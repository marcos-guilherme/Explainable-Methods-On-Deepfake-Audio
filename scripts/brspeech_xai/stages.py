"""Estágios do pipeline. Cada função recebe um RunContext e lê/grava artefatos em disco."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import artifacts as A
from . import bands
from .adaptation import make_p_spoof_ad
from .config import RunConfig
from .data import SPOOF_LABEL, build_balanced_split
from .features import mel_band_features
from .logging_utils import progress
from .metrics import compute_eer, quadrant
from .occlusion import (bootstrap_ci, mel_band_edges, occlusion_drop,
                        per_band_occlusion_drop, stratified_idx, to_16k_mono)
from .stats import (band_assoc_signed, confirmatory_tests, convergence_bands,
                    cross_spine_agreement, paired_intervention_test,
                    spearman_intraclass, top_features_by_rho)

POOL_SPLIT = "pool"  # nome lógico do split de análise no modo cross-fit


def _analysis_split(cfg: RunConfig) -> str:
    """Split lógico sobre o qual roda a análise (pool no cross-fit, senão o eval)."""
    return POOL_SPLIT if cfg.adapt.cross_fit else cfg.data.eval_split


def _present_detectors(master) -> list[str]:
    """Detectores com score disponível na tabela mestra (ordem fixa zs, ad).

    Encoders só-extratores (has_zero_shot=False) não geram D_zs; os estágios a
    jusante devem iterar apenas os detectores presentes, em vez de assumir zs+ad.
    """
    return [t for t in ("zs", "ad") if f"p_spoof_{t}" in master.columns]


def _p_spoof_zs(embedder, audios, srs) -> np.ndarray:
    """P(spoof) zero-shot (D_zs) do encoder, exigindo suporte a zero-shot.

    O pipeline atual compara D_zs vs D_ad. Encoders só-extratores (has_zero_shot=False)
    ainda não têm o caminho de D_zs implementado nos estágios; ao registrar o primeiro,
    estender features/occlusion/report para pular D_zs de forma consistente.
    """
    if not getattr(embedder, "has_zero_shot", False):
        raise NotImplementedError(
            f"encoder {getattr(embedder, 'name', '?')!r} não tem zero-shot; o pipeline "
            "D_zs ainda não suporta encoders só-extratores (ver spec de encoders)."
        )
    return np.asarray(embedder.spoof_prob_batch(audios, srs))


@dataclass
class RunContext:
    cfg: RunConfig
    paths: A.RunPaths
    logger: object
    embedder: object = None  # encoder carregado sob demanda (build_encoder)

    def get_embedder(self):
        if self.embedder is None:
            from .encoders import build_encoder
            self.embedder = build_encoder(self.cfg.model, self.cfg.device)
        return self.embedder


def _collect_specs(cfg: RunConfig) -> list[tuple[str, str, int]]:
    """(nome lógico, split-fonte, n por classe) a coletar, por modo."""
    if cfg.adapt.cross_fit:
        return [(POOL_SPLIT, cfg.data.analysis_split, cfg.data.n_analysis_per_class)]
    return [(cfg.data.train_split, cfg.data.train_split, cfg.data.n_train_per_class),
            (cfg.data.eval_split, cfg.data.eval_split, cfg.data.n_test_per_class)]


def stage_collect(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    rows = []
    for name, split, n in _collect_specs(cfg):
        audios, srs, labels, prov = build_balanced_split(
            cfg.data.dataset_id, split, n, loader=cfg.data.loader, seed=cfg.seed)
        for i, pr in enumerate(prov):
            rows.append({"split": name, "idx": i, **pr, "sample_rate": srs[i]})
        np.save(p.path(f"audios_{name}.npy"),
                np.array([a for a in audios], dtype=object), allow_pickle=True)
        np.save(p.path(f"srs_{name}.npy"), np.array(srs, dtype=np.int32))
    A.save_table(pd.DataFrame(rows), p.path("samples.parquet"))


def stage_embeddings(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    enc = ctx.get_embedder()
    for name, _split, _n in _collect_specs(cfg):
        audios = list(np.load(p.path(f"audios_{name}.npy"), allow_pickle=True))
        srs = list(np.load(p.path(f"srs_{name}.npy")))
        if ctx.logger:
            ctx.logger.info(f"embeddings '{name}': {len(audios)} clipes ({enc.name})")
        emb = enc.extract_embeddings(audios, [int(s) for s in srs])
        A.save_npy(emb.astype(np.float32), p.path(f"emb_{name}.npy"))


def stage_adapt(ctx: RunContext) -> None:
    if ctx.cfg.adapt.cross_fit:
        _adapt_crossfit(ctx)
    else:
        _adapt_legacy(ctx)


def _adapt_crossfit(ctx: RunContext) -> None:
    """Head cross-fitted: P(spoof) OOF por K-fold + head final (p/ oclusão).

    O D_zs (congelado) pontua tudo direto; o D_ad recebe scores out-of-fold, sem
    vazamento. O EER-precheck usa os scores OOF (estimativa honesta)."""
    from .adaptation import build_head, crossfit_oof_scores
    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    pool = samples[samples.split == POOL_SPLIT].reset_index(drop=True)
    y = pool["label"].to_numpy()
    emb = A.load_npy(p.path(f"emb_{POOL_SPLIT}.npy"))
    make = lambda: build_head(cfg.adapt.head, cfg.seed)
    p_ad = crossfit_oof_scores(emb, y, make, n_splits=cfg.adapt.cv_folds,
                               seed=cfg.seed, spoof_label=SPOOF_LABEL)
    # Head final no pool inteiro: usado só pela oclusão (contrafactual do modelo).
    final_head = make()
    final_head.fit(emb, y)
    joblib.dump(final_head, p.path("d_ad.joblib"))
    eer_ad, _ = compute_eer(p_ad, y)
    enc = ctx.get_embedder()
    precheck = {"mode": "crossfit", "head": cfg.adapt.head, "cv_folds": cfg.adapt.cv_folds,
                "n_pool": int(len(y)), "eer_ad": float(eer_ad),
                "has_zero_shot": bool(getattr(enc, "has_zero_shot", False))}
    # D_zs (congelado) pontua o pool direto, só quando o encoder expõe zero-shot.
    if getattr(enc, "has_zero_shot", False):
        audios = list(np.load(p.path(f"audios_{POOL_SPLIT}.npy"), allow_pickle=True))
        srs = [int(s) for s in np.load(p.path(f"srs_{POOL_SPLIT}.npy"))]
        p_zs = _p_spoof_zs(enc, audios, srs)
        eer_zs, _ = compute_eer(p_zs, y)
        precheck.update(eer_zs=float(eer_zs), improved=bool(eer_ad < eer_zs))
        A.save_npy(p_zs.astype(np.float32), p.path("p_spoof_zs.npy"))
    A.save_json(precheck, p.path("eer_precheck.json"))
    A.save_npy(p_ad.astype(np.float32), p.path("p_spoof_ad.npy"))


def _adapt_legacy(ctx: RunContext) -> None:
    """Modo antigo: head treina no split train e pontua o split test (holdout)."""
    from .adaptation import build_head
    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    y_train = samples.loc[samples.split == cfg.data.train_split, "label"].to_numpy()
    y_test = samples.loc[samples.split == cfg.data.eval_split, "label"].to_numpy()
    emb_train = A.load_npy(p.path(f"emb_{cfg.data.train_split}.npy"))
    emb_test = A.load_npy(p.path(f"emb_{cfg.data.eval_split}.npy"))
    ad_head = build_head(cfg.adapt.head, cfg.seed)
    ad_head.fit(emb_train, y_train)
    joblib.dump(ad_head, p.path("d_ad.joblib"))
    p_ad = ad_head.predict_proba(emb_test)[:, SPOOF_LABEL]
    eer_ad, _ = compute_eer(p_ad, y_test)
    enc = ctx.get_embedder()
    precheck = {"mode": "holdout", "head": cfg.adapt.head, "eer_ad": float(eer_ad),
                "has_zero_shot": bool(getattr(enc, "has_zero_shot", False))}
    # D_zs (congelado) pontua o split de avaliação, só quando há zero-shot.
    if getattr(enc, "has_zero_shot", False):
        audios = list(np.load(p.path(f"audios_{cfg.data.eval_split}.npy"), allow_pickle=True))
        srs = [int(s) for s in np.load(p.path(f"srs_{cfg.data.eval_split}.npy"))]
        p_zs = _p_spoof_zs(enc, audios, srs)
        eer_zs, _ = compute_eer(p_zs, y_test)
        precheck.update(eer_zs=float(eer_zs), improved=bool(eer_ad < eer_zs))
        A.save_npy(p_zs.astype(np.float32), p.path("p_spoof_zs.npy"))
    A.save_json(precheck, p.path("eer_precheck.json"))
    A.save_npy(p_ad.astype(np.float32), p.path("p_spoof_ad.npy"))


def stage_features(ctx: RunContext) -> None:
    """Tabela mestra: score/quadrante por detector + energia log-mel por banda (μ/σ)."""
    cfg, p = ctx.cfg, ctx.paths
    split = _analysis_split(cfg)
    samples = A.load_table(p.path("samples.parquet"))
    test = samples[samples.split == split].reset_index(drop=True)
    audios = list(np.load(p.path(f"audios_{split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(p.path(f"srs_{split}.npy"))]
    y = test["label"].to_numpy()
    if ctx.logger:
        ctx.logger.info(f"features log-mel: {len(audios)} clipes × "
                        f"{cfg.bands.n_bands} bandas (μ/σ)")
    feats = np.vstack([mel_band_features(a, s)
                       for a, s in progress(list(zip(audios, srs)),
                                            desc="features log-mel", unit="clip")])
    df = pd.DataFrame({"sample_id": test.index, "ground_truth": y})
    # D_zs só existe quando o encoder expõe zero-shot (arquivo gravado no stage_adapt).
    if p.path("p_spoof_zs.npy").exists():
        p_zs = A.load_npy(p.path("p_spoof_zs.npy"))
        thr_zs = compute_eer(p_zs, y)[1]
        df["p_spoof_zs"] = p_zs
        df["pred_zs"] = (p_zs >= thr_zs).astype(int)
        df["quadrant_zs"] = [quadrant(t, pz) for t, pz in zip(df.ground_truth, df.pred_zs)]
    p_ad = A.load_npy(p.path("p_spoof_ad.npy"))
    thr_ad = compute_eer(p_ad, y)[1]
    df["p_spoof_ad"] = p_ad
    df["pred_ad"] = (p_ad >= thr_ad).astype(int)
    df["quadrant_ad"] = [quadrant(t, pa) for t, pa in zip(df.ground_truth, df.pred_ad)]
    for j, name in enumerate(bands.BAND_COLS):
        df[name] = feats[:, j]
    A.save_table(df, p.path("master_table.parquet"))


def stage_association(ctx: RunContext) -> None:
    """Espinha 1 (associativa): Spearman intra-classe energia-de-banda↔P(spoof)."""
    cfg, p = ctx.cfg, ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    spearman = pd.concat([spearman_intraclass(master, t, bands.BAND_COLS)
                          for t in _present_detectors(master)], ignore_index=True)
    A.save_table(spearman, p.path("spearman_table.csv"))
    if _plots_on(ctx):
        from .plotting import (plot_association_profile,
                               plot_association_profile_signed,
                               plot_association_profile_single,
                               plot_spearman_scatter, set_plot_style)
        set_plot_style()
        plot_association_profile(spearman, p.path("figures"), top_n=cfg.association.top_n)
        plot_association_profile_signed(spearman, p.path("figures"))
        for tag in ("zs", "ad"):
            if (spearman.detector == tag).any():
                plot_association_profile_single(spearman, tag, p.path("figures"))
        plot_spearman_scatter(master, spearman, p.path("figures"))


def _convergence_intervention(cfg, master, audios, srs, edges, targets):
    """Teste de convergência H1->H2 por intervenção, ORIENTADO pelo sinal (ver plot).

    Rankeia |ρ| associativo (H1) numa metade dos clipes e, na outra metade, oclui banda a
    banda as mais e as menos associadas. O efeito causal de cada banda é orientado pelo
    sinal da sua associação (`drop x sinal(ρ)`): assim, tanto remover uma pista de spoof
    (derruba P(spoof)) quanto remover uma pista de bonafide (sobe P(spoof)) contam como
    efeito POSITIVO quando são coerentes com o H1. Isso evita o cancelamento que ocorre
    ao ocluir em grupo bandas de sinais opostos. Metades disjuntas evitam circularidade.

    O valor por clipe é a média (sobre as bandas do grupo) do efeito orientado. Uma
    diferença mediana positiva (top - bottom) indica que as bandas mais associadas pelo
    H1 têm efeito causal maior e no sentido previsto: as duas análises convergem.

    Returns:
        (pairs, table): `pairs` = {tag: (oriented_top, oriented_bottom)} por clipe;
        `table` = DataFrame com um teste pareado por detector.
    """
    k = cfg.occlusion.convergence_k or max(1, cfg.bands.n_bands // 3)
    perm = np.random.default_rng(cfg.seed).permutation(len(master))
    half = len(perm) // 2
    rank_master, test_master = master.iloc[perm[:half]], master.iloc[perm[half:]]
    pairs, rows = {}, []
    for tag, quad_col, fn in targets:
        sp = spearman_intraclass(rank_master, tag, bands.BAND_COLS)
        top, bottom = convergence_bands(sp, tag, cfg.bands.n_bands, k)
        signs = np.sign(np.nan_to_num(band_assoc_signed(sp, tag, cfg.bands.n_bands)))
        idx = stratified_idx(test_master, quad_col, cfg.occlusion.per_quadrant, seed=cfg.seed)
        sub_audios = [to_16k_mono(audios[i], srs[i]) for i in idx]
        sub_srs = [16000] * len(sub_audios)
        sel = top + bottom  # oclui só as bandas dos dois grupos (uma por vez)
        drops = per_band_occlusion_drop(fn, sub_audios, sub_srs, edges, sel,
                                        desc=f"convergência {tag} (top/bottom-{k})")
        oriented = drops * signs[sel]                 # orienta cada banda pelo sentido de H1
        o_top = oriented[:, :len(top)].mean(axis=1)   # efeito médio nas mais associadas
        o_bottom = oriented[:, len(top):].mean(axis=1)  # efeito médio nas menos associadas
        stat = paired_intervention_test(o_top, o_bottom, seed=cfg.seed,
                                        n_boot=cfg.occlusion.n_boot)
        rows.append({"detector": tag, "k_bands": k,
                     "top_bands": ";".join(map(str, top)),
                     "bottom_bands": ";".join(map(str, bottom)), **stat})
        pairs[tag] = (o_top, o_bottom)
    return pairs, pd.DataFrame(rows)


def stage_occlusion(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    split = _analysis_split(cfg)
    master = A.load_table(p.path("master_table.parquet"))
    audios = list(np.load(p.path(f"audios_{split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(p.path(f"srs_{split}.npy"))]
    enc = ctx.get_embedder()
    head = joblib.load(p.path("d_ad.joblib"))
    _, p_ad_from_audio = make_p_spoof_ad(head, enc)
    edges = mel_band_edges(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
    # D_ad sempre; D_zs só quando o encoder expõe zero-shot.
    targets = []
    if getattr(enc, "has_zero_shot", False):
        targets.append(("zs", "quadrant_zs", lambda a, s: enc.spoof_prob(a, s)))
    targets.append(("ad", "quadrant_ad", p_ad_from_audio))
    rows = []
    for tag, quad_col, fn in targets:
        idx = stratified_idx(master, quad_col, cfg.occlusion.per_quadrant, seed=cfg.seed)
        # A oclusão sempre filtra a 16 kHz (bordas < Nyquist para qualquer sr nativo).
        sub_audios = [to_16k_mono(audios[i], srs[i]) for i in idx]
        sub_srs = [16000] * len(sub_audios)
        if ctx.logger:
            ctx.logger.info(f"oclusão {tag}: {len(sub_audios)} clipes × "
                            f"{cfg.bands.n_bands} bandas")
        # (n_clips, n_bands): queda por clipe; média + IC 95% por bootstrap sobre os clipes.
        drops = occlusion_drop(fn, sub_audios, sub_srs, edges, desc=f"oclusão {tag}")
        mean_drop = drops.mean(axis=0)
        ci_low, ci_high = bootstrap_ci(drops, n_boot=cfg.occlusion.n_boot, seed=cfg.seed)
        for bi in range(len(edges) - 1):
            rows.append({"detector": tag, "band_hz_low": float(edges[bi]),
                         "band_hz_high": float(edges[bi + 1]),
                         "mean_p_spoof_drop": float(mean_drop[bi]),
                         "ci_low": float(ci_low[bi]), "ci_high": float(ci_high[bi]),
                         "n": int(drops.shape[0])})
    occ = pd.DataFrame(rows)
    A.save_table(occ, p.path("occlusion_table.csv"))
    # Convergência das duas espinhas na mesma grade de bandas (causal × associativo).
    spearman = A.load_table(p.path("spearman_table.csv"))
    agree = cross_spine_agreement(occ, spearman)
    A.save_table(agree, p.path("spine_agreement.csv"))
    # Convergência por INTERVENÇÃO (H1->H2): teste pareado por clipe, mais robusto que o
    # ρ agregado sobre poucas bandas. Ranking e oclusão usam metades disjuntas de clipes.
    conv_pairs, conv = _convergence_intervention(cfg, master, audios, srs, edges, targets)
    A.save_table(conv, p.path("convergence_intervention.csv"))
    if _plots_on(ctx):
        from .plotting import (plot_convergence_intervention, plot_occlusion_bands,
                               plot_occlusion_overlay, plot_spine_convergence,
                               set_plot_style)
        set_plot_style()
        plot_occlusion_bands(edges, occ, p.path("figures"))
        plot_occlusion_overlay(edges, occ, p.path("figures"))
        plot_spine_convergence(edges, occ, spearman, agree, p.path("figures"))
        plot_convergence_intervention(conv_pairs, conv, p.path("figures"))


def stage_confirmatory(ctx: RunContext) -> None:
    """H3: Welch/Levene + FDR nas bandas de maior |ρ| (seleção via Espinha 1)."""
    p = ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    spearman = A.load_table(p.path("spearman_table.csv"))
    top = top_features_by_rho(spearman, ctx.cfg.association.top_n)
    dets = _present_detectors(master)
    conf = pd.concat([confirmatory_tests(master, t, f"quadrant_{t}", top) for t in dets],
                     ignore_index=True)
    A.save_table(conf, p.path("confirmatory_tests.csv"))
    if _plots_on(ctx):
        from .plotting import (plot_confirmatory_box, plot_confirmatory_effects,
                               set_plot_style)
        set_plot_style()
        top_box = top[:6]  # H3: até 6 features mais salientes (boxplots de apoio)
        for t in dets:
            plot_confirmatory_box(master, conf, top_box, t, f"quadrant_{t}", p.path("figures"))
            # Figura principal de H3: tamanho de efeito (média TN-FP e variância TP-FN).
            plot_confirmatory_effects(master, conf, top, t, f"quadrant_{t}", p.path("figures"))


def stage_report(ctx: RunContext) -> None:
    p = ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    y = master["ground_truth"].to_numpy()
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    rows = []
    for tag in _present_detectors(master):
        eer, _ = compute_eer(master[f"p_spoof_{tag}"].to_numpy(), y)
        pred = master[f"pred_{tag}"].to_numpy()
        rows.append({"detector": tag, "eer": float(eer),
                     "mcc": float(matthews_corrcoef(y, pred)),
                     "accuracy": float(accuracy_score(y, pred))})
    A.save_table(pd.DataFrame(rows), p.path("performance_table.csv"))
    if _plots_on(ctx):
        from .plotting import plot_det, set_plot_style
        set_plot_style()
        plot_det(master, p.path("figures"))
    A.save_json({"stages": "complete", "config_hash": ctx.cfg.config_hash()},
                p.path("run_manifest.json"))


def _plots_on(ctx: RunContext) -> bool:
    return getattr(ctx, "plots", True)
