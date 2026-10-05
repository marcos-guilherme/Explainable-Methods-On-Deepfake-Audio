"""Estágios do pipeline. Cada função recebe um RunContext e lê/grava artefatos em disco."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from . import artifacts as A
from . import bands
from .adaptation import crossfit_oof_scores, fit_head, make_p_spoof_ad, score_head
from .config import RunConfig
from .data import SPOOF_LABEL, build_balanced_split
from .features import mel_band_features
from .logging_utils import progress
from .metrics import calibrate_threshold, evaluate_at_threshold, quadrant
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


def _has_explicit_calibration(cfg: RunConfig) -> bool:
    return bool(cfg.data.calibration_split) and cfg.data.n_calibration_per_class > 0


def _collect_specs(cfg: RunConfig) -> list[tuple[str, str, int]]:
    """(nome lógico, split-fonte, n por classe) a coletar, por modo."""
    has_cal_split = bool(cfg.data.calibration_split)
    has_cal_quota = cfg.data.n_calibration_per_class > 0
    if has_cal_split != has_cal_quota:
        raise ValueError(
            "data.calibration_split e data.n_calibration_per_class devem ser definidos juntos"
        )
    if cfg.adapt.cross_fit:
        if has_cal_split:
            raise ValueError(
                "adapt.cross_fit não pode ser combinado com calibração holdout explícita"
            )
        return [(POOL_SPLIT, cfg.data.analysis_split, cfg.data.n_analysis_per_class)]
    specs = [(cfg.data.train_split, cfg.data.train_split, cfg.data.n_train_per_class)]
    if _has_explicit_calibration(cfg):
        specs.append(
            (
                cfg.data.calibration_split,
                cfg.data.calibration_split,
                cfg.data.n_calibration_per_class,
            )
        )
    specs.append((cfg.data.eval_split, cfg.data.eval_split, cfg.data.n_test_per_class))
    return specs


def _load_split_labels(samples: pd.DataFrame, split: str) -> np.ndarray:
    return samples.loc[samples.split == split, "label"].to_numpy()


def _load_analysis_audio(
    paths: A.RunPaths,
    split: str,
    samples: pd.DataFrame,
) -> tuple[list[np.ndarray], list[int]]:
    """Lê cache legado de áudio ou recarrega WAVs canônicos pelo catálogo."""
    audio_path = paths.path(f"audios_{split}.npy")
    srs_path = paths.path(f"srs_{split}.npy")
    if audio_path.is_file() and srs_path.is_file():
        return (
            list(np.load(audio_path, allow_pickle=True)),
            [int(sr) for sr in np.load(srs_path)],
        )
    if "processed_path" not in samples.columns:
        raise ValueError(
            f"áudio ausente para split={split}: cache .npy e processed_path indisponíveis"
        )
    audios: list[np.ndarray] = []
    srs: list[int] = []
    for raw_path in samples["processed_path"]:
        path = str(raw_path)
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"falha ao ler processed_path: {path}") from exc
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0 or not np.all(np.isfinite(waveform)):
            raise ValueError(f"áudio canônico inválido em processed_path: {path}")
        audios.append(waveform)
        srs.append(int(sr))
    return audios, srs


def _build_thresholds_payload(
    *,
    calibration_source: str,
    calibration_fallback_train: bool,
    detectors: dict[str, dict],
) -> dict:
    return {
        "schema_version": 1,
        "calibration_source": calibration_source,
        "calibration_fallback_train": bool(calibration_fallback_train),
        "detectors": detectors,
    }


def stage_collect(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    rows = []
    for name, split, n in _collect_specs(cfg):
        audios, srs, labels, prov = build_balanced_split(
            cfg.data.dataset_id,
            split,
            n,
            loader=cfg.data.loader,
            seed=cfg.seed,
            dataset_kind=cfg.data.dataset_kind,
            manifest_path=cfg.data.manifest_path,
        )
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

    Limiares vêm dos scores OOF do pool (fonte de calibração); nunca de outro alvo."""
    from .adaptation import build_head

    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    pool = samples[samples.split == POOL_SPLIT].reset_index(drop=True)
    y = pool["label"].to_numpy()
    emb = A.load_npy(p.path(f"emb_{POOL_SPLIT}.npy"))
    make = lambda: build_head(cfg.adapt.head, cfg.seed)
    p_ad = crossfit_oof_scores(
        emb, y, make, n_splits=cfg.adapt.cv_folds, seed=cfg.seed, spoof_label=SPOOF_LABEL
    )
    final_head = fit_head(emb, y, head=cfg.adapt.head, seed=cfg.seed)
    joblib.dump(final_head, p.path("d_ad.joblib"))

    cal_source = "pool/crossfit_oof"
    detectors = {"ad": calibrate_threshold(p_ad, y, cal_source, "ad")}
    enc = ctx.get_embedder()
    precheck = {
        "mode": "crossfit",
        "head": cfg.adapt.head,
        "cv_folds": cfg.adapt.cv_folds,
        "n_pool": int(len(y)),
        "calibration_source": cal_source,
        "calibration_fallback_train": False,
        "has_zero_shot": bool(getattr(enc, "has_zero_shot", False)),
        "ad_calibration": detectors["ad"],
        "ad_eval": evaluate_at_threshold(
            p_ad, y, detectors["ad"]["threshold"],
            condition="crossfit_pool", source=cal_source, target=POOL_SPLIT,
        ),
    }
    if getattr(enc, "has_zero_shot", False):
        audios = list(np.load(p.path(f"audios_{POOL_SPLIT}.npy"), allow_pickle=True))
        srs = [int(s) for s in np.load(p.path(f"srs_{POOL_SPLIT}.npy"))]
        p_zs = _p_spoof_zs(enc, audios, srs)
        detectors["zs"] = calibrate_threshold(p_zs, y, cal_source, "zs")
        zs_eval = evaluate_at_threshold(
            p_zs, y, detectors["zs"]["threshold"],
            condition="crossfit_pool", source=cal_source, target=POOL_SPLIT,
        )
        precheck.update(
            zs_calibration=detectors["zs"],
            zs_eval=zs_eval,
            improved=bool(
                detectors["ad"]["calibration_eer"] < detectors["zs"]["calibration_eer"]
            ),
        )
        A.save_npy(p_zs.astype(np.float32), p.path("p_spoof_zs.npy"))
    A.save_json(
        _build_thresholds_payload(
            calibration_source=cal_source,
            calibration_fallback_train=False,
            detectors=detectors,
        ),
        p.path("thresholds.json"),
    )
    A.save_json(precheck, p.path("eer_precheck.json"))
    A.save_npy(p_ad.astype(np.float32), p.path("p_spoof_ad.npy"))


def _adapt_legacy(ctx: RunContext) -> None:
    """Holdout: fit só no train; calibra limiar na fonte; pontua eval/test."""
    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    train_split = cfg.data.train_split
    eval_split = cfg.data.eval_split
    y_train = _load_split_labels(samples, train_split)
    y_eval = _load_split_labels(samples, eval_split)
    emb_train = A.load_npy(p.path(f"emb_{train_split}.npy"))
    emb_eval = A.load_npy(p.path(f"emb_{eval_split}.npy"))

    ad_head = fit_head(emb_train, y_train, head=cfg.adapt.head, seed=cfg.seed)
    joblib.dump(ad_head, p.path("d_ad.joblib"))

    if _has_explicit_calibration(cfg):
        cal_split = cfg.data.calibration_split
        y_cal = _load_split_labels(samples, cal_split)
        emb_cal = A.load_npy(p.path(f"emb_{cal_split}.npy"))
        cal_fallback = False
    else:
        # Calibração in-sample legada (HF/BRSpeech): usa o próprio train após fit.
        # Nunca usa labels do eval/test para definir limiar.
        cal_split = train_split
        y_cal = y_train
        emb_cal = emb_train
        cal_fallback = True

    p_ad_cal = score_head(ad_head, emb_cal)
    detectors = {"ad": calibrate_threshold(p_ad_cal, y_cal, cal_split, "ad")}
    p_ad_eval = score_head(ad_head, emb_eval)

    enc = ctx.get_embedder()
    precheck = {
        "mode": "holdout",
        "head": cfg.adapt.head,
        "calibration_source": cal_split,
        "calibration_fallback_train": cal_fallback,
        "has_zero_shot": bool(getattr(enc, "has_zero_shot", False)),
        "ad_calibration": detectors["ad"],
        "ad_eval": evaluate_at_threshold(
            p_ad_eval, y_eval, detectors["ad"]["threshold"],
            condition="holdout", source=cal_split, target=eval_split,
        ),
    }
    if getattr(enc, "has_zero_shot", False):
        audios_cal = list(np.load(p.path(f"audios_{cal_split}.npy"), allow_pickle=True))
        srs_cal = [int(s) for s in np.load(p.path(f"srs_{cal_split}.npy"))]
        p_zs_cal = _p_spoof_zs(enc, audios_cal, srs_cal)
        detectors["zs"] = calibrate_threshold(p_zs_cal, y_cal, cal_split, "zs")

        audios_eval = list(np.load(p.path(f"audios_{eval_split}.npy"), allow_pickle=True))
        srs_eval = [int(s) for s in np.load(p.path(f"srs_{eval_split}.npy"))]
        p_zs_eval = _p_spoof_zs(enc, audios_eval, srs_eval)
        precheck.update(
            zs_calibration=detectors["zs"],
            zs_eval=evaluate_at_threshold(
                p_zs_eval, y_eval, detectors["zs"]["threshold"],
                condition="holdout", source=cal_split, target=eval_split,
            ),
            improved=bool(
                detectors["ad"]["calibration_eer"] < detectors["zs"]["calibration_eer"]
            ),
        )
        A.save_npy(p_zs_eval.astype(np.float32), p.path("p_spoof_zs.npy"))
    A.save_json(
        _build_thresholds_payload(
            calibration_source=cal_split,
            calibration_fallback_train=cal_fallback,
            detectors=detectors,
        ),
        p.path("thresholds.json"),
    )
    A.save_json(precheck, p.path("eer_precheck.json"))
    A.save_npy(p_ad_eval.astype(np.float32), p.path("p_spoof_ad.npy"))


def stage_features(ctx: RunContext) -> None:
    """Tabela mestra: score/quadrante por detector + energia log-mel por banda (μ/σ)."""
    cfg, p = ctx.cfg, ctx.paths
    split = _analysis_split(cfg)
    samples = A.load_table(p.path("samples.parquet"))
    test = samples[samples.split == split].reset_index(drop=True)
    audios, srs = _load_analysis_audio(p, split, test)
    y = test["label"].to_numpy()
    thresholds = A.load_json(p.path("thresholds.json"))
    if thresholds.get("schema_version") != 1:
        raise ValueError("thresholds.json tem schema_version inválida")
    det_thr = thresholds.get("detectors")
    if not isinstance(det_thr, dict):
        raise ValueError("thresholds.json deve conter detectors")

    def _threshold(detector: str) -> float:
        try:
            value = float(det_thr[detector]["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"thresholds.json não contém threshold válido para {detector}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"threshold de {detector} deve ser finito")
        return value

    p_ad = A.load_npy(p.path("p_spoof_ad.npy"))
    thr_ad = _threshold("ad")
    expected_n = len(test)
    if len(audios) != expected_n or len(srs) != expected_n or len(p_ad) != expected_n:
        raise ValueError(
            "artefatos de análise têm comprimentos incompatíveis: "
            f"samples={expected_n}, audios={len(audios)}, srs={len(srs)}, ad={len(p_ad)}"
        )
    p_zs = None
    thr_zs = None
    if p.path("p_spoof_zs.npy").exists():
        p_zs = A.load_npy(p.path("p_spoof_zs.npy"))
        thr_zs = _threshold("zs")
        if len(p_zs) != expected_n:
            raise ValueError(
                f"artefato zs tem comprimento {len(p_zs)}, esperado {expected_n}"
            )
    if ctx.logger:
        ctx.logger.info(f"features log-mel: {len(audios)} clipes × "
                        f"{cfg.bands.n_bands} bandas (μ/σ)")
    feats = np.vstack([mel_band_features(a, s)
                       for a, s in progress(list(zip(audios, srs)),
                                            desc="features log-mel", unit="clip")])
    sample_ids = test["sample_id"] if "sample_id" in test.columns else test.index
    df = pd.DataFrame({"sample_id": sample_ids, "ground_truth": y})
    # D_zs só existe quando o encoder expõe zero-shot (arquivo gravado no stage_adapt).
    if p_zs is not None:
        df["p_spoof_zs"] = p_zs
        df["pred_zs"] = (p_zs >= thr_zs).astype(int)
        df["quadrant_zs"] = [quadrant(t, pz) for t, pz in zip(df.ground_truth, df.pred_zs)]
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
    samples = A.load_table(p.path("samples.parquet"))
    analysis_samples = samples[samples.split == split].reset_index(drop=True)
    audios, srs = _load_analysis_audio(p, split, analysis_samples)
    enc = ctx.get_embedder()
    head = joblib.load(p.path("d_ad.joblib"))
    _, p_ad_from_audio = make_p_spoof_ad(head, enc)
    edges = mel_band_edges(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
    # D_ad sempre; D_zs só quando o encoder expõe zero-shot.
    targets = []
    if "zs" in _present_detectors(master):
        if not getattr(enc, "has_zero_shot", False):
            raise ValueError("master contém D_zs, mas o encoder não oferece zero-shot")
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
    thresholds = A.load_json(p.path("thresholds.json"))
    target = _analysis_split(ctx.cfg)
    rows = []
    for tag in _present_detectors(master):
        scores = master[f"p_spoof_{tag}"].to_numpy()
        thr = float(thresholds["detectors"][tag]["threshold"])
        pred = master[f"pred_{tag}"].to_numpy()
        eval_out = evaluate_at_threshold(
            scores, y, thr,
            condition="report", source=thresholds["calibration_source"], target=target,
        )
        tf = eval_out["threshold_free"]
        fx = eval_out["fixed_threshold"]
        rows.append({
            "detector": tag,
            "target": target,
            "calibration_source": thresholds["calibration_source"],
            "eer_diagnostic": float(tf["eer_diagnostic"]),
            "roc_auc": float(tf["roc_auc"]),
            "average_precision": float(tf["average_precision"]),
            "threshold_fixed": thr,
            "mcc_at_fixed_threshold": float(fx["mcc"]),
            "accuracy_at_fixed_threshold": float(fx["accuracy"]),
            "tpr_at_fixed_threshold": float(fx["tpr"]),
            "fpr_at_fixed_threshold": float(fx["fpr"]),
            "fnr_at_fixed_threshold": float(fx["fnr"]),
            # Compatibilidade com consumidores legados da tabela de performance.
            "eer": float(tf["eer_diagnostic"]),
            "mcc": float(fx["mcc"]),
            "accuracy": float(fx["accuracy"]),
        })
        assert np.array_equal(pred, (scores >= thr).astype(int))
    A.save_table(pd.DataFrame(rows), p.path("performance_table.csv"))
    if _plots_on(ctx):
        from .plotting import plot_det, set_plot_style
        set_plot_style()
        plot_det(master, p.path("figures"))
    A.save_json({"stages": "complete", "config_hash": ctx.cfg.config_hash()},
                p.path("run_manifest.json"))


def _plots_on(ctx: RunContext) -> bool:
    return getattr(ctx, "plots", True)
