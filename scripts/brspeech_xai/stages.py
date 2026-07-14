"""Estágios do pipeline. Cada função recebe um RunContext e lê/grava artefatos em disco."""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import artifacts as A
from .config import RunConfig
from .data import SPOOF_LABEL, build_balanced_split
from .features import MFCC_COLS, mfcc_features
from .inference import extract_embeddings, make_p_spoof_ad, score_audios
from .metrics import compute_eer, quadrant
from .occlusion import mel_band_edges, occlusion_drop, stratified_idx, to_16k_mono
from .stats import confirmatory_tests
from .surrogate import fit_surrogate, shap_importance

FEATURE_NAMES = MFCC_COLS


@dataclass
class RunContext:
    cfg: RunConfig
    paths: A.RunPaths
    logger: object
    detector: object = None  # carregado sob demanda (build_detector)

    def get_detector(self):
        if self.detector is None:
            from .model import build_detector
            self.detector = build_detector(self.cfg.model.checkpoint,
                                           self.cfg.model.spoof_index, self.cfg.device)
        return self.detector


def stage_collect(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    rows = []
    for split, n in [(cfg.data.train_split, cfg.data.n_train_per_class),
                     (cfg.data.eval_split, cfg.data.n_test_per_class)]:
        audios, srs, labels, prov = build_balanced_split(
            cfg.data.dataset_id, split, n, loader=cfg.data.loader, seed=cfg.seed)
        for i, pr in enumerate(prov):
            rows.append({"split": split, "idx": i, **pr, "sample_rate": srs[i]})
        np.save(p.path(f"audios_{split}.npy"),
                np.array([a for a in audios], dtype=object), allow_pickle=True)
        np.save(p.path(f"srs_{split}.npy"), np.array(srs, dtype=np.int32))
    A.save_table(pd.DataFrame(rows), p.path("samples.parquet"))


def stage_embeddings(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    det = ctx.get_detector()
    for split in (cfg.data.train_split, cfg.data.eval_split):
        audios = list(np.load(p.path(f"audios_{split}.npy"), allow_pickle=True))
        srs = list(np.load(p.path(f"srs_{split}.npy")))
        emb = extract_embeddings(det, audios, [int(s) for s in srs])
        A.save_npy(emb.astype(np.float32), p.path(f"emb_{split}.npy"))


def stage_adapt(ctx: RunContext) -> None:
    from sklearn.linear_model import LogisticRegression
    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    y_train = samples.loc[samples.split == cfg.data.train_split, "label"].to_numpy()
    y_test = samples.loc[samples.split == cfg.data.eval_split, "label"].to_numpy()
    emb_train = A.load_npy(p.path(f"emb_{cfg.data.train_split}.npy"))
    emb_test = A.load_npy(p.path(f"emb_{cfg.data.eval_split}.npy"))
    logreg = LogisticRegression(max_iter=1000).fit(emb_train, y_train)
    joblib.dump(logreg, p.path("d_ad.joblib"))
    # pré-check de EER: D_zs vs D_ad no test
    det = ctx.get_detector()
    audios = list(np.load(p.path(f"audios_{cfg.data.eval_split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(p.path(f"srs_{cfg.data.eval_split}.npy"))]
    p_zs = np.array(score_audios(det, audios, srs))
    p_ad = logreg.predict_proba(emb_test)[:, SPOOF_LABEL]
    eer_zs, _ = compute_eer(p_zs, y_test)
    eer_ad, _ = compute_eer(p_ad, y_test)
    A.save_json({"eer_zs": float(eer_zs), "eer_ad": float(eer_ad),
                 "improved": bool(eer_ad < eer_zs)}, p.path("eer_precheck.json"))
    A.save_npy(p_zs.astype(np.float32), p.path("p_spoof_zs.npy"))
    A.save_npy(p_ad.astype(np.float32), p.path("p_spoof_ad.npy"))


def stage_master_mfcc(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    samples = A.load_table(p.path("samples.parquet"))
    test = samples[samples.split == cfg.data.eval_split].reset_index(drop=True)
    audios = list(np.load(p.path(f"audios_{cfg.data.eval_split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(p.path(f"srs_{cfg.data.eval_split}.npy"))]
    p_zs = A.load_npy(p.path("p_spoof_zs.npy"))
    p_ad = A.load_npy(p.path("p_spoof_ad.npy"))
    y = test["label"].to_numpy()
    thr_zs = compute_eer(p_zs, y)[1]
    thr_ad = compute_eer(p_ad, y)[1]
    feats = np.vstack([mfcc_features(a, s) for a, s in zip(audios, srs)])
    df = pd.DataFrame({"sample_id": test.index, "ground_truth": y,
                       "p_spoof_zs": p_zs, "pred_zs": (p_zs >= thr_zs).astype(int),
                       "p_spoof_ad": p_ad, "pred_ad": (p_ad >= thr_ad).astype(int)})
    df["quadrant_zs"] = [quadrant(t, pz) for t, pz in zip(df.ground_truth, df.pred_zs)]
    df["quadrant_ad"] = [quadrant(t, pa) for t, pa in zip(df.ground_truth, df.pred_ad)]
    for j, name in enumerate(FEATURE_NAMES):
        df[name] = feats[:, j]
    A.save_table(df, p.path("master_table.parquet"))


def stage_shap(ctx: RunContext) -> None:
    p = ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    X = master[FEATURE_NAMES]
    parts = []
    for tag, col in [("zs", "p_spoof_zs"), ("ad", "p_spoof_ad")]:
        surr, fid = fit_surrogate(X, master[col].to_numpy(), seed=ctx.cfg.seed)
        joblib.dump({"surrogate": surr, "fidelity": fid}, p.path(f"surrogate_{tag}.joblib"))
        parts.append(shap_importance(surr, X, tag))
    imp = pd.concat(parts, ignore_index=True)
    A.save_table(imp, p.path("shap_importance.csv"))
    if _plots_on(ctx):
        from .plotting import plot_shap_compare, set_plot_style
        set_plot_style()
        plot_shap_compare(imp, p.path("figures"), top_n=ctx.cfg.shap.top_n)


def stage_occlusion(ctx: RunContext) -> None:
    cfg, p = ctx.cfg, ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    audios = list(np.load(p.path(f"audios_{cfg.data.eval_split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(p.path(f"srs_{cfg.data.eval_split}.npy"))]
    det = ctx.get_detector()
    logreg = joblib.load(p.path("d_ad.joblib"))
    _, p_ad_from_audio = make_p_spoof_ad(logreg, det)
    edges = mel_band_edges(cfg.occlusion.n_bands, cfg.occlusion.f_min, cfg.occlusion.f_max)
    rows = []
    for tag, quad_col, fn in [("zs", "quadrant_zs", lambda a, s: det.spoof_prob(a, s)),
                              ("ad", "quadrant_ad", p_ad_from_audio)]:
        idx = stratified_idx(master, quad_col, cfg.occlusion.per_quadrant, seed=cfg.seed)
        # A oclusão sempre filtra a 16 kHz (bordas < Nyquist para qualquer sr nativo).
        sub_audios = [to_16k_mono(audios[i], srs[i]) for i in idx]
        sub_srs = [16000] * len(sub_audios)
        drops = occlusion_drop(fn, sub_audios, sub_srs, edges)  # (n_bands,) de queda média
        for bi in range(len(edges) - 1):
            rows.append({"detector": tag, "band_hz_low": float(edges[bi]),
                         "band_hz_high": float(edges[bi + 1]),
                         "mean_p_spoof_drop": float(drops[bi])})
    occ = pd.DataFrame(rows)
    A.save_table(occ, p.path("occlusion_table.csv"))
    if _plots_on(ctx):
        from .plotting import plot_occlusion_bands, set_plot_style
        set_plot_style()
        dz = occ[occ.detector == "zs"]["mean_p_spoof_drop"].to_numpy()
        da = occ[occ.detector == "ad"]["mean_p_spoof_drop"].to_numpy()
        plot_occlusion_bands(edges, dz, da, p.path("figures"))


def stage_confirmatory(ctx: RunContext) -> None:
    p = ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    imp = A.load_table(p.path("shap_importance.csv"))
    top = (imp.groupby("feature")["mean_abs_shap"].max()
           .sort_values(ascending=False).head(ctx.cfg.shap.top_n).index.tolist())
    parts = [confirmatory_tests(master, "zs", "quadrant_zs", top),
             confirmatory_tests(master, "ad", "quadrant_ad", top)]
    A.save_table(pd.concat(parts, ignore_index=True), p.path("confirmatory_tests.csv"))


def stage_report(ctx: RunContext) -> None:
    p = ctx.paths
    master = A.load_table(p.path("master_table.parquet"))
    y = master["ground_truth"].to_numpy()
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    rows = []
    for tag in ("zs", "ad"):
        surr = joblib.load(p.path(f"surrogate_{tag}.joblib"))["fidelity"]
        eer, _ = compute_eer(master[f"p_spoof_{tag}"].to_numpy(), y)
        pred = master[f"pred_{tag}"].to_numpy()
        rows.append({"detector": tag, "eer": float(eer),
                     "mcc": float(matthews_corrcoef(y, pred)),
                     "accuracy": float(accuracy_score(y, pred)),
                     "surrogate_r2": surr["r2"], "surrogate_spearman": surr["spearman"]})
    A.save_table(pd.DataFrame(rows), p.path("performance_table.csv"))
    A.save_json({"stages": "complete", "config_hash": ctx.cfg.config_hash()},
                p.path("run_manifest.json"))


def _plots_on(ctx: RunContext) -> bool:
    return getattr(ctx, "plots", True)
