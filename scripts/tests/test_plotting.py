"""Smoke tests das figuras (backend Agg): rodam com dados sintéticos e salvam PDF+PNG."""
import numpy as np
import pandas as pd

from brspeech_xai import plotting as P
from brspeech_xai.bands import BAND_EDGES, N_BANDS
from brspeech_xai.stats import cross_spine_agreement

FEATS = ([f"band{k}_mean" for k in range(1, N_BANDS + 1)]
         + [f"band{k}_std" for k in range(1, N_BANDS + 1)])


def _master(n=120, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 2, size=n)
    df = pd.DataFrame({f: rng.normal(size=n) for f in FEATS})
    df["ground_truth"] = gt
    df["p_spoof_zs"] = np.clip(0.5 + 0.2 * (gt - 0.5) + 0.05 * rng.normal(size=n), 0, 1)
    df["p_spoof_ad"] = np.clip(0.5 + 0.3 * (gt - 0.5) + 0.05 * rng.normal(size=n), 0, 1)
    quads = rng.choice(["TP", "TN", "FP", "FN"], size=n)
    df["quadrant_zs"] = quads
    df["quadrant_ad"] = rng.choice(["TP", "TN", "FP", "FN"], size=n)
    return df


def _spearman_df():
    rows = []
    for d in ("zs", "ad"):
        for f in FEATS:
            for c in ("bonafide", "spoof"):
                rho = ((abs(hash((f, d, c))) % 100) / 100) - 0.5
                rows.append({"detector": d, "feature": f, "class": c, "rho": rho,
                             "p_value": 0.01, "n": 60, "q_value_fdr": 0.03})
    return pd.DataFrame(rows)


def _occ_df():
    rows = []
    edges = np.array([20.0, 1000.0, 3000.0, 7900.0])
    for d in ("zs", "ad"):
        for bi in range(len(edges) - 1):
            m = (bi - 1) * 0.02
            rows.append({"detector": d, "band_hz_low": edges[bi], "band_hz_high": edges[bi + 1],
                         "mean_p_spoof_drop": m, "ci_low": m - 0.01, "ci_high": m + 0.01, "n": 50})
    return edges, pd.DataFrame(rows)


def test_association_and_occlusion(tmp_path):
    P.set_plot_style()
    P.plot_association_profile(_spearman_df(), tmp_path, top_n=5)
    P.plot_association_profile_signed(_spearman_df(), tmp_path)
    edges, occ = _occ_df()
    P.plot_occlusion_bands(edges, occ, tmp_path)
    P.plot_occlusion_overlay(edges, occ, tmp_path)
    for name in ("association_profile_zs_vs_ad", "association_profile_signed_zs_vs_ad",
                 "occlusion_bands_zs_vs_ad", "occlusion_overlay_zs_vs_ad"):
        assert (tmp_path / f"{name}.pdf").exists() and (tmp_path / f"{name}.png").exists()


def _occ_full():
    """Tabela de oclusão nas N_BANDS bandas reais (para a figura de convergência)."""
    lows, highs = BAND_EDGES[:-1], BAND_EDGES[1:]
    rows = []
    for d in ("zs", "ad"):
        for b in range(N_BANDS):
            m = (b + 1) * 0.01
            rows.append({"detector": d, "band_hz_low": lows[b], "band_hz_high": highs[b],
                         "mean_p_spoof_drop": m, "ci_low": m - 0.005, "ci_high": m + 0.005,
                         "n": 50})
    return pd.DataFrame(rows)


def test_spine_convergence(tmp_path):
    P.set_plot_style()
    occ = _occ_full()
    spearman = _spearman_df()
    agree = cross_spine_agreement(occ, spearman)
    P.plot_spine_convergence(BAND_EDGES, occ, spearman, agree, tmp_path)
    assert (tmp_path / "spine_convergence_zs_vs_ad.pdf").exists()


def test_confirmatory_box_and_scatter_and_det(tmp_path):
    P.set_plot_style()
    master = _master()
    top = FEATS[:4]
    conf = pd.DataFrame([
        {"detector": "zs", "feature": top[0], "test": "welch_TN_vs_FP",
         "statistic": 1.0, "p_value": 0.001, "q_value_fdr": 0.004},
        {"detector": "zs", "feature": top[0], "test": "levene_TP_vs_FN",
         "statistic": 1.0, "p_value": 0.2, "q_value_fdr": 0.3},
    ])
    P.plot_confirmatory_box(master, conf, top, "zs", "quadrant_zs", tmp_path)
    assert (tmp_path / "confirmatory_box_zs.pdf").exists()

    P.plot_confirmatory_effects(master, conf, top, "zs", "quadrant_zs", tmp_path, n_boot=100)
    assert (tmp_path / "confirmatory_effects_zs.pdf").exists()

    spearman = pd.DataFrame([
        {"detector": d, "feature": f, "class": c, "rho": (0.3 if c == "spoof" else -0.2),
         "p_value": 0.01, "n": 60, "q_value_fdr": 0.03}
        for d in ("zs", "ad") for f in top for c in ("bonafide", "spoof")
    ])
    P.plot_spearman_scatter(master, spearman, tmp_path)
    assert (tmp_path / "spearman_scatter.pdf").exists()

    P.plot_det(master, tmp_path)
    assert (tmp_path / "det_zs_vs_ad.pdf").exists()
