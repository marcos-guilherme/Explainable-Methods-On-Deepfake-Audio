import numpy as np
import pandas as pd

from brspeech_xai.bands import BAND_EDGES, N_BANDS
from brspeech_xai.stats import (confirmatory_tests, cross_spine_agreement,
                                 spearman_intraclass, top_features_by_rho)


def test_confirmatory_runs_on_top_features():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "mfcc2_mean": rng.normal(size=n),
        "p_spoof_zs": rng.uniform(size=n),
        "quadrant_zs": rng.choice(["TP", "TN", "FP", "FN"], size=n),
    })
    out = confirmatory_tests(df, "zs", "quadrant_zs", ["mfcc2_mean"])
    assert {"detector", "feature", "test", "statistic", "p_value", "q_value_fdr"}.issubset(out.columns)


def test_spearman_intraclass_columns_and_sign():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    gt = rng.integers(0, 2, size=n)
    # score monotonicamente crescente com x dentro de cada classe -> rho positivo
    score = np.clip(0.5 + 0.1 * x + 0.01 * rng.normal(size=n), 0, 1)
    df = pd.DataFrame({"mfcc2_mean": x, "ground_truth": gt, "p_spoof_zs": score})
    out = spearman_intraclass(df, "zs", ["mfcc2_mean"])
    assert {"detector", "feature", "class", "rho", "p_value", "n", "q_value_fdr"}.issubset(out.columns)
    assert set(out["class"]) == {"bonafide", "spoof"}
    assert (out["rho"] > 0).all()


def test_top_features_by_rho_ranks_by_abs_max():
    df = pd.DataFrame([
        {"detector": "zs", "feature": "mfcc1_mean", "class": "spoof", "rho": 0.10},
        {"detector": "zs", "feature": "mfcc2_mean", "class": "spoof", "rho": -0.80},
        {"detector": "ad", "feature": "mfcc3_mean", "class": "bonafide", "rho": 0.50},
    ])
    assert top_features_by_rho(df, top_n=2) == ["mfcc2_mean", "mfcc3_mean"]


def test_cross_spine_agreement_aligned():
    lows, highs = BAND_EDGES[:-1], BAND_EDGES[1:]
    occ_rows, sp_rows = [], []
    for det in ("zs", "ad"):
        for b in range(N_BANDS):
            occ_rows.append({"detector": det, "band_hz_low": lows[b],
                             "band_hz_high": highs[b], "mean_p_spoof_drop": (b + 1) * 0.01})
            sp_rows.append({"detector": det, "feature": f"band{b + 1}_mean",
                            "class": "spoof", "rho": (b + 1) * 0.05})
    out = cross_spine_agreement(pd.DataFrame(occ_rows), pd.DataFrame(sp_rows))
    assert set(out["detector"]) == {"zs", "ad"}
    assert (out["rho_causal_vs_assoc"] > 0.9).all()   # perfis monotonicamente alinhados
    assert (out["n_bands"] == N_BANDS).all()
