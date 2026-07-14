import numpy as np
import pandas as pd

from brspeech_xai.surrogate import fit_surrogate, shap_importance

FEATURES = [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 26)), columns=FEATURES)
    target = (X["mfcc1_mean"] * 0.8 + rng.normal(scale=0.1, size=n)).to_numpy()
    return X, target


def test_fit_surrogate_reports_fidelity():
    X, target = _toy_frame()
    surr, fid = fit_surrogate(X, target)
    assert "r2" in fid and "spearman" in fid
    assert -1.0 <= fid["r2"] <= 1.0


def test_shap_importance_shape():
    X, target = _toy_frame()
    surr, _ = fit_surrogate(X, target)
    imp = shap_importance(surr, X, "zs")
    assert set(["detector", "feature", "mean_abs_shap"]).issubset(imp.columns)
    assert len(imp) == 26
