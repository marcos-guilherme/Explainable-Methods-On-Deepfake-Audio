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


def test_fit_surrogate_respects_max_depth():
    X, target = _toy_frame()
    surr, _ = fit_surrogate(X, target, n_estimators=20, max_depth=4)
    assert surr.n_estimators == 20
    assert all(est.get_depth() <= 4 for est in surr.estimators_)


def test_shap_importance_shape():
    X, target = _toy_frame()
    surr, _ = fit_surrogate(X, target)
    imp = shap_importance(surr, X, "zs")
    assert set(["detector", "feature", "mean_abs_shap"]).issubset(imp.columns)
    assert len(imp) == 26


def test_shap_importance_subsamples_rows():
    # 26 features de importância independem de quantas linhas foram explicadas.
    X, target = _toy_frame(n=300)
    surr, _ = fit_surrogate(X, target, n_estimators=20, max_depth=4)
    imp_full = shap_importance(surr, X, "zs", max_samples=None)
    imp_sub = shap_importance(surr, X, "zs", max_samples=50)
    assert len(imp_full) == 26 and len(imp_sub) == 26
