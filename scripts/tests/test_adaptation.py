"""Testes do head de adaptação e do scoring out-of-fold (só sklearn, sem torch)."""
import numpy as np

from brspeech_xai.adaptation import build_head, crossfit_oof_scores


def _separable(n=200, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.normal(size=(n, dim))
    x[y == 1, 0] += 3.0  # sinal separável na 1a dimensão
    return x, y


def test_build_head_variants():
    for head in ("logistic", "mlp"):
        clf = build_head(head, seed=0)
        assert hasattr(clf, "fit") and hasattr(clf, "predict_proba")


def test_crossfit_oof_shape_and_quality():
    x, y = _separable()
    make = lambda: build_head("logistic", seed=0)
    oof = crossfit_oof_scores(x, y, make, n_splits=5, seed=0, spoof_label=1)
    assert oof.shape == (len(y),)
    assert not np.isnan(oof).any()
    assert (oof >= 0).all() and (oof <= 1).all()
    # dados separáveis: score médio da classe spoof > da bonafide
    assert oof[y == 1].mean() > oof[y == 0].mean()


def test_crossfit_no_leakage_is_every_sample_scored():
    x, y = _separable(n=120)
    make = lambda: build_head("logistic", seed=1)
    oof = crossfit_oof_scores(x, y, make, n_splits=4, seed=1, spoof_label=1)
    assert np.isfinite(oof).all()
