"""Testes do head de adaptação e do scoring out-of-fold (só sklearn, sem torch)."""
import numpy as np
import pytest

from brspeech_xai.adaptation import build_head, crossfit_oof_scores, fit_head, score_head


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


def test_fit_head_and_score_head_roundtrip():
    x, y = _separable(n=60)
    head = fit_head(x, y, head="logistic", seed=2)
    scores = score_head(head, x)
    assert scores.shape == (len(y),)
    assert scores[y == 1].mean() > scores[y == 0].mean()


@pytest.mark.parametrize(
    "labels",
    [
        np.zeros(10, dtype=int),
        np.arange(10) % 3,
    ],
)
def test_fit_head_requires_both_binary_classes(labels):
    x = np.zeros((10, 4), dtype=float)
    with pytest.raises(ValueError, match="classes|binários"):
        fit_head(x, labels)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([[0.2, 0.8]]),
        np.array([[0.2, 0.8], [0.3, np.nan]]),
        np.array([[0.2, 0.8], [-0.1, 1.1]]),
    ],
)
def test_score_head_rejects_invalid_predict_proba(probabilities):
    class _Head:
        classes_ = np.array([0, 1])

        def predict_proba(self, _emb):
            return probabilities

    with pytest.raises(ValueError, match="predict_proba"):
        score_head(_Head(), np.zeros((2, 3)))


def test_crossfit_no_leakage_is_every_sample_scored():
    x, y = _separable(n=120)
    make = lambda: build_head("logistic", seed=1)
    oof = crossfit_oof_scores(x, y, make, n_splits=4, seed=1, spoof_label=1)
    assert np.isfinite(oof).all()


def test_crossfit_rejects_more_folds_than_minority_samples():
    x, y = _separable(n=8)
    make = lambda: build_head("logistic", seed=1)
    with pytest.raises(ValueError, match="n_splits"):
        crossfit_oof_scores(x, y, make, n_splits=5, seed=1)
