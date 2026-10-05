import numpy as np
import pytest

from brspeech_xai.metrics import (
    calibrate_threshold,
    compute_eer,
    evaluate,
    evaluate_at_threshold,
    quadrant,
)


def test_perfect_separation_zero_eer():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    eer, thr = compute_eer(scores, labels)
    assert eer < 1e-6
    assert 0.2 <= thr <= 0.8


def test_quadrant_mapping():
    assert quadrant(1, 1) == "TP"
    assert quadrant(0, 0) == "TN"
    assert quadrant(0, 1) == "FP"
    assert quadrant(1, 0) == "FN"


def test_calibrate_threshold_keys():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    out = calibrate_threshold(scores, labels, "train", "ad")
    assert {"threshold", "calibration_eer", "n", "counts", "source_split"}.issubset(out)


def test_evaluate_at_threshold_has_both_sections():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    out = evaluate_at_threshold(scores, labels, 0.5)
    assert "threshold_free" in out and "fixed_threshold" in out


def test_constant_scores_produce_finite_eer_threshold():
    scores = np.full(20, 0.5)
    labels = np.array([0] * 10 + [1] * 10)
    eer, threshold = compute_eer(scores, labels)
    assert eer == pytest.approx(0.5)
    assert np.isfinite(threshold)


@pytest.mark.parametrize("threshold", [np.inf, -np.inf, np.nan])
def test_evaluate_at_threshold_rejects_nonfinite_threshold(threshold):
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="threshold"):
        evaluate_at_threshold(scores, labels, threshold)


def test_evaluate_keys():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    m = evaluate(scores, labels)
    assert {"eer", "eer_threshold"}.issubset(m.keys())
