import numpy as np

from brspeech_xai.metrics import compute_eer, evaluate, quadrant


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


def test_evaluate_keys():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    m = evaluate(scores, labels)
    assert {"eer", "eer_threshold"}.issubset(m.keys())
