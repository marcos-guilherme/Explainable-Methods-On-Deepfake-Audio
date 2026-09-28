"""Métricas de detecção: EER, avaliação consolidada e rótulo de quadrante."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Calcula o Equal Error Rate e o limiar correspondente.

    Usa a curva ROC e procura o ponto onde a FNR (1 - TPR) mais se aproxima da FPR.

    Args:
        scores: Array (N,) com a probabilidade de spoof.
        labels: Array (N,) com os rótulos binários (1 = spoof, 0 = bonafide).

    Returns:
        Tupla `(eer, threshold)`: o EER em fração [0, 1] e o limiar que o atinge.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def evaluate(scores: np.ndarray, labels: np.ndarray, condition: str = "integro") -> dict[str, float]:
    """Consolida EER, MCC e acurácia (em dois limiares) para um conjunto de scores.

    Returns:
        Dicionário com: `condition`, `eer` (%), `eer_threshold`, `mcc_eer`, `acc_eer`,
        `mcc_0.5`, `acc_0.5`.
    """
    eer, thr = compute_eer(scores, labels)
    pred_eer = (scores >= thr).astype(int)
    pred_half = (scores >= 0.5).astype(int)
    return {
        "condition": condition,
        "eer": eer * 100.0,
        "eer_threshold": thr,
        "mcc_eer": matthews_corrcoef(labels, pred_eer),
        "acc_eer": accuracy_score(labels, pred_eer),
        "mcc_0.5": matthews_corrcoef(labels, pred_half),
        "acc_0.5": accuracy_score(labels, pred_half),
    }


def quadrant(y_true: int, y_pred: int) -> str:
    """Rótulo do quadrante (positivo = spoof = 1)."""
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(y_true), int(y_pred))]
