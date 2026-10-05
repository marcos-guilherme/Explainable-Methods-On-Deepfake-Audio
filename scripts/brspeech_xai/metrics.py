"""Métricas de detecção: EER, calibração de limiar e avaliação consolidada."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def _validate_scores_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if s.shape[0] != y.shape[0]:
        raise ValueError(
            f"scores e labels devem ter o mesmo comprimento: {s.shape[0]} vs {y.shape[0]}"
        )
    if s.shape[0] < min_samples:
        raise ValueError(f"amostras insuficientes: {s.shape[0]} < {min_samples}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores deve conter apenas valores finitos")
    if not np.all(np.isfinite(y)):
        raise ValueError("labels deve conter apenas valores finitos")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("labels devem ser binários (0=bonafide, 1=spoof)")
    if len(np.unique(y)) < 2:
        raise ValueError("labels devem conter as duas classes")
    return s, y.astype(int)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Calcula o Equal Error Rate e o limiar correspondente.

    Usa a curva ROC e procura o ponto onde a FNR (1 - TPR) mais se aproxima da FPR.

    Args:
        scores: Array (N,) com a probabilidade de spoof.
        labels: Array (N,) com os rótulos binários (1 = spoof, 0 = bonafide).

    Returns:
        Tupla `(eer, threshold)`: o EER em fração [0, 1] e o limiar que o atinge.
    """
    s, y = _validate_scores_labels(scores, labels)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    fnr = 1.0 - tpr
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("curva ROC não produziu limiar EER finito")
    candidates = np.flatnonzero(finite)
    idx = int(candidates[np.argmin(np.abs(fnr[finite] - fpr[finite]))])
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    source_split: str,
    detector: str,
    method: str = "eer",
) -> dict:
    """Calibra limiar somente a partir de scores/labels da fonte (nunca eval/target).

    Returns:
        Estrutura JSON-safe com threshold, EER de calibração, n e contagens por classe.
    """
    if method != "eer":
        raise ValueError(f"método de calibração desconhecido: {method!r}")
    s, y = _validate_scores_labels(scores, labels)
    eer, thr = compute_eer(s, y)
    return {
        "detector": str(detector),
        "source_split": str(source_split),
        "method": method,
        "threshold": float(thr),
        "calibration_eer": float(eer),
        "n": int(len(y)),
        "counts": {
            "bonafide": int((y == 0).sum()),
            "spoof": int((y == 1).sum()),
        },
    }


def _fixed_threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    if not np.isfinite(threshold):
        raise ValueError("threshold deve ser finito")
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    tpr = tp / pos if pos else float("nan")
    fpr = fp / neg if neg else float("nan")
    fnr = fn / pos if pos else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    condition: str = "integro",
    source: str = "eval",
    target: str = "eval",
) -> dict:
    """Separa métricas threshold-free (diagnóstico) de métricas com limiar fixo.

    ``threshold_free`` inclui ROC-AUC, average precision e EER diagnóstico — nenhum
    deles usa ``threshold`` para classificação. ``fixed_threshold`` reporta acurácia,
    MCC e taxas com o limiar já calibrado na fonte.
    """
    s, y = _validate_scores_labels(scores, labels)
    eer_diag, _ = compute_eer(s, y)
    return {
        "condition": condition,
        "source": source,
        "target": target,
        "threshold_free": {
            "roc_auc": float(roc_auc_score(y, s)),
            "average_precision": float(average_precision_score(y, s)),
            "eer_diagnostic": float(eer_diag),
        },
        "fixed_threshold": _fixed_threshold_metrics(s, y, float(threshold)),
    }


def evaluate(scores: np.ndarray, labels: np.ndarray, condition: str = "integro") -> dict[str, float]:
    """Wrapper legado: EER + MCC/acurácia no limiar-EER e em 0.5.

    Novos fluxos devem preferir ``calibrate_threshold`` + ``evaluate_at_threshold``.
    """
    s, y = _validate_scores_labels(scores, labels)
    eer, thr = compute_eer(s, y)
    pred_eer = (s >= thr).astype(int)
    pred_half = (s >= 0.5).astype(int)
    return {
        "condition": condition,
        "eer": eer * 100.0,
        "eer_threshold": thr,
        "mcc_eer": matthews_corrcoef(y, pred_eer),
        "acc_eer": accuracy_score(y, pred_eer),
        "mcc_0.5": matthews_corrcoef(y, pred_half),
        "acc_0.5": accuracy_score(y, pred_half),
    }


def quadrant(y_true: int, y_pred: int) -> str:
    """Rótulo do quadrante (positivo = spoof = 1)."""
    return {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(y_true), int(y_pred))]
