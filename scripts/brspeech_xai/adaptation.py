"""Head de adaptação leve (sobre embeddings congelados) e scoring cross-fitted.

O encoder XLS-R permanece congelado; aqui só se ajusta um classificador leve
(regressão logística ou um MLP pequeno) e, opcionalmente, obtêm-se scores
out-of-fold (OOF) por K-fold, sem vazamento entre treino e avaliação do head.
Depende apenas de scikit-learn (sem torch), para cross-fitar rápido.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _make_logistic(seed: int):
    """Regressão logística: P(spoof) calibrado, baseline do estudo."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def _make_mlp(seed: int):
    """MLP pequeno (1 camada oculta): head mais expressivo."""
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=400,
                         early_stopping=True, n_iter_no_change=10, random_state=seed)


# Registry de heads: nome -> fábrica do classificador. Para plugar uma head nova (ex.:
# gradient boosting, SVM), basta adicionar uma entrada aqui. Contrato esperado da head:
# objeto estilo scikit-learn com fit(X, y), predict_proba(X) e classes_.
HEADS = {
    "logistic": _make_logistic,
    "mlp": _make_mlp,
}


def build_head(head: str = "logistic", seed: int = 42):
    """Constrói o head de adaptação: StandardScaler + classificador do registry ``HEADS``."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        make_clf = HEADS[head]
    except KeyError:
        raise ValueError(
            f"head desconhecido: {head!r}. Disponíveis: {sorted(HEADS)}"
        ) from None
    return make_pipeline(StandardScaler(), make_clf(seed))


def make_p_spoof_ad(head, embedder):
    """Fábrica do detector adaptado (D_ad): head treinada + encoder para pontuar áudio bruto.

    Returns:
        ``(p_spoof_ad, p_spoof_ad_from_audio)``:
          - ``p_spoof_ad(emb) -> np.ndarray``: P(spoof) para embeddings (N, D). Classe
            positiva = spoof (coluna 1).
          - ``p_spoof_ad_from_audio(audio, sr) -> float``: P(spoof) de um único áudio bruto
            (usado pela oclusão de D_ad), re-extraindo o embedding pelo encoder.
    """
    def p_spoof_ad(emb: np.ndarray) -> np.ndarray:
        return head.predict_proba(emb)[:, 1]

    def p_spoof_ad_from_audio(audio: np.ndarray, sr: int) -> float:
        emb = embedder.extract_embeddings([audio], [sr])
        return float(p_spoof_ad(emb)[0])

    return p_spoof_ad, p_spoof_ad_from_audio


def crossfit_oof_scores(emb: np.ndarray, y: np.ndarray,
                        make: Callable[[], object], n_splits: int = 5,
                        seed: int = 42, spoof_label: int = 1) -> np.ndarray:
    """P(spoof) out-of-fold para cada amostra via StratifiedKFold.

    Em cada fold, um head NOVO (via ``make()``) treina nos K−1 folds e pontua o
    fold retido. Assim nenhum score é produzido por um head que viu aquela amostra.
    """
    from sklearn.model_selection import StratifiedKFold

    emb = np.asarray(emb)
    y = np.asarray(y)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(emb, y):
        head = make()
        head.fit(emb[tr], y[tr])
        classes = list(head.classes_)
        col = classes.index(spoof_label) if spoof_label in classes else -1
        oof[va] = head.predict_proba(emb[va])[:, col]
    return oof
