"""Head de adaptação leve (sobre embeddings congelados) e scoring cross-fitted.

O encoder XLS-R permanece congelado; aqui só se ajusta um classificador leve
(regressão logística ou um MLP pequeno) e, opcionalmente, obtêm-se scores
out-of-fold (OOF) por K-fold, sem vazamento entre treino e avaliação do head.
Depende apenas de scikit-learn (sem torch), para cross-fitar rápido.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .data import SPOOF_LABEL


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


def _validate_fit_inputs(emb_train: np.ndarray, y_train: np.ndarray) -> None:
    emb = np.asarray(emb_train)
    y = np.asarray(y_train)
    if emb.ndim != 2:
        raise ValueError(f"emb_train deve ser 2D, obteve shape {emb.shape}")
    if y.ndim != 1:
        raise ValueError(f"y_train deve ser 1D, obteve shape {y.shape}")
    if emb.shape[0] != y.shape[0]:
        raise ValueError(
            f"emb_train e y_train devem ter o mesmo nº de linhas: "
            f"{emb.shape[0]} vs {y.shape[0]}"
        )
    if emb.shape[0] == 0:
        raise ValueError("emb_train não pode ser vazio")
    if not np.all(np.isfinite(emb)):
        raise ValueError("emb_train deve conter apenas valores finitos")
    if not np.all(np.isfinite(y)):
        raise ValueError("y_train deve conter apenas valores finitos")
    classes = set(np.unique(y).tolist())
    if not classes.issubset({0, 1}):
        raise ValueError("y_train deve conter rótulos binários (0=bonafide, 1=spoof)")
    if classes != {0, 1}:
        raise ValueError("y_train deve conter as duas classes binárias")


def _validate_score_inputs(head, emb: np.ndarray) -> None:
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"emb deve ser 2D, obteve shape {emb.shape}")
    if emb.shape[0] == 0:
        raise ValueError("emb não pode ser vazio")
    if not np.all(np.isfinite(emb)):
        raise ValueError("emb deve conter apenas valores finitos")
    if not hasattr(head, "predict_proba"):
        raise ValueError("head deve expor predict_proba após fit")


def spoof_column_index(head, spoof_label: int = SPOOF_LABEL) -> int:
    """Índice da coluna P(spoof) em ``predict_proba``, via ``head.classes_``."""
    classes = list(getattr(head, "classes_", []))
    if not classes:
        raise ValueError("head não treinado ou sem classes_")
    if spoof_label not in classes:
        raise ValueError(
            f"spoof_label={spoof_label} ausente em classes_={classes}"
        )
    return classes.index(spoof_label)


def fit_head(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    *,
    head: str = "logistic",
    seed: int = 42,
):
    """Ajusta o head somente em ``(emb_train, y_train)``; não pontua nem calibra."""
    _validate_fit_inputs(emb_train, y_train)
    model = build_head(head, seed)
    model.fit(np.asarray(emb_train), np.asarray(y_train))
    return model


def score_head(head, emb: np.ndarray, *, spoof_label: int = SPOOF_LABEL) -> np.ndarray:
    """Pontua embeddings com head já ajustado; não refaz fit."""
    _validate_score_inputs(head, emb)
    col = spoof_column_index(head, spoof_label)
    probabilities = np.asarray(head.predict_proba(np.asarray(emb)), dtype=np.float64)
    n_classes = len(getattr(head, "classes_", []))
    if probabilities.ndim != 2 or probabilities.shape != (len(emb), n_classes):
        raise ValueError(
            "predict_proba retornou shape inválido: "
            f"{probabilities.shape}, esperado {(len(emb), n_classes)}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predict_proba retornou probabilidades não finitas")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("predict_proba retornou valores fora de [0, 1]")
    return probabilities[:, col]


def make_p_spoof_ad(head, embedder):
    """Fábrica do detector adaptado (D_ad): head treinada + encoder para pontuar áudio bruto.

    Returns:
        ``(p_spoof_ad, p_spoof_ad_from_audio)``:
          - ``p_spoof_ad(emb) -> np.ndarray``: P(spoof) para embeddings (N, D).
          - ``p_spoof_ad_from_audio(audio, sr) -> float``: P(spoof) de um único áudio bruto
            (usado pela oclusão de D_ad), re-extraindo o embedding pelo encoder.
    """
    def p_spoof_ad(emb: np.ndarray) -> np.ndarray:
        return score_head(head, emb)

    def p_spoof_ad_from_audio(audio: np.ndarray, sr: int) -> float:
        emb = embedder.extract_embeddings([audio], [sr])
        return float(p_spoof_ad(emb)[0])

    return p_spoof_ad, p_spoof_ad_from_audio


def crossfit_oof_scores(emb: np.ndarray, y: np.ndarray,
                        make: Callable[[], object], n_splits: int = 5,
                        seed: int = 42, spoof_label: int = SPOOF_LABEL) -> np.ndarray:
    """P(spoof) out-of-fold para cada amostra via StratifiedKFold.

    Em cada fold, um head NOVO (via ``make()``) treina nos K−1 folds e pontua o
    fold retido. Assim nenhum score é produzido por um head que viu aquela amostra.
    """
    from sklearn.model_selection import StratifiedKFold

    emb = np.asarray(emb)
    y = np.asarray(y)
    _validate_fit_inputs(emb, y)
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits deve ser inteiro >= 2")
    minority_count = int(np.min(np.unique(y, return_counts=True)[1]))
    if n_splits > minority_count:
        raise ValueError(
            f"n_splits={n_splits} excede amostras da classe minoritária={minority_count}"
        )
    oof = np.full(len(y), np.nan, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(emb, y):
        head = make()
        head.fit(emb[tr], y[tr])
        oof[va] = score_head(head, emb[va], spoof_label=spoof_label)
    if not np.all(np.isfinite(oof)):
        raise ValueError("cross-fit não produziu scores finitos para todas as amostras")
    return oof
