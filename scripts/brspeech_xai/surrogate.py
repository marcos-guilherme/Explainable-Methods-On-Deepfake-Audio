"""Surrogate interpretável (Random Forest) sobre MFCCs + importância via SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .features import MFCC_COLS


def fit_surrogate(X_mfcc, target_scores: np.ndarray, seed: int = 42,
                  n_estimators: int = 200, max_depth: int | None = 12):
    """Treina uma floresta que imita o score do detector a partir dos MFCCs.

    Split interno para reportar fidelidade honesta (R² e Spearman no held-out).
    `max_depth` limita a profundidade das árvores: além de regularizar, reduz
    drasticamente o custo do TreeSHAP (que cresce com profundidade × nº de árvores).

    Returns:
        (modelo, dict de fidelidade com chaves `r2` e `spearman`).
    """
    Xtr, Xte, ytr, yte = train_test_split(X_mfcc, target_scores, test_size=0.3, random_state=seed)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    fidelity = {"r2": float(rf.score(Xte, yte)), "spearman": float(spearmanr(pred, yte).statistic)}
    return rf, fidelity


def shap_importance(surrogate, X_mfcc, detector_tag: str, max_samples: int | None = 1000,
                    seed: int = 42) -> pd.DataFrame:
    """Importância média |SHAP| por feature MFCC para um surrogate.

    A importância é uma MÉDIA de |SHAP| sobre as linhas; para mantê-la barata em
    conjuntos grandes, explicamos no máximo `max_samples` linhas (amostradas de
    forma determinística). `max_samples=None` explica todas as linhas.

    Returns:
        DataFrame com colunas `detector`, `feature`, `mean_abs_shap` (26 linhas).
    """
    feature_names = list(X_mfcc.columns) if hasattr(X_mfcc, "columns") else MFCC_COLS
    X_expl = X_mfcc
    if max_samples is not None and len(X_mfcc) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X_mfcc), size=max_samples, replace=False))
        X_expl = X_mfcc.iloc[idx] if hasattr(X_mfcc, "iloc") else X_mfcc[idx]
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X_expl, check_additivity=False)
    return pd.DataFrame({"detector": detector_tag, "feature": feature_names,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
