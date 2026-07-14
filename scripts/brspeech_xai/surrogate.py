"""Surrogate interpretável (Random Forest) sobre MFCCs + importância via SHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .features import MFCC_COLS


def fit_surrogate(X_mfcc, target_scores: np.ndarray, seed: int = 42):
    """Treina uma floresta que imita o score do detector a partir dos MFCCs.

    Split interno para reportar fidelidade honesta (R² e Spearman no held-out).

    Returns:
        (modelo, dict de fidelidade com chaves `r2` e `spearman`).
    """
    Xtr, Xte, ytr, yte = train_test_split(X_mfcc, target_scores, test_size=0.3, random_state=seed)
    rf = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    fidelity = {"r2": float(rf.score(Xte, yte)), "spearman": float(spearmanr(pred, yte).statistic)}
    return rf, fidelity


def shap_importance(surrogate, X_mfcc, detector_tag: str) -> pd.DataFrame:
    """Importância média |SHAP| por feature MFCC para um surrogate.

    Returns:
        DataFrame com colunas `detector`, `feature`, `mean_abs_shap` (26 linhas).
    """
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X_mfcc, check_additivity=False)
    feature_names = list(X_mfcc.columns) if hasattr(X_mfcc, "columns") else MFCC_COLS
    return pd.DataFrame({"detector": detector_tag, "feature": feature_names,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
