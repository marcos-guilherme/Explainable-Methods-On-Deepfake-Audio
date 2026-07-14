"""Testes confirmatórios sobre os MFCCs de maior SHAP, com correção FDR (Benjamini–Hochberg)."""
from __future__ import annotations

import pandas as pd
from scipy.stats import false_discovery_control, levene, ttest_ind


def confirmatory_tests(master: pd.DataFrame, detector_tag: str, quad_col: str,
                       top_features: list[str]) -> pd.DataFrame:
    """Welch (média, TN vs FP) e Levene (variância, TP vs FN) nos MFCCs do SHAP, com FDR.

    Returns:
        DataFrame com colunas `detector, feature, test, statistic, p_value, q_value_fdr`.
    """
    rows = []
    g_vn = master[master[quad_col] == "TN"]
    g_fp = master[master[quad_col] == "FP"]
    g_tp = master[master[quad_col] == "TP"]
    g_fn = master[master[quad_col] == "FN"]
    for f in top_features:
        if len(g_vn) > 1 and len(g_fp) > 1:
            t = ttest_ind(g_vn[f], g_fp[f], equal_var=False)
            rows.append({"detector": detector_tag, "feature": f, "test": "welch_TN_vs_FP",
                         "statistic": float(t.statistic), "p_value": float(t.pvalue)})
        if len(g_tp) > 2 and len(g_fn) > 2:
            lv = levene(g_tp[f], g_fn[f])
            rows.append({"detector": detector_tag, "feature": f, "test": "levene_TP_vs_FN",
                         "statistic": float(lv.statistic), "p_value": float(lv.pvalue)})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_value_fdr"] = false_discovery_control(df["p_value"].to_numpy())
    return df
