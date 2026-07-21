"""Associação (Spearman intra-classe) e testes confirmatórios (Welch/Levene) sobre os
MFCCs de maior |ρ|, com correção FDR (Benjamini–Hochberg)."""
from __future__ import annotations

import pandas as pd
from scipy.stats import false_discovery_control, levene, spearmanr, ttest_ind

# ground_truth: 1 = spoof, 0 = bonafide
_CLASS_NAME = {0: "bonafide", 1: "spoof"}


def confirmatory_tests(master: pd.DataFrame, detector_tag: str, quad_col: str,
                       top_features: list[str]) -> pd.DataFrame:
    """Welch (média, TN vs FP) e Levene (variância, TP vs FN) nos MFCCs salientes, com FDR.

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


def top_features_by_rho(spearman: pd.DataFrame, top_n: int) -> list[str]:
    """Seleciona as top-N features pela maior |ρ| (máximo sobre classes e detectores).

    Substitui a antiga seleção por SHAP: aqui a saliência é a força de associação
    monotônica intra-classe entre a feature e P(spoof).
    """
    strength = spearman.assign(abs_rho=spearman["rho"].abs())
    ranked = (strength.groupby("feature")["abs_rho"].max()
              .sort_values(ascending=False))
    return ranked.head(top_n).index.tolist()


def spearman_intraclass(master: pd.DataFrame, detector_tag: str,
                        features: list[str]) -> pd.DataFrame:
    """Spearman entre cada MFCC e P(spoof), DENTRO de cada classe (evita correlação
    espúria induzida pela separação bonafide/spoof), com FDR por detector.

    Returns:
        DataFrame com colunas `detector, feature, class, rho, p_value, n, q_value_fdr`.
    """
    score_col = f"p_spoof_{detector_tag}"
    rows = []
    for cls_val, cls_name in _CLASS_NAME.items():
        g = master[master["ground_truth"] == cls_val]
        if len(g) < 3:
            continue
        scores = g[score_col]
        for f in features:
            res = spearmanr(g[f], scores)
            rows.append({"detector": detector_tag, "feature": f, "class": cls_name,
                         "rho": float(res.statistic), "p_value": float(res.pvalue),
                         "n": int(len(g))})
    df = pd.DataFrame(rows)
    if len(df):
        df["q_value_fdr"] = false_discovery_control(df["p_value"].to_numpy())
    return df
