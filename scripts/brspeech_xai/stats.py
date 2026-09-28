"""Associação (Spearman intra-classe) e testes confirmatórios (Welch/Levene) sobre as
features de maior |ρ|, com correção FDR (Benjamini–Hochberg)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, levene, spearmanr, ttest_ind, wilcoxon

from .bands import band_index

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


def cross_spine_agreement(occlusion: pd.DataFrame,
                          spearman: pd.DataFrame) -> pd.DataFrame:
    """Concordância entre as duas espinhas na MESMA grade de bandas, por detector.

    Para cada detector correlaciona (Spearman) o perfil CAUSAL (|queda de oclusão|
    por banda) com o perfil ASSOCIATIVO (max|ρ| por banda, sobre μ/σ e classes).
    Um ρ alto indica que as bandas causalmente importantes são também as mais
    associadas ao score --- validade convergente verificável.

    Returns:
        DataFrame com colunas `detector, rho_causal_vs_assoc, p_value, n_bands`.
    """
    sp = spearman.assign(abs_rho=spearman["rho"].abs())
    sp["band"] = [band_index(f)[0] for f in sp["feature"]]
    rows = []
    for det in ("zs", "ad"):
        occ = occlusion[occlusion.detector == det].sort_values("band_hz_low")
        if occ.empty:
            continue
        causal = np.abs(occ["mean_p_spoof_drop"].to_numpy())
        spd = sp[sp.detector == det]
        assoc = spd.groupby("band")["abs_rho"].max()
        assoc = np.array([assoc.get(b, np.nan) for b in range(len(causal))])
        ok = ~np.isnan(assoc)
        res = spearmanr(causal[ok], assoc[ok]) if ok.sum() >= 3 else None
        rows.append({"detector": det,
                     "rho_causal_vs_assoc": float(res.statistic) if res else float("nan"),
                     "p_value": float(res.pvalue) if res else float("nan"),
                     "n_bands": int(ok.sum())})
    return pd.DataFrame(rows)


def band_assoc_strength(spearman: pd.DataFrame, detector_tag: str,
                        n_bands: int) -> np.ndarray:
    """max|ρ| associativo por banda (sobre μ/σ e classes) para um detector.

    Returns:
        Array (n_bands,) com a força de associação de cada banda; NaN se ausente.
    """
    sp = spearman[spearman.detector == detector_tag].assign(abs_rho=lambda d: d["rho"].abs())
    sp = sp.assign(band=[band_index(f)[0] for f in sp["feature"]])
    m = sp.groupby("band")["abs_rho"].max()
    return np.array([float(m.get(b, np.nan)) for b in range(n_bands)])


def band_assoc_signed(spearman: pd.DataFrame, detector_tag: str,
                      n_bands: int) -> np.ndarray:
    """ρ COM SINAL (o de maior |ρ|) por banda, para um detector.

    Complementa `band_assoc_strength` (que só devolve |ρ|): aqui preservamos o sentido
    da associação de cada banda. ρ>0 = mais energia na banda anda junto com mais P(spoof)
    (pista de spoof); ρ<0 = mais energia anda junto com menos P(spoof) (pista de bonafide).
    Usado para orientar o efeito causal da oclusão pelo sentido apontado pelo H1.

    Returns:
        Array (n_bands,) com o ρ (com sinal) de cada banda; NaN se ausente.
    """
    sp = spearman[spearman.detector == detector_tag].copy()
    sp["band"] = [band_index(f)[0] for f in sp["feature"]]
    sp["abs_rho"] = sp["rho"].abs()
    out = np.full(n_bands, np.nan)
    for band, grp in sp.groupby("band"):
        if 0 <= int(band) < n_bands:
            out[int(band)] = float(grp.loc[grp["abs_rho"].idxmax(), "rho"])
    return out


def convergence_bands(spearman: pd.DataFrame, detector_tag: str, n_bands: int,
                      k: int) -> tuple[list[int], list[int]]:
    """Bandas top-k e bottom-k por força de associação |ρ| (H1), para um detector.

    Usado para escolher, de forma independente da oclusão, quais bandas o H1 diz
    serem mais e menos associadas ao score, antes de testar o efeito causal delas.

    Returns:
        (top_bands, bottom_bands), índices 0-based ordenados por |ρ| decrescente
        (top) e crescente (bottom). Bandas com |ρ| ausente vão para o fim do ranking.
    """
    strength = band_assoc_strength(spearman, detector_tag, n_bands)
    order = np.argsort(np.nan_to_num(strength, nan=-np.inf))  # ascendente
    bottom = order[:k].tolist()
    top = order[::-1][:k].tolist()
    return top, bottom


def paired_intervention_test(drop_top: np.ndarray, drop_bottom: np.ndarray,
                             seed: int = 42, n_boot: int = 1000) -> dict:
    """Teste pareado por clipe da convergência H1→H2.

    Compara, no mesmo clipe, a queda de P(spoof) ao ocluir as bandas mais associadas
    (`drop_top`) versus as menos associadas (`drop_bottom`). Uma diferença positiva
    indica que as bandas apontadas pelo H1 também causam mais queda (as duas espinhas
    convergem). Reporta a diferença mediana com IC 95% (bootstrap de pares), o
    tamanho de efeito de Cohen d_z e o p de Wilcoxon (postos sinalizados).

    Returns:
        dict com n_pairs, median_diff, ci_low, ci_high, cohen_dz, wilcoxon_p,
        mean_drop_top, mean_drop_bottom.
    """
    a = np.asarray(drop_top, dtype=np.float64)
    b = np.asarray(drop_bottom, dtype=np.float64)
    diff = a - b
    n = int(len(diff))
    if n == 0:
        return {"n_pairs": 0, "median_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "cohen_dz": float("nan"),
                "wilcoxon_p": float("nan"), "mean_drop_top": float("nan"),
                "mean_drop_bottom": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.array([np.median(diff[rng.integers(0, n, n)]) for _ in range(n_boot)])
    sd = diff.std(ddof=1) if n > 1 else 0.0
    dz = float(diff.mean() / sd) if sd > 0 else float("nan")
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:  # todas as diferenças nulas => sem teste
        p = float("nan")
    return {"n_pairs": n, "median_diff": float(np.median(diff)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "cohen_dz": dz, "wilcoxon_p": p,
            "mean_drop_top": float(a.mean()), "mean_drop_bottom": float(b.mean())}


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
