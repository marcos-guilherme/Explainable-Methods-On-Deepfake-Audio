"""Figuras (padrão ACM sigconf), em backend não-interativo (Agg) para execução headless."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend headless; deve vir antes de importar pyplot

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from .features import mfcc_label  # noqa: E402

# Paleta segura para daltônicos (ordem estável entre figuras).
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Larguras de coluna do template ACM sigconf (polegadas).
COL_WIDTH_SINGLE = 3.33
COL_WIDTH_DOUBLE = 7.0

# Cores/rótulos fixos por detector e por classe (consistentes entre figuras).
DETECTOR_COLOR = {"zs": CB_PALETTE[0], "ad": CB_PALETTE[1]}
DETECTOR_LABEL = {"zs": r"$D_\mathrm{zs}$ (zero-shot)", "ad": r"$D_\mathrm{ad}$ (adapted)"}
CLASS_COLOR = {"bonafide": CB_PALETTE[0], "spoof": CB_PALETTE[1]}
# Ordem fixa dos quadrantes nos boxplots.
QUADRANT_ORDER = ["TN", "FP", "TP", "FN"]


def set_plot_style() -> None:
    """Define um estilo de figura consistente (padrão de paper ACM sigconf)."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.prop_cycle": mpl.cycler(color=CB_PALETTE),
    })


def save_fig(fig, name: str, figures_dir: str | Path) -> None:
    """Salva a figura em PDF vetorial (para o LaTeX) e PNG 300 DPI (preview)."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(figures_dir / f"{name}.png")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "figure"


def plot_confusion(scores: np.ndarray, labels: np.ndarray, threshold: float,
                   title: str, figures_dir: str | Path):
    """Matriz de confusão para um dado limiar de decisão (salva PDF + PNG)."""
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Bonafide", "Spoof"])
    ax.set_yticks([0, 1], ["Bonafide", "Spoof"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, f"confusion_{_slug(title)}", figures_dir)
    return fig


def _assoc_panel(ax, strength_panel: pd.DataFrame, top_n: int, title: str):
    """Painel de barras horizontais top-N por |ρ| (máx. sobre classes), D_zs vs D_ad."""
    top = (strength_panel.groupby("feature")["abs_rho"].max()
           .sort_values(ascending=False).head(top_n).index.tolist())
    piv = (strength_panel[strength_panel["feature"].isin(top)]
           .pivot(index="feature", columns="detector", values="abs_rho")
           .reindex(top))
    detectors = [d for d in ("zs", "ad") if d in piv.columns]
    y = np.arange(len(top))
    h = 0.8 / max(len(detectors), 1)
    for k, tag in enumerate(detectors):
        ax.barh(y + (k - (len(detectors) - 1) / 2) * h, piv[tag].to_numpy(), height=h,
                color=DETECTOR_COLOR[tag], label=DETECTOR_LABEL[tag])
    ax.set_yticks(y, [mfcc_label(f) for f in top])
    ax.invert_yaxis()
    ax.set_xlabel(r"max$_\mathrm{class}\,|\rho|$ (MFCC vs. $P$(spoof))")
    ax.set_title(title, fontsize=8, loc="left")


def plot_association_profile(spearman: pd.DataFrame, figures_dir: str | Path,
                             top_n: int = 10):
    """Perfil de associação MFCC↔P(spoof) (Espinha 1) em dois painéis: coeficientes
    de média (μ) e de desvio (σ). Força = |ρ| de Spearman intra-classe (máx. sobre
    classes), barras horizontais top-N por painel, D_zs vs D_ad (H1, descritiva)."""
    strength = (spearman.assign(abs_rho=spearman["rho"].abs())
                .groupby(["detector", "feature"])["abs_rho"].max().reset_index())
    is_std = strength["feature"].str.endswith("_std")
    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH_SINGLE, 4.6))
    _assoc_panel(axes[0], strength[~is_std], top_n, r"Mean coefficients ($\mu$)")
    _assoc_panel(axes[1], strength[is_std], top_n, r"Std. coefficients ($\sigma$)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "association_profile_zs_vs_ad", figures_dir)
    return fig


SUSTAIN_SPOOF_COLOR = "#d62728"     # queda positiva: ocluir derruba P(spoof)
SUSTAIN_BONAFIDE_COLOR = "#1f77b4"  # queda negativa: ocluir sobe P(spoof)


def _occlusion_panel(ax, centers, mean_drop, ci_low, ci_high, title):
    """Um painel divergente de oclusão espectral (área com sinal + faixa de IC)."""
    ax.fill_between(centers, 0, mean_drop, where=mean_drop >= 0, interpolate=True,
                    color=SUSTAIN_SPOOF_COLOR, alpha=0.45, label="Sustains spoof")
    ax.fill_between(centers, 0, mean_drop, where=mean_drop < 0, interpolate=True,
                    color=SUSTAIN_BONAFIDE_COLOR, alpha=0.45, label="Sustains bonafide")
    ax.fill_between(centers, ci_low, ci_high, color="#333333", alpha=0.15, linewidth=0,
                    label="95% CI")
    ax.plot(centers, mean_drop, color="#333333", lw=1.0, alpha=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    max_abs = float(np.max(np.abs(np.concatenate([ci_low, ci_high, mean_drop])))) or 1.0
    ax.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    ax.set_ylabel(r"$\Delta$ P(spoof)")
    ax.set_title(title, fontsize=8, loc="left")


def plot_occlusion_bands(edges, occ: pd.DataFrame, figures_dir: str | Path):
    """Perfil espectral divergente da oclusão por detector (D_zs em cima, D_ad embaixo).

    Área vermelha = banda sustenta spoof (ocluir derruba P(spoof)); azul = sustenta
    bonafide (ocluir sobe P(spoof)); faixa cinza = IC 95% (bootstrap) da queda média.
    Espera `occ` com colunas: detector, band_hz_low, band_hz_high, mean_p_spoof_drop,
    ci_low, ci_high.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    labels = {"zs": r"$D_\mathrm{zs}$ (zero-shot)", "ad": r"$D_\mathrm{ad}$ (adapted)"}
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_WIDTH_SINGLE, 4.2))
    for ax, tag in zip(axes, ["zs", "ad"]):
        sub = occ[occ.detector == tag].sort_values("band_hz_low")
        _occlusion_panel(ax, centers, sub["mean_p_spoof_drop"].to_numpy(),
                         sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(),
                         labels.get(tag, tag))
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Frequency band center (Hz)")
    fig.tight_layout()
    save_fig(fig, "occlusion_bands_zs_vs_ad", figures_dir)
    return fig


def plot_occlusion_overlay(edges, occ: pd.DataFrame, figures_dir: str | Path):
    """Comparação direta D_zs vs D_ad: curvas de queda média sobrepostas em um único
    eixo, cada uma com faixa de IC 95% (bootstrap). Complementa a figura divergente.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(COL_WIDTH_SINGLE, 2.8))
    for tag in ("zs", "ad"):
        sub = occ[occ.detector == tag].sort_values("band_hz_low")
        if sub.empty:
            continue
        mean = sub["mean_p_spoof_drop"].to_numpy()
        lo, hi = sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy()
        color = DETECTOR_COLOR[tag]
        ax.fill_between(centers, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax.plot(centers, mean, color=color, lw=1.4, marker="o", ms=3,
                label=DETECTOR_LABEL[tag])
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("Frequency band center (Hz)")
    ax.set_ylabel(r"$\Delta$ P(spoof) on occlusion")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "occlusion_overlay_zs_vs_ad", figures_dir)
    return fig


def _stars(q: float | None) -> str:
    """Estrelas de significância a partir do q-value (pós-FDR)."""
    if q is None or np.isnan(q):
        return "n/a"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def _lookup_q(conf: pd.DataFrame, detector: str, feature: str, test: str) -> float | None:
    sel = conf[(conf.detector == detector) & (conf.feature == feature)
               & (conf.test == test)]
    return float(sel["q_value_fdr"].iloc[0]) if len(sel) else None


def plot_confirmatory_box(master: pd.DataFrame, conf: pd.DataFrame,
                          top_features: list[str], detector_tag: str, quad_col: str,
                          figures_dir: str | Path):
    """Boxplots por quadrante das top-features (H3). Um subplot por feature; anota
    Welch (média, TN vs FP) e Levene (variância, TP vs FN) com estrelas pós-FDR."""
    n = len(top_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_WIDTH_DOUBLE, 2.2 * nrows),
                             squeeze=False)
    quad_colors = {"TN": CB_PALETTE[0], "FP": CB_PALETTE[1],
                   "TP": CB_PALETTE[2], "FN": CB_PALETTE[3]}
    for ax_i, feat in enumerate(top_features):
        ax = axes[ax_i // ncols][ax_i % ncols]
        data, present = [], []
        for q in QUADRANT_ORDER:
            vals = master.loc[master[quad_col] == q, feat].to_numpy()
            if len(vals):
                data.append(vals)
                present.append(q)
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, tick_labels=present, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, q in zip(bp["boxes"], present):
            patch.set_facecolor(quad_colors[q])
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        q_mean = _lookup_q(conf, detector_tag, feat, "welch_TN_vs_FP")
        q_var = _lookup_q(conf, detector_tag, feat, "levene_TP_vs_FN")
        ax.set_title(f"{mfcc_label(feat)}\nTN-FP {_stars(q_mean)} | TP-FN {_stars(q_var)}",
                     fontsize=7, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(DETECTOR_LABEL[detector_tag], fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"confirmatory_box_{detector_tag}", figures_dir)
    return fig


def plot_spearman_scatter(master: pd.DataFrame, spearman: pd.DataFrame,
                          figures_dir: str | Path):
    """Dispersão do MFCC de maior |ρ| vs P(spoof), por detector, colorido por classe.
    Anota ρ e q (pós-FDR) por classe. Associação intra-classe (evita efeito espúrio)."""
    detectors = [d for d in ("zs", "ad") if (spearman.detector == d).any()]
    if not detectors:
        return None
    fig, axes = plt.subplots(1, len(detectors), squeeze=False,
                             figsize=(COL_WIDTH_DOUBLE, 3.0))
    for k, tag in enumerate(detectors):
        ax = axes[0][k]
        sp = spearman[spearman.detector == tag]
        feat = sp.loc[sp["rho"].abs().idxmax(), "feature"]
        score_col = f"p_spoof_{tag}"
        for cls_val, cls_name in ((0, "bonafide"), (1, "spoof")):
            g = master[master["ground_truth"] == cls_val]
            ax.scatter(g[feat], g[score_col], s=6, alpha=0.35,
                       color=CLASS_COLOR[cls_name], label=cls_name, edgecolors="none")
        # anotação de ρ/q por classe
        lines = []
        for cls_name in ("bonafide", "spoof"):
            row = sp[(sp["feature"] == feat) & (sp["class"] == cls_name)]
            if len(row):
                rho = row["rho"].iloc[0]
                q = row["q_value_fdr"].iloc[0] if "q_value_fdr" in row else float("nan")
                lines.append(rf"{cls_name}: $\rho$={rho:.2f} ({_stars(q)})")
        ax.set_title(f"{DETECTOR_LABEL[tag]}\n{mfcc_label(feat)}", fontsize=8, loc="left")
        ax.set_xlabel(mfcc_label(feat))
        ax.set_ylabel("P(spoof)")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=6.5, bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0.4))
        ax.legend(loc="lower right", fontsize=6.5, markerscale=1.5)
    fig.tight_layout()
    save_fig(fig, "spearman_scatter", figures_dir)
    return fig


def plot_det(master: pd.DataFrame, figures_dir: str | Path):
    """Curva DET (FNR × FPR em eixos de desvio-normal) para D_zs e D_ad, com o ponto
    de EER marcado em cada curva (P0: pré-condição de desempenho)."""
    y = master["ground_truth"].to_numpy()
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    fig, ax = plt.subplots(figsize=(COL_WIDTH_SINGLE, 3.2))
    for tag in ("zs", "ad"):
        scores = master[f"p_spoof_{tag}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
        fnr = 1.0 - tpr
        m = (fpr > 0) & (fnr > 0)
        ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), color=DETECTOR_COLOR[tag],
                lw=1.4, label=DETECTOR_LABEL[tag])
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = (fpr[i] + fnr[i]) / 2.0
        ax.plot(norm.ppf(fpr[i]), norm.ppf(fnr[i]), "o", color=DETECTOR_COLOR[tag],
                ms=5, label=f"EER {eer * 100:.1f}%")
    ax.plot([norm.ppf(ticks[0]), norm.ppf(ticks[-1])],
            [norm.ppf(ticks[0]), norm.ppf(ticks[-1])], ls="--", color="grey", lw=0.7)
    ax.set_xticks([norm.ppf(t) for t in ticks], [f"{t * 100:g}" for t in ticks])
    ax.set_yticks([norm.ppf(t) for t in ticks], [f"{t * 100:g}" for t in ticks])
    ax.set_xlabel("False positive rate (%)")
    ax.set_ylabel("False negative rate (%)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "det_zs_vs_ad", figures_dir)
    return fig
