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
from sklearn.metrics import confusion_matrix  # noqa: E402

# Paleta segura para daltônicos (ordem estável entre figuras).
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Larguras de coluna do template ACM sigconf (polegadas).
COL_WIDTH_SINGLE = 3.33
COL_WIDTH_DOUBLE = 7.0


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


def plot_shap_compare(df: pd.DataFrame, figures_dir: str | Path, top_n: int = 10):
    """Barras horizontais das top-N features MFCC por mean(|SHAP|), D_zs vs D_ad."""
    top = (df.groupby("feature")["mean_abs_shap"].max()
             .sort_values(ascending=False).head(top_n).index.tolist())
    piv = df[df["feature"].isin(top)].pivot(index="feature", columns="detector",
                                            values="mean_abs_shap").loc[top]
    fig, ax = plt.subplots(figsize=(COL_WIDTH_SINGLE, 3.0))
    piv.plot.barh(ax=ax)
    ax.set_xlabel("mean(|SHAP|)"); ax.set_ylabel("MFCC feature"); ax.invert_yaxis()
    ax.legend(title="detector")
    fig.tight_layout()
    save_fig(fig, "shap_importance_zs_vs_ad", figures_dir)
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
                    label="95\\% CI")
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
