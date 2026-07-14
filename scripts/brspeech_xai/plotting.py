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


def plot_occlusion_bands(edges, drops_zs, drops_ad, figures_dir: str | Path):
    """Queda de P(spoof) por banda de frequência (D_zs vs D_ad)."""
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(COL_WIDTH_SINGLE, 2.6))
    ax.plot(centers, drops_zs, marker="o", label="D_zs")
    ax.plot(centers, drops_ad, marker="s", label="D_ad")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Frequency band center (Hz)"); ax.set_ylabel(r"$\Delta$ P(spoof)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "occlusion_bands_zs_vs_ad", figures_dir)
    return fig
