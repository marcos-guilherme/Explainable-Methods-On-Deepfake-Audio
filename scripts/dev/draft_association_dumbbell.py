"""Ideia guardada (nao usada no pipeline): Figura de associacao no formato
dumbbell com seta zs->ad (Abordagem A do brainstorming de 2026-08-10). Preferimos
as barras por enquanto. Gera um PNG em /tmp para avaliacao.

Uso: PYTHONPATH=scripts python scripts/dev/draft_association_dumbbell.py [run_dir]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brspeech_xai.plotting import (
    set_plot_style, band_label, band_index, DETECTOR_COLOR, DETECTOR_LABEL,
    COL_WIDTH_DOUBLE,
)

FLIP = "#c1121f"       # seta que inverte de sentido (vermelho)
SAME = "#b9b9b9"       # seta que mantem o sentido (cinza)
WEAK = 0.10            # zona "fraco"

import sys
run = sys.argv[1] if len(sys.argv) > 1 else "results/default-20260720-191814"
sp = pd.read_csv(f"{run}/spearman_table.csv")
set_plot_style()

sp = sp.assign(abs_rho=sp["rho"].abs())
idx = sp.groupby(["detector", "feature"])["abs_rho"].idxmax()
signed = sp.loc[idx, ["detector", "feature", "rho"]].rename(columns={"rho": "signed_rho"})
is_std = signed["feature"].str.endswith("_std")
order = lambda df: sorted(df["feature"].unique(), key=lambda f: band_index(f)[0])
mean_feats, std_feats = order(signed[~is_std]), order(signed[is_std])
lim = 1.15 * max(float(np.nanmax(np.abs(signed["signed_rho"].to_numpy()))), 0.05)


def panel(ax, df, feats, title):
    piv = df.pivot(index="feature", columns="detector", values="signed_rho").reindex(feats)
    y = np.arange(len(feats))
    ax.axvspan(-WEAK, WEAK, color="gray", alpha=0.12, lw=0, zorder=0)
    ax.axvline(0.0, color="black", lw=1.0, ls="--", zorder=1)
    a = piv["zs"].to_numpy(float)
    b = piv["ad"].to_numpy(float)
    for i in range(len(feats)):
        if not (np.isfinite(a[i]) and np.isfinite(b[i])):
            continue
        conn = FLIP if a[i] * b[i] < 0 else SAME
        ax.annotate("", xy=(b[i], y[i]), xytext=(a[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=conn,
                                    lw=2.2 if conn == FLIP else 1.4,
                                    shrinkA=3, shrinkB=3), zorder=2)
    ax.scatter(a, y, s=42, color=DETECTOR_COLOR["zs"], zorder=3,
               label=DETECTOR_LABEL.get("zs", "zero-shot"))
    ax.scatter(b, y, s=42, color=DETECTOR_COLOR["ad"], zorder=3,
               label=DETECTOR_LABEL.get("ad", "adaptado"))
    ax.set_yticks(y, [band_label(f).split("\u00b7")[0] for f in feats], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(feats) - 0.4, -0.6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(0.995, 1.02, "spoof \u2192", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7, color="#555555")
    ax.text(0.005, 1.02, "\u2190 bonafide", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7, color="#555555")


fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_WIDTH_DOUBLE, 6.4))
panel(axes[0], signed[~is_std], mean_feats, "Energia m\u00e9dia (\u03bc)")
panel(axes[1], signed[is_std], std_feats, "Variabilidade no tempo (\u03c3)")
axes[-1].set_xlabel(r"$\rho$ de Spearman com sinal   ($\to$ spoof, $\leftarrow$ bonafide)")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([0], [0], marker="o", ls="", color=DETECTOR_COLOR["zs"], label="zero-shot"),
    Line2D([0], [0], marker="o", ls="", color=DETECTOR_COLOR["ad"], label="adaptado"),
    Line2D([0], [0], color=FLIP, lw=2.2, label="inverte de sentido"),
    Patch(color="gray", alpha=0.12, label="fraco (|\u03c1|<0,1)"),
]
axes[0].legend(handles=handles, loc="lower right", fontsize=6.5, ncol=2, framealpha=0.9)
fig.suptitle("Associa\u00e7\u00e3o por faixa: o sentido muda do zero-shot para o adaptado (H1)",
             fontsize=9.5)
fig.tight_layout()
out = "/tmp/draft_dumbbell.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
