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

from .bands import band_index, band_label  # noqa: E402
from .metrics import compute_eer  # noqa: E402

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


def _assoc_panel(ax, strength_panel: pd.DataFrame, title: str):
    """Painel de barras de |ρ| por banda (ordenado por frequência), D_zs vs D_ad."""
    feats = sorted(strength_panel["feature"].unique(), key=lambda f: band_index(f)[0])
    piv = (strength_panel.pivot(index="feature", columns="detector", values="abs_rho")
           .reindex(feats))
    detectors = [d for d in ("zs", "ad") if d in piv.columns]
    x = np.arange(len(feats))
    w = 0.8 / max(len(detectors), 1)
    for k, tag in enumerate(detectors):
        ax.bar(x + (k - (len(detectors) - 1) / 2) * w, piv[tag].to_numpy(), width=w,
               color=DETECTOR_COLOR[tag], label=DETECTOR_LABEL[tag])
    ax.set_xticks(x, [band_label(f).split("\u00b7")[0] for f in feats],
                  rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel(r"max$_\mathrm{class}\,|\rho|$")
    ax.set_title(title, fontsize=8, loc="left")


def plot_association_profile(spearman: pd.DataFrame, figures_dir: str | Path,
                             top_n: int = 10):
    """Perfil de associação energia-de-banda↔P(spoof) (Espinha 1), ordenado por
    frequência, em dois painéis: energia média (μ) e desvio (σ) por banda. Força =
    |ρ| de Spearman intra-classe (máx. sobre classes), D_zs vs D_ad (H1, descritiva).
    O argumento `top_n` é mantido por compatibilidade e ignorado (mostra todas)."""
    strength = (spearman.assign(abs_rho=spearman["rho"].abs())
                .groupby(["detector", "feature"])["abs_rho"].max().reset_index())
    is_std = strength["feature"].str.endswith("_std")
    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH_SINGLE, 4.6))
    _assoc_panel(axes[0], strength[~is_std], r"Band energy $\mu$")
    _assoc_panel(axes[1], strength[is_std], r"Band energy $\sigma$")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "association_profile_zs_vs_ad", figures_dir)
    return fig


def plot_association_profile_single(spearman: pd.DataFrame, detector_tag: str,
                                    figures_dir: str | Path):
    """Perfil de associação |ρ| por banda para UM detector (painéis μ e σ).

    Mesma leitura do `plot_association_profile`, mas isolando um detector para
    inspecioná-lo sem as barras do outro por cima. Salva `association_profile_{tag}`
    (H1, descritiva). Força = |ρ| de Spearman intra-classe (máx. sobre classes)."""
    strength = (spearman[spearman.detector == detector_tag]
                .assign(abs_rho=lambda d: d["rho"].abs())
                .groupby(["detector", "feature"])["abs_rho"].max().reset_index())
    is_std = strength["feature"].str.endswith("_std")
    lab = DETECTOR_LABEL[detector_tag]
    dot = " \u00b7 "
    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH_SINGLE, 4.6))
    _assoc_panel(axes[0], strength[~is_std], lab + dot + r"Band energy $\mu$")
    _assoc_panel(axes[1], strength[is_std], lab + dot + r"Band energy $\sigma$")
    fig.tight_layout()
    save_fig(fig, f"association_profile_{detector_tag}", figures_dir)
    return fig


_FLIP_SHADE = "#f2c744"


def _signed_assoc_panel(ax, panel_df: pd.DataFrame, feats: list[str], title: str,
                        show_xlabels: bool = True):
    """Painel de barras de ρ COM SINAL por banda (+ pra cima = spoof, - pra baixo =
    bonafide), D_zs vs D_ad, ordenado por frequência. Faixas onde os dois detectores
    têm sinais opostos ficam sombreadas (a associação inverte)."""
    piv = (panel_df.pivot(index="feature", columns="detector", values="signed_rho")
           .reindex(feats))
    detectors = [d for d in ("zs", "ad") if d in piv.columns]
    x = np.arange(len(feats))
    if len(detectors) == 2:
        a, b = piv["zs"].to_numpy(float), piv["ad"].to_numpy(float)
        for i in range(len(feats)):
            if np.isfinite(a[i]) and np.isfinite(b[i]) and a[i] * b[i] < 0:
                ax.axvspan(i - 0.5, i + 0.5, color=_FLIP_SHADE, alpha=0.22, lw=0,
                           zorder=0)
    w = 0.8 / max(len(detectors), 1)
    for k, tag in enumerate(detectors):
        ax.bar(x + (k - (len(detectors) - 1) / 2) * w, piv[tag].to_numpy(), width=w,
               color=DETECTOR_COLOR[tag], label=DETECTOR_LABEL[tag], zorder=3)
    ax.axhline(0.0, color="black", lw=1.0, zorder=2)
    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels([band_label(f).split("\u00b7")[0] for f in feats],
                           rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"$\rho$ com sinal")
    ax.set_title(title, fontsize=9, loc="left")


def plot_association_profile_signed(spearman: pd.DataFrame, figures_dir: str | Path):
    """Perfil de associação COM SINAL e em formato largo (H1, descritiva).

    Diferente de `plot_association_profile` (que mostra |ρ|), aqui a barra guarda o
    sinal do ρ da classe de maior |ρ|: pra cima quando mais energia acompanha score
    de spoof (ρ>0) e pra baixo quando acompanha bonafide (ρ<0). Dois painéis: energia
    média (μ) e desvio (σ) por banda, D_zs vs D_ad."""
    sp = spearman.assign(abs_rho=spearman["rho"].abs())
    idx = sp.groupby(["detector", "feature"])["abs_rho"].idxmax()
    signed = (sp.loc[idx, ["detector", "feature", "rho"]]
              .rename(columns={"rho": "signed_rho"}))
    is_std = signed["feature"].str.endswith("_std")
    order = lambda df: sorted(df["feature"].unique(), key=lambda f: band_index(f)[0])
    mean_feats, std_feats = order(signed[~is_std]), order(signed[is_std])
    lim = 1.15 * max(float(np.nanmax(np.abs(signed["signed_rho"].to_numpy()))), 0.05)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_WIDTH_DOUBLE, 5.2))
    _signed_assoc_panel(axes[0], signed[~is_std], mean_feats,
                        r"Energia média ($\mu$)", show_xlabels=False)
    _signed_assoc_panel(axes[1], signed[is_std], std_feats,
                        r"Variabilidade no tempo ($\sigma$)", show_xlabels=True)
    for ax in axes:
        ax.set_ylim(-lim, lim)
        ax.text(0.995, 0.95, "(+) rumo a spoof", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#555555")
        ax.text(0.995, 0.05, "(\u2212) rumo a bonafide", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#555555")
    from matplotlib.patches import Patch
    handles = [Patch(color=DETECTOR_COLOR["zs"], label="zero-shot"),
               Patch(color=DETECTOR_COLOR["ad"], label="adaptado"),
               Patch(color=_FLIP_SHADE, alpha=0.5, label="inverte de sentido")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, ncol=3)
    axes[-1].set_xlabel("Faixa de frequência (kHz)")
    fig.tight_layout()
    save_fig(fig, "association_profile_signed_zs_vs_ad", figures_dir)
    return fig


def plot_energy_distribution(master: pd.DataFrame, figures_dir: str | Path):
    """Distribuição de energia log-mel por faixa de frequência, bonafide vs spoof.

    Dois painéis (energia média μ e variabilidade σ): média + faixa do IQR (q25–q75)
    por classe, com o tamanho de efeito (Cohen's d de spoof−bonafide) anotado por banda
    para ancorar quão diferente é cada faixa. A linha central é a média, coerente com o
    sinal de d. Verdade de base acústica que motiva H1/H2."""
    bands = list(range(1, 9))
    x = np.arange(len(bands))
    freq = [band_label(f"band{i}_mean").split("\u00b7")[0] for i in bands]
    bon = master[master["ground_truth"] == 0]
    spo = master[master["ground_truth"] == 1]
    bon_c, spo_c = CLASS_COLOR["bonafide"], CLASS_COLOR["spoof"]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_WIDTH_DOUBLE, 6.0))
    panels = [("mean", axes[0], r"Energia média ($\mu$) por banda"),
              ("std", axes[1], r"Variabilidade no tempo ($\sigma$) por banda")]
    for suffix, ax, title in panels:
        cols = [f"band{i}_{suffix}" for i in bands]
        for df, col, lab in [(bon, bon_c, "bonafide"), (spo, spo_c, "spoof")]:
            mean = df[cols].mean().to_numpy()
            q1 = df[cols].quantile(0.25).to_numpy()
            q3 = df[cols].quantile(0.75).to_numpy()
            ax.fill_between(x, q1, q3, color=col, alpha=0.18, lw=0)
            ax.plot(x, mean, color=col, lw=2, marker="o", ms=4, label=lab, zorder=3)
        trans = ax.get_xaxis_transform()
        for i, c in enumerate(cols):
            d = _cohens_d(spo[c].to_numpy(), bon[c].to_numpy())
            strong = bool(np.isfinite(d) and abs(d) >= 0.2)
            color = (spo_c if d > 0 else bon_c) if strong else "#9a9a9a"
            ax.text(i, 0.965, f"d={d:+.2f}", transform=trans, ha="center", va="top",
                    fontsize=6.3, color=color,
                    fontweight="bold" if strong else "normal")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("energia log-mel")
        ax.margins(y=0.14)
    axes[0].legend(loc="lower left", fontsize=8)
    axes[-1].set_xticks(x, freq, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Faixa de frequência (kHz)")
    fig.suptitle("Distribuição de energia por frequência: bonafide vs spoof  "
                 r"(média + IQR; $d$ = spoof$-$bonafide por banda)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "energy_distribution_bonafide_vs_spoof", figures_dir)
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
    tags = [t for t in ("zs", "ad") if not occ[occ.detector == t].empty]
    fig, axes = plt.subplots(len(tags), 1, sharex=True, squeeze=False,
                             figsize=(COL_WIDTH_SINGLE, 2.1 * len(tags)))
    axes = axes[:, 0]
    for ax, tag in zip(axes, tags):
        sub = occ[occ.detector == tag].sort_values("band_hz_low")
        _occlusion_panel(ax, centers, sub["mean_p_spoof_drop"].to_numpy(),
                         sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy(),
                         DETECTOR_LABEL[tag])
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
    if conf is None or len(conf) == 0 or "detector" not in conf.columns:
        return None
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
        ax.set_title(f"{band_label(feat)}\nTN-FP {_stars(q_mean)} | TP-FN {_stars(q_var)}",
                     fontsize=7, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(DETECTOR_LABEL[detector_tag], fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"confirmatory_box_{detector_tag}", figures_dir)
    return fig


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Diferença de médias padronizada (pooled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def _log2_std_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """log2 da razão de desvios-padrão (a/b): >0 => a mais variável que b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    return float(np.log2(sa / sb)) if sa > 0 and sb > 0 else np.nan


def _boot_ci(a, b, stat, n_boot=1000, seed=42):
    """IC 95% percentil por bootstrap reamostrando cada grupo (a, b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        vals[i] = stat(ra, rb)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def plot_confirmatory_effects(master: pd.DataFrame, conf: pd.DataFrame,
                              features: list[str], detector_tag: str, quad_col: str,
                              figures_dir: str | Path, n_boot: int = 1000, seed: int = 42):
    """H3 em tamanho de efeito (forest plot): uma linha por banda, dois painéis.

    Esquerda: Cohen's d (média TN vs FP) com IC 95%. Direita: log2 da razão de
    desvios (variância TP vs FN) com IC 95%. Marcador cheio = q<0,05 (pós-FDR);
    vazio = não significativo. Linha no zero = sem efeito. Foca magnitude+direção,
    complementando (e sendo mais honesto que) os boxplots."""
    feats = sorted(features, key=lambda f: band_index(f))
    y = np.arange(len(feats))[::-1]  # primeira feature no topo
    d_pt, d_lo, d_hi, d_sig = [], [], [], []
    r_pt, r_lo, r_hi, r_sig = [], [], [], []
    for f in feats:
        tn = master.loc[master[quad_col] == "TN", f].to_numpy()
        fp = master.loc[master[quad_col] == "FP", f].to_numpy()
        tp = master.loc[master[quad_col] == "TP", f].to_numpy()
        fn = master.loc[master[quad_col] == "FN", f].to_numpy()
        d_pt.append(_cohens_d(tn, fp))
        lo, hi = _boot_ci(tn, fp, _cohens_d, n_boot, seed)
        d_lo.append(lo); d_hi.append(hi)
        r_pt.append(_log2_std_ratio(tp, fn))
        lo, hi = _boot_ci(tp, fn, _log2_std_ratio, n_boot, seed)
        r_lo.append(lo); r_hi.append(hi)
        qm = _lookup_q(conf, detector_tag, f, "welch_TN_vs_FP")
        qv = _lookup_q(conf, detector_tag, f, "levene_TP_vs_FN")
        d_sig.append(qm is not None and qm < 0.05)
        r_sig.append(qv is not None and qv < 0.05)

    fig, (axL, axR) = plt.subplots(1, 2, sharey=True,
                                   figsize=(COL_WIDTH_DOUBLE, 0.46 * len(feats) + 1.8))
    panels = [
        (axL, d_pt, d_lo, d_hi, d_sig, CB_PALETTE[1],
         "Does average energy differ?",
         "when real audio is wrongly rejected (TN vs FP)",
         r"Cohen's $d$   ($\approx$0: groups barely differ)"),
        (axR, r_pt, r_lo, r_hi, r_sig, CB_PALETTE[2],
         "Does variability differ?",
         "when a spoof slips through (TP vs FN)",
         r"$\log_2$ std ratio   (>0: caught spoof varies more)")]
    for ax, pt, lo, hi, sig, color, title, subtitle, xlabel in panels:
        pt = np.array(pt, float)
        err = np.array([pt - np.array(lo, float), np.array(hi, float) - pt])
        for i in range(len(feats)):
            filled = sig[i]
            ax.errorbar(pt[i], y[i], xerr=[[err[0, i]], [err[1, i]]], fmt="o", ms=5,
                        color=color, ecolor=color, elinewidth=1.2, capsize=2.5,
                        markerfacecolor=color if filled else "white",
                        markeredgecolor=color, markeredgewidth=1.2)
        ax.axvline(0.0, color="black", lw=1.0, ls="--")
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8, loc="left")
    # Faixa cinza = "efeito pequeno" (|d|<0,2), para o leitor ver que as médias mal mudam.
    axL.axvspan(-0.2, 0.2, color="gray", alpha=0.12, lw=0)
    axL.text(0.0, 0.015, "small effect", transform=axL.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=6.5, color="#666666")
    axL.margins(y=0.06)
    axL.set_yticks(y, [band_label(f) for f in feats], fontsize=7)
    # Legenda de significância (cheio = significativo apos correcao FDR).
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="#555", markerfacecolor="#555",
                      ls="", label=r"significant ($q<0.05$)"),
               Line2D([0], [0], marker="o", color="#555", markerfacecolor="white",
                      ls="", label="not significant")]
    axR.legend(handles=handles, loc="lower right", fontsize=6.5, framealpha=0.9)
    fig.suptitle(f"{DETECTOR_LABEL[detector_tag]} — cues linked to errors (H3)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, f"confirmatory_effects_{detector_tag}", figures_dir)
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
        ax.set_title(f"{DETECTOR_LABEL[tag]}\n{band_label(feat)}", fontsize=8, loc="left")
        ax.set_xlabel(band_label(feat))
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
    for tag in [t for t in ("zs", "ad") if f"p_spoof_{t}" in master.columns]:
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


def _band_assoc_strength(spearman: pd.DataFrame, tag: str, n_bands: int) -> np.ndarray:
    """max|ρ| por banda (sobre μ/σ e classes) para um detector, ordenado por banda."""
    sp = spearman[spearman.detector == tag].assign(abs_rho=lambda d: d["rho"].abs())
    sp = sp.assign(band=[band_index(f)[0] for f in sp["feature"]])
    m = sp.groupby("band")["abs_rho"].max()
    return np.array([float(m.get(b, np.nan)) for b in range(n_bands)])


def plot_eer_threshold(master: pd.DataFrame, figures_dir: str | Path):
    """Taxas de erro em função do limiar de decisão (D_zs em cima, D_ad embaixo).

    Didática do EER: para cada detector, duas curvas cruzam no ponto de operação de
    EER --- a taxa de rejeitar bonafide (FPR, cai com o limiar) e a de deixar passar
    spoof (FNR, sobe com o limiar). O cruzamento (FPR = FNR) é o EER. Complementa a
    curva DET (mesma informação, com o limiar explícito no eixo x)."""
    y = master["ground_truth"].to_numpy()
    # Escala logit no eixo x: os scores saturam perto de 1, então varremos o limiar em
    # espaçamento logit para o cruzamento (EER) ficar visível em vez de colado em 1.0.
    xticks = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    thr = 1.0 / (1.0 + np.exp(-np.linspace(-8.0, 8.0, 801)))   # ~ (3e-4 .. 1-3e-4)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_WIDTH_SINGLE, 4.2))
    for ax, tag in zip(axes, ("zs", "ad")):
        s = master[f"p_spoof_{tag}"].to_numpy()
        bona, spoof = s[y == 0], s[y == 1]
        fpr = np.array([(bona >= t).mean() for t in thr])   # rejeitar bonafide
        fnr = np.array([(spoof < t).mean() for t in thr])   # deixar passar spoof
        eer, eer_thr = compute_eer(s, y)
        eer_thr = float(np.clip(eer_thr, thr[0], thr[-1]))
        ax.plot(thr, fpr, color=CB_PALETTE[1], lw=1.6,
                label="reject bonafide (FPR)")
        ax.plot(thr, fnr, color=CB_PALETTE[0], lw=1.6,
                label="miss spoof (FNR)")
        ax.axvline(eer_thr, color="grey", ls="--", lw=0.7)
        ax.plot([eer_thr], [eer], "o", color="black", ms=5, zorder=5)
        ax.annotate(f"EER {eer * 100:.1f}%", xy=(eer_thr, eer),
                    xytext=(-8, 8), textcoords="offset points", fontsize=8,
                    fontweight="bold", ha="right")
        ax.set_xscale("logit")
        ax.set_xlim(thr[0], thr[-1])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Error rate")
        ax.set_title(DETECTOR_LABEL[tag], fontsize=8, loc="left")
    axes[0].legend(loc="center left", fontsize=7, ncol=1)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{t:g}" for t in xticks])
    axes[-1].set_xlabel(r"Decision threshold on $P(\mathrm{spoof})$ (logit scale)")
    fig.tight_layout()
    save_fig(fig, "eer_threshold_zs_vs_ad", figures_dir)
    return fig


def plot_spine_convergence(edges, occ: pd.DataFrame, spearman: pd.DataFrame,
                           agreement: pd.DataFrame, figures_dir: str | Path):
    """Convergência das duas espinhas na MESMA grade de bandas (D_zs em cima, D_ad
    embaixo): barras = |queda causal| da oclusão; linha = |ρ| associativo (eixo à
    direita). O título traz o ρ de concordância (Spearman) por detector."""
    centers = (edges[:-1] + edges[1:]) / 2
    n_bands = len(centers)
    tags = [t for t in ("zs", "ad") if not occ[occ.detector == t].empty]
    fig, axes = plt.subplots(len(tags), 1, sharex=True, squeeze=False,
                             figsize=(COL_WIDTH_SINGLE, 2.1 * len(tags)))
    axes = axes[:, 0]
    for ax, tag in zip(axes, tags):
        sub = occ[occ.detector == tag].sort_values("band_hz_low")
        causal = np.abs(sub["mean_p_spoof_drop"].to_numpy())
        assoc = _band_assoc_strength(spearman, tag, n_bands)
        ax.bar(centers, causal, width=np.diff(edges) * 0.9, align="center",
               color=DETECTOR_COLOR[tag], alpha=0.35, label="causal $|\\Delta P|$")
        ax2 = ax.twinx()
        ax2.plot(centers, assoc, color="#333333", lw=1.3, marker="o", ms=3,
                 label=r"assoc.\ $|\rho|$")
        ax2.set_ylim(bottom=0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"$|\Delta P(\mathrm{spoof})|$")
        ax2.set_ylabel(r"$|\rho|$")
        row = agreement[agreement.detector == tag]
        rho_txt = f"$\\rho$={row['rho_causal_vs_assoc'].iloc[0]:.2f}" if len(row) else ""
        ax.set_title(f"{DETECTOR_LABEL[tag]}  (agreement {rho_txt})", fontsize=8, loc="left")
    axes[-1].set_xlabel("Frequency band center (Hz)")
    fig.tight_layout()
    save_fig(fig, "spine_convergence_zs_vs_ad", figures_dir)
    return fig


def plot_convergence_intervention(pairs: dict, conv: pd.DataFrame,
                                  figures_dir: str | Path):
    """Convergência H1→H2 por intervenção causal orientada (teste pareado por clipe).

    Um painel por detector. Cada painel mostra a distribuição, por clipe, do efeito
    causal ORIENTADO ao ocluir as bandas MAIS associadas (escolhidas pelo H1) versus as
    MENOS associadas. Orientado = a queda de P(spoof) é multiplicada pelo sinal da
    associação da banda, então remover uma pista de spoof (P(spoof) cai) e remover uma
    pista de bonafide (P(spoof) sobe) contam ambas como efeito POSITIVO quando são
    coerentes com o H1. As bandas são escolhidas numa partição de clipes independente da
    usada aqui, para evitar circularidade. A diferença mediana (mais − menos), o IC 95% e
    o p de Wilcoxon aparecem no título: uma diferença positiva indica que as bandas
    apontadas pelo H1 têm efeito causal maior e no sentido previsto (as duas análises
    convergem).

    Args:
        pairs: {tag: (oriented_top, oriented_bottom)} com o efeito orientado por clipe.
        conv: tabela de `paired_intervention_test` por detector.
    """
    tags = [t for t in ("zs", "ad") if t in pairs]
    fig, axes = plt.subplots(1, len(tags), figsize=(COL_WIDTH_DOUBLE, 3.4),
                             squeeze=False)
    axes = axes[0]
    for ax, tag in zip(axes, tags):
        d_top, d_bottom = pairs[tag]
        data = [np.asarray(d_top, float), np.asarray(d_bottom, float)]
        parts = ax.violinplot(data, positions=[0, 1], widths=0.8,
                              showmeans=False, showextrema=False, showmedians=True)
        for body, col in zip(parts["bodies"], (DETECTOR_COLOR[tag], "#999999")):
            body.set_facecolor(col)
            body.set_alpha(0.45)
            body.set_edgecolor("#333333")
        parts["cmedians"].set_color("#333333")
        ax.axhline(0.0, color="black", lw=0.8, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["mais\nassociadas (H1)", "menos\nassociadas (H1)"],
                           fontsize=8)
        ax.set_ylabel("efeito causal orientado por clipe\n(positivo = coerente com H1)",
                      fontsize=8)
        row = conv[conv.detector == tag]
        if len(row):
            r = row.iloc[0]
            title = (f"{DETECTOR_LABEL[tag]}\n"
                     f"$\\Delta$mediana = {r.median_diff:.3f} "
                     f"[{r.ci_low:.3f}, {r.ci_high:.3f}]")
            ax.set_title(title, fontsize=8, loc="left")
            ax.text(0.98, 0.03, f"$d_z$={r.cohen_dz:.2f} · p={r.wilcoxon_p:.1e}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color="#555555")
    fig.tight_layout()
    save_fig(fig, "convergence_intervention_zs_vs_ad", figures_dir)
    return fig
