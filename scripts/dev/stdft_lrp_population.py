"""Espectrograma de atenção POPULACIONAL (STDFT-LRP) médio sobre muitos clipes, por encoder.

Diferente do ``stdft_lrp_compare`` (um clipe por célula), aqui acumulamos a relevância STDFT-LRP
no MESMO grid tempo-frequência sobre TODOS os clipes do split e tiramos a média. Isso é possível
porque os clipes têm comprimento fixo (num_samples), então a grade do STFT é idêntica entre eles.

Duas figuras, grade dos 3 codificadores conservativos (wav2vec2, hubert, wavlm):
  * magnitude:  média de |relevância|(t,f)  -> ONDE, no plano tempo-frequência, o modelo se apoia
                (direção-agnóstico), 1 linha x 3 colunas.
  * por direção: média de max(R,0) ("rumo a spoof") e de max(-R,0) ("rumo a bonafide"),
                2 linhas (direção) x 3 colunas (encoder).

RESSALVA de leitura: o eixo do TEMPO não é semanticamente alinhado entre enunciados (cada áudio
fala em instantes diferentes). Leia a estrutura em FREQUÊNCIA; a estrutura no tempo só reflete
efeitos sistemáticos (ex.: silêncio no começo/fim). A escala de cor é robusta (99º percentil):
por painel na magnitude, por coluna (modelo) na versão por direção; compare a forma, não a
intensidade absoluta entre modelos (a magnitude do wavlm é inflada pela completude).

Gera cada figura em duas faixas: cheia (--stdft-fmax, default 4 kHz) e um zoom nos graves
(--zoom-fmax, default 1 kHz). Os arrays acumulados ficam em cache (npz), então mudar o recorte
de frequência depois é instantâneo com --from-cache (sem reprocessar os clipes).

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/stdft_lrp_population.py --results-root /workspace/results
    python /workspace/scripts/dev/stdft_lrp_population.py --limit 200          # teste rápido
    python /workspace/scripts/dev/stdft_lrp_population.py --from-cache         # só replota
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brspeech_xai import artifacts as A
from brspeech_xai import dft_lrp
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger, progress
from dft_lrp_ad import relevance_for_clip
from stdft_lrp_compare import (_build_lrp_model, _discover_runs, _order_slugs, _split_of)


def _accumulate_encoder(emb, model, audios, srs, indices, device, win, hop, eps, log, tag):
    """Média das relevâncias STDFT-LRP de um encoder sobre os clipes dados.

    Devolve (times, freqs, mean_mag, mean_spoof, mean_bona) no grid fixo tempo-frequência.
    mean_mag = média de |R|; mean_spoof = média de max(R,0); mean_bona = média de max(-R,0).
    """
    acc_mag = acc_spoof = acc_bona = None
    times = freqs = None
    n = 0
    for i in progress(indices, desc=f"STDFT pop {tag}", unit="clip"):
        wav16k = emb._to_16k_mono(audios[i], srs[i])
        x_time, r_time, _logit = relevance_for_clip(model, emb._processor, wav16k, device)
        times, freqs, r_tf, _ = dft_lrp.stdft_lrp(x_time, r_time, 16000,
                                                  win_size=win, hop=hop, eps=eps)
        r_tf = np.asarray(r_tf, dtype=np.float64)          # (tempo, freq)
        if acc_mag is None:
            acc_mag = np.zeros_like(r_tf)
            acc_spoof = np.zeros_like(r_tf)
            acc_bona = np.zeros_like(r_tf)
        acc_mag += np.abs(r_tf)
        acc_spoof += np.clip(r_tf, 0, None)
        acc_bona += np.clip(-r_tf, 0, None)
        n += 1
    log.info(f"[{tag}] acumulados {n} clipes")
    return times, freqs, acc_mag / n, acc_spoof / n, acc_bona / n


def _plot_magnitude(profs, fmax, out_png, out_pdf, range_txt=""):
    """Grade 1 x N: média de |relevância|(t,f) por encoder (viridis, escala robusta por painel)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    n = len(profs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), squeeze=False)
    for c, p in enumerate(profs):
        ax = axes[0][c]
        mask = p["freqs"] <= fmax
        f = p["freqs"][mask]
        M = p["mag"][:, mask].T                            # (freq, tempo)
        v = float(np.percentile(M, 99)) or 1.0
        ax.pcolormesh(p["times"], f, M, cmap="viridis", vmin=0, vmax=v, shading="auto")
        ax.annotate(f"{p['short']}\n(conservative)", xy=(0.5, 1.06), xycoords="axes fraction",
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Frequency (Hz)" if c == 0 else "")
        if c != 0:
            ax.tick_params(labelleft=False)
    fig.suptitle(f"Population STDFT-LRP attention (mean |relevance| over the test split){range_txt}",
                 y=1.02, fontsize=11)
    fig.text(0.5, -0.02,
             "Brighter = more relevance in time-frequency, averaged over all clips. Read the "
             "FREQUENCY structure; the time axis is not aligned across utterances. Per-panel "
             "robust scale (99th pct): compare shape, not absolute intensity across models.",
             ha="center", va="top", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_by_direction(profs, fmax, out_png, out_pdf, range_txt=""):
    """Grade 2 x N: média de max(R,0) (rumo a spoof) e max(-R,0) (rumo a bonafide) por encoder.

    Escala de cor compartilhada por coluna (modelo), sobre as duas direções, para dar para comparar
    a assimetria spoof vs bonafide dentro de cada modelo (viridis, escala robusta)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    n = len(profs)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 5.2), squeeze=False)
    row_label = ("toward spoof", "toward bonafide")
    for c, p in enumerate(profs):
        mask = p["freqs"] <= fmax
        f = p["freqs"][mask]
        S = p["spoof"][:, mask].T
        B = p["bona"][:, mask].T
        v = float(np.percentile(np.concatenate([S.ravel(), B.ravel()]), 99)) or 1.0
        for r, M in enumerate((S, B)):
            ax = axes[r][c]
            ax.pcolormesh(p["times"], f, M, cmap="viridis", vmin=0, vmax=v, shading="auto")
            if r == 0:
                ax.annotate(f"{p['short']}\n(conservative)", xy=(0.5, 1.08),
                            xycoords="axes fraction", ha="center", va="bottom",
                            fontweight="bold", fontsize=9)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("t [s]")
            if c == 0:
                ax.set_ylabel(f"{row_label[r]}\nFrequency (Hz)", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
    fig.suptitle(f"Population STDFT-LRP by direction (mean over the test split){range_txt}",
                 y=0.995, fontsize=11)
    fig.text(0.5, 0.005,
             "Top: mean relevance pushing toward spoof; bottom: toward bonafide. Averaged over all "
             "clips. Read the FREQUENCY structure; the time axis is not aligned across utterances. "
             "Color scale shared per column (per model): compare shape, not absolute intensity.",
             ha="center", va="bottom", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def _freq_marginal(arr: np.ndarray) -> np.ndarray:
    """Média no tempo -> perfil por frequência (arr é (tempo, freq))."""
    return np.asarray(arr, dtype=np.float64).mean(axis=0)


def _plot_freq_magnitude(profs, fmax, out_png, out_pdf, range_txt=""):
    """Marginal em frequência da magnitude: share de |relevância| por frequência, 1 linha/encoder.

    Colapsa o eixo do tempo (média) dos mapas populacionais e normaliza cada encoder para somar
    100% no espectro (share), como o mapa de saliência. Direção-agnóstico: mostra ONDE cada modelo
    se apoia, comparável entre modelos apesar da magnitude bruta do wavlm ser inflada."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, p in enumerate(profs):
        c = plt.cm.tab10.colors[i % 10]
        marg = _freq_marginal(p["mag"])
        share = 100.0 * marg / (marg.sum() + 1e-12)        # share sobre todo o espectro
        peak = float(p["freqs"][int(np.argmax(marg))])
        ax.plot(p["freqs"], share, lw=1.3, color=c,
                label=f"{p['short']} (peak ~{peak:.0f} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Share of mean |relevance| (% per bin)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, fmax)
    ax.set_title(f"Population STDFT-LRP frequency marginal — magnitude{range_txt}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_freq_direction(profs, fmax, out_png, out_pdf, range_txt=""):
    """Marginal em frequência por direção: rumo a spoof e rumo a bonafide, share por frequência.

    Dois painéis (spoof/bonafide), 1 linha por encoder. Normaliza cada encoder pelo total das duas
    direções, então a área sob as duas curvas soma 100% (mostra onde e para que lado o modelo puxa
    em cada frequência)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    titles = ("toward spoof", "toward bonafide")
    for i, p in enumerate(profs):
        c = plt.cm.tab10.colors[i % 10]
        sp = _freq_marginal(p["spoof"])
        bo = _freq_marginal(p["bona"])
        total = sp.sum() + bo.sum() + 1e-12                # normaliza pelo total das duas direções
        for ax, marg in zip(axes, (sp, bo)):
            ax.plot(p["freqs"], 100.0 * marg / total, lw=1.3, color=c, label=p["short"])
    for ax, t in zip(axes, titles):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, fmax)
        ax.set_ylim(bottom=0)
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of mean relevance (% per bin)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Population STDFT-LRP frequency marginal — by direction{range_txt}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_freq_strip_magnitude(profs, fmax, out_png, out_pdf, range_txt=""):
    """'Espectrograma sem tempo': relevância média por frequência como tira vertical por modelo.

    Colapsa o tempo (média) e desenha uma coluna por encoder: eixo Y = frequência, cor = relevância
    (viridis). Cada coluna é normalizada pelo próprio máximo (0..1) para ler QUAIS frequências
    acendem em cada modelo, mantendo a cara de espectrograma, mas sem o eixo do tempo (que não é
    alinhado entre enunciados)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    mask0 = profs[0]["freqs"] <= fmax
    f = profs[0]["freqs"][mask0]
    cols = []
    for p in profs:
        m = p["freqs"] <= fmax
        marg = _freq_marginal(p["mag"])[m]
        cols.append(marg / (marg.max() + 1e-12))
    M = np.column_stack(cols)                              # (freq, modelos)
    n = len(profs)
    fig, ax = plt.subplots(figsize=(1.5 * n + 2.2, 4.6))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=(0, n, float(f[0]), float(f[-1])))
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels([p["short"] for p in profs], fontsize=9)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Population STDFT-LRP — frequency importance (time collapsed){range_txt}",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("relative relevance (per model, 0–1)", fontsize=8)
    fig.text(0.5, -0.02,
             "Time averaged out: each column is the mean |relevance| per frequency for one model, "
             "normalized to its own max. Brighter = frequency the model relies on more.",
             ha="center", va="top", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_freq_strip_direction(profs, fmax, out_png, out_pdf, range_txt=""):
    """Mesma tira vertical (frequência no Y, viridis), separada por direção da relevância.

    Dois painéis (rumo a spoof / rumo a bonafide). Cada modelo é normalizado pelo próprio máximo
    sobre as duas direções, então dá para ver a assimetria spoof vs bonafide dentro do modelo."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    f = profs[0]["freqs"][profs[0]["freqs"] <= fmax]
    n = len(profs)
    sp_cols, bo_cols = [], []
    for p in profs:
        m = p["freqs"] <= fmax
        sp = _freq_marginal(p["spoof"])[m]
        bo = _freq_marginal(p["bona"])[m]
        denom = max(sp.max(), bo.max()) + 1e-12            # por modelo, sobre as duas direções
        sp_cols.append(sp / denom)
        bo_cols.append(bo / denom)
    mats = (np.column_stack(sp_cols), np.column_stack(bo_cols))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (1.5 * n + 1.0), 4.6), sharey=True)
    im = None
    for ax, M, t in zip(axes, mats, ("toward spoof", "toward bonafide")):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=(0, n, float(f[0]), float(f[-1])))
        ax.set_xticks([j + 0.5 for j in range(n)])
        ax.set_xticklabels([p["short"] for p in profs], fontsize=8)
        ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("Frequency (Hz)")
    cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.046)
    cb.set_label("relative relevance (per model, 0–1)", fontsize=8)
    fig.suptitle("Population STDFT-LRP — frequency importance by direction "
                 f"(time collapsed){range_txt}", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bands_from_marginal(marg: np.ndarray, freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Média da marginal contínua dentro de cada faixa [edges[i], edges[i+1]) -> vetor (n_bandas,)."""
    vals = np.zeros(len(edges) - 1, dtype=np.float64)
    for i in range(len(edges) - 1):
        hi_incl = i == len(edges) - 2
        m = (freqs >= edges[i]) & ((freqs <= edges[i + 1]) if hi_incl else (freqs < edges[i + 1]))
        vals[i] = float(marg[m].mean()) if m.any() else 0.0
    return vals


def _plot_band_strip_magnitude(profs, edges, out_png, out_pdf):
    """Tira vertical nas 24 faixas mel (frequência no Y, viridis), magnitude, tempo colapsado.

    Igual à tira contínua, mas a marginal é agregada na MESMA grade de bandas de H1/H2/DFT-LRP,
    então dá para casar a leitura banda a banda. Cada coluna (modelo) normalizada pelo próprio máximo."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    n = len(profs)
    cols = []
    for p in profs:
        b = _bands_from_marginal(_freq_marginal(p["mag"]), p["freqs"], edges)
        cols.append(b / (b.max() + 1e-12))
    M = np.column_stack(cols)                              # (n_bandas, modelos)
    fig, ax = plt.subplots(figsize=(1.5 * n + 2.2, 5.0))
    im = ax.pcolormesh(np.arange(n + 1), edges, M, cmap="viridis", vmin=0, vmax=1, shading="flat")
    ax.set_xticks([j + 0.5 for j in range(n)])
    ax.set_xticklabels([p["short"] for p in profs], fontsize=9)
    ax.set_ylabel(f"Frequency (Hz) — {len(edges) - 1} mel bands")
    ax.set_title("Population STDFT-LRP — band importance (time collapsed)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("relative relevance (per model, 0–1)", fontsize=8)
    fig.text(0.5, -0.02,
             "Time averaged out and binned into the shared 24 mel bands. Brighter = band the model "
             "relies on more (each column normalized to its own max).",
             ha="center", va="top", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_band_strip_direction(profs, edges, out_png, out_pdf):
    """Mesma tira nas 24 faixas mel, separada por direção (rumo a spoof / rumo a bonafide)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    n = len(profs)
    sp_cols, bo_cols = [], []
    for p in profs:
        sp = _bands_from_marginal(_freq_marginal(p["spoof"]), p["freqs"], edges)
        bo = _bands_from_marginal(_freq_marginal(p["bona"]), p["freqs"], edges)
        denom = max(sp.max(), bo.max()) + 1e-12
        sp_cols.append(sp / denom)
        bo_cols.append(bo / denom)
    mats = (np.column_stack(sp_cols), np.column_stack(bo_cols))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (1.5 * n + 1.0), 5.0), sharey=True)
    im = None
    for ax, M, t in zip(axes, mats, ("toward spoof", "toward bonafide")):
        im = ax.pcolormesh(np.arange(n + 1), edges, M, cmap="viridis", vmin=0, vmax=1, shading="flat")
        ax.set_xticks([j + 0.5 for j in range(n)])
        ax.set_xticklabels([p["short"] for p in profs], fontsize=8)
        ax.set_title(t, fontsize=10)
    axes[0].set_ylabel(f"Frequency (Hz) — {len(edges) - 1} mel bands")
    cb = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.046)
    cb.set_label("relative relevance (per model, 0–1)", fontsize=8)
    fig.suptitle("Population STDFT-LRP — band importance by direction (time collapsed)", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_cache(profs, path: Path) -> None:
    """Guarda os arrays acumulados (npz) para replotar recortes de frequência sem reprocessar."""
    d = {"shorts": np.array([p["short"] for p in profs])}
    for i, p in enumerate(profs):
        for k in ("times", "freqs", "mag", "spoof", "bona"):
            d[f"{k}_{i}"] = p[k]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def _load_cache(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=False)
    shorts = [str(s) for s in z["shorts"]]
    return [{"short": shorts[i], "times": z[f"times_{i}"], "freqs": z[f"freqs_{i}"],
             "mag": z[f"mag_{i}"], "spoof": z[f"spoof_{i}"], "bona": z[f"bona_{i}"]}
            for i in range(len(shorts))]


def _plot_all(profs, out_dir: Path, full_fmax: float, zoom_fmax: float, log,
              band_edges=None) -> None:
    """Gera as figuras (magnitude e por direção) em duas faixas: cheia e zoom nos graves.

    Se ``band_edges`` for dado, gera também as tiras agregadas nas faixas mel (24 bandas)."""
    khz = f"{zoom_fmax / 1000:g}"
    for fmax, suffix, rng in ((full_fmax, "", ""),
                              (zoom_fmax, f"_{khz}khz", f" — 0–{khz} kHz")):
        _plot_magnitude(profs, fmax, out_dir / f"stdft_lrp_population_magnitude{suffix}.png",
                        out_dir / f"stdft_lrp_population_magnitude{suffix}.pdf", range_txt=rng)
        _plot_by_direction(profs, fmax, out_dir / f"stdft_lrp_population_direction{suffix}.png",
                           out_dir / f"stdft_lrp_population_direction{suffix}.pdf", range_txt=rng)
        # Marginais em frequência (colapsam o tempo), como o mapa de saliência.
        _plot_freq_magnitude(profs, fmax, out_dir / f"stdft_lrp_population_freq_magnitude{suffix}.png",
                             out_dir / f"stdft_lrp_population_freq_magnitude{suffix}.pdf", range_txt=rng)
        _plot_freq_direction(profs, fmax, out_dir / f"stdft_lrp_population_freq_direction{suffix}.png",
                             out_dir / f"stdft_lrp_population_freq_direction{suffix}.pdf", range_txt=rng)
        # 'Espectrograma sem tempo': tira vertical por modelo (frequência no Y, viridis).
        _plot_freq_strip_magnitude(profs, fmax, out_dir / f"stdft_lrp_population_strip_magnitude{suffix}.png",
                                   out_dir / f"stdft_lrp_population_strip_magnitude{suffix}.pdf", range_txt=rng)
        _plot_freq_strip_direction(profs, fmax, out_dir / f"stdft_lrp_population_strip_direction{suffix}.png",
                                   out_dir / f"stdft_lrp_population_strip_direction{suffix}.pdf", range_txt=rng)
    # Tiras agregadas nas 24 faixas mel (faixa completa), casando com H1/H2/DFT-LRP.
    if band_edges is not None:
        _plot_band_strip_magnitude(profs, band_edges,
                                   out_dir / "stdft_lrp_population_bandstrip_magnitude.png",
                                   out_dir / "stdft_lrp_population_bandstrip_magnitude.pdf")
        _plot_band_strip_direction(profs, band_edges,
                                   out_dir / "stdft_lrp_population_bandstrip_direction.png",
                                   out_dir / "stdft_lrp_population_bandstrip_direction.pdf")
    log.info(f"figuras salvas em: {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stdft_lrp_population")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--limit", type=int, default=None,
                    help="usa só os primeiros N clipes do split (default: todos)")
    ap.add_argument("--stdft-win", type=int, default=512)
    ap.add_argument("--stdft-hop", type=int, default=128)
    ap.add_argument("--stdft-fmax", type=float, default=4000.0, help="fmax da versão cheia")
    ap.add_argument("--zoom-fmax", type=float, default=1000.0, help="fmax da versão ampliada")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--from-cache", action="store_true",
                    help="não reprocessa: carrega os arrays do npz e só replota (recortes rápidos)")
    ap.add_argument("--exclude", nargs="*", default=["mms-300m"],
                    help="encoders (short) a não plotar. Default: mms-300m (removido por ora). "
                         "O cache mantém todos; isto só filtra na hora de plotar.")
    ap.add_argument("--n-bands", type=int, default=24, help="nº de faixas mel para a tira banded")
    ap.add_argument("--band-fmin", type=float, default=20.0)
    ap.add_argument("--band-fmax", type=float, default=7900.0)
    args = ap.parse_args(argv)
    excl = set(args.exclude or [])
    from brspeech_xai.bands import mel_band_edges
    band_edges = mel_band_edges(args.n_bands, args.band_fmin, args.band_fmax)

    log = get_logger()
    results_root = Path(args.results_root)
    out_dir = results_root / "_aggregate"
    cache_path = out_dir / "stdft_lrp_population.npz"

    if args.from_cache:
        if not cache_path.exists():
            raise SystemExit(f"cache não encontrado: {cache_path} (rode uma vez sem --from-cache)")
        log.info(f"replotando do cache: {cache_path}")
        profs = _load_cache(cache_path)
        if excl:
            profs = [p for p in profs if p["short"] not in excl]
            log.info(f"excluídos da plotagem: {sorted(excl)}")
        _plot_all(profs, out_dir, args.stdft_fmax, args.zoom_fmax, log, band_edges=band_edges)
        return 0

    runs = _discover_runs(results_root)
    if not runs:
        raise SystemExit(f"nenhuma run hf_ssl com d_ad.joblib em {results_root}")
    slugs = _order_slugs(runs)
    short = {s: s.replace("hf_ssl-", "") for s in slugs}
    if excl:
        slugs = [s for s in slugs if short[s] not in excl]
        log.info(f"excluídos: {sorted(excl)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"encoders: {[short[s] for s in slugs]} | device={device}")

    profs = []
    for slug in slugs:
        run_dir = runs[slug]
        cfg = load_config(run_dir / "config.resolved.yaml")
        emb, model, conservative = _build_lrp_model(cfg, run_dir, device)
        if not conservative:
            log.info(f"[{short[slug]}] pulando (não-conservativo)")
            del emb, model
            continue
        split = _split_of(cfg)
        audios = list(np.load(run_dir / f"audios_{split}.npy", allow_pickle=True))
        srs = [int(s) for s in np.load(run_dir / f"srs_{split}.npy")]
        indices = range(len(audios) if args.limit is None else min(args.limit, len(audios)))
        log.info(f"[{short[slug]}] {cfg.model.checkpoint} | split={split} | "
                 f"{len(indices)} clipes")
        times, freqs, mag, spoof, bona = _accumulate_encoder(
            emb, model, audios, srs, indices, device, args.stdft_win, args.stdft_hop,
            args.eps, log, short[slug])
        profs.append({"short": short[slug], "times": times, "freqs": freqs,
                      "mag": mag, "spoof": spoof, "bona": bona})
        del emb, model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not profs:
        raise SystemExit("nenhum encoder conservativo encontrado")

    _save_cache(profs, cache_path)
    log.info(f"cache salvo: {cache_path}")
    _plot_all(profs, out_dir, args.stdft_fmax, args.zoom_fmax, log, band_edges=band_edges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
