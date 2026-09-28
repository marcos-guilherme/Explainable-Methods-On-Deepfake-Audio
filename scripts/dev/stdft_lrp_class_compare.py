"""STDFT-LRP separado por direção (spoof vs bonafide), em viridis, comparando encoders.

Complementa o ``stdft_lrp_compare`` (mapa divergente vermelho/azul num único painel). Aqui, para
o MESMO clipe, isolamos a evidência de cada lado em dois mapas de MAGNITUDE (viridis, fundo escuro
e energia clara, como um espectrograma):

  * "rumo a spoof"    = parte positiva da relevância (max(R, 0));
  * "rumo a bonafide" = parte negativa da relevância (max(-R, 0)).

Uma figura por clipe, grade 2 linhas (direção) x 3 colunas (encoder). A escala de cor é
compartilhada POR COLUNA (por modelo, 99º percentil sobre as duas direções), então dentro de um
modelo dá para comparar a assimetria spoof vs bonafide; entre modelos, compare a forma, não a
intensidade (os três encoders usam o método conservativo AttnLRP).

Uso (dentro do container):
    python /workspace/scripts/dev/stdft_lrp_class_compare.py \
        --results-root /workspace/results --clip-indices 187 1016
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brspeech_xai import artifacts as A
from brspeech_xai import dft_lrp
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger
from dft_lrp_ad import relevance_for_clip
from stdft_lrp_compare import _build_lrp_model, _discover_runs, _order_slugs, _split_of


def _plot_class_grid(clip_idx, true_cls, encoders, cols, fmax, out_png, out_pdf):
    """Grade 2 (direção) x N (encoder) para um clipe, em viridis.

    encoders: lista de (short_name, conservative_bool).
    cols: lista (por encoder) de dict com times, freqs, r_tf, logit, pred.
    """
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    ncols = len(encoders)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 4.6), squeeze=False)
    row_labels = ("toward spoof", "toward bonafide")
    for c, ((short, conservative), data) in enumerate(zip(encoders, cols)):
        times, freqs, r_tf, logit, pred = (data[k] for k in
                                           ("times", "freqs", "r_tf", "logit", "pred"))
        mask = freqs <= fmax
        f = freqs[mask]
        R = r_tf[:, mask]                                  # (tempo, freq)
        pos = np.clip(R, 0, None).T                        # rumo a spoof (magnitude)
        neg = np.clip(-R, 0, None).T                       # rumo a bonafide (magnitude)
        vmax = float(np.percentile(np.abs(R), 99)) or 1.0  # escala compartilhada por coluna
        for row, mag in ((0, pos), (1, neg)):
            ax = axes[row][c]
            ax.pcolormesh(times, f, mag, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
            if row == 0:
                backend = "conservative" if conservative else "gxi ref."
                ax.set_title(f"{short}\n({backend}) | logit={logit:+.1f}→{pred}",
                             fontsize=8, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{row_labels[row]}\nFrequency (Hz)", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel("t [s]")
            else:
                ax.tick_params(labelbottom=False)
    fig.suptitle(f"STDFT-LRP magnitude by direction — {true_cls} clip #{clip_idx} "
                 "(same audio across encoders)", y=0.995, fontsize=11)
    fig.text(0.5, 0.005,
             "Rows isolate each side: brighter = more relevance in that direction. Color scale "
             "shared per column (per model); compare shape across models, not intensity. "
             "All three encoders use the conservative method (AttnLRP).",
             ha="center", va="bottom", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stdft_lrp_class_compare")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--clip-indices", type=int, nargs="+", default=[187, 1016],
                    help="índices de clipe (posição em audios_<split>.npy, a mesma entre runs); "
                         "uma figura por clipe. Default: 187 (spoof) e 1016 (bonafide).")
    ap.add_argument("--stdft-win", type=int, default=512)
    ap.add_argument("--stdft-hop", type=int, default=128)
    ap.add_argument("--stdft-fmax", type=float, default=4000.0)
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args(argv)

    log = get_logger()
    results_root = Path(args.results_root)
    runs = _discover_runs(results_root)
    if not runs:
        raise SystemExit(f"nenhuma run hf_ssl com d_ad.joblib em {results_root}")
    slugs = _order_slugs(runs)
    short = {s: s.replace("hf_ssl-", "") for s in slugs}
    log.info(f"encoders: {[short[s] for s in slugs]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_cfg = load_config(runs[slugs[0]] / "config.resolved.yaml")
    split = _split_of(ref_cfg)
    ref_master = A.load_table(runs[slugs[0]] / "master_table.parquet")
    audios = list(np.load(runs[slugs[0]] / f"audios_{split}.npy", allow_pickle=True))
    srs = [int(s) for s in np.load(runs[slugs[0]] / f"srs_{split}.npy")]
    clip_cls = {i: ("spoof" if int(ref_master["ground_truth"].iloc[i]) == 1 else "bonafide")
                for i in args.clip_indices}

    # cells[(clip_idx, col)] -> dict(times, freqs, r_tf, logit, pred). Um modelo por vez (memória).
    cells: dict[tuple[int, int], dict] = {}
    encoders_meta = []
    for c, slug in enumerate(slugs):
        run_dir = runs[slug]
        cfg = load_config(run_dir / "config.resolved.yaml")
        emb, model, conservative = _build_lrp_model(cfg, run_dir, device)
        encoders_meta.append((short[slug], conservative))
        log.info(f"[{short[slug]}] {cfg.model.checkpoint} | "
                 f"{'conservativo' if conservative else 'gxi (referência)'}")
        for i in args.clip_indices:
            wav16k = emb._to_16k_mono(audios[i], srs[i])
            x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
            times, freqs, r_tf, _ = dft_lrp.stdft_lrp(x_time, r_time, 16000,
                                                      win_size=args.stdft_win, hop=args.stdft_hop,
                                                      eps=args.eps)
            cells[(i, c)] = {"times": times, "freqs": freqs, "r_tf": r_tf, "logit": logit,
                             "pred": "spoof" if logit > 0 else "bonafide"}
        del emb, model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_dir = results_root / "_aggregate"
    for i in args.clip_indices:
        cols = [cells[(i, c)] for c in range(len(slugs))]
        stem = f"stdft_lrp_class_compare_{clip_cls[i]}_clip{i:04d}"
        out_png = out_dir / f"{stem}.png"
        _plot_class_grid(i, clip_cls[i], encoders_meta, cols, args.stdft_fmax,
                         out_png, out_png.with_suffix(".pdf"))
        log.info(f"figura salva: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
