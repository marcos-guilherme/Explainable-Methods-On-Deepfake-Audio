"""Compara o STDFT-LRP entre encoders HF SSL usando os MESMOS clipes.

Monta uma grade clipes (linhas) x encoders (colunas): cada célula é o mapa tempo-frequência da
relevância que o D_ad atribui à decisão de spoof naquele áudio. Como as runs HF SSL default
compartilham a mesma ordem de split (mesmo seed), o índice de clipe aponta para o mesmo áudio em
todas; carregamos os áudios só de uma run de referência.

Backend por encoder: wav2vec2, hubert e wavlm usam o método conservativo (AttnLRP; o wavlm tem
regra CP-LRP dedicada para a atenção com viés de posição relativa e gating). Encoders sem regra
conservativa cairiam no Gradient x Input (gxi) como referência. A escala de cor é por célula
(robusta, 99º percentil), então compare a FORMA e o SINAL, não a intensidade absoluta entre
células.

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/stdft_lrp_compare.py \
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
from dft_lrp_ad import SSLDetectorAD, port_logistic_head, relevance_for_clip

# Ordem de leitura das colunas (referência primeiro); slugs 'hf_ssl-<nome>'.
_COL_ORDER = ("wav2vec2", "hubert", "wavlm")


def _discover_runs(results_root: Path) -> dict[str, Path]:
    """Última run por encoder HF SSL, com os artefatos necessários para o LRP."""
    runs: dict[str, Path] = {}
    for slug_dir in sorted(results_root.glob("hf_ssl-*")):
        cands = [d for d in slug_dir.glob("*")
                 if (d / "master_table.parquet").exists()
                 and (d / "config.resolved.yaml").exists()
                 and (d / "d_ad.joblib").exists()]
        if cands:
            runs[slug_dir.name] = max(cands, key=lambda d: d.name)  # timestamp no nome ordena
    return runs


def _order_slugs(runs: dict[str, Path]) -> list[str]:
    """Ordena os slugs por _COL_ORDER (o que casar primeiro), o resto no fim."""
    def rank(slug: str) -> int:
        for i, name in enumerate(_COL_ORDER):
            if name in slug:
                return i
        return len(_COL_ORDER)
    return sorted(runs, key=rank)


def _split_of(cfg) -> str:
    return "pool" if cfg.adapt.cross_fit else cfg.data.eval_split


def _is_conservative(checkpoint: str) -> bool:
    # mms/xls-r são arquitetura wav2vec2 (atenção padrão), cobertos pelo patch conservativo.
    c = checkpoint.lower()
    return any(k in c for k in ("wav2vec2", "hubert", "wavlm", "mms", "xls"))


def _build_lrp_model(cfg, run_dir: Path, device: str):
    """Encoder eager (patcheado se conservativo) + head logístico -> SSLDetectorAD, e o embedder
    (só para processor e reamostragem)."""
    from transformers import AutoModel
    import joblib

    from brspeech_xai.encoders.hf_ssl import HFSSLEmbedder

    emb = HFSSLEmbedder(checkpoint=cfg.model.checkpoint, layer=cfg.model.layer,
                        pooling=cfg.model.pooling, device=device,
                        num_samples=cfg.audio.num_samples, attn_implementation="eager")
    encoder = AutoModel.from_pretrained(cfg.model.checkpoint,
                                        attn_implementation="eager").to(device).eval()
    encoder.requires_grad_(False)
    conservative = _is_conservative(cfg.model.checkpoint)
    if conservative:
        from brspeech_xai.attnlrp import patch_ssl_encoder_for_attnlrp
        patch_ssl_encoder_for_attnlrp(encoder, attention="cp")
    head = joblib.load(run_dir / "d_ad.joblib")
    w, b = port_logistic_head(head)
    model = SSLDetectorAD(encoder, cfg.model.layer, w, b).to(device).eval()
    return emb, model, conservative


def _plot_grid(clips, encoders, cells, fmax, out_png, out_pdf):
    """Grade clipes (linhas) x encoders (colunas), cada célula um heatmap tempo-frequência.

    clips: lista de (idx, true_class).
    encoders: lista de (short_name, conservative_bool) na ordem das colunas.
    cells: dict (r, c) -> (times, freqs, r_tf, x_time, sr, logit, pred, enc_p_spoof).
    """
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    nrows, ncols = len(clips), len(encoders)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows),
                             squeeze=False)
    for r, (_idx, true_cls) in enumerate(clips):
        for c, (short, conservative) in enumerate(encoders):
            ax = axes[r][c]
            times, freqs, r_tf, _x, _sr, logit, pred, enc_p = cells[(r, c)]
            mask = freqs <= fmax
            f = freqs[mask]
            R = r_tf[:, mask].T                            # (freq, tempo)
            v = float(np.percentile(np.abs(R), 99)) or 1.0  # escala simétrica robusta por célula
            ax.pcolormesh(times, f, R, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
            hit = "OK" if (pred == true_cls) else "MISS"
            ax.set_title(f"logit={logit:+.1f} -> {pred} [{hit}]", fontsize=7, loc="left")
            if r == 0:
                backend = "conservative" if conservative else "gxi ref."
                ax.annotate(f"{short}\n({backend})", xy=(0.5, 1.22), xycoords="axes fraction",
                            ha="center", va="bottom", fontweight="bold", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{true_cls} clip\nFrequency (Hz)", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if r == nrows - 1:
                ax.set_xlabel("t [s]")
            else:
                ax.tick_params(labelbottom=False)
    fig.suptitle("STDFT-LRP relevance — same clips across encoders "
                 "(red = toward spoof, blue = toward bonafide)", y=0.995, fontsize=11)
    fig.text(0.5, 0.005,
             "Rows = the SAME audio clip (true class). Columns = encoders explaining that audio. "
             "Color scale is per cell (99th pct): read shape and sign, not absolute intensity "
             "across cells. All three encoders use the conservative method (AttnLRP).",
             ha="center", va="bottom", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stdft_lrp_compare")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--clip-indices", type=int, nargs="+", default=[187, 1016],
                    help="índices de clipe (posição em audios_<split>.npy, a mesma entre runs); "
                         "um por linha da grade. Default: 187 (spoof) e 1016 (bonafide).")
    ap.add_argument("--stdft-win", type=int, default=512)
    ap.add_argument("--stdft-hop", type=int, default=128)
    ap.add_argument("--stdft-fmax", type=float, default=4000.0)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--out", default=None, help="PNG de saída (default: <root>/_aggregate/...)")
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

    # Áudios e rótulos verdadeiros vêm de uma run de referência (todas alinhadas pelo seed).
    ref_run = runs[slugs[0]]
    ref_cfg = load_config(ref_run / "config.resolved.yaml")
    split = _split_of(ref_cfg)
    ref_master = A.load_table(ref_run / "master_table.parquet")
    audios = list(np.load(ref_run / f"audios_{split}.npy", allow_pickle=True))
    srs = [int(s) for s in np.load(ref_run / f"srs_{split}.npy")]

    clips = []  # (idx, true_class)
    for i in args.clip_indices:
        gt = int(ref_master["ground_truth"].iloc[i])
        clips.append((i, "spoof" if gt == 1 else "bonafide"))
    log.info(f"clipes: {[(i, c) for i, c in clips]}")

    cells: dict[tuple[int, int], tuple] = {}
    encoders_meta = []  # (short, conservative) por coluna
    for c, slug in enumerate(slugs):
        run_dir = runs[slug]
        cfg = load_config(run_dir / "config.resolved.yaml")
        emb, model, conservative = _build_lrp_model(cfg, run_dir, device)
        master = A.load_table(run_dir / "master_table.parquet")
        p_by_pos = master["p_spoof_ad"].astype(float).to_numpy()
        encoders_meta.append((short[slug], conservative))
        log.info(f"[{short[slug]}] {cfg.model.checkpoint} | "
                 f"{'conservativo' if conservative else 'gxi (referência)'}")
        for r, (i, _tc) in enumerate(clips):
            wav16k = emb._to_16k_mono(audios[i], srs[i])
            x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
            times, freqs, r_tf, _ = dft_lrp.stdft_lrp(x_time, r_time, 16000,
                                                      win_size=args.stdft_win, hop=args.stdft_hop,
                                                      eps=args.eps)
            pred = "spoof" if logit > 0 else "bonafide"
            cells[(r, c)] = (times, freqs, r_tf, x_time, 16000, logit, pred,
                             float(p_by_pos[i]))
        del emb, model
        if device == "cuda":
            torch.cuda.empty_cache()

    out_png = (Path(args.out) if args.out
               else results_root / "_aggregate" / "stdft_lrp_compare.png")
    out_pdf = out_png.with_suffix(".pdf")
    _plot_grid(clips, encoders_meta, cells, args.stdft_fmax, out_png, out_pdf)
    log.info(f"figura salva: {out_png}")
    log.info(f"figura salva: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
