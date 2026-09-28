"""Compara o attention roll-out entre encoders HF SSL usando os MESMOS clipes.

Escolhe 4 clipes de referência (1 por quadrante do D_ad de uma run de referência) e monta
uma grade clipes × encoders: cada célula mostra como aquele backbone atende ao MESMO áudio.
Assim a comparação entre encoders é justa (mesma entrada). O P(spoof) anotado por célula é o
do próprio encoder, para deixar visível se eles concordam na decisão.

As runs HF SSL default compartilham a mesma ordem de split (mesmo seed), então o ``sample_id``
aponta para o mesmo áudio em todas; carregamos os áudios só da run de referência.

Ressalva (a mesma do rollout por-encoder): é atenção TEMPORAL do backbone congelado, não a
decisão da logística e não é espectral.

Uso (dentro do container):
    python /workspace/scripts/dev/attention_rollout_compare.py \
        --results-root /workspace/results --ref-slug wavlm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from attention_rollout import _QUADRANTS, _pick_examples, _rollout_for_clip, frame_times
from brspeech_xai import artifacts as A
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger


def _discover_runs(results_root: Path) -> dict[str, Path]:
    """Última run por encoder HF SSL (slug = pasta 'hf_ssl-*'), com artefatos necessários."""
    runs: dict[str, Path] = {}
    for slug_dir in sorted(results_root.glob("hf_ssl-*")):
        cands = [d for d in slug_dir.glob("*")
                 if (d / "master_table.parquet").exists()
                 and (d / "config.resolved.yaml").exists()]
        if cands:
            runs[slug_dir.name] = max(cands, key=lambda d: d.name)  # timestamp no nome ordena
    return runs


def _split_of(cfg) -> str:
    return "pool" if cfg.adapt.cross_fit else cfg.data.eval_split


def _enc_p_spoof(run_dir: Path, sample_ids: list[int]) -> dict[int, float]:
    """P(spoof) do D_ad daquele encoder para cada sample_id (lido do master_table)."""
    master = A.load_table(run_dir / "master_table.parquet")
    by_id = dict(zip(master["sample_id"].astype(int), master["p_spoof_ad"].astype(float)))
    return {sid: by_id.get(int(sid), float("nan")) for sid in sample_ids}


def _plot_grid(clips, encoders, cells, num_samples, sample_rate, out_png, out_pdf):
    """Grade clipes (linhas) × encoders (colunas).

    clips: lista de (quadrant_tag, quadrant_label, sample_id, ref_p_spoof).
    encoders: lista de nomes curtos de encoder (ordem das colunas).
    cells: dict (row_idx, col_idx) -> (waveform, relevance, enc_p_spoof).
    """
    from brspeech_xai.plotting import CB_PALETTE, set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    wave_color, rel_color = "0.6", CB_PALETTE[1]
    nrows, ncols = len(clips), len(encoders)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.0 * nrows),
                             sharex=True, squeeze=False)
    dur = num_samples / float(sample_rate)
    for r, (_tag, qlabel, _sid, ref_p) in enumerate(clips):
        for c, enc in enumerate(encoders):
            ax = axes[r][c]
            wav, rel, enc_p = cells[(r, c)]
            t_wav = np.linspace(0.0, dur, len(wav))
            w = wav / (np.max(np.abs(wav)) + 1e-9)
            ax.plot(t_wav[::3], w[::3], color=wave_color, lw=0.4, alpha=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_yticks([])
            t_rel = frame_times(len(rel), num_samples, sample_rate)
            rr = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
            ax2 = ax.twinx()
            ax2.fill_between(t_rel, rr, color=rel_color, alpha=0.30)
            ax2.plot(t_rel, rr, color=rel_color, lw=1.1)
            ax2.set_ylim(0, 1.05)
            ax2.set_yticks([])
            ax.set_title(f"P(spoof)={enc_p:.2f}", loc="right", fontsize=7, color=rel_color)
            if r == 0:
                ax.annotate(enc, xy=(0.5, 1.28), xycoords="axes fraction",
                            ha="center", va="bottom", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{qlabel}\n(ref P={ref_p:.2f})", fontsize=8)
    for c in range(ncols):
        axes[-1][c].set_xlabel("time (s)")
    fig.suptitle("Temporal attention roll-out — same clips across encoders", y=0.995)
    fig.text(0.5, 0.004,
             "Rows = clips (quadrant of the reference detector). Columns = encoders attending "
             "the SAME audio. Per-cell P(spoof) is that encoder's own decision. "
             "Backbone temporal attention only: not the logistic decision, not frequency-specific.",
             ha="center", va="bottom", fontsize=6.5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="attention_rollout_compare")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--ref-slug", default="wavlm",
                    help="substring do slug do encoder de referência (escolhe os clipes)")
    ap.add_argument("--head-fusion", default="mean", choices=["mean", "max", "min"])
    ap.add_argument("--out", default=None, help="PNG de saída (default: <root>/_aggregate/...)")
    args = ap.parse_args(argv)

    log = get_logger()
    from brspeech_xai.encoders.hf_ssl import HFSSLEmbedder

    results_root = Path(args.results_root)
    runs = _discover_runs(results_root)
    if not runs:
        raise SystemExit(f"nenhuma run hf_ssl com artefatos em {results_root}")
    ref_slug = next((s for s in runs if args.ref_slug in s), None)
    if ref_slug is None:
        raise SystemExit(f"ref-slug {args.ref_slug!r} não encontrado entre {sorted(runs)}")

    # Ordena as colunas com a referência primeiro (leitura mais natural).
    slugs = [ref_slug] + [s for s in runs if s != ref_slug]
    log.info(f"encoders: {slugs} | referência: {ref_slug}")

    ref_run = runs[ref_slug]
    ref_cfg = load_config(ref_run / "config.resolved.yaml")
    split = _split_of(ref_cfg)
    ref_master = A.load_table(ref_run / "master_table.parquet")
    picks = _pick_examples(ref_master)
    clips = []  # (tag, label, sample_id, ref_p_spoof)
    for tag, label, _pred in _QUADRANTS:
        item = picks[tag]
        if item is None:
            log.warning(f"quadrante {tag} vazio na referência; será ignorado")
            continue
        sid, p_ad, _gt = item
        clips.append((tag, label, sid, p_ad))
    if not clips:
        raise SystemExit("nenhum clipe de referência disponível")
    sample_ids = [sid for _t, _l, sid, _p in clips]

    audios = list(np.load(ref_run / f"audios_{split}.npy", allow_pickle=True))
    srs = [int(s) for s in np.load(ref_run / f"srs_{split}.npy")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = {s: s.replace("hf_ssl-", "") for s in slugs}
    cells: dict[tuple[int, int], tuple] = {}
    for c, slug in enumerate(slugs):
        run_dir = runs[slug]
        cfg = load_config(run_dir / "config.resolved.yaml")
        log.info(f"[{short[slug]}] {cfg.model.checkpoint} (device={device})")
        emb = HFSSLEmbedder(checkpoint=cfg.model.checkpoint, layer=cfg.model.layer,
                            pooling=cfg.model.pooling, device=device,
                            num_samples=cfg.audio.num_samples, attn_implementation="eager")
        p_by_id = _enc_p_spoof(run_dir, sample_ids)
        for r, (_tag, _label, sid, _p) in enumerate(clips):
            wav, rel = _rollout_for_clip(emb, audios[sid], srs[sid],
                                         head_fusion=args.head_fusion)
            cells[(r, c)] = (wav, rel, p_by_id[sid])
        del emb
        if device == "cuda":
            torch.cuda.empty_cache()

    out_png = Path(args.out) if args.out else results_root / "_aggregate" / "attention_rollout_compare.png"
    out_pdf = out_png.with_suffix(".pdf")
    _plot_grid(clips, [short[s] for s in slugs], cells,
               ref_cfg.audio.num_samples, ref_cfg.audio.sample_rate, out_png, out_pdf)
    log.info(f"figura salva: {out_png}")
    log.info(f"figura salva: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
