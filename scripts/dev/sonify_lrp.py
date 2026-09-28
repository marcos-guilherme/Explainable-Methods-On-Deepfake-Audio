"""Sonifica a explicação DFT-LRP: gera áudio ouvível do que empurra a decisão para cada lado.

Ideia (análoga ao soft-masking do L2I, Parekh et al. 2024, mas com bins de frequência no lugar de
componentes NMF): para um clipe, pegamos a relevância tempo-frequência R(t,f) que o detector
adaptado (D_ad) atribui à decisão de spoof (STDFT-LRP conservativo). Separamos por direção,
R+ = max(R,0) (rumo a spoof) e R- = max(-R,0) (rumo a bonafide), viramos cada uma numa máscara
suave M(t,f) em [0,1] e a aplicamos à magnitude do STFT do próprio clipe, mantendo a fase original.
A STFT inversa (mesma janela Hann + WOLA da análise, logo invertível) devolve um .wav que "acende"
só as regiões usadas pelo modelo. Assim dá para OUVIR o que leva a spoof e o que leva a bonafide.

Compara os mesmos clipes entre os encoders conservativos (wav2vec2, hubert, wavlm). Saídas por
clipe: original.wav e, por encoder, toward_spoof_<enc>.wav e toward_bonafide_<enc>.wav; mais uma
figura ligando o áudio à envoltória temporal da relevância e um manifest.csv com rótulo/decisão.

RESSALVA: o soft-masking introduz artefatos (mesmo confound da oclusão). É ilustração qualitativa,
não prova, exatamente como o próprio L2I ressalva.

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/sonify_lrp.py --results-root /workspace/results \
        --clip-indices 187 1016
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from brspeech_xai import artifacts as A
from brspeech_xai import dft_lrp
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger
from dft_lrp_ad import relevance_for_clip
from stdft_lrp_compare import (_build_lrp_model, _discover_runs, _order_slugs, _split_of)


def _auto_clips(ref_master, per_quadrant: int) -> list[tuple[int, str]]:
    """Escolhe clipes variados por quadrante da matriz de confusão (referência = 1º encoder).

    Para cada quadrante pega os ``per_quadrant`` casos mais informativos:
      TP (spoof certo) e FN (spoof errado como bonafide): spoof verdadeiro;
      TN (bonafide certo) e FP (bonafide errado como spoof): bonafide verdadeiro.
    Acertos ordenados pela maior confiança; erros pela maior confiança no lado ERRADO (pior caso).
    """
    gt = ref_master["ground_truth"].astype(int).to_numpy()
    p = ref_master["p_spoof_ad"].astype(float).to_numpy()
    pred = (p > 0.5).astype(int)
    quad = {
        "TP": (np.where((gt == 1) & (pred == 1))[0], -p),   # spoof certo, mais confiante
        "FN": (np.where((gt == 1) & (pred == 0))[0], p),    # spoof visto como bonafide (pior)
        "TN": (np.where((gt == 0) & (pred == 0))[0], p),    # bonafide certo, mais confiante
        "FP": (np.where((gt == 0) & (pred == 1))[0], -p),   # bonafide visto como spoof (pior)
    }
    out: list[tuple[int, str]] = []
    for name, (idx, key) in quad.items():
        if len(idx) == 0:
            continue
        order = idx[np.argsort(key[idx])][:per_quadrant]
        for i in order:
            out.append((int(i), "spoof" if gt[i] == 1 else "bonafide"))
    return out


def _write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    """Grava PCM 16 bits, normalizado ao pico do próprio sinal (para ficar audível)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(x))) or 1.0
    xi = np.int16(np.clip(x / peak, -1.0, 1.0) * 32767)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, xi)


def _softmask_reconstruct(x_time: np.ndarray, r_tf: np.ndarray, win: int, hop: int,
                          direction: str, pctl: float = 99.0, floor: float = 0.0) -> np.ndarray:
    """STFT inversa do clipe com a magnitude ponderada pela máscara de relevância de uma direção.

    Replica EXATAMENTE a análise de ``dft_lrp.stdft_lrp`` (pad win//2, janela Hann, WOLA por
    ``wsum``), então com máscara 1 recupera o sinal original. A máscara vem de R+ (spoof) ou R-
    (bonafide), normalizada pelo percentil ``pctl`` sobre o clipe todo e cortada em [0,1]. ``floor``
    mantém um piso de energia fora das regiões relevantes (0 = isola só o que o modelo usa).
    """
    x = np.asarray(x_time, dtype=np.float64).ravel()
    n = x.shape[-1]
    pad = win // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    npad = xp.shape[-1]
    w = np.hanning(win)
    starts = np.arange(0, npad - win + 1, hop)
    wsum = np.zeros(npad, dtype=np.float64)
    for s in starts:
        wsum[s:s + win] += w
    raw = np.clip(r_tf, 0, None) if direction == "spoof" else np.clip(-r_tf, 0, None)
    p = float(np.percentile(raw, pctl)) if raw.any() else 1.0
    mask = np.clip(raw / (p + 1e-12), 0.0, 1.0)            # (M, F) em [0,1]
    if floor:
        mask = floor + (1.0 - floor) * mask
    y = np.zeros(npad, dtype=np.float64)
    for m, s in enumerate(starts):
        sl = slice(s, s + win)
        c = np.fft.rfft(xp[sl] * w)
        y[sl] += np.fft.irfft(c * mask[m], n=win)
    y = y / np.where(wsum > 0, wsum, 1.0)
    return y[pad:pad + n]


def _plot_envelopes(clip_tag: str, true_cls: str, per_enc: list[dict], times, out_png, out_pdf):
    """Companheira visual: por encoder, forma de onda + envoltória temporal da relevância.

    Vermelho: soma por quadro de R+ (rumo a spoof); azul: soma de R- (rumo a bonafide). Ajuda a
    'ver' o que se ouve nos .wav (quais instantes acendem para cada lado)."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    n = len(per_enc)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 2.1 * n), squeeze=False, sharex=True)
    for r, d in enumerate(per_enc):
        ax = axes[r][0]
        x = d["x_time"]
        tx = np.arange(len(x)) / 16000.0
        ax.plot(tx, x / (np.max(np.abs(x)) or 1.0), color="0.7", lw=0.5, zorder=1)
        es = d["env_spoof"] / (d["env_spoof"].max() or 1.0)
        eb = d["env_bona"] / (d["env_bona"].max() or 1.0)
        ax.fill_between(times, 0, es, color="#c0392b", alpha=0.45, zorder=2, label="toward spoof")
        ax.fill_between(times, 0, -eb, color="#2c66b0", alpha=0.45, zorder=2, label="toward bonafide")
        hit = "OK" if d["pred"] == true_cls else "MISS"
        ax.set_ylabel(f"{d['short']}", fontsize=9)
        ax.set_title(f"logit={d['logit']:+.1f} -> {d['pred']} [{hit}]  (p_spoof={d['p_spoof']:.2f})",
                     fontsize=8, loc="left")
        ax.set_ylim(-1.05, 1.05)
        if r == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1][0].set_xlabel("t [s]")
    fig.suptitle(f"DFT-LRP relevance envelope — clip {clip_tag} (true: {true_cls})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sonify_lrp")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--clip-indices", type=int, nargs="+", default=[187, 1016],
                    help="índices de clipe (posição em audios_<split>.npy, a mesma entre runs). "
                         "Default: 187 (spoof) e 1016 (bonafide). Ignorado se --per-quadrant > 0.")
    ap.add_argument("--per-quadrant", type=int, default=0,
                    help="se > 0, escolhe automaticamente N clipes por quadrante (TP/FN/TN/FP) da "
                         "matriz de confusão do encoder de referência (variedade de fácil e difícil).")
    ap.add_argument("--stdft-win", type=int, default=512)
    ap.add_argument("--stdft-hop", type=int, default=128)
    ap.add_argument("--pctl", type=float, default=99.0,
                    help="percentil para normalizar a máscara (maior = máscara mais seletiva)")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="piso de energia fora das regiões relevantes (0 = isola só o que o modelo usa)")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--exclude", nargs="*", default=["mms-300m"],
                    help="encoders (short) a não sonificar. Default: mms-300m (removido por ora).")
    ap.add_argument("--out", default=None, help="dir de saída (default: <root>/_aggregate/sonification)")
    args = ap.parse_args(argv)

    log = get_logger()
    results_root = Path(args.results_root)
    runs = _discover_runs(results_root)
    if not runs:
        raise SystemExit(f"nenhuma run hf_ssl com d_ad.joblib em {results_root}")
    slugs = _order_slugs(runs)
    short = {s: s.replace("hf_ssl-", "") for s in slugs}
    excl = set(args.exclude or [])
    if excl:
        slugs = [s for s in slugs if short[s] not in excl]
        log.info(f"excluídos: {sorted(excl)}")
    log.info(f"encoders: {[short[s] for s in slugs]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ref_run = runs[slugs[0]]
    ref_cfg = load_config(ref_run / "config.resolved.yaml")
    split = _split_of(ref_cfg)
    ref_master = A.load_table(ref_run / "master_table.parquet")
    audios = list(np.load(ref_run / f"audios_{split}.npy", allow_pickle=True))
    srs = [int(s) for s in np.load(ref_run / f"srs_{split}.npy")]

    if args.per_quadrant > 0:
        clips = _auto_clips(ref_master, args.per_quadrant)
    else:
        clips = [(i, "spoof" if int(ref_master["ground_truth"].iloc[i]) == 1 else "bonafide")
                 for i in args.clip_indices]
    log.info(f"clipes ({len(clips)}): {clips}")

    out_dir = Path(args.out) if args.out else results_root / "_aggregate" / "sonification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_clip[idx] -> list de dicts por encoder (para a figura); manifest acumula as decisões.
    per_clip: dict[int, dict] = {i: {"true": tc, "encs": []} for i, tc in clips}
    manifest_rows = []

    for slug in slugs:
        run_dir = runs[slug]
        cfg = load_config(run_dir / "config.resolved.yaml")
        emb, model, conservative = _build_lrp_model(cfg, run_dir, device)
        master = A.load_table(run_dir / "master_table.parquet")
        p_by_pos = master["p_spoof_ad"].astype(float).to_numpy()
        log.info(f"[{short[slug]}] {cfg.model.checkpoint} | "
                 f"{'conservativo' if conservative else 'gxi (referência)'}")
        for i, true_cls in clips:
            wav16k = emb._to_16k_mono(audios[i], srs[i])
            x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
            times, _freqs, r_tf, _ = dft_lrp.stdft_lrp(x_time, r_time, 16000,
                                                       win_size=args.stdft_win, hop=args.stdft_hop,
                                                       eps=args.eps)
            pred = "spoof" if logit > 0 else "bonafide"
            clip_tag = f"{i:05d}_{true_cls}"
            clip_dir = out_dir / clip_tag
            # Original só uma vez por clipe (mesmo áudio entre encoders).
            orig_path = clip_dir / "original.wav"
            if not orig_path.exists():
                _write_wav(orig_path, x_time)
            y_sp = _softmask_reconstruct(x_time, r_tf, args.stdft_win, args.stdft_hop,
                                         "spoof", pctl=args.pctl, floor=args.floor)
            y_bo = _softmask_reconstruct(x_time, r_tf, args.stdft_win, args.stdft_hop,
                                         "bonafide", pctl=args.pctl, floor=args.floor)
            _write_wav(clip_dir / f"toward_spoof_{short[slug]}.wav", y_sp)
            _write_wav(clip_dir / f"toward_bonafide_{short[slug]}.wav", y_bo)
            per_clip[i]["encs"].append({
                "short": short[slug], "x_time": np.asarray(x_time, dtype=np.float64),
                "env_spoof": np.clip(r_tf, 0, None).sum(axis=1),
                "env_bona": np.clip(-r_tf, 0, None).sum(axis=1),
                "logit": float(logit), "pred": pred, "p_spoof": float(p_by_pos[i]),
                "times": times,
            })
            manifest_rows.append({"clip": clip_tag, "index": i, "true_class": true_cls,
                                  "encoder": short[slug], "logit": round(float(logit), 3),
                                  "pred": pred, "p_spoof_ad": round(float(p_by_pos[i]), 4),
                                  "hit": int(pred == true_cls)})
        del emb, model
        if device == "cuda":
            torch.cuda.empty_cache()

    for i, true_cls in clips:
        d = per_clip[i]
        clip_tag = f"{i:05d}_{true_cls}"
        clip_dir = out_dir / clip_tag
        times = d["encs"][0]["times"]
        _plot_envelopes(clip_tag, true_cls, d["encs"], times,
                        clip_dir / "relevance_envelope.png", clip_dir / "relevance_envelope.pdf")

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        wr.writeheader()
        wr.writerows(manifest_rows)
    log.info(f"sonificação salva em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
