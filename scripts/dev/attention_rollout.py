"""Attention Roll-out temporal (ilustrativo) para encoders SSL do HuggingFace.

Gera uma figura qualitativa que mostra ONDE, no tempo, o backbone Transformer congelado
(wav2vec2 / hubert / wavlm) concentra a atenção, usando o algoritmo de Attention Roll-out
(Abnar & Zuidema, 2020): consolida o fluxo de atenção por todas as camadas, multiplicando
recursivamente as matrizes de atenção ponderadas por uma conexão residual.

Ressalva importante (registrada de propósito): o rollout resume a atenção do BACKBONE, no
eixo TEMPORAL. Isso NÃO é a decisão da nossa regressão logística (D_ad) nem uma explicação
espectral (frequência). O quadrante (TP/FN/TN/FP) dá o contexto do clipe (é mesmo um spoof
pego, um spoof que passou, etc.), mas a atenção mostrada é genérica do backbone. É material
complementar e ilustrativo, ao lado de H1/H2/H3.

Escopo: encoders HF SSL (``encoder=hf_ssl``). XLS-R (fairseq) fica fora.

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/attention_rollout.py \
        --run-dir /workspace/results/hf_ssl-wavlm-base/wavlm-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brspeech_xai.artifacts import RunPaths
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger

# Quadrantes do detector adaptado, com o rótulo legível e a classe de predição (pred_ad).
# pred_ad=1 => o clipe foi predito como spoof; escolhemos como exemplo o de decisão mais
# confiante (P(spoof) mais extremo no sentido da predição), para ilustrar um caso nítido.
_QUADRANTS = [
    ("TP", "TP: spoof correctly flagged", 1),
    ("FN", "FN: spoof missed", 0),
    ("TN", "TN: bonafide accepted", 0),
    ("FP", "FP: bonafide flagged", 1),
]


def compute_rollout(attentions, head_fusion: str = "mean",
                    residual_weight: float = 0.5) -> torch.Tensor:
    """Consolida o fluxo de atenção (Attention Roll-out) por todas as camadas.

    Args:
        attentions: tupla com uma matriz por camada, formato (batch, heads, T, T).
        head_fusion: como fundir as cabeças de atenção ('mean', 'max' ou 'min').
        residual_weight: peso da identidade (conexão residual). 0.5 = padrão da literatura.

    Returns:
        Matriz de atenção conjunta (T, T). A linha i pondera o quanto a posição i "olha"
        para cada posição de origem, já acumulado da entrada até a última camada.
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    eye = torch.eye(seq_len, device=device)
    joint = eye.clone()
    for layer_attn in attentions:
        a = layer_attn[0]  # descarta o batch (assume batch_size=1): (heads, T, T)
        if head_fusion == "mean":
            fused = a.mean(dim=0)
        elif head_fusion == "max":
            fused = a.max(dim=0)[0]
        elif head_fusion == "min":
            fused = a.min(dim=0)[0]
        else:
            raise ValueError("head_fusion deve ser 'mean', 'max' ou 'min'")
        a_new = residual_weight * eye + (1.0 - residual_weight) * fused
        a_new = a_new / a_new.sum(dim=-1, keepdim=True)  # renormaliza por linha
        joint = torch.matmul(a_new, joint)
    return joint


def temporal_relevance(joint: torch.Tensor) -> np.ndarray:
    """Relevância temporal por frame: atenção média recebida (média sobre as consultas).

    Modelos de áudio SSL não têm token [CLS] de resumo, então usamos a média por coluna do
    rollout: quanto cada instante é atendido, em média, por todos os demais.
    """
    return joint.mean(dim=0).detach().cpu().numpy()


def frame_times(n_frames: int, num_samples: int, sample_rate: int) -> np.ndarray:
    """Eixo temporal (segundos) para ``n_frames`` frames de um clipe de comprimento fixo."""
    duration = num_samples / float(sample_rate)
    return np.linspace(0.0, duration, n_frames)


def _pick_examples(master):
    """Escolhe 1 clipe por quadrante do D_ad (o de decisão mais confiante).

    Returns:
        dict quadrante -> (sample_id, p_spoof_ad, ground_truth) ou None se vazio.
    """
    picks: dict[str, tuple[int, float, int] | None] = {}
    for tag, _label, pred in _QUADRANTS:
        sub = master[master["quadrant_ad"] == tag]
        if len(sub) == 0:
            picks[tag] = None
            continue
        # pred spoof => mais confiante é o maior P(spoof); pred bonafide => o menor.
        idx = sub["p_spoof_ad"].idxmax() if pred == 1 else sub["p_spoof_ad"].idxmin()
        row = sub.loc[idx]
        picks[tag] = (int(row["sample_id"]), float(row["p_spoof_ad"]), int(row["ground_truth"]))
    return picks


def _rollout_for_clip(emb, audio: np.ndarray, sr: int, head_fusion: str = "mean"):
    """Pré-processa um clipe como o encoder faz e devolve (waveform_16k, relevância)."""
    wav = emb._to_16k_mono(audio, sr)  # mono, 16 kHz, comprimento fixo (~4.04s)
    inputs = emb._processor([wav], sampling_rate=16000, return_tensors="pt",
                            padding=True, return_attention_mask=True)
    input_values = inputs["input_values"].to(emb.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(emb.device)
    with torch.inference_mode():
        out = emb._model(input_values, attention_mask=attention_mask, output_attentions=True)
    atts = getattr(out, "attentions", None)
    if not atts or atts[0] is None:
        raise RuntimeError("o modelo não retornou matrizes de atenção; carregue-o com "
                           "attn_implementation='eager' (sdpa devolve None).")
    rel = temporal_relevance(compute_rollout(atts, head_fusion=head_fusion))
    return wav, rel


def _plot(examples, checkpoint: str, out_png: Path, out_pdf: Path,
          num_samples: int, sample_rate: int) -> None:
    from brspeech_xai.plotting import CB_PALETTE, set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    wave_color = "0.6"
    rel_color = CB_PALETTE[1]  # laranja (consistente com D_ad nas outras figuras)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.6), sharex=True)
    for ax, (tag, label, _pred) in zip(axes, _QUADRANTS):
        item = examples[tag]
        if item is None:
            ax.text(0.5, 0.5, f"{label}\n(sem exemplo neste quadrante)",
                    ha="center", va="center", transform=ax.transAxes, color="0.4")
            ax.set_yticks([])
            continue
        wav, rel, p_ad = item
        t_wav = np.linspace(0.0, num_samples / float(sample_rate), len(wav))
        w = wav / (np.max(np.abs(wav)) + 1e-9)  # normaliza para [-1, 1] só para exibir
        ax.plot(t_wav[::3], w[::3], color=wave_color, lw=0.4, alpha=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([])
        ax.set_ylabel("waveform", color=wave_color)

        t_rel = frame_times(len(rel), num_samples, sample_rate)
        rr = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)  # 0..1 só para exibir
        ax2 = ax.twinx()
        ax2.fill_between(t_rel, rr, color=rel_color, alpha=0.30)
        ax2.plot(t_rel, rr, color=rel_color, lw=1.2)
        thr = np.percentile(rr, 90)
        ax2.axhline(thr, color=rel_color, ls=":", lw=0.9)
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([])
        ax2.set_ylabel("attention", color=rel_color)
        ax.set_title(f"{label}   (P(spoof)={p_ad:.2f})", loc="left")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Temporal attention roll-out — backbone {checkpoint}", y=0.995)
    fig.text(0.5, 0.005,
             "Where the frozen backbone attends over time (dotted line = top-10% frames). "
             "Not the logistic decision and not frequency-specific.",
             ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="attention_rollout")
    ap.add_argument("--run-dir", required=True, help="diretório de uma run de encoder hf_ssl")
    ap.add_argument("--head-fusion", default="mean", choices=["mean", "max", "min"])
    args = ap.parse_args(argv)

    log = get_logger()
    run_dir = Path(args.run_dir)
    paths = RunPaths(root=run_dir)
    cfg = load_config(paths.path("config.resolved.yaml"))
    if cfg.model.encoder != "hf_ssl":
        raise SystemExit(f"encoder={cfg.model.encoder!r} não suportado; "
                         "o rollout só cobre os encoders HF SSL (encoder=hf_ssl).")

    from brspeech_xai import artifacts as A
    from brspeech_xai.encoders.hf_ssl import HFSSLEmbedder

    split = "pool" if cfg.adapt.cross_fit else cfg.data.eval_split
    master = A.load_table(paths.path("master_table.parquet"))
    audios = list(np.load(paths.path(f"audios_{split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(paths.path(f"srs_{split}.npy"))]

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"attention roll-out: {cfg.model.checkpoint} | split={split} | device={device}")
    emb = HFSSLEmbedder(checkpoint=cfg.model.checkpoint, layer=cfg.model.layer,
                        pooling=cfg.model.pooling, device=device,
                        num_samples=cfg.audio.num_samples, attn_implementation="eager")
    # output_attentions só funciona com atenção "eager"; o padrão sdpa (wav2vec2/hubert)
    # devolve atenções None. Recarrega o backbone em modo eager só para o rollout.
    from transformers import AutoModel
    emb._model = AutoModel.from_pretrained(
        cfg.model.checkpoint, attn_implementation="eager").to(device).eval()

    picks = _pick_examples(master)
    examples: dict = {}
    for tag, label, _pred in _QUADRANTS:
        item = picks[tag]
        if item is None:
            log.warning(f"quadrante {tag} vazio; painel ficará em branco")
            examples[tag] = None
            continue
        sid, p_ad, _gt = item
        wav, rel = _rollout_for_clip(emb, audios[sid], srs[sid], head_fusion=args.head_fusion)
        log.info(f"{label}: sample_id={sid} P(spoof)={p_ad:.3f} frames={len(rel)}")
        examples[tag] = (wav, rel, p_ad)

    out_png = paths.figure("attention_rollout_ad.png")
    out_pdf = paths.figure("attention_rollout_ad.pdf")
    _plot(examples, cfg.model.checkpoint.split("/")[-1], out_png, out_pdf,
          cfg.audio.num_samples, cfg.audio.sample_rate)
    log.info(f"figura salva: {out_png}")
    log.info(f"figura salva: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
