"""DFT-LRP para o detector adaptado (D_ad) sobre um encoder SSL do HuggingFace.

Terceira via de explicação, ao lado da associação (H1) e da oclusão (H2): olha POR DENTRO do
modelo e mede a relevância que o D_ad atribui a cada faixa de frequência ao decidir "spoof".
Combina duas peças da literatura:

  * AttnLRP (Achtibat et al., 2024), reimplementado à mão em ``brspeech_xai.attnlrp``: faz a
    relevância atravessar o Transformer (atenção, normalização, GELU) de modo conservativo até
    a entrada de áudio no tempo. Suporte conservativo: wav2vec2, hubert e wavlm.
  * DFT-LRP (Vielhaben et al., 2024): leva a relevância do tempo para a frequência por uma
    camada de inspeção virtual (a forma fechada testada em ``brspeech_xai.dft_lrp``).

Escopo: encoder ``hf_ssl`` com head ``logistic`` (o D_ad é uma regressão logística sobre os
embeddings congelados). Só o D_ad existe no caminho HuggingFace (encoder é só extrator).

CORREÇÃO DO MÉTODO (critério intrínseco, sem comparar com nada externo):
  1) camada DFT: sum_k R_k == sum_n R_n (exata por construção, vale para qualquer backend);
  2) CERTIFICADO: com os vieses do encoder zerados, a rede fica homogênea de grau 1 e a
     relevância da entrada recupera (logit - b_head) na mosca (Euler). Resíduo ~0 => regras certas.
  3) COMPLETUDE (não é erro): com os vieses reais, a relevância da entrada explica a PARTE do logit
     que depende da forma de onda; o restante é carregado pelos vieses internos (parcela reportada
     na decomposição). Isso é propriedade conhecida do Input×Gradient com viés, não defeito.

NÃO introduzir viés metodológico: o resultado fiel usa as regras puras (--norm-stab 0). O botão
--norm-stab (κ>0) só existe para diagnóstico; ele altera a explicação e quebra o certificado.

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/dft_lrp_ad.py \
        --run-dir /workspace/results/hf_ssl-wav2vec2-base/wav2vec2-YYYYMMDD-HHMMSS \
        --backend attnlrp        # ou: gxi (Gradient×Input, referência não-conservativa)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brspeech_xai import artifacts as A
from brspeech_xai import dft_lrp
from brspeech_xai.config import load_config
from brspeech_xai.lrp_detector import (
    SSLDetectorAD,
    _legacy_ssl_detector_ad,
    conservation_certificate,
    port_logistic_head,
    relevance_for_clip,
)
from brspeech_xai.logging_utils import get_logger
from brspeech_xai.occlusion import bootstrap_ci, mel_band_edges, stratified_idx


def _plot(edges, mean_rel, ci_low, ci_high, checkpoint, backend, out_png, out_pdf):
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    centers = (edges[:-1] + edges[1:]) / 2
    sustain_spoof, sustain_bona = "#d62728", "#1f77b4"
    fig, ax = plt.subplots(figsize=(3.33, 2.8))
    ax.fill_between(centers, 0, mean_rel, where=mean_rel >= 0, interpolate=True,
                    color=sustain_spoof, alpha=0.45, label="toward spoof")
    ax.fill_between(centers, 0, mean_rel, where=mean_rel < 0, interpolate=True,
                    color=sustain_bona, alpha=0.45, label="toward bonafide")
    ax.fill_between(centers, ci_low, ci_high, color="#333333", alpha=0.15, lw=0,
                    label="95% CI")
    ax.plot(centers, mean_rel, color="#333333", lw=1.0, alpha=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("Frequency band center (Hz)")
    ax.set_ylabel("DFT-LRP relevance (spoof logit)")
    ax.set_title(f"DFT-LRP band relevance — $D_{{ad}}$ ({checkpoint}, {backend})",
                 fontsize=8, loc="left")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def logit_only(model: SSLDetectorAD, processor, wav16k: np.ndarray, device: str) -> float:
    """Só o logit de spoof (sem backward), para escolher clipes exemplo pela confiança."""
    inputs = processor([wav16k], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        return float(model(inputs["input_values"].to(device))[0])


def _plot_stdft(times, freqs, r_tf, x_time, sr, fmax, title, out_png, out_pdf):
    """Figura estilo STDFT-LRP (Vielhaben et al.): heatmap tempo-frequência da relevância, com
    a marginal em frequência (à esquerda) e a forma de onda (embaixo).

    Cor divergente centrada em 0: vermelho = rumo a spoof, azul = rumo a bonafide.
    """
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    set_plot_style()
    mask = freqs <= fmax
    f = freqs[mask]
    R = r_tf[:, mask].T                                # (freq, tempo)
    v = np.percentile(np.abs(R), 99) or 1.0           # escala simétrica robusta
    t_wave = np.arange(len(x_time)) / sr
    marg = r_tf[:, mask].sum(axis=0)                  # marginal em frequência (soma no tempo)

    fig = plt.figure(figsize=(5.2, 3.6))
    gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[3, 1], wspace=0.05, hspace=0.08)
    ax_main = fig.add_subplot(gs[0, 1])
    ax_freq = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_wave = fig.add_subplot(gs[1, 1], sharex=ax_main)

    ax_main.pcolormesh(times, f, R, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax_main.set_title(title, fontsize=8, loc="left")
    ax_main.tick_params(labelbottom=False, labelleft=False)

    ax_freq.plot(marg, f, color="#333333", lw=0.9)
    ax_freq.axvline(0, color="grey", ls="--", lw=0.6)
    ax_freq.set_ylabel("Frequency (Hz)")
    ax_freq.set_xlabel("rel.", fontsize=7)
    ax_freq.invert_xaxis()                             # relevância cresce em direção ao heatmap

    ax_wave.plot(t_wave, x_time, color="grey", lw=0.4)
    ax_wave.set_xlabel("t [s]")
    ax_wave.set_yticks([])
    ax_wave.set_xlim(times.min(), times.max())
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_salience(centers, mean_share, ci_low, ci_high, checkpoint, backend, out_png, out_pdf):
    """Espectro de saliência: quanto da atenção (relevância em módulo) cai em cada frequência.

    ``mean_share`` é a fração média (por clipe) da relevância absoluta em cada bin fino, em %.
    Direção-agnóstico (spoof e bonafide contam igual): mostra ONDE o modelo olha, não para que lado.
    """
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.fill_between(centers, 0, mean_share, color="#6a3d9a", alpha=0.35, label="attention share")
    ax.fill_between(centers, ci_low, ci_high, color="#333333", alpha=0.15, lw=0, label="95% CI")
    ax.plot(centers, mean_share, color="#6a3d9a", lw=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Attention share |relevance| (% per bin)")
    ax.set_title(f"DFT-LRP frequency attention — $D_{{ad}}$ ({checkpoint}, {backend})",
                 fontsize=8, loc="left")
    ax.legend(loc="best", fontsize=7)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_by_class(edges, groups, checkpoint, backend, out_png, out_pdf):
    """Perfil por classe PREDITA (spoof vs bonafide) na mesma figura.

    ``groups``: dict classe -> (mean, ci_low, ci_high, n). Uma linha por classe, com a faixa de
    confiança. Condicionar por classe evita o cancelamento de sinais opostos da média global.
    """
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    centers = (edges[:-1] + edges[1:]) / 2
    colors = {"predicted spoof": "#d62728", "predicted bonafide": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for label, (mean_rel, lo, hi, n) in groups.items():
        c = colors.get(label, "#333333")
        ax.fill_between(centers, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(centers, mean_rel, color=c, lw=1.2, label=f"{label} (n={n})")
    ax.axhline(0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("Frequency band center (Hz)")
    ax.set_ylabel("DFT-LRP relevance (spoof logit)")
    ax.set_title(f"DFT-LRP by predicted class — $D_{{ad}}$ ({checkpoint}, {backend})",
                 fontsize=8, loc="left")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def run_stdft_examples(model, emb, audios, srs, idx, args, cfg, paths, backend, log,
                       master=None) -> int:
    """Gera figuras STDFT-LRP de clipes exemplo e encerra (rápido, sem o perfil agregado).

    Sem ``--clip-indices``: escolhe os clipes mais confiantes de cada lado (maior e menor logit
    num pool pequeno), pois a relevância fica mais legível quando a decisão é clara.

    Com ``--clip-indices``: usa exatamente esses índices (posição em ``audios_<split>.npy``, a
    mesma entre runs), para comparar o MESMO clipe entre modelos. O nome do arquivo usa a classe
    VERDADEIRA (``ground_truth`` do master), então o par fica estável mesmo se um modelo errar.
    Uma figura por clipe.
    """
    device = model.w.device.type
    tag = backend if args.norm_stab <= 0 else f"{backend}_ns{args.norm_stab:g}"
    ckpt = cfg.model.checkpoint.split("/")[-1]

    if args.clip_indices:
        picks = [int(i) for i in args.clip_indices]
        log.info(f"STDFT-LRP clipes fixos: {picks} | win={args.stdft_win} hop={args.stdft_hop} "
                 f"| fmax={args.stdft_fmax:g} Hz")
    else:
        # Pool espalhado por TODO o idx estratificado (não só o começo), para cobrir os quadrantes
        # do D_ad e ter exemplos dos dois lados (spoof e bonafide), não só de um quadrante.
        pool_size = max(60, 4 * args.stdft_examples)
        step = max(1, len(idx) // pool_size)
        pool = list(idx[::step])
        logits = np.array([logit_only(model, emb._processor,
                                      emb._to_16k_mono(audios[i], srs[i]), device) for i in pool])
        order = np.argsort(logits)                      # ascendente: bonafide -> spoof
        n = args.stdft_examples
        n_sp = (n + 1) // 2                             # metade mais "spoof", metade mais "bonafide"
        picks = [pool[k] for k in order[-n_sp:][::-1]] + [pool[k] for k in order[:n - n_sp]]
        log.info(f"STDFT-LRP exemplos: {len(picks)} clipes | win={args.stdft_win} "
                 f"hop={args.stdft_hop} | fmax={args.stdft_fmax:g} Hz")

    def _true_cls(i: int) -> str | None:
        if master is not None and "ground_truth" in getattr(master, "columns", []):
            return "spoof" if int(master["ground_truth"].iloc[i]) == 1 else "bonafide"
        return None

    for i in picks:
        wav16k = emb._to_16k_mono(audios[i], srs[i])
        x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
        times, freqs, r_tf, _ = dft_lrp.stdft_lrp(x_time, r_time, 16000,
                                                  win_size=args.stdft_win, hop=args.stdft_hop,
                                                  eps=args.eps)
        pred = "spoof" if logit > 0 else "bonafide"
        true_cls = _true_cls(i)
        cls_for_name = true_cls or pred                 # nome pela classe real quando disponível
        cls_tag = f" [{true_cls}]" if true_cls else ""
        title = (f"STDFT-LRP $D_{{ad}}$ ({ckpt}) — clip #{i}{cls_tag}: "
                 f"logit={logit:+.2f} → {pred}")
        stem = f"stdft_lrp_ad_{tag}_clip{i:04d}_{cls_for_name}"
        out_png = paths.figure(f"{stem}.png")
        out_pdf = paths.figure(f"{stem}.pdf")
        _plot_stdft(times, freqs, r_tf, x_time, 16000, args.stdft_fmax, title, out_png, out_pdf)
        cons = abs(float(r_tf.sum()) - float(r_time.sum()))
        log.info(f"  clip #{i} (true={true_cls}, pred={pred}, logit={logit:+.2f}): "
                 f"STDFT conserva |Δ|={cons:.2e} | {out_png}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dft_lrp_ad")
    ap.add_argument("--run-dir", required=True, help="diretório de uma run de encoder hf_ssl")
    ap.add_argument("--backend", default="attnlrp", choices=["attnlrp", "gxi"],
                    help="attnlrp (à mão, conservativo, wav2vec2/hubert/wavlm) ou gxi (Gradient×Input)")
    ap.add_argument("--attn-rule", default="cp", choices=["cp", "uniform"],
                    help="regra da atenção no backend attnlrp: cp (conservativo) ou uniform")
    ap.add_argument("--norm-stab", type=float, default=0.0,
                    help="DIAGNÓSTICO apenas. κ do piso no std do backward das normas. 0 = regra da "
                         "identidade pura (resultado fiel). κ>0 é um botão arbitrário que altera a "
                         "explicação (introduz viés) e quebra o certificado de conservação; não usar "
                         "no resultado reportado.")
    ap.add_argument("--norm-stab-scope", default="extractor", choices=["extractor", "all"],
                    help="onde κ vale (só com --norm-stab>0): extractor (front-end conv) ou all")
    ap.add_argument("--per-quadrant", type=int, default=150,
                    help="clipes por quadrante do D_ad (mesma amostragem da oclusão)")
    ap.add_argument("--salience-bins", type=int, default=256,
                    help="bins finos (lineares) do espectro de saliência em frequência")
    ap.add_argument("--stdft-examples", type=int, default=0,
                    help=">0: gera N figuras STDFT-LRP (heatmap tempo-frequência) de clipes exemplo "
                         "e encerra, sem rodar o perfil agregado. Escolhe os clipes mais confiantes.")
    ap.add_argument("--clip-indices", type=int, nargs="+", default=None,
                    help="índices fixos de clipe (posição em audios_<split>.npy, a mesma entre "
                         "runs) para o modo STDFT; usa estes em vez de escolher por confiança. "
                         "Serve para comparar o MESMO clipe entre modelos. Ativa o modo STDFT.")
    ap.add_argument("--stdft-win", type=int, default=512,
                    help="janela da STDFT em amostras (16 kHz: 512 = 32 ms, resolução ~31 Hz)")
    ap.add_argument("--stdft-hop", type=int, default=128,
                    help="salto da STDFT em amostras (16 kHz: 128 = 8 ms)")
    ap.add_argument("--stdft-fmax", type=float, default=4000.0,
                    help="frequência máxima exibida no heatmap STDFT-LRP (Hz)")
    ap.add_argument("--eps", type=float, default=1e-9, help="estabilização de R_n/x_n na DFT")
    args = ap.parse_args(argv)

    log = get_logger()
    run_dir = Path(args.run_dir)
    paths = A.RunPaths(root=run_dir)
    cfg = load_config(paths.path("config.resolved.yaml"))
    if cfg.model.encoder != "hf_ssl":
        raise SystemExit(f"encoder={cfg.model.encoder!r} não suportado; DFT-LRP cobre só hf_ssl.")

    backend = args.backend
    ckpt_l = cfg.model.checkpoint.lower()
    # mms/xls-r são arquitetura wav2vec2 (Wav2Vec2Attention), então o patch conservativo os cobre.
    attnlrp_supported = any(k in ckpt_l for k in ("wav2vec2", "hubert", "wavlm", "mms", "xls"))
    if backend == "attnlrp" and not attnlrp_supported:
        raise SystemExit(
            f"backend=attnlrp (à mão, conservativo) cobre wav2vec2/hubert/wavlm/mms/xls-r; "
            f"checkpoint={cfg.model.checkpoint!r}. Use --backend gxi para outros modelos.")

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModel

    from brspeech_xai.encoders.hf_ssl import HFSSLEmbedder
    import joblib

    # Reusa o embedder só pelo pré-processamento (processor + _to_16k_mono); o grafo de LRP
    # usa um AutoModel próprio em modo eager (atenção explícita, exigida pela regra de atenção).
    emb = HFSSLEmbedder(checkpoint=cfg.model.checkpoint, layer=cfg.model.layer,
                        pooling=cfg.model.pooling, device=device,
                        num_samples=cfg.audio.num_samples, attn_implementation="eager")
    encoder = AutoModel.from_pretrained(cfg.model.checkpoint,
                                        attn_implementation="eager").to(device).eval()
    encoder.requires_grad_(False)                     # só queremos o grad da entrada

    if backend == "attnlrp":
        from brspeech_xai.attnlrp import patch_ssl_encoder_for_attnlrp
        counts = patch_ssl_encoder_for_attnlrp(encoder, attention=args.attn_rule,
                                               norm_stabilizer=args.norm_stab,
                                               norm_stab_scope=args.norm_stab_scope)
        log.info(f"AttnLRP patch (à mão, atenção={args.attn_rule}, "
                 f"norm_stab={args.norm_stab:g}@{args.norm_stab_scope}): {counts}")
        if min(counts.values()) == 0:
            log.warning("algum tipo de módulo não foi patcheado; o portão de conservação vai "
                        "denunciar se a relevância deixar de conservar.")

    head = joblib.load(paths.path("d_ad.joblib"))
    w, b = port_logistic_head(head)
    model = _legacy_ssl_detector_ad(encoder, cfg.model.layer, w, b).to(device).eval()

    split = "pool" if cfg.adapt.cross_fit else cfg.data.eval_split
    master = A.load_table(paths.path("master_table.parquet"))
    audios = list(np.load(paths.path(f"audios_{split}.npy"), allow_pickle=True))
    srs = [int(s) for s in np.load(paths.path(f"srs_{split}.npy"))]
    idx = stratified_idx(master, "quadrant_ad", args.per_quadrant, seed=cfg.seed)
    log.info(f"DFT-LRP [{backend}] {cfg.model.checkpoint} | split={split} | "
             f"{len(idx)} clipes | device={device}")

    # CERTIFICADO de correção (só attnlrp com regras puras): zera os vieses do encoder e confere
    # que a relevância da entrada recupera (logit - b_head) na mosca. Prova que a matemática está
    # certa; o gap com vieses (medido abaixo) é a parcela do logit que os vieses carregam, não bug.
    if backend == "attnlrp" and args.norm_stab <= 0:
        cert_wavs = [emb._to_16k_mono(audios[i], srs[i]) for i in idx[:3]]
        cert = conservation_certificate(model, emb._processor, cert_wavs, device, float(b))
        status = "OK" if cert < 1e-3 else "FALHOU"
        log.info(f"certificado de conservação (vieses zerados): resíduo relativo máx={cert:.2e} "
                 f"[{status}] (esperado ~0; independe do modelo)")

    # MODO EXEMPLOS: figura STDFT-LRP (heatmap tempo-frequência) de poucos clipes e encerra.
    # Ativa se --stdft-examples>0 OU se índices fixos foram passados (--clip-indices).
    if args.stdft_examples > 0 or args.clip_indices:
        return run_stdft_examples(model, emb, audios, srs, idx, args, cfg, paths, backend, log,
                                  master=master)

    edges = mel_band_edges(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
    # Grid fino LINEAR para o espectro de saliência (mesmos limites de frequência de H1/H2).
    sal_edges = np.linspace(cfg.bands.f_min, cfg.bands.f_max, args.salience_bins + 1)
    rel_bands, cons_dft = [], []
    sum_r, logit_mb, pred_spoof = [], [], []           # relevância de entrada; logit-b_head; predição
    salience = []                                       # distribuição de |relevância| por clipe (fina)
    from brspeech_xai.logging_utils import progress
    for i in progress(idx, desc=f"DFT-LRP {backend}", unit="clip"):
        wav16k = emb._to_16k_mono(audios[i], srs[i])
        x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
        freqs = dft_lrp.rfft_frequencies(len(x_time), 16000)
        r_freq = dft_lrp.time_to_freq_relevance(x_time, r_time, eps=args.eps)
        rel_bands.append(dft_lrp.aggregate_to_bands(r_freq, freqs, edges))
        cons_dft.append(float(dft_lrp.conservation_error(x_time, r_time, r_freq)))
        sum_r.append(float(r_time.sum()))
        logit_mb.append(logit - float(b))
        pred_spoof.append(logit > 0.0)                 # decisão do D_ad: spoof se logit de spoof > 0
        # Saliência: |relevância| por bin fino, normalizada por clipe (distribuição de atenção),
        # para clipes de logit grande não dominarem a média. Direção-agnóstico (spoof=bonafide).
        sal = dft_lrp.aggregate_to_bands(np.abs(r_freq), freqs, sal_edges)
        total = sal.sum()
        salience.append(sal / total if total > 0 else sal)
    rel_bands = np.vstack(rel_bands)                  # (n_clips, n_bands)
    salience = np.vstack(salience)                     # (n_clips, salience_bins)
    sum_r = np.asarray(sum_r)
    logit_mb = np.asarray(logit_mb)
    pred_spoof = np.asarray(pred_spoof)
    gap = logit_mb - sum_r                             # parcela do logit carregada pelos vieses

    # PORTÃO da DFT: exato sempre (a ponte tempo->frequência conserva por construção).
    log.info(f"conservação DFT (|sum_k R_k - sum_n R_n|): "
             f"máx={max(cons_dft):.2e} média={np.mean(cons_dft):.2e}")
    # COMPLETUDE (não é erro): a relevância da entrada explica a parte do logit que depende da
    # forma de onda; o resto é carregado pelos vieses internos. Reportamos a decomposição.
    frac_bias = float(np.mean(np.abs(gap))) / (float(np.mean(np.abs(logit_mb))) + 1e-12)
    log.info("completude Input×Gradient (decomposição, NÃO é erro):")
    log.info(f"  média sum R_input={sum_r.mean():+.3f} | média (logit-b_head)={logit_mb.mean():+.3f} "
             f"| média parcela dos vieses={gap.mean():+.3f}")
    log.info(f"  |vieses|/|logit-b_head| médio = {frac_bias:.1%} "
             "(o certificado acima mostra que, sem vieses, este gap → 0)")

    mean_rel = rel_bands.mean(axis=0)
    ci_low, ci_high = bootstrap_ci(rel_bands, n_boot=cfg.occlusion.n_boot, seed=cfg.seed)
    # Sufixo com κ para não sobrescrever ao comparar níveis de estabilização (κ=0 vs κ>0).
    tag = backend if args.norm_stab <= 0 else f"{backend}_ns{args.norm_stab:g}"
    import pandas as pd
    rows = [{"detector": "ad", "band_hz_low": float(edges[j]),
             "band_hz_high": float(edges[j + 1]), "mean_relevance": float(mean_rel[j]),
             "ci_low": float(ci_low[j]), "ci_high": float(ci_high[j]),
             "n": int(rel_bands.shape[0]), "backend": backend,
             "norm_stab": float(args.norm_stab)}
            for j in range(len(edges) - 1)]
    A.save_table(pd.DataFrame(rows), paths.path(f"lrp_table_{tag}.csv"))
    A.save_npy(rel_bands.astype(np.float32), paths.path(f"lrp_band_relevance_ad_{tag}.npy"))

    out_png = paths.figure(f"dft_lrp_bands_ad_{tag}.png")
    out_pdf = paths.figure(f"dft_lrp_bands_ad_{tag}.pdf")
    _plot(edges, mean_rel, ci_low, ci_high, cfg.model.checkpoint.split("/")[-1], tag,
          out_png, out_pdf)
    log.info(f"tabela salva: {paths.path(f'lrp_table_{tag}.csv')}")
    log.info(f"figura salva: {out_png}")

    # PERFIL POR CLASSE PREDITA: condiciona (spoof vs bonafide) para não cancelar sinais opostos.
    groups, class_rows = {}, []
    for label, mask in (("predicted spoof", pred_spoof), ("predicted bonafide", ~pred_spoof)):
        if mask.sum() < 2:
            log.warning(f"classe '{label}' com n={int(mask.sum())} clipes; pulando no perfil por classe.")
            continue
        sub = rel_bands[mask]
        m = sub.mean(axis=0)
        lo, hi = bootstrap_ci(sub, n_boot=cfg.occlusion.n_boot, seed=cfg.seed)
        groups[label] = (m, lo, hi, int(mask.sum()))
        class_rows += [{"detector": "ad", "pred_class": label.split()[-1],
                        "band_hz_low": float(edges[j]), "band_hz_high": float(edges[j + 1]),
                        "mean_relevance": float(m[j]), "ci_low": float(lo[j]),
                        "ci_high": float(hi[j]), "n": int(mask.sum()), "backend": backend}
                       for j in range(len(edges) - 1)]
    if groups:
        A.save_table(pd.DataFrame(class_rows), paths.path(f"lrp_table_{tag}_byclass.csv"))
        cls_png = paths.figure(f"dft_lrp_bands_ad_{tag}_byclass.png")
        cls_pdf = paths.figure(f"dft_lrp_bands_ad_{tag}_byclass.pdf")
        _plot_by_class(edges, groups, cfg.model.checkpoint.split("/")[-1], tag, cls_png, cls_pdf)
        n_sp = int(pred_spoof.sum())
        log.info(f"perfil por classe: spoof n={n_sp} | bonafide n={len(pred_spoof) - n_sp}")
        log.info(f"figura por classe salva: {cls_png}")

    # ESPECTRO DE SALIÊNCIA: onde (em frequência) o modelo presta mais atenção, em %.
    sal_centers = (sal_edges[:-1] + sal_edges[1:]) / 2
    sal_mean = salience.mean(axis=0) * 100.0
    sal_lo, sal_hi = bootstrap_ci(salience * 100.0, n_boot=cfg.occlusion.n_boot, seed=cfg.seed)
    sal_rows = [{"detector": "ad", "freq_hz_low": float(sal_edges[j]),
                 "freq_hz_high": float(sal_edges[j + 1]), "attention_share_pct": float(sal_mean[j]),
                 "ci_low": float(sal_lo[j]), "ci_high": float(sal_hi[j]),
                 "n": int(salience.shape[0]), "backend": backend}
                for j in range(len(sal_edges) - 1)]
    A.save_table(pd.DataFrame(sal_rows), paths.path(f"lrp_salience_{tag}.csv"))
    sal_png = paths.figure(f"dft_lrp_salience_ad_{tag}.png")
    sal_pdf = paths.figure(f"dft_lrp_salience_ad_{tag}.pdf")
    _plot_salience(sal_centers, sal_mean, sal_lo, sal_hi, cfg.model.checkpoint.split("/")[-1],
                   tag, sal_png, sal_pdf)
    peak = sal_centers[int(np.argmax(sal_mean))]
    log.info(f"saliência: pico de atenção em ~{peak:.0f} Hz | figura: {sal_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
