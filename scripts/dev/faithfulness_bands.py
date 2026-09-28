"""Validação de fidelidade por bandas: a explicação DFT-LRP aponta as faixas que sustentam
(keep) e derrubam (delete) a decisão mais do que faixas aleatórias?

Transforma a sonificação (ilustrativa) em evidência (verificável). Para cada clipe do split de
teste: rankeamos as 24 faixas mel pela relevância DFT-LRP orientada à classe PREVISTA, filtramos
o áudio mantendo/removendo as top-k faixas (band-pass/band-stop na STFT, invertível), e devolvemos
o áudio filtrado ao MESMO detector adaptado (D_ad). Comparamos com uma baseline aleatória de mesma
quantidade de faixas: o que vale como evidência é o GAP contra o aleatório, que isola a explicação
do artefato genérico do filtro (mesmo confound de H2).

Métricas (vocabulário da área):
  * suficiência   = p_pred quando MANTEMOS só as top-k faixas (alto => a evidência basta);
  * comprehensiveness = p_pred(orig) - p_pred(delete_k) (alto => a evidência é necessária).
Resumo AOPC: média das curvas ao longo de k, e o gap DFT-LRP menos aleatório.

Incerteza: IC 95% por bootstrap SOBRE OS CLIPES (a unidade de variação é o clipe, não a banda).

Fase 1: um encoder (wav2vec2), todos os clipes de teste. Extensão a hubert/wavlm reusa o mesmo
código via --encoder.

Uso (dentro do container, onde há GPU e os áudios crus):
    python /workspace/scripts/dev/faithfulness_bands.py \
        --results-root /workspace/results --encoder wav2vec2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from brspeech_xai import dft_lrp

# Imports pesados (torch, transformers, artifacts) são adiados para dentro de ``main``/``score_waves``
# para que as funções puras do filtro (testadas em CI local sem GPU) importem só numpy.

_KS_DEFAULT = (1, 2, 3, 4, 6, 8, 12)


def band_of_bin(freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Índice de banda (0..n_bands-1) de cada bin de frequência.

    Toda frequência é atribuída a exatamente uma banda: bins abaixo da 1ª borda caem na banda 0
    e acima da última borda caem na última banda. Assim keep(todas as bandas) = manter tudo
    (identidade) e delete(todas) = zerar tudo, o que torna o filtro exatamente invertível.
    No interior a partição coincide com ``dft_lrp.aggregate_to_bands`` (semiaberta [lo, hi)).
    """
    n_bands = len(edges) - 1
    return np.clip(np.digitize(freqs, edges) - 1, 0, n_bands - 1)


def band_mask_bins(freqs: np.ndarray, edges: np.ndarray, keep_idx) -> np.ndarray:
    """Máscara binária por bin (1 nas bandas de ``keep_idx``, 0 fora), constante no tempo."""
    bob = band_of_bin(freqs, edges)
    keep = np.zeros(freqs.shape[0], dtype=np.float64)
    keep_set = {int(i) for i in keep_idx}
    for b in keep_set:
        keep[bob == b] = 1.0
    return keep


def reconstruct_with_band_mask(x_time: np.ndarray, mask_bins: np.ndarray,
                               win: int, hop: int) -> np.ndarray:
    """STFT inversa do clipe mantendo só os bins de ``mask_bins`` (band-pass/stop invertível).

    Replica EXATAMENTE a análise de ``dft_lrp.stdft_lrp`` (pad win//2, Hann, WOLA por ``wsum``).
    Com máscara 1 recupera o sinal original; com máscara 0 devolve zeros. A máscara é real e em
    {0,1}, então escala a magnitude preservando a fase.
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
    y = np.zeros(npad, dtype=np.float64)
    for s in starts:
        sl = slice(s, s + win)
        c = np.fft.rfft(xp[sl] * w)
        y[sl] += np.fft.irfft(c * mask_bins, n=win)
    y = y / np.where(wsum > 0, wsum, 1.0)
    return y[pad:pad + n]


def rank_bands_for_clip(r_tf: np.ndarray, freqs: np.ndarray, edges: np.ndarray,
                        pred_spoof: bool) -> np.ndarray:
    """Ordena as bandas (desc) pela relevância DFT-LRP orientada à classe prevista.

    Marginaliza a relevância no tempo, agrega por banda e orienta pelo sinal da classe prevista
    (spoof usa +R; bonafide usa -R). Retorna os índices de banda do mais para o menos relevante.
    """
    r_freq = r_tf.sum(axis=0)                       # (F,) marginal temporal
    r_band = dft_lrp.aggregate_to_bands(r_freq, freqs, edges)   # (n_bands,)
    oriented = r_band if pred_spoof else -r_band
    return np.argsort(-oriented)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                  seed: int = 42) -> tuple[float, float]:
    """IC percentil por bootstrap da MÉDIA de um vetor 1D (reamostra clipes com reposição)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        m = float(v.mean()) if v.size else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def score_waves(model, waves: list[np.ndarray], device: str,
                batch_size: int = 16) -> np.ndarray:
    """Logit de spoof do D_ad para uma lista de formas de onda já normalizadas (feed direto)."""
    import torch

    logits: list[float] = []
    with torch.no_grad():
        for i in range(0, len(waves), batch_size):
            chunk = waves[i:i + batch_size]
            t = torch.as_tensor(np.stack(chunk), dtype=torch.float32, device=device)
            logits.extend(model(t).detach().cpu().numpy().ravel().tolist())
    return np.asarray(logits, dtype=np.float64)


def process_clip(model, r_tf, x_time, logit, freqs, edges, win, hop, ks, n_random,
                 rng, device) -> list[tuple[str, int, float]]:
    """Re-scoring keep/delete/random por k para um clipe. Devolve (condition, k, p_pred)."""
    n_bands = len(edges) - 1
    pred_spoof = logit > 0
    ranking = rank_bands_for_clip(r_tf, freqs, edges, pred_spoof)
    all_bands = np.arange(n_bands)

    # Monta todas as variantes do clipe e pontua num único lote (rótulo -> índice).
    waves: list[np.ndarray] = []
    labels: list[tuple[str, int, int]] = []   # (condition, k, draw)
    for k in ks:
        top = ranking[:k]
        waves.append(reconstruct_with_band_mask(x_time, band_mask_bins(freqs, edges, top), win, hop))
        labels.append(("keep_dftlrp", k, 0))
        keep_del = np.setdiff1d(all_bands, top, assume_unique=False)
        waves.append(reconstruct_with_band_mask(x_time, band_mask_bins(freqs, edges, keep_del), win, hop))
        labels.append(("delete_dftlrp", k, 0))
        for d in range(n_random):
            rand = rng.choice(n_bands, size=k, replace=False)
            waves.append(reconstruct_with_band_mask(x_time, band_mask_bins(freqs, edges, rand), win, hop))
            labels.append(("keep_random", k, d))
            rand_del = np.setdiff1d(all_bands, rand, assume_unique=False)
            waves.append(reconstruct_with_band_mask(x_time, band_mask_bins(freqs, edges, rand_del), win, hop))
            labels.append(("delete_random", k, d))

    logits = score_waves(model, waves, device)
    p = _sigmoid(logits) if pred_spoof else _sigmoid(-logits)   # p da classe PREVISTA

    # Reduz: média sobre draws para as condições aleatórias.
    acc: dict[tuple[str, int], list[float]] = {}
    for (cond, k, _d), pv in zip(labels, p):
        acc.setdefault((cond, k), []).append(float(pv))
    return [(cond, k, float(np.mean(v))) for (cond, k), v in acc.items()]


def _aggregate(rows, ks, classes):
    """Média + IC 95% (bootstrap sobre clipes) de p_pred por (class, condition, k)."""
    out = {}
    for cls in classes:
        for cond in ("keep_dftlrp", "keep_random", "delete_dftlrp", "delete_random"):
            for k in ks:
                vals = np.asarray([r["p_pred"] for r in rows
                                   if r["true_class"] == cls and r["condition"] == cond
                                   and r["k"] == k], dtype=np.float64)
                if vals.size == 0:
                    continue
                lo, hi = _bootstrap_ci(vals)
                out[(cls, cond, k)] = (float(vals.mean()), lo, hi, int(vals.size))
    return out


def _plot(agg, ks, classes, p_orig, out_png, out_pdf):
    """Curvas de fidelidade: p_pred vs nº de faixas, por classe. keep (verde), delete (vermelho);
    linha cheia = DFT-LRP, tracejada = aleatório. O gap cheio-tracejado é a evidência."""
    from brspeech_xai.plotting import set_plot_style
    import matplotlib.pyplot as plt

    set_plot_style()
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 4.0), squeeze=False)
    # cor = operação (keep verde / delete vermelho); traço+marcador = fonte (DFT-LRP cheia·o /
    # aleatório tracejada·x), para as quatro entradas ficarem distinguíveis na legenda.
    style = {
        "keep_dftlrp": ("#2e7d32", "-", "o", "keep top-k (DFT-LRP)"),
        "keep_random": ("#2e7d32", "--", "x", "keep k (random)"),
        "delete_dftlrp": ("#c0392b", "-", "o", "delete top-k (DFT-LRP)"),
        "delete_random": ("#c0392b", "--", "x", "delete k (random)"),
    }
    kx = np.asarray(ks, dtype=float)
    for j, cls in enumerate(classes):
        ax = axes[0][j]
        for cond, (color, ls, marker, label) in style.items():
            pts = [(k,) + agg[(cls, cond, k)][:3] for k in ks if (cls, cond, k) in agg]
            if not pts:
                continue
            kk = np.asarray([p[0] for p in pts], dtype=float)
            m = np.asarray([p[1] for p in pts])
            lo = np.asarray([p[2] for p in pts])
            hi = np.asarray([p[3] for p in pts])
            ax.plot(kk, m, color=color, ls=ls, marker=marker, ms=4, label=label)
            ax.fill_between(kk, lo, hi, color=color, alpha=0.12)
        if cls in p_orig:
            ax.axhline(p_orig[cls], color="0.4", ls=":", lw=1.0,
                       label=f"original (p={p_orig[cls]:.2f})")
        ax.set_title(f"true: {cls}", fontsize=10)
        ax.set_xlabel("number of bands (k of 24)")
        ax.set_ylabel("p(predicted class)")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(kx)
        if j == 0:
            ax.legend(fontsize=7, loc="best", handlelength=3.0)
    fig.suptitle("Faithfulness by frequency bands (keep / delete vs random)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def _read_curves_csv(path: Path):
    """Recarrega o agg (class, condition, k) -> (mean, lo, hi, n) do CSV de curvas, para replotar."""
    agg = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            agg[(r["true_class"], r["condition"], int(r["k"]))] = (
                float(r["p_pred_mean"]), float(r["ci_low"]), float(r["ci_high"]), int(r["n_clips"]))
    return agg


def _write_curves_csv(path: Path, agg, ks, classes):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "condition", "true_class", "p_pred_mean", "ci_low", "ci_high", "n_clips"])
        for cls in classes:
            for cond in ("keep_dftlrp", "keep_random", "delete_dftlrp", "delete_random"):
                for k in ks:
                    if (cls, cond, k) not in agg:
                        continue
                    m, lo, hi, n = agg[(cls, cond, k)]
                    wr.writerow([k, cond, cls, round(m, 5), round(lo, 5), round(hi, 5), n])


def _write_aopc_csv(path: Path, rows, ks, classes, p_orig_per_clip):
    """AOPC por clipe (média nas curvas) e IC 95% por bootstrap sobre clipes.

    sufficiency        = média_k p_pred(keep_k)          (DFT-LRP e random)
    comprehensiveness  = média_k [p_orig - p_pred(delete_k)] (DFT-LRP e random)
    """
    by_clip: dict[int, dict[tuple[str, int], float]] = {}
    cls_of: dict[int, str] = {}
    for r in rows:
        by_clip.setdefault(r["index"], {})[(r["condition"], r["k"])] = r["p_pred"]
        cls_of[r["index"]] = r["true_class"]

    def aopc(idx, cond, kind):
        vals = []
        for k in ks:
            if (cond, k) not in by_clip[idx]:
                return None
            pk = by_clip[idx][(cond, k)]
            vals.append(pk if kind == "suff" else (p_orig_per_clip[idx] - pk))
        return float(np.mean(vals))

    specs = [("sufficiency", "keep_dftlrp", "suff"), ("sufficiency", "keep_random", "suff"),
             ("comprehensiveness", "delete_dftlrp", "comp"),
             ("comprehensiveness", "delete_random", "comp")]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "condition", "true_class", "aopc", "ci_low", "ci_high", "n_clips"])
        for cls in classes:
            idxs = [i for i in by_clip if cls_of[i] == cls]
            for metric, cond, kind in specs:
                vals = np.asarray([aopc(i, cond, kind) for i in idxs
                                   if aopc(i, cond, kind) is not None], dtype=np.float64)
                if vals.size == 0:
                    continue
                lo, hi = _bootstrap_ci(vals)
                wr.writerow([metric, cond, cls, round(float(vals.mean()), 5),
                             round(lo, 5), round(hi, 5), int(vals.size)])


def select_stratified_indices(master, per_class: int, seed: int = 42) -> list[int]:
    """~per_class clipes por classe, estratificados pelos quadrantes de D_ad (quadrant_ad).

    Cada classe tem dois quadrantes (spoof: TP/FN; bonafide: TN/FP). Aloca a cota da classe entre
    eles incluindo primeiro os raros (FN/FP) por inteiro e completando com o comum, para que os
    erros (essenciais para a leitura) nunca fiquem de fora.
    """
    rng = np.random.default_rng(seed)
    gt = master["ground_truth"].astype(int).to_numpy()
    quad = master["quadrant_ad"].to_numpy()
    chosen: list[int] = []
    class_quads = {"spoof": (1, ["FN", "TP"]), "bonafide": (0, ["FP", "TN"])}
    for _cls, (lab, quads) in class_quads.items():
        pools = {q: np.where((gt == lab) & (quad == q))[0] for q in quads}
        order = sorted(quads, key=lambda q: len(pools[q]))   # raros primeiro
        remaining, slots = per_class, len(order)
        for j, q in enumerate(order):
            share = remaining // (slots - j)
            t = min(len(pools[q]), share)
            pool = pools[q]
            sel = pool if t >= len(pool) else rng.choice(pool, t, replace=False)
            chosen.extend(int(x) for x in sel)
            remaining -= len(sel)
    return sorted(chosen)


def _bands_hz(edges: np.ndarray, idx) -> str:
    """Rótulo curto das faixas (kHz) para o manifesto de áudio, ordenadas por frequência."""
    idx = sorted(int(i) for i in idx)
    return "; ".join(f"{edges[i] / 1000:.2f}-{edges[i + 1] / 1000:.2f}" for i in idx)


def export_audio_clips(model, emb, relevance_for_clip, audios, srs, gt, clip_idx, audio_k,
                       freqs, edges, win, hop, eps, device, out_dir, write_wav, log, seed=0):
    """Grava, para poucos clipes, o áudio original e as intervenções por banda (top-k e aleatório).

    Mesma operação que o experimento pontua (band-pass/stop invertível), para o ouvido conferir o
    que a métrica mede. Também re-pontua cada versão no D_ad e registra p(classe prevista) no
    manifesto, para casar percepção com número.
    """
    rng = np.random.default_rng(seed)
    n_bands = len(edges) - 1
    all_bands = np.arange(n_bands)
    rows = []
    for i in clip_idx:
        wav16k = emb._to_16k_mono(audios[i], srs[i])
        x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
        _t, _f, r_tf, _s = dft_lrp.stdft_lrp(x_time, r_time, 16000, win_size=win, hop=hop, eps=eps)
        pred_spoof = logit > 0
        true_cls = "spoof" if gt[i] == 1 else "bonafide"
        pred_cls = "spoof" if pred_spoof else "bonafide"
        ranking = rank_bands_for_clip(r_tf, freqs, edges, pred_spoof)
        top = ranking[:audio_k]
        rand = rng.choice(n_bands, size=audio_k, replace=False)
        variants = {
            "original": None,
            f"keep_top{audio_k}": top,
            f"delete_top{audio_k}": np.setdiff1d(all_bands, top),
            f"keep_rand{audio_k}": rand,
            f"delete_rand{audio_k}": np.setdiff1d(all_bands, rand),
        }
        clip_tag = f"{i:05d}_{true_cls}"
        d = out_dir / clip_tag
        waves, names = [], []
        for name, keep in variants.items():
            y = x_time if keep is None else reconstruct_with_band_mask(
                x_time, band_mask_bins(freqs, edges, keep), win, hop)
            write_wav(d / f"{name}.wav", y)
            waves.append(np.asarray(y, dtype=np.float64))
            names.append(name)
        p = _sigmoid(score_waves(model, waves, device) * (1 if pred_spoof else -1))
        for name, pv in zip(names, p):
            keep = variants[name]
            rows.append({"clip": clip_tag, "index": i, "true_class": true_cls, "pred": pred_cls,
                         "variant": name, "p_pred": round(float(pv), 4),
                         "kept_bands_khz": "all" if keep is None else _bands_hz(edges, keep)})
        log.info(f"[{clip_tag}] pred={pred_cls} p_orig={float(_sigmoid(logit if pred_spoof else -logit)):.2f} "
                 f"| top{audio_k} bands (kHz): {_bands_hz(edges, top)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    log.info(f"áudios de fidelidade salvos em: {out_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="faithfulness_bands")
    ap.add_argument("--results-root", default="/workspace/results")
    ap.add_argument("--encoder", default="wav2vec2",
                    help="substring do slug do encoder (default: wav2vec2)")
    ap.add_argument("--ks", type=int, nargs="+", default=list(_KS_DEFAULT))
    ap.add_argument("--n-random", type=int, default=5, help="sorteios da baseline aleatória por k")
    ap.add_argument("--stdft-win", type=int, default=512)
    ap.add_argument("--stdft-hop", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = todos os clipes do split")
    ap.add_argument("--per-class", type=int, default=0,
                    help=">0: amostra ~N clipes por classe estratificados por quadrante "
                         "(inclui FN/FP raros). Substitui --max-clips.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="dir de saída (default: <root>/_aggregate/faithfulness)")
    ap.add_argument("--audio-only", action="store_true",
                    help="não roda a métrica; só grava .wav de poucos clipes para ouvir")
    ap.add_argument("--audio-clips", type=int, nargs="+", default=[187, 1016],
                    help="índices de clipe para exportar áudio (default: 187 spoof, 1016 bonafide)")
    ap.add_argument("--audio-k", type=int, default=6, help="nº de faixas top-k na exportação de áudio")
    ap.add_argument("--replot", action="store_true",
                    help="só refaz a figura a partir do CSV existente (recomputa a linha 'original')")
    args = ap.parse_args(argv)

    import torch

    from brspeech_xai import artifacts as A
    from brspeech_xai.bands import mel_band_edges
    from brspeech_xai.config import load_config
    from brspeech_xai.logging_utils import get_logger, progress
    from dft_lrp_ad import relevance_for_clip
    from stdft_lrp_compare import (_build_lrp_model, _discover_runs, _order_slugs, _split_of)

    log = get_logger()
    results_root = Path(args.results_root)
    runs = _discover_runs(results_root)
    if not runs:
        raise SystemExit(f"nenhuma run hf_ssl com d_ad.joblib em {results_root}")
    slugs = _order_slugs(runs)
    match = [s for s in slugs if args.encoder in s]
    if not match:
        raise SystemExit(f"encoder {args.encoder!r} não encontrado entre {slugs}")
    slug = match[0]
    run_dir = runs[slug]
    short = slug.replace("hf_ssl-", "")
    log.info(f"encoder: {short} | run: {run_dir.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(run_dir / "config.resolved.yaml")
    split = _split_of(cfg)
    emb, model, conservative = _build_lrp_model(cfg, run_dir, device)
    if not conservative:
        log.warning(f"{short} não é conservativo; a relevância cai no Gradient×Input (referência).")
    edges = mel_band_edges(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
    freqs = dft_lrp.rfft_frequencies(args.stdft_win, 16000)

    master = A.load_table(run_dir / "master_table.parquet")
    gt = master["ground_truth"].astype(int).to_numpy()
    audios = list(np.load(run_dir / f"audios_{split}.npy", allow_pickle=True))
    srs = [int(s) for s in np.load(run_dir / f"srs_{split}.npy")]

    out_dir = Path(args.out) if args.out else results_root / "_aggregate" / "faithfulness"
    if args.audio_only:
        from sonify_lrp import _write_wav
        # Separa por encoder para permitir comparar o mesmo clipe entre modelos sem sobrescrever.
        audio_dir = out_dir / "audio" / short
        log.info(f"modo áudio: clipes {args.audio_clips} | top-{args.audio_k} de {cfg.bands.n_bands}")
        export_audio_clips(model, emb, relevance_for_clip, audios, srs, gt, args.audio_clips,
                           args.audio_k, freqs, edges, args.stdft_win, args.stdft_hop, args.eps,
                           device, audio_dir, _write_wav, log, seed=args.seed)
        return 0

    classes = ["spoof", "bonafide"]
    if args.replot:
        agg = _read_curves_csv(out_dir / f"faithfulness_bands_{short}.csv")
        ks = sorted({k for (_c, _cond, k) in agg})
        idxs = select_stratified_indices(master, args.per_class, args.seed) if args.per_class > 0 \
            else list(range(len(audios)))
        p_by_cls = {c: [] for c in classes}
        for i in progress(idxs, desc=f"p_orig {short}", unit="clip"):
            wav16k = emb._to_16k_mono(audios[i], srs[i])
            _x, _r, logit = relevance_for_clip(model, emb._processor, wav16k, device)
            cls = "spoof" if gt[i] == 1 else "bonafide"
            p_by_cls[cls].append(float(_sigmoid(logit if logit > 0 else -logit)))
        p_orig = {c: float(np.mean(v)) for c, v in p_by_cls.items() if v}
        _plot(agg, ks, classes, p_orig,
              out_dir / f"faithfulness_bands_{short}.png",
              out_dir / f"faithfulness_bands_{short}.pdf")
        log.info(f"figura replotada em: {out_dir}")
        return 0

    if args.per_class > 0:
        clip_indices = select_stratified_indices(master, args.per_class, args.seed)
        how = f"estratificado ~{args.per_class}/classe"
    else:
        n_clips = len(audios) if args.max_clips <= 0 else min(args.max_clips, len(audios))
        clip_indices = list(range(n_clips))
        how = "todos" if args.max_clips <= 0 else f"sequencial {n_clips}"
    log.info(f"clipes: {len(clip_indices)} ({how}, split={split}) | bandas={cfg.bands.n_bands} "
             f"| ks={args.ks} | n_random={args.n_random}")

    rng = np.random.default_rng(args.seed)
    rows = []
    p_orig_per_clip: dict[int, float] = {}
    for i in progress(clip_indices, desc=f"faithfulness {short}", unit="clip"):
        wav16k = emb._to_16k_mono(audios[i], srs[i])
        x_time, r_time, logit = relevance_for_clip(model, emb._processor, wav16k, device)
        _t, _f, r_tf, _s = dft_lrp.stdft_lrp(x_time, r_time, 16000, win_size=args.stdft_win,
                                             hop=args.stdft_hop, eps=args.eps)
        pred_spoof = logit > 0
        true_cls = "spoof" if gt[i] == 1 else "bonafide"
        p_orig_per_clip[i] = float(_sigmoid(logit if pred_spoof else -logit))
        for cond, k, pv in process_clip(model, r_tf, x_time, logit, freqs, edges,
                                        args.stdft_win, args.stdft_hop, args.ks,
                                        args.n_random, rng, device):
            rows.append({"index": i, "true_class": true_cls,
                         "pred_class": "spoof" if pred_spoof else "bonafide",
                         "condition": cond, "k": k, "p_pred": pv})

    agg = _aggregate(rows, args.ks, classes)
    p_orig = {cls: float(np.mean([p_orig_per_clip[i] for i in p_orig_per_clip
                                  if (("spoof" if gt[i] == 1 else "bonafide") == cls)]))
              for cls in classes}

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(agg, args.ks, classes, p_orig,
          out_dir / f"faithfulness_bands_{short}.png",
          out_dir / f"faithfulness_bands_{short}.pdf")
    _write_curves_csv(out_dir / f"faithfulness_bands_{short}.csv", agg, args.ks, classes)
    _write_aopc_csv(out_dir / f"faithfulness_aopc_{short}.csv", rows, args.ks, classes,
                    p_orig_per_clip)
    log.info(f"fidelidade salva em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
