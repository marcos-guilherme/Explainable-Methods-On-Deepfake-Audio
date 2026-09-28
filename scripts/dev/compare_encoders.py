"""Agrega os resultados de várias runs (uma por encoder) em tabela + figuras comparativas.

Cada encoder roda em uma run separada (o `config_hash` inclui `model.encoder`, então os
artefatos ficam isolados por diretório). A comparação é sempre sobre o detector ADAPTADO
(`ad`): os encoders só-extratores (hf_ssl) não têm zero-shot, então usar o `ad` do XLS-R
também mantém a comparação justa (mesmo tipo de detector). Como todos compartilham a mesma
grade de bandas e a mesma feature (energia log-mel), as pistas espectrais vivem no mesmo
eixo de Hz e podem ser sobrepostas diretamente.

Por padrão o `mms-300m` é excluído de TODAS as figuras (``--exclude``), por ora. Para readicioná-lo
use ``--exclude`` sem valor; ``--peer-exclude`` permite tirar labels só das figuras peer.

Lê de cada run: `performance_table.csv`, `occlusion_table.csv`, `spearman_table.csv`,
`convergence_intervention.csv` e `config.resolved.yaml`. Gera, em `_aggregate/`:
  - `encoder_comparison.csv`: EER/MCC/acc do D_ad (e EER do D_zs se houver).
  - `encoder_comparison_eer.{pdf,png}`: barras de EER do D_ad (desempenho; menor = melhor).
  - `encoder_occlusion_overlay.{pdf,png}` (H2): perfil causal por banda (queda de P(spoof)
    ao ocluir), com IC 95%, uma linha por encoder no mesmo eixo de Hz.
  - `encoder_assoc_heatmap.{pdf,png}` (H1): mapa encoder × banda do ρ de Spearman com sinal
    (associação com spoof); mostra em que faixas os encoders concordam/divergem.
  - `encoder_convergence_forest.{pdf,png}` (H1->H2): por encoder, a diferença de queda ao
    ocluir bandas top-assoc vs bottom-assoc (IC 95%); positivo = a associação prevê o efeito
    causal (as duas análises convergem).
  - `encoder_lrp_overlay.{pdf,png}` (terceira lente, DFT-LRP): relevância por banda que o D_ad
    atribui a spoof, uma linha por encoder; cheia = backend conservativo (attnlrp), tracejada =
    referência gxi. Só encoders hf_ssl (o XLS-R fairseq fica fora desta lente).
  - `encoder_lrp_overlay_byclass.{pdf,png}`: a mesma relevância DFT-LRP condicionada à classe
    PREDITA (dois painéis: predito spoof / predito bonafide), uma linha por encoder.
  - `encoder_lrp_salience_overlay.{pdf,png}`: saliência DFT-LRP (fração da |relevância| por
    frequência, direção-agnóstica) sobreposta, uma cor por encoder.
  - `encoder_lrp_correlation.{pdf,png}` + `.csv`: concordância cruzada (Spearman) entre encoders
    do perfil de relevância DFT-LRP por banda (com sinal). Matriz par a par do ranking de quais
    bandas pesam; baixo = as arquiteturas discordam sobre quais frequências importam.
  - `encoder_lrp_top_bands.{pdf,png}` + `.csv`: tabela das bandas mais importantes (Top-K por
    |relevância| DFT-LRP) por encoder, com a linha Top 1 destacada; mostra que cada modelo
    elege uma faixa diferente como a mais importante.
  - `encoder_pred_agreement.{pdf,png}`: matriz de Spearman do P(spoof) do D_ad entre encoders,
    clipe a clipe (só possível porque as runs compartilham os mesmos clipes de teste).

Sem pandas de propósito (só csv/numpy/matplotlib/yaml), para rodar em qualquer ambiente:
    # explícito (uma pasta de run por encoder)
    python scripts/dev/compare_encoders.py \
        --run-dirs results/xlsr_fairseq/default-A results/hf_ssl-hubert-base-ls960/hubert-B
    # automático: varre results/, pega a run mais recente de cada pasta de encoder
    python scripts/dev/compare_encoders.py --results-root results
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ANALYSIS_DETECTOR = "ad"  # comparação sempre no detector adaptado (ver docstring)
AGGREGATE_DIR = "_aggregate"


def _read_csv(path: Path) -> list[dict]:
    """Lê um CSV como lista de dicts (uma linha por registro)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _encoder_id(run: Path) -> tuple[str, str]:
    """(encoder, checkpoint) a partir do config.resolved.yaml da run."""
    cfg = yaml.safe_load((run / "config.resolved.yaml").read_text()) or {}
    model = cfg.get("model", {}) or {}
    return str(model.get("encoder", "?")), str(model.get("checkpoint", "?"))


def _label(encoder: str, checkpoint: str) -> str:
    """Rótulo curto e legível para a figura (em inglês, p/ o paper)."""
    if encoder == "xlsr_fairseq":
        return "XLS-R"
    if encoder == "hf_ssl":
        return checkpoint.split("/")[-1]  # ex.: hubert-base-ls960
    return encoder


def discover_runs(results_root: str) -> list[str]:
    """Varre results/ e devolve a run mais recente de cada pasta de encoder.

    Estrutura esperada (layout atual): results/<encoder>/<run_id>/. Ignora a pasta de
    agregado (prefixo '_') e runs 'flat' legadas na raiz de results/ (sem model.encoder no
    config, viram label '?'). Dentro de cada encoder, prefere runs 'default' a 'smoke'.
    """
    root = Path(results_root)
    runs = []
    for enc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if enc_dir.name.startswith("_"):
            continue
        sub = [d for d in enc_dir.iterdir()
               if d.is_dir() and (d / "performance_table.csv").exists()]
        if not sub:
            continue
        pool = [d for d in sub if not d.name.startswith("smoke")] or sub
        runs.append(str(sorted(pool)[-1]))                 # mais recente por nome (timestamp)
    return runs


def collect(run_dirs: list[str]) -> list[dict]:
    """Uma linha por encoder com EER/MCC/acc do D_ad (e EER do D_zs se houver)."""
    rows = []
    for rd in run_dirs:
        run = Path(rd)
        perf = {r["detector"]: r for r in _read_csv(run / "performance_table.csv")}
        encoder, checkpoint = _encoder_id(run)
        ad = perf["ad"]
        rows.append({"run_dir": run.name, "encoder": encoder, "checkpoint": checkpoint,
                     "label": _label(encoder, checkpoint),
                     "eer_ad": float(ad["eer"]), "mcc_ad": float(ad["mcc"]),
                     "accuracy_ad": float(ad["accuracy"]),
                     "eer_zs": float(perf["zs"]["eer"]) if "zs" in perf else float("nan")})
    rows.sort(key=lambda r: r["eer_ad"])
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_eer(rows: list[dict], out_dir: Path) -> None:
    """Barras horizontais de EER do D_ad por encoder (menor = melhor)."""
    fig, ax = plt.subplots(figsize=(5.0, 0.5 + 0.5 * len(rows)))
    y = range(len(rows))
    eer = np.array([r["eer_ad"] * 100 for r in rows])
    ax.barh(list(y), eer, color="#D55E00", alpha=0.85)
    ax.set_yticks(list(y), [r["label"] for r in rows])
    ax.invert_yaxis()  # melhor (menor EER) no topo
    ax.set_xlabel("Adapted-head EER (%)")
    for i, v in enumerate(eer):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_comparison_eer.pdf")
    fig.savefig(out_dir / "encoder_comparison_eer.png", dpi=300)
    plt.close(fig)


def _occlusion_ad(run: Path) -> dict | None:
    """Perfil causal (H2) do D_ad: queda de P(spoof) por banda + IC, ordenado por Hz."""
    f = run / "occlusion_table.csv"
    if not f.exists():
        return None
    rows = [r for r in _read_csv(f) if r["detector"] == ANALYSIS_DETECTOR]
    if not rows:
        return None
    rows.sort(key=lambda r: float(r["band_hz_low"]))
    lo = np.array([float(r["band_hz_low"]) for r in rows])
    hi = np.array([float(r["band_hz_high"]) for r in rows])
    return {"center": (lo + hi) / 2.0,
            "mean": np.array([float(r["mean_p_spoof_drop"]) for r in rows]),
            "ci_low": np.array([float(r["ci_low"]) for r in rows]),
            "ci_high": np.array([float(r["ci_high"]) for r in rows])}


def _signed_assoc_by_band(run: Path) -> dict | None:
    """ρ de Spearman COM SINAL por banda (H1) do D_ad: |ρ| máximo sobre μ/σ e classes.

    Para cada banda, escolhe entre as features band{n}_mean/std e as classes o ρ de maior
    módulo, preservando o sinal (espelha o perfil de associação com sinal de cada run).
    """
    f = run / "spearman_table.csv"
    if not f.exists():
        return None
    best: dict[int, tuple[float, float]] = {}   # banda -> (|rho|, rho com sinal)
    for r in _read_csv(f):
        if r["detector"] != ANALYSIS_DETECTOR:
            continue
        m = re.match(r"band(\d+)_", r["feature"])
        if not m:
            continue
        band, rho = int(m.group(1)), float(r["rho"])
        if band not in best or abs(rho) > best[band][0]:
            best[band] = (abs(rho), rho)
    return {b: v[1] for b, v in best.items()} or None


def _convergence_ad(run: Path) -> dict | None:
    """Linha do teste de convergência (H1->H2) do D_ad."""
    f = run / "convergence_intervention.csv"
    if not f.exists():
        return None
    rows = [r for r in _read_csv(f) if r["detector"] == ANALYSIS_DETECTOR]
    return rows[0] if rows else None


def _lrp_ad(run: Path) -> dict | None:
    """Terceira lente (DFT-LRP) do D_ad: relevância por banda + IC, ordenada por Hz.

    Lê ``lrp_table_<tag>.csv`` (gerada por ``dft_lrp_ad.py``). Prefere o backend conservativo
    (``attnlrp``, wav2vec2/hubert/wavlm) ao de referência (``gxi``). Ignora variantes de
    diagnóstico (κ>0, sufixo ``_ns...``), que alteram a explicação de propósito.
    """
    cands = [c for c in sorted(run.glob("lrp_table_*.csv"))
             if "_ns" not in c.stem.split("lrp_table_")[-1] and "_byclass" not in c.stem]
    if not cands:
        return None
    cands.sort(key=lambda c: 0 if c.stem.endswith("attnlrp") else 1)  # conservativo antes
    rows = [r for r in _read_csv(cands[0]) if r["detector"] == ANALYSIS_DETECTOR]
    if not rows:
        return None
    rows.sort(key=lambda r: float(r["band_hz_low"]))
    lo = np.array([float(r["band_hz_low"]) for r in rows])
    hi = np.array([float(r["band_hz_high"]) for r in rows])
    return {"center": (lo + hi) / 2.0,
            "mean": np.array([float(r["mean_relevance"]) for r in rows]),
            "ci_low": np.array([float(r["ci_low"]) for r in rows]),
            "ci_high": np.array([float(r["ci_high"]) for r in rows]),
            "backend": rows[0].get("backend", "?")}


def _lrp_byclass_ad(run: Path) -> dict | None:
    """Relevância DFT-LRP por banda condicionada à CLASSE PREDITA (spoof/bonafide) do D_ad.

    Lê ``lrp_table_<tag>_byclass.csv``. Prefere o backend conservativo (attnlrp). Devolve
    ``{"backend": tag, "classes": {classe -> {center, mean, ci_low, ci_high}}}``.
    """
    cands = [c for c in sorted(run.glob("lrp_table_*_byclass.csv"))
             if "_ns" not in c.stem]
    if not cands:
        return None
    cands.sort(key=lambda c: 0 if "attnlrp" in c.stem else 1)  # conservativo antes
    rows = [r for r in _read_csv(cands[0]) if r["detector"] == ANALYSIS_DETECTOR]
    if not rows:
        return None
    classes: dict[str, dict] = {}
    for cls in ("spoof", "bonafide"):
        sub = [r for r in rows if r.get("pred_class") == cls]
        if not sub:
            continue
        sub.sort(key=lambda r: float(r["band_hz_low"]))
        lo = np.array([float(r["band_hz_low"]) for r in sub])
        hi = np.array([float(r["band_hz_high"]) for r in sub])
        classes[cls] = {"center": (lo + hi) / 2.0,
                        "mean": np.array([float(r["mean_relevance"]) for r in sub]),
                        "ci_low": np.array([float(r["ci_low"]) for r in sub]),
                        "ci_high": np.array([float(r["ci_high"]) for r in sub])}
    if not classes:
        return None
    return {"backend": rows[0].get("backend", "?"), "classes": classes}


def _salience_ad(run: Path) -> dict | None:
    """Espectro de saliência DFT-LRP do D_ad: fração da atenção (|relevância|) por bin fino.

    Lê ``lrp_salience_<tag>.csv``. Prefere o backend conservativo (attnlrp). Direção-agnóstico:
    mostra ONDE o modelo olha, não para que lado.
    """
    cands = [c for c in sorted(run.glob("lrp_salience_*.csv")) if "_ns" not in c.stem]
    if not cands:
        return None
    cands.sort(key=lambda c: 0 if c.stem.endswith("attnlrp") else 1)  # conservativo antes
    rows = [r for r in _read_csv(cands[0]) if r["detector"] == ANALYSIS_DETECTOR]
    if not rows:
        return None
    rows.sort(key=lambda r: float(r["freq_hz_low"]))
    lo = np.array([float(r["freq_hz_low"]) for r in rows])
    hi = np.array([float(r["freq_hz_high"]) for r in rows])
    share = np.array([float(r["attention_share_pct"]) for r in rows])
    center = (lo + hi) / 2.0
    return {"center": center, "share": share,
            "ci_low": np.array([float(r["ci_low"]) for r in rows]),
            "ci_high": np.array([float(r["ci_high"]) for r in rows]),
            "backend": rows[0].get("backend", "?"),
            "peak_hz": float(center[int(np.argmax(share))])}


def collect_profiles(run_dirs: list[str], order: list[str]) -> list[dict]:
    """Perfis por encoder (occ/assoc/convergência/lrp), ordenados por ``order`` (labels)."""
    profs = []
    for rd in run_dirs:
        run = Path(rd)
        encoder, checkpoint = _encoder_id(run)
        profs.append({"label": _label(encoder, checkpoint),
                      "occ": _occlusion_ad(run), "assoc": _signed_assoc_by_band(run),
                      "conv": _convergence_ad(run), "lrp": _lrp_ad(run),
                      "lrp_byclass": _lrp_byclass_ad(run), "salience": _salience_ad(run)})
    rank = {lbl: i for i, lbl in enumerate(order)}
    return sorted(profs, key=lambda p: rank.get(p["label"], len(rank)))


def plot_occlusion_overlay(profs: list[dict], out_dir: Path) -> None:
    """H2: perfil causal por banda (queda de P(spoof) ao ocluir), IC 95%, por encoder."""
    have = [p for p in profs if p["occ"] is not None]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, p in enumerate(have):
        occ, c = p["occ"], plt.cm.tab10.colors[i % 10]
        ax.plot(occ["center"], occ["mean"], marker="o", ms=3, lw=1.4, color=c,
                label=p["label"])
        ax.fill_between(occ["center"], occ["ci_low"], occ["ci_high"], color=c, alpha=0.15)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\Delta$ P(spoof) when band removed")
    ax.set_title("Causal spectral profile (H2), adapted head — higher = band sustains spoof")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_occlusion_overlay.pdf")
    fig.savefig(out_dir / "encoder_occlusion_overlay.png", dpi=300)
    plt.close(fig)


def plot_assoc_heatmap(profs: list[dict], out_dir: Path) -> None:
    """H1: mapa encoder × banda do ρ de Spearman com sinal (associação com spoof)."""
    series = [(p["label"], p["assoc"]) for p in profs if p["assoc"] is not None]
    if not series:
        return
    bands = sorted(set().union(*[set(s) for _, s in series]))
    matrix = np.full((len(series), len(bands)), np.nan)
    for r, (_, s) in enumerate(series):
        for c, b in enumerate(bands):
            if b in s:
                matrix[r, c] = s[b]
    vmax = float(np.nanmax(np.abs(matrix)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(0.32 * len(bands) + 2.5, 0.55 * len(series) + 1.8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(series)), [lbl for lbl, _ in series])
    # Rótulos do eixo x em Hz (centro da banda), a partir de algum perfil com oclusão.
    centers = next((p["occ"]["center"] for p in profs if p["occ"] is not None), None)
    if centers is not None and len(centers) == len(bands):
        step = max(1, len(bands) // 6)
        ticks = list(range(0, len(bands), step))
        ax.set_xticks(ticks, [f"{centers[t]:.0f}" for t in ticks])
        ax.set_xlabel("Frequency (Hz, band center)")
    else:
        ax.set_xlabel("Band index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("signed Spearman ρ (spoof association)")
    ax.set_title("Associative spectral profile (H1), adapted head")
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_assoc_heatmap.pdf")
    fig.savefig(out_dir / "encoder_assoc_heatmap.png", dpi=300)
    plt.close(fig)


def plot_convergence_forest(profs: list[dict], out_dir: Path) -> None:
    """H1->H2: diferença de queda (top-assoc − bottom-assoc) por encoder, com IC 95%."""
    rows = [(p["label"], p["conv"]) for p in profs if p["conv"] is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(rows) + 2.0))
    los = [float(cv["ci_low"]) for _, cv in rows]
    his = [float(cv["ci_high"]) for _, cv in rows]
    span = (max(his + [0.0]) - min(los + [0.0])) or 1.0
    for i, (_, cv) in enumerate(rows):
        md, lo, hi = float(cv["median_diff"]), float(cv["ci_low"]), float(cv["ci_high"])
        ax.errorbar(md, i, xerr=[[md - lo], [hi - md]], fmt="o", ms=6,
                    color="#0072B2", capsize=3)
        ax.text(hi + 0.02 * span, i, f"d_z={float(cv['cohen_dz']):.2f}",
                va="center", fontsize=8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(range(len(rows)), [lbl for lbl, _ in rows])
    ax.invert_yaxis()
    ax.margins(x=0.18)  # espaço p/ o rótulo d_z não ser cortado
    ax.set_xlabel(r"Median $\Delta$drop (top-assoc $-$ bottom-assoc bands)")
    ax.set_title("H1→H2 convergence (positive = association predicts causal effect)",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_convergence_forest.pdf")
    fig.savefig(out_dir / "encoder_convergence_forest.png", dpi=300)
    plt.close(fig)


def plot_lrp_overlay(profs: list[dict], out_dir: Path) -> None:
    """Terceira lente (DFT-LRP): relevância por banda que o D_ad atribui a spoof, por encoder.

    Positivo = a faixa empurra a decisão rumo a spoof; negativo = rumo a bonafide. Linha cheia =
    backend conservativo (attnlrp, wav2vec2/hubert/wavlm); tracejada = referência não-conservativa
    (gxi). Só encoders hf_ssl entram (o XLS-R fairseq fica fora desta lente). As magnitudes só
    são diretamente comparáveis entre encoders conservativos; leia o gxi pela FORMA/sinal.
    """
    have = [p for p in profs if p.get("lrp") is not None]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, p in enumerate(have):
        lrp, c = p["lrp"], plt.cm.tab10.colors[i % 10]
        conservative = lrp["backend"] == "attnlrp"
        ax.plot(lrp["center"], lrp["mean"], marker="o", ms=3, lw=1.4, color=c,
                ls="-" if conservative else "--",
                label=f"{p['label']} ({'conservative' if conservative else 'gxi ref.'})")
        ax.fill_between(lrp["center"], lrp["ci_low"], lrp["ci_high"], color=c, alpha=0.12)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("DFT-LRP relevance (spoof logit)")
    ax.set_title("Internal spectral relevance (DFT-LRP), adapted head\n"
                 "(positive = pushes toward spoof)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_lrp_overlay.pdf")
    fig.savefig(out_dir / "encoder_lrp_overlay.png", dpi=300)
    plt.close(fig)


def plot_lrp_overlay_byclass(profs: list[dict], out_dir: Path) -> None:
    """Terceira lente por CLASSE PREDITA: relevância DFT-LRP por banda, dois painéis.

    Painel esquerdo: clipes preditos spoof; direito: preditos bonafide. Uma linha por encoder
    (cheia = conservativo attnlrp; tracejada = referência gxi). Condicionar por classe evita o
    cancelamento de sinais opostos que a média global sofre. Positivo = a banda empurra a decisão
    rumo a spoof; negativo = rumo a bonafide. Só encoders hf_ssl (o XLS-R fica fora desta lente).
    """
    have = [p for p in profs if p.get("lrp_byclass") is not None]
    if not have:
        return
    classes = ("spoof", "bonafide")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True, sharey=True)
    for ax, cls in zip(axes, classes):
        for i, p in enumerate(have):
            d = p["lrp_byclass"]["classes"].get(cls)
            if d is None:
                continue
            c = plt.cm.tab10.colors[i % 10]
            conservative = p["lrp_byclass"]["backend"] == "attnlrp"
            ax.plot(d["center"], d["mean"], marker="o", ms=3, lw=1.4, color=c,
                    ls="-" if conservative else "--",
                    label=f"{p['label']} ({'conservative' if conservative else 'gxi ref.'})")
            ax.fill_between(d["center"], d["ci_low"], d["ci_high"], color=c, alpha=0.12)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"predicted {cls}")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("DFT-LRP relevance (spoof logit)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Internal spectral relevance by predicted class (DFT-LRP), adapted head\n"
                 "(positive = pushes toward spoof)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "encoder_lrp_overlay_byclass.pdf")
    fig.savefig(out_dir / "encoder_lrp_overlay_byclass.png", dpi=300)
    plt.close(fig)


def plot_salience_overlay(profs: list[dict], out_dir: Path) -> None:
    """Saliência DFT-LRP sobreposta: fração da atenção (|relevância|) por frequência, por encoder.

    Uma linha por encoder (cheia = conservativo attnlrp; tracejada = referência gxi). Eixo x
    linear em Hz; y em % por bin. Direção-agnóstico (spoof e bonafide contam igual): mostra ONDE
    cada modelo se apoia no espectro. Só encoders hf_ssl (o XLS-R fica fora desta lente).
    """
    have = [p for p in profs if p.get("salience") is not None]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, p in enumerate(have):
        s, c = p["salience"], plt.cm.tab10.colors[i % 10]
        conservative = s["backend"] == "attnlrp"
        ax.plot(s["center"], s["share"], lw=1.3, color=c, ls="-" if conservative else "--",
                label=f"{p['label']} (peak ~{s['peak_hz']:.0f} Hz"
                      f"{'' if conservative else ', gxi ref.'})")
        ax.fill_between(s["center"], s["ci_low"], s["ci_high"], color=c, alpha=0.10, lw=0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Attention share |relevance| (% per bin)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # Duas versões: espectro cheio e um zoom em 0-2 kHz (onde a atenção se concentra).
    base_title = "DFT-LRP frequency attention, adapted head (where each encoder looks"
    ax.set_title(f"{base_title} in the spectrum)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_lrp_salience_overlay.pdf")
    fig.savefig(out_dir / "encoder_lrp_salience_overlay.png", dpi=300)
    ax.set_xlim(0, 2000)
    ax.set_title(f"{base_title}, 0–2 kHz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_lrp_salience_overlay_2khz.pdf")
    fig.savefig(out_dir / "encoder_lrp_salience_overlay_2khz.png", dpi=300)
    plt.close(fig)


def _spearman_matrix(cols: np.ndarray) -> np.ndarray:
    """Matriz de correlação de Spearman entre colunas (Pearson sobre os postos)."""
    ranks = np.argsort(np.argsort(cols, axis=0), axis=0).astype(float)
    return np.corrcoef(ranks, rowvar=False)


def _fmt_hz(hz: float) -> str:
    """Frequência legível: Hz abaixo de 1 kHz, kHz acima."""
    return f"{hz:.0f} Hz" if hz < 1000 else f"{hz / 1000:.2f} kHz"


def plot_lrp_top_bands_table(profs: list[dict], out_dir: Path, k: int = 5) -> None:
    """Tabela das bandas mais importantes (|relevância| DFT-LRP) por encoder.

    Colunas: encoders. Linhas: Top 1..k (banda de maior |relevância| primeiro). A linha Top 1
    fica destacada, para ver de imediato qual faixa cada modelo elege como a mais importante
    (e que elas raramente coincidem). Também escreve ``encoder_lrp_top_bands.csv`` e imprime,
    por banda top-1, o conjunto de encoders que a elegeram. Ranking por magnitude (ignora o
    lado spoof/bonafide).
    """
    got = _lrp_relevance_matrix(profs)
    if got is None:
        print("aviso: <2 encoders com DFT-LRP; pulando encoder_lrp_top_bands.")
        return
    labels, centers, cols = got
    centers = np.asarray(centers, dtype=float)
    absv = np.abs(cols)                                   # (n_bandas, n_enc)
    n_bands, n_enc = absv.shape
    k = min(k, n_bands)
    cell = [["" for _ in range(n_enc)] for _ in range(k)]
    csv_rows, top1_by_band = [], {}
    for j in range(n_enc):
        order = np.argsort(-absv[:, j], kind="stable")[:k]
        for r, bi in enumerate(order):
            cell[r][j] = _fmt_hz(centers[bi])
            csv_rows.append({"encoder": labels[j], "rank": r + 1,
                             "band_center_hz": round(float(centers[bi]), 1),
                             "abs_relevance": round(float(absv[bi, j]), 4),
                             "signed_relevance": round(float(cols[bi, j]), 4)})
        top1_by_band.setdefault(_fmt_hz(centers[order[0]]), []).append(labels[j])

    fig, ax = plt.subplots(figsize=(2.1 * n_enc + 1.6, 0.5 * k + 1.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=[f"Top {i + 1}" for i in range(k)],
                   colLabels=labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(n_enc):                                # destaca a linha Top 1 (row 1: header=0)
        tbl[(1, j)].set_facecolor("#FFF2CC")
    ax.set_title("Most important DFT-LRP bands per encoder (by |relevance|)",
                 fontsize=10, pad=14)
    fig.savefig(out_dir / "encoder_lrp_top_bands.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "encoder_lrp_top_bands.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_rows, out_dir / "encoder_lrp_top_bands.csv")
    print("\nDFT-LRP top-1 band (by |relevance|) -> encoders:")
    for band, encs in sorted(top1_by_band.items(), key=lambda kv: -len(kv[1])):
        print(f"  {band:>10}: {', '.join(encs)}")


def _lrp_relevance_matrix(profs: list[dict]) -> tuple[list[str], list[float], np.ndarray] | None:
    """Matriz banda × encoder da relevância DFT-LRP com sinal (perfil do overlay).

    Alinha os encoders pela mesma grade de bandas (centros em Hz). Devolve
    (labels, centros_hz, matriz) com matriz de forma (n_bandas, n_encoders), ou None se
    houver menos de dois encoders com DFT-LRP.
    """
    have = [p for p in profs if p.get("lrp") is not None]
    if len(have) < 2:
        return None
    keyed = [{round(float(c), 2): float(m)
              for c, m in zip(p["lrp"]["center"], p["lrp"]["mean"])} for p in have]
    common = sorted(set.intersection(*[set(d) for d in keyed]))
    if len(common) < 3:
        return None
    labels = [p["label"] for p in have]
    cols = np.column_stack([[d[k] for k in common] for d in keyed])  # (n_bandas, n_enc)
    return labels, common, cols


def _annot_heatmap(ax, mat: np.ndarray, labels: list[str], title: str) -> "matplotlib.image.AxesImage":
    """Heatmap de correlação (-1..1, RdBu_r) com valores anotados; devolve o objeto de imagem."""
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j]) > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    return im


def plot_lrp_correlation(profs: list[dict], out_dir: Path) -> None:
    """Concordância cruzada (Spearman) entre encoders dos perfis de relevância DFT-LRP por banda.

    Matriz par a par do Spearman: mede se as arquiteturas ordenam as bandas do mesmo jeito
    (quais frequências pesam), ignorando a escala bruta da relevância, que difere entre
    arquiteturas. Baixo = as arquiteturas discordam sobre quais frequências importam.
    Calculada sobre a relevância por banda COM SINAL (o perfil do ``encoder_lrp_overlay``),
    backend conservativo.
    """
    got = _lrp_relevance_matrix(profs)
    if got is None:
        print("aviso: <2 encoders com DFT-LRP; pulando encoder_lrp_correlation.")
        return
    labels, _, cols = got
    n_bands = cols.shape[0]
    spearman = _spearman_matrix(cols)
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.2, 1.1 * len(labels) + 2.0))
    im = _annot_heatmap(ax, spearman, labels, "")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ")
    ax.set_title(f"Cross-encoder agreement of DFT-LRP band relevance\n"
                 f"Spearman rank of which bands matter (n={n_bands} bands)", fontsize=10)
    fig.savefig(out_dir / "encoder_lrp_correlation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "encoder_lrp_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # CSV par a par (triângulo superior) + resumo no stdout.
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append({"encoder_a": labels[i], "encoder_b": labels[j],
                          "spearman": round(float(spearman[i, j]), 4)})
    _write_csv(pairs, out_dir / "encoder_lrp_correlation.csv")
    print("\nDFT-LRP cross-encoder Spearman (signed band relevance):")
    for p in pairs:
        print(f"  {p['encoder_a']:>16} vs {p['encoder_b']:<16} Spearman={p['spearman']:+.2f}")


def plot_pred_agreement(run_dirs: list[str], out_dir: Path) -> None:
    """Concordância clipe a clipe entre encoders: Spearman do P(spoof) do D_ad.

    Só é possível porque as runs compartilham exatamente os mesmos clipes de teste
    (mesmos `sample_id`). Um ρ alto entre dois encoders indica que eles pontuam os
    mesmos clipes de forma parecida (ordenam o risco de spoof de modo semelhante),
    mesmo que a EER difira. Lê `master_table.parquet` (requer pandas); se pandas não
    estiver disponível, a figura é pulada com aviso, sem quebrar as demais.
    """
    try:
        import pandas as pd
    except ImportError:
        print("aviso: pandas ausente; pulando encoder_pred_agreement.")
        return
    series = {}
    for rd in run_dirs:
        run = Path(rd)
        f = run / "master_table.parquet"
        if not f.exists():
            continue
        m = pd.read_parquet(f, columns=["sample_id", "p_spoof_ad"])
        series[_label(*_encoder_id(run))] = m.set_index("sample_id")["p_spoof_ad"]
    if len(series) < 2:
        return
    labels = sorted(series)
    common = set.intersection(*[set(series[k].index) for k in labels])
    if len(common) < 50:
        print(f"aviso: só {len(common)} clipes em comum; pulando pred_agreement.")
        return
    base = sorted(common)
    cols = np.column_stack([series[k].loc[base].to_numpy() for k in labels])
    rho = _spearman_matrix(cols)
    fig, ax = plt.subplots(figsize=(0.8 * len(labels) + 2.5, 0.8 * len(labels) + 2.0))
    im = ax.imshow(rho, cmap="YlGnBu", vmin=float(np.min(rho[rho < 1])), vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if rho[i, j] > 0.8 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ de P(spoof)")
    ax.set_title(f"Concordância de previsões entre encoders (D_ad, {len(base)} clipes)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_pred_agreement.pdf")
    fig.savefig(out_dir / "encoder_pred_agreement.png", dpi=300)
    plt.close(fig)


def _print_table(rows: list[dict]) -> None:
    """Tabela curta no stdout (sem pandas)."""
    header = f"{'label':<22}{'eer_ad':>9}{'eer_zs':>9}{'mcc_ad':>9}{'acc_ad':>9}"
    print(header)
    for r in rows:
        print(f"{r['label']:<22}{r['eer_ad']:>9.4f}{r['eer_zs']:>9.4f}"
              f"{r['mcc_ad']:>9.4f}{r['accuracy_ad']:>9.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_encoders")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dirs", nargs="+", help="diretórios de run (um por encoder)")
    src.add_argument("--results-root", help="varre results/ e pega a run mais recente por encoder")
    ap.add_argument("--out", default=None,
                    help="pasta de saída (default: <results-root>/_aggregate ou results/_aggregate)")
    ap.add_argument("--exclude", nargs="*", default=["mms-300m"],
                    help="labels a excluir de TODAS as figuras (peer e DFT-LRP). Default: mms-300m "
                         "(removido por ora). Use '--exclude' sem valor para não excluir nada.")
    ap.add_argument("--peer-exclude", nargs="*", default=[],
                    help="labels a excluir só das figuras peer (EER/H1/H2/convergência/"
                         "concordância), mantidos nas figuras DFT-LRP. Default: vazio.")
    args = ap.parse_args(argv)
    if args.results_root:
        run_dirs = discover_runs(args.results_root)
        default_out = Path(args.results_root) / AGGREGATE_DIR
    else:
        run_dirs = args.run_dirs
        default_out = Path("results") / AGGREGATE_DIR
    if not run_dirs:
        print("nenhuma run com performance_table.csv encontrada.")
        return 1
    out_dir = Path(args.out) if args.out else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exclusão global: remove os labels de --exclude de TODAS as figuras (peer e DFT-LRP).
    global_excl = set(args.exclude or [])
    if global_excl:
        run_dirs = [rd for rd in run_dirs
                    if _label(*_encoder_id(Path(rd))) not in global_excl]
        print(f"exclude(global)={sorted(global_excl)} (fora de TODAS as figuras)")

    # Figuras peer: opcionalmente excluem labels adicionais (só das peer), via --peer-exclude.
    exclude = set(args.peer_exclude or [])
    peer_dirs = [rd for rd in run_dirs if _label(*_encoder_id(Path(rd))) not in exclude]
    if exclude:
        print(f"peer_exclude={sorted(exclude)} (fora das figuras EER/H1/H2/convergência)")

    peer_rows = collect(peer_dirs)
    _write_csv(peer_rows, out_dir / "encoder_comparison.csv")
    plot_eer(peer_rows, out_dir)
    peer_profs = collect_profiles(peer_dirs, order=[r["label"] for r in peer_rows])
    plot_occlusion_overlay(peer_profs, out_dir)   # H2
    plot_assoc_heatmap(peer_profs, out_dir)        # H1
    plot_convergence_forest(peer_profs, out_dir)   # H1->H2
    plot_pred_agreement(peer_dirs, out_dir)        # concordância clipe a clipe (D_ad)

    # Terceira lente (DFT-LRP): usa os encoders restantes após a exclusão global.
    # Com `--exclude` vazio, o mms-300m entra como proxy do front-end do XLS-R.
    all_rows = collect(run_dirs)
    all_profs = collect_profiles(run_dirs, order=[r["label"] for r in all_rows])
    plot_lrp_overlay(all_profs, out_dir)          # terceira lente (DFT-LRP)
    plot_lrp_overlay_byclass(all_profs, out_dir)  # terceira lente por classe predita
    plot_salience_overlay(all_profs, out_dir)     # saliência DFT-LRP sobreposta por encoder
    plot_lrp_correlation(all_profs, out_dir)      # concordância cruzada (Spearman) do DFT-LRP
    plot_lrp_top_bands_table(all_profs, out_dir)  # tabela das bandas mais importantes por encoder

    _print_table(all_rows)
    print(f"\nsalvo em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
