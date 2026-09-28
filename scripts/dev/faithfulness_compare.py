"""Compara a fidelidade DFT-LRP entre encoders num único quadro de barras.

Lê os CSVs ``faithfulness_aopc_<encoder>.csv`` (gerados por ``faithfulness_bands.py``)
e desenha, para cada métrica (sufficiency, comprehensiveness) e classe (spoof, bonafide),
o AOPC das faixas DFT-LRP contra a baseline aleatória, com intervalo de confiança de 95%.

A comparação que importa é DFT-LRP vs aleatório dentro do mesmo painel: em sufficiency,
DFT-LRP acima do aleatório indica que as faixas bastam; em comprehensiveness, DFT-LRP
acima do aleatório indica que as faixas são necessárias.

Uso:
    python scripts/dev/faithfulness_compare.py \
        --faith-dir results/_aggregate/faithfulness \
        --out results/_aggregate/faithfulness_compare
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ordem e rótulos curtos dos encoders (do mais para o menos fiel observado).
ENCODER_ORDER = ["wav2vec2-base", "hubert-base-ls960", "wavlm-base"]
SHORT = {"wav2vec2-base": "wav2vec2", "hubert-base-ls960": "hubert", "wavlm-base": "wavlm"}

METRICS = ["sufficiency", "comprehensiveness"]
CLASSES = ["spoof", "bonafide"]
# condição DFT-LRP e condição aleatória por métrica.
COND = {
    "sufficiency": ("keep_dftlrp", "keep_random"),
    "comprehensiveness": ("delete_dftlrp", "delete_random"),
}
# cor por métrica (verde = manter/keep; vermelho = remover/delete), coerente com a curva.
COLOR = {"sufficiency": "#2e7d32", "comprehensiveness": "#c0392b"}


def _read_aopc(path: Path) -> dict:
    """Indexa o CSV por (metric, condition, true_class) -> (aopc, ci_low, ci_high)."""
    table = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r["metric"], r["condition"], r["true_class"])
            table[key] = (float(r["aopc"]), float(r["ci_low"]), float(r["ci_high"]))
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faith-dir", default="results/_aggregate/faithfulness")
    ap.add_argument("--out", default="results/_aggregate/faithfulness_compare")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS,
                    help="quais métricas plotar (default: ambas)")
    args = ap.parse_args()

    metrics = [m for m in METRICS if m in args.metrics]

    faith_dir = Path(args.faith_dir)
    data = {}
    for enc in ENCODER_ORDER:
        p = faith_dir / f"faithfulness_aopc_{enc}.csv"
        if p.exists():
            data[enc] = _read_aopc(p)
    if not data:
        raise SystemExit(f"nenhum faithfulness_aopc_*.csv em {faith_dir}")

    encoders = [e for e in ENCODER_ORDER if e in data]
    x = np.arange(len(encoders), dtype=float)
    width = 0.36

    nrows = len(metrics)
    fig, axes = plt.subplots(nrows, 2, figsize=(10.5, 3.8 * nrows), sharex=True,
                             squeeze=False)
    for i, metric in enumerate(metrics):
        cond_lrp, cond_rand = COND[metric]
        base = COLOR[metric]
        for j, cls in enumerate(CLASSES):
            ax = axes[i, j]
            lrp_y, lrp_err = [], [[], []]
            rnd_y, rnd_err = [], [[], []]
            for enc in encoders:
                a, lo, hi = data[enc][(metric, cond_lrp, cls)]
                lrp_y.append(a); lrp_err[0].append(a - lo); lrp_err[1].append(hi - a)
                a, lo, hi = data[enc][(metric, cond_rand, cls)]
                rnd_y.append(a); rnd_err[0].append(a - lo); rnd_err[1].append(hi - a)

            ax.bar(x - width / 2, lrp_y, width, yerr=lrp_err, capsize=3,
                   color=base, label="DFT-LRP")
            ax.bar(x + width / 2, rnd_y, width, yerr=rnd_err, capsize=3,
                   color=base, alpha=0.35, hatch="//", edgecolor="white",
                   label="aleatório")

            ax.set_title(f"{metric} — {cls}", fontsize=11)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            if j == 0:
                ax.set_ylabel("AOPC (p_pred)" if metric == "sufficiency"
                              else "AOPC (queda de p_pred)")
            if i == nrows - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([SHORT[e] for e in encoders])

    axes[0, 0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Faithfulness by frequency bands: DFT-LRP vs random (AOPC, 95% CI)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {out}.png / {out}.pdf")


if __name__ == "__main__":
    main()
