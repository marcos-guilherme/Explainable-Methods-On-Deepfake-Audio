"""Agrega os resultados de várias runs (uma por encoder) numa tabela e figura de EER.

Cada encoder roda em uma run separada (o `config_hash` inclui `model.encoder`, então os
artefatos ficam isolados por diretório). Este passo lê o `performance_table.csv` e o
`config.resolved.yaml` de cada run e monta:
  - `encoder_comparison.csv`: encoder, checkpoint, eer_ad, eer_zs (se houver), mcc/acc do D_ad.
  - `encoder_comparison_eer.{pdf,png}`: barras de EER do D_ad por encoder (rótulos em inglês).

Uso (fora do container; só pandas/matplotlib/yaml):
    # explícito (uma pasta de run por encoder)
    PYTHONPATH=scripts python scripts/dev/compare_encoders.py \
        --run-dirs results/xlsr_fairseq/default-A results/hf_ssl-hubert-base-ls960/hubert-B
    # automático: varre results/, pega a run mais recente de cada pasta de encoder
    PYTHONPATH=scripts python scripts/dev/compare_encoders.py --results-root results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


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


AGGREGATE_DIR = "_aggregate"


def discover_runs(results_root: str) -> list[str]:
    """Varre results/ e devolve a run mais recente de cada pasta de encoder.

    Estrutura esperada: results/<encoder>/<run_id>/. Ignora a pasta de agregado
    (prefixo '_') e aceita também runs 'flat' legadas (performance_table na raiz).
    """
    root = Path(results_root)
    runs = []
    for enc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if enc_dir.name.startswith("_"):
            continue
        if (enc_dir / "performance_table.csv").exists():   # run flat (legado)
            runs.append(str(enc_dir))
            continue
        sub = [d for d in enc_dir.iterdir()
               if d.is_dir() and (d / "performance_table.csv").exists()]
        if sub:
            runs.append(str(sorted(sub)[-1]))              # mais recente por nome (timestamp)
    return runs


def collect(run_dirs: list[str]) -> pd.DataFrame:
    """Lê cada run e devolve uma linha por encoder com EER do D_ad (e D_zs se houver)."""
    rows = []
    for rd in run_dirs:
        run = Path(rd)
        perf = pd.read_csv(run / "performance_table.csv").set_index("detector")
        encoder, checkpoint = _encoder_id(run)
        row = {"run_dir": run.name, "encoder": encoder, "checkpoint": checkpoint,
               "label": _label(encoder, checkpoint),
               "eer_ad": float(perf.loc["ad", "eer"]),
               "mcc_ad": float(perf.loc["ad", "mcc"]),
               "accuracy_ad": float(perf.loc["ad", "accuracy"])}
        row["eer_zs"] = float(perf.loc["zs", "eer"]) if "zs" in perf.index else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("eer_ad").reset_index(drop=True)


def plot_eer(df: pd.DataFrame, out_dir: Path) -> None:
    """Barras horizontais de EER do D_ad por encoder (menor = melhor)."""
    fig, ax = plt.subplots(figsize=(5.0, 0.5 + 0.5 * len(df)))
    y = range(len(df))
    ax.barh(list(y), (df["eer_ad"] * 100).to_numpy(), color="#D55E00", alpha=0.85)
    ax.set_yticks(list(y), df["label"].tolist())
    ax.invert_yaxis()  # melhor (menor EER) no topo
    ax.set_xlabel("Adapted-head EER (%)")
    for i, v in enumerate(df["eer_ad"] * 100):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "encoder_comparison_eer.pdf")
    fig.savefig(out_dir / "encoder_comparison_eer.png", dpi=300)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_encoders")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dirs", nargs="+", help="diretórios de run (um por encoder)")
    src.add_argument("--results-root", help="varre results/ e pega a run mais recente por encoder")
    ap.add_argument("--out", default=None,
                    help="pasta de saída (default: <results-root>/_aggregate ou results/_aggregate)")
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
    df = collect(run_dirs)
    df.to_csv(out_dir / "encoder_comparison.csv", index=False)
    plot_eer(df, out_dir)
    print(df.to_string(index=False))
    print(f"\nsalvo em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
