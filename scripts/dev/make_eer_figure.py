"""Renderiza a figura didática de EER (taxas de erro × limiar, D_zs vs D_ad) a partir
de um diretório de run existente (usa a master_table.parquet). Complementa a curva DET.

Uso:
    PYTHONPATH=scripts /tmp/figvenv/bin/python scripts/dev/make_eer_figure.py \
        --run-dir results/default-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from brspeech_xai.plotting import plot_eer_threshold, set_plot_style


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_eer_figure")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="pasta de figuras (default: <run-dir>/figures)")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    master = pd.read_parquet(run / "master_table.parquet")
    out = Path(args.out) if args.out else run / "figures"
    set_plot_style()
    plot_eer_threshold(master, out)
    print(f"salvo: {out}/eer_threshold_zs_vs_ad.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
