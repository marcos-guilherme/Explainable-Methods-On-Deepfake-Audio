"""Renderiza o perfil de associação COM SINAL (barras pra cima/baixo, formato largo)
a partir do spearman_table.csv de um run existente. Complementa a versão em |ρ|.

Uso:
    PYTHONPATH=scripts /tmp/figvenv/bin/python scripts/dev/make_assoc_signed_figure.py \
        --run-dir results/default-YYYYMMDD-HHMMSS
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from brspeech_xai.plotting import plot_association_profile_signed, set_plot_style


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_assoc_signed_figure")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="pasta de figuras (default: <run-dir>/figures)")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    spearman = pd.read_csv(run / "spearman_table.csv")
    out = Path(args.out) if args.out else run / "figures"
    set_plot_style()
    plot_association_profile_signed(spearman, out)
    print(f"salvo: {out}/association_profile_signed_zs_vs_ad.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
