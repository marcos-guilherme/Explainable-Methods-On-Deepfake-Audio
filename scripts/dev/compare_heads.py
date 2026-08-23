"""Compara a EER cross-validada (OOF) de diferentes heads a partir dos embeddings
de um run existente (emb_pool.npy + samples.parquet). Não usa GPU nem o encoder.

Uso:
    PYTHONPATH=scripts /tmp/figvenv/bin/python scripts/dev/compare_heads.py \
        --run-dir results/default-YYYYMMDD-HHMMSS --heads logistic mlp --folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brspeech_xai.adaptation import build_head, crossfit_oof_scores
from brspeech_xai.metrics import compute_eer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_heads")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="pool")
    ap.add_argument("--heads", nargs="+", default=["logistic", "mlp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    samples = pd.read_parquet(run / "samples.parquet")
    y = samples.loc[samples.split == args.split, "label"].to_numpy()
    emb = np.load(run / f"emb_{args.split}.npy")
    print(f"pool: n={len(y)}  spoof={int(y.sum())}  bonafide={int((y == 0).sum())}")
    for head in args.heads:
        make = lambda h=head: build_head(h, args.seed)
        oof = crossfit_oof_scores(emb, y, make, n_splits=args.folds,
                                  seed=args.seed, spoof_label=1)
        eer, thr = compute_eer(oof, y)
        print(f"head={head:9s}  EER(OOF)={eer * 100:6.2f}%  thr={thr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
