"""Retoma o pipeline em um diretório de run EXISTENTE (o CLI padrão sempre cria um
diretório novo com timestamp). Útil para reprocessar estágios após mudança de código
reaproveitando artefatos pesados já persistidos (áudios/embeddings/scores).

Uso (dentro do container):
    python /workspace/scripts/dev/resume_run.py \
        --config /workspace/scripts/configs/default.yaml \
        --run-dir /workspace/results/default-YYYYMMDD-HHMMSS \
        --from features --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

from brspeech_xai.artifacts import RunPaths
from brspeech_xai.config import load_config
from brspeech_xai.logging_utils import get_logger
from brspeech_xai.pipeline import run_stages
from brspeech_xai.seed import set_seed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resume_run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.device == "auto":
        import torch
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    paths = RunPaths(root=Path(args.run_dir))
    logger = get_logger(logfile=paths.path("run.log"))
    logger.info(f"RESUME run_dir={args.run_dir} device={cfg.device} "
                f"hash={cfg.config_hash()}")
    run_stages(cfg, paths, logger, ctx_extra={"plots": not args.no_plots},
               start=args.start, only=args.only, force=args.force)
    logger.info(f"pipeline concluído: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
