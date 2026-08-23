"""Entrada de linha de comando do pipeline."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .config import dump_resolved, encoder_slug, load_config
from .logging_utils import get_logger
from .pipeline import run_stages
from .seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="brspeech_xai")
    ap.add_argument("--config", required=True, help="caminho do YAML de config")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override chave.aninhada=valor (repetível)")
    ap.add_argument("--from", dest="start", default=None, help="começa deste estágio")
    ap.add_argument("--only", default=None, help="roda só este estágio")
    ap.add_argument("--force", action="store_true", help="ignora 'done markers'")
    ap.add_argument("--no-plots", action="store_true", help="não gera figuras")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)
    if cfg.device == "auto":
        import torch
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)
    run_id = f"{cfg.run_name}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    # Agrupa as runs por versão de encoder: results/<encoder>/<run_id>/.
    root = Path(cfg.output_dir) / encoder_slug(cfg.model) / run_id
    from .artifacts import RunPaths
    paths = RunPaths(root=root)
    logger = get_logger(logfile=paths.path("run.log"))
    dump_resolved(cfg, paths.path("config.resolved.yaml"))
    logger.info("run_id=%s device=%s hash=%s", run_id, cfg.device, cfg.config_hash())
    run_stages(cfg, paths, logger, ctx_extra={"plots": not args.no_plots},
               start=args.start, only=args.only, force=args.force)
    logger.info("pipeline concluído: %s", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
