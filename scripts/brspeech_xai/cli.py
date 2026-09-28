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
    _log_run_header(logger, cfg, run_id, root)
    run_stages(cfg, paths, logger, ctx_extra={"plots": not args.no_plots},
               start=args.start, only=args.only, force=args.force)
    logger.info(f"pipeline concluído: {root}")
    return 0


def _log_run_header(logger, cfg, run_id: str, root) -> None:
    """Cabeçalho do run: dá o panorama (encoder, device, grade, dados) num relance."""
    logger.info("=" * 60)
    logger.info(f"run_id={run_id} | encoder={cfg.model.encoder} | device={cfg.device}")
    logger.info(f"checkpoint={cfg.model.checkpoint} | hash={cfg.config_hash()}")
    logger.info(f"bandas={cfg.bands.n_bands} "
                f"[{cfg.bands.f_min:.0f}-{cfg.bands.f_max:.0f} Hz] | "
                f"head={cfg.adapt.head} | cross_fit={cfg.adapt.cross_fit}")
    logger.info(f"dados: treino={cfg.data.n_train_per_class}/classe, "
                f"análise={cfg.data.n_analysis_per_class}/classe "
                f"(split={cfg.data.analysis_split})")
    logger.info(f"saída: {root}")
    logger.info("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
