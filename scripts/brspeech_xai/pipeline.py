"""Orquestração resumível dos estágios (skip via 'done markers' por hash de config)."""
from __future__ import annotations

import time

from . import artifacts as A
from . import stages as S
from .logging_utils import format_duration

STAGES = [
    ("collect", S.stage_collect),
    ("embeddings", S.stage_embeddings),
    ("adapt", S.stage_adapt),
    ("features", S.stage_features),
    ("association", S.stage_association),
    ("occlusion", S.stage_occlusion),
    ("confirmatory", S.stage_confirmatory),
    ("report", S.stage_report),
]


def _selected(start=None, only=None):
    if only:
        return [(n, f) for n, f in STAGES if n == only]
    if start:
        idx = [n for n, _ in STAGES].index(start)
        return STAGES[idx:]
    return list(STAGES)


def run_stages(cfg, paths, logger, ctx_extra, start=None, only=None, force=False):
    from .bands import configure_bands
    # Reconfigura a grade compartilhada (H1 e H2) ANTES de qualquer estágio.
    configure_bands(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
    chash = cfg.config_hash()
    selected = _selected(start, only)
    total = len(selected)
    durations: dict[str, float] = {}
    for i, (name, fn) in enumerate(selected, 1):
        if not force and A.is_done(paths, name, chash):
            if logger:
                logger.info(f"[{i}/{total}] estágio '{name}' já concluído; pulando")
            continue
        if logger:
            logger.info(f"[{i}/{total}] ▶ estágio: {name}")
        started = time.perf_counter()
        ctx = _make_ctx(cfg, paths, logger, ctx_extra)
        fn(ctx)
        elapsed = time.perf_counter() - started
        durations[name] = elapsed
        A.write_done(paths, name, chash, meta={"seconds": round(elapsed, 3)})
        if logger:
            done = format_duration(sum(durations.values()))
            logger.info(f"[{i}/{total}] ✓ '{name}' em {format_duration(elapsed)}"
                        f" (acumulado {done})")
    if logger and durations:
        _log_timing_summary(logger, durations)


def _log_timing_summary(logger, durations: dict[str, float]) -> None:
    """Tabela final: tempo por estágio e fração do total (ajuda a achar gargalos)."""
    total = sum(durations.values())
    logger.info("resumo de tempos por estágio:")
    for name, seconds in durations.items():
        share = 100.0 * seconds / total if total else 0.0
        logger.info(f"  {name:<13}{format_duration(seconds):>10}  ({share:4.1f}%)")
    logger.info(f"  {'TOTAL':<13}{format_duration(total):>10}")


def _make_ctx(cfg, paths, logger, ctx_extra):
    from .stages import RunContext
    ctx = RunContext(cfg=cfg, paths=paths, logger=logger)
    for k, v in (ctx_extra or {}).items():
        setattr(ctx, k, v)
    return ctx
