"""Orquestração resumível dos estágios (skip via 'done markers' por hash de config)."""
from __future__ import annotations

from . import artifacts as A
from . import stages as S

STAGES = [
    ("collect", S.stage_collect),
    ("embeddings", S.stage_embeddings),
    ("adapt", S.stage_adapt),
    ("master_mfcc", S.stage_master_mfcc),
    ("shap", S.stage_shap),
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
    chash = cfg.config_hash()
    for name, fn in _selected(start, only):
        if not force and A.is_done(paths, name, chash):
            if logger:
                logger.info("estágio '%s' já concluído; pulando", name)
            continue
        if logger:
            logger.info(">> executando estágio: %s", name)
        ctx = _make_ctx(cfg, paths, logger, ctx_extra)
        fn(ctx)
        A.write_done(paths, name, chash, meta={})


def _make_ctx(cfg, paths, logger, ctx_extra):
    from .stages import RunContext
    ctx = RunContext(cfg=cfg, paths=paths, logger=logger)
    for k, v in (ctx_extra or {}).items():
        setattr(ctx, k, v)
    return ctx
