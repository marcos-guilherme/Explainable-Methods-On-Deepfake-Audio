"""Escopos XAI da matriz: completo na diagonal e reduzido sob corpus shift."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import joblib

from . import artifacts as A
from .config import RunConfig
from .matrix import LANGUAGES, MatrixPaths, _validate_configs
from .stages import (
    RunContext,
    stage_association,
    stage_confirmatory,
    stage_features,
    stage_occlusion,
    stage_report,
)


def _analysis_root(paths: MatrixPaths, source: str, target: str) -> Path:
    if source == target:
        return paths.root / "xai" / source
    return paths.root / "shift" / f"{source}_to_{target}"


def _prepare_scope_artifacts(
    paths: MatrixPaths,
    source: str,
    target: str,
) -> A.RunPaths:
    root = A.RunPaths(_analysis_root(paths, source, target))
    catalog = A.load_table(paths.catalog(target, "test")).copy()
    catalog["split"] = "test"
    A.save_table(catalog, root.path("samples.parquet"))
    A.save_npy(A.load_npy(paths.cell_scores(source, target)), root.path("p_spoof_ad.npy"))
    A.save_json(A.load_json(paths.thresholds(source)), root.path("thresholds.json"))
    A.save_json(A.load_json(paths.cell_metrics(source, target)), root.path("cell_metrics.json"))
    joblib.dump(joblib.load(paths.model(source)), root.path("d_ad.joblib"))
    return root


def _validate_xai_inputs(paths: MatrixPaths) -> None:
    for target in LANGUAGES:
        catalog = A.load_table(paths.catalog(target, "test"))
        if "processed_path" not in catalog.columns:
            raise ValueError(f"catálogo test de {target} não contém processed_path")
        missing = [
            str(raw_path)
            for raw_path in catalog["processed_path"]
            if not Path(str(raw_path)).is_file()
        ]
        if missing:
            raise ValueError(
                f"catálogo test de {target} contém processed_path ausente: {missing[0]}"
            )


def _context(
    cfg: RunConfig,
    paths: A.RunPaths,
    *,
    embedder,
    logger,
    per_quadrant: int | None = None,
    plots: bool = False,
) -> RunContext:
    scoped = copy.deepcopy(cfg)
    scoped.data.eval_split = "test"
    scoped.adapt.cross_fit = False
    if per_quadrant is not None:
        scoped.occlusion.per_quadrant = min(
            scoped.occlusion.per_quadrant,
            per_quadrant,
        )
    ctx = RunContext(cfg=scoped, paths=paths, logger=logger, embedder=embedder)
    ctx.plots = plots
    return ctx


def run_xai_scopes(
    configs: Mapping[str, RunConfig],
    paths: MatrixPaths | str | Path,
    *,
    embedder=None,
    reduced_per_quadrant: int = 25,
    plots: bool = False,
    logger=None,
) -> None:
    """Executa H1/H2/H3 diagonal; off-diagonal usa H1 completo e H2 reduzido."""
    from .encoders import build_encoder

    _validate_configs(configs)
    if reduced_per_quadrant <= 0:
        raise ValueError("reduced_per_quadrant deve ser positivo")
    paths = paths if isinstance(paths, MatrixPaths) else MatrixPaths(paths)
    _validate_xai_inputs(paths)
    if embedder is None:
        reference = configs[LANGUAGES[0]]
        embedder = build_encoder(reference.model, reference.device)

    for source in LANGUAGES:
        for target in LANGUAGES:
            scoped_paths = _prepare_scope_artifacts(paths, source, target)
            diagonal = source == target
            ctx = _context(
                configs[target],
                scoped_paths,
                embedder=embedder,
                logger=logger,
                per_quadrant=None if diagonal else reduced_per_quadrant,
                plots=plots,
            )
            stage_features(ctx)
            stage_association(ctx)
            stage_occlusion(ctx)
            if diagonal:
                stage_confirmatory(ctx)
                stage_report(ctx)
                scope = {
                    "schema_version": 1,
                    "source": source,
                    "target": target,
                    "scope": "full_diagonal_xai",
                    "stages": ["features", "association", "occlusion", "confirmatory", "report"],
                    "confirmatory_h3": True,
                    "interpretation": "within_language_test",
                    "model_and_threshold_from": source,
                    "evaluation_data_from": target,
                    "threshold_calibration_source": f"{source}/calibration",
                    "canonical_cell_metrics": str(paths.cell_metrics(source, target)),
                }
            else:
                scope = {
                    "schema_version": 1,
                    "source": source,
                    "target": target,
                    "scope": "reduced_cross_corpus_shift",
                    "stages": [
                        "features_all_target_test",
                        "association_h1_all_target_test",
                        "occlusion_h2_reduced",
                    ],
                    "confirmatory_h3": False,
                    "occlusion_per_quadrant": ctx.cfg.occlusion.per_quadrant,
                    "interpretation": "external_validation_under_corpus_shift",
                    "claim_limit": "not_evidence_of_universal_deepfake_explanations",
                    "model_and_threshold_from": source,
                    "evaluation_data_from": target,
                    "threshold_calibration_source": f"{source}/calibration",
                    "canonical_cell_metrics": str(paths.cell_metrics(source, target)),
                }
            A.save_json(scope, scoped_paths.path("scope.json"))
            if logger:
                logger.info(f"XAI {source}->{target}: {scope['scope']}")
