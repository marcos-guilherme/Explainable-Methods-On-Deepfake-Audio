"""Sensibilidade do probe linear à reamostragem estratificada do treino (embeddings fixos)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .adaptation import fit_head, score_head
from .encoder_suite import LANGUAGE_ORDER, LAYER_ORDER, ROLE_ORDER
from .layerwise_paths import (
    LayerwiseSuitePaths,
    assert_catalog_aligned_with_embedding_metadata,
    read_embedding_metadata_json,
)
from .layerwise_probes import (
    LANGUAGE_ORDER as _PROBE_LANGUAGES,
    _default_metrics,
    _validate_inputs,
    _validate_layers,
    _validate_scores,
)
from .metrics import calibrate_threshold


DEFAULT_PROBE_STABILITY_SEEDS: tuple[int, ...] = (42, 43, 44)
STRATIFIED_TRAIN_BOOTSTRAP_METHOD = "stratified_bootstrap_with_replacement_per_class"
# Regressão logística sklearn (solver padrão) é determinística; a variação vem só do bootstrap.
_PROBE_REFIT_RANDOM_STATE = 0
_PROBE_STABILITY_SCOPE_PT = (
    "Análise de sensibilidade à amostra finita de treino do probe linear por "
    "reamostragem estratificada com reposição (mesmos embeddings materializados)."
)
_PROBE_STABILITY_DISCLAIMER = (
    "Stratified bootstrap sensitivity of the linear probe training set only; "
    "does not measure XAI attribution stability, encoder/XAI end-to-end seed "
    "stability, or recomputed explanations."
)
_SUMMARY_KEYS = ("profile", "layer", "source", "target")


def validate_bootstrap_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a non-empty sequence")
    if len(seeds) == 0:
        raise ValueError("seeds must be a non-empty sequence")
    normalized: list[int] = []
    seen: set[int] = set()
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("each seed must be an integer")
        if seed in seen:
            raise ValueError(f"duplicate bootstrap seed: {seed}")
        seen.add(seed)
        normalized.append(seed)
    return tuple(normalized)


def _bootstrap_rng_seed(bootstrap_seed: int, source: str) -> int:
    digest = hashlib.sha256(
        f"stratified_probe_train_bootstrap|{bootstrap_seed}|{source}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**31)


def stratified_train_bootstrap_indices(
    labels: np.ndarray,
    *,
    bootstrap_seed: int,
    source: str,
) -> np.ndarray:
    """Bootstrap estratificado: reposição dentro de cada classe, mantendo contagens."""
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("train labels must be non-empty")
    rng = np.random.default_rng(_bootstrap_rng_seed(bootstrap_seed, source))
    parts: list[np.ndarray] = []
    for class_label in (0, 1):
        pool = np.flatnonzero(labels == class_label)
        if pool.size == 0:
            raise ValueError(
                f"train labels for source {source!r} must contain class {class_label}"
            )
        parts.append(rng.choice(pool, size=pool.size, replace=True))
    return np.concatenate(parts).astype(np.int64, copy=False)


def _bootstrap_provenance_payload(
    *,
    labels: np.ndarray,
    indices: np.ndarray,
    sample_ids: Sequence[str],
    bootstrap_seed: int,
    source: str,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    boot_labels = labels[indices]
    original_counts = {
        str(class_label): int((labels == class_label).sum()) for class_label in (0, 1)
    }
    bootstrap_counts = {
        str(class_label): int((boot_labels == class_label).sum())
        for class_label in (0, 1)
    }
    if bootstrap_counts != original_counts:
        raise ValueError("bootstrap must preserve per-class train counts")
    index_list = [int(value) for value in indices.tolist()]
    canonical = json.dumps(index_list, separators=(",", ":"), sort_keys=False)
    return {
        "schema_version": 1,
        "resampling_method": STRATIFIED_TRAIN_BOOTSTRAP_METHOD,
        "bootstrap_seed": bootstrap_seed,
        "source_language": source,
        "calibration_and_test": "fixed_original_splits",
        "probe_fit_random_state": _PROBE_REFIT_RANDOM_STATE,
        "n_train": int(labels.size),
        "original_class_counts": original_counts,
        "bootstrap_class_counts": bootstrap_counts,
        "train_indices": index_list,
        "train_indices_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "sample_ids": [str(item) for item in sample_ids],
        "bootstrap_sample_ids": [str(sample_ids[index]) for index in index_list],
    }


def _write_json_atomic(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_catalog_embedding_alignment(
    paths: LayerwiseSuitePaths,
    profile_id: str,
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]],
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    *,
    languages: Sequence[str] | None = None,
) -> None:
    """Fail-closed: catálogo regenerado deve coincidir com metadata/arrays persistidos."""
    selected = tuple(languages) if languages is not None else tuple(catalogs.keys())
    for language in selected:
        if language not in catalogs:
            raise ValueError(f"catalogs lack language {language!r}")
        if language not in embeddings:
            raise ValueError(f"embeddings lack language {language!r}")
        for role in ROLE_ORDER:
            catalog = catalogs[language][role]
            if "sample_id" not in catalog.columns:
                raise ValueError(
                    f"catalog lacks sample_id for {profile_id}/{language}/{role}"
                )
            array = embeddings[language][role]
            metadata_path = paths.embedding_metadata(profile_id, language, role)
            metadata = read_embedding_metadata_json(metadata_path)
            assert_catalog_aligned_with_embedding_metadata(
                profile_id=profile_id,
                language=language,
                role=role,
                catalog_sample_ids=catalog["sample_id"].tolist(),
                array=array,
                metadata=metadata,
            )


def load_materialized_embeddings(
    paths: LayerwiseSuitePaths,
    profile_id: str,
    *,
    languages: Sequence[str] = LANGUAGE_ORDER,
) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(paths, LayerwiseSuitePaths):
        raise TypeError("paths must be a LayerwiseSuitePaths instance")
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for language in languages:
        embeddings[language] = {}
        for role in ROLE_ORDER:
            array_path = paths.embeddings(profile_id, language, role)
            if not array_path.is_file():
                raise ValueError(
                    f"missing embedding array for {profile_id}/{language}/{role}: "
                    f"{array_path}"
                )
            array = np.load(array_path, allow_pickle=False)
            if array.dtype != np.float32:
                raise ValueError(
                    f"embeddings {profile_id}/{language}/{role} must be float32"
                )
            embeddings[language][role] = array
    return embeddings


def load_catalogs_from_execution_plan(
    suite_root: Path,
    *,
    languages: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], int]:
    plan_path = suite_root / "execution_plan.json"
    if not plan_path.is_file():
        raise ValueError(f"execution_plan.json is missing under {suite_root}")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"execution_plan.json is unreadable: {plan_path}") from exc
    if not isinstance(plan, dict) or "seed" not in plan or "inputs" not in plan:
        raise ValueError("execution_plan.json lacks seed or inputs")
    suite_seed = int(plan["seed"])
    inputs = plan["inputs"]
    if not isinstance(inputs, dict):
        raise ValueError("execution_plan inputs must be an object")
    selected = tuple(languages) if languages is not None else tuple(inputs.keys())
    catalogs: dict[str, dict[str, pd.DataFrame]] = {}
    from .config import load_config
    from .data import build_balanced_split

    for language in selected:
        if language not in inputs:
            raise ValueError(f"execution_plan lacks inputs for language {language!r}")
        entry = inputs[language]
        config_path = Path(str(entry["config_path"])).expanduser()
        manifest_path = Path(str(entry["manifest_path"])).expanduser()
        if not config_path.is_file():
            raise ValueError(f"config_path is missing for {language}: {config_path}")
        if not manifest_path.is_file():
            raise ValueError(
                f"manifest_path is missing for {language}: {manifest_path}"
            )
        cfg = load_config(config_path)
        cfg.data.manifest_path = str(manifest_path)
        catalogs[language] = {}
        for role in ROLE_ORDER:
            split = {
                "train": cfg.data.train_split,
                "calibration": cfg.data.calibration_split,
                "test": cfg.data.eval_split,
            }[role]
            quota = {
                "train": cfg.data.n_train_per_class,
                "calibration": cfg.data.n_calibration_per_class,
                "test": cfg.data.n_test_per_class,
            }[role]
            _audios, _srs, labels, provenance = build_balanced_split(
                cfg.data.dataset_id,
                split,
                quota,
                loader=cfg.data.loader,
                seed=suite_seed,
                dataset_kind=cfg.data.dataset_kind,
                manifest_path=cfg.data.manifest_path,
            )
            frame = pd.DataFrame(provenance)
            frame["label"] = np.asarray(labels, dtype=np.int64)
            catalogs[language][role] = frame
    return catalogs, suite_seed


def _metric_values(metrics: dict) -> tuple[float, float]:
    try:
        roc_auc = float(metrics["threshold_free"]["roc_auc"])
        mcc = float(metrics["fixed_threshold"]["mcc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metrics function returned an incompatible artifact") from exc
    if not np.isfinite(roc_auc) or not np.isfinite(mcc):
        raise ValueError("roc_auc and mcc must be finite for probe stability")
    return roc_auc, mcc


def evaluate_probe_performance_for_seed(
    profile_id: str,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    layers: tuple[int, ...],
    bootstrap_seed: int,
    paths: LayerwiseSuitePaths | None = None,
    fit_head_fn: Callable = fit_head,
    score_head_fn: Callable = score_head,
    calibrate_threshold_fn: Callable = calibrate_threshold,
    metrics_fn: Callable = _default_metrics,
) -> pd.DataFrame:
    """Re-fit probes após bootstrap estratificado do treino; calibração/teste fixos."""
    layers = _validate_layers(layers)
    validate_bootstrap_seeds((bootstrap_seed,))
    data = _validate_inputs(embeddings, catalogs, layers)
    rows: list[dict] = []
    bootstrap_cache: dict[str, np.ndarray] = {}
    for layer in layers:
        for source in _PROBE_LANGUAGES:
            train, train_catalog, train_labels = data[source]["train"]
            if source not in bootstrap_cache:
                indices = stratified_train_bootstrap_indices(
                    train_labels,
                    bootstrap_seed=bootstrap_seed,
                    source=source,
                )
                bootstrap_cache[source] = indices
                if paths is not None:
                    payload = _bootstrap_provenance_payload(
                        labels=train_labels,
                        indices=indices,
                        sample_ids=train_catalog["sample_id"].tolist(),
                        bootstrap_seed=bootstrap_seed,
                        source=source,
                    )
                    _write_json_atomic(
                        payload,
                        paths.probe_stability_train_bootstrap(
                            profile_id, bootstrap_seed, source
                        ),
                    )
            train_indices = bootstrap_cache[source]
            boot_train_x = train[train_indices, layer, :]
            boot_train_y = train_labels[train_indices]
            calibration, _calibration_catalog, calibration_labels = data[source][
                "calibration"
            ]
            head = fit_head_fn(
                boot_train_x,
                boot_train_y,
                head="logistic",
                seed=_PROBE_REFIT_RANDOM_STATE,
            )
            calibration_scores = _validate_scores(
                score_head_fn(head, calibration[:, layer, :]),
                expected_rows=len(calibration),
                context=f"{source}/calibration/layer_{layer:02d}",
            )
            calibration_artifact = calibrate_threshold_fn(
                calibration_scores,
                calibration_labels,
                f"{source}/calibration/layer_{layer:02d}",
                "ad",
            )
            threshold = float(calibration_artifact["threshold"])
            for target in _PROBE_LANGUAGES:
                target_embeddings, _target_catalog, target_labels = data[target]["test"]
                scores = _validate_scores(
                    score_head_fn(head, target_embeddings[:, layer, :]),
                    expected_rows=len(target_embeddings),
                    context=f"{source}->{target}/layer_{layer:02d}",
                )
                condition = (
                    "cross_corpus_shift" if source != target else "diagonal"
                )
                metrics = metrics_fn(
                    scores,
                    target_labels,
                    threshold,
                    condition=condition,
                    source=source,
                    target=target,
                )
                roc_auc, mcc = _metric_values(metrics)
                rows.append(
                    {
                        "seed": bootstrap_seed,
                        "profile": profile_id,
                        "layer": layer,
                        "source": source,
                        "target": target,
                        "n": int(len(target_labels)),
                        "threshold": threshold,
                        "roc_auc": roc_auc,
                        "mcc": mcc,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def aggregate_probe_stability(per_seed: pd.DataFrame) -> pd.DataFrame:
    required = {*_SUMMARY_KEYS, "seed", "roc_auc", "mcc"}
    if not required.issubset(per_seed.columns):
        missing = sorted(required - set(per_seed.columns))
        raise ValueError(f"per-seed frame lacks columns: {missing}")
    grouped = per_seed.groupby(list(_SUMMARY_KEYS), sort=True)
    summary = grouped.agg(
        n_seeds=("seed", "count"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", _aggregate_std),
        roc_auc_min=("roc_auc", "min"),
        roc_auc_max=("roc_auc", "max"),
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", _aggregate_std),
        mcc_min=("mcc", "min"),
        mcc_max=("mcc", "max"),
    ).reset_index()
    for column in ("roc_auc_std", "mcc_std"):
        summary[column] = summary[column].fillna(0.0).astype(float)
    column_order = [
        "profile",
        "layer",
        "source",
        "target",
        "n_seeds",
        "roc_auc_mean",
        "roc_auc_std",
        "roc_auc_min",
        "roc_auc_max",
        "mcc_mean",
        "mcc_std",
        "mcc_min",
        "mcc_max",
    ]
    return summary[column_order]


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_probe_stability_manifest(
    paths: LayerwiseSuitePaths,
    profile_id: str,
    *,
    seeds: Sequence[int],
    suite_seed: int,
    layers: Sequence[int],
) -> None:
    payload = {
        "schema_version": 2,
        "profile": profile_id,
        "suite_data_seed": suite_seed,
        "bootstrap_seeds": list(seeds),
        "layers": list(layers),
        "resampling_method": STRATIFIED_TRAIN_BOOTSTRAP_METHOD,
        "scope": _PROBE_STABILITY_SCOPE_PT,
        "disclaimer": _PROBE_STABILITY_DISCLAIMER,
        "does_not_measure": [
            "xai_attribution_stability",
            "attnlrp_recomputation",
            "embedding_recomputation",
            "end_to_end_suite_seed_stability",
        ],
        "probe_fit_random_state": _PROBE_REFIT_RANDOM_STATE,
        "probe_fit_random_state_note": (
            "Held fixed; sklearn logistic default solver is deterministic."
        ),
    }
    manifest_path = paths.probe_stability_manifest(profile_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_probe_stability_for_profile(
    suite_root: Path,
    profile_id: str,
    *,
    layers: tuple[int, ...] = LAYER_ORDER,
    seeds: Sequence[int] = DEFAULT_PROBE_STABILITY_SEEDS,
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    embeddings: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    verify_embedding_alignment: bool = True,
    fit_head_fn: Callable = fit_head,
    score_head_fn: Callable = score_head,
) -> pd.DataFrame:
    validated_seeds = validate_bootstrap_seeds(seeds)
    paths = LayerwiseSuitePaths(suite_root)
    if catalogs is None:
        catalogs, suite_seed = load_catalogs_from_execution_plan(suite_root)
    else:
        plan_path = suite_root / "execution_plan.json"
        suite_seed = (
            int(json.loads(plan_path.read_text(encoding="utf-8"))["seed"])
            if plan_path.is_file()
            else 42
        )
    if embeddings is None:
        embeddings = load_materialized_embeddings(
            paths, profile_id, languages=tuple(catalogs.keys())
        )
    if verify_embedding_alignment:
        verify_catalog_embedding_alignment(
            paths,
            profile_id,
            catalogs,
            embeddings,
            languages=tuple(catalogs.keys()),
        )
    seed_frames: list[pd.DataFrame] = []
    for bootstrap_seed in validated_seeds:
        frame = evaluate_probe_performance_for_seed(
            profile_id,
            embeddings,
            catalogs,
            layers=layers,
            bootstrap_seed=bootstrap_seed,
            paths=paths,
            fit_head_fn=fit_head_fn,
            score_head_fn=score_head_fn,
        )
        _atomic_csv(frame, paths.probe_stability_performance(profile_id, bootstrap_seed))
        seed_frames.append(frame)
    combined = pd.concat(seed_frames, ignore_index=True)
    summary = aggregate_probe_stability(combined)
    _atomic_csv(summary, paths.probe_stability_summary(profile_id))
    _write_probe_stability_manifest(
        paths,
        profile_id,
        seeds=validated_seeds,
        suite_seed=suite_seed,
        layers=layers,
    )
    return summary


def _discover_profiles(
    paths: LayerwiseSuitePaths,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    discovered = sorted(
        child.name
        for child in paths.root.iterdir()
        if child.is_dir()
        and (child / "embeddings").is_dir()
        and child.name not in {"aggregates", ".stage-artifacts", ".state"}
    )
    if not discovered:
        raise ValueError(f"no profile directories with embeddings under {paths.root}")
    return tuple(discovered)


def run_probe_stability(
    suite_root: Path | str,
    profiles: Sequence[str] | None = None,
    *,
    layers: tuple[int, ...] = LAYER_ORDER,
    seeds: Sequence[int] = DEFAULT_PROBE_STABILITY_SEEDS,
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
    verify_embedding_alignment: bool = True,
    fit_head_fn: Callable = fit_head,
    score_head_fn: Callable = score_head,
) -> dict[str, object]:
    validated_seeds = validate_bootstrap_seeds(seeds)
    root = Path(suite_root)
    paths = LayerwiseSuitePaths(root)
    selected = _discover_profiles(paths, profiles)
    summaries: dict[str, pd.DataFrame] = {}
    for profile_id in selected:
        summaries[profile_id] = run_probe_stability_for_profile(
            root,
            profile_id,
            layers=layers,
            seeds=validated_seeds,
            catalogs=catalogs,
            verify_embedding_alignment=verify_embedding_alignment,
            fit_head_fn=fit_head_fn,
            score_head_fn=score_head_fn,
        )
    return {
        "suite_root": str(root),
        "profiles": list(selected),
        "seeds": list(validated_seeds),
        "layers": list(layers),
        "resampling_method": STRATIFIED_TRAIN_BOOTSTRAP_METHOD,
        "scope": _PROBE_STABILITY_SCOPE_PT,
        "disclaimer": _PROBE_STABILITY_DISCLAIMER,
        "summaries": {
            key: frame.to_dict(orient="records") for key, frame in summaries.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sensibilidade do probe linear à reamostragem estratificada do treino "
            "(seeds de bootstrap) usando embeddings materializados; sem recomputar "
            "encoder/XAI."
        )
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "Completed encoder-suite output root (repeat for HuBERT/WavLM/Wav2Vec2 runs)."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Profile IDs to evaluate (default: auto-detect under each suite root).",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=list(LAYER_ORDER),
        help="Transformer layers to re-fit (default: 1..12).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_STABILITY_SEEDS),
        help=(
            "Bootstrap seeds for stratified train resampling (default: 42 43 44)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layers = tuple(_validate_layers(tuple(args.layers)))
    validated_seeds = validate_bootstrap_seeds(tuple(args.seeds))
    results = []
    for suite_root in args.suite_root:
        results.append(
            run_probe_stability(
                suite_root,
                profiles=args.profiles,
                layers=layers,
                seeds=validated_seeds,
                verify_embedding_alignment=True,
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
