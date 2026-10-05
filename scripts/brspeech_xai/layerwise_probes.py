"""Probes lineares por camada e matriz trilíngue sem leakage de alvo."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .adaptation import fit_head, score_head
from .layerwise_paths import LANGUAGES, LayerwiseSuitePaths
from .metrics import calibrate_threshold, evaluate_at_threshold


LANGUAGE_ORDER = ("eng", "por", "zho")
ROLE_ORDER = ("train", "calibration", "test")
_REQUIRED_CATALOG_COLUMNS = frozenset({"sample_id", "label"})


def _validate_layers(layers: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("layers must be a non-empty tuple")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= 12
        for layer in layers
    ):
        raise ValueError("each layer must be an integer from 1 through 12")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicates")
    return layers


def _validate_catalog(
    catalog: pd.DataFrame,
    *,
    language: str,
    role: str,
    expected_rows: int,
) -> np.ndarray:
    location = f"{language}/{role}"
    if not isinstance(catalog, pd.DataFrame):
        raise ValueError(f"catalog {location} must be a pandas DataFrame")
    if not _REQUIRED_CATALOG_COLUMNS.issubset(catalog.columns):
        raise ValueError(
            f"catalog {location} must contain columns "
            f"{sorted(_REQUIRED_CATALOG_COLUMNS)}"
        )
    if len(catalog) != expected_rows:
        raise ValueError(
            f"catalog {location} length does not match embedding rows: "
            f"{len(catalog)} vs {expected_rows}"
        )

    sample_ids = catalog["sample_id"].tolist()
    if any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in sample_ids
    ):
        raise ValueError(f"catalog {location} sample_id values must be non-empty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"catalog {location} sample_id values must be unique")

    labels = catalog["label"].to_numpy()
    if labels.ndim != 1 or not np.all(pd.notna(labels)):
        raise ValueError(f"catalog {location} labels must be finite binary values")
    try:
        finite = np.isfinite(labels.astype(np.float64))
    except (TypeError, ValueError):
        finite = np.zeros(len(labels), dtype=bool)
    if not finite.all() or not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(f"catalog {location} labels must be binary")
    if role in {"train", "calibration"} and set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"catalog {location} labels must contain both classes")
    return labels.astype(np.int64, copy=False)


def _validate_inputs(
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]],
    layers: tuple[int, ...],
) -> dict[str, dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray]]]:
    if not isinstance(embeddings, Mapping) or set(embeddings) != set(LANGUAGES):
        raise ValueError(f"embedding languages must be exactly {sorted(LANGUAGES)}")
    if not isinstance(catalogs, Mapping) or set(catalogs) != set(LANGUAGES):
        raise ValueError(f"catalog languages must be exactly {sorted(LANGUAGES)}")

    validated: dict[
        str, dict[str, tuple[np.ndarray, pd.DataFrame, np.ndarray]]
    ] = {}
    hidden_size: int | None = None
    for language in LANGUAGE_ORDER:
        language_embeddings = embeddings[language]
        language_catalogs = catalogs[language]
        if (
            not isinstance(language_embeddings, Mapping)
            or set(language_embeddings) != set(ROLE_ORDER)
            or not isinstance(language_catalogs, Mapping)
            or set(language_catalogs) != set(ROLE_ORDER)
        ):
            raise ValueError(
                f"embedding and catalog roles for {language} must be exactly "
                f"{list(ROLE_ORDER)}"
            )
        validated[language] = {}
        for role in ROLE_ORDER:
            array = language_embeddings[role]
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 3
                or array.shape[0] == 0
                or array.shape[2] == 0
            ):
                raise ValueError(
                    f"embeddings {language}/{role} must have shape [N, L, H]"
                )
            if array.dtype != np.float32:
                raise ValueError(f"embeddings {language}/{role} must have dtype float32")
            if not np.isfinite(array).all():
                raise ValueError(
                    f"embeddings {language}/{role} must contain only finite values"
                )
            if max(layers) >= array.shape[1]:
                raise ValueError(
                    f"requested layer is outside embeddings {language}/{role} "
                    f"with {array.shape[1]} hidden states"
                )
            if hidden_size is None:
                hidden_size = int(array.shape[2])
            elif array.shape[2] != hidden_size:
                raise ValueError("all embeddings must share the same hidden size")

            catalog = language_catalogs[role]
            labels = _validate_catalog(
                catalog,
                language=language,
                role=role,
                expected_rows=array.shape[0],
            )
            validated[language][role] = (array, catalog, labels)
    return validated


def _json_dump(payload: dict, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _publish_directory(destination: Path, writer: Callable[[Path], None]) -> None:
    """Build a complete directory beside its destination, then swap it into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    backup = destination.with_name(f".{destination.name}.{token}.bak")
    temporary.mkdir()
    moved_old = False
    try:
        writer(temporary)
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        try:
            os.replace(temporary, destination)
        except BaseException:
            if moved_old:
                os.replace(backup, destination)
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _threshold_payload(source: str, layer: int, calibration: dict) -> dict:
    return {
        "schema_version": 1,
        "source_language": source,
        "layer": layer,
        "calibration_source": f"{source}/calibration/layer_{layer:02d}",
        "calibration_fallback_train": False,
        "detectors": {"ad": calibration},
    }


def _persist_probe(
    paths: LayerwiseSuitePaths,
    profile_id: str,
    layer: int,
    source: str,
    head: object,
    calibration: dict,
) -> object:
    probe_path = paths.probe(profile_id, layer, source)
    threshold_path = paths.thresholds(profile_id, layer, source)
    probe_dir = probe_path.parent

    def write(directory: Path) -> None:
        joblib.dump(head, directory / probe_path.name)
        _json_dump(
            _threshold_payload(source, layer, calibration),
            directory / threshold_path.name,
        )

    _publish_directory(probe_dir, write)
    try:
        return joblib.load(probe_path)
    except Exception as exc:
        raise ValueError(
            f"incompatible persisted head artifact for {source}/layer_{layer:02d}"
        ) from exc


def _default_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    condition: str,
    source: str,
    target: str,
) -> dict:
    if set(np.unique(labels).tolist()) == {0, 1}:
        return evaluate_at_threshold(
            scores,
            labels,
            threshold,
            condition=condition,
            source=source,
            target=target,
        )
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "condition": condition,
        "source": source,
        "target": target,
        "threshold_free": {
            "roc_auc": None,
            "average_precision": None,
            "eer_diagnostic": None,
        },
        "fixed_threshold": {
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
        },
    }


def _validate_scores(scores: object, *, expected_rows: int, context: str) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or len(array) != expected_rows:
        raise ValueError(
            f"scores for {context} must be one-dimensional with {expected_rows} rows"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"scores for {context} must contain only finite values")
    return array


def _metric_summary(metrics: dict) -> tuple[float, float]:
    try:
        auc_value = metrics["threshold_free"]["roc_auc"]
        accuracy = float(metrics["fixed_threshold"]["accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metrics function returned an incompatible artifact") from exc
    auc = float("nan") if auc_value is None else float(auc_value)
    if (not np.isnan(auc) and not np.isfinite(auc)) or not np.isfinite(accuracy):
        raise ValueError("metrics function returned non-finite required metrics")
    return auc, accuracy


def run_layerwise_probe_matrix(
    profile_id: str,
    embeddings: Mapping[str, Mapping[str, np.ndarray]],
    catalogs: Mapping[str, Mapping[str, pd.DataFrame]],
    paths: LayerwiseSuitePaths,
    *,
    layers: tuple[int, ...] = tuple(range(1, 13)),
    seed: int = 42,
    fit_head_fn: Callable = fit_head,
    score_head_fn: Callable = score_head,
    calibrate_threshold_fn: Callable = calibrate_threshold,
    metrics_fn: Callable = _default_metrics,
) -> pd.DataFrame:
    """Fit once per ``(layer, source)`` and evaluate all three target tests."""
    if not isinstance(paths, LayerwiseSuitePaths):
        raise TypeError("paths must be a LayerwiseSuitePaths instance")
    layers = _validate_layers(layers)
    paths.profile(profile_id)
    data = _validate_inputs(embeddings, catalogs, layers)

    rows: list[dict] = []
    for layer in layers:
        for source in LANGUAGE_ORDER:
            train, _train_catalog, train_labels = data[source]["train"]
            calibration, _calibration_catalog, calibration_labels = data[source][
                "calibration"
            ]
            head = fit_head_fn(
                train[:, layer, :],
                train_labels,
                head="logistic",
                seed=seed,
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
            if not isinstance(calibration_artifact, dict):
                raise ValueError("calibration function returned an incompatible artifact")
            try:
                threshold = float(calibration_artifact["threshold"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "calibration artifact must contain a numeric threshold"
                ) from exc
            if not np.isfinite(threshold):
                raise ValueError("calibration threshold must be finite")
            persisted_head = _persist_probe(
                paths,
                profile_id,
                layer,
                source,
                head,
                calibration_artifact,
            )

            for target in LANGUAGE_ORDER:
                target_embeddings, target_catalog, target_labels = data[target]["test"]
                scores = _validate_scores(
                    score_head_fn(persisted_head, target_embeddings[:, layer, :]),
                    expected_rows=len(target_embeddings),
                    context=f"{source}->{target}/layer_{layer:02d}",
                )
                corpus_shift = source != target
                condition = "cross_corpus_shift" if corpus_shift else "diagonal"
                metrics = metrics_fn(
                    scores,
                    target_labels,
                    threshold,
                    condition=condition,
                    source=source,
                    target=target,
                )
                if not isinstance(metrics, dict):
                    raise ValueError("metrics function returned an incompatible artifact")
                auc, accuracy = _metric_summary(metrics)
                predictions = (scores >= threshold).astype(np.int64)
                prediction_table = pd.DataFrame(
                    {
                        "sample_id": target_catalog["sample_id"].tolist(),
                        "y_true": target_labels,
                        "score": scores,
                        "prediction": predictions,
                    }
                )

                cell_dir = paths.cell(profile_id, layer, source, target)

                def write_cell(directory: Path) -> None:
                    np.save(directory / "scores.npy", scores, allow_pickle=False)
                    prediction_table.to_parquet(
                        directory / "predictions.parquet",
                        index=False,
                    )
                    _json_dump(metrics, directory / "metrics.json")

                _publish_directory(cell_dir, write_cell)
                rows.append(
                    {
                        "profile": profile_id,
                        "layer": layer,
                        "source": source,
                        "target": target,
                        "n": int(len(target_labels)),
                        "threshold": threshold,
                        "auc": auc,
                        "accuracy": accuracy,
                        "corpus_shift": corpus_shift,
                    }
                )

    performance = pd.DataFrame(rows)
    performance_path = paths.profile(profile_id) / "layerwise_performance.csv"
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = performance_path.with_name(
        f".{performance_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        performance.to_csv(temporary, index=False)
        os.replace(temporary, performance_path)
    finally:
        temporary.unlink(missing_ok=True)
    return performance
