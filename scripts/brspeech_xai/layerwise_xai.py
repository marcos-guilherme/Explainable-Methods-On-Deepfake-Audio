"""Coorte fixa e execução AttnLRP -> DFT/STDFT por célula layer-wise."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from .adaptation import score_head
from .attnlrp import ensure_ssl_encoder_attnlrp, patch_ssl_encoder_for_attnlrp
from .dft_lrp import (
    aggregate_to_bands,
    rfft_frequencies,
    stdft_lrp,
    time_to_freq_relevance,
)
from .layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
    sample_id_catalog_hash,
)
from .lrp_detector import (
    ConservationDiagnostics,
    SSLDetectorAD,
    conservation_certificate,
    port_logistic_head,
    relevance_for_clip,
    verify_score_equivalence,
)
from .xai_registry import get_encoder_spec


resolve_active_xai_generation = resolve_active_generation


_CATALOG_COLUMNS = frozenset({"sample_id", "label", "processed_path"})
_PREDICTION_COLUMNS = frozenset({"sample_id", "y_true", "score", "prediction"})
_SCORE_RECOMPUTE_RTOL = 1e-5
_SCORE_RECOMPUTE_ATOL = 7e-3


def _validated_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(catalog, pd.DataFrame):
        raise ValueError("catalog must be a pandas DataFrame")
    missing = _CATALOG_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError(f"catalog is missing required columns: {sorted(missing)}")
    ids = catalog["sample_id"].tolist()
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("sample_id values must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    labels = catalog["label"].to_numpy()
    if labels.dtype.kind not in {"i", "u"} or not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("labels must be binary integers")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("catalog must contain both classes")
    paths = catalog["processed_path"].tolist()
    if any(not isinstance(value, (str, os.PathLike)) or not str(value) for value in paths):
        raise ValueError("processed_path values must be non-empty paths")
    return catalog


def select_fixed_cohort(
    catalog: pd.DataFrame,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Select equal class quotas using only ``seed|sample_id`` SHA-256 order."""
    catalog = _validated_catalog(catalog)
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise ValueError("per_class must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    selected: list[pd.DataFrame] = []
    for label in (0, 1):
        class_rows = catalog.loc[catalog["label"] == label].copy()
        if len(class_rows) < per_class:
            raise ValueError(
                f"class {label} does not have enough samples for per_class={per_class}"
            )
        class_rows["_cohort_hash"] = class_rows["sample_id"].map(
            lambda sample_id: hashlib.sha256(
                f"{int(seed)}|{sample_id}".encode("utf-8")
            ).hexdigest()
        )
        selected.append(
            class_rows.sort_values(
                ["_cohort_hash", "sample_id"], kind="mergesort"
            ).iloc[:per_class]
        )
    return pd.concat(selected, ignore_index=True).drop(columns="_cohort_hash")


def _validate_edges(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or values[0] < 0
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("frequency_edges must be finite, non-negative, and increasing")
    return values


def _load_threshold(path: Path, *, source: str, layer: int) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_threshold = payload["detectors"]["ad"]["threshold"]
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raise TypeError("threshold must be numeric")
        threshold = float(raw_threshold)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid threshold artifact: {path}") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or type(payload.get("source_language")) is not str
        or payload["source_language"] != source
        or type(payload.get("layer")) is not int
        or payload["layer"] != layer
    ):
        raise ValueError(
            f"threshold schema/source/layer mismatch for {source}/layer_{layer:02d}"
        )
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite: {path}")
    return threshold


def _strict_binary_column(frame: pd.DataFrame, column: str, *, context: str) -> np.ndarray:
    values = frame[column].to_numpy()
    if values.dtype.kind not in {"i", "u"} or not set(values.tolist()).issubset({0, 1}):
        raise ValueError(f"{column} must contain binary integer values: {context}")
    return values.astype(np.int64, copy=False)


def _read_prediction_cell(
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
    catalog: pd.DataFrame,
    cohort_ids: tuple[str, ...],
) -> pd.DataFrame:
    probe_path = paths.probe(profile, layer, source)
    threshold_path = paths.thresholds(profile, layer, source)
    if not probe_path.is_file():
        raise ValueError(f"probe artifact is missing: {probe_path}")
    if not threshold_path.is_file():
        raise ValueError(f"threshold artifact is missing: {threshold_path}")
    cell_dir = paths.cell(profile, layer, source, target)
    prediction_path = cell_dir / "predictions.parquet"
    scores_path = cell_dir / "scores.npy"
    if not prediction_path.is_file() or not scores_path.is_file():
        raise ValueError(f"cell scores/predictions are missing: {cell_dir}")
    try:
        predictions = pd.read_parquet(prediction_path)
        scores = np.load(scores_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"cell scores/predictions are unreadable: {cell_dir}") from exc
    if not _PREDICTION_COLUMNS.issubset(predictions.columns):
        raise ValueError(f"cell prediction columns are incompatible: {cell_dir}")
    expected_ids = tuple(catalog["sample_id"])
    actual_ids = tuple(predictions["sample_id"])
    if actual_ids != expected_ids:
        raise ValueError(f"cell sample_id catalog is not aligned: {cell_dir}")
    if scores.ndim != 1 or len(scores) != len(predictions):
        raise ValueError(f"cell scores have incompatible shape: {cell_dir}")
    prediction_scores = predictions["score"].to_numpy()
    if (
        prediction_scores.dtype.kind not in {"f", "i", "u"}
        or not np.isfinite(scores).all()
        or not np.isfinite(prediction_scores).all()
    ):
        raise ValueError(f"cell scores/predictions must be finite: {cell_dir}")
    y_true = _strict_binary_column(predictions, "y_true", context=str(cell_dir))
    prediction = _strict_binary_column(predictions, "prediction", context=str(cell_dir))
    if not np.allclose(scores, prediction_scores.astype(np.float64)):
        raise ValueError(f"cell scores.npy is not aligned with predictions: {cell_dir}")
    if not np.array_equal(
        y_true,
        catalog["label"].to_numpy(),
    ):
        raise ValueError(f"cell labels are not aligned with target catalog: {cell_dir}")
    if prediction.shape != y_true.shape:
        raise ValueError(f"cell predictions have incompatible shape: {cell_dir}")
    indexed = predictions.set_index("sample_id", drop=False)
    try:
        selected = indexed.loc[list(cohort_ids)].reset_index(drop=True)
    except KeyError as exc:
        raise ValueError(f"cohort IDs are absent from cell: {cell_dir}") from exc
    if tuple(selected["sample_id"]) != cohort_ids:
        raise ValueError(f"cohort IDs diverged in cell: {cell_dir}")
    return selected


def _validate_embeddings(
    embeddings: Mapping[str, Mapping[str, Mapping[str, object]]],
    profiles: tuple[str, ...],
    targets: tuple[str, ...],
    catalogs: Mapping[str, pd.DataFrame],
    layers: tuple[int, ...],
) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(embeddings, Mapping):
        raise ValueError("embeddings must be mapped by profile and target")
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for profile in profiles:
        if profile not in embeddings or not isinstance(embeddings[profile], Mapping):
            raise ValueError(f"embeddings are missing profile {profile!r}")
        aligned[profile] = {}
        for target in targets:
            if target not in embeddings[profile]:
                raise ValueError(f"embeddings are missing {profile}/{target}")
            artifact = embeddings[profile][target]
            if not isinstance(artifact, Mapping) or set(artifact) != {
                "values",
                "metadata",
            }:
                raise ValueError(
                    f"embeddings {profile}/{target} must carry values and cache metadata"
                )
            array = artifact["values"]
            metadata = artifact["metadata"]
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"embedding cache metadata is invalid for {profile}/{target}"
                )
            sample_ids = metadata.get("sample_ids")
            if (
                not isinstance(array, np.ndarray)
                or array.ndim != 3
                or array.dtype != np.float32
                or len(array) != len(catalogs[target])
                or max(layers) >= array.shape[1]
                or array.shape[2] == 0
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    f"embeddings {profile}/{target} must be finite float32 [N,L,H] "
                    "and contain every requested layer"
                )
            if (
                isinstance(sample_ids, (str, bytes))
                or not isinstance(sample_ids, Sequence)
                or len(sample_ids) != len(array)
                or any(
                    not isinstance(sample_id, str) or not sample_id
                    for sample_id in sample_ids
                )
                or len(set(sample_ids)) != len(sample_ids)
            ):
                raise ValueError(
                    f"embedding metadata sample_ids are invalid for {profile}/{target}"
                )
            if (
                metadata.get("schema_version") != 1
                or metadata.get("profile") != profile
                or metadata.get("language") != target
                or metadata.get("role") != "test"
                or metadata.get("shape") != list(array.shape)
                or metadata.get("dtype") != "float32"
                or metadata.get("sample_ids_sha256")
                != sample_id_catalog_hash(sample_ids)
            ):
                raise ValueError(
                    f"embedding cache metadata does not match values/context for "
                    f"{profile}/{target}"
                )
            expected_ids = catalogs[target]["sample_id"].tolist()
            if set(sample_ids) != set(expected_ids):
                raise ValueError(
                    f"embedding metadata sample_ids do not match catalog for {profile}/{target}"
                )
            positions = {sample_id: index for index, sample_id in enumerate(sample_ids)}
            aligned[profile][target] = array[
                np.asarray([positions[sample_id] for sample_id in expected_ids])
            ]
    return aligned


def _quadrant(y_true: int, prediction: int) -> str:
    return ("TP" if prediction else "FN") if y_true == 1 else (
        "FP" if prediction else "TN"
    )


def _summary_rows(
    samples: pd.DataFrame,
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    grouping: str,
    group_column: str | None,
) -> list[dict]:
    rows: list[dict] = []
    groups = [("all", samples)] if group_column is None else list(samples.groupby(group_column))
    for group_value, frame in groups:
        signed = np.stack(frame["band_signed"].to_numpy())
        absolute = np.stack(frame["band_abs_normalized"].to_numpy())
        for measure, values in (("signed", signed), ("absolute_normalized", absolute)):
            n = len(values)
            mean = values.mean(axis=0)
            sem = (
                values.std(axis=0, ddof=1) / math.sqrt(n)
                if n > 1
                else np.zeros(values.shape[1])
            )
            for band, (center, error) in enumerate(zip(mean, sem), start=1):
                rows.append(
                    {
                        "profile": profile,
                        "layer": layer,
                        "source": source,
                        "target": target,
                        "grouping": grouping,
                        "group": group_value,
                        "measure": measure,
                        "band": band,
                        "n": n,
                        "mean": float(center),
                        "ci_low": float(center - 1.96 * error),
                        "ci_high": float(center + 1.96 * error),
                    }
                )
    return rows


def _prepare_cell(
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    cohort: pd.DataFrame,
    target_catalog: pd.DataFrame,
    target_embeddings: np.ndarray,
    predictions: pd.DataFrame,
    paths: LayerwiseSuitePaths,
    head_loader: Callable,
    threshold_loader: Callable,
    port_head_fn: Callable,
    verify_equivalence_fn: Callable,
    score_head_fn: Callable,
) -> dict:
    threshold = threshold_loader(
        paths.thresholds(profile, layer, source), source=source, layer=layer
    )
    head = head_loader(paths.probe(profile, layer, source))
    weight, bias = port_head_fn(head)
    catalog_index = {
        sample_id: index for index, sample_id in enumerate(target_catalog["sample_id"])
    }
    cohort_indices = np.asarray(
        [catalog_index[sample_id] for sample_id in cohort["sample_id"]]
    )
    selected_embeddings = target_embeddings[cohort_indices, layer, :]
    equivalence_error = verify_equivalence_fn(
        head, selected_embeddings, weight, bias
    )
    if not np.isfinite(equivalence_error):
        raise ValueError("score equivalence error must be finite")
    expected_scores = np.asarray(
        score_head_fn(head, selected_embeddings), dtype=np.float64
    )
    actual_scores = predictions["score"].to_numpy(dtype=np.float64)
    if (
        expected_scores.shape != actual_scores.shape
        or not np.isfinite(expected_scores).all()
        or not np.allclose(expected_scores, actual_scores, rtol=1e-5, atol=1e-6)
    ):
        raise ValueError(
            f"preflight cell score diverges from persisted head for "
            f"{profile}/layer_{layer:02d}/{source}->{target}"
        )
    return {
        "threshold": threshold,
        "weight": weight,
        "bias": bias,
        "equivalence_error": float(equivalence_error),
    }


def run_layer_xai_cell(
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    cohort: pd.DataFrame,
    target_catalog: pd.DataFrame,
    target_embeddings: np.ndarray,
    predictions: pd.DataFrame,
    paths: LayerwiseSuitePaths,
    encoder: object,
    processor: object,
    prepared: Mapping[str, object],
    frequency_edges: np.ndarray,
    stdft_sample_ids: frozenset[str],
    sample_rate: int = 16000,
    conservation_tolerance: float = 1e-3,
    device: str = "cpu",
    audio_loader: Callable = None,
    detector_factory: Callable = SSLDetectorAD,
    relevance_fn: Callable = relevance_for_clip,
    temporal_certificate_fn: Callable = conservation_certificate,
    dft_fn: Callable = time_to_freq_relevance,
    aggregate_fn: Callable = aggregate_to_bands,
    stdft_fn: Callable = stdft_lrp,
) -> dict:
    """Execute one preflighted profile/layer/source/target XAI cell."""
    if audio_loader is None:
        raise ValueError("audio_loader is required")
    if not np.isfinite(conservation_tolerance) or conservation_tolerance <= 0:
        raise ValueError("conservation_tolerance must be finite and positive")
    threshold = float(prepared["threshold"])
    weight = np.asarray(prepared["weight"])
    bias = float(prepared["bias"])
    equivalence_error = float(prepared["equivalence_error"])
    model = detector_factory(encoder, layer, weight, bias)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("detector_factory must return a torch.nn.Module")
    encoder_dtype = next(
        (
            parameter.dtype
            for parameter in encoder.parameters()
            if parameter.is_floating_point()
        ),
        torch.float32,
    )
    model = model.to(device=device, dtype=encoder_dtype)
    model.eval()
    expected_device = torch.device(device)
    for name in ("w", "b"):
        buffer = getattr(model, name, None)
        if not isinstance(buffer, torch.Tensor):
            raise ValueError(f"detector must expose tensor buffer {name}")
        if (
            buffer.device.type != expected_device.type
            or (
                expected_device.index is not None
                and buffer.device.index != expected_device.index
            )
        ):
            raise ValueError(f"detector buffer {name} is not on device {device}")
    prediction_index = predictions.set_index("sample_id")
    sample_rows: list[dict] = []
    stdft_payloads: dict[str, tuple[np.ndarray, ...]] = {}
    loaded_wavs: dict[str, np.ndarray] = {}
    for row in cohort.itertuples(index=False):
        sample_id = row.sample_id
        processed_path = Path(row.processed_path)
        if not processed_path.is_file():
            raise ValueError(f"processed_path is invalid for {sample_id}: {processed_path}")
        loaded = audio_loader(processed_path)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("audio_loader must return (waveform, sample_rate)")
        wav, actual_rate = loaded
        wav = np.asarray(wav)
        if (
            wav.ndim != 1
            or wav.size == 0
            or not np.isfinite(wav).all()
            or actual_rate != sample_rate
        ):
            raise ValueError(
                f"canonical audio for {sample_id} must be finite mono at {sample_rate} Hz"
            )
        loaded_wavs[sample_id] = wav

    validation_sample_id = str(cohort.iloc[0]["sample_id"])
    conservation = temporal_certificate_fn(
        model,
        processor,
        [loaded_wavs[validation_sample_id]],
        device,
        bias,
        conservation_tolerance,
    )
    if not isinstance(conservation, ConservationDiagnostics):
        raise ValueError(
            "bias-zeroed model/rule conservation returned invalid diagnostics"
        )
    diagnostic_values = (
        conservation.evidence,
        conservation.relevance_sum,
        conservation.absolute_error,
        conservation.relative_error,
        conservation.absolute_tolerance,
        conservation.relative_tolerance,
        conservation.bound,
    )
    if (
        not all(np.isfinite(value) for value in diagnostic_values)
        or conservation.absolute_error < 0
        or conservation.relative_error < 0
        or conservation.absolute_tolerance < 0
        or conservation.relative_tolerance <= 0
        or conservation.bound <= 0
        or conservation.relative_tolerance != conservation_tolerance
    ):
        raise ValueError(
            "bias-zeroed model/rule conservation returned invalid diagnostics"
        )
    bias_zeroed_validation_residual = float(conservation.relative_error)
    if not conservation.accepted:
        raise ValueError(
            "bias-zeroed model/rule conservation exceeded tolerance: "
            f"absolute_error={conservation.absolute_error}, "
            f"bound={conservation.bound}, "
            f"relative_error={conservation.relative_error}"
        )

    for row in cohort.itertuples(index=False):
        sample_id = row.sample_id
        wav = loaded_wavs[sample_id]
        x_time, r_time, logit = relevance_fn(model, processor, wav, device)
        x_time = np.asarray(x_time, dtype=np.float64)
        r_time = np.asarray(r_time, dtype=np.float64)
        if (
            x_time.ndim != 1
            or r_time.shape != x_time.shape
            or x_time.size == 0
            or not np.isfinite(x_time).all()
            or not np.isfinite(r_time).all()
            or not np.isfinite(logit)
        ):
            raise ValueError(f"non-finite or incompatible AttnLRP output for {sample_id}")
        bias_inclusive_gap = abs(float(r_time.sum()) - float(logit)) / (
            abs(float(logit)) + 1e-9
        )
        if not np.isfinite(bias_inclusive_gap):
            raise ValueError(f"bias-inclusive attribution gap is non-finite for {sample_id}")
        cell_row = prediction_index.loc[sample_id]
        score = float(cell_row["score"])
        probability = float(1.0 / (1.0 + np.exp(-float(logit))))
        prediction = int(cell_row["prediction"])
        if prediction != int(score >= threshold):
            raise ValueError(f"prediction/threshold divergence for {sample_id}")
        score_recompute_absolute_error = abs(probability - score)
        # Cached embeddings are produced in inference batches, while AttnLRP
        # recomputes one clip with gradients. CUDA reduction order can cause a
        # small numerical drift, so preserve the fixed-threshold decision,
        # bound the score difference, and publish the observed error.
        if not np.isclose(
            probability,
            score,
            rtol=_SCORE_RECOMPUTE_RTOL,
            atol=_SCORE_RECOMPUTE_ATOL,
        ):
            raise ValueError(
                f"logit/score divergence for {sample_id}: {probability} vs {score}"
            )
        recomputed_prediction = int(probability >= threshold)
        recomputed_prediction_disagrees = recomputed_prediction != prediction

        r_freq = np.asarray(dft_fn(x_time, r_time), dtype=np.float64)
        expected_bins = x_time.size // 2 + 1
        if (
            r_freq.shape != (expected_bins,)
            or not np.isfinite(r_freq).all()
            or frequency_edges[-1] > sample_rate / 2
        ):
            raise ValueError(f"invalid DFT relevance or frequency edges for {sample_id}")
        denominator = abs(float(r_time.sum())) + 1e-9
        dft_residual = abs(float(r_freq.sum() - r_time.sum())) / denominator
        if not np.isfinite(dft_residual) or dft_residual > conservation_tolerance:
            raise ValueError(
                f"DFT conservation exceeded tolerance for {sample_id}: {dft_residual}"
            )
        freqs = rfft_frequencies(x_time.size, sample_rate)
        band_signed = np.asarray(
            aggregate_fn(r_freq, freqs, frequency_edges), dtype=np.float64
        )
        band_absolute = np.asarray(
            aggregate_fn(np.abs(r_freq), freqs, frequency_edges), dtype=np.float64
        )
        if (
            band_signed.shape != (frequency_edges.size - 1,)
            or not np.isfinite(band_signed).all()
            or band_absolute.shape != band_signed.shape
            or not np.isfinite(band_absolute).all()
            or np.any(band_absolute < 0)
        ):
            raise ValueError(f"invalid band relevance for {sample_id}")
        absolute_total = float(band_absolute.sum())
        band_absolute_normalized = (
            band_absolute / absolute_total
            if absolute_total > 0
            else np.zeros_like(band_absolute)
        )
        y_true = int(row.label)
        sample_rows.append(
            {
                "profile": profile,
                "layer": layer,
                "source": source,
                "target": target,
                "sample_id": sample_id,
                "y_true": y_true,
                "processed_path": str(row.processed_path),
                "score": score,
                "recomputed_score": probability,
                "score_recompute_absolute_error": score_recompute_absolute_error,
                "prediction": prediction,
                "recomputed_prediction": recomputed_prediction,
                "recomputed_prediction_disagrees": (
                    recomputed_prediction_disagrees
                ),
                "quadrant": _quadrant(y_true, prediction),
                "logit": float(logit),
                "equivalence_error": float(equivalence_error),
                "bias_inclusive_attribution_gap": bias_inclusive_gap,
                "dft_residual": dft_residual,
                "band_signed": band_signed.tolist(),
                "band_abs_normalized": band_absolute_normalized.tolist(),
            }
        )
        if sample_id in stdft_sample_ids:
            payload = tuple(
                np.asarray(value) for value in stdft_fn(x_time, r_time, sample_rate)
            )
            if len(payload) != 4 or any(not np.isfinite(value).all() for value in payload):
                raise ValueError(f"invalid STDFT relevance for {sample_id}")
            relevance_time_sum = float(np.asarray(r_time, dtype=np.float64).sum())
            relevance_tf_sum = float(
                np.asarray(payload[2], dtype=np.float64).sum()
            )
            conservation_absolute_error = abs(
                relevance_tf_sum - relevance_time_sum
            )
            conservation_relative_error = conservation_absolute_error / (
                abs(relevance_time_sum) + 1e-9
            )
            if (
                not np.isfinite(conservation_relative_error)
                or conservation_relative_error > conservation_tolerance
            ):
                raise ValueError(
                    "STDFT conservation exceeded tolerance for "
                    f"{sample_id}: {conservation_relative_error}"
                )
            stdft_payloads[sample_id] = payload + (
                np.asarray(relevance_time_sum),
                np.asarray(relevance_tf_sum),
                np.asarray(conservation_absolute_error),
                np.asarray(conservation_relative_error),
            )

    samples = pd.DataFrame(sample_rows)
    overall = pd.DataFrame(
        _summary_rows(
            samples,
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            grouping="all",
            group_column=None,
        )
    )
    byclass = pd.DataFrame(
        _summary_rows(
            samples,
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            grouping="y_true",
            group_column="y_true",
        )
        + _summary_rows(
            samples,
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            grouping="prediction",
            group_column="prediction",
        )
    )
    destination = paths.layer_xai(profile, layer, source, target)

    def write(directory: Path) -> None:
        serializable = samples.copy()
        serializable.to_parquet(directory / "sample_relevance.parquet", index=False)
        overall.to_csv(directory / "dft_band_relevance.csv", index=False)
        byclass.to_csv(directory / "dft_band_relevance_byclass.csv", index=False)
        counts = {name: int((samples["quadrant"] == name).sum()) for name in ("TP", "TN", "FP", "FN")}
        (directory / "attnlrp_conservation.json").write_text(
            json.dumps(
                {
                    "profile": profile,
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "n": len(samples),
                    "validation_kind": "bias_zeroed_model_rule_check",
                    "validation_sample_id": validation_sample_id,
                    "bias_zeroed_validation_residual": (
                        bias_zeroed_validation_residual
                    ),
                    "bias_zeroed_validation_evidence": conservation.evidence,
                    "bias_zeroed_validation_relevance_sum": (
                        conservation.relevance_sum
                    ),
                    "bias_zeroed_validation_absolute_error": (
                        conservation.absolute_error
                    ),
                    "bias_zeroed_validation_absolute_tolerance": (
                        conservation.absolute_tolerance
                    ),
                    "bias_zeroed_validation_relative_tolerance": (
                        conservation.relative_tolerance
                    ),
                    "bias_zeroed_validation_bound": conservation.bound,
                    "bias_zeroed_validation_accepted": conservation.accepted,
                    "max_bias_inclusive_attribution_gap": float(
                        samples["bias_inclusive_attribution_gap"].max()
                    ),
                    "max_score_recompute_absolute_error": float(
                        samples["score_recompute_absolute_error"].max()
                    ),
                    "recomputed_prediction_disagreement_count": int(
                        samples["recomputed_prediction_disagrees"].sum()
                    ),
                    "score_recompute_rtol": _SCORE_RECOMPUTE_RTOL,
                    "score_recompute_atol": _SCORE_RECOMPUTE_ATOL,
                    "max_dft_residual": float(samples["dft_residual"].max()),
                    "max_stdft_conservation_relative_error": max(
                        (
                            float(payload[7])
                            for payload in stdft_payloads.values()
                        ),
                        default=0.0,
                    ),
                    "tolerance": conservation_tolerance,
                    "quadrant_counts": counts,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        examples = directory / "stdft_examples"
        examples.mkdir()
        for sample_id, payload in stdft_payloads.items():
            (
                times,
                freqs,
                relevance,
                spectrum,
                relevance_time_sum,
                relevance_tf_sum,
                conservation_absolute_error,
                conservation_relative_error,
            ) = payload
            artifact_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
            np.savez(
                examples / f"{artifact_id}.npz",
                sample_id=np.asarray(sample_id),
                times=times,
                freqs=freqs,
                relevance=relevance,
                spectrum=spectrum,
                relevance_time_sum=relevance_time_sum,
                relevance_tf_sum=relevance_tf_sum,
                conservation_absolute_error=conservation_absolute_error,
                conservation_relative_error=conservation_relative_error,
                conservation_tolerance=np.asarray(conservation_tolerance),
            )

    publish_generation(destination, write, role="layer_xai")
    return {
        "profile": profile,
        "layer": layer,
        "source": source,
        "target": target,
        "n": len(samples),
        "stdft_n": len(stdft_payloads),
        "bias_zeroed_validation_residual": bias_zeroed_validation_residual,
        "max_bias_inclusive_attribution_gap": float(
            samples["bias_inclusive_attribution_gap"].max()
        ),
        "max_score_recompute_absolute_error": float(
            samples["score_recompute_absolute_error"].max()
        ),
        "max_dft_residual": float(samples["dft_residual"].max()),
        "max_stdft_conservation_relative_error": max(
            (float(payload[7]) for payload in stdft_payloads.values()),
            default=0.0,
        ),
        **{name.lower(): int((samples["quadrant"] == name).sum()) for name in ("TP", "TN", "FP", "FN")},
    }


def run_layerwise_xai(
    *,
    profiles: Sequence[str],
    catalogs: Mapping[str, pd.DataFrame],
    embeddings: Mapping[str, Mapping[str, Mapping[str, object]]],
    paths: LayerwiseSuitePaths,
    frequency_edges: np.ndarray,
    encoder_factory: Callable,
    layers: tuple[int, ...] = tuple(range(1, 13)),
    sources: tuple[str, ...] = ("eng", "por", "zho"),
    targets: tuple[str, ...] = ("eng", "por", "zho"),
    per_class: int = 25,
    stdft_per_class: int = 2,
    seed: int = 42,
    sample_rate: int = 16000,
    conservation_tolerance: float = 1e-3,
    device: str = "cpu",
    encoder_spec_fn: Callable = get_encoder_spec,
    patch_encoder_fn: Callable = patch_ssl_encoder_for_attnlrp,
    head_loader: Callable = joblib.load,
    threshold_loader: Callable = _load_threshold,
    port_head_fn: Callable = port_logistic_head,
    verify_equivalence_fn: Callable = verify_score_equivalence,
    score_head_fn: Callable = score_head,
    cohort_observer: Callable | None = None,
    **cell_dependencies,
) -> pd.DataFrame:
    """Preflight every cell, then execute one fixed cohort per target."""
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be a non-empty sequence without duplicates")
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= 12
            for layer in layers
        )
    ):
        raise ValueError("layers must be unique integers from 1 through 12")
    if (
        isinstance(stdft_per_class, bool)
        or not isinstance(stdft_per_class, int)
        or stdft_per_class < 0
        or stdft_per_class > per_class
    ):
        raise ValueError("stdft_per_class must be between zero and per_class")
    if not isinstance(paths, LayerwiseSuitePaths):
        raise TypeError("paths must be a LayerwiseSuitePaths instance")
    edges = _validate_edges(frequency_edges)

    validated_catalogs: dict[str, pd.DataFrame] = {}
    cohorts: dict[str, pd.DataFrame] = {}
    cohort_ids: dict[str, tuple[str, ...]] = {}
    stdft_ids: dict[str, frozenset[str]] = {}
    for target in targets:
        if target not in catalogs:
            raise ValueError(f"catalog is missing target {target!r}")
        catalog = _validated_catalog(catalogs[target])
        validated_catalogs[target] = catalog
        cohort = select_fixed_cohort(catalog, per_class=per_class, seed=seed)
        for row in cohort.itertuples(index=False):
            if not Path(row.processed_path).is_file():
                raise ValueError(
                    f"processed_path is invalid for {row.sample_id}: {row.processed_path}"
                )
        cohorts[target] = cohort
        cohort_ids[target] = tuple(cohort["sample_id"])
        chosen: list[str] = []
        for label in (0, 1):
            chosen.extend(
                cohort.loc[cohort["label"] == label, "sample_id"]
                .iloc[:stdft_per_class]
                .tolist()
            )
        stdft_ids[target] = frozenset(chosen)

    specs: dict[str, object] = {}
    for profile in profiles:
        spec = encoder_spec_fn(profile)
        specs[profile] = spec
        capabilities = frozenset(getattr(spec, "capabilities", ()))
        attention_rule = getattr(spec, "attention_rule", None)
        required_attention = {
            "cp_lrp": "attnlrp_cp",
            "uniform_lrp": "attnlrp_uniform",
        }.get(attention_rule)
        if required_attention is None:
            raise ValueError(f"profile {profile!r} has an invalid attention_rule")
        missing = {required_attention, "dft_lrp"} - capabilities
        if missing:
            raise ValueError(
                f"profile {profile!r} lacks required XAI capability: {sorted(missing)}"
            )
    aligned_embeddings = _validate_embeddings(
        embeddings, profiles, targets, validated_catalogs, layers
    )

    preflight: dict[tuple[str, int, str, str], pd.DataFrame] = {}
    prepared_cells: dict[tuple[str, int, str, str], dict] = {}
    for profile in profiles:
        for layer in layers:
            for source in sources:
                for target in targets:
                    predictions = _read_prediction_cell(
                        paths,
                        profile,
                        layer,
                        source,
                        target,
                        validated_catalogs[target],
                        cohort_ids[target],
                    )
                    if tuple(predictions["sample_id"]) != cohort_ids[target]:
                        raise ValueError("cohort IDs diverged before backward")
                    key = (profile, layer, source, target)
                    preflight[key] = predictions
                    prepared_cells[key] = _prepare_cell(
                        profile=profile,
                        layer=layer,
                        source=source,
                        target=target,
                        cohort=cohorts[target],
                        target_catalog=validated_catalogs[target],
                        target_embeddings=aligned_embeddings[profile][target],
                        predictions=predictions,
                        paths=paths,
                        head_loader=head_loader,
                        threshold_loader=threshold_loader,
                        port_head_fn=port_head_fn,
                        verify_equivalence_fn=verify_equivalence_fn,
                        score_head_fn=score_head_fn,
                    )

    rows: list[dict] = []
    runtimes: dict[str, tuple[object, object]] = {}
    for profile in profiles:
        encoder, processor = encoder_factory(profile)
        runtimes[profile] = (encoder, processor)
        ensure_ssl_encoder_attnlrp(
            encoder,
            attention_rule=str(getattr(specs[profile], "attention_rule")),
            capabilities=frozenset(getattr(specs[profile], "capabilities", ())),
            patch_fn=patch_encoder_fn,
        )

    for profile in profiles:
        encoder, processor = runtimes[profile]
        for layer in layers:
            for source in sources:
                for target in targets:
                    ids = cohort_ids[target]
                    if cohort_observer is not None:
                        cohort_observer(profile, layer, source, target, ids)
                    rows.append(
                        run_layer_xai_cell(
                            profile=profile,
                            layer=layer,
                            source=source,
                            target=target,
                            cohort=cohorts[target],
                            target_catalog=validated_catalogs[target],
                            target_embeddings=aligned_embeddings[profile][target],
                            predictions=preflight[(profile, layer, source, target)],
                            paths=paths,
                            encoder=encoder,
                            processor=processor,
                            prepared=prepared_cells[
                                (profile, layer, source, target)
                            ],
                            frequency_edges=edges,
                            stdft_sample_ids=stdft_ids[target],
                            sample_rate=sample_rate,
                            conservation_tolerance=conservation_tolerance,
                            device=device,
                            **cell_dependencies,
                        )
                    )
    return pd.DataFrame(rows)
