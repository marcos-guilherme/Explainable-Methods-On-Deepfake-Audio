from __future__ import annotations

import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio import AUDIO_METADATA_COLUMNS, verify_processed_manifest_row
from .core.hashing import sha256_file
from .manifest import MANIFEST_COLUMNS, validate_manifest_rows
from .profiles.english import ENGLISH_PROFILE


SILENCE_THRESHOLD = 1e-4
_SPLIT_ORDER = {split: index for index, split in enumerate(ENGLISH_PROFILE.splits)}
BASELINE_FEATURES = [
    "source_duration_seconds",
    "source_rms",
    "source_peak",
    "source_silence_ratio",
    "source_sample_rate",
    "source_channels",
]
_PROCESSED_COLUMNS = MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
_AUDIT_MEASURES = [
    "source_duration_seconds",
    "source_rms",
    "source_peak",
    "source_silence_ratio",
    "source_sample_rate",
    "source_channels",
    "processed_duration_seconds",
    "processed_rms",
    "processed_peak",
    "processed_silence_ratio",
    "processed_sample_rate",
    "processed_channels",
    "source_rms_dbfs",
    "processed_rms_dbfs",
    "source_estimated_snr_db",
    "processed_estimated_snr_db",
]


@dataclass(frozen=True)
class AuditSummary:
    sample_count: int
    counts: dict[str, list[dict[str, Any]]]
    cross_split_speakers: list[dict[str, str]]
    duplicate_hashes: dict[str, list[dict[str, Any]]]
    missing_values: dict[str, list[dict[str, str]]]
    missing_paths: list[dict[str, str]]
    distributions: dict[str, list[dict[str, Any]]]
    cross_split_source_groups: list[dict[str, Any]]
    duplicate_source_groups: list[dict[str, Any]]
    silence_definition: dict[str, Any]


@dataclass(frozen=True)
class BaselineReport:
    balanced_accuracy: float
    roc_auc: float
    f1: float
    coefficients: list[dict[str, float | str]]
    intercept: float
    class_counts: dict[str, dict[str, int]]
    seed: int
    convergence: dict[str, int | bool]
    feature_order: list[str]


def audit_manifest(frame: pd.DataFrame, output_dir: Path) -> AuditSummary:
    """Audit a canonical processed manifest and publish immutable reports."""
    _validate_processed_manifest(frame)
    enriched = _sample_audit_frame(frame)
    summary = AuditSummary(
        sample_count=len(enriched),
        counts=_counts(enriched),
        cross_split_speakers=_cross_split_speakers(enriched),
        duplicate_hashes={
            "source": _duplicate_hashes(enriched, "sha256_source"),
            "processed": _duplicate_hashes(enriched, "sha256_processed"),
        },
        missing_values=_missing_values(enriched),
        missing_paths=[],
        distributions=_distributions(enriched),
        cross_split_source_groups=_cross_split_source_groups(enriched),
        duplicate_source_groups=_duplicate_source_groups(enriched),
        silence_definition={
            "formula": (
                "count(abs(decoded_sample) <= threshold) / "
                "count(decoded_sample)"
            ),
            "threshold": SILENCE_THRESHOLD,
            "sample_scope": "all decoded samples across all channels",
        },
    )
    _publish_reports(summary, enriched, Path(output_dir))
    return summary


def fit_technical_baseline(
    frame: pd.DataFrame,
    seed: int = 42,
) -> BaselineReport:
    """Fit a technical-shortcut diagnostic on train and evaluate on dev."""
    _validate_processed_manifest(frame)
    _validate_baseline_partitions(frame)
    enriched = _sample_audit_frame(frame)
    features = enriched.loc[:, BASELINE_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Technical baseline contains non-finite feature values")

    train_mask = enriched["split"] == "train"
    dev_mask = enriched["split"] == "dev"
    x_train = features.loc[train_mask].to_numpy(dtype=np.float64)
    x_dev = features.loc[dev_mask].to_numpy(dtype=np.float64)
    y_train = (enriched.loc[train_mask, "label"] == "generated").astype(int)
    y_dev = (enriched.loc[dev_mask, "label"] == "generated").astype(int)
    max_iterations = 1_000
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=seed,
                    solver="liblinear",
                    max_iter=max_iterations,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_dev)
    probabilities = pipeline.predict_proba(x_dev)[:, 1]
    classifier = pipeline.named_steps["classifier"]
    iterations = int(classifier.n_iter_[0])
    convergence_warnings = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    return BaselineReport(
        balanced_accuracy=float(balanced_accuracy_score(y_dev, predictions)),
        roc_auc=float(roc_auc_score(y_dev, probabilities)),
        f1=float(f1_score(y_dev, predictions)),
        coefficients=[
            {"feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(
                BASELINE_FEATURES,
                classifier.coef_[0],
                strict=True,
            )
        ],
        intercept=float(classifier.intercept_[0]),
        class_counts={
            split: {
                label: int(
                    (
                        (enriched["split"] == split)
                        & (enriched["label"] == label)
                    ).sum()
                )
                for label in ("generated", "pristine")
            }
            for split in ("train", "dev")
        },
        seed=seed,
        convergence={
            "converged": convergence_warnings == 0
            and iterations < max_iterations,
            "iterations": iterations,
            "max_iterations": max_iterations,
            "warnings": convergence_warnings,
        },
        feature_order=list(BASELINE_FEATURES),
    )


def audit_and_publish(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[AuditSummary, BaselineReport]:
    """Decode each source/output once, then publish all reports as one set."""
    audited = _build_audited_table(frame, progress)
    summary = _summary_from_audited(audited)
    baseline = _baseline_from_audited(audited)
    _publish_report_set(summary, audited, baseline, Path(output_dir))
    return summary, baseline


def _build_audited_table(
    frame: pd.DataFrame,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    if list(frame.columns) != _PROCESSED_COLUMNS:
        raise ValueError("Audit requires the processed canonical schema")
    validate_manifest_rows(frame, additional_columns=AUDIO_METADATA_COLUMNS)
    records: list[dict[str, Any]] = []
    total = len(frame)
    if progress is not None and total == 0:
        progress(0, 0)
    for completed, (_, row) in enumerate(frame.iterrows(), start=1):
        record = row.to_dict()
        for prefix, path_column in (
            ("source", "source_path"),
            ("processed", "processed_path"),
        ):
            path = Path(row[path_column])
            source_stat = path.stat()
            try:
                info = sf.info(path)
                samples, sample_rate = sf.read(
                    path, dtype="float64", always_2d=True
                )
            finally:
                os.utime(
                    path, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns)
                )
            if samples.size == 0 or not np.isfinite(samples).all():
                raise ValueError(f"Invalid decoded audio during audit: {path}")
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
            expected_hash = row[f"sha256_{prefix}"]
            if prefix == "processed" and _sha256_path(path) != expected_hash:
                raise ValueError(f"sha256_processed diverges for {row['utt_id']}")
            expected_values = {
                f"{prefix}_sample_rate": int(sample_rate),
                f"{prefix}_channels": int(samples.shape[1]),
                f"{prefix}_frames": int(samples.shape[0]),
                f"{prefix}_duration_seconds": samples.shape[0] / sample_rate,
                f"{prefix}_peak": peak,
                f"{prefix}_rms": rms,
            }
            for name, expected in expected_values.items():
                actual = float(row[name])
                if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                    raise ValueError(f"{name} diverges for {row['utt_id']}")
            record[f"{prefix}_format"] = info.format
            record[f"{prefix}_subtype"] = info.subtype
            if prefix == "processed":
                record["output_format"] = info.format
                record["output_subtype"] = info.subtype
            record[f"{prefix}_silence_ratio"] = float(
                np.count_nonzero(np.abs(samples) <= SILENCE_THRESHOLD)
                / samples.size
            )
            record[f"{prefix}_rms_dbfs"] = (
                float(20.0 * np.log10(rms)) if rms > 0 else None
            )
            record[f"{prefix}_rms_dbfs_unavailable_reason"] = (
                None if rms > 0 else "decoded RMS is zero"
            )
            estimate, reason = _estimated_snr_proxy(samples, sample_rate)
            record[f"{prefix}_estimated_snr_db"] = estimate
            record[f"{prefix}_estimated_snr_db_unavailable_reason"] = reason
        records.append(record)
        if progress is not None and (completed % 1000 == 0 or completed == total):
            progress(completed, total)
    return pd.DataFrame(records)


def _estimated_snr_proxy(
    samples: np.ndarray, sample_rate: int
) -> tuple[float | None, str | None]:
    """Estimate SNR as active-sample RMS / lowest-decile frame RMS."""
    mono = samples.mean(axis=1)
    frame_size = max(1, int(sample_rate * 0.02))
    frame_rms = [
        float(np.sqrt(np.mean(np.square(mono[start : start + frame_size]))))
        for start in range(0, len(mono), frame_size)
        if len(mono[start : start + frame_size])
    ]
    positive = np.asarray([value for value in frame_rms if value > 0])
    signal_rms = float(np.sqrt(np.mean(np.square(mono))))
    if signal_rms == 0:
        return None, "decoded signal RMS is zero"
    if positive.size < 2:
        return None, "insufficient non-silent 20 ms frames for noise-floor proxy"
    noise = float(np.quantile(positive, 0.1))
    if noise <= 0:
        return None, "estimated noise-floor proxy is zero"
    return float(20.0 * np.log10(signal_rms / noise)), None


def _summary_from_audited(enriched: pd.DataFrame) -> AuditSummary:
    counts = _counts(enriched)
    for dimensions in (
        ("split", "label", "protocol_codec"),
        ("split", "label", "source_format", "source_subtype"),
        ("split", "label", "output_format", "output_subtype"),
    ):
        counts["_".join(dimensions)] = _structured_count(enriched, dimensions)
    if "source_group_id" in enriched:
        grouped = enriched.loc[enriched["source_group_id"].fillna("") != ""]
        counts["source_group"] = [
            {"source_group_id": str(group_id), "count": int(count)}
            for group_id, count in grouped.groupby("source_group_id", sort=True).size().items()
        ]
    return AuditSummary(
        sample_count=len(enriched),
        counts=counts,
        cross_split_speakers=_cross_split_speakers(enriched),
        duplicate_hashes={
            "source": _duplicate_hashes(enriched, "sha256_source"),
            "processed": _duplicate_hashes(enriched, "sha256_processed"),
        },
        missing_values=_missing_values(enriched),
        missing_paths=[],
        distributions=_distributions(enriched),
        cross_split_source_groups=_cross_split_source_groups(enriched),
        duplicate_source_groups=_duplicate_source_groups(enriched),
        silence_definition={
            "formula": "count(abs(decoded_sample) <= threshold) / count(decoded_sample)",
            "threshold": SILENCE_THRESHOLD,
            "sample_scope": "all decoded samples across all channels",
        },
    )


def _baseline_from_audited(enriched: pd.DataFrame) -> BaselineReport:
    _validate_baseline_partitions(enriched)
    features = enriched.loc[:, BASELINE_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Technical baseline contains non-finite feature values")
    train_mask = enriched["split"] == "train"
    dev_mask = enriched["split"] == "dev"
    y_train = (enriched.loc[train_mask, "label"] == "generated").astype(int)
    y_dev = (enriched.loc[dev_mask, "label"] == "generated").astype(int)
    maximum = 1_000
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=42, solver="liblinear", max_iter=maximum
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features.loc[train_mask], y_train)
    predictions = model.predict(features.loc[dev_mask])
    probabilities = model.predict_proba(features.loc[dev_mask])[:, 1]
    classifier = model.named_steps["classifier"]
    iterations = int(classifier.n_iter_[0])
    warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    return BaselineReport(
        balanced_accuracy=float(balanced_accuracy_score(y_dev, predictions)),
        roc_auc=float(roc_auc_score(y_dev, probabilities)),
        f1=float(f1_score(y_dev, predictions)),
        coefficients=[
            {"feature": name, "coefficient": float(value)}
            for name, value in zip(
                BASELINE_FEATURES, classifier.coef_[0], strict=True
            )
        ],
        intercept=float(classifier.intercept_[0]),
        class_counts={
            split: {
                label: int(
                    ((enriched["split"] == split) & (enriched["label"] == label)).sum()
                )
                for label in ("generated", "pristine")
            }
            for split in ("train", "dev")
        },
        seed=42,
        convergence={
            "converged": warning_count == 0 and iterations < maximum,
            "iterations": iterations,
            "max_iterations": maximum,
            "warnings": warning_count,
        },
        feature_order=list(BASELINE_FEATURES),
    )


def _sha256_path(path: Path) -> str:
    return sha256_file(path)


def _publish_report_set(
    summary: Any,
    samples: pd.DataFrame,
    baseline: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_value = asdict(summary) if hasattr(summary, "__dataclass_fields__") else summary
    baseline_value = (
        asdict(baseline) if hasattr(baseline, "__dataclass_fields__") else baseline
    )
    payloads = {
        "audit.json": (
            json.dumps(summary_value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode(),
        "audit.csv": samples.to_csv(index=False).encode(),
        "technical_baseline.json": (
            json.dumps(baseline_value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    }
    destinations = {name: output_dir / name for name in payloads}
    existing = {name: path for name, path in destinations.items() if path.exists()}
    if existing:
        conflicts = [
            name
            for name, path in existing.items()
            if path.read_bytes() != payloads[name]
        ]
        if conflicts:
            raise FileExistsError(
                "Report-set conflict with existing content: "
                + ", ".join(sorted(conflicts))
            )
        if len(existing) == len(payloads):
            return
    temporary: dict[str, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for name, payload in payloads.items():
            if name in existing:
                continue
            path = _temporary_path(output_dir, name)
            with path.open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            temporary[name] = path
        for name, source in temporary.items():
            destination = destinations[name]
            os.link(source, destination)
            published.append((source, destination))
    except BaseException:
        for source, destination in published:
            if _same_file(source, destination):
                destination.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _structured_count(
    frame: pd.DataFrame,
    dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    if any(column not in frame.columns for column in dimensions):
        return []
    records = []
    grouped = frame.groupby(list(dimensions), sort=True, dropna=False).size()
    for key, count in grouped.items():
        values = key if isinstance(key, tuple) else (key,)
        records.append(
            {
                **dict(zip(dimensions, values, strict=True)),
                "count": int(count),
            }
        )
    return records


def _nonempty_source_groups(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_group_id" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["source_group_id"].fillna("").astype(str) != ""]


def _cross_split_source_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = _nonempty_source_groups(frame)
    result = []
    for group_id, rows in grouped.groupby("source_group_id", sort=True):
        splits = sorted(
            set(rows["split"]),
            key=lambda split: _SPLIT_ORDER.get(split, len(_SPLIT_ORDER)),
        )
        if len(splits) > 1:
            result.append(
                {"source_group_id": str(group_id), "splits": splits}
            )
    return result


def _duplicate_source_groups(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = _nonempty_source_groups(frame)
    result = []
    for group_id, rows in grouped.groupby("source_group_id", sort=True):
        if len(rows) < 2:
            continue
        result.append(
            {
                "source_group_id": str(group_id),
                "samples": [
                    {"utt_id": str(row["utt_id"]), "split": str(row["split"])}
                    for _, row in rows.sort_values(
                        ["split", "utt_id"], kind="mergesort"
                    ).iterrows()
                ],
            }
        )
    return result


def _validate_processed_manifest(frame: pd.DataFrame) -> None:
    if list(frame.columns) != _PROCESSED_COLUMNS:
        raise ValueError("Audit requires the processed canonical schema")
    validate_manifest_rows(frame, additional_columns=AUDIO_METADATA_COLUMNS)
    for _, row in frame.iterrows():
        verify_processed_manifest_row(row)


def _validate_baseline_partitions(frame: pd.DataFrame) -> None:
    if frame["utt_id"].duplicated().any():
        duplicate = frame.loc[frame["utt_id"].duplicated(), "utt_id"].iloc[0]
        raise ValueError(f"Duplicate utt_id across manifest: {duplicate}")
    for split in ("train", "dev"):
        labels = set(frame.loc[frame["split"] == split, "label"])
        if labels != {"pristine", "generated"}:
            raise ValueError(
                f"Technical baseline requires both classes in {split}"
            )
    train_speakers = set(frame.loc[frame["split"] == "train", "spk_id"])
    dev_speakers = set(frame.loc[frame["split"] == "dev", "spk_id"])
    overlap = sorted(train_speakers & dev_speakers)
    if overlap:
        raise ValueError(
            "Technical baseline refuses speaker leakage between train and dev: "
            + ", ".join(overlap)
        )


def _sample_audit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    source_ratios: list[float] = []
    processed_ratios: list[float] = []
    for _, row in enriched.iterrows():
        source_ratios.append(_silence_ratio(Path(row["source_path"])))
        processed_ratios.append(_silence_ratio(Path(row["processed_path"])))
    enriched["source_silence_ratio"] = source_ratios
    enriched["processed_silence_ratio"] = processed_ratios
    return enriched


def _silence_ratio(path: Path) -> float:
    source_stat = path.stat()
    try:
        samples, _ = sf.read(path, dtype="float64", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read audio for silence audit: {path}") from exc
    finally:
        os.utime(
            path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
    if samples.size == 0:
        raise ValueError(f"Audio is empty during silence audit: {path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio has non-finite samples during silence audit: {path}")
    return float(np.count_nonzero(np.abs(samples) <= SILENCE_THRESHOLD) / samples.size)


def _counts(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    dimensions = ["split", "label", "spk_id", "attack_id"]
    display_names = {
        "spk_id": "speaker",
        "attack_id": "attack",
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for length in range(1, len(dimensions) + 1):
        for selected in combinations(dimensions, length):
            name = "_".join(
                display_names.get(item, item)
                for item in selected
            )
            grouped = frame.groupby(
                list(selected),
                dropna=False,
                sort=True,
            ).size()
            records: list[dict[str, Any]] = []
            for key, count in grouped.items():
                values = key if isinstance(key, tuple) else (key,)
                record = {
                    display_names.get(column, column): value
                    for column, value in zip(selected, values, strict=True)
                }
                record["count"] = int(count)
                records.append(record)
            result[name] = records
    return result


def _cross_split_speakers(
    frame: pd.DataFrame,
) -> list[dict[str, str]]:
    speakers = (
        frame.groupby("spk_id", sort=True)["split"]
        .agg(
            lambda values: sorted(
                set(values),
                key=lambda value: _SPLIT_ORDER[value],
            )
        )
    )
    result: list[dict[str, str]] = []
    for speaker, splits in speakers.items():
        if len(splits) < 2:
            continue
        for left, right in combinations(splits, 2):
            result.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "spk_id": speaker,
                }
            )
    return result


def _duplicate_hashes(
    frame: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    grouped = frame.groupby(column, sort=True, dropna=False)
    for value, group in grouped:
        if len(group) < 2:
            continue
        duplicates.append(
            {
                "sha256": str(value),
                "samples": [
                    {"utt_id": row["utt_id"], "split": row["split"]}
                    for _, row in group.sort_index().iterrows()
                ],
            }
        )
    return duplicates


def _missing_values(frame: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for column in frame.columns:
        missing = frame[column].isna()
        if not missing.any():
            continue
        result[column] = [
            {"utt_id": row["utt_id"], "split": row["split"]}
            for _, row in frame.loc[missing].iterrows()
        ]
    return result


def _distributions(
    frame: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    grouped = frame.groupby(["split", "label"], sort=True)
    for measure in _AUDIT_MEASURES:
        if measure not in frame.columns:
            continue
        records: list[dict[str, Any]] = []
        for key, group in grouped:
            values = pd.to_numeric(group[measure], errors="coerce")
            finite = values[np.isfinite(values)]
            split, label = key
            records.append(
                {
                    "split": split,
                    "label": label,
                    **_describe(finite, len(values)),
                }
            )
        result[measure] = records
    return result


def _describe(
    finite: pd.Series,
    total: int,
) -> dict[str, float | int | None]:
    if finite.empty:
        return {
            "count": 0,
            "missing_or_nonfinite": total,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(len(finite)),
        "missing_or_nonfinite": int(total - len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "p25": float(finite.quantile(0.25)),
        "median": float(finite.median()),
        "p75": float(finite.quantile(0.75)),
        "max": float(finite.max()),
    }


def _publish_reports(
    summary: AuditSummary,
    samples: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [
        output_dir / "audit.json",
        output_dir / "audit.csv",
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError("An audit report destination already exists")

    temporary_paths: list[Path] = []
    published: list[tuple[Path, Path]] = []
    try:
        json_path = _temporary_path(output_dir, "audit.json")
        temporary_paths.append(json_path)
        with json_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(
                asdict(summary),
                output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

        csv_path = _temporary_path(output_dir, "audit.csv")
        temporary_paths.append(csv_path)
        with csv_path.open("w", encoding="utf-8", newline="") as output:
            samples.to_csv(output, index=False)
            output.flush()
            os.fsync(output.fileno())

        for temporary, destination in zip(
            temporary_paths,
            destinations,
            strict=True,
        ):
            os.link(temporary, destination)
            published.append((temporary, destination))
    except BaseException:
        for temporary, destination in published:
            if _same_file(temporary, destination):
                destination.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _temporary_path(directory: Path, report_name: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{report_name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False
