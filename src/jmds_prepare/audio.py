from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from .core.hashing import sha256_file, sha256_file_preserving_times
from .manifest import (
    MANIFEST_COLUMNS,
    validate_manifest_rows,
    write_manifest_atomic,
)
from .storage import estimates
from .storage.layout import DataLayout, english_layout


AUDIO_METADATA_COLUMNS = [
    "source_sample_rate",
    "source_channels",
    "source_frames",
    "source_duration_seconds",
    "source_peak",
    "source_rms",
    "processed_sample_rate",
    "processed_channels",
    "processed_frames",
    "processed_duration_seconds",
    "processed_peak",
    "processed_rms",
]

@dataclass(frozen=True)
class AudioMetadata:
    source_sample_rate: int
    source_channels: int
    source_frames: int
    source_duration_seconds: float
    source_peak: float
    source_rms: float
    processed_sample_rate: int
    processed_channels: int
    processed_frames: int
    processed_duration_seconds: float
    processed_peak: float
    processed_rms: float
    sha256_source: str
    sha256_processed: str


def process_audio(
    source: Path,
    destination: Path,
    sample_rate: int,
) -> AudioMetadata:
    """Create a deterministic PCM-16 mono WAV without modifying the source."""
    _validate_sample_rate(sample_rate)
    source = Path(source)
    destination = Path(destination)
    try:
        source_stat = source.stat()
    except OSError as exc:
        raise ValueError(f"Could not read audio source: {source}") from exc

    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination audio paths must differ")

    temporary_path: Path | None = None
    try:
        samples, source_rate = _read_source(source)
        source_frames, source_channels = samples.shape
        source_peak, source_rms = _statistics(samples)
        with np.errstate(over="ignore", invalid="ignore"):
            row_scale = np.max(np.abs(samples), axis=1)
            scaled = np.divide(
                samples,
                row_scale[:, None],
                out=np.zeros_like(samples),
                where=row_scale[:, None] != 0,
            )
            mono = row_scale * scaled.mean(axis=1, dtype=np.float64)
        _require_finite(mono, "mono conversion")
        with np.errstate(over="ignore", invalid="ignore"):
            processed = _resample(mono, source_rate, sample_rate)
        _require_finite(processed, "resampling")
        processed = np.clip(processed, -1.0, 1.0)
        _require_finite(processed, "clipping")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp.wav",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with sf.SoundFile(
                temporary,
                mode="w",
                samplerate=sample_rate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            ) as output:
                output.write(processed)
                output.flush()
            temporary.flush()
            os.fsync(temporary.fileno())

        _validate_output(temporary_path, sample_rate, len(processed))
        candidate_hash = _sha256(temporary_path)
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if not _file_matches_hash(destination, candidate_hash):
                raise FileExistsError(
                    f"Existing processed audio conflicts with deterministic output: "
                    f"{destination}"
                )
        else:
            temporary_path.unlink()
            temporary_path = None

        output_samples, output_rate = sf.read(
            destination,
            dtype="float64",
            always_2d=True,
        )
        output_peak, output_rms = _statistics(output_samples)
        output_frames, output_channels = output_samples.shape
        return AudioMetadata(
            source_sample_rate=source_rate,
            source_channels=source_channels,
            source_frames=source_frames,
            source_duration_seconds=source_frames / source_rate,
            source_peak=source_peak,
            source_rms=source_rms,
            processed_sample_rate=int(output_rate),
            processed_channels=output_channels,
            processed_frames=output_frames,
            processed_duration_seconds=output_frames / output_rate,
            processed_peak=output_peak,
            processed_rms=output_rms,
            sha256_source=_sha256(source),
            sha256_processed=_sha256(destination),
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        os.utime(
            source,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )


def process_manifest(
    manifest: pd.DataFrame,
    data_root: Path,
    destination: Path,
    sample_rate: int = 16_000,
) -> pd.DataFrame:
    """Process all rows and atomically publish the enriched manifest."""
    _validate_sample_rate(sample_rate)
    frame = manifest.copy()
    expected_columns = MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
    columns = list(frame.columns)
    if columns == MANIFEST_COLUMNS:
        is_processed = False
        if any(
            row["processed_path"] != "" or row["sha256_processed"] != ""
            for _, row in frame.iterrows()
        ):
            raise ValueError("raw manifest processed fields must be empty")
        validate_manifest_rows(frame)
    elif columns == expected_columns:
        is_processed = True
        validate_manifest_rows(frame, additional_columns=AUDIO_METADATA_COLUMNS)
    else:
        raise ValueError(
            "Audio manifest columns must be the raw or processed canonical schema"
        )

    layout = english_layout(data_root)
    destination = Path(destination)
    resolved_manifest = destination.resolve()
    planned: list[tuple[int, Path, Path]] = []
    seen_destinations: set[Path] = set()
    planned_directories: set[Path] = set()
    resolved_sources: set[Path] = set()
    for position, (_, row) in enumerate(frame.iterrows()):
        utt_id = row["utt_id"]
        split = row["split"]
        label = row["label"]
        source = Path(row["source_path"])
        resolved_sources.add(source.resolve())
        output_root = layout.processed_dir(split, label)
        output = output_root / f"{utt_id}.wav"
        resolved_root = output_root.resolve()
        resolved_output = output.resolve()
        planned_directories.add(resolved_root)
        if resolved_output.parent != resolved_root:
            raise ValueError(
                f"Processed destination for {utt_id} escapes expected directory"
            )
        if resolved_output in seen_destinations:
            raise ValueError(f"Duplicate destination path: {resolved_output}")
        seen_destinations.add(resolved_output)

        if is_processed:
            existing_path = row["processed_path"]
            if (
                not existing_path
                or Path(existing_path).resolve() != resolved_output
            ):
                raise ValueError(f"processed_path diverges for {utt_id}")
            if not output.is_file():
                raise ValueError(
                    f"processed_path deterministic file is missing for {utt_id}"
                )
        elif row["processed_path"] or row["sha256_processed"]:
            raise ValueError(
                f"raw manifest processed fields must be empty for {utt_id}"
            )
        planned.append((position, source, output))

    planned_paths = seen_destinations | planned_directories
    if any(
        path != resolved_manifest
        and path.is_relative_to(resolved_manifest)
        for path in planned_paths
    ):
        raise ValueError(
            f"manifest destination collides because it is an ancestor of a "
            f"planned output or directory: {destination}"
        )
    if (
        resolved_manifest in resolved_sources
        or resolved_manifest in seen_destinations
        or resolved_manifest in planned_directories
        or (destination.exists() and destination.is_dir())
    ):
        raise ValueError(
            f"manifest destination collides with a source, output, or directory: "
            f"{destination}"
        )

    _require_processing_space(frame, layout, sample_rate)
    records: list[dict[str, Any]] = []
    for position, source, output in planned:
        metadata = process_audio(source, output, sample_rate)
        row = frame.iloc[position]
        if is_processed:
            _validate_processed_metadata(row, metadata)
            continue
        record = row.loc[MANIFEST_COLUMNS].to_dict()
        record["processed_path"] = str(output)
        record["sha256_processed"] = metadata.sha256_processed
        record.update(
            {
                name: value
                for name, value in asdict(metadata).items()
                if name in AUDIO_METADATA_COLUMNS
            }
        )
        records.append(record)

    processed = (
        frame.copy()
        if is_processed
        else pd.DataFrame(records, columns=expected_columns)
    )
    write_manifest_atomic(processed, destination)
    return processed


def _require_processing_space(
    frame: pd.DataFrame, layout: DataLayout, sample_rate: int
) -> None:
    estimate = estimates.processing_space_estimate(frame, layout, sample_rate)
    root = estimates.existing_ancestor(layout.data_root)
    free = int(shutil.disk_usage(root).free)
    if free < estimate.required_bytes:
        raise estimates.InsufficientProcessingSpaceError(
            f"Refusing to process audio: required {estimate.required_bytes} "
            f"bytes, free {free} bytes at {root} (missing outputs "
            f"{estimate.missing_outputs} (allocated "
            f"{estimate.missing_output_allocated_bytes}); largest temporary "
            f"{estimate.largest_temporary_bytes}; manifest reserve "
            f"{estimate.manifest_reserve_bytes}; safety reserve "
            f"{estimate.safety_reserve_bytes})"
        )


def verify_processed_manifest_row(row: pd.Series) -> AudioMetadata:
    """Recalculate and verify one processed manifest row without writing."""
    missing_columns = [
        column
        for column in MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
        if column not in row.index
    ]
    if missing_columns:
        raise ValueError(
            "Processed manifest row is missing columns: "
            + ", ".join(missing_columns)
        )

    utt_id = row["utt_id"]
    source = Path(row["source_path"])
    processed = Path(row["processed_path"])
    if not source.is_file():
        raise FileNotFoundError(f"Source audio is missing for {utt_id}: {source}")
    if not processed.is_file():
        raise FileNotFoundError(
            f"Processed audio is missing for {utt_id}: {processed}"
        )
    if source.resolve() == processed.resolve():
        raise ValueError(f"source_path and processed_path must differ for {utt_id}")

    source_hash = _sha256_preserving_times(source)
    processed_hash = _sha256_preserving_times(processed)
    if row["sha256_source"] != source_hash:
        raise ValueError(f"sha256_source diverges for {utt_id}")
    if row["sha256_processed"] != processed_hash:
        raise ValueError(f"sha256_processed diverges for {utt_id}")

    source_samples, source_rate = _read_audio_preserving_times(source)
    processed_samples, processed_rate = _read_audio_preserving_times(processed)
    source_frames, source_channels = source_samples.shape
    processed_frames, processed_channels = processed_samples.shape
    source_peak, source_rms = _statistics(source_samples)
    processed_peak, processed_rms = _statistics(processed_samples)
    metadata = AudioMetadata(
        source_sample_rate=source_rate,
        source_channels=source_channels,
        source_frames=source_frames,
        source_duration_seconds=source_frames / source_rate,
        source_peak=source_peak,
        source_rms=source_rms,
        processed_sample_rate=processed_rate,
        processed_channels=processed_channels,
        processed_frames=processed_frames,
        processed_duration_seconds=processed_frames / processed_rate,
        processed_peak=processed_peak,
        processed_rms=processed_rms,
        sha256_source=source_hash,
        sha256_processed=processed_hash,
    )
    _validate_processed_metadata(row, metadata)
    return metadata


def _read_source(source: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = sf.read(
            source,
            dtype="float64",
            always_2d=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read audio source: {source}") from exc
    if (
        not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) <= 0
    ):
        raise ValueError(f"Audio source has invalid sample rate: {source}")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"Audio source is empty: {source}")
    _require_finite(samples, f"decoded source {source}")
    return samples, int(sample_rate)


def _read_audio_preserving_times(path: Path) -> tuple[np.ndarray, int]:
    source_stat = path.stat()
    try:
        return _read_source(path)
    finally:
        os.utime(
            path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )


def _resample(
    samples: np.ndarray,
    source_rate: int,
    destination_rate: int,
) -> np.ndarray:
    if source_rate == destination_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, destination_rate)
    up = destination_rate // divisor
    down = source_rate // divisor
    return np.asarray(resample_poly(samples, up, down), dtype=np.float64)


def _validate_output(path: Path, sample_rate: int, frames: int) -> None:
    try:
        info = sf.info(path)
        samples, actual_rate = sf.read(path, dtype="float64", always_2d=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not validate processed audio: {path}") from exc
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate != sample_rate
        or info.frames != frames
        or actual_rate != sample_rate
        or samples.shape != (frames, 1)
        or not np.isfinite(samples).all()
    ):
        raise ValueError(f"Processed audio validation failed: {path}")


def _statistics(samples: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return 0.0, 0.0
    scaled = samples / peak
    rms_factor = float(
        np.sqrt(np.mean(np.square(scaled), dtype=np.float64))
    )
    rms = peak * rms_factor
    if not math.isfinite(rms) and rms_factor <= 1.0:
        rms = peak
    return peak, rms


def _require_finite(samples: np.ndarray, stage: str) -> None:
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio contains non-finite samples after {stage}")


def _validate_processed_metadata(
    row: pd.Series,
    metadata: AudioMetadata,
) -> None:
    calculated = asdict(metadata)
    utt_id = row["utt_id"]
    if row["sha256_source"] != metadata.sha256_source:
        raise ValueError(f"sha256_source diverges for {utt_id}")
    if row["sha256_processed"] != metadata.sha256_processed:
        raise ValueError(f"sha256_processed diverges for {utt_id}")

    integer_fields = {
        "source_sample_rate",
        "source_channels",
        "source_frames",
        "processed_sample_rate",
        "processed_channels",
        "processed_frames",
    }
    for column in AUDIO_METADATA_COLUMNS:
        actual = row[column]
        expected = calculated[column]
        if column in integer_fields:
            matches = (
                isinstance(actual, Integral)
                and not isinstance(actual, bool)
                and int(actual) == expected
            )
        else:
            matches = (
                isinstance(actual, Real)
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        if not matches:
            raise ValueError(f"{column} diverges for {utt_id}")


def _file_matches_hash(path: Path, expected_hash: str) -> bool:
    try:
        return path.is_file() and _sha256(path) == expected_hash
    except OSError:
        return False


def _validate_sample_rate(sample_rate: int) -> None:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or int(sample_rate) <= 0
    ):
        raise ValueError("sample_rate must be a positive integer")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _sha256_preserving_times(path: Path) -> str:
    return sha256_file_preserving_times(path, hasher=_sha256)
