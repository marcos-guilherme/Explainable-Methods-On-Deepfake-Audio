"""Storage-size estimates for acquisition planning, from headers and file sizes."""

from __future__ import annotations

import os
import wave
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

import soundfile as sf

from ..core import paths
from ..profiles.english import ENGLISH_PROFILE
from .layout import DataLayout, english_layout

if TYPE_CHECKING:
    import pandas as pd

# Canonical RIFF/fmt/data header libsndfile writes for PCM-16 mono WAV.
WAV_PCM16_HEADER_BYTES = 44
# Allocation unit assumed for every new file (the NTFS/ext4 default).
CLUSTER_BYTES = 4096
# The manifest is reserved twice: the published CSV plus its atomic temporary.
# Each copy gets max(1 MiB, 2048 bytes per row); real rows are well below 1 KiB.
MANIFEST_RESERVE_MIN_BYTES = 1 << 20
MANIFEST_RESERVE_BYTES_PER_ROW = 2048
# Headroom kept free after processing: max(1 GiB, 1% of the subtotal).
SAFETY_RESERVE_MIN_BYTES = 1 << 30
SAFETY_RESERVE_PERCENT = 1


class InsufficientProcessingSpaceError(RuntimeError):
    """Raised before processing when free space cannot hold the missing WAVs."""


@dataclass(frozen=True)
class ProcessingSpaceEstimate:
    """Conservative disk requirement of one ``process_manifest`` run.

    Only outputs missing from disk are counted as new files. The largest
    planned WAV is always added once, because ``process_audio`` writes a
    temporary candidate even when the deterministic output already exists.
    """

    rows: int
    missing_outputs: int
    missing_output_bytes: int
    missing_output_allocated_bytes: int
    largest_temporary_bytes: int
    manifest_reserve_bytes: int
    safety_reserve_bytes: int

    @property
    def subtotal_bytes(self) -> int:
        return (
            self.missing_output_allocated_bytes
            + self.largest_temporary_bytes
            + self.manifest_reserve_bytes
        )

    @property
    def required_bytes(self) -> int:
        return self.subtotal_bytes + self.safety_reserve_bytes


class EstimateConfig(Protocol):
    jmds_root: Path
    data_root: Path
    sample_rate: int


def acquisition_storage_state(
    files: dict[str, Any],
    archive_root: Path,
    raw_root: Path,
    ledger_path: Path,
    *,
    read_extraction_ledger: Callable[[Path], pd.DataFrame],
) -> dict[str, int]:
    extracted_flac_bytes = sum(
        path.stat().st_size
        for split in ENGLISH_PROFILE.splits
        for path in (raw_root / split).glob("*.flac")
        if path.is_file()
    )
    current_archive_bytes = 0
    for name in files:
        candidates = [
            archive_root / name,
            archive_root / f"{name}.partial",
        ]
        if archive_root.is_dir():
            candidates.extend(archive_root.glob(f"{name}.invalid*"))
        current_archive_bytes += sum(
            candidate.stat().st_size
            for candidate in candidates
            if candidate.is_file()
        )

    completed_archives: set[str] = set()
    ledger = read_extraction_ledger(ledger_path)
    if not ledger.empty:
        for archive_name, rows in ledger.groupby("archive_name", sort=False):
            file = files.get(str(archive_name))
            if file is not None and all(
                str(checksum).lower() == file.checksum.lower()
                for checksum in rows["archive_md5"]
            ):
                completed_archives.add(str(archive_name))

    remaining = [
        file for name, file in files.items() if name not in completed_archives
    ]
    remaining_flac_upper_bound = sum(file.size for file in remaining)
    largest_active = max((file.size for file in remaining), default=0)
    active_working_set = current_archive_bytes + largest_active
    total_peak = (
        extracted_flac_bytes
        + current_archive_bytes
        + remaining_flac_upper_bound
        + largest_active
    )
    return {
        "extracted_flac_bytes": extracted_flac_bytes,
        "current_archive_partial_quarantine_bytes": current_archive_bytes,
        "remaining_flac_upper_bound_bytes": remaining_flac_upper_bound,
        "largest_active_remaining_tar_bytes": largest_active,
        "maximum_active_tar_working_set_bytes": active_working_set,
        "conservative_total_acquisition_peak_bytes": total_peak,
    }


def audio_storage_estimate(
    config: EstimateConfig,
    protocols: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    result: dict[str, dict[str, Any]] = {
        label: {
            "files_measured": 0,
            "files_pending_extraction": 0,
            "processed_pcm16_bytes": 0,
        }
        for label in ("generated", "pristine")
    }
    layout = english_layout(config.data_root)
    tasks: list[tuple[str, Path]] = []
    generated_ids = {
        split: set(
            protocol.loc[protocol["label"] == "generated", "utt_id"].astype(str)
        )
        for split, protocol in protocols.items()
    }
    generated_count = sum(len(ids) for ids in generated_ids.values())
    if generated_count > 1_000:
        bulk = bulk_generated_pcm16_bytes(config, generated_ids)
        result["generated"].update(bulk)
    for split, protocol in protocols.items():
        for row in protocol.to_dict("records"):
            label = str(row["label"])
            utt_id = str(row["utt_id"])
            if label == "generated":
                if generated_count > 1_000:
                    continue
                path = (
                    ENGLISH_PROFILE.generated_split_dir(config.jmds_root, split)
                    / f"{utt_id}.wav"
                )
            else:
                path = layout.raw_split_dir(split) / f"{utt_id}.flac"
            tasks.append((label, path))
    maximum_single_output = result["generated"].pop(
        "maximum_single_output_bytes", 0
    )
    workers = min(32, max(4, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        measurements = executor.map(
            lambda task: measure_audio_header(task, config.sample_rate),
            tasks,
        )
        for label, output_bytes in measurements:
            if output_bytes is None:
                result[label]["files_pending_extraction"] += 1
                continue
            result[label]["files_measured"] += 1
            result[label]["processed_pcm16_bytes"] += output_bytes
            maximum_single_output = max(maximum_single_output, output_bytes)
    pristine = result["pristine"]
    pristine["status"] = (
        "pending until extraction"
        if pristine["files_pending_extraction"]
        else "complete from current headers"
    )
    result["generated"]["status"] = (
        "complete from current headers"
        if not result["generated"]["files_pending_extraction"]
        else "incomplete: generated source missing"
    )
    return {
        **result,
        "maximum_single_output_bytes": maximum_single_output,
    }


def bulk_generated_pcm16_bytes(
    config: EstimateConfig,
    generated_ids: dict[str, set[str]],
) -> dict[str, Any]:
    total_bytes = 0
    maximum_single_output = 0
    measured = 0
    missing = 0
    for split, wanted_ids in generated_ids.items():
        directory = ENGLISH_PROFILE.generated_split_dir(config.jmds_root, split)
        entries = {
            Path(entry.name).stem: entry
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.lower().endswith(".wav")
        }
        present_ids = sorted(wanted_ids & set(entries))
        sample_ids = (
            [present_ids[index] for index in {0, len(present_ids) // 2, -1}]
            if present_ids
            else []
        )
        for utt_id in sample_ids:
            entry = entries[utt_id]
            with _preserved_times(Path(entry.path)), wave.open(entry.path, "rb") as source:
                if (
                    source.getframerate() != config.sample_rate
                    or source.getnchannels() != 1
                    or source.getsampwidth() != 2
                    or entry.stat().st_size - source.getnframes() * 2 != 44
                ):
                    raise ValueError(
                        "Generated WAV storage shortcut requires PCM-16 mono "
                        f"{config.sample_rate} Hz with a 44-byte header: {entry.path}"
                    )
        for utt_id in present_ids:
            output_bytes = entries[utt_id].stat().st_size - 44
            total_bytes += output_bytes
            maximum_single_output = max(maximum_single_output, output_bytes)
        measured += len(present_ids)
        missing += len(wanted_ids - set(entries))
    return {
        "files_measured": measured,
        "files_pending_extraction": missing,
        "processed_pcm16_bytes": total_bytes,
        "maximum_single_output_bytes": maximum_single_output,
        "measurement_method": (
            "actual WAV sizes minus 44-byte headers; beginning/middle/end "
            "headers in each split verified as PCM-16 mono target-rate"
        ),
    }


def measure_audio_header(
    task: tuple[str, Path], sample_rate: int
) -> tuple[str, int | None]:
    label, path = task
    if not path.is_file():
        return label, None
    frames, source_rate = _read_frames_and_rate(path)
    return label, _resampled_frames(frames, source_rate, sample_rate) * 2


def remaining_processed_wav_bytes(
    manifest: pd.DataFrame,
    layout: DataLayout,
    sample_rate: int,
) -> int:
    """Bytes of the PCM-16 mono WAVs still missing for a validated manifest.

    Each missing output counts its resampled payload, from the source header,
    plus the WAV header; outputs already on disk add nothing.
    """
    return sum(
        size
        for exists, size in _planned_wav_sizes(manifest, layout, sample_rate)
        if not exists
    )


def processing_space_estimate(
    manifest: pd.DataFrame,
    layout: DataLayout,
    sample_rate: int,
) -> ProcessingSpaceEstimate:
    """Conservative space needed to process a validated manifest."""
    planned = _planned_wav_sizes(manifest, layout, sample_rate)
    return processing_space_estimate_from_sizes(
        [size for exists, size in planned if not exists],
        [size for _, size in planned],
        rows=len(manifest),
    )


def processing_space_estimate_from_sizes(
    missing_sizes: Iterable[int],
    planned_sizes: Iterable[int],
    *,
    rows: int,
) -> ProcessingSpaceEstimate:
    """Build the estimate from exact WAV sizes of missing and all planned outputs."""
    missing = list(missing_sizes)
    allocated = sum(_cluster_rounded(size) for size in missing)
    largest_temporary = max(
        (_cluster_rounded(size) for size in planned_sizes), default=0
    )
    manifest_reserve = 2 * max(
        MANIFEST_RESERVE_MIN_BYTES, rows * MANIFEST_RESERVE_BYTES_PER_ROW
    )
    subtotal = allocated + largest_temporary + manifest_reserve
    safety = max(
        SAFETY_RESERVE_MIN_BYTES, _ceil_div(subtotal * SAFETY_RESERVE_PERCENT, 100)
    )
    return ProcessingSpaceEstimate(
        rows=rows,
        missing_outputs=len(missing),
        missing_output_bytes=sum(missing),
        missing_output_allocated_bytes=allocated,
        largest_temporary_bytes=largest_temporary,
        manifest_reserve_bytes=manifest_reserve,
        safety_reserve_bytes=safety,
    )


def _planned_wav_sizes(
    manifest: pd.DataFrame,
    layout: DataLayout,
    sample_rate: int,
) -> list[tuple[bool, int]]:
    """(output exists, exact WAV size) for every manifest row, in order."""
    rows = [
        (
            (
                layout.processed_dir(row["split"], row["label"])
                / f"{row['utt_id']}.wav"
            ).exists(),
            Path(row["source_path"]),
        )
        for _, row in manifest.iterrows()
    ]
    if not rows:
        return []
    workers = min(32, max(4, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        payloads = list(
            executor.map(
                lambda row: _source_pcm16_payload(row[1], sample_rate), rows
            )
        )
    return [
        (exists, payload + WAV_PCM16_HEADER_BYTES)
        for (exists, _), payload in zip(rows, payloads, strict=True)
    ]


def _source_pcm16_payload(source: Path, sample_rate: int) -> int:
    try:
        frames, source_rate = _read_frames_and_rate(source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read audio source: {source}") from exc
    if frames <= 0 or source_rate <= 0:
        raise ValueError(
            f"Invalid audio header for {source}: frames={frames}, "
            f"sample_rate={source_rate}"
        )
    return _resampled_frames(frames, source_rate, sample_rate) * 2


def _read_frames_and_rate(path: Path) -> tuple[int, int]:
    with _preserved_times(path):
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as source:
                    return source.getnframes(), source.getframerate()
            except (wave.Error, EOFError):
                pass
        info = sf.info(path)
        return int(info.frames), int(info.samplerate)


@contextmanager
def _preserved_times(path: Path) -> Iterator[None]:
    """Restore atime/mtime after reading, even when the read fails."""
    stat = path.stat()
    try:
        yield
    finally:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _resampled_frames(frames: int, source_rate: int, sample_rate: int) -> int:
    return _ceil_div(frames * sample_rate, source_rate)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _cluster_rounded(size: int) -> int:
    return _ceil_div(size, CLUSTER_BYTES) * CLUSTER_BYTES


def existing_ancestor(path: Path) -> Path:
    # A delegating function, not a re-export: callers and layer contracts
    # identify this storage-level entry point by its defining module.
    return paths.existing_ancestor(path)
