"""Build and verify portable XAI VM bundles."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import zstandard

from jmds_prepare.core.hashing import sha256_file, sha256_file_preserving_times
from jmds_prepare.core.xai_sample import (
    SUPPORTED_LANGUAGES,
    XAI_SAMPLE_COLUMNS,
    XaiSample,
)
from jmds_prepare.pipelines.xai_audit import validate_processed_wav_format
from jmds_prepare.pipelines.xai_dataset import serialize_xai_samples_csv
from jmds_prepare.profiles.xai import XAI_ROWS_PER_LANGUAGE
from jmds_prepare.storage.xai_layout import XaiLayout

BUNDLE_SCHEMA = "jmds_prepare.xai_bundle"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_VERSION = 1
CANONICAL_ROWS_PER_LANGUAGE = XAI_ROWS_PER_LANGUAGE
_LANGUAGES = tuple(SUPPORTED_LANGUAGES)
_CLASS_BY_LABEL = {0: "bonafide", 1: "spoof"}
_ARCHIVE_MIN_OVERHEAD_BYTES = 1024 * 1024


class BundleValidationError(ValueError):
    """Raised when a bundle or source manifest violates the bundle contract."""


@dataclass(frozen=True)
class BundleResult:
    """Summary returned by bundle build and verification operations."""

    output: Path
    file_count: int
    total_bytes: int
    counts_by_language: Mapping[str, int]
    dry_run: bool = False


@dataclass(frozen=True)
class _PlannedSample:
    sample: XaiSample
    source: Path
    relative_destination: str
    size_bytes: int


def build_xai_bundle(
    manifests: Mapping[str, Path],
    output: Path,
    *,
    allow_noncanonical_counts: bool = False,
    dry_run: bool = False,
    archive: bool = False,
) -> BundleResult:
    """Validate three XAI manifests and atomically publish a portable bundle."""
    output = Path(output)
    if set(manifests) != set(_LANGUAGES):
        raise BundleValidationError("manifests must contain exactly eng, por, zho")
    if output.exists():
        raise FileExistsError(f"Bundle output already exists: {output}")
    if archive and not dry_run:
        _require_archive_outputs_absent(output)

    planned_by_language: dict[str, tuple[_PlannedSample, ...]] = {}
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    for language in _LANGUAGES:
        manifest_path = Path(manifests[language])
        planned = _read_source_manifest(manifest_path, expected_language=language)
        for item in planned:
            duplicate_id = item.sample.sample_id in seen_ids
            duplicate_destination = item.relative_destination in seen_destinations
            if duplicate_id or duplicate_destination:
                details = []
                if duplicate_id:
                    details.append(f"Duplicate sample_id: {item.sample.sample_id}")
                if duplicate_destination:
                    details.append(
                        f"Duplicate destination: {item.relative_destination}"
                    )
                raise BundleValidationError("; ".join(details))
            seen_ids.add(item.sample.sample_id)
            seen_destinations.add(item.relative_destination)
        planned_by_language[language] = planned

    counts = {language: len(planned_by_language[language]) for language in _LANGUAGES}
    _validate_counts(counts, allow_noncanonical_counts=allow_noncanonical_counts)
    total_bytes = sum(
        item.size_bytes
        for language in _LANGUAGES
        for item in planned_by_language[language]
    )
    result = BundleResult(
        output=output,
        file_count=sum(counts.values()),
        total_bytes=total_bytes,
        counts_by_language=counts,
        dry_run=dry_run,
    )
    if dry_run:
        return result

    manifest_bytes = sum(Path(path).stat().st_size for path in manifests.values())
    receipt_allowance = max(4096, result.file_count * 512)
    estimated_bundle_bytes = total_bytes + manifest_bytes + receipt_allowance
    archive_allowance = (
        _archive_space_allowance(
            estimated_bundle_bytes,
            entry_count=result.file_count + len(_LANGUAGES) + 1,
        )
        if archive
        else 0
    )
    _require_disk_space(
        output.parent,
        required_bytes=estimated_bundle_bytes + archive_allowance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        receipt_files: list[dict[str, Any]] = []
        manifest_receipts: dict[str, dict[str, Any]] = {}
        bundled_samples: dict[str, tuple[XaiSample, ...]] = {}
        for language in _LANGUAGES:
            samples: list[XaiSample] = []
            for item in planned_by_language[language]:
                destination = _confined_bundle_path(staging, item.relative_destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item.source, destination)
                copied_hash = sha256_file(destination)
                if copied_hash != item.sample.sha256_processed:
                    raise BundleValidationError(
                        f"Copied SHA-256 mismatch for {item.sample.sample_id}"
                    )
                samples.append(
                    replace(item.sample, processed_path=item.relative_destination)
                )
                receipt_files.append(
                    _file_receipt(
                        destination,
                        relative_path=item.relative_destination,
                        kind="audio",
                        sample=item.sample,
                    )
                )
            bundled_samples[language] = tuple(samples)
            manifest_relative = f"manifests/{language}/xai_samples.csv"
            manifest_destination = _confined_bundle_path(staging, manifest_relative)
            manifest_destination.parent.mkdir(parents=True, exist_ok=True)
            manifest_destination.write_bytes(
                serialize_xai_samples_csv(bundled_samples[language])
            )
            manifest_entry = _file_receipt(
                manifest_destination,
                relative_path=manifest_relative,
                kind="manifest",
                language=language,
            )
            receipt_files.append(manifest_entry)
            manifest_receipts[language] = {
                "path": manifest_relative,
                "sha256": manifest_entry["sha256"],
                "size_bytes": manifest_entry["size_bytes"],
                "sample_count": len(samples),
            }

        receipt_files.sort(key=lambda entry: entry["path"])
        receipt = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_version": BUNDLE_VERSION,
            "files": receipt_files,
            "manifests": manifest_receipts,
            "counts": _aggregate_counts(bundled_samples, receipt_files),
        }
        (staging / "bundle_receipt.json").write_text(
            json.dumps(
                receipt,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if output.exists():
            raise FileExistsError(f"Bundle output already exists: {output}")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if archive:
        _create_archive(output)
    return result


def archive_xai_bundle(
    bundle: Path,
    *,
    allow_noncanonical_counts: bool = False,
) -> tuple[Path, Path]:
    """Verify an existing bundle, then publish its archive and hash sidecar."""
    bundle = Path(bundle)
    _require_archive_outputs_absent(bundle)
    verify_xai_bundle(
        bundle,
        allow_noncanonical_counts=allow_noncanonical_counts,
    )
    receipt_path = bundle / "bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_files = receipt["files"]
    verified_bundle_bytes = sum(
        int(entry["size_bytes"]) for entry in receipt_files
    ) + receipt_path.stat().st_size
    _require_disk_space(
        bundle.parent,
        required_bytes=_archive_space_allowance(
            verified_bundle_bytes,
            entry_count=len(receipt_files) + 1,
        ),
    )
    return _create_archive(bundle)


def _create_archive(bundle: Path) -> tuple[Path, Path]:
    archive = bundle.with_name(f"{bundle.name}.tar.zst")
    sidecar = archive.with_name(f"{archive.name}.sha256")
    _require_archive_outputs_absent(bundle)
    archive_temp: Path | None = None
    sidecar_temp: Path | None = None
    archive_published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{archive.name}.tmp-",
            dir=archive.parent,
            delete=False,
        ) as raw_archive:
            archive_temp = Path(raw_archive.name)
            compressor = zstandard.ZstdCompressor()
            with compressor.stream_writer(raw_archive, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    tar.add(
                        bundle,
                        arcname=bundle.name,
                        filter=_normalize_tar_metadata,
                    )
            raw_archive.flush()
            os.fsync(raw_archive.fileno())

        digest = sha256_file(archive_temp)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{sidecar.name}.tmp-",
            dir=sidecar.parent,
            delete=False,
        ) as raw_sidecar:
            sidecar_temp = Path(raw_sidecar.name)
            raw_sidecar.write(f"{digest}  {archive.name}\n".encode("ascii"))
            raw_sidecar.flush()
            os.fsync(raw_sidecar.fileno())

        _publish_without_overwrite(archive_temp, archive)
        archive_published = True
        _publish_without_overwrite(sidecar_temp, sidecar)
    except BaseException:
        if archive_published:
            archive.unlink(missing_ok=True)
        raise
    finally:
        if archive_temp is not None:
            archive_temp.unlink(missing_ok=True)
        if sidecar_temp is not None:
            sidecar_temp.unlink(missing_ok=True)
    return archive, sidecar


def _normalize_tar_metadata(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.mode = 0o755 if member.isdir() else 0o644
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    return member


def _require_archive_outputs_absent(bundle: Path) -> None:
    archive = bundle.with_name(f"{bundle.name}.tar.zst")
    sidecar = archive.with_name(f"{archive.name}.sha256")
    for path in (archive, sidecar):
        if path.exists():
            raise FileExistsError(f"Archive output already exists: {path}")


def _publish_without_overwrite(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem temporary file without replacement."""
    try:
        os.link(source, destination)
    except FileExistsError:
        raise FileExistsError(f"Archive output already exists: {destination}") from None


def verify_xai_bundle(
    bundle: Path,
    *,
    allow_noncanonical_counts: bool = False,
) -> BundleResult:
    """Verify receipt, manifests, files, hashes, WAV format and sample counts."""
    bundle = Path(bundle)
    if not bundle.is_dir():
        raise BundleValidationError(f"Bundle directory does not exist: {bundle}")
    receipt_path = bundle / "bundle_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"Invalid bundle receipt: {receipt_path}") from exc
    _validate_receipt_header(receipt)

    files = receipt.get("files")
    if not isinstance(files, list):
        raise BundleValidationError("Receipt files must be a list")
    seen_paths: set[str] = set()
    receipt_by_path: dict[str, Mapping[str, Any]] = {}
    verified_path_by_relative: dict[str, Path] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise BundleValidationError("Receipt file entries must be objects")
        relative = _require_receipt_text(entry, "path")
        kind = _require_receipt_text(entry, "kind")
        if kind not in {"audio", "manifest"}:
            raise BundleValidationError(
                f"Unsupported receipt kind for {relative}: {kind}"
            )
        if relative in seen_paths:
            raise BundleValidationError(f"Duplicate receipt path: {relative}")
        seen_paths.add(relative)
        receipt_by_path[relative] = entry
        path = _confined_bundle_path(bundle, relative)
        if not path.is_file():
            raise BundleValidationError(f"Missing bundled file: {relative}")
        expected_size = entry.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise BundleValidationError(f"Invalid receipt size for {relative}")
        if path.stat().st_size != expected_size:
            raise BundleValidationError(f"Size mismatch for {relative}")
        expected_hash = _require_receipt_text(entry, "sha256")
        if sha256_file(path) != expected_hash:
            raise BundleValidationError(f"SHA-256 mismatch for {relative}")
        verified_path_by_relative[relative] = path

    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != receipt_path
    }
    if actual_files != seen_paths:
        missing = sorted(seen_paths - actual_files)
        extra = sorted(actual_files - seen_paths)
        raise BundleValidationError(
            f"Receipt file set mismatch; missing={missing}, extra={extra}"
        )

    manifest_metadata = receipt.get("manifests")
    if not isinstance(manifest_metadata, dict) or set(manifest_metadata) != set(
        _LANGUAGES
    ):
        raise BundleValidationError("Receipt must describe exactly eng, por, zho manifests")

    samples_by_language: dict[str, tuple[XaiSample, ...]] = {}
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    for language in _LANGUAGES:
        expected_manifest_path = f"manifests/{language}/xai_samples.csv"
        metadata = manifest_metadata[language]
        if not isinstance(metadata, dict) or metadata.get("path") != expected_manifest_path:
            raise BundleValidationError(f"Invalid manifest receipt for {language}")
        file_entry = receipt_by_path.get(expected_manifest_path)
        if file_entry is None or file_entry.get("kind") != "manifest":
            raise BundleValidationError(f"Manifest missing from receipt files: {language}")
        for field in ("sha256", "size_bytes"):
            if metadata.get(field) != file_entry.get(field):
                raise BundleValidationError(
                    f"Manifest receipt {field} mismatch for {language}"
                )
        samples = _read_bundled_manifest(
            bundle / expected_manifest_path,
            expected_language=language,
            receipt_by_path=receipt_by_path,
            verified_path_by_relative=verified_path_by_relative,
        )
        if metadata.get("sample_count") != len(samples):
            raise BundleValidationError(f"Manifest sample_count mismatch for {language}")
        for sample in samples:
            expected = _bundle_relative_destination(sample)
            if sample.sample_id in seen_ids:
                raise BundleValidationError(f"Duplicate sample_id: {sample.sample_id}")
            if expected in seen_destinations:
                raise BundleValidationError(f"Duplicate destination: {expected}")
            seen_ids.add(sample.sample_id)
            seen_destinations.add(expected)
        samples_by_language[language] = samples

    expected_receipt_paths = seen_destinations | {
        f"manifests/{language}/xai_samples.csv" for language in _LANGUAGES
    }
    if seen_paths != expected_receipt_paths:
        missing = sorted(expected_receipt_paths - seen_paths)
        extra = sorted(seen_paths - expected_receipt_paths)
        raise BundleValidationError(
            f"Receipt paths mismatch; missing={missing}, extra={extra}"
        )

    counts = {language: len(samples_by_language[language]) for language in _LANGUAGES}
    _validate_counts(counts, allow_noncanonical_counts=allow_noncanonical_counts)
    expected_counts = _aggregate_counts(samples_by_language, list(files))
    if receipt.get("counts") != expected_counts:
        raise BundleValidationError("Receipt aggregate counts mismatch")
    total_bytes = sum(
        int(receipt_by_path[relative]["size_bytes"])
        for relative in seen_destinations
    )
    return BundleResult(
        output=bundle,
        file_count=sum(counts.values()),
        total_bytes=total_bytes,
        counts_by_language=counts,
    )


def _read_source_manifest(
    manifest_path: Path,
    *,
    expected_language: str,
) -> tuple[_PlannedSample, ...]:
    rows = _read_csv_rows(manifest_path)
    planned: list[_PlannedSample] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            sample = XaiSample.from_dict(row, enforce_column_order=True)
        except ValueError as exc:
            raise BundleValidationError(
                f"Invalid manifest {manifest_path} row {row_number}: {exc}"
            ) from exc
        if sample.language != expected_language:
            raise BundleValidationError(
                f"Manifest language mismatch for {sample.sample_id}: "
                f"expected {expected_language}, got {sample.language}"
            )
        source = _resolve_source_path(manifest_path, sample.processed_path)
        if not source.is_file():
            raise BundleValidationError(f"Missing processed file: {source}")
        actual_hash = sha256_file_preserving_times(source)
        if actual_hash != sample.sha256_processed:
            raise BundleValidationError(
                f"sha256_processed mismatch for {sample.sample_id}"
            )
        try:
            validate_processed_wav_format(source)
        except ValueError as exc:
            raise BundleValidationError(str(exc)) from exc
        relative_destination = _bundle_relative_destination(sample)
        planned.append(
            _PlannedSample(
                sample=sample,
                source=source,
                relative_destination=relative_destination,
                size_bytes=source.stat().st_size,
            )
        )
    return tuple(planned)


def _read_bundled_manifest(
    manifest_path: Path,
    *,
    expected_language: str,
    receipt_by_path: Mapping[str, Mapping[str, Any]],
    verified_path_by_relative: Mapping[str, Path],
) -> tuple[XaiSample, ...]:
    rows = _read_csv_rows(manifest_path)
    samples: list[XaiSample] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            sample = XaiSample.from_dict(row, enforce_column_order=True)
        except ValueError as exc:
            raise BundleValidationError(
                f"Invalid manifest {manifest_path} row {row_number}: {exc}"
            ) from exc
        if sample.language != expected_language:
            raise BundleValidationError(
                f"Manifest language mismatch for {sample.sample_id}"
            )
        expected_relative = _bundle_relative_destination(sample)
        if sample.processed_path != expected_relative:
            raise BundleValidationError(
                f"Noncanonical processed_path for {sample.sample_id}: "
                f"{sample.processed_path}"
            )
        entry = receipt_by_path.get(expected_relative)
        if entry is None or entry.get("kind") != "audio":
            raise BundleValidationError(
                f"Audio missing from receipt for {sample.sample_id}"
            )
        for field, expected in (
            ("language", sample.language),
            ("role", sample.role),
            ("class", _CLASS_BY_LABEL[sample.label]),
            ("sample_id", sample.sample_id),
        ):
            if entry.get(field) != expected:
                raise BundleValidationError(
                    f"Receipt {field} mismatch for {sample.sample_id}"
                )
        if entry.get("sha256") != sample.sha256_processed:
            raise BundleValidationError(
                f"Manifest SHA-256 mismatch for {sample.sample_id}"
            )
        path = verified_path_by_relative[expected_relative]
        try:
            validate_processed_wav_format(path)
        except ValueError as exc:
            raise BundleValidationError(str(exc)) from exc
        samples.append(sample)
    return tuple(samples)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BundleValidationError(f"Missing manifest: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != XAI_SAMPLE_COLUMNS:
                missing = [
                    column
                    for column in XAI_SAMPLE_COLUMNS
                    if column not in (reader.fieldnames or [])
                ]
                if missing:
                    raise BundleValidationError(
                        "Missing required columns: " + ", ".join(missing)
                    )
                raise BundleValidationError(
                    "Column order does not match the canonical schema"
                )
            return [dict(row) for row in reader]
    except UnicodeError as exc:
        raise BundleValidationError(f"Manifest is not UTF-8: {path}") from exc


def _resolve_source_path(manifest_path: Path, value: str) -> Path:
    source = Path(value)
    if source.is_absolute():
        return source
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "bundle_receipt.json").is_file():
            return _confined_bundle_path(candidate, PurePosixPath(value).as_posix())
    raise BundleValidationError(
        "Relative processed_path requires bundle_receipt.json at or above "
        f"the manifest directory: {value}"
    )


def _bundle_relative_destination(sample: XaiSample) -> str:
    layout = XaiLayout(Path("."))
    destination = layout.sample_path(
        sample.language,
        sample.role,
        sample.label,
        sample.sample_id,
    )
    return destination.as_posix().removeprefix("./")


def _confined_bundle_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise BundleValidationError("Bundle-relative path must be non-empty")
    if "\\" in relative:
        raise BundleValidationError(f"Bundle path must use POSIX separators: {relative}")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts:
        raise BundleValidationError(f"Bundle path is not confined: {relative}")
    destination = root.joinpath(*posix.parts)
    resolved_root = root.resolve()
    resolved = destination.resolve()
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise BundleValidationError(f"Bundle path escapes root: {relative}")
    return destination


def _file_receipt(
    path: Path,
    *,
    relative_path: str,
    kind: str,
    sample: XaiSample | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": relative_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "kind": kind,
    }
    if sample is not None:
        entry.update(
            {
                "language": sample.language,
                "role": sample.role,
                "class": _CLASS_BY_LABEL[sample.label],
                "sample_id": sample.sample_id,
            }
        )
    elif language is not None:
        entry["language"] = language
    return entry


def _aggregate_counts(
    samples_by_language: Mapping[str, Sequence[XaiSample]],
    files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    samples = [
        sample
        for language in _LANGUAGES
        for sample in samples_by_language[language]
    ]
    return {
        "samples_total": len(samples),
        "samples_by_language": {
            language: len(samples_by_language[language]) for language in _LANGUAGES
        },
        "samples_by_role": dict(
            sorted(Counter(sample.role for sample in samples).items())
        ),
        "samples_by_class": {
            class_name: sum(
                1
                for sample in samples
                if _CLASS_BY_LABEL[sample.label] == class_name
            )
            for class_name in ("bonafide", "spoof")
        },
        "receipt_file_count": len(files),
        "receipt_file_bytes": sum(int(entry["size_bytes"]) for entry in files),
    }


def _validate_counts(
    counts: Mapping[str, int],
    *,
    allow_noncanonical_counts: bool,
) -> None:
    if allow_noncanonical_counts:
        return
    for language in _LANGUAGES:
        actual = counts[language]
        if actual != CANONICAL_ROWS_PER_LANGUAGE:
            raise BundleValidationError(
                f"Expected {CANONICAL_ROWS_PER_LANGUAGE} samples for {language}, "
                f"got {actual}; use allow_noncanonical_counts only for fixtures "
                "and development"
            )


def _require_disk_space(path: Path, *, required_bytes: int) -> None:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = shutil.disk_usage(probe).free
    if available < required_bytes:
        raise BundleValidationError(
            f"Insufficient disk space: need {required_bytes} bytes, "
            f"have {available} bytes"
        )


def _archive_space_allowance(bundle_bytes: int, *, entry_count: int) -> int:
    overhead = max(
        _ARCHIVE_MIN_OVERHEAD_BYTES,
        bundle_bytes // 100,
        entry_count * 4096,
    )
    return bundle_bytes + overhead


def _validate_receipt_header(receipt: Any) -> None:
    if not isinstance(receipt, dict):
        raise BundleValidationError("Bundle receipt must be a JSON object")
    if receipt.get("schema") != BUNDLE_SCHEMA:
        raise BundleValidationError("Unsupported bundle receipt schema")
    if receipt.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleValidationError("Unsupported bundle receipt schema_version")
    if receipt.get("bundle_version") != BUNDLE_VERSION:
        raise BundleValidationError("Unsupported bundle_version")


def _require_receipt_text(entry: Mapping[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"Invalid receipt {field}")
    return value
