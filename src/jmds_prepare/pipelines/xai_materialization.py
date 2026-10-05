"""Selective XAI materialization with archive preflight and content-idempotent outputs."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Literal

from jmds_prepare.audio import AudioMetadata, process_audio
from jmds_prepare.core.hashing import sha256_file, sha256_file_preserving_times
from jmds_prepare.core.xai_sample import XaiSample
from jmds_prepare.pipelines.xai_audit import validate_processed_wav_format
from jmds_prepare.pipelines.xai_selection import SelectedCandidate, SelectionCandidate
from jmds_prepare.storage.xai_layout import XaiLayout, slugify_token

ArchiveKind = Literal["tar", "zip", "rar"]
MaterializationMode = Literal["reuse_processed", "process_source", "extract_archive"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_DIGEST_LENGTH = 12
_MAX_SAMPLE_ID_LENGTH = 200
ArchiveSpecKey = tuple[str, str]
ExtractedPathKey = tuple[str, str, str]

# CORAA train materialization requires UnRAR (or UnRAR.exe) and all five
# multi-volume parts from the pinned CORAA v1.1 catalog:
# train.part1.rar .. train.part5.rar in the same directory.
CORAA_TRAIN_RAR_VOLUME_NAMES: tuple[str, ...] = tuple(
    f"train.part{index}.rar" for index in range(1, 6)
)


class MaterializationError(ValueError):
    """Base error for selective XAI materialization."""


class PreflightError(MaterializationError):
    """Raised when preflight rejects one or more selected candidates."""


class ArchiveResolutionError(PreflightError):
    """Raised when an archive member cannot be resolved 1:1."""


class DestinationConflictError(MaterializationError, FileExistsError):
    """Raised when a destination exists with different content."""


@dataclass(frozen=True)
class ArchiveSpec:
    """Archive location(s) and container kind for one corpus/native split pair."""

    corpus: str
    native_split: str
    kind: ArchiveKind
    volume_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        paths = tuple(Path(path) for path in self.volume_paths)
        object.__setattr__(self, "volume_paths", paths)
        for field in ("corpus", "native_split"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must be non-empty")
        if self.kind not in {"tar", "zip", "rar"}:
            raise ValueError(f"Unsupported archive kind: {self.kind}")
        if not paths:
            raise ValueError("volume_paths must be non-empty")
        resolved = [path.resolve() for path in paths]
        if len(resolved) != len(set(resolved)):
            raise ValueError("volume_paths must be unique")
        if self.kind in {"tar", "zip"} and len(paths) != 1:
            raise ValueError(f"{self.kind} archives require exactly one volume path")
        if self.kind == "rar" and not paths[0].name.lower().endswith(".rar"):
            raise ValueError("RAR volume_paths must start with a .rar part1 archive")

    @property
    def archive_path(self) -> Path:
        """Primary archive path (part1 for RAR, sole path for tar/zip)."""
        return self.volume_paths[0]


def standard_archive_spec(
    *,
    corpus: str,
    native_split: str,
    archive_path: Path,
) -> ArchiveSpec:
    """Build the canonical archive kind and volume set for known XAI corpora."""
    archive_path = Path(archive_path)
    if corpus == "AISHELL-3":
        return ArchiveSpec(corpus, native_split, "tar", (archive_path,))
    if corpus == "CORAA":
        if native_split in {"dev", "test"}:
            return ArchiveSpec(corpus, native_split, "zip", (archive_path,))
        if native_split == "train":
            directory = archive_path.parent
            volume_paths = tuple(
                directory / name for name in CORAA_TRAIN_RAR_VOLUME_NAMES
            )
            return ArchiveSpec(corpus, native_split, "rar", volume_paths)
        raise ValueError(f"Unsupported CORAA native split: {native_split}")
    raise ValueError(f"Unsupported archive corpus: {corpus}")


@dataclass(frozen=True)
class ArchiveMemberRecord:
    member_name: str
    member_size: int
    volume_path: Path | None = None


@dataclass(frozen=True)
class ArchiveInventory:
    members_by_ref: Mapping[str, ArchiveMemberRecord]
    duplicate_refs: tuple[str, ...]
    rejected_unsafe_members: tuple[str, ...]

    def resolve_ref(self, original_ref: str) -> ArchiveMemberRecord:
        if original_ref in self.duplicate_refs:
            raise ArchiveResolutionError(
                f"Duplicate archive member for original_ref: {original_ref}"
            )
        try:
            return self.members_by_ref[original_ref]
        except KeyError as exc:
            raise ArchiveResolutionError(
                f"Missing archive member for original_ref: {original_ref}"
            ) from exc


@dataclass(frozen=True)
class PreflightReport:
    archive_inventories: Mapping[ArchiveSpecKey, ArchiveInventory]


@dataclass(frozen=True)
class _ProcessedOutput:
    item: SelectedCandidate
    sample_id: str
    staging_wav: Path
    destination: Path
    metadata: AudioMetadata


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


def default_subprocess_runner(*args, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(*args, **kwargs)


def derive_sample_id(
    *,
    language: str,
    role: str,
    label: int,
    corpus: str,
    candidate_id: str,
) -> str:
    """Return a stable, injective filesystem-safe sample identifier."""
    identity = f"{language}:{role}:{label}:{corpus}:{candidate_id}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:_IDENTITY_DIGEST_LENGTH]
    label_slug = "bonafide" if label == 0 else "spoof"
    prefix = (
        f"{language}-{role}-{label_slug}-"
        f"{slugify_token(corpus)}-{slugify_token(candidate_id)}"
    )
    suffix = f"-{digest}"
    max_prefix_length = _MAX_SAMPLE_ID_LENGTH - len(suffix)
    if len(prefix) > max_prefix_length:
        prefix = prefix[:max_prefix_length].rstrip("-")
    sample_id = f"{prefix}{suffix}"
    XaiLayout.validate_sample_id(sample_id)
    return sample_id


def inventory_archive_spec(
    spec: ArchiveSpec,
    *,
    archive_runner: SubprocessRunner = default_subprocess_runner,
    unrar_executable: str = "UnRAR",
) -> ArchiveInventory:
    _validate_archive_volumes(spec)
    if spec.kind == "tar":
        return _inventory_tar(spec.archive_path)
    if spec.kind == "zip":
        return _inventory_zip(spec.archive_path)
    if spec.kind == "rar":
        return _inventory_rar(
            spec,
            runner=archive_runner,
            executable=unrar_executable,
        )
    raise ValueError(f"Unsupported archive kind: {spec.kind}")


def preflight_materialization(
    selected: Sequence[SelectedCandidate],
    layout: XaiLayout,
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    *,
    archive_runner: SubprocessRunner = default_subprocess_runner,
    unrar_executable: str = "UnRAR",
) -> PreflightReport:
    """Validate every selected candidate before any cache write occurs."""
    issues: list[str] = []
    inventories: dict[ArchiveSpecKey, ArchiveInventory] = {}

    for key in _required_archive_keys(selected):
        spec = archive_specs.get(key)
        if spec is None:
            issues.append(f"Missing ArchiveSpec for {key[0]}/{key[1]}")
            continue
        try:
            _validate_archive_volumes(spec)
        except PreflightError as exc:
            issues.append(str(exc))
            continue
        try:
            inventories[key] = inventory_archive_spec(
                spec,
                archive_runner=archive_runner,
                unrar_executable=unrar_executable,
            )
        except PreflightError as exc:
            issues.append(str(exc))
        except (OSError, subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
            issues.append(f"Archive listing failed for {key[0]}/{key[1]}: {exc}")

    seen_candidate_ids: set[str] = set()
    seen_original_refs: set[str] = set()
    seen_sample_ids: set[str] = set()
    seen_destinations: set[Path] = set()

    for item in selected:
        candidate = item.candidate
        _validate_materialization_mode(candidate.materialization_mode)

        if candidate.candidate_id in seen_candidate_ids:
            issues.append(f"Duplicate candidate_id: {candidate.candidate_id}")
        seen_candidate_ids.add(candidate.candidate_id)

        if candidate.original_ref in seen_original_refs:
            issues.append(f"Duplicate original_ref: {candidate.original_ref}")
        seen_original_refs.add(candidate.original_ref)

        sample_id = derive_sample_id(
            language=candidate.language,
            role=item.role,
            label=candidate.label,
            corpus=candidate.corpus,
            candidate_id=candidate.candidate_id,
        )
        if sample_id in seen_sample_ids:
            issues.append(f"Duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)

        if candidate.materialization_mode == "extract_archive":
            key = (candidate.corpus, candidate.native_split)
            spec = archive_specs.get(key)
            inventory = inventories.get(key)
            if _is_unsafe_member_path(candidate.original_ref):
                issues.append(
                    f"Unsafe original_ref for {candidate.candidate_id}: "
                    f"{candidate.original_ref}"
                )
                continue
            if spec is None:
                issues.append(
                    f"Missing ArchiveSpec for archive candidate {candidate.candidate_id}"
                )
                continue
            if inventory is None:
                continue
            if candidate.original_ref in inventory.rejected_unsafe_members:
                issues.append(
                    f"Unsafe archive member for {candidate.candidate_id}: "
                    f"{candidate.original_ref}"
                )
                continue
            if candidate.original_ref in inventory.duplicate_refs:
                issues.append(
                    f"Duplicate archive member for {candidate.candidate_id}: "
                    f"{candidate.original_ref}"
                )
                continue
            if candidate.original_ref not in inventory.members_by_ref:
                issues.append(
                    f"Missing archive member for {candidate.candidate_id}: "
                    f"{candidate.original_ref}"
                )
                continue
        elif candidate.materialization_mode == "reuse_processed":
            original_source = Path(candidate.original_ref)
            if not original_source.is_file():
                issues.append(
                    f"Missing original source for {candidate.candidate_id}: "
                    f"{original_source}"
                )
                continue
            processed = Path(candidate.local_source_path)
            if not processed.is_file():
                issues.append(
                    f"Missing local source for {candidate.candidate_id}: {processed}"
                )
                continue
            if candidate.expected_sha256_source is not None:
                try:
                    actual_source = sha256_file_preserving_times(original_source)
                except OSError as exc:
                    issues.append(
                        f"Could not hash source for {candidate.candidate_id}: {exc}"
                    )
                    continue
                if actual_source != candidate.expected_sha256_source:
                    issues.append(
                        f"sha256_source mismatch for {candidate.candidate_id}"
                    )
            if candidate.expected_sha256_processed is not None:
                try:
                    actual_processed = sha256_file_preserving_times(processed)
                except OSError as exc:
                    issues.append(
                        f"Could not hash processed audio for {candidate.candidate_id}: "
                        f"{exc}"
                    )
                    continue
                if actual_processed != candidate.expected_sha256_processed:
                    issues.append(
                        f"sha256_processed mismatch for {candidate.candidate_id}"
                    )
            try:
                validate_processed_wav_format(processed)
            except ValueError as exc:
                issues.append(
                    f"Invalid reused processed audio for {candidate.candidate_id}: {exc}"
                )
        elif candidate.local_source_path is None:
            issues.append(
                f"Missing local_source_path for {candidate.candidate_id} "
                f"({candidate.materialization_mode})"
            )
            continue
        else:
            source = Path(candidate.local_source_path)
            if not source.is_file():
                issues.append(
                    f"Missing local source for {candidate.candidate_id}: {source}"
                )
                continue
            if (
                candidate.materialization_mode == "process_source"
                and candidate.expected_sha256_source is not None
            ):
                try:
                    actual = sha256_file_preserving_times(source)
                except OSError as exc:
                    issues.append(
                        f"Could not hash source for {candidate.candidate_id}: {exc}"
                    )
                    continue
                if actual != candidate.expected_sha256_source:
                    issues.append(
                        f"sha256_source mismatch for {candidate.candidate_id}"
                    )

        if candidate.materialization_mode != "reuse_processed":
            destination = layout.sample_path(
                candidate.language,
                item.role,
                candidate.label,
                sample_id,
            )
            resolved_destination = destination.resolve()
            if resolved_destination in seen_destinations:
                issues.append(f"Duplicate destination path: {resolved_destination}")
            seen_destinations.add(resolved_destination)

    if issues:
        raise PreflightError("; ".join(sorted(set(issues))))

    return PreflightReport(archive_inventories=inventories)


def materialize_selected(
    selected: Sequence[SelectedCandidate],
    layout: XaiLayout,
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    *,
    sample_rate: int = 16_000,
    archive_runner: SubprocessRunner = default_subprocess_runner,
    unrar_executable: str = "UnRAR",
) -> tuple[XaiSample, ...]:
    """Materialize selected candidates into canonical XAI samples."""
    report = preflight_materialization(
        selected,
        layout,
        archive_specs,
        archive_runner=archive_runner,
        unrar_executable=unrar_executable,
    )
    ordered = _canonical_materialization_order(selected)
    layout.staging_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=layout.staging_dir,
        prefix="run-",
    ) as run_staging_name:
        run_staging = Path(run_staging_name)
        extracted_paths = _batch_extract_archives(
            ordered,
            report=report,
            archive_specs=archive_specs,
            run_staging=run_staging,
            archive_runner=archive_runner,
            unrar_executable=unrar_executable,
        )

        reuse_samples: dict[tuple[str, int], XaiSample] = {}
        processed_outputs: list[_ProcessedOutput] = []

        for item in ordered:
            candidate = item.candidate
            sample_id = derive_sample_id(
                language=candidate.language,
                role=item.role,
                label=candidate.label,
                corpus=candidate.corpus,
                candidate_id=candidate.candidate_id,
            )
            if candidate.materialization_mode == "reuse_processed":
                reuse_samples[(item.candidate.candidate_id, item.selection_rank)] = (
                    _materialize_reuse_processed(item, sample_id=sample_id)
                )
                continue

            destination = layout.sample_path(
                candidate.language,
                item.role,
                candidate.label,
                sample_id,
            )
            staging_wav = run_staging / "processed" / f"{sample_id}.wav"
            if candidate.materialization_mode == "process_source":
                source = Path(candidate.local_source_path)
            elif candidate.materialization_mode == "extract_archive":
                key: ExtractedPathKey = (
                    candidate.corpus,
                    candidate.native_split,
                    candidate.original_ref,
                )
                source = extracted_paths[key]
            else:
                raise MaterializationError(
                    f"Unsupported materialization_mode: {candidate.materialization_mode}"
                )

            metadata = process_audio(source, staging_wav, sample_rate)
            processed_outputs.append(
                _ProcessedOutput(
                    item=item,
                    sample_id=sample_id,
                    staging_wav=staging_wav,
                    destination=destination,
                    metadata=metadata,
                )
            )

        _preflight_publish_conflicts(processed_outputs)
        _publish_processed_outputs(processed_outputs)

        samples: list[XaiSample] = []
        for item in ordered:
            candidate = item.candidate
            key = (candidate.candidate_id, item.selection_rank)
            if candidate.materialization_mode == "reuse_processed":
                samples.append(reuse_samples[key])
                continue
            output = next(
                planned
                for planned in processed_outputs
                if planned.item.candidate.candidate_id == candidate.candidate_id
                and planned.item.selection_rank == item.selection_rank
            )
            samples.append(
                _build_xai_sample(
                    output.item,
                    sample_id=output.sample_id,
                    processed_path=str(output.destination),
                    sha256_source=output.metadata.sha256_source,
                    sha256_processed=output.metadata.sha256_processed,
                )
            )
        return tuple(samples)


def _batch_extract_archives(
    ordered: Sequence[SelectedCandidate],
    *,
    report: PreflightReport,
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    run_staging: Path,
    archive_runner: SubprocessRunner,
    unrar_executable: str,
) -> dict[ExtractedPathKey, Path]:
    grouped_refs: dict[ArchiveSpecKey, list[str]] = {}
    for item in ordered:
        candidate = item.candidate
        if candidate.materialization_mode != "extract_archive":
            continue
        key = (candidate.corpus, candidate.native_split)
        grouped_refs.setdefault(key, []).append(candidate.original_ref)

    extracted: dict[ExtractedPathKey, Path] = {}
    for key, refs in grouped_refs.items():
        spec = archive_specs[key]
        inventory = report.archive_inventories[key]
        staging_root = (
            run_staging
            / "archives"
            / slugify_token(key[0])
            / slugify_token(key[1])
        )
        batch = _extract_archive_batch(
            spec,
            refs,
            inventory,
            staging_root,
            archive_runner=archive_runner,
            unrar_executable=unrar_executable,
        )
        for ref, path in batch.items():
            extracted[(key[0], key[1], ref)] = path
    return extracted


def _extract_archive_batch(
    spec: ArchiveSpec,
    refs: Sequence[str],
    inventory: ArchiveInventory,
    staging_root: Path,
    *,
    archive_runner: SubprocessRunner,
    unrar_executable: str,
) -> dict[str, Path]:
    members_by_ref: dict[str, ArchiveMemberRecord] = {}
    for ref in refs:
        members_by_ref[ref] = inventory.resolve_ref(ref)

    if spec.kind == "rar":
        members_by_volume: dict[Path, list[ArchiveMemberRecord]] = {}
        for member in members_by_ref.values():
            if member.volume_path is None:
                raise ArchiveResolutionError(
                    f"RAR member is missing its volume path: {member.member_name}"
                )
            members_by_volume.setdefault(member.volume_path, []).append(member)

        extracted_by_ref: dict[str, Path] = {}
        for volume_path, members in members_by_volume.items():
            member_names = list(
                dict.fromkeys(member.member_name for member in members)
            )
            extracted_members = _extract_rar_members(
                volume_path,
                member_names=member_names,
                destination_dir=staging_root,
                archive_runner=archive_runner,
                executable=unrar_executable,
            )
            for ref, member in members_by_ref.items():
                if member.volume_path == volume_path:
                    extracted_by_ref[ref] = extracted_members[member.member_name]
        return extracted_by_ref

    member_names: list[str] = []
    ref_by_member: dict[str, str] = {}
    for ref, member in members_by_ref.items():
        member_names.append(member.member_name)
        ref_by_member[member.member_name] = ref

    unique_members = list(dict.fromkeys(member_names))
    if spec.kind == "tar":
        extracted_members = _extract_tar_members_batch(
            spec.archive_path,
            unique_members,
            staging_root,
        )
    elif spec.kind == "zip":
        extracted_members = _extract_zip_members_batch(
            spec.archive_path,
            unique_members,
            staging_root,
        )
    else:
        raise ValueError(f"Unsupported archive kind: {spec.kind}")

    return {
        ref_by_member[member_name]: extracted_members[member_name]
        for member_name in member_names
    }


def _materialize_reuse_processed(
    item: SelectedCandidate,
    *,
    sample_id: str,
) -> XaiSample:
    candidate = item.candidate
    processed = Path(candidate.local_source_path)
    sha256_processed = candidate.expected_sha256_processed
    if sha256_processed is None:
        sha256_processed = sha256_file_preserving_times(processed)
    sha256_source = candidate.expected_sha256_source
    if sha256_source is None:
        sha256_source = sha256_file_preserving_times(Path(candidate.original_ref))
    return _build_xai_sample(
        item,
        sample_id=sample_id,
        processed_path=str(processed),
        sha256_source=sha256_source,
        sha256_processed=sha256_processed,
    )


def _preflight_publish_conflicts(outputs: Sequence[_ProcessedOutput]) -> None:
    for output in outputs:
        destination = output.destination
        if not destination.exists():
            continue
        existing_hash = sha256_file(destination)
        if existing_hash != output.metadata.sha256_processed:
            raise DestinationConflictError(
                f"Destination exists with different content: {destination}"
            )


def _publish_processed_outputs(outputs: Sequence[_ProcessedOutput]) -> None:
    newly_published: list[Path] = []
    try:
        for output in outputs:
            destination = output.destination
            if (
                destination.exists()
                and sha256_file(destination) == output.metadata.sha256_processed
            ):
                continue
            _publish_temporary(output.staging_wav, destination)
            newly_published.append(destination)
    except BaseException:
        for destination in newly_published:
            destination.unlink(missing_ok=True)
        raise


def _build_xai_sample(
    item: SelectedCandidate,
    *,
    sample_id: str,
    processed_path: str,
    sha256_source: str,
    sha256_processed: str,
) -> XaiSample:
    candidate = item.candidate
    return XaiSample(
        sample_id=sample_id,
        language=candidate.language,
        role=item.role,
        label=candidate.label,
        corpus=candidate.corpus,
        native_split=candidate.native_split,
        original_ref=candidate.original_ref,
        processed_path=processed_path,
        speaker_id=candidate.speaker_id,
        group_id=candidate.group_id,
        attack_id=candidate.attack_id,
        sha256_source=_coerce_sha256(sha256_source, field="sha256_source"),
        sha256_processed=_coerce_sha256(sha256_processed, field="sha256_processed"),
        selection_seed=item.selection_seed,
        selection_rank=item.selection_rank,
        selection_reason=item.selection_reason,
        selection_source=item.selection_source,
    )


def _extract_tar_members_batch(
    archive_path: Path,
    member_names: Sequence[str],
    staging_root: Path,
) -> dict[str, Path]:
    staging_root.mkdir(parents=True, exist_ok=True)
    destinations = {
        member_name: _staging_member_path(staging_root, member_name)
        for member_name in member_names
    }
    missing = [
        member_name
        for member_name in member_names
        if not destinations[member_name].is_file()
    ]
    if not missing:
        return destinations

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member_name in missing:
            member = archive.getmember(member_name)
            if not member.isreg() or _is_unsafe_member_path(member.name):
                raise ArchiveResolutionError(
                    f"Unsafe or non-regular member: {member_name}"
                )
            destination = destinations[member_name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ArchiveResolutionError(f"Could not read TAR member: {member_name}")
            temporary_path: Path | None = None
            try:
                with closing(source):
                    temporary_path = _write_temporary_copy(
                        source,
                        destination.parent,
                        destination.name,
                    )
                _publish_temporary(temporary_path, destination)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    for member_name in member_names:
        destination = destinations[member_name]
        if not destination.is_file():
            raise ArchiveResolutionError(
                f"TAR extraction did not produce member: {member_name}"
            )
    return destinations


def _extract_zip_members_batch(
    archive_path: Path,
    member_names: Sequence[str],
    staging_root: Path,
) -> dict[str, Path]:
    staging_root.mkdir(parents=True, exist_ok=True)
    destinations = {
        member_name: _staging_member_path(staging_root, member_name)
        for member_name in member_names
    }
    missing = [
        member_name
        for member_name in member_names
        if not destinations[member_name].is_file()
    ]
    if not missing:
        return destinations

    with zipfile.ZipFile(archive_path) as archive:
        for member_name in missing:
            try:
                info = archive.getinfo(member_name)
            except KeyError as exc:
                raise ArchiveResolutionError(
                    f"Missing ZIP member: {member_name}"
                ) from exc
            if info.is_dir() or _is_unsafe_member_path(member_name):
                raise ArchiveResolutionError(
                    f"Unsafe or non-regular member: {member_name}"
                )
            destination = destinations[member_name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with archive.open(info) as source:
                    temporary_path = _write_temporary_copy(
                        source,
                        destination.parent,
                        destination.name,
                    )
                _publish_temporary(temporary_path, destination)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    for member_name in member_names:
        destination = destinations[member_name]
        if not destination.is_file():
            raise ArchiveResolutionError(
                f"ZIP extraction did not produce member: {member_name}"
            )
    return destinations


def _extract_rar_members(
    volume_path: Path,
    *,
    member_names: Sequence[str],
    destination_dir: Path,
    archive_runner: SubprocessRunner,
    executable: str,
) -> dict[str, Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        member_name: _staging_member_path(
            destination_dir,
            _normalize_rar_member_path(member_name),
        )
        for member_name in member_names
    }
    missing = [
        member_name
        for member_name in member_names
        if not destinations[member_name].is_file()
    ]
    if not missing:
        return destinations

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination_dir,
        prefix=".extract-list.",
        suffix=".txt",
        delete=False,
    ) as list_file:
        list_path = Path(list_file.name)
        list_file.write("\n".join(missing))
        list_file.flush()
        os.fsync(list_file.fileno())

    try:
        completed = _run_unrar(
            archive_runner,
            [
                executable,
                "x",
                "-o-",
                "-p-",
                "-idq",
                str(volume_path),
                f"@{list_path}",
                _unrar_destination_path(destination_dir),
            ],
        )
        if completed.returncode != 0:
            raise MaterializationError(
                f"Selective RAR extraction failed for {volume_path}: "
                f"{completed.stderr or completed.stdout}"
            )
    finally:
        list_path.unlink(missing_ok=True)

    for member_name in member_names:
        destination = destinations[member_name]
        if not destination.is_file():
            raise ArchiveResolutionError(
                f"RAR extraction did not produce member: {member_name}"
            )
    return destinations


def _inventory_rar(
    spec: ArchiveSpec,
    *,
    runner: SubprocessRunner,
    executable: str,
) -> ArchiveInventory:
    members_by_ref: dict[str, list[tuple[str, Path]]] = {}
    unsafe: list[str] = []
    for volume_path in spec.volume_paths:
        completed = _run_unrar(
            runner,
            [executable, "lb", "-p-", str(volume_path)],
        )
        if completed.returncode != 0:
            raise PreflightError(
                f"Archive listing failed for {volume_path.name}: "
                f"{completed.stderr or completed.stdout}"
            )
        for line in completed.stdout.splitlines():
            member_name = line.strip()
            if not member_name:
                continue
            normalized_ref = _normalize_rar_member_path(member_name)
            if _is_unsafe_member_path(normalized_ref):
                unsafe.append(member_name)
                continue
            members_by_ref.setdefault(normalized_ref, []).append(
                (member_name, volume_path)
            )
    duplicate_refs = tuple(
        sorted(ref for ref, members in members_by_ref.items() if len(members) > 1)
    )
    records = {
        ref: ArchiveMemberRecord(
            member_name=members[0][0],
            member_size=0,
            volume_path=members[0][1],
        )
        for ref, members in members_by_ref.items()
        if len(members) == 1
    }
    return ArchiveInventory(
        members_by_ref=records,
        duplicate_refs=duplicate_refs,
        rejected_unsafe_members=tuple(unsafe),
    )


def _validate_archive_volumes(spec: ArchiveSpec) -> None:
    missing = [path for path in spec.volume_paths if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise PreflightError(f"Missing archive volume(s): {joined}")


def _run_unrar(
    runner: SubprocessRunner,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreflightError(
            f"UnRAR executable not found: {args[0]}"
        ) from exc


def _unrar_destination_path(destination_dir: Path) -> str:
    text = str(destination_dir)
    if os.name == "nt" and not text.endswith("\\"):
        return f"{text}\\"
    return text


def _normalize_rar_member_path(member_name: str) -> str:
    return member_name.replace("\\", "/")


def _staging_member_path(staging_root: Path, member_name: str) -> Path:
    if _is_unsafe_member_path(member_name):
        raise ArchiveResolutionError(f"Unsafe archive member path: {member_name}")
    destination = staging_root.joinpath(*PurePosixPath(member_name).parts)
    resolved = destination.resolve()
    resolved_root = staging_root.resolve()
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise ArchiveResolutionError(
            f"Archive member escapes staging root: {member_name}"
        )
    return destination


def _inventory_tar(archive_path: Path) -> ArchiveInventory:
    members_by_name: dict[str, list[tarfile.TarInfo]] = {}
    unsafe: list[str] = []
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if not member.isreg():
                continue
            if _is_unsafe_member_path(member.name):
                unsafe.append(member.name)
                continue
            members_by_name.setdefault(member.name, []).append(member)
    return _finalize_inventory(members_by_name, unsafe)


def _inventory_zip(archive_path: Path) -> ArchiveInventory:
    members_by_name: dict[str, list[zipfile.ZipInfo]] = {}
    unsafe: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if _is_unsafe_member_path(info.filename):
                unsafe.append(info.filename)
                continue
            members_by_name.setdefault(info.filename, []).append(info)
    return _finalize_inventory(
        members_by_name,
        unsafe,
        size_getter=lambda item: item.file_size,
    )


def _finalize_inventory(
    members_by_name: Mapping[str, Sequence[object]],
    unsafe: Sequence[str],
    *,
    size_getter: Callable[[object], int] | None = None,
) -> ArchiveInventory:
    duplicate_refs = tuple(
        sorted(name for name, members in members_by_name.items() if len(members) > 1)
    )
    records = {
        name: ArchiveMemberRecord(
            member_name=name,
            member_size=size_getter(members[0]) if size_getter else 0,
        )
        for name, members in members_by_name.items()
        if len(members) == 1
    }
    return ArchiveInventory(
        members_by_ref=records,
        duplicate_refs=duplicate_refs,
        rejected_unsafe_members=tuple(unsafe),
    )


def _write_temporary_copy(
    source: BinaryIO,
    directory: Path,
    destination_name: str,
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=directory,
        prefix=f".{destination_name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        while chunk := source.read(1024 * 1024):
            temporary.write(chunk)
        temporary.flush()
        os.fsync(temporary.fileno())
    return temporary_path


def _publish_temporary(temporary_path: Path, destination: Path) -> None:
    digest = sha256_file(temporary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary_path, destination)
    except FileExistsError:
        if not destination.is_file() or sha256_file(destination) != digest:
            raise DestinationConflictError(
                f"Destination exists with different content: {destination}"
            )
    else:
        temporary_path.unlink(missing_ok=True)
        return
    temporary_path.unlink(missing_ok=True)


def _required_archive_keys(selected: Sequence[SelectedCandidate]) -> set[ArchiveSpecKey]:
    return {
        (item.candidate.corpus, item.candidate.native_split)
        for item in selected
        if item.candidate.materialization_mode == "extract_archive"
    }


def _canonical_materialization_order(
    selected: Sequence[SelectedCandidate],
) -> tuple[SelectedCandidate, ...]:
    role_index = {"train": 0, "calibration": 1, "test": 2}
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                role_index[item.role],
                item.candidate.label,
                item.selection_rank,
                item.candidate.candidate_id,
            ),
        )
    )


def _validate_materialization_mode(mode: str) -> None:
    if mode not in {"reuse_processed", "process_source", "extract_archive"}:
        raise ValueError(f"Unsupported materialization_mode: {mode}")


def _coerce_sha256(value: str, *, field: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value}")
    return value


def _is_unsafe_member_path(name: str) -> bool:
    if "\\" in name:
        return True
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    return (
        posix_path.is_absolute()
        or ".." in posix_path.parts
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    )
