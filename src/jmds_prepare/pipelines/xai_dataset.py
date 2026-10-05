"""Orchestration, validation and publication for canonical XAI datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from jmds_prepare import __version__
from jmds_prepare.core.hashing import sha256_file_preserving_times
from jmds_prepare.core.xai_sample import SUPPORTED_ROLES, XAI_SAMPLE_COLUMNS, XaiSample
from jmds_prepare.pipelines.xai_audit import (
    XaiSampleAudit,
    audit_processed_sample,
    summarize_sample_audits,
    validate_processed_wav_format,
)
from jmds_prepare.pipelines.xai_materialization import (
    ArchiveSpec,
    ArchiveSpecKey,
    materialize_selected,
)
from jmds_prepare.pipelines.xai_selection import (
    SelectedCandidate,
    SelectionCandidate,
    select_language_candidates,
)
from jmds_prepare.profiles.xai import SHARED_SPLIT_DISJOINT_FIELDS, XaiLanguageProfile
from jmds_prepare.storage.xai_layout import XaiLayout

PROVENANCE_SCHEMA_VERSION = 1
CANONICAL_SAMPLE_RATE = 16_000


class DatasetValidationError(ValueError):
    """Raised when a materialized XAI sample set fails pre-publication checks."""


class SelectedValidationError(DatasetValidationError):
    """Raised when selected candidates fail pre-materialization checks."""


def require_canonical_sample_rate(sample_rate: int) -> None:
    """Reject non-canonical processed sample rates before selection/materialization."""
    if sample_rate != CANONICAL_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate must be exactly {CANONICAL_SAMPLE_RATE}, got {sample_rate}"
        )


@dataclass(frozen=True)
class XaiDatasetDeps:
    select_candidates: Callable[..., tuple[SelectedCandidate, ...]]
    materialize_selected: Callable[..., tuple[XaiSample, ...]]
    audit_sample: Callable[..., XaiSampleAudit]
    summarize_audits: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    sha256_file: Callable[[Path], str]
    software_version: str = __version__


@dataclass(frozen=True)
class XaiDatasetResult:
    samples: tuple[XaiSample, ...]
    selected: tuple[SelectedCandidate, ...]
    audits: tuple[XaiSampleAudit, ...]


def build_xai_dataset(
    candidates: Sequence[SelectionCandidate],
    *,
    profile: XaiLanguageProfile,
    seed: int,
    layout: XaiLayout,
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    sample_rate: int,
    input_paths: Mapping[str, Path],
    deps: XaiDatasetDeps,
    unrar_executable: str = "UnRAR",
    archive_runner: Callable[..., Any] | None = None,
    source_provenance_path: Path | None = None,
) -> XaiDatasetResult:
    """Select, materialize, audit, validate and publish one language dataset."""
    require_canonical_sample_rate(sample_rate)
    selected = deps.select_candidates(candidates, profile, seed)
    validate_selected_before_materialization(selected, profile=profile)
    materialize_kwargs: dict[str, Any] = {
        "sample_rate": sample_rate,
        "unrar_executable": unrar_executable,
    }
    if archive_runner is not None:
        materialize_kwargs["archive_runner"] = archive_runner
    samples = deps.materialize_selected(
        selected,
        layout,
        archive_specs,
        **materialize_kwargs,
    )
    audits = tuple(
        deps.audit_sample(Path(sample.processed_path), sample_id=sample.sample_id)
        for sample in samples
    )
    validate_xai_sample_set(
        samples,
        profile=profile,
        layout=layout,
        preserve_existing_paths=profile.language_code == "eng",
    )
    selection_report = build_selection_report(
        candidates=candidates,
        selected=selected,
        profile=profile,
        seed=seed,
    )
    audio_audit = deps.summarize_audits(samples, audits)
    provenance = build_xai_provenance(
        profile=profile,
        seed=seed,
        sample_rate=sample_rate,
        input_paths=input_paths,
        archive_specs=archive_specs,
        layout=layout,
        sha256_file=deps.sha256_file,
        software_version=deps.software_version,
        source_provenance_path=source_provenance_path,
    )
    artifacts = {
        layout.samples_csv(profile.language_code): serialize_xai_samples_csv(samples),
        layout.selection_report_json(profile.language_code): serialize_json(
            selection_report
        ),
        layout.audio_audit_json(profile.language_code): serialize_json(audio_audit),
        layout.provenance_json(profile.language_code): serialize_json(provenance),
    }
    deps.publish_artifacts(artifacts)
    return XaiDatasetResult(samples=samples, selected=selected, audits=audits)


def serialize_xai_samples_csv(samples: Sequence[XaiSample]) -> bytes:
    """Serialize samples to canonical UTF-8 CSV with LF line endings."""
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(XAI_SAMPLE_COLUMNS)
    for sample in samples:
        row = sample.to_dict()
        ordered = [row[column] for column in XAI_SAMPLE_COLUMNS]
        XaiSample.from_dict(
            dict(zip(XAI_SAMPLE_COLUMNS, ordered, strict=True)),
            enforce_column_order=True,
        )
        writer.writerow(ordered)
    return buffer.getvalue().encode("utf-8")


def serialize_json(payload: Mapping[str, Any]) -> bytes:
    """Serialize one JSON artifact deterministically."""
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_xai_sample_set(
    samples: Sequence[XaiSample],
    *,
    profile: XaiLanguageProfile,
    layout: XaiLayout,
    preserve_existing_paths: bool = False,
) -> None:
    """Validate quotas, uniqueness, files, hashes, format and disjoint fields."""
    expected_total = profile.rows_per_language()
    if len(samples) != expected_total:
        raise DatasetValidationError(
            f"Expected {expected_total} samples, got {len(samples)}"
        )

    counts: Counter[tuple[str, int]] = Counter(
        (sample.role, sample.label) for sample in samples
    )
    _validate_role_label_quotas(counts, profile, error_type=DatasetValidationError)

    seen_sample_ids: set[str] = set()
    seen_original_refs: set[str] = set()
    seen_processed_paths: set[str] = set()
    values_by_role: dict[tuple[str, str, str], dict[str, set[str]]] = {}

    for sample in samples:
        if sample.language != profile.language_code:
            raise DatasetValidationError(
                f"Sample language mismatch: {sample.language}"
            )
        if sample.label != profile.label_for_corpus(sample.corpus):
            raise DatasetValidationError(
                f"Label/corpus mismatch for {sample.sample_id}"
            )
        allowed_roles = profile.corpus_native_split_roles[sample.corpus][
            sample.native_split
        ]
        if sample.role not in allowed_roles:
            raise DatasetValidationError(
                f"Role {sample.role!r} is not allowed for "
                f"{sample.corpus}/{sample.native_split}"
            )

        for field, seen in (
            ("sample_id", seen_sample_ids),
            ("original_ref", seen_original_refs),
            ("processed_path", seen_processed_paths),
        ):
            value = getattr(sample, field)
            if value in seen:
                raise DatasetValidationError(f"Duplicate {field}: {value}")
            seen.add(value)

        processed = Path(sample.processed_path)
        if not processed.is_file():
            raise DatasetValidationError(f"Missing processed file: {processed}")
        if preserve_existing_paths:
            pass
        elif not processed.resolve().is_relative_to(layout.data_dir.resolve()):
            raise DatasetValidationError(
                f"Processed path escapes layout cache: {processed}"
            )

        original = Path(sample.original_ref)
        if original.is_file():
            actual_source = sha256_file_preserving_times(original)
            if actual_source != sample.sha256_source:
                raise DatasetValidationError(
                    f"sha256_source mismatch for {sample.sample_id}"
                )
        actual_processed = sha256_file_preserving_times(processed)
        if actual_processed != sample.sha256_processed:
            raise DatasetValidationError(
                f"sha256_processed mismatch for {sample.sample_id}"
            )
        try:
            validate_processed_wav_format(processed)
        except ValueError as exc:
            raise DatasetValidationError(str(exc)) from exc

        disjoint_fields = profile.shared_split_disjoint_fields.get(
            sample.corpus, {}
        ).get(sample.native_split, ())
        for field_name in disjoint_fields:
            value = _field_value(sample, field_name)
            if value is None:
                continue
            key = (sample.corpus, sample.native_split, field_name)
            values_by_role.setdefault(key, {}).setdefault(sample.role, set()).add(
                value
            )

    for (corpus, native_split, field_name), role_values in values_by_role.items():
        roles = sorted(role_values)
        for left_index, left_role in enumerate(roles):
            for right_role in roles[left_index + 1 :]:
                overlap = role_values[left_role] & role_values[right_role]
                if overlap:
                    joined = ", ".join(sorted(overlap))
                    raise DatasetValidationError(
                        f"overlap in {corpus}/{native_split}/{field_name} between "
                        f"{left_role} and {right_role}: {joined}"
                    )


def validate_selected_before_materialization(
    selected: Sequence[SelectedCandidate],
    *,
    profile: XaiLanguageProfile,
) -> None:
    """Validate quotas and structural invariants before any cache write."""
    expected_total = profile.rows_per_language()
    if len(selected) != expected_total:
        raise SelectedValidationError(
            f"Expected {expected_total} selected candidates, got {len(selected)}"
        )

    counts: Counter[tuple[str, int]] = Counter(
        (item.role, item.candidate.label) for item in selected
    )
    _validate_role_label_quotas(counts, profile, error_type=SelectedValidationError)

    seen_candidate_ids: set[str] = set()
    seen_original_refs: set[str] = set()
    for item in selected:
        candidate = item.candidate
        if candidate.language != profile.language_code:
            raise SelectedValidationError(
                f"Candidate language mismatch: {candidate.language}"
            )
        if candidate.label != profile.label_for_corpus(candidate.corpus):
            raise SelectedValidationError(
                f"Label/corpus mismatch for {candidate.candidate_id}"
            )
        allowed_roles = profile.corpus_native_split_roles[candidate.corpus][
            candidate.native_split
        ]
        if item.role not in allowed_roles:
            raise SelectedValidationError(
                f"Role {item.role!r} is not allowed for "
                f"{candidate.corpus}/{candidate.native_split}"
            )
        if candidate.candidate_id in seen_candidate_ids:
            raise SelectedValidationError(
                f"Duplicate candidate_id: {candidate.candidate_id}"
            )
        seen_candidate_ids.add(candidate.candidate_id)
        if candidate.original_ref in seen_original_refs:
            raise SelectedValidationError(
                f"Duplicate original_ref: {candidate.original_ref}"
            )
        seen_original_refs.add(candidate.original_ref)

    for verification in _disjoint_verifications(profile, selected):
        if not verification["verified"]:
            raise SelectedValidationError(
                "disjoint overlap detected: "
                f"{verification['corpus']}/{verification['native_split']}"
            )

    if profile.paired_selection:
        _validate_paired_speakers(selected, profile)


def build_selection_report(
    *,
    candidates: Sequence[SelectionCandidate],
    selected: Sequence[SelectedCandidate],
    profile: XaiLanguageProfile,
    seed: int,
) -> dict[str, Any]:
    """Summarize deterministic selection without claiming utterance pairing."""
    candidate_counts = _count_candidates(candidates, profile)
    selected_counts = _count_selected(selected)
    return {
        "seed": seed,
        "language": profile.language_code,
        "targets": {
            role: profile.per_class_targets[role] for role in SUPPORTED_ROLES
        },
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "reason_counts": dict(Counter(item.selection_reason for item in selected)),
        "source_counts": dict(
            Counter(item.selection_source for item in selected)
        ),
        "disjoint_verifications": _disjoint_verifications(profile, selected),
        "paired_selection": profile.paired_selection,
        "paired_utterances": False,
    }


def build_xai_provenance(
    *,
    profile: XaiLanguageProfile,
    seed: int,
    sample_rate: int,
    input_paths: Mapping[str, Path],
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    layout: XaiLayout,
    sha256_file: Callable[[Path], str],
    software_version: str = __version__,
    source_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance record for one language dataset."""
    inputs = _input_evidence(input_paths, archive_specs, sha256_file)
    if source_provenance_path is not None:
        inputs.append(
            _file_evidence("source_provenance", source_provenance_path, sha256_file)
        )
        inputs.sort(key=lambda entry: entry["key"])
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "language": profile.language_code,
        "inputs": inputs,
        "config": {
            "profile": profile.language_slug,
            "seed": seed,
            "sample_rate": sample_rate,
            "per_class_targets": {
                role: profile.per_class_targets[role] for role in SUPPORTED_ROLES
            },
            "paired_selection": profile.paired_selection,
            "shared_split_disjoint_fields": _serialize_disjoint_fields(profile),
        },
        "software": {
            "name": "jmds-prepare",
            "version": software_version,
        },
        "outputs": [
            str(layout.samples_csv(profile.language_code)),
            str(layout.selection_report_json(profile.language_code)),
            str(layout.audio_audit_json(profile.language_code)),
            str(layout.provenance_json(profile.language_code)),
        ],
        "limitations": _limitations_for_language(profile.language_code),
    }
    if profile.language_code == "zho":
        payload["aishell3"] = {
            "pristine_candidate_source": "reconciled_aishell3_metadata_manifest",
            "reconciliation_details": "upstream_mandarin_metadata_provenance",
        }
    return payload


def _limitations_for_language(language: str) -> dict[str, Any]:
    if language == "eng":
        return {
            "paired_by_speaker": True,
            "paired_utterances": False,
        }
    limitations: dict[str, Any] = {
        "paired_samples": False,
        "class_corpus_confound": True,
        "external_validation_under_corpus_shift": True,
    }
    if language == "por":
        limitations["coraa"] = {
            "local_only_no_redistribution": True,
            "license": "CC-BY-NC-ND-4.0",
        }
    return limitations


def _validate_role_label_quotas(
    counts: Counter[tuple[str, int]],
    profile: XaiLanguageProfile,
    *,
    error_type: type[Exception],
) -> None:
    for role in SUPPORTED_ROLES:
        target = profile.per_class_targets[role]
        for label in (0, 1):
            actual = counts[(role, label)]
            if actual != target:
                raise error_type(
                    f"quota mismatch for role={role!r} label={label}: "
                    f"expected {target}, got {actual}"
                )


def _validate_paired_speakers(
    selected: Sequence[SelectedCandidate],
    profile: XaiLanguageProfile,
) -> None:
    by_role: dict[str, dict[int, set[str]]] = {}
    for item in selected:
        speaker = item.candidate.speaker_id
        if speaker is None:
            raise SelectedValidationError(
                f"Paired selection requires speaker_id for {item.candidate.candidate_id}"
            )
        by_role.setdefault(item.role, {0: set(), 1: set()})[item.candidate.label].add(
            speaker
        )
    for role in SUPPORTED_ROLES:
        role_speakers = by_role.get(role, {0: set(), 1: set()})
        if role_speakers[0] != role_speakers[1]:
            raise SelectedValidationError(
                f"paired speaker mismatch for role={role!r}"
            )


def _input_evidence(
    input_paths: Mapping[str, Path],
    archive_specs: Mapping[ArchiveSpecKey, ArchiveSpec],
    sha256_file: Callable[[Path], str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key, path in sorted(input_paths.items()):
        evidence.append(_file_evidence(key, path, sha256_file))
    seen_paths: set[Path] = {path.resolve() for path in input_paths.values()}
    for (corpus, native_split), spec in sorted(archive_specs.items()):
        for index, volume in enumerate(spec.volume_paths):
            resolved = volume.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            evidence.append(
                _file_evidence(
                    f"archive:{corpus}/{native_split}:{index}",
                    volume,
                    sha256_file,
                )
            )
    return evidence


def _file_evidence(
    key: str,
    path: Path,
    sha256_file: Callable[[Path], str],
) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "key": key,
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _serialize_disjoint_fields(profile: XaiLanguageProfile) -> dict[str, Any]:
    serialized: dict[str, dict[str, list[str]]] = {}
    for corpus, split_fields in profile.shared_split_disjoint_fields.items():
        serialized[corpus] = {
            native_split: list(fields)
            for native_split, fields in split_fields.items()
        }
    return serialized


def _count_candidates(
    candidates: Sequence[SelectionCandidate],
    profile: XaiLanguageProfile,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for candidate in candidates:
        if candidate.language != profile.language_code:
            continue
        corpus_entry = grouped.setdefault(candidate.corpus, {})
        split_entry = corpus_entry.setdefault(candidate.native_split, {})
        for role in profile.corpus_native_split_roles[candidate.corpus][
            candidate.native_split
        ]:
            role_entry = split_entry.setdefault(role, {"0": 0, "1": 0})
            role_entry[str(candidate.label)] += 1
    return grouped


def _count_selected(selected: Sequence[SelectedCandidate]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for item in selected:
        candidate = item.candidate
        corpus_entry = grouped.setdefault(candidate.corpus, {})
        split_entry = corpus_entry.setdefault(candidate.native_split, {})
        role_entry = split_entry.setdefault(item.role, {"0": 0, "1": 0})
        role_entry[str(candidate.label)] += 1
    return grouped


def _disjoint_verifications(
    profile: XaiLanguageProfile,
    selected: Sequence[SelectedCandidate],
) -> list[dict[str, Any]]:
    verifications: list[dict[str, Any]] = []
    for corpus, split_fields in profile.shared_split_disjoint_fields.items():
        for native_split, fields in split_fields.items():
            role_values: dict[str, set[str]] = {}
            for item in selected:
                candidate = item.candidate
                if candidate.corpus != corpus or candidate.native_split != native_split:
                    continue
                for field_name in fields:
                    value = _field_value_from_candidate(candidate, field_name)
                    if value is None:
                        continue
                    role_values.setdefault(item.role, set()).add(value)
            overlap: list[dict[str, Any]] = []
            roles = sorted(role_values)
            for left_index, left_role in enumerate(roles):
                for right_role in roles[left_index + 1 :]:
                    shared = sorted(role_values[left_role] & role_values[right_role])
                    if shared:
                        overlap.append(
                            {
                                "roles": [left_role, right_role],
                                "values": shared,
                            }
                        )
            verifications.append(
                {
                    "corpus": corpus,
                    "native_split": native_split,
                    "fields": list(fields),
                    "verified": not overlap,
                    "overlap": overlap,
                }
            )
    return verifications


def _field_value(sample: XaiSample, field_name: str) -> str | None:
    if field_name not in SHARED_SPLIT_DISJOINT_FIELDS:
        raise ValueError(f"Unsupported disjoint field: {field_name}")
    if field_name == "speaker_id":
        return sample.speaker_id
    if field_name == "group_id":
        return sample.group_id
    raise ValueError(f"Unsupported disjoint field: {field_name}")


def _field_value_from_candidate(
    candidate: SelectionCandidate, field_name: str
) -> str | None:
    if field_name == "speaker_id":
        return candidate.speaker_id
    if field_name == "group_id":
        return candidate.group_id
    raise ValueError(f"Unsupported disjoint field: {field_name}")


def default_xai_dataset_deps() -> XaiDatasetDeps:
    from jmds_prepare.core.hashing import sha256_file
    from jmds_prepare.core.publication import publish_artifact_set_idempotent

    return XaiDatasetDeps(
        select_candidates=select_language_candidates,
        materialize_selected=materialize_selected,
        audit_sample=audit_processed_sample,
        summarize_audits=summarize_sample_audits,
        publish_artifacts=publish_artifact_set_idempotent,
        sha256_file=sha256_file,
    )
