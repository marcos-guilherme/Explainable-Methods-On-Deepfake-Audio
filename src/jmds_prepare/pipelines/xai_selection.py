"""Pure, corpus-agnostic deterministic selection between native manifests and materialization."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from jmds_prepare.audio import AUDIO_METADATA_COLUMNS
from jmds_prepare.core.xai_sample import SUPPORTED_LANGUAGES, SUPPORTED_ROLES
from jmds_prepare.manifest import MANIFEST_COLUMNS
from jmds_prepare.profiles.xai import XaiLanguageProfile
from jmds_prepare.sources.aishell3 import AISHELL3_COLUMNS
from jmds_prepare.sources.coraa import CORAA_COLUMNS
from jmds_prepare.sources.jmds import JMDS_COLUMNS

_SUPPORTED_LABELS = frozenset({0, 1})
_SPARSE_OPTIONAL = frozenset({"unk", "-", ""})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROLE_INDEX = {role: index for index, role in enumerate(SUPPORTED_ROLES)}

_ENGLISH_PROCESSED_COLUMNS = MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
_CORAA_ADAPTER_COLUMNS = (*CORAA_COLUMNS, "split", "metadata_source_file", "metadata_source_row")
_MLAAD_ADAPTER_COLUMNS = (
    *JMDS_COLUMNS,
    "split",
    "metadata_source_file",
    "metadata_source_row",
    "audio_path",
)
_ADD_ADAPTER_COLUMNS = _MLAAD_ADAPTER_COLUMNS


class SelectionError(ValueError):
    """Base error for deterministic XAI selection."""


class InsufficientCapacityError(SelectionError):
    """Raised when a role/class quota cannot be satisfied."""


class DuplicateCandidateError(SelectionError):
    """Raised when candidate identifiers or references are duplicated."""


_MATERIALIZATION_MODES = frozenset(
    {"reuse_processed", "process_source", "extract_archive"}
)


@dataclass(frozen=True)
class SelectionCandidate:
    """Immutable pre-materialization candidate from a native manifest row."""

    candidate_id: str
    language: str
    label: int
    corpus: str
    native_split: str
    original_ref: str
    local_source_path: str | None
    speaker_id: str | None
    group_id: str | None
    attack_id: str | None
    metadata_source: str
    materialization_mode: str
    expected_sha256_source: str | None = None
    expected_sha256_processed: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _coerce_label(self.label))
        object.__setattr__(
            self, "speaker_id", _normalize_optional_text(self.speaker_id)
        )
        object.__setattr__(self, "group_id", _normalize_optional_text(self.group_id))
        object.__setattr__(self, "attack_id", _normalize_optional_text(self.attack_id))
        object.__setattr__(
            self,
            "local_source_path",
            _normalize_optional_text(self.local_source_path),
        )
        object.__setattr__(
            self,
            "expected_sha256_source",
            _normalize_optional_sha256(self.expected_sha256_source),
        )
        object.__setattr__(
            self,
            "expected_sha256_processed",
            _normalize_optional_sha256(self.expected_sha256_processed),
        )
        for field in (
            "candidate_id",
            "language",
            "corpus",
            "native_split",
            "original_ref",
            "metadata_source",
            "materialization_mode",
        ):
            _require_non_empty_text(getattr(self, field), field=field)
        if self.materialization_mode not in _MATERIALIZATION_MODES:
            raise ValueError(
                f"Unsupported materialization_mode: {self.materialization_mode}"
            )
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")
        if self.label == 0 and self.attack_id is not None:
            raise ValueError("attack_id must be empty for label 0")
        if self.materialization_mode == "extract_archive" and self.local_source_path:
            raise ValueError(
                "extract_archive candidates must not populate local_source_path"
            )
        if self.materialization_mode in {
            "reuse_processed",
            "process_source",
        } and not self.local_source_path:
            raise ValueError(
                f"{self.materialization_mode} candidates must populate "
                "local_source_path"
            )
        if (
            self.materialization_mode == "reuse_processed"
            and self.expected_sha256_processed is None
        ):
            raise ValueError(
                "reuse_processed candidates must populate expected_sha256_processed"
            )


@dataclass(frozen=True)
class SelectedCandidate:
    """One deterministically selected candidate bound to an experimental role."""

    candidate: SelectionCandidate
    role: str
    selection_seed: int
    selection_rank: int
    selection_reason: str
    selection_source: str

    def __post_init__(self) -> None:
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(f"Unsupported role: {self.role}")
        object.__setattr__(
            self,
            "selection_seed",
            _coerce_non_negative_int(self.selection_seed, field="selection_seed"),
        )
        object.__setattr__(
            self,
            "selection_rank",
            _coerce_non_negative_int(self.selection_rank, field="selection_rank"),
        )
        for field in ("selection_reason", "selection_source"):
            _require_non_empty_text(getattr(self, field), field=field)


def adapt_english_processed(
    frame: pd.DataFrame,
    *,
    metadata_source: str,
) -> tuple[SelectionCandidate, ...]:
    actual_columns = list(frame.columns)
    if actual_columns == MANIFEST_COLUMNS:
        raise ValueError(
            "english_processed adapter requires the full processed schema "
            f"({len(_ENGLISH_PROCESSED_COLUMNS)} columns including audio metadata), "
            "not the raw manifest schema"
        )
    _validate_columns(frame, _ENGLISH_PROCESSED_COLUMNS, source="english_processed")
    _validate_unique_values(frame, "utt_id", source="english_processed")
    _validate_unique_values(frame, "source_path", source="english_processed")
    candidates: list[SelectionCandidate] = []
    for row in frame.to_dict("records"):
        utt_id = str(row["utt_id"])
        processed_path = str(row["processed_path"]).strip()
        processed_hash = str(row["sha256_processed"]).strip()
        _require_processed_english_row(
            utt_id=utt_id,
            processed_path=processed_path,
            sha256_processed=processed_hash,
        )
        label_name = row["label"]
        if label_name == "pristine":
            label = 0
            corpus = "ASVspoof2024"
        elif label_name == "generated":
            label = 1
            corpus = "JMDS"
        else:
            raise ValueError(f"Unsupported english_processed label: {label_name}")
        group_id = str(row["source_group_id"]).strip() or None
        attack_id = str(row["attack_id"]).strip()
        source_hash = str(row["sha256_source"]).strip()
        candidates.append(
            SelectionCandidate(
                candidate_id=utt_id,
                language=str(row["language"]),
                label=label,
                corpus=corpus,
                native_split=str(row["split"]),
                original_ref=str(row["source_path"]),
                local_source_path=processed_path,
                speaker_id=str(row["spk_id"]).strip() or None,
                group_id=group_id,
                attack_id=None if label == 0 else (attack_id or None),
                metadata_source=metadata_source,
                materialization_mode="reuse_processed",
                expected_sha256_source=source_hash or None,
                expected_sha256_processed=processed_hash,
            )
        )
    return tuple(candidates)


def adapt_coraa_metadata(
    frame: pd.DataFrame,
    *,
    metadata_source: str,
) -> tuple[SelectionCandidate, ...]:
    _validate_columns(frame, _CORAA_ADAPTER_COLUMNS, source="CORAA")
    _validate_unique_values(frame, "file_path", source="CORAA")
    candidates: list[SelectionCandidate] = []
    for row in frame.to_dict("records"):
        file_path = str(row["file_path"])
        candidates.append(
            SelectionCandidate(
                candidate_id=file_path,
                language="por",
                label=0,
                corpus="CORAA",
                native_split=str(row["split"]),
                original_ref=file_path,
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source=metadata_source,
                materialization_mode="extract_archive",
            )
        )
    return tuple(candidates)


def adapt_mlaad_metadata(
    frame: pd.DataFrame,
    *,
    metadata_source: str,
) -> tuple[SelectionCandidate, ...]:
    _validate_columns(frame, _MLAAD_ADAPTER_COLUMNS, source="MLAAD")
    _validate_unique_values(frame, "utt_id", source="MLAAD")
    _validate_unique_values(frame, "audio_path", source="MLAAD")
    candidates: list[SelectionCandidate] = []
    for row in frame.to_dict("records"):
        audio_path = str(row["audio_path"])
        speaker_id = _optional_identifier(row["spk_id"])
        attack_id = _optional_identifier(row["attack_id"])
        candidates.append(
            SelectionCandidate(
                candidate_id=str(row["utt_id"]),
                language="por",
                label=1,
                corpus="MLAAD",
                native_split=str(row["split"]),
                original_ref=audio_path,
                local_source_path=audio_path,
                speaker_id=speaker_id,
                group_id=None,
                attack_id=attack_id,
                metadata_source=metadata_source,
                materialization_mode="process_source",
            )
        )
    return tuple(candidates)


def adapt_aishell3_metadata(
    frame: pd.DataFrame,
    *,
    metadata_source: str,
) -> tuple[SelectionCandidate, ...]:
    _validate_columns(frame, AISHELL3_COLUMNS, source="AISHELL-3")
    _validate_unique_values(frame, "utt_id", source="AISHELL-3")
    _validate_unique_values(frame, "archive_member_path", source="AISHELL-3")
    candidates: list[SelectionCandidate] = []
    for row in frame.to_dict("records"):
        archive_member_path = str(row["archive_member_path"])
        candidates.append(
            SelectionCandidate(
                candidate_id=str(row["utt_id"]),
                language="zho",
                label=0,
                corpus="AISHELL-3",
                native_split=str(row["split"]),
                original_ref=archive_member_path,
                local_source_path=None,
                speaker_id=str(row["speaker_id"]).strip() or None,
                group_id=None,
                attack_id=None,
                metadata_source=metadata_source,
                materialization_mode="extract_archive",
            )
        )
    return tuple(candidates)


def adapt_add_metadata(
    frame: pd.DataFrame,
    *,
    metadata_source: str,
) -> tuple[SelectionCandidate, ...]:
    _validate_columns(frame, _ADD_ADAPTER_COLUMNS, source="ADD")
    _validate_unique_values(frame, "utt_id", source="ADD")
    _validate_unique_values(frame, "audio_path", source="ADD")
    candidates: list[SelectionCandidate] = []
    for row in frame.to_dict("records"):
        audio_path = str(row["audio_path"])
        candidates.append(
            SelectionCandidate(
                candidate_id=str(row["utt_id"]),
                language="zho",
                label=1,
                corpus="ADD",
                native_split=str(row["split"]),
                original_ref=audio_path,
                local_source_path=audio_path,
                speaker_id=_optional_identifier(row["spk_id"]),
                group_id=None,
                attack_id=_optional_identifier(row["attack_id"]),
                metadata_source=metadata_source,
                materialization_mode="process_source",
            )
        )
    return tuple(candidates)


def select_language_candidates(
    candidates: Sequence[SelectionCandidate],
    profile: XaiLanguageProfile,
    seed: int,
) -> tuple[SelectedCandidate, ...]:
    """Select deterministic role/class quotas from native manifest candidates."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Invalid seed: {seed}")

    _validate_candidate_pool(candidates, profile)
    blocked_values = _BlockedSplitValues()
    selected: list[SelectedCandidate] = []

    for role in _role_allocation_order(profile):
        target = profile.per_class_targets[role]
        if profile.paired_selection:
            role_selected = _select_paired_role(
                candidates=candidates,
                profile=profile,
                role=role,
                target=target,
                seed=seed,
                blocked_values=blocked_values,
            )
        else:
            role_selected = _select_unpaired_role(
                candidates=candidates,
                profile=profile,
                role=role,
                target=target,
                seed=seed,
                blocked_values=blocked_values,
            )
        selected.extend(role_selected)

    ordered = _canonical_selection_order(selected)
    _assert_unique_selection(ordered)
    return ordered


def _assert_unique_selection(selected: Sequence[SelectedCandidate]) -> None:
    seen_ids: set[str] = set()
    seen_refs: set[str] = set()
    for item in selected:
        candidate_id = item.candidate.candidate_id
        original_ref = item.candidate.original_ref
        if candidate_id in seen_ids:
            raise SelectionError(
                f"Duplicate candidate_id in selection result: {candidate_id}"
            )
        if original_ref in seen_refs:
            raise SelectionError(
                f"Duplicate original_ref in selection result: {original_ref}"
            )
        seen_ids.add(candidate_id)
        seen_refs.add(original_ref)


def _role_allocation_order(profile: XaiLanguageProfile) -> tuple[str, ...]:
    """Allocate constrained shared splits from smaller role quotas to larger ones."""
    must_precede: dict[str, set[str]] = {role: set() for role in SUPPORTED_ROLES}

    for corpus, split_fields in profile.shared_split_disjoint_fields.items():
        for native_split, fields in split_fields.items():
            if not fields:
                continue
            roles = profile.corpus_native_split_roles[corpus][native_split]
            if len(roles) < 2:
                continue
            ordered = sorted(
                roles,
                key=lambda role: (
                    profile.per_class_targets[role],
                    _ROLE_INDEX[role],
                ),
            )
            for earlier, later in zip(ordered, ordered[1:]):
                must_precede[later].add(earlier)

    remaining = set(SUPPORTED_ROLES)
    order: list[str] = []
    while remaining:
        ready = sorted(
            (
                role
                for role in remaining
                if must_precede[role].issubset(set(order))
            ),
            key=_ROLE_INDEX.__getitem__,
        )
        if not ready:
            raise SelectionError(
                "Unable to derive role allocation order from shared split constraints"
            )
        chosen = ready[0]
        order.append(chosen)
        remaining.remove(chosen)
    return tuple(order)


def _canonical_selection_order(
    selected: Sequence[SelectedCandidate],
) -> tuple[SelectedCandidate, ...]:
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                _ROLE_INDEX[item.role],
                item.candidate.label,
                item.selection_rank,
                item.candidate.candidate_id,
            ),
        )
    )


def _select_unpaired_role(
    *,
    candidates: Sequence[SelectionCandidate],
    profile: XaiLanguageProfile,
    role: str,
    target: int,
    seed: int,
    blocked_values: _BlockedSplitValues,
) -> list[SelectedCandidate]:
    selected: list[SelectedCandidate] = []
    for label in (0, 1):
        pool, eligibility = _eligible_candidates(
            candidates,
            profile=profile,
            role=role,
            label=label,
            blocked_values=blocked_values,
        )
        conserve_speakers = _pool_requires_speaker_conservation(
            pool, profile=profile, role=role
        )
        ordered = _select_label_picks(
            pool,
            target=target,
            seed=seed,
            conserve_speakers=conserve_speakers,
        )
        if len(ordered) < target:
            raise InsufficientCapacityError(
                f"Insufficient candidates for role={role!r} label={label}: "
                f"need {target}, have {len(ordered)}, "
                f"excluded_missing_disjoint_field="
                f"{eligibility.excluded_missing_disjoint_field}"
            )
        picks = ordered[:target]
        for rank, candidate in enumerate(picks):
            selected.append(
                SelectedCandidate(
                    candidate=candidate,
                    role=role,
                    selection_seed=seed,
                    selection_rank=rank,
                    selection_reason="unpaired_class_quota",
                    selection_source=candidate.metadata_source,
                )
            )
        _register_blocked_values(
            blocked_values,
            profile=profile,
            role=role,
            picks=picks,
        )
    return selected


def _select_paired_role(
    *,
    candidates: Sequence[SelectionCandidate],
    profile: XaiLanguageProfile,
    role: str,
    target: int,
    seed: int,
    blocked_values: _BlockedSplitValues,
) -> list[SelectedCandidate]:
    eligibility_by_label: dict[int, _RoleLabelEligibility] = {}
    pools: dict[int, list[SelectionCandidate]] = {}
    for label in (0, 1):
        pool, eligibility = _eligible_candidates(
            candidates,
            profile=profile,
            role=role,
            label=label,
            blocked_values=blocked_values,
        )
        pools[label] = pool
        eligibility_by_label[label] = eligibility
    speakers = _paired_speakers(pools[0], pools[1])
    if not speakers:
        excluded = sum(
            stats.excluded_missing_disjoint_field
            for stats in eligibility_by_label.values()
        )
        raise InsufficientCapacityError(
            f"No paired speakers available for role={role!r}, "
            f"excluded_missing_disjoint_field={excluded}"
        )

    speaker_order = sorted(
        speakers,
        key=lambda speaker: _selection_digest(seed, f"speaker:{speaker}"),
    )
    by_speaker: dict[str, dict[int, list[SelectionCandidate]]] = {}
    for label in (0, 1):
        for candidate in pools[label]:
            speaker = candidate.speaker_id
            if speaker is None or speaker not in speakers:
                continue
            by_speaker.setdefault(speaker, {0: [], 1: []})[label].append(candidate)

    selected_by_label: dict[int, list[SelectionCandidate]] = {0: [], 1: []}
    remaining = target
    for speaker in speaker_order:
        if remaining <= 0:
            break
        class_pools = by_speaker[speaker]
        available = min(len(class_pools[0]), len(class_pools[1]), remaining)
        if available <= 0:
            continue
        for label in (0, 1):
            ordered = _sorted_candidates(class_pools[label], seed=seed)
            selected_by_label[label].extend(ordered[:available])
        remaining -= available

    if remaining > 0:
        excluded = sum(
            stats.excluded_missing_disjoint_field
            for stats in eligibility_by_label.values()
        )
        raise InsufficientCapacityError(
            f"Insufficient paired capacity for role={role!r}: "
            f"need {target}, selected {target - remaining}, "
            f"excluded_missing_disjoint_field={excluded}"
        )

    selected: list[SelectedCandidate] = []
    for label in (0, 1):
        picks = selected_by_label[label]
        if len(picks) != target:
            excluded = eligibility_by_label[label].excluded_missing_disjoint_field
            raise InsufficientCapacityError(
                f"Paired selection failed to balance role={role!r} label={label}, "
                f"excluded_missing_disjoint_field={excluded}"
            )
        for rank, candidate in enumerate(
            _sorted_candidates(picks, seed=seed)
        ):
            selected.append(
                SelectedCandidate(
                    candidate=candidate,
                    role=role,
                    selection_seed=seed,
                    selection_rank=rank,
                    selection_reason="paired_speaker_quota",
                    selection_source=candidate.metadata_source,
                )
            )
        _register_blocked_values(
            blocked_values,
            profile=profile,
            role=role,
            picks=picks,
        )
    return selected


def _pool_requires_speaker_conservation(
    pool: Sequence[SelectionCandidate],
    *,
    profile: XaiLanguageProfile,
    role: str,
) -> bool:
    seen: set[tuple[str, str]] = set()
    for candidate in pool:
        key = (candidate.corpus, candidate.native_split)
        if key in seen:
            continue
        seen.add(key)
        if _should_conserve_speakers(
            profile,
            role=role,
            corpus=candidate.corpus,
            native_split=candidate.native_split,
        ):
            return True
    return False


def _should_conserve_speakers(
    profile: XaiLanguageProfile,
    *,
    role: str,
    corpus: str,
    native_split: str,
) -> bool:
    fields = profile.shared_split_disjoint_fields.get(corpus, {}).get(native_split, ())
    if not fields or "speaker_id" not in fields:
        return False
    sibling_roles = profile.corpus_native_split_roles[corpus][native_split]
    if len(sibling_roles) < 2:
        return False
    role_quota = profile.per_class_targets[role]
    return any(
        profile.per_class_targets[sibling] > role_quota
        for sibling in sibling_roles
        if sibling != role
    )


def _select_label_picks(
    pool: Sequence[SelectionCandidate],
    *,
    target: int,
    seed: int,
    conserve_speakers: bool,
) -> list[SelectionCandidate]:
    if conserve_speakers:
        return _select_speaker_conserving(pool, target=target, seed=seed)
    return _sorted_candidates(pool, seed=seed)[:target]


def _select_speaker_conserving(
    pool: Sequence[SelectionCandidate],
    *,
    target: int,
    seed: int,
) -> list[SelectionCandidate]:
    """Pack picks into fewer speakers to preserve disjoint pools for larger sibling roles."""
    with_speaker = [candidate for candidate in pool if candidate.speaker_id]
    without_speaker = [candidate for candidate in pool if candidate.speaker_id is None]
    by_speaker: dict[str, list[SelectionCandidate]] = {}
    for candidate in with_speaker:
        assert candidate.speaker_id is not None
        by_speaker.setdefault(candidate.speaker_id, []).append(candidate)

    speaker_order = sorted(
        by_speaker,
        key=lambda speaker: (
            -len(by_speaker[speaker]),
            _selection_digest(seed, f"speaker:{speaker}"),
            speaker,
        ),
    )
    picks: list[SelectionCandidate] = []
    for speaker in speaker_order:
        picks.extend(_sorted_candidates(by_speaker[speaker], seed=seed))
        if len(picks) >= target:
            return picks[:target]
    picks.extend(_sorted_candidates(without_speaker, seed=seed))
    return picks[:target]


@dataclass(frozen=True)
class _RoleLabelEligibility:
    excluded_missing_disjoint_field: int = 0


def _eligible_candidates(
    candidates: Sequence[SelectionCandidate],
    *,
    profile: XaiLanguageProfile,
    role: str,
    label: int,
    blocked_values: _BlockedSplitValues,
) -> tuple[list[SelectionCandidate], _RoleLabelEligibility]:
    eligible: list[SelectionCandidate] = []
    excluded_missing_disjoint_field = 0
    for candidate in candidates:
        if candidate.language != profile.language_code:
            raise SelectionError(
                f"Candidate language mismatch: {candidate.language} != "
                f"{profile.language_code}"
            )
        if candidate.label != label:
            continue
        expected_label = profile.label_for_corpus(candidate.corpus)
        if expected_label != label:
            raise SelectionError(
                f"Candidate label mismatch for corpus {candidate.corpus!r}: "
                f"{label} != {expected_label}"
            )
        roles = profile.corpus_native_split_roles[candidate.corpus][
            candidate.native_split
        ]
        if role not in roles:
            continue
        disjoint_fields = profile.shared_split_disjoint_fields.get(
            candidate.corpus, {}
        ).get(candidate.native_split, ())
        if disjoint_fields and not _has_usable_disjoint_field(
            candidate, disjoint_fields
        ):
            excluded_missing_disjoint_field += 1
            continue
        if blocked_values.conflicts(candidate, disjoint_fields, role=role):
            continue
        eligible.append(candidate)
    return eligible, _RoleLabelEligibility(
        excluded_missing_disjoint_field=excluded_missing_disjoint_field
    )


def _paired_speakers(
    label_zero: Sequence[SelectionCandidate],
    label_one: Sequence[SelectionCandidate],
) -> set[str]:
    zero_speakers = {
        candidate.speaker_id
        for candidate in label_zero
        if candidate.speaker_id is not None
    }
    one_speakers = {
        candidate.speaker_id
        for candidate in label_one
        if candidate.speaker_id is not None
    }
    return zero_speakers & one_speakers


def _register_blocked_values(
    blocked_values: _BlockedSplitValues,
    *,
    profile: XaiLanguageProfile,
    role: str,
    picks: Sequence[SelectionCandidate],
) -> None:
    for candidate in picks:
        disjoint_fields = profile.shared_split_disjoint_fields.get(
            candidate.corpus, {}
        ).get(candidate.native_split, ())
        if not disjoint_fields:
            continue
        sibling_roles = [
            mapped_role
            for mapped_role in profile.corpus_native_split_roles[candidate.corpus][
                candidate.native_split
            ]
            if mapped_role != role
        ]
        if not sibling_roles:
            continue
        blocked_values.register(candidate, disjoint_fields, role=role)


@dataclass
class _BlockedSplitValues:
    by_split: dict[tuple[str, str, str], set[tuple[str, str]]] | None = None

    def __post_init__(self) -> None:
        if self.by_split is None:
            self.by_split = {}

    def conflicts(
        self,
        candidate: SelectionCandidate,
        fields: Sequence[str],
        *,
        role: str,
    ) -> bool:
        assert self.by_split is not None
        for field_name, value in _non_empty_disjoint_values(candidate, fields):
            key = (candidate.corpus, candidate.native_split, field_name)
            used = self.by_split.get(key, set())
            if any(
                stored_role != role and stored_value == value
                for stored_role, stored_value in used
            ):
                return True
        return False

    def register(
        self,
        candidate: SelectionCandidate,
        fields: Sequence[str],
        *,
        role: str,
    ) -> None:
        assert self.by_split is not None
        for field_name, value in _non_empty_disjoint_values(candidate, fields):
            key = (candidate.corpus, candidate.native_split, field_name)
            self.by_split.setdefault(key, set()).add((role, value))


def _non_empty_disjoint_values(
    candidate: SelectionCandidate,
    fields: Sequence[str],
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for field_name in fields:
        value = _field_value(candidate, field_name)
        if value is not None:
            values.append((field_name, value))
    return values


def _has_usable_disjoint_field(
    candidate: SelectionCandidate,
    fields: Sequence[str],
) -> bool:
    return bool(_non_empty_disjoint_values(candidate, fields))


def _field_value(candidate: SelectionCandidate, field_name: str) -> str | None:
    if field_name == "speaker_id":
        return candidate.speaker_id
    if field_name == "group_id":
        return candidate.group_id
    raise ValueError(f"Unsupported disjoint field: {field_name}")


def _validate_candidate_pool(
    candidates: Sequence[SelectionCandidate],
    profile: XaiLanguageProfile,
) -> None:
    seen_ids: set[str] = set()
    seen_refs: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen_ids:
            raise DuplicateCandidateError(
                f"Duplicate candidate_id: {candidate.candidate_id}"
            )
        seen_ids.add(candidate.candidate_id)
        if candidate.original_ref in seen_refs:
            raise DuplicateCandidateError(
                f"Duplicate original_ref: {candidate.original_ref}"
            )
        seen_refs.add(candidate.original_ref)
        if candidate.language != profile.language_code:
            raise SelectionError(
                f"Candidate language mismatch: {candidate.language} != "
                f"{profile.language_code}"
            )
        if candidate.corpus not in profile.corpus_labels:
            raise SelectionError(f"Unknown corpus for profile: {candidate.corpus}")
        if profile.label_for_corpus(candidate.corpus) != candidate.label:
            raise SelectionError(
                f"Candidate label mismatch for corpus {candidate.corpus!r}"
            )


def _sorted_candidates(
    candidates: Sequence[SelectionCandidate],
    *,
    seed: int,
) -> list[SelectionCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            _selection_digest(seed, candidate.candidate_id),
            candidate.candidate_id,
        ),
    )


def _selection_digest(seed: int, token: str) -> bytes:
    payload = f"{seed}:{token}".encode()
    return hashlib.sha256(payload).digest()


def _validate_columns(
    frame: pd.DataFrame,
    expected: Sequence[str],
    *,
    source: str,
) -> None:
    actual = list(frame.columns)
    if actual != list(expected):
        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if missing:
            raise ValueError(
                f"{source} adapter missing required columns: {', '.join(missing)}"
            )
        if extra:
            raise ValueError(
                f"{source} adapter has unexpected columns: {', '.join(extra)}"
            )
        raise ValueError(f"{source} adapter columns are not in the required order")


def _validate_unique_values(frame: pd.DataFrame, column: str, *, source: str) -> None:
    duplicates = frame.loc[frame[column].duplicated(keep=False), column].unique()
    if len(duplicates):
        joined = ", ".join(str(value) for value in sorted(duplicates))
        raise ValueError(f"Duplicate {source} {column} value(s): {joined}")


def _optional_identifier(value: Any) -> str | None:
    text = str(value).strip()
    if not text or text in _SPARSE_OPTIONAL:
        return None
    return text


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_label(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Unsupported label: {value}")
    if value not in _SUPPORTED_LABELS:
        raise ValueError(f"Unsupported label: {value}")
    return value


def _coerce_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def _require_non_empty_text(value: Any, *, field: str) -> None:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} must be non-empty")


def _normalize_optional_sha256(value: Any) -> str | None:
    text = _normalize_optional_text(value)
    if text is None:
        return None
    if text != text.lower() or not _SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"Invalid sha256 hash: {text}")
    return text


def _require_processed_english_row(
    *,
    utt_id: str,
    processed_path: str,
    sha256_processed: str,
) -> None:
    if not processed_path:
        raise ValueError(
            f"english_processed row {utt_id} must populate processed_path"
        )
    if not sha256_processed:
        raise ValueError(
            f"english_processed row {utt_id} must populate sha256_processed"
        )
    if sha256_processed != sha256_processed.lower():
        raise ValueError(
            f"english_processed row {utt_id} has invalid sha256_processed: "
            f"{sha256_processed}"
        )
    if not _SHA256_PATTERN.fullmatch(sha256_processed):
        raise ValueError(
            f"english_processed row {utt_id} has invalid sha256_processed: "
            f"{sha256_processed}"
        )
