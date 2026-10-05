"""Tests for deterministic XAI candidate selection and manifest adapters."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from jmds_prepare.audio import AUDIO_METADATA_COLUMNS
from jmds_prepare.core.xai_sample import SUPPORTED_ROLES
from jmds_prepare.manifest import MANIFEST_COLUMNS
from jmds_prepare.pipelines import xai_selection as xai_selection_module
from jmds_prepare.pipelines.xai_selection import (
    DuplicateCandidateError,
    InsufficientCapacityError,
    SelectionCandidate,
    SelectedCandidate,
    SelectionError,
    adapt_add_metadata,
    adapt_aishell3_metadata,
    adapt_coraa_metadata,
    adapt_english_processed,
    adapt_mlaad_metadata,
    select_language_candidates,
)
from jmds_prepare.profiles.xai import XaiLanguageProfile


def _tiny_eng_profile(*, paired: bool = True) -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="eng",
        language_slug="english",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"ASVspoof2024": 0, "JMDS": 1},
        corpus_native_split_roles={
            "ASVspoof2024": {"train": ("train",), "dev": ("calibration", "test")},
            "JMDS": {"train": ("train",), "dev": ("calibration", "test")},
        },
        shared_split_disjoint_fields={
            "ASVspoof2024": {"dev": ("speaker_id", "group_id")},
            "JMDS": {"dev": ("speaker_id", "group_id")},
        },
        paired_selection=paired,
    )


def _tiny_por_profile() -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="por",
        language_slug="portuguese",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"CORAA": 0, "MLAAD": 1},
        corpus_native_split_roles={
            "CORAA": {
                "train": ("train",),
                "dev": ("calibration",),
                "test": ("test",),
            },
            "MLAAD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={},
        paired_selection=False,
    )


def _tiny_zho_profile() -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="zho",
        language_slug="mandarin",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"AISHELL-3": 0, "ADD": 1},
        corpus_native_split_roles={
            "AISHELL-3": {"train": ("train", "calibration"), "test": ("test",)},
            "ADD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={"AISHELL-3": {"train": ("speaker_id",)}},
        paired_selection=False,
    )


_ENGLISH_PROCESSED_COLUMNS = MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
_AUDIO_METADATA_DEFAULTS = {
    "source_sample_rate": "48000",
    "source_channels": "1",
    "source_frames": "1000",
    "source_duration_seconds": "0.02",
    "source_peak": "0.5",
    "source_rms": "0.1",
    "processed_sample_rate": "16000",
    "processed_channels": "1",
    "processed_frames": "320",
    "processed_duration_seconds": "0.02",
    "processed_peak": "0.5",
    "processed_rms": "0.1",
}


def _english_processed_row(**overrides: Any) -> dict[str, str]:
    row = {
        "utt_id": "T_0000000001",
        "spk_id": "T_0001",
        "gender": "F",
        "language": "eng",
        "dataset": "ASVspoof2024",
        "split": "train",
        "label": "pristine",
        "attack_id": "pristine",
        "protocol_codec": "",
        "source_group_id": "",
        "source_path": "/raw/train/T_0000000001.flac",
        "processed_path": "/data/processed/train/T_0000000001.wav",
        "sha256_source": "a" * 64,
        "sha256_processed": "b" * 64,
        "metadata_source": "JMDS+ASVspoof5-verified",
        **_AUDIO_METADATA_DEFAULTS,
    }
    row.update(overrides)
    return row


def _candidate(
    *,
    candidate_id: str,
    language: str,
    label: int,
    corpus: str,
    native_split: str,
    original_ref: str,
    speaker_id: str | None = None,
    group_id: str | None = None,
    attack_id: str | None = None,
    local_source_path: str | None = None,
    metadata_source: str = "fixture",
    materialization_mode: str = "process_source",
    expected_sha256_source: str | None = None,
    expected_sha256_processed: str | None = None,
) -> SelectionCandidate:
    if materialization_mode == "reuse_processed" and expected_sha256_processed is None:
        expected_sha256_processed = "b" * 64
    if materialization_mode == "extract_archive":
        local_source_path = None
    elif local_source_path is None:
        local_source_path = original_ref
    return SelectionCandidate(
        candidate_id=candidate_id,
        language=language,
        label=label,
        corpus=corpus,
        native_split=native_split,
        original_ref=original_ref,
        local_source_path=local_source_path,
        speaker_id=speaker_id,
        group_id=group_id,
        attack_id=attack_id,
        metadata_source=metadata_source,
        materialization_mode=materialization_mode,
        expected_sha256_source=expected_sha256_source,
        expected_sha256_processed=expected_sha256_processed,
    )


def _eng_train_pool() -> list[SelectionCandidate]:
    return [
        _candidate(
            candidate_id=f"asv-{idx:02d}",
            language="eng",
            label=0,
            corpus="ASVspoof2024",
            native_split="train",
            original_ref=f"raw/train/{idx}.flac",
            speaker_id=f"S{idx // 2:02d}",
            local_source_path=f"/data/processed/train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"jmds-{idx:02d}",
            language="eng",
            label=1,
            corpus="JMDS",
            native_split="train",
            original_ref=f"gen/train/{idx}.wav",
            speaker_id=f"S{idx // 2:02d}",
            attack_id="A01",
            local_source_path=f"/data/processed/gen/train/{idx}.wav",
        )
        for idx in range(4)
    ]


def _eng_dev_pool(*, cal_speakers: tuple[str, ...], test_speakers: tuple[str, ...]) -> list[SelectionCandidate]:
    rows: list[SelectionCandidate] = []
    for speaker in cal_speakers:
        rows.append(
            _candidate(
                candidate_id=f"asv-cal-{speaker}",
                language="eng",
                label=0,
                corpus="ASVspoof2024",
                native_split="dev",
                original_ref=f"raw/dev/{speaker}.flac",
                speaker_id=speaker,
                local_source_path=f"/data/processed/dev/{speaker}.wav",
            )
        )
        rows.append(
            _candidate(
                candidate_id=f"jmds-cal-{speaker}",
                language="eng",
                label=1,
                corpus="JMDS",
                native_split="dev",
                original_ref=f"gen/dev/{speaker}.wav",
                speaker_id=speaker,
                attack_id="A01",
                local_source_path=f"/data/processed/gen/dev/{speaker}.wav",
            )
        )
    for speaker in test_speakers:
        rows.append(
            _candidate(
                candidate_id=f"asv-test-{speaker}",
                language="eng",
                label=0,
                corpus="ASVspoof2024",
                native_split="dev",
                original_ref=f"raw/dev/test/{speaker}.flac",
                speaker_id=speaker,
                local_source_path=f"/data/processed/dev/test/{speaker}.wav",
            )
        )
        rows.append(
            _candidate(
                candidate_id=f"jmds-test-{speaker}",
                language="eng",
                label=1,
                corpus="JMDS",
                native_split="dev",
                original_ref=f"gen/dev/test/{speaker}.wav",
                speaker_id=speaker,
                attack_id="A01",
                local_source_path=f"/data/processed/gen/dev/test/{speaker}.wav",
            )
        )
    return rows


def _selected_ids(selected: tuple[SelectedCandidate, ...]) -> frozenset[str]:
    return frozenset(item.candidate.candidate_id for item in selected)


def _count_by_role_label(
    selected: tuple[SelectedCandidate, ...],
) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for item in selected:
        key = (item.role, item.candidate.label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_selection_is_invariant_to_input_order():
    profile = _tiny_eng_profile()
    pool = _eng_train_pool() + _eng_dev_pool(
        cal_speakers=("C1", "C2"), test_speakers=("T1", "T2")
    )
    forward = select_language_candidates(pool, profile, seed=7)
    reverse = select_language_candidates(list(reversed(pool)), profile, seed=7)
    assert _selected_ids(forward) == _selected_ids(reverse)


def test_same_seed_repeats_and_different_seed_changes_selection():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    first = select_language_candidates(pool, profile, seed=11)
    again = select_language_candidates(pool, profile, seed=11)
    other = select_language_candidates(pool, profile, seed=12)
    assert _selected_ids(first) == _selected_ids(again)
    assert _selected_ids(first) != _selected_ids(other)


def test_exact_quotas_for_tiny_profiles():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    selected = select_language_candidates(pool, profile, seed=3)
    assert _count_by_role_label(selected) == {
        ("train", 0): 2,
        ("train", 1): 2,
        ("calibration", 0): 1,
        ("calibration", 1): 1,
        ("test", 0): 1,
        ("test", 1): 1,
    }


def test_fails_before_output_when_capacity_is_insufficient():
    profile = _tiny_eng_profile()
    pool = _eng_train_pool()[:3]
    with pytest.raises(InsufficientCapacityError):
        select_language_candidates(pool, profile, seed=1)


def test_capacity_error_reports_missing_disjoint_field_exclusions():
    profile = XaiLanguageProfile(
        language_code="zho",
        language_slug="mandarin",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"AISHELL-3": 0, "ADD": 1},
        corpus_native_split_roles={
            "AISHELL-3": {"train": ("train", "calibration"), "test": ("test",)},
            "ADD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={"AISHELL-3": {"train": ("speaker_id",)}},
        paired_selection=False,
    )
    pool = [
        _candidate(
            candidate_id=f"ai-no-spk-{idx}",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="train",
            original_ref=f"train/wav/none/utt{idx}.wav",
            speaker_id=None,
        )
        for idx in range(5)
    ] + [
        _candidate(
            candidate_id="add-train-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="train",
            original_ref="/jmds/train/1.wav",
        ),
    ]
    with pytest.raises(
        InsufficientCapacityError,
        match="excluded_missing_disjoint_field=5",
    ):
        select_language_candidates(pool, profile, seed=1)


def test_selection_result_enforces_unique_candidate_ids_and_refs():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    selected = select_language_candidates(pool, profile, seed=0)
    candidate_ids = [item.candidate.candidate_id for item in selected]
    original_refs = [item.candidate.original_ref for item in selected]
    assert len(candidate_ids) == len(set(candidate_ids))
    assert len(original_refs) == len(set(original_refs))

    duplicate = SelectedCandidate(
        candidate=selected[0].candidate,
        role="train",
        selection_seed=0,
        selection_rank=99,
        selection_reason="test",
        selection_source="fixture",
    )
    with pytest.raises(SelectionError, match="Duplicate candidate_id"):
        xai_selection_module._assert_unique_selection((selected[0], duplicate))


def test_rejects_duplicate_candidate_id():
    profile = _tiny_por_profile()
    duplicate = _candidate(
        candidate_id="dup",
        language="por",
        label=0,
        corpus="CORAA",
        native_split="train",
        original_ref="a.wav",
    )
    pool = [duplicate, duplicate]
    with pytest.raises(DuplicateCandidateError, match="candidate_id"):
        select_language_candidates(pool, profile, seed=1)


def test_rejects_duplicate_original_ref():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id="a",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref="same.wav",
        ),
        _candidate(
            candidate_id="b",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref="same.wav",
        ),
    ]
    with pytest.raises(DuplicateCandidateError, match="original_ref"):
        select_language_candidates(pool, profile, seed=1)


def test_english_paired_selection_balances_classes_per_speaker():
    profile = _tiny_eng_profile()
    pool = _eng_train_pool() + _eng_dev_pool(
        cal_speakers=("C1", "C2"), test_speakers=("T1", "T2")
    )
    selected = select_language_candidates(pool, profile, seed=5)
    for role in ("train", "calibration", "test"):
        role_rows = [item for item in selected if item.role == role]
        by_speaker: dict[str, dict[int, int]] = {}
        for item in role_rows:
            speaker = item.candidate.speaker_id
            assert speaker is not None
            counts = by_speaker.setdefault(speaker, {0: 0, 1: 0})
            counts[item.candidate.label] += 1
        for speaker, counts in by_speaker.items():
            assert counts[0] == counts[1], f"speaker {speaker} unbalanced in {role}"


def test_english_calibration_and_test_speakers_are_disjoint():
    profile = _tiny_eng_profile()
    pool = _eng_train_pool() + _eng_dev_pool(
        cal_speakers=("C1", "C2"), test_speakers=("T1", "T2")
    )
    selected = select_language_candidates(pool, profile, seed=5)
    cal_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "calibration" and item.candidate.speaker_id
    }
    test_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "test" and item.candidate.speaker_id
    }
    assert cal_speakers.isdisjoint(test_speakers)


def test_english_disjoint_blocking_uses_group_id_when_present():
    profile = _tiny_eng_profile()
    pool = [
        _candidate(
            candidate_id="asv-g1",
            language="eng",
            label=0,
            corpus="ASVspoof2024",
            native_split="dev",
            original_ref="raw/dev/g1.flac",
            speaker_id="SX",
            group_id="G1",
            local_source_path="/data/dev/g1.wav",
        ),
        _candidate(
            candidate_id="jmds-g1",
            language="eng",
            label=1,
            corpus="JMDS",
            native_split="dev",
            original_ref="gen/dev/g1.wav",
            speaker_id="SX",
            group_id="G1",
            attack_id="A01",
            local_source_path="/data/gen/dev/g1.wav",
        ),
        _candidate(
            candidate_id="asv-g2",
            language="eng",
            label=0,
            corpus="ASVspoof2024",
            native_split="dev",
            original_ref="raw/dev/g2.flac",
            speaker_id="SY",
            group_id="G2",
            local_source_path="/data/dev/g2.wav",
        ),
        _candidate(
            candidate_id="jmds-g2",
            language="eng",
            label=1,
            corpus="JMDS",
            native_split="dev",
            original_ref="gen/dev/g2.wav",
            speaker_id="SY",
            group_id="G2",
            attack_id="A01",
            local_source_path="/data/gen/dev/g2.wav",
        ),
        *_eng_train_pool(),
    ]
    selected = select_language_candidates(pool, profile, seed=9)
    cal_groups = {
        item.candidate.group_id
        for item in selected
        if item.role == "calibration" and item.candidate.group_id
    }
    test_groups = {
        item.candidate.group_id
        for item in selected
        if item.role == "test" and item.candidate.group_id
    }
    assert cal_groups.isdisjoint(test_groups)


def test_shared_split_allocates_smaller_role_quota_before_larger():
    profile = XaiLanguageProfile(
        language_code="zho",
        language_slug="mandarin",
        per_class_targets={"train": 4, "calibration": 2, "test": 1},
        corpus_labels={"AISHELL-3": 0, "ADD": 1},
        corpus_native_split_roles={
            "AISHELL-3": {"train": ("train", "calibration"), "test": ("test",)},
            "ADD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={"AISHELL-3": {"train": ("speaker_id",)}},
        paired_selection=False,
    )
    aishell_train = [
        _candidate(
            candidate_id=f"ai-{speaker}-{idx}",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="train",
            original_ref=f"train/wav/{speaker}/utt{idx}.wav",
            speaker_id=speaker,
        )
        for speaker in ("spkA", "spkB", "spkC")
        for idx in range(5)
    ]
    add_pool = [
        _candidate(
            candidate_id=f"add-train-{idx}",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="train",
            original_ref=f"/jmds/train/{idx}.wav",
        )
        for idx in range(8)
    ] + [
        _candidate(
            candidate_id=f"add-cal-{idx}",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="dev",
            original_ref=f"/jmds/dev/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="add-test-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="eval",
            original_ref="/jmds/eval/1.wav",
        ),
        _candidate(
            candidate_id="ai-test-1",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="test",
            original_ref="test/wav/spkZ/utt.wav",
            speaker_id="spkZ",
        ),
    ]
    selected = select_language_candidates(aishell_train + add_pool, profile, seed=1)
    cal_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "calibration" and item.candidate.corpus == "AISHELL-3"
    }
    train_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "train" and item.candidate.corpus == "AISHELL-3"
    }
    assert cal_speakers.isdisjoint(train_speakers)
    assert len([item for item in selected if item.role == "calibration"]) == 4


def test_selection_output_is_canonical_role_label_rank_order():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    selected = select_language_candidates(pool, profile, seed=0)
    role_indices = [SUPPORTED_ROLES.index(item.role) for item in selected]
    assert role_indices == sorted(role_indices)
    for role in SUPPORTED_ROLES:
        for label in (0, 1):
            ranks = [
                item.selection_rank
                for item in selected
                if item.role == role and item.candidate.label == label
            ]
            assert ranks == list(range(len(ranks)))


def test_aishell_train_and_calibration_speakers_are_disjoint():
    profile = _tiny_zho_profile()
    pool = [
        _candidate(
            candidate_id=f"ai-train-{idx}",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="train",
            original_ref=f"train/wav/spk{idx}/utt.wav",
            speaker_id=f"spk{idx}",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"add-train-{idx}",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="train",
            original_ref=f"/jmds/train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="add-dev-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="dev",
            original_ref="/jmds/dev/1.wav",
        ),
        _candidate(
            candidate_id="add-eval-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="eval",
            original_ref="/jmds/eval/1.wav",
        ),
        _candidate(
            candidate_id="ai-test-1",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="test",
            original_ref="test/wav/spkZ/utt.wav",
            speaker_id="spkZ",
        ),
    ]
    selected = select_language_candidates(pool, profile, seed=4)
    train_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "train" and item.candidate.speaker_id
    }
    cal_speakers = {
        item.candidate.speaker_id
        for item in selected
        if item.role == "calibration" and item.candidate.speaker_id
    }
    assert train_speakers.isdisjoint(cal_speakers)


def test_portuguese_and_mandarin_do_not_assert_utterance_pairing():
    por_profile = _tiny_por_profile()
    por_pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    por_selected = select_language_candidates(por_pool, por_profile, seed=2)
    por_train = [item for item in por_selected if item.role == "train"]
    assert len({item.candidate.candidate_id for item in por_train}) == 4

    zho_profile = _tiny_zho_profile()
    zho_pool = [
        _candidate(
            candidate_id=f"ai-train-{idx}",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="train",
            original_ref=f"train/wav/spk{idx}/utt.wav",
            speaker_id=f"spk{idx}",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"add-train-{idx}",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="train",
            original_ref=f"/jmds/train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="add-dev-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="dev",
            original_ref="/jmds/dev/1.wav",
        ),
        _candidate(
            candidate_id="add-eval-1",
            language="zho",
            label=1,
            corpus="ADD",
            native_split="eval",
            original_ref="/jmds/eval/1.wav",
        ),
        _candidate(
            candidate_id="ai-test-1",
            language="zho",
            label=0,
            corpus="AISHELL-3",
            native_split="test",
            original_ref="test/wav/spkZ/utt.wav",
            speaker_id="spkZ",
        ),
    ]
    zho_selected = select_language_candidates(zho_pool, zho_profile, seed=2)
    assert all(item.selection_reason for item in zho_selected)
    por_reasons = {item.selection_reason for item in por_selected}
    zho_reasons = {item.selection_reason for item in zho_selected}
    assert por_reasons == {"unpaired_class_quota"}
    assert zho_reasons == {"unpaired_class_quota"}


def test_selection_rank_is_zero_based_within_role_and_label():
    profile = _tiny_por_profile()
    pool = [
        _candidate(
            candidate_id=f"c-{idx}",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="train",
            original_ref=f"train/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id=f"m-{idx}",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="train",
            original_ref=f"audio/{idx}.wav",
        )
        for idx in range(4)
    ] + [
        _candidate(
            candidate_id="c-dev-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="dev",
            original_ref="dev/1.wav",
        ),
        _candidate(
            candidate_id="m-dev-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="dev",
            original_ref="dev/audio/1.wav",
        ),
        _candidate(
            candidate_id="c-test-1",
            language="por",
            label=0,
            corpus="CORAA",
            native_split="test",
            original_ref="test/1.wav",
        ),
        _candidate(
            candidate_id="m-test-1",
            language="por",
            label=1,
            corpus="MLAAD",
            native_split="eval",
            original_ref="eval/audio/1.wav",
        ),
    ]
    selected = select_language_candidates(pool, profile, seed=0)
    for role in profile.per_class_targets:
        for label in (0, 1):
            ranks = sorted(
                item.selection_rank
                for item in selected
                if item.role == role and item.candidate.label == label
            )
            assert ranks == list(range(len(ranks)))


def test_adapt_english_processed_maps_labels_corpora_and_paths():
    frame = pd.DataFrame(
        [
            _english_processed_row(),
            _english_processed_row(
                utt_id="T_0000000002",
                spk_id="T_0002",
                gender="M",
                dataset="JMDS",
                label="generated",
                attack_id="A01",
                source_group_id="G99",
                source_path="/gen/train/T_0000000002.wav",
                processed_path="/data/processed/generated/train/T_0000000002.wav",
                sha256_source="c" * 64,
                sha256_processed="d" * 64,
            ),
        ],
        columns=_ENGLISH_PROCESSED_COLUMNS,
    )
    candidates = adapt_english_processed(frame, metadata_source="english_processed.csv")
    assert len(candidates) == 2
    pristine = candidates[0]
    generated = candidates[1]
    assert pristine.label == 0
    assert pristine.corpus == "ASVspoof2024"
    assert pristine.candidate_id == "T_0000000001"
    assert pristine.original_ref == "/raw/train/T_0000000001.flac"
    assert pristine.local_source_path == "/data/processed/train/T_0000000001.wav"
    assert pristine.materialization_mode == "reuse_processed"
    assert pristine.expected_sha256_source == "a" * 64
    assert pristine.expected_sha256_processed == "b" * 64
    assert pristine.speaker_id == "T_0001"
    assert pristine.group_id is None
    assert generated.label == 1
    assert generated.materialization_mode == "reuse_processed"
    assert generated.corpus == "JMDS"
    assert generated.group_id == "G99"
    assert generated.attack_id == "A01"


def test_adapt_english_processed_requires_full_processed_schema():
    raw_only = pd.DataFrame(
        [_english_processed_row(processed_path="", sha256_processed="")],
        columns=MANIFEST_COLUMNS,
    )
    with pytest.raises(ValueError, match="processed schema|audio metadata|27"):
        adapt_english_processed(raw_only, metadata_source="english_processed.csv")

    incomplete = pd.DataFrame([_english_processed_row()], columns=MANIFEST_COLUMNS)
    with pytest.raises(ValueError, match="processed schema|audio metadata|27"):
        adapt_english_processed(incomplete, metadata_source="english_processed.csv")


def test_adapt_english_processed_rejects_rows_missing_processed_hashes():
    frame = pd.DataFrame(
        [
            _english_processed_row(),
            _english_processed_row(
                utt_id="T_0000000003",
                split="dev",
                source_path="/raw/dev/T_0000000003.flac",
                processed_path="",
                sha256_processed="",
            ),
        ],
        columns=_ENGLISH_PROCESSED_COLUMNS,
    )
    with pytest.raises(ValueError, match="processed_path|sha256_processed"):
        adapt_english_processed(frame, metadata_source="english_processed.csv")


def test_adapt_coraa_metadata_schema_and_mapping():
    frame = pd.DataFrame(
        [
            {
                "file_path": "train/wav/spk/file.wav",
                "task": "annotation",
                "variety": "standard",
                "dataset": "CORAA",
                "accent": "neutro",
                "speech_genre": "conversational",
                "speech_style": "impersonation",
                "up_votes": "1",
                "down_votes": "0",
                "votes_for_hesitation": "0",
                "votes_for_filled_pause": "0",
                "votes_for_noise_or_low_voice": "0",
                "votes_for_second_voice": "0",
                "votes_for_no_identified_problem": "0",
                "text": "Olá",
                "split": "train",
                "metadata_source_file": "coraa.csv",
                "metadata_source_row": "2",
            }
        ]
    )
    candidates = adapt_coraa_metadata(frame, metadata_source="coraa.csv")
    assert len(candidates) == 1
    item = candidates[0]
    assert item.label == 0
    assert item.corpus == "CORAA"
    assert item.candidate_id == "train/wav/spk/file.wav"
    assert item.original_ref == "train/wav/spk/file.wav"
    assert item.native_split == "train"
    assert item.speaker_id is None


def test_adapt_mlaad_metadata_schema_and_mapping():
    frame = pd.DataFrame(
        [
            {
                "spk_id": "T_0001",
                "utt_id": "T_0000000001",
                "gender": "F",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "por",
                "dataset": "MLAAD",
                "split": "train",
                "metadata_source_file": "jmds.csv",
                "metadata_source_row": "2",
                "audio_path": "/jmds/train/T_0000000001.wav",
            }
        ]
    )
    candidates = adapt_mlaad_metadata(frame, metadata_source="jmds.csv")
    item = candidates[0]
    assert item.label == 1
    assert item.corpus == "MLAAD"
    assert item.candidate_id == "T_0000000001"
    assert item.original_ref == "/jmds/train/T_0000000001.wav"
    assert item.local_source_path == "/jmds/train/T_0000000001.wav"
    assert item.speaker_id == "T_0001"
    assert item.attack_id == "A01"


def test_adapt_aishell3_metadata_schema_and_mapping():
    frame = pd.DataFrame(
        [
            {
                "utt_id": "utt-1",
                "split": "train",
                "archive_member_path": "train/wav/spk1/utt-1.wav",
                "speaker_id": "spk1",
                "gender": "F",
                "age": "20",
                "accent": "standard",
                "transcription": "你好",
                "pinyin": "ni hao",
                "prosody_available": "yes",
                "metadata_source_file": "archive.tgz",
                "metadata_source_row": "10",
            }
        ]
    )
    candidates = adapt_aishell3_metadata(frame, metadata_source="archive.tgz")
    item = candidates[0]
    assert item.label == 0
    assert item.corpus == "AISHELL-3"
    assert item.candidate_id == "utt-1"
    assert item.original_ref == "train/wav/spk1/utt-1.wav"
    assert item.speaker_id == "spk1"


def test_adapt_add_metadata_schema_and_mapping():
    frame = pd.DataFrame(
        [
            {
                "spk_id": "unk",
                "utt_id": "T_0000123456",
                "gender": "unk",
                "codec": "-",
                "attack_id": "unk",
                "label": "generated",
                "native": "yes",
                "language": "zho",
                "dataset": "ADD",
                "split": "train",
                "metadata_source_file": "jmds.csv",
                "metadata_source_row": "2",
                "audio_path": "/jmds/train/T_0000123456.wav",
            }
        ]
    )
    candidates = adapt_add_metadata(frame, metadata_source="jmds.csv")
    item = candidates[0]
    assert item.label == 1
    assert item.corpus == "ADD"
    assert item.candidate_id == "T_0000123456"
    assert item.original_ref == "/jmds/train/T_0000123456.wav"
    assert item.speaker_id is None


@pytest.mark.parametrize(
    ("adapter", "frame"),
    [
        (
            adapt_english_processed,
            pd.DataFrame({"utt_id": ["x"]}),
        ),
        (
            adapt_coraa_metadata,
            pd.DataFrame({"file_path": ["x"]}),
        ),
    ],
)
def test_adapters_reject_incomplete_schema(adapter, frame: pd.DataFrame):
    with pytest.raises(ValueError, match="missing|required|columns"):
        adapter(frame, metadata_source="fixture.csv")


def test_selected_candidate_carries_explicit_reason_and_source():
    candidate = _candidate(
        candidate_id="c-0",
        language="por",
        label=0,
        corpus="CORAA",
        native_split="train",
        original_ref="train/0.wav",
    )
    selected = SelectedCandidate(
        candidate=candidate,
        role="train",
        selection_seed=1,
        selection_rank=0,
        selection_reason="unpaired_class_quota",
        selection_source="coraa.csv",
    )
    assert selected.selection_reason == "unpaired_class_quota"
    assert selected.selection_source == "coraa.csv"
