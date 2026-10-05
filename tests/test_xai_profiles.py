"""Tests for immutable XAI selection profiles by language."""

from __future__ import annotations

import pytest

from jmds_prepare.core.xai_sample import SUPPORTED_LANGUAGES, SUPPORTED_ROLES
from jmds_prepare.profiles.english import ENGLISH_PROFILE
from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE
from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE
_MINIMAL_POR_CORPUS_ROLES = {
    "CORAA": {"train": ("train",)},
    "MLAAD": {"train": ("train",)},
}
_MINIMAL_ENG_CORPUS_ROLES = {
    "ASVspoof2024": {"dev": ("calibration", "test")},
    "JMDS": {"dev": ("calibration", "test")},
}

from jmds_prepare.profiles.xai import (
    ENGLISH_XAI_PROFILE,
    MANDARIN_XAI_PROFILE,
    PORTUGUESE_XAI_PROFILE,
    SHARED_SPLIT_DISJOINT_FIELDS,
    XAI_PER_CLASS_TARGETS,
    XAI_PROFILE_BY_LANGUAGE,
    XAI_ROWS_PER_LANGUAGE,
    XAI_TOTAL_ROWS,
    XaiLanguageProfile,
    per_class_row_total,
    per_language_row_total,
)


def test_shared_row_totals():
    assert per_class_row_total("train") == 3_612
    assert per_class_row_total("calibration") == 1_204
    assert per_class_row_total("test") == 1_206
    assert per_language_row_total() == XAI_ROWS_PER_LANGUAGE == 6_022
    assert XAI_TOTAL_ROWS == 18_066


def test_every_profile_meets_per_class_targets():
    for profile in (ENGLISH_XAI_PROFILE, PORTUGUESE_XAI_PROFILE, MANDARIN_XAI_PROFILE):
        assert dict(profile.per_class_targets) == {
            "train": 1_806,
            "calibration": 602,
            "test": 603,
        }
        assert profile.rows_per_language() == 6_022


def test_role_vocabulary_is_shared_with_contract():
    assert tuple(XAI_PER_CLASS_TARGETS) == SUPPORTED_ROLES


def test_english_native_split_roles_and_paired_flag():
    profile = ENGLISH_XAI_PROFILE

    assert profile.language_code == "eng"
    assert profile.paired_selection is True
    assert dict(profile.corpus_labels) == {"ASVspoof2024": 0, "JMDS": 1}
    assert dict(profile.corpus_native_split_roles["ASVspoof2024"]) == {
        "train": ("train",),
        "dev": ("calibration", "test"),
    }
    assert dict(profile.corpus_native_split_roles["JMDS"]) == {
        "train": ("train",),
        "dev": ("calibration", "test"),
    }
    assert dict(profile.shared_split_disjoint_fields["ASVspoof2024"]) == {
        "dev": ("speaker_id", "group_id"),
    }
    assert dict(profile.shared_split_disjoint_fields["JMDS"]) == {
        "dev": ("speaker_id", "group_id"),
    }


def test_portuguese_native_split_roles_and_unpaired_flag():
    profile = PORTUGUESE_XAI_PROFILE

    assert profile.language_code == "por"
    assert profile.paired_selection is False
    assert dict(profile.corpus_labels) == {"CORAA": 0, "MLAAD": 1}
    assert profile.shared_split_disjoint_fields == {}
    assert dict(profile.corpus_native_split_roles["CORAA"]) == {
        "train": ("train",),
        "dev": ("calibration",),
        "test": ("test",),
    }
    assert dict(profile.corpus_native_split_roles["MLAAD"]) == {
        "train": ("train",),
        "dev": ("calibration",),
        "eval": ("test",),
    }


def test_mandarin_native_split_roles_and_shared_split_blocking():
    profile = MANDARIN_XAI_PROFILE

    assert profile.language_code == "zho"
    assert profile.paired_selection is False
    assert dict(profile.corpus_labels) == {"AISHELL-3": 0, "ADD": 1}
    assert dict(profile.shared_split_disjoint_fields["AISHELL-3"]) == {
        "train": ("speaker_id",),
    }
    assert profile.shared_split_disjoint_fields.get("ADD", {}) == {}
    assert dict(profile.corpus_native_split_roles["AISHELL-3"]) == {
        "train": ("train", "calibration"),
        "test": ("test",),
    }
    assert dict(profile.corpus_native_split_roles["ADD"]) == {
        "train": ("train",),
        "dev": ("calibration",),
        "eval": ("test",),
    }


def test_profile_registry_covers_all_languages():
    assert set(XAI_PROFILE_BY_LANGUAGE) == set(SUPPORTED_LANGUAGES)
    assert XAI_PROFILE_BY_LANGUAGE["eng"] is ENGLISH_XAI_PROFILE
    assert XAI_PROFILE_BY_LANGUAGE["por"] is PORTUGUESE_XAI_PROFILE
    assert XAI_PROFILE_BY_LANGUAGE["zho"] is MANDARIN_XAI_PROFILE


def test_xai_profiles_align_with_existing_language_profiles():
    assert ENGLISH_XAI_PROFILE.language_code == ENGLISH_PROFILE.language_code
    assert ENGLISH_XAI_PROFILE.language_slug == ENGLISH_PROFILE.language_slug
    assert PORTUGUESE_XAI_PROFILE.language_code == PORTUGUESE_PROFILE.language_code
    assert PORTUGUESE_XAI_PROFILE.language_slug == PORTUGUESE_PROFILE.language_slug
    assert MANDARIN_XAI_PROFILE.language_code == MANDARIN_PROFILE.language_code
    assert MANDARIN_XAI_PROFILE.language_slug == MANDARIN_PROFILE.language_slug


def test_profile_mappings_are_read_only():
    profile = ENGLISH_XAI_PROFILE
    with pytest.raises(TypeError):
        profile.per_class_targets["train"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.corpus_native_split_roles["ASVspoof2024"]["train"] = ("test",)  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.shared_split_disjoint_fields["ASVspoof2024"]["dev"] = ("speaker_id",)  # type: ignore[index]


def test_profile_instances_are_immutable():
    profile = ENGLISH_XAI_PROFILE
    with pytest.raises(Exception):
        profile.paired_selection = False  # type: ignore[misc]


def test_xai_language_profile_rejects_mutation_of_inputs():
    mutable_targets = {"train": 1_806, "calibration": 602, "test": 603}
    mutable_roles = {
        "CORAA": {"train": ("train",), "dev": ("calibration",), "test": ("test",)},
        "MLAAD": {"train": ("train",), "dev": ("calibration",), "eval": ("test",)},
    }
    profile = XaiLanguageProfile(
        language_code="por",
        language_slug="portuguese",
        per_class_targets=mutable_targets,
        corpus_labels={"CORAA": 0, "MLAAD": 1},
        corpus_native_split_roles=mutable_roles,
        shared_split_disjoint_fields={},
        paired_selection=False,
    )
    mutable_targets["train"] = 1
    mutable_roles["CORAA"]["train"] = ("test",)
    assert dict(profile.per_class_targets)["train"] == 1_806
    assert profile.corpus_native_split_roles["CORAA"]["train"] == ("train",)


def test_xai_language_profile_rejects_unknown_language_code():
    with pytest.raises(ValueError, match="Unsupported language_code"):
        XaiLanguageProfile(
            language_code="fra",
            language_slug="french",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles=_MINIMAL_POR_CORPUS_ROLES,
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_incomplete_role_targets():
    with pytest.raises(ValueError, match="per_class_targets must contain exactly"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets={"train": 1_806, "calibration": 602},
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles=_MINIMAL_POR_CORPUS_ROLES,
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_unknown_roles_in_mappings():
    with pytest.raises(ValueError, match="Unsupported role"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                "CORAA": {"train": ("holdout",)},
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_duplicate_roles_for_one_split():
    with pytest.raises(ValueError, match="duplicate roles"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                "CORAA": {"train": ("train", "train")},
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_requires_disjoint_fields_for_multi_role_splits():
    with pytest.raises(
        ValueError,
        match="feeds multiple roles and must declare shared_split_disjoint_fields",
    ):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                "CORAA": {"train": ("train", "calibration")},
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_disjoint_fields_on_single_role_split():
    with pytest.raises(ValueError, match="feeds a single role"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                "CORAA": {"train": ("train",)},
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={"CORAA": {"train": ("speaker_id",)}},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_unknown_disjoint_field():
    with pytest.raises(ValueError, match="Unsupported shared_split_disjoint_field"):
        XaiLanguageProfile(
            language_code="eng",
            language_slug="english",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"ASVspoof2024": 0, "JMDS": 1},
            corpus_native_split_roles=_MINIMAL_ENG_CORPUS_ROLES,
            shared_split_disjoint_fields={
                "ASVspoof2024": {"dev": ("speaker_id", "unknown_field")},
                "JMDS": {"dev": ("speaker_id", "group_id")},
            },
            paired_selection=True,
        )


def test_shared_split_disjoint_field_vocabulary_is_frozen():
    assert SHARED_SPLIT_DISJOINT_FIELDS == frozenset({"speaker_id", "group_id"})


def test_corpus_labels_are_explicit_and_validated():
    assert ENGLISH_XAI_PROFILE.label_for_corpus("ASVspoof2024") == 0
    assert ENGLISH_XAI_PROFILE.label_for_corpus("JMDS") == 1
    with pytest.raises(ValueError, match="Unknown corpus"):
        ENGLISH_XAI_PROFILE.label_for_corpus("CORAA")
    with pytest.raises(ValueError, match="must map exactly one corpus to label 0"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 0},
            corpus_native_split_roles=_MINIMAL_POR_CORPUS_ROLES,
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_requires_exact_bool_paired_selection():
    with pytest.raises(ValueError, match="paired_selection must be a bool"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles=_MINIMAL_POR_CORPUS_ROLES,
            shared_split_disjoint_fields={},
            paired_selection=1,
        )


def test_xai_language_profile_rejects_non_string_corpus_keys():
    with pytest.raises(ValueError, match="corpus_native_split_roles keys must be strings"):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                1: {"train": ("train",)},  # type: ignore[dict-item]
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={},
            paired_selection=False,
        )


def test_xai_language_profile_rejects_non_string_native_split_keys():
    with pytest.raises(
        ValueError, match="corpus_native_split_roles\\['CORAA'\\] keys must be strings"
    ):
        XaiLanguageProfile(
            language_code="por",
            language_slug="portuguese",
            per_class_targets=dict(XAI_PER_CLASS_TARGETS),
            corpus_labels={"CORAA": 0, "MLAAD": 1},
            corpus_native_split_roles={
                "CORAA": {1: ("train",)},  # type: ignore[dict-item]
                "MLAAD": {"train": ("train",)},
            },
            shared_split_disjoint_fields={},
            paired_selection=False,
        )
