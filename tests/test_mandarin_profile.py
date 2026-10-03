"""Tests for the immutable Mandarin profile and AISHELL/JMDS path helpers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE


def test_mandarin_profile_pins_add_layout():
    root = Path("E:/JMDS")
    assert MANDARIN_PROFILE.generated_wav_path(
        root, "train", "T_0000123456"
    ) == (
        root
        / "dataset"
        / "Chinese_ADD_Generated"
        / "train"
        / "wav"
        / "T_0000123456.wav"
    )
    assert dict(MANDARIN_PROFILE.expected_generated_counts) == {
        "train": 7146,
        "dev": 7497,
        "eval": 9999,
    }


def test_mandarin_artifact_names_match_spec():
    names = MANDARIN_PROFILE.artifact_names
    assert names.pristine_manifest == "aishell3_metadata.csv"
    assert names.generated_manifest == "jmds_add_generated_metadata.csv"
    assert names.pristine_profile == "aishell3_metadata_profile.json"
    assert names.generated_profile == "jmds_add_generated_metadata_profile.json"
    assert names.summary == "mandarin_metadata_summary.json"
    assert names.provenance == "mandarin_metadata_provenance.json"


def test_mandarin_profile_holds_canonical_values():
    profile = MANDARIN_PROFILE

    assert profile.language_code == "zho"
    assert profile.language_slug == "mandarin"
    assert profile.generated_dataset == "ADD"
    assert profile.generated_dir_name == "Chinese_ADD_Generated"
    assert profile.generated_audio_subdir == "wav"
    assert profile.protocol_splits == ("train", "dev", "eval")
    assert profile.aishell_splits == ("train", "test")
    assert dict(profile.split_id_prefixes) == {
        "train": "T",
        "dev": "D",
        "eval": "E",
    }
    assert profile.excluded_jmds_pristine_count == 4410


def test_mandarin_aishell_archive_members():
    members = dict(MANDARIN_PROFILE.aishell_archive_members)
    assert members["spk-info"] == "spk-info.txt"
    assert members["train/content"] == "train/content.txt"
    assert members["test/content"] == "test/content.txt"
    assert members["train/label"] == "train/label_train-set.txt"


def test_mandarin_profile_path_helpers(tmp_path):
    profile = MANDARIN_PROFILE
    jmds_root = tmp_path / "jmds"

    assert profile.jmds_protocol_path(jmds_root, "dev") == (
        jmds_root / "cm_protocols" / "open_v2_dev.cm.csv"
    )


def test_mandarin_profile_is_deeply_immutable():
    profile = MANDARIN_PROFILE

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.language_code = "por"  # type: ignore[misc]
    for mapping in (
        profile.expected_generated_counts,
        profile.split_id_prefixes,
        profile.aishell_archive_members,
    ):
        with pytest.raises(TypeError):
            mapping["train"] = "changed"  # type: ignore[index]
    assert isinstance(profile.protocol_splits, tuple)
    assert isinstance(profile.aishell_splits, tuple)


def test_mandarin_profile_rejects_unsupported_splits(tmp_path):
    with pytest.raises(ValueError, match="Unsupported split"):
        MANDARIN_PROFILE.jmds_protocol_path(tmp_path, "test")
    with pytest.raises(ValueError, match="Unsupported split"):
        MANDARIN_PROFILE.generated_wav_path(tmp_path, "test", "T_0000000001")
