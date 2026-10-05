"""Tests for the immutable Portuguese profile and shared JMDS contracts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE
from jmds_prepare.protocols import JMDS_COLUMNS, read_jmds_protocol

FIXTURES = Path(__file__).parent / "fixtures"


def test_portuguese_profile_pins_real_layout():
    root = Path("E:/JMDS")
    assert PORTUGUESE_PROFILE.generated_wav_path(
        root, "train", "T_0000025757"
    ) == (
        root
        / "dataset"
        / "Portugese_MLAAD_Generated"
        / "train"
        / "wav"
        / "T_0000025757.wav"
    )
    assert dict(PORTUGUESE_PROFILE.expected_generated_counts) == {
        "train": 1806,
        "dev": 602,
        "eval": 603,
    }


def test_portuguese_profile_mappings_are_immutable():
    with pytest.raises(TypeError):
        PORTUGUESE_PROFILE.expected_coraa_counts["train"] = 1


def test_portuguese_profile_holds_canonical_values():
    profile = PORTUGUESE_PROFILE

    assert profile.language_code == "por"
    assert profile.language_slug == "portuguese"
    assert profile.generated_dataset == "MLAAD"
    assert profile.generated_dir_name == "Portugese_MLAAD_Generated"
    assert profile.generated_audio_subdir == "wav"
    assert profile.protocol_splits == ("train", "dev", "eval")
    assert profile.coraa_splits == ("train", "dev", "test")
    assert dict(profile.coraa_metadata_files) == {
        "train": "metadata_train_final.csv",
        "dev": "metadata_dev_final.csv",
        "test": "metadata_test_final.csv",
    }
    assert dict(profile.expected_coraa_counts) == {
        "train": 382_258,
        "dev": 7_522,
        "test": 12_676,
    }
    assert dict(profile.split_id_prefixes) == {
        "train": "T",
        "dev": "D",
        "eval": "E",
    }
    assert profile.accepted_labels == frozenset({"pristine", "generated"})
    assert profile.jmds_release == "v2"
    assert profile.coraa_version == "v1.1"
    assert profile.coraa_revision == "719c91226a79f5f9a8984145f15f29626eabc29a"
    assert profile.coraa_license == "CC-BY-NC-ND-4.0"


def test_portuguese_profile_path_helpers(tmp_path):
    profile = PORTUGUESE_PROFILE
    jmds_root = tmp_path / "jmds"
    coraa_root = tmp_path / "coraa"

    assert profile.jmds_protocol_path(jmds_root, "train") == (
        jmds_root / "cm_protocols" / "open_v2_train.cm.csv"
    )
    assert profile.coraa_metadata_path(coraa_root, "dev") == (
        coraa_root / "metadata_dev_final.csv"
    )


def test_portuguese_profile_is_deeply_immutable():
    profile = PORTUGUESE_PROFILE

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.language_code = "eng"  # type: ignore[misc]
    for mapping in (
        profile.coraa_metadata_files,
        profile.expected_coraa_counts,
        profile.expected_generated_counts,
        profile.split_id_prefixes,
    ):
        with pytest.raises(TypeError):
            mapping["train"] = "changed"  # type: ignore[index]
    assert isinstance(profile.protocol_splits, tuple)
    assert isinstance(profile.coraa_splits, tuple)


def test_portuguese_profile_rejects_unsupported_splits(tmp_path):
    with pytest.raises(ValueError, match="Unsupported split"):
        PORTUGUESE_PROFILE.jmds_protocol_path(tmp_path, "test")
    with pytest.raises(ValueError, match="Unsupported split"):
        PORTUGUESE_PROFILE.generated_wav_path(tmp_path, "test", "T_0000000001")
    with pytest.raises(ValueError, match="Unsupported split"):
        PORTUGUESE_PROFILE.coraa_metadata_path(tmp_path, "eval")


@pytest.mark.parametrize("split", ["eval", "test"])
def test_english_reader_rejects_unsupported_split(split):
    with pytest.raises(ValueError, match=split):
        read_jmds_protocol(FIXTURES / "jmds_train.csv", split)


def test_english_reader_rejects_reordered_columns(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    reordered = [JMDS_COLUMNS[1], JMDS_COLUMNS[0], *JMDS_COLUMNS[2:]]
    path = tmp_path / "jmds.csv"
    protocol[reordered].to_csv(path, index=False)

    with pytest.raises(ValueError, match="required order"):
        read_jmds_protocol(path, "train")


def test_english_reader_rejects_malformed_ids(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    protocol.loc[0, "utt_id"] = "bad-id"
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Malformed.*utt_id"):
        read_jmds_protocol(path, "train")


def test_english_reader_rejects_duplicate_ids(tmp_path):
    protocol = pd.read_csv(FIXTURES / "jmds_train.csv", dtype=str)
    protocol.loc[1, "utt_id"] = protocol.loc[0, "utt_id"]
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_jmds_protocol(path, "train")
