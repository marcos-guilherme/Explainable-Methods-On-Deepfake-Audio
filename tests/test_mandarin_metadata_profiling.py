"""Tests for Mandarin metadata profile, summary and provenance builders."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.pipelines.mandarin_metadata_profiling import (
    build_aishell3_profile,
    build_jmds_add_profile,
    build_mandarin_summary,
)
from jmds_prepare.pipelines.mandarin_metadata_provenance import (
    build_mandarin_metadata_provenance,
)
from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE, MandarinProfile
from jmds_prepare.sources.aishell3 import (
    AISHELL3_COLUMNS,
    read_aishell3_metadata,
)
from jmds_prepare.sources.jmds_add import read_mandarin_generated

FIXTURES = Path(__file__).parent / "fixtures"
AISHELL_FIXTURE = FIXTURES / "aishell3_mini.tgz"
JMDS_FIXTURE = FIXTURES / "jmds_add_metadata.csv"

_SENSITIVE_AISHELL_FIELDS = frozenset(
    {
        "utt_id",
        "archive_member_path",
        "transcription",
        "pinyin",
        "metadata_source_file",
        "metadata_source_row",
    }
)
_SENSITIVE_JMDS_FIELDS = frozenset(
    {"spk_id", "utt_id", "audio_path", "metadata_source_file", "metadata_source_row"}
)


def _profile_with_train_count(count: int) -> MandarinProfile:
    return replace(
        MANDARIN_PROFILE,
        expected_generated_counts={"train": count, "dev": 7497, "eval": 9999},
    )


@pytest.fixture
def jmds_root(tmp_path: Path) -> Path:
    root = tmp_path / "jmds"
    for utt_id in ("T_0000123456", "T_0000654321"):
        wav_path = MANDARIN_PROFILE.generated_wav_path(root, "train", utt_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFFxxxxWAVEfmt ")
    return root


@pytest.fixture
def aishell_frame_and_reconciliation():
    return read_aishell3_metadata(AISHELL_FIXTURE, profile=MANDARIN_PROFILE)


@pytest.fixture
def aishell_frame(aishell_frame_and_reconciliation):
    frame, _ = aishell_frame_and_reconciliation
    return frame


@pytest.fixture
def reconciliation(aishell_frame_and_reconciliation):
    _, reconciliation = aishell_frame_and_reconciliation
    return reconciliation


@pytest.fixture
def jmds_frame(jmds_root: Path) -> pd.DataFrame:
    return read_mandarin_generated(
        JMDS_FIXTURE,
        split="train",
        jmds_root=jmds_root,
        profile=_profile_with_train_count(2),
    )


def _field_by_name(profile: dict, name: str) -> dict:
    return next(field for field in profile["fields"] if field["name"] == name)


def _assert_json_native(value: object) -> None:
    json.dumps(value)


def test_build_aishell3_profile_includes_reconciliation(
    aishell_frame: pd.DataFrame,
    reconciliation,
):
    profile = build_aishell3_profile(aishell_frame, reconciliation=reconciliation)

    assert profile["source"] == "AISHELL3"
    assert profile["sample_count"] == 4
    assert profile["counts_by_split"] == {"test": 2, "train": 2}
    assert profile["reconciliation"]["official_test_sample_count"] == 23262
    assert profile["reconciliation"]["observed_test_content_line_count"] == 3
    assert (
        "24,773" in profile["reconciliation"]["divergence_explanation"]
        or "24773" in profile["reconciliation"]["divergence_explanation"]
    )
    test_report = profile["reconciliation"]["by_split"]["test"]
    assert test_report["matched_count"] == 2
    assert test_report["content_without_wav"] == ["SSB00050005"]
    assert [field["name"] for field in profile["fields"]] == sorted(AISHELL3_COLUMNS)
    _assert_json_native(profile)


def test_aishell3_profile_hides_sensitive_values(aishell_frame: pd.DataFrame, reconciliation):
    profile = build_aishell3_profile(aishell_frame, reconciliation=reconciliation)

    for field in profile["fields"]:
        assert "values" not in field
        if field["name"] in _SENSITIVE_AISHELL_FIELDS:
            assert "value_counts" not in field
            assert "length_statistics" in field

    transcription = _field_by_name(profile, "transcription")
    text_lengths = aishell_frame["transcription"].map(len)
    assert transcription["length_statistics"]["min_length"] == int(text_lengths.min())
    assert transcription["length_statistics"]["max_length"] == int(text_lengths.max())


def test_aishell3_profile_includes_categorical_counts(
    aishell_frame: pd.DataFrame,
    reconciliation,
):
    profile = build_aishell3_profile(aishell_frame, reconciliation=reconciliation)
    gender = _field_by_name(profile, "gender")
    prosody = _field_by_name(profile, "prosody_available")

    assert gender["value_counts"] == [
        {"count": 4, "value": "female"},
    ]
    assert prosody["value_counts"] == [
        {"count": 2, "value": "no"},
        {"count": 2, "value": "yes"},
    ]
    assert _field_by_name(profile, "speaker_id")["origin"] == (
        "Derived from AISHELL-3 utt_id prefix"
    )


def test_build_jmds_add_profile_structure_and_counts(jmds_frame: pd.DataFrame):
    profile = build_jmds_add_profile(jmds_frame)

    assert profile["source"] == "JMDS_ADD"
    assert profile["sample_count"] == 2
    assert profile["counts_by_split"] == {"train": 2}
    assert profile["audio_availability"]["resolved_count"] == 2
    assert profile["audio_availability"]["missing_count"] == 0
    _assert_json_native(profile)


def test_jmds_add_profile_hides_ids_and_paths(jmds_frame: pd.DataFrame):
    profile = build_jmds_add_profile(jmds_frame)

    for field in profile["fields"]:
        assert "values" not in field
        if field["name"] in _SENSITIVE_JMDS_FIELDS:
            assert "value_counts" not in field
            assert "length_statistics" in field


def test_jmds_add_profile_reports_sparse_categorical_values(jmds_frame: pd.DataFrame):
    profile = build_jmds_add_profile(jmds_frame)

    assert _field_by_name(profile, "attack_id")["value_counts"] == [
        {"count": 2, "value": "unk"},
    ]
    assert _field_by_name(profile, "codec")["value_counts"] == [
        {"count": 2, "value": "-"},
    ]
    assert _field_by_name(profile, "native")["value_counts"] == [
        {"count": 2, "value": "yes"},
    ]


def test_build_mandarin_summary_keeps_sources_separate(
    aishell_frame: pd.DataFrame,
    jmds_frame: pd.DataFrame,
):
    summary = build_mandarin_summary(aishell_frame, jmds_frame)

    assert summary["comparison_limitations"]["paired_samples"] is False
    assert "sample_pairs" not in summary
    assert summary["sources"]["AISHELL3"]["label_role"] == "pristine"
    assert summary["sources"]["JMDS_ADD"]["label_role"] == "generated"
    assert summary["sources"]["AISHELL3"]["sample_count"] == 4
    assert summary["sources"]["JMDS_ADD"]["sample_count"] == 2
    assert summary["comparable_characteristics"]["paired_samples"] is False
    summary_text = json.dumps(summary)
    for forbidden in ("archive_member_path", "audio_path", "utt_id", "你好", "训练"):
        assert forbidden not in summary_text
    assert "timestamp" not in summary_text
    _assert_json_native(summary)


def test_build_mandarin_metadata_provenance_records_inputs_and_exclusion(
    tmp_path: Path,
    reconciliation,
):
    archive_path = tmp_path / "data_aishell3.tgz"
    train_protocol = tmp_path / "open_v2_train.cm.csv"
    dev_protocol = tmp_path / "open_v2_dev.cm.csv"
    eval_protocol = tmp_path / "open_v2_eval.cm.csv"
    archive_path.write_bytes(b"archive-bytes")
    train_protocol.write_text("train-bytes\n", encoding="utf-8")
    dev_protocol.write_text("dev-bytes\n", encoding="utf-8")
    eval_protocol.write_text("eval-bytes\n", encoding="utf-8")

    def fake_sha256(path: Path) -> str:
        return {
            str(archive_path): "deadbeef",
            str(train_protocol): "train-hash",
            str(dev_protocol): "dev-hash",
            str(eval_protocol): "eval-hash",
        }[str(path)]

    provenance = build_mandarin_metadata_provenance(
        profile=MANDARIN_PROFILE,
        input_paths={
            "aishell_archive": archive_path,
            "jmds_train": train_protocol,
            "jmds_dev": dev_protocol,
            "jmds_eval": eval_protocol,
        },
        sha256_file=fake_sha256,
        reconciliation=reconciliation,
    )

    assert provenance["excluded_jmds_pristine_count"] == 4410
    assert provenance["aishell3"]["license"] == "Apache-2.0"
    assert provenance["inputs"][0]["sha256"] == "deadbeef"
    assert provenance["inputs"][0]["key"] == "aishell_archive"
    assert {entry["key"] for entry in provenance["inputs"]} == {
        "aishell_archive",
        "jmds_dev",
        "jmds_eval",
        "jmds_train",
    }
    assert provenance["aishell3"]["reconciliation"]["official_test_sample_count"] == 23262
    assert provenance["jmds_add"]["generated_dataset"] == "ADD"
    assert provenance["jmds_add"]["missing_reasons"]["attack_id"] == (
        "All ADD rows report attack_id=unk; no attack taxonomy is encoded."
    )
    assert provenance["jmds_add"]["missing_reasons"]["gender"] == (
        "All ADD rows report gender=unk; speaker gender is not provided."
    )
    assert provenance["jmds_add"]["missing_reasons"]["spk_id"] == (
        "All ADD rows report spk_id=unk; speaker identity is not provided."
    )
    assert provenance["jmds_add"]["missing_reasons"]["codec"] == (
        "All ADD rows report codec=-; no codec label is provided."
    )
    assert "aishell3_metadata_profile" in provenance["artifact_schemas"]
    assert "timestamp" not in json.dumps(provenance)
    _assert_json_native(provenance)
