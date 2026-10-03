"""Tests for Portuguese metadata profile, summary and provenance builders."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.pipelines.metadata_profiling import (
    build_coraa_profile,
    build_jmds_mlaad_profile,
    build_portuguese_summary,
)
from jmds_prepare.pipelines.metadata_profiling_common import profile_field
from jmds_prepare.pipelines.metadata_provenance import build_metadata_provenance
from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE, PortugueseProfile
from jmds_prepare.sources.coraa import CORAA_COLUMNS, read_coraa_metadata
from jmds_prepare.sources.jmds_mlaad import read_portuguese_generated

FIXTURES = Path(__file__).parent / "fixtures"
CORAA_FIXTURE = FIXTURES / "coraa_metadata.csv"
JMDS_FIXTURE = FIXTURES / "jmds_mlaad_metadata.csv"

_SENSITIVE_CORAA_FIELDS = frozenset(
    {"file_path", "text", "metadata_source_file", "metadata_source_row"}
)
_SENSITIVE_JMDS_FIELDS = frozenset(
    {"spk_id", "utt_id", "audio_path", "metadata_source_file", "metadata_source_row"}
)
_VOTE_FIELDS = frozenset(
    {
        "up_votes",
        "down_votes",
        "votes_for_hesitation",
        "votes_for_filled_pause",
        "votes_for_noise_or_low_voice",
        "votes_for_second_voice",
        "votes_for_no_identified_problem",
    }
)


def _profile_with_train_count(count: int) -> PortugueseProfile:
    return replace(
        PORTUGUESE_PROFILE,
        expected_generated_counts={"train": count, "dev": 602, "eval": 603},
    )


@pytest.fixture
def jmds_root(tmp_path: Path) -> Path:
    root = tmp_path / "jmds"
    for utt_id in ("T_0000000001", "T_0000000002"):
        wav_path = PORTUGUESE_PROFILE.generated_wav_path(root, "train", utt_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFFxxxxWAVEfmt ")
    return root


@pytest.fixture
def coraa_frame() -> pd.DataFrame:
    return read_coraa_metadata(
        CORAA_FIXTURE, split="train", profile=PORTUGUESE_PROFILE
    )


@pytest.fixture
def jmds_frame(jmds_root: Path) -> pd.DataFrame:
    return read_portuguese_generated(
        JMDS_FIXTURE,
        split="train",
        jmds_root=jmds_root,
        profile=_profile_with_train_count(2),
    )


def _field_by_name(profile: dict, name: str) -> dict:
    return next(field for field in profile["fields"] if field["name"] == name)


def _assert_json_native(value: object) -> None:
    json.dumps(value)


def test_profile_field_reports_missing_and_distinct():
    frame = pd.DataFrame({"split": ["train", ""], "task": ["a", "b"]})
    payload = profile_field(
        frame,
        "split",
        description="split label",
        origin="derived",
        categorical=frozenset({"split"}),
        length_fields=frozenset(),
        vote_fields=frozenset(),
    )
    assert payload["missing_count"] == 1
    assert payload["distinct_count"] == 2
    assert payload["value_counts"] == [
        {"count": 1, "value": ""},
        {"count": 1, "value": "train"},
    ]


def test_build_coraa_profile_structure_and_counts(coraa_frame: pd.DataFrame):
    profile = build_coraa_profile(coraa_frame)

    assert profile["source"] == "CORAA"
    assert profile["sample_count"] == 3
    assert profile["counts_by_split"] == {"train": 3}
    assert [field["name"] for field in profile["fields"]] == sorted(
        [*CORAA_COLUMNS, "split", "metadata_source_file", "metadata_source_row"]
    )
    _assert_json_native(profile)


def test_coraa_profile_field_metadata_and_categorical_counts(coraa_frame: pd.DataFrame):
    profile = build_coraa_profile(coraa_frame)
    file_path = _field_by_name(profile, "file_path")
    task = _field_by_name(profile, "task")

    assert file_path["origin"] == "CORAA metadata CSV"
    assert file_path["observed_type"] == "string"
    assert file_path["nullable"] is False
    assert file_path["missing_count"] == 0
    assert file_path["distinct_count"] == 3
    assert "value_counts" not in file_path
    assert file_path["length_statistics"] == {
        "count": 3,
        "min_length": len("train/wav/speaker2/file2.wav"),
        "max_length": len("train/wav/speaker3/file3.wav"),
        "mean_length": pytest.approx(
            sum(len(value) for value in coraa_frame["file_path"]) / 3
        ),
    }

    assert task["value_counts"] == [
        {"count": 1, "value": "annotation"},
        {"count": 1, "value": "annotation_and_transcription"},
        {"count": 1, "value": "transcription"},
    ]
    assert task["origin"] == "CORAA metadata CSV"
    assert _field_by_name(profile, "split")["origin"] == (
        "Derived from source filename and validation"
    )
    assert _field_by_name(profile, "metadata_source_file")["origin"] == (
        "Derived during metadata extraction"
    )


def test_coraa_profile_hides_sensitive_values_and_profiles_votes(
    coraa_frame: pd.DataFrame,
):
    profile = build_coraa_profile(coraa_frame)

    for field in profile["fields"]:
        assert "values" not in field
        if field["name"] in _SENSITIVE_CORAA_FIELDS:
            assert "value_counts" not in field
        if field["name"] == "text":
            text_lengths = coraa_frame["text"].map(len)
            assert field["length_statistics"]["min_length"] == int(text_lengths.min())
            assert field["length_statistics"]["max_length"] == int(text_lengths.max())

    up_votes = _field_by_name(profile, "up_votes")
    assert up_votes["missing_count"] == 1
    assert up_votes["nullable"] is True
    assert up_votes["numeric_statistics"] == {
        "max": 2,
        "mean": pytest.approx(1.5),
        "min": 1,
        "non_missing_count": 2,
    }


def test_profile_statistics_are_json_safe_for_empty_values():
    coraa = pd.DataFrame(
        [
            {
                **{column: "" for column in CORAA_COLUMNS},
                "file_path": "",
                "split": "train",
                "metadata_source_file": "",
                "metadata_source_row": "",
            }
        ]
    )
    profile = build_coraa_profile(coraa)

    assert _field_by_name(profile, "up_votes")["numeric_statistics"] == {
        "non_missing_count": 0,
        "min": None,
        "max": None,
        "mean": None,
    }

    empty_profile = build_coraa_profile(coraa.iloc[0:0])
    assert _field_by_name(empty_profile, "text")["length_statistics"] == {
        "count": 0,
        "min_length": None,
        "max_length": None,
        "mean_length": None,
    }
    _assert_json_native(empty_profile)


def test_build_jmds_mlaad_profile_structure_and_counts(jmds_frame: pd.DataFrame):
    profile = build_jmds_mlaad_profile(jmds_frame)

    assert profile["source"] == "JMDS_MLAAD"
    assert profile["sample_count"] == 2
    assert profile["counts_by_split"] == {"train": 2}
    assert profile["audio_availability"]["resolved_count"] == 2
    assert profile["audio_availability"]["missing_count"] == 0
    _assert_json_native(profile)


def test_jmds_profile_hides_ids_and_paths(jmds_frame: pd.DataFrame):
    profile = build_jmds_mlaad_profile(jmds_frame)

    for field in profile["fields"]:
        assert "values" not in field
        if field["name"] in _SENSITIVE_JMDS_FIELDS:
            assert "value_counts" not in field
            assert "length_statistics" in field

    utt_id = _field_by_name(profile, "utt_id")
    assert utt_id["distinct_count"] == 2
    assert utt_id["length_statistics"]["min_length"] == len("T_0000000001")


def test_jmds_profile_includes_sorted_categorical_counts(jmds_frame: pd.DataFrame):
    profile = build_jmds_mlaad_profile(jmds_frame)
    gender = _field_by_name(profile, "gender")

    assert gender["value_counts"] == [
        {"count": 1, "value": "F"},
        {"count": 1, "value": "M"},
    ]
    assert gender["origin"] == "JMDS protocol CSV"
    assert _field_by_name(profile, "audio_path")["origin"] == (
        "Derived by resolving generated WAV paths"
    )


def test_build_portuguese_summary_keeps_sources_separate(
    coraa_frame: pd.DataFrame,
    jmds_frame: pd.DataFrame,
):
    summary = build_portuguese_summary(coraa_frame, jmds_frame)

    assert summary["sources"]["CORAA"]["label_role"] == "pristine"
    assert summary["sources"]["JMDS_MLAAD"]["label_role"] == "generated"
    assert summary["sources"]["CORAA"]["sample_count"] == 3
    assert summary["sources"]["JMDS_MLAAD"]["sample_count"] == 2
    assert "sample_pairs" not in summary
    assert summary["comparison_limitations"]["paired_samples"] is False
    assert summary["sources"]["CORAA"]["distributions"]["dataset"] == [
        {"count": 3, "value": "CORAA"},
    ]
    assert summary["sources"]["CORAA"]["distributions"]["subcorpus"] == [
        {"count": 3, "value": "CORAA"},
    ]
    assert summary["sources"]["JMDS_MLAAD"]["distributions"]["dataset"] == [
        {"count": 2, "value": "MLAAD"}
    ]
    assert set(summary["sources"]["CORAA"]["quality_indicators"]) == _VOTE_FIELDS
    for indicator in summary["sources"]["CORAA"]["quality_indicators"].values():
        assert set(indicator) == {
            "missing_count",
            "non_missing_count",
            "numeric_statistics",
        }
    assert summary["comparable_characteristics"]["paired_samples"] is False
    assert summary["comparable_characteristics"]["characteristics"]["dataset"] == {
        "CORAA": {"available": True, "field": "dataset"},
        "JMDS_MLAAD": {"available": True, "field": "dataset"},
    }
    summary_text = json.dumps(summary)
    for forbidden in ("file_path", "audio_path", "utt_id", '"text"'):
        assert forbidden not in summary_text
    assert "timestamp" not in json.dumps(summary)
    _assert_json_native(summary)


def test_build_metadata_provenance_records_inputs_and_exclusion(
    tmp_path: Path,
):
    coraa_path = tmp_path / "metadata_train_final.csv"
    jmds_path = tmp_path / "open_v2_train.cm.csv"
    coraa_path.write_text("coraa-bytes\n", encoding="utf-8")
    jmds_path.write_text("jmds-bytes\n", encoding="utf-8")

    def fake_sha256(path: Path) -> str:
        return {
            str(coraa_path): "coraa-hash",
            str(jmds_path): "jmds-hash",
        }[str(path)]

    provenance = build_metadata_provenance(
        profile=PORTUGUESE_PROFILE,
        input_paths={
            "coraa_train": coraa_path,
            "jmds_train": jmds_path,
        },
        sha256_file=fake_sha256,
    )

    assert provenance["coraa"]["license"] == PORTUGUESE_PROFILE.coraa_license
    assert provenance["coraa"]["revision"] == PORTUGUESE_PROFILE.coraa_revision
    assert provenance["coraa"]["version"] == PORTUGUESE_PROFILE.coraa_version
    assert provenance["coraa"]["usage_policy"] == {
        "adapted_material_sharing_permitted": False,
        "commercial_use_permitted": False,
        "preserve_originals": True,
        "statement": PORTUGUESE_PROFILE.coraa_usage_policy_statement,
    }
    assert provenance["excluded_jmds_pristine_count"] == 1_000
    assert provenance["excluded_jmds_pristine_statement"] == (
        "Portuguese JMDS pristine rows referencing anonymized CORAA IDs were "
        "excluded; no public sample-level mapping exists between JMDS pristine "
        "IDs and CORAA file paths."
    )
    assert provenance["inputs"] == [
        {
            "key": "coraa_train",
            "path": str(coraa_path),
            "sha256": "coraa-hash",
            "size_bytes": coraa_path.stat().st_size,
        },
        {
            "key": "jmds_train",
            "path": str(jmds_path),
            "sha256": "jmds-hash",
            "size_bytes": jmds_path.stat().st_size,
        },
    ]
    assert "coraa_metadata_profile" in provenance["artifact_schemas"]
    assert provenance["field_origins"]["coraa_metadata_csv"]["file_path"] == (
        "CORAA metadata CSV"
    )
    assert provenance["field_origins"]["jmds_mlaad_generated_metadata_csv"][
        "utt_id"
    ] == "JMDS protocol CSV"
    assert provenance["jmds_mlaad"] == {
        "jmds_release": "v2",
        "generated_dataset": "MLAAD",
        "generated_dataset_version": None,
        "license": None,
        "missing_reasons": {
            "generated_dataset_version": (
                "No separate MLAAD version is stated in the JMDS v2 protocol."
            ),
            "license": "No MLAAD license is stated in the local JMDS v2 metadata.",
        },
    }
    assert "jmds_mlaad" in provenance["artifact_schemas"][
        "portuguese_metadata_provenance"
    ]["sections"]
    assert "timestamp" not in json.dumps(provenance)
    _assert_json_native(provenance)
