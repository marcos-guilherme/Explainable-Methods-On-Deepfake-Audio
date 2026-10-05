"""Tests for the corpus-agnostic XAI sample contract."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from jmds_prepare.core.xai_sample import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_ROLES,
    XAI_SAMPLE_COLUMNS,
    XaiSample,
)


def _valid_row(**overrides: Any) -> dict[str, str]:
    row = {
        "sample_id": "eng-train-bonafide-00001",
        "language": "eng",
        "role": "train",
        "label": "0",
        "corpus": "ASVspoof2024",
        "native_split": "train",
        "original_ref": "raw/asvspoof5/train/T_0000000001.flac",
        "processed_path": "data/eng/train/T_0000000001.wav",
        "speaker_id": "T_0001",
        "group_id": "",
        "attack_id": "",
        "sha256_source": "a" * 64,
        "sha256_processed": "b" * 64,
        "selection_seed": "42",
        "selection_rank": "0",
        "selection_reason": "paired_train_quota",
        "selection_source": "jmds_protocol:open_v2_train.cm.csv",
    }
    row.update(overrides)
    return row


def _valid_sample(**overrides: Any) -> XaiSample:
    return XaiSample.from_dict(_valid_row(**overrides))


def test_canonical_column_order_is_fixed():
    assert XAI_SAMPLE_COLUMNS == [
        "sample_id",
        "language",
        "role",
        "label",
        "corpus",
        "native_split",
        "original_ref",
        "processed_path",
        "speaker_id",
        "group_id",
        "attack_id",
        "sha256_source",
        "sha256_processed",
        "selection_seed",
        "selection_rank",
        "selection_reason",
        "selection_source",
    ]


def test_supported_vocabularies_are_frozen():
    assert SUPPORTED_LANGUAGES == ("eng", "por", "zho")
    assert SUPPORTED_ROLES == ("train", "calibration", "test")


def test_from_dict_and_to_dict_roundtrip():
    row = _valid_row()
    sample = XaiSample.from_dict(row)

    assert sample.sample_id == row["sample_id"]
    assert sample.language == "eng"
    assert sample.role == "train"
    assert sample.label == 0
    assert sample.speaker_id == "T_0001"
    assert sample.group_id is None
    assert sample.attack_id is None
    assert sample.selection_seed == 42
    assert sample.selection_rank == 0

    roundtrip = sample.to_dict()
    assert list(roundtrip.keys()) == XAI_SAMPLE_COLUMNS
    assert roundtrip == row
    assert XaiSample.from_dict(roundtrip) == sample


def test_direct_construction_is_validated():
    sample = _valid_sample()
    direct = XaiSample(**dataclasses.asdict(sample))
    assert direct == sample

    with pytest.raises(ValueError, match="Unsupported language"):
        XaiSample(
            **{
                **dataclasses.asdict(sample),
                "language": "fra",
            }
        )


@pytest.mark.parametrize(
    "bad_label",
    [True, False, 1.0, "0", 2],
)
def test_direct_construction_rejects_non_canonical_label(bad_label: Any):
    base = dataclasses.asdict(_valid_sample())
    base["label"] = bad_label
    with pytest.raises(ValueError, match="Unsupported label"):
        XaiSample(**base)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_seed", True),
        ("selection_seed", 42.0),
        ("selection_seed", "42"),
        ("selection_rank", False),
        ("selection_rank", 1.5),
        ("selection_rank", "0"),
    ],
)
def test_direct_construction_rejects_non_canonical_int_fields(
    field: str, value: Any
):
    base = dataclasses.asdict(_valid_sample())
    base[field] = value
    with pytest.raises(ValueError, match=f"Invalid {field}"):
        XaiSample(**base)


def test_direct_construction_normalizes_optional_whitespace():
    base = dataclasses.asdict(_valid_sample(label="1", attack_id="A01"))
    base["speaker_id"] = "  T_0001  "
    base["group_id"] = "  "
    base["attack_id"] = "\tA01\t"
    sample = XaiSample(**base)

    assert sample.speaker_id == "T_0001"
    assert sample.group_id is None
    assert sample.attack_id == "A01"


def test_direct_construction_requires_string_text_fields():
    base = dataclasses.asdict(_valid_sample())
    base["sample_id"] = 123
    with pytest.raises(ValueError, match="sample_id must be a string"):
        XaiSample(**base)


@pytest.mark.parametrize("field", ["sha256_source", "sha256_processed"])
def test_from_dict_rejects_uppercase_sha256(field: str):
    row = _valid_row(**{field: ("A" * 64)})
    with pytest.raises(ValueError, match=f"Invalid {field}"):
        XaiSample.from_dict(row)


@pytest.mark.parametrize("field", ["sha256_source", "sha256_processed"])
def test_direct_construction_rejects_uppercase_sha256(field: str):
    base = dataclasses.asdict(_valid_sample())
    base[field] = "A" * 64
    with pytest.raises(ValueError, match=f"Invalid {field}"):
        XaiSample(**base)


def test_sha256_is_canonical_lowercase():
    sample = XaiSample.from_dict(_valid_row())
    assert sample.sha256_source == "a" * 64
    assert sample.sha256_processed == "b" * 64


def test_optional_fields_accept_empty_strings():
    row = _valid_row(speaker_id="", group_id="", attack_id="")
    sample = XaiSample.from_dict(row)

    assert sample.speaker_id is None
    assert sample.group_id is None
    assert sample.attack_id is None
    assert sample.to_dict()["speaker_id"] == ""


def test_optional_fields_normalize_whitespace_only_to_none():
    row = _valid_row(speaker_id="   ", group_id="\t", attack_id=" ")
    sample = XaiSample.from_dict(row)

    assert sample.speaker_id is None
    assert sample.group_id is None
    assert sample.attack_id is None


def test_spoof_label_is_one():
    sample = XaiSample.from_dict(_valid_row(label="1", attack_id="A01"))
    assert sample.label == 1
    assert sample.attack_id == "A01"


def test_spoof_label_allows_missing_attack_id():
    sample = XaiSample.from_dict(_valid_row(label="1", attack_id=""))
    assert sample.label == 1
    assert sample.attack_id is None


def test_bonafide_label_rejects_attack_id():
    with pytest.raises(ValueError, match="attack_id must be empty for label 0"):
        XaiSample.from_dict(_valid_row(label="0", attack_id="A01"))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("language", "fra", "Unsupported language"),
        ("role", "validation", "Unsupported role"),
        ("label", "2", "Unsupported label"),
        ("label", "bonafide", "Unsupported label"),
        ("sha256_source", "short", "Invalid sha256_source"),
        ("sha256_processed", "g" * 63, "Invalid sha256_processed"),
        ("selection_seed", "not-int", "Invalid selection_seed"),
        ("selection_rank", "-1", "Invalid selection_rank"),
        ("sample_id", "", "sample_id must be non-empty"),
        ("corpus", "", "corpus must be non-empty"),
        ("native_split", "", "native_split must be non-empty"),
        ("original_ref", "", "original_ref must be non-empty"),
        ("processed_path", "", "processed_path must be non-empty"),
        ("selection_reason", "", "selection_reason must be non-empty"),
        ("selection_source", "", "selection_source must be non-empty"),
    ],
)
def test_from_dict_rejects_invalid_values(field: str, value: str, match: str):
    row = _valid_row(**{field: value})
    with pytest.raises(ValueError, match=match):
        XaiSample.from_dict(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", True),
        ("label", False),
        ("label", 1.0),
        ("label", "01"),
        ("label", " 0"),
        ("label", "0 "),
        ("selection_seed", True),
        ("selection_seed", 42.0),
        ("selection_seed", "042"),
        ("selection_seed", " 42"),
        ("selection_rank", False),
        ("selection_rank", 1.0),
        ("selection_rank", "01"),
        ("selection_rank", " 0"),
    ],
)
def test_from_dict_rejects_silent_coercions(field: str, value: Any):
    row = _valid_row(**{field: value})
    with pytest.raises(ValueError):
        XaiSample.from_dict(row)


def test_from_dict_accepts_int_label_and_rank():
    sample = XaiSample.from_dict(
        _valid_row(label=0, selection_seed=7, selection_rank=0)
    )
    assert sample.label == 0
    assert sample.selection_seed == 7
    assert sample.selection_rank == 0


def test_from_dict_rejects_missing_columns():
    row = _valid_row()
    del row["selection_source"]
    with pytest.raises(ValueError, match="Missing required columns"):
        XaiSample.from_dict(row)


def test_from_dict_rejects_extra_columns():
    row = _valid_row(extra="unexpected")
    with pytest.raises(ValueError, match="Unexpected columns"):
        XaiSample.from_dict(row)


def test_from_dict_rejects_wrong_column_order_when_strict():
    row = _valid_row()
    shuffled = {key: row[key] for key in reversed(XAI_SAMPLE_COLUMNS)}
    with pytest.raises(ValueError, match="Column order"):
        XaiSample.from_dict(shuffled, enforce_column_order=True)


def test_validate_dict_accepts_valid_row_without_constructing():
    row = _valid_row()
    XaiSample.validate_dict(row)
    with pytest.raises(ValueError):
        XaiSample.validate_dict(_valid_row(label="9"))


def test_sample_is_immutable():
    sample = XaiSample.from_dict(_valid_row())
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.language = "por"  # type: ignore[misc]
