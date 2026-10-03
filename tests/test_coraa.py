from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE
from jmds_prepare.sources.coraa import CORAA_COLUMNS, read_coraa_metadata

FIXTURES = Path(__file__).parent / "fixtures"
CORAA_FIXTURE = FIXTURES / "coraa_metadata.csv"


def test_reads_valid_coraa_metadata_with_traceability():
    frame = read_coraa_metadata(
        CORAA_FIXTURE, split="train", profile=PORTUGUESE_PROFILE
    )

    assert list(frame.columns) == [
        *CORAA_COLUMNS,
        "split",
        "metadata_source_file",
        "metadata_source_row",
    ]
    assert frame["metadata_source_row"].tolist() == ["2", "3", "4"]
    assert frame["metadata_source_file"].tolist() == [
        CORAA_FIXTURE.name,
        CORAA_FIXTURE.name,
        CORAA_FIXTURE.name,
    ]
    assert frame["split"].tolist() == ["train", "train", "train"]
    assert frame["file_path"].is_unique
    assert frame["task"].tolist() == [
        "annotation",
        "transcription",
        "annotation_and_transcription",
    ]
    assert frame.loc[1, "up_votes"] == ""
    assert frame.loc[1, "text"] == "Este e um texto de transcricao"


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (CORAA_COLUMNS[:-1], "missing columns"),
        (CORAA_COLUMNS + ["duration"], "extra columns"),
    ],
)
def test_rejects_missing_or_extra_coraa_columns(tmp_path, columns, message):
    path = tmp_path / "coraa.csv"
    pd.DataFrame(columns=columns).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_coraa_metadata(path, split="train", profile=PORTUGUESE_PROFILE)


def test_rejects_reordered_coraa_columns(tmp_path):
    protocol = pd.read_csv(CORAA_FIXTURE, dtype=str, keep_default_na=False)
    reordered = [CORAA_COLUMNS[1], CORAA_COLUMNS[0], *CORAA_COLUMNS[2:]]
    path = tmp_path / "coraa.csv"
    protocol[reordered].to_csv(path, index=False)

    with pytest.raises(ValueError, match="required order"):
        read_coraa_metadata(path, split="train", profile=PORTUGUESE_PROFILE)


@pytest.mark.parametrize(
    ("file_path", "message"),
    [
        ("dev/wav/speaker/file.wav", "split"),
        ("/train/wav/speaker/file.wav", "absolute"),
        ("train/wav/../speaker/file.wav", r"\.\."),
        (r"train\wav\speaker\file.wav", "backslash"),
        ("", r"(?i)empty.*file_path"),
    ],
)
def test_rejects_invalid_coraa_file_paths(tmp_path, file_path, message):
    protocol = pd.read_csv(CORAA_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[0, "file_path"] = file_path
    path = tmp_path / "coraa.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_coraa_metadata(path, split="train", profile=PORTUGUESE_PROFILE)


def test_rejects_duplicate_coraa_file_paths(tmp_path):
    protocol = pd.read_csv(CORAA_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[1, "file_path"] = protocol.loc[0, "file_path"]
    path = tmp_path / "coraa.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate.*file_path"):
        read_coraa_metadata(path, split="train", profile=PORTUGUESE_PROFILE)


@pytest.mark.parametrize("split", ["eval", "unsupported"])
def test_rejects_unsupported_coraa_splits(split):
    with pytest.raises(ValueError, match=split):
        read_coraa_metadata(CORAA_FIXTURE, split=split, profile=PORTUGUESE_PROFILE)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("up_votes", "abc", "up_votes"),
        ("down_votes", "-1", "down_votes"),
        ("votes_for_hesitation", "1.5", "votes_for_hesitation"),
        ("votes_for_filled_pause", "x", "votes_for_filled_pause"),
        ("votes_for_noise_or_low_voice", "-2", "votes_for_noise_or_low_voice"),
        ("votes_for_second_voice", "bad", "votes_for_second_voice"),
        ("votes_for_no_identified_problem", "-3", "votes_for_no_identified_problem"),
    ],
)
def test_rejects_invalid_coraa_vote_fields(tmp_path, field, value, message):
    protocol = pd.read_csv(CORAA_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[0, field] = value
    path = tmp_path / "coraa.csv"
    protocol.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_coraa_metadata(path, split="train", profile=PORTUGUESE_PROFILE)
