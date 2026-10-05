"""Strict CORAA metadata CSV adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiles.portuguese import PortugueseProfile
from .jmds_common import validate_schema, validate_split

CORAA_COLUMNS = [
    "file_path",
    "task",
    "variety",
    "dataset",
    "accent",
    "speech_genre",
    "speech_style",
    "up_votes",
    "down_votes",
    "votes_for_hesitation",
    "votes_for_filled_pause",
    "votes_for_noise_or_low_voice",
    "votes_for_second_voice",
    "votes_for_no_identified_problem",
    "text",
]

_VOTE_COLUMNS = (
    "up_votes",
    "down_votes",
    "votes_for_hesitation",
    "votes_for_filled_pause",
    "votes_for_noise_or_low_voice",
    "votes_for_second_voice",
    "votes_for_no_identified_problem",
)

_TRACEABILITY_COLUMNS = ("split", "metadata_source_file", "metadata_source_row")


def read_coraa_metadata(
    path: Path,
    *,
    split: str,
    profile: PortugueseProfile,
) -> pd.DataFrame:
    validate_split(split, profile.coraa_splits)
    metadata = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_schema(list(metadata.columns), CORAA_COLUMNS, source="CORAA")
    _validate_file_paths(metadata, split=split)
    _validate_vote_fields(metadata)
    _validate_unique_file_paths(metadata)
    return _with_traceability(metadata, path=path, split=split)


def _validate_file_paths(frame: pd.DataFrame, *, split: str) -> None:
    paths = frame["file_path"]
    empty = paths == ""
    if empty.any():
        raise ValueError("Empty CORAA file_path value(s)")

    absolute = paths.map(lambda value: Path(value).is_absolute() or value.startswith("/"))
    if absolute.any():
        raise ValueError("CORAA file_path must be relative, not absolute")

    backslash = paths.str.contains("\\", regex=False)
    if backslash.any():
        raise ValueError("CORAA file_path must not contain backslashes")

    parent_refs = paths.str.contains("..", regex=False)
    if parent_refs.any():
        raise ValueError("CORAA file_path must not contain '..'")

    split_prefix = f"{split}/"
    wrong_split = ~paths.str.startswith(split_prefix)
    if wrong_split.any():
        raise ValueError(
            f"CORAA file_path must start with split prefix {split_prefix!r}"
        )


def _validate_vote_fields(frame: pd.DataFrame) -> None:
    for column in _VOTE_COLUMNS:
        values = frame[column]
        nonempty = values != ""
        if not nonempty.any():
            continue
        invalid = nonempty & ~values.str.fullmatch(r"\d+")
        if invalid.any():
            invalid_values = sorted(values.loc[invalid].unique())
            joined = ", ".join(invalid_values)
            raise ValueError(f"Invalid CORAA {column} value(s): {joined}")


def _validate_unique_file_paths(frame: pd.DataFrame) -> None:
    duplicates = sorted(
        frame.loc[frame["file_path"].duplicated(keep=False), "file_path"].unique()
    )
    if duplicates:
        raise ValueError(
            f"Duplicate CORAA file_path value(s): {', '.join(duplicates)}"
        )


def _with_traceability(
    frame: pd.DataFrame, *, path: Path, split: str
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["split"] = split
    enriched["metadata_source_file"] = Path(path).name
    enriched["metadata_source_row"] = (enriched.index + 2).astype(str)
    return enriched[[*CORAA_COLUMNS, *_TRACEABILITY_COLUMNS]]
