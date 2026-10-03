"""Profile-parameterized JMDS schema, split, ID and duplicate validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd


def validate_schema(
    actual: Sequence[str], expected: Sequence[str], *, source: str
) -> None:
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"extra columns: {', '.join(extra)}")
        raise ValueError(f"Invalid {source} schema ({'; '.join(details)})")
    if list(actual) != list(expected):
        raise ValueError(f"{source} columns are not in the required order")


def validate_split(split: str, supported: Iterable[str]) -> None:
    if split not in supported:
        raise ValueError(f"Unsupported split: {split}")


def validate_unique_ids(
    frame: pd.DataFrame, *, source: str, id_column: str = "utt_id"
) -> None:
    duplicates = sorted(
        frame.loc[frame[id_column].duplicated(keep=False), id_column].unique()
    )
    if duplicates:
        raise ValueError(
            f"Duplicate {source} {id_column} value(s): {', '.join(duplicates)}"
        )


def validate_utterance_ids(
    frame: pd.DataFrame,
    *,
    split: str,
    prefixes: Mapping[str, str],
    source: str,
) -> None:
    prefix = prefixes[split]
    valid = frame["utt_id"].str.fullmatch(rf"{prefix}_\d{{10}}")
    malformed = sorted(frame.loc[~valid, "utt_id"].unique())
    if malformed:
        raise ValueError(
            f"Malformed {source} utt_id value(s) for {split}: "
            f"{', '.join(malformed)}"
        )
