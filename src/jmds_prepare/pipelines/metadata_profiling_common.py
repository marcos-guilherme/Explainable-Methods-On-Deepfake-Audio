"""Shared deterministic profiling helpers for two-source metadata pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_field(
    frame: pd.DataFrame,
    column: str,
    *,
    description: str,
    origin: str,
    categorical: frozenset[str],
    length_fields: frozenset[str],
    vote_fields: frozenset[str],
) -> dict[str, Any]:
    series = frame[column]
    missing_count = int((series == "").sum())
    payload: dict[str, Any] = {
        "description": description,
        "distinct_count": int(series.nunique(dropna=False)),
        "missing_count": missing_count,
        "name": column,
        "nullable": missing_count > 0,
        "observed_type": "string",
        "origin": origin,
    }
    if column in length_fields:
        payload["length_statistics"] = length_statistics(series)
    elif column in vote_fields:
        payload["numeric_statistics"] = numeric_statistics(series)
    elif column in categorical:
        payload["value_counts"] = value_counts(series)
    return payload


def counts_by_split(frame: pd.DataFrame) -> dict[str, int]:
    if "split" not in frame.columns or frame.empty:
        return {}
    counts = frame["split"].value_counts().to_dict()
    return {split: int(count) for split, count in sorted(counts.items())}


def length_statistics(series: pd.Series) -> dict[str, float | int | None]:
    lengths = series.map(len)
    if lengths.empty:
        return {
            "count": 0,
            "max_length": None,
            "mean_length": None,
            "min_length": None,
        }
    return {
        "count": int(len(lengths)),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "min_length": int(lengths.min()),
    }


def numeric_statistics(series: pd.Series) -> dict[str, float | int | None]:
    values = series.loc[series != ""].astype(int)
    if values.empty:
        return {
            "max": None,
            "mean": None,
            "min": None,
            "non_missing_count": 0,
        }
    return {
        "max": int(values.max()),
        "mean": float(values.mean()),
        "min": int(values.min()),
        "non_missing_count": int(len(values)),
    }


def value_counts(series: pd.Series) -> list[dict[str, Any]]:
    counts = series.value_counts()
    rows = [{"count": int(counts[value]), "value": value} for value in counts.index]
    return sorted(rows, key=lambda row: row["value"])
