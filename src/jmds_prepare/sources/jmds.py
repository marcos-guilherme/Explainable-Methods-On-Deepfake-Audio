"""JMDS countermeasure protocol schema, reader and utterance-ID rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiles.english import ENGLISH_PROFILE
from .jmds_common import (
    validate_schema,
    validate_split,
    validate_unique_ids,
    validate_utterance_ids,
)


JMDS_COLUMNS = [
    "spk_id",
    "utt_id",
    "gender",
    "codec",
    "attack_id",
    "label",
    "native",
    "language",
    "dataset",
]

_SUPPORTED_SPLITS = frozenset(ENGLISH_PROFILE.protocol_splits)
_JMDS_LABELS = frozenset(ENGLISH_PROFILE.label_equivalence)
_SPLIT_PREFIX = ENGLISH_PROFILE.split_id_prefixes


def read_jmds_protocol(
    path: Path,
    split: str,
    *,
    language: str | None = None,
) -> pd.DataFrame:
    validate_split(split, _SUPPORTED_SPLITS)
    protocol = pd.read_csv(path, dtype=str, keep_default_na=False)
    actual_columns = list(protocol.columns)
    validate_schema(actual_columns, JMDS_COLUMNS, source="JMDS")

    if language is not None:
        protocol = protocol.loc[protocol["language"] == language].copy()
    validate_unique_ids(protocol, source="JMDS")
    validate_utterance_ids(
        protocol, split=split, prefixes=_SPLIT_PREFIX, source="JMDS"
    )
    unsupported = sorted(set(protocol["label"]) - _JMDS_LABELS)
    if unsupported:
        raise ValueError(f"Unsupported JMDS label(s): {', '.join(unsupported)}")
    return protocol


def _validate_split(split: str) -> None:
    validate_split(split, _SUPPORTED_SPLITS)


def _validate_unique_ids(protocol: pd.DataFrame, source: str) -> None:
    validate_unique_ids(protocol, source=source)


def _validate_utterance_ids(
    protocol: pd.DataFrame, split: str, source: str
) -> None:
    validate_utterance_ids(
        protocol, split=split, prefixes=_SPLIT_PREFIX, source=source
    )
