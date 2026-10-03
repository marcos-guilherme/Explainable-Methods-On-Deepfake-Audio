"""Deterministic JSON profile and summary builders for Mandarin metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..sources.aishell3 import AISHELL3_COLUMNS, AishellReconciliation
from ..sources.jmds import JMDS_COLUMNS
from .metadata_profiling_common import (
    counts_by_split,
    profile_field,
    value_counts,
)

_JMDS_TRACEABILITY = (
    "split",
    "metadata_source_file",
    "metadata_source_row",
    "audio_path",
)

_AISHELL_CATEGORICAL = frozenset(
    {"split", "gender", "age", "accent", "prosody_available"}
)
_AISHELL_LENGTH = frozenset(
    {
        "utt_id",
        "archive_member_path",
        "speaker_id",
        "transcription",
        "pinyin",
        "metadata_source_file",
        "metadata_source_row",
    }
)

_JMDS_CATEGORICAL = frozenset(
    {
        "gender",
        "codec",
        "attack_id",
        "label",
        "native",
        "language",
        "dataset",
        "split",
    }
)
_JMDS_LENGTH = frozenset(
    {
        "spk_id",
        "utt_id",
        "audio_path",
        "metadata_source_file",
        "metadata_source_row",
    }
)

_AISHELL_DESCRIPTIONS = {
    "utt_id": "AISHELL-3 utterance identifier",
    "split": "Derived split label from archive layout",
    "archive_member_path": "Internal archive path to the WAV member",
    "speaker_id": "Speaker identifier derived from the utterance id prefix",
    "gender": "Speaker gender from spk-info.txt",
    "age": "Speaker age group from spk-info.txt",
    "accent": "Speaker accent region from spk-info.txt",
    "transcription": "Utterance transcription from content.txt",
    "pinyin": "Prosody label text when available",
    "prosody_available": "Whether a train prosody label exists for the utterance",
    "metadata_source_file": "Source archive member filename for traceability",
    "metadata_source_row": "1-based source content row number as a string",
}

_JMDS_DESCRIPTIONS = {
    "spk_id": "Speaker identifier from the JMDS protocol",
    "utt_id": "Utterance identifier from the JMDS protocol",
    "gender": "Speaker gender code",
    "codec": "Codec label from the protocol",
    "attack_id": "Attack identifier for generated samples",
    "label": "Sample label in the JMDS protocol",
    "native": "Native-speaker indicator",
    "language": "Language code in the JMDS protocol",
    "dataset": "Dataset identifier in the JMDS protocol",
    "split": "Derived split label from the protocol file",
    "metadata_source_file": "Source protocol filename for traceability",
    "metadata_source_row": "1-based source protocol row number as a string",
    "audio_path": "Resolved generated WAV path when available",
}

_AISHELL_FIELD_ORIGINS = {
    **{column: "AISHELL-3 archive metadata" for column in AISHELL3_COLUMNS},
    "split": "Derived from archive layout and validation",
    "speaker_id": "Derived from AISHELL-3 utt_id prefix",
    "metadata_source_file": "Derived during metadata extraction",
    "metadata_source_row": "Derived during metadata extraction",
}

_JMDS_FIELD_ORIGINS = {
    **{column: "JMDS protocol CSV" for column in JMDS_COLUMNS},
    "split": "Derived from protocol filename and validation",
    "metadata_source_file": "Derived during metadata extraction",
    "metadata_source_row": "Derived during metadata extraction",
    "audio_path": "Derived by resolving generated WAV paths",
}


def build_aishell3_profile(
    frame: pd.DataFrame,
    *,
    reconciliation: AishellReconciliation,
) -> dict[str, Any]:
    return {
        "source": "AISHELL3",
        "sample_count": len(frame),
        "counts_by_split": counts_by_split(frame),
        "reconciliation": _reconciliation_payload(reconciliation),
        "fields": [
            profile_field(
                frame,
                column,
                description=_AISHELL_DESCRIPTIONS[column],
                origin=_AISHELL_FIELD_ORIGINS[column],
                categorical=_AISHELL_CATEGORICAL,
                length_fields=_AISHELL_LENGTH,
                vote_fields=frozenset(),
            )
            for column in sorted(AISHELL3_COLUMNS)
        ],
    }


def build_jmds_add_profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [*JMDS_COLUMNS, *_JMDS_TRACEABILITY]
    audio_paths = frame["audio_path"] if "audio_path" in frame.columns else pd.Series(dtype=str)
    resolved = audio_paths != ""
    return {
        "source": "JMDS_ADD",
        "sample_count": len(frame),
        "counts_by_split": counts_by_split(frame),
        "audio_availability": {
            "resolved_count": int(resolved.sum()),
            "missing_count": int((~resolved).sum()),
        },
        "fields": [
            profile_field(
                frame,
                column,
                description=_JMDS_DESCRIPTIONS[column],
                origin=_JMDS_FIELD_ORIGINS[column],
                categorical=_JMDS_CATEGORICAL,
                length_fields=_JMDS_LENGTH,
                vote_fields=frozenset(),
            )
            for column in sorted(columns)
        ],
    }


def build_mandarin_summary(
    aishell: pd.DataFrame, jmds: pd.DataFrame
) -> dict[str, Any]:
    return {
        "comparison_limitations": {
            "paired_samples": False,
            "statement": (
                "AISHELL-3 pristine metadata and JMDS/ADD generated metadata describe "
                "distinct corpora; no sample-level pairing is asserted."
            ),
        },
        "comparable_characteristics": {
            "paired_samples": False,
            "statement": (
                "Availability is compared at field level only; values remain "
                "source-specific and do not identify paired samples."
            ),
            "characteristics": {
                "accent": {
                    "AISHELL3": {"available": True, "field": "accent"},
                    "JMDS_ADD": {"available": False, "field": None},
                },
                "age": {
                    "AISHELL3": {"available": True, "field": "age"},
                    "JMDS_ADD": {"available": False, "field": None},
                },
                "attack": {
                    "AISHELL3": {"available": False, "field": None},
                    "JMDS_ADD": {"available": True, "field": "attack_id"},
                },
                "codec": {
                    "AISHELL3": {"available": False, "field": None},
                    "JMDS_ADD": {"available": True, "field": "codec"},
                },
                "dataset": {
                    "AISHELL3": {"available": False, "field": None},
                    "JMDS_ADD": {"available": True, "field": "dataset"},
                },
                "gender": {
                    "AISHELL3": {"available": True, "field": "gender"},
                    "JMDS_ADD": {"available": True, "field": "gender"},
                },
                "native_speaker": {
                    "AISHELL3": {"available": False, "field": None},
                    "JMDS_ADD": {"available": True, "field": "native"},
                },
                "prosody": {
                    "AISHELL3": {"available": True, "field": "prosody_available"},
                    "JMDS_ADD": {"available": False, "field": None},
                },
                "speaker": {
                    "AISHELL3": {"available": True, "field": "speaker_id"},
                    "JMDS_ADD": {"available": True, "field": "spk_id"},
                },
                "split": {
                    "AISHELL3": {"available": True, "field": "split"},
                    "JMDS_ADD": {"available": True, "field": "split"},
                },
                "transcription": {
                    "AISHELL3": {"available": True, "field": "transcription"},
                    "JMDS_ADD": {"available": False, "field": None},
                },
            },
        },
        "sources": {
            "AISHELL3": _source_summary(
                aishell,
                label_role="pristine",
                distributions={
                    "accent": "accent",
                    "age": "age",
                    "gender": "gender",
                    "prosody_available": "prosody_available",
                },
            ),
            "JMDS_ADD": _source_summary(
                jmds,
                label_role="generated",
                distributions={
                    "attack_id": "attack_id",
                    "codec": "codec",
                    "dataset": "dataset",
                    "gender": "gender",
                    "native": "native",
                },
            ),
        },
    }


def _reconciliation_payload(
    reconciliation: AishellReconciliation,
) -> dict[str, Any]:
    return {
        "by_split": {
            split: dict(report)
            for split, report in sorted(reconciliation.by_split.items())
        },
        "official_test_sample_count": reconciliation.official_test_sample_count,
        "observed_test_content_line_count": (
            reconciliation.observed_test_content_line_count
        ),
        "divergence_explanation": reconciliation.divergence_explanation,
    }


def _source_summary(
    frame: pd.DataFrame,
    *,
    label_role: str,
    distributions: dict[str, str],
) -> dict[str, Any]:
    return {
        "counts_by_split": counts_by_split(frame),
        "distributions": {
            name: value_counts(frame[column])
            for name, column in sorted(distributions.items())
        },
        "label_role": label_role,
        "sample_count": len(frame),
    }
