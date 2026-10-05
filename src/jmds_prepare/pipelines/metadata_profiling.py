"""Deterministic JSON profile and summary builders for Portuguese metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..sources.coraa import CORAA_COLUMNS
from ..sources.jmds import JMDS_COLUMNS
from .metadata_profiling_common import (
    counts_by_split,
    length_statistics,
    numeric_statistics,
    profile_field,
    value_counts,
)

_CORAA_TRACEABILITY = ("split", "metadata_source_file", "metadata_source_row")
_JMDS_TRACEABILITY = (
    "split",
    "metadata_source_file",
    "metadata_source_row",
    "audio_path",
)

_CORAA_CATEGORICAL = frozenset(
    {
        "task",
        "variety",
        "dataset",
        "accent",
        "speech_genre",
        "speech_style",
        "split",
    }
)
_CORAA_LENGTH = frozenset(
    {"file_path", "text", "metadata_source_file", "metadata_source_row"}
)
_CORAA_VOTE = frozenset(
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

_CORAA_DESCRIPTIONS = {
    "file_path": "Relative CORAA audio path within the split tree",
    "task": "Annotation task type assigned to the sample",
    "variety": "Portuguese variety label",
    "dataset": "Source dataset identifier",
    "accent": "Regional accent label",
    "speech_genre": "Speech genre category",
    "speech_style": "Speech style category",
    "up_votes": "Crowd up-vote count for quality review",
    "down_votes": "Crowd down-vote count for quality review",
    "votes_for_hesitation": "Votes flagging hesitation",
    "votes_for_filled_pause": "Votes flagging filled pauses",
    "votes_for_noise_or_low_voice": "Votes flagging noise or low voice",
    "votes_for_second_voice": "Votes flagging a second voice",
    "votes_for_no_identified_problem": "Votes reporting no identified problem",
    "text": "Transcription or annotation text",
    "split": "Derived split label from the source file",
    "metadata_source_file": "Source CSV filename for traceability",
    "metadata_source_row": "1-based source CSV row number as a string",
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

_CORAA_FIELD_ORIGINS = {
    **{column: "CORAA metadata CSV" for column in CORAA_COLUMNS},
    "split": "Derived from source filename and validation",
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


def build_coraa_profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [*CORAA_COLUMNS, *_CORAA_TRACEABILITY]
    return {
        "source": "CORAA",
        "sample_count": int(len(frame)),
        "counts_by_split": counts_by_split(frame),
        "fields": [
            profile_field(
                frame,
                column,
                description=_CORAA_DESCRIPTIONS[column],
                origin=_CORAA_FIELD_ORIGINS[column],
                categorical=_CORAA_CATEGORICAL,
                length_fields=_CORAA_LENGTH,
                vote_fields=_CORAA_VOTE,
            )
            for column in sorted(columns)
        ],
    }


def build_jmds_mlaad_profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [*JMDS_COLUMNS, *_JMDS_TRACEABILITY]
    audio_paths = frame["audio_path"] if "audio_path" in frame.columns else pd.Series(dtype=str)
    resolved = audio_paths != ""
    return {
        "source": "JMDS_MLAAD",
        "sample_count": int(len(frame)),
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


def build_portuguese_summary(
    coraa: pd.DataFrame, jmds: pd.DataFrame
) -> dict[str, Any]:
    return {
        "comparison_limitations": {
            "paired_samples": False,
            "statement": (
                "CORAA pristine metadata and JMDS/MLAAD generated metadata describe "
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
                    "CORAA": {"available": True, "field": "accent"},
                    "JMDS_MLAAD": {"available": False, "field": None},
                },
                "attack": {
                    "CORAA": {"available": False, "field": None},
                    "JMDS_MLAAD": {"available": True, "field": "attack_id"},
                },
                "codec": {
                    "CORAA": {"available": False, "field": None},
                    "JMDS_MLAAD": {"available": True, "field": "codec"},
                },
                "dataset": {
                    "CORAA": {"available": True, "field": "dataset"},
                    "JMDS_MLAAD": {"available": True, "field": "dataset"},
                },
                "gender": {
                    "CORAA": {"available": False, "field": None},
                    "JMDS_MLAAD": {"available": True, "field": "gender"},
                },
                "native_speaker": {
                    "CORAA": {"available": False, "field": None},
                    "JMDS_MLAAD": {"available": True, "field": "native"},
                },
                "quality_votes": {
                    "CORAA": {"available": True, "field": "vote fields"},
                    "JMDS_MLAAD": {"available": False, "field": None},
                },
                "speech_genre": {
                    "CORAA": {"available": True, "field": "speech_genre"},
                    "JMDS_MLAAD": {"available": False, "field": None},
                },
                "speech_style": {
                    "CORAA": {"available": True, "field": "speech_style"},
                    "JMDS_MLAAD": {"available": False, "field": None},
                },
                "split": {
                    "CORAA": {"available": True, "field": "split"},
                    "JMDS_MLAAD": {"available": True, "field": "split"},
                },
                "task": {
                    "CORAA": {"available": True, "field": "task"},
                    "JMDS_MLAAD": {"available": False, "field": None},
                },
            },
        },
        "sources": {
            "CORAA": _source_summary(
                coraa,
                label_role="pristine",
                distributions={
                    "accent": "accent",
                    "dataset": "dataset",
                    "speech_genre": "speech_genre",
                    "speech_style": "speech_style",
                    "subcorpus": "dataset",
                    "task": "task",
                    "variety": "variety",
                },
            ),
            "JMDS_MLAAD": _source_summary(
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


def _source_summary(
    frame: pd.DataFrame,
    *,
    label_role: str,
    distributions: dict[str, str],
) -> dict[str, Any]:
    summary = {
        "counts_by_split": counts_by_split(frame),
        "distributions": {
            name: value_counts(frame[column])
            for name, column in sorted(distributions.items())
        },
        "label_role": label_role,
        "sample_count": int(len(frame)),
    }
    if label_role == "pristine":
        summary["quality_indicators"] = {
            field: {
                "missing_count": int((frame[field] == "").sum()),
                "non_missing_count": int((frame[field] != "").sum()),
                "numeric_statistics": numeric_statistics(frame[field]),
            }
            for field in sorted(_CORAA_VOTE)
        }
    return summary
