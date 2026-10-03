"""Input evidence and field-origin provenance for Mandarin metadata artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..profiles.mandarin import MandarinProfile
from ..sources.aishell3 import AISHELL3_COLUMNS, AishellReconciliation
from ..sources.jmds import JMDS_COLUMNS
from .mandarin_metadata_profiling import _reconciliation_payload
from .metadata_provenance_common import build_input_evidence

_EXCLUDED_JMDS_PRISTINE_STATEMENT = (
    "Mandarin JMDS pristine rows referencing anonymized AISHELL-3 IDs were "
    "excluded; no public sample-level mapping exists between JMDS pristine "
    "IDs and AISHELL-3 archive utterance names."
)

_AISHELL_FIELD_ORIGINS = {
    "accent": "AISHELL-3 archive metadata",
    "age": "AISHELL-3 archive metadata",
    "archive_member_path": "AISHELL-3 archive metadata",
    "gender": "AISHELL-3 archive metadata",
    "metadata_source_file": "derived during metadata extraction",
    "metadata_source_row": "derived during metadata extraction",
    "pinyin": "AISHELL-3 archive metadata",
    "prosody_available": "derived during metadata extraction",
    "speaker_id": "derived from AISHELL-3 utt_id prefix",
    "split": "derived from archive layout and validation",
    "transcription": "AISHELL-3 archive metadata",
    "utt_id": "AISHELL-3 archive metadata",
}

_JMDS_FIELD_ORIGINS = {
    "attack_id": "JMDS protocol CSV",
    "audio_path": "derived by resolving generated WAV paths",
    "codec": "JMDS protocol CSV",
    "dataset": "JMDS protocol CSV",
    "gender": "JMDS protocol CSV",
    "label": "JMDS protocol CSV",
    "language": "JMDS protocol CSV",
    "metadata_source_file": "derived during metadata extraction",
    "metadata_source_row": "derived during metadata extraction",
    "native": "JMDS protocol CSV",
    "spk_id": "JMDS protocol CSV",
    "split": "derived from protocol filename and validation",
    "utt_id": "JMDS protocol CSV",
}


def build_mandarin_metadata_provenance(
    *,
    profile: MandarinProfile,
    input_paths: Mapping[str, Path],
    sha256_file: Callable[[Path], str],
    reconciliation: AishellReconciliation,
) -> dict[str, Any]:
    return {
        "artifact_schemas": {
            "aishell3_metadata_csv": {
                "columns": list(AISHELL3_COLUMNS),
                "field_origins": dict(sorted(_AISHELL_FIELD_ORIGINS.items())),
            },
            "aishell3_metadata_profile": {
                "fields": sorted(AISHELL3_COLUMNS),
            },
            "jmds_add_generated_metadata_csv": {
                "columns": [
                    *JMDS_COLUMNS,
                    "split",
                    "metadata_source_file",
                    "metadata_source_row",
                    "audio_path",
                ],
                "field_origins": dict(sorted(_JMDS_FIELD_ORIGINS.items())),
            },
            "jmds_add_generated_metadata_profile": {
                "fields": sorted(
                    {
                        *JMDS_COLUMNS,
                        "split",
                        "metadata_source_file",
                        "metadata_source_row",
                        "audio_path",
                    }
                ),
            },
            "mandarin_metadata_provenance": {
                "sections": sorted(
                    [
                        "aishell3",
                        "artifact_schemas",
                        "excluded_jmds_pristine_count",
                        "excluded_jmds_pristine_statement",
                        "field_origins",
                        "inputs",
                        "jmds_add",
                    ]
                ),
            },
            "mandarin_metadata_summary": {
                "sections": sorted(
                    [
                        "comparable_characteristics",
                        "comparison_limitations",
                        "sources",
                    ]
                ),
            },
        },
        "aishell3": {
            "license": "Apache-2.0",
            "reconciliation": _reconciliation_payload(reconciliation),
            "usage_policy": {
                "preserve_originals": True,
                "statement": (
                    "AISHELL-3 metadata is read from the local archive without "
                    "extracting or modifying WAV members."
                ),
            },
        },
        "excluded_jmds_pristine_count": profile.excluded_jmds_pristine_count,
        "excluded_jmds_pristine_statement": _EXCLUDED_JMDS_PRISTINE_STATEMENT,
        "field_origins": {
            "aishell3_metadata_csv": dict(sorted(_AISHELL_FIELD_ORIGINS.items())),
            "jmds_add_generated_metadata_csv": dict(
                sorted(_JMDS_FIELD_ORIGINS.items())
            ),
        },
        "inputs": build_input_evidence(input_paths, sha256_file),
        "jmds_add": {
            "generated_dataset": profile.generated_dataset,
            "generated_dataset_version": None,
            "jmds_release": "v2",
            "license": None,
            "missing_reasons": {
                "attack_id": (
                    "All ADD rows report attack_id=unk; no attack taxonomy is encoded."
                ),
                "codec": (
                    "All ADD rows report codec=-; no codec label is provided."
                ),
                "gender": (
                    "All ADD rows report gender=unk; speaker gender is not provided."
                ),
                "generated_dataset_version": (
                    "No separate ADD version is stated in the JMDS v2 protocol."
                ),
                "license": (
                    "No ADD license is stated in the local JMDS v2 metadata."
                ),
                "spk_id": (
                    "All ADD rows report spk_id=unk; speaker identity is not provided."
                ),
            },
        },
    }
