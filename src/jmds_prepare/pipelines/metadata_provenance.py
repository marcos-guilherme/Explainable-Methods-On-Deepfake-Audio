"""Input evidence and field-origin provenance for Portuguese metadata artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..profiles.portuguese import PortugueseProfile
from ..sources.coraa import CORAA_COLUMNS
from ..sources.jmds import JMDS_COLUMNS
from .metadata_provenance_common import build_input_evidence

_EXCLUDED_JMDS_PRISTINE_STATEMENT = (
    "Portuguese JMDS pristine rows referencing anonymized CORAA IDs were "
    "excluded; no public sample-level mapping exists between JMDS pristine "
    "IDs and CORAA file paths."
)

_CORAA_FIELD_ORIGINS = {
    "accent": "CORAA metadata CSV",
    "dataset": "CORAA metadata CSV",
    "down_votes": "CORAA metadata CSV",
    "file_path": "CORAA metadata CSV",
    "metadata_source_file": "derived during metadata extraction",
    "metadata_source_row": "derived during metadata extraction",
    "speech_genre": "CORAA metadata CSV",
    "speech_style": "CORAA metadata CSV",
    "split": "derived from source filename and validation",
    "task": "CORAA metadata CSV",
    "text": "CORAA metadata CSV",
    "up_votes": "CORAA metadata CSV",
    "variety": "CORAA metadata CSV",
    "votes_for_filled_pause": "CORAA metadata CSV",
    "votes_for_hesitation": "CORAA metadata CSV",
    "votes_for_no_identified_problem": "CORAA metadata CSV",
    "votes_for_noise_or_low_voice": "CORAA metadata CSV",
    "votes_for_second_voice": "CORAA metadata CSV",
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


def build_metadata_provenance(
    *,
    profile: PortugueseProfile,
    input_paths: Mapping[str, Path],
    sha256_file: Callable[[Path], str],
) -> dict[str, Any]:
    return {
        "artifact_schemas": {
            "coraa_metadata_csv": {
                "columns": [
                    *CORAA_COLUMNS,
                    "split",
                    "metadata_source_file",
                    "metadata_source_row",
                ],
                "field_origins": dict(sorted(_CORAA_FIELD_ORIGINS.items())),
            },
            "coraa_metadata_profile": {
                "fields": sorted(
                    {
                        *CORAA_COLUMNS,
                        "split",
                        "metadata_source_file",
                        "metadata_source_row",
                    }
                ),
            },
            "jmds_mlaad_generated_metadata_csv": {
                "columns": [
                    *JMDS_COLUMNS,
                    "split",
                    "metadata_source_file",
                    "metadata_source_row",
                    "audio_path",
                ],
                "field_origins": dict(sorted(_JMDS_FIELD_ORIGINS.items())),
            },
            "jmds_mlaad_generated_metadata_profile": {
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
            "portuguese_metadata_provenance": {
                "sections": sorted(
                    [
                        "artifact_schemas",
                        "coraa",
                        "excluded_jmds_pristine_count",
                        "excluded_jmds_pristine_statement",
                        "field_origins",
                        "inputs",
                        "jmds_mlaad",
                    ]
                ),
            },
            "portuguese_metadata_summary": {
                "sections": sorted(
                    [
                        "comparable_characteristics",
                        "comparison_limitations",
                        "sources",
                    ]
                ),
            },
        },
        "coraa": {
            "license": profile.coraa_license,
            "revision": profile.coraa_revision,
            "usage_policy": {
                "adapted_material_sharing_permitted": False,
                "commercial_use_permitted": False,
                "preserve_originals": True,
                "statement": profile.coraa_usage_policy_statement,
            },
            "version": profile.coraa_version,
        },
        "excluded_jmds_pristine_count": profile.excluded_jmds_pristine_count,
        "excluded_jmds_pristine_statement": _EXCLUDED_JMDS_PRISTINE_STATEMENT,
        "field_origins": {
            "coraa_metadata_csv": dict(sorted(_CORAA_FIELD_ORIGINS.items())),
            "jmds_mlaad_generated_metadata_csv": dict(
                sorted(_JMDS_FIELD_ORIGINS.items())
            ),
        },
        "inputs": build_input_evidence(input_paths, sha256_file),
        "jmds_mlaad": {
            "generated_dataset": profile.generated_dataset,
            "generated_dataset_version": None,
            "jmds_release": profile.jmds_release,
            "license": None,
            "missing_reasons": {
                "generated_dataset_version": (
                    "No separate MLAAD version is stated in the JMDS v2 protocol."
                ),
                "license": (
                    "No MLAAD license is stated in the local JMDS v2 metadata."
                ),
            },
        },
    }
