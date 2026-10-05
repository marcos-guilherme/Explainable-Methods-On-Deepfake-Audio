"""Corpus-agnostic primitives for two-source metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoSourceArtifactNames:
    """Filenames for the six published two-source metadata artifacts."""

    pristine_manifest: str
    generated_manifest: str
    pristine_profile: str
    generated_profile: str
    summary: str
    provenance: str
