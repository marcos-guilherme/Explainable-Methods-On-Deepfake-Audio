"""Output paths for the Mandarin metadata extraction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jmds_prepare.profiles.mandarin import MANDARIN_ARTIFACT_NAMES
from jmds_prepare.storage.two_source_metadata_layout import TwoSourceMetadataLayout


@dataclass(frozen=True)
class MandarinMetadataLayout(TwoSourceMetadataLayout):
    """Immutable mapping from ``output_root`` to the six published artifacts."""

    def __init__(self, output_root: Path) -> None:
        super().__init__(Path(output_root), MANDARIN_ARTIFACT_NAMES)

    @property
    def aishell3_metadata_csv(self) -> Path:
        return self.pristine_metadata_csv

    @property
    def jmds_add_generated_metadata_csv(self) -> Path:
        return self.generated_metadata_csv

    @property
    def aishell3_metadata_profile_json(self) -> Path:
        return self.pristine_metadata_profile_json

    @property
    def jmds_add_generated_metadata_profile_json(self) -> Path:
        return self.generated_metadata_profile_json

    @property
    def mandarin_metadata_summary_json(self) -> Path:
        return self.summary_json

    @property
    def mandarin_metadata_provenance_json(self) -> Path:
        return self.provenance_json
