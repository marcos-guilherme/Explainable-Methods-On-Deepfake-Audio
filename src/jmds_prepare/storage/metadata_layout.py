"""Output paths for the Portuguese metadata extraction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jmds_prepare.profiles.portuguese import PORTUGUESE_ARTIFACT_NAMES
from jmds_prepare.storage.two_source_metadata_layout import TwoSourceMetadataLayout


@dataclass(frozen=True)
class PortugueseMetadataLayout(TwoSourceMetadataLayout):
    """Immutable mapping from ``output_root`` to the six published artifacts."""

    def __init__(self, output_root: Path) -> None:
        super().__init__(Path(output_root), PORTUGUESE_ARTIFACT_NAMES)

    @property
    def coraa_metadata_csv(self) -> Path:
        return self.pristine_metadata_csv

    @property
    def jmds_mlaad_generated_metadata_csv(self) -> Path:
        return self.generated_metadata_csv

    @property
    def coraa_metadata_profile_json(self) -> Path:
        return self.pristine_metadata_profile_json

    @property
    def jmds_mlaad_generated_metadata_profile_json(self) -> Path:
        return self.generated_metadata_profile_json

    @property
    def portuguese_metadata_summary_json(self) -> Path:
        return self.summary_json

    @property
    def portuguese_metadata_provenance_json(self) -> Path:
        return self.provenance_json
