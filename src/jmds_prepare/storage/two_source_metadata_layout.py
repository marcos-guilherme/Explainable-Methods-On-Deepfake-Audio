"""Generic layout for two-source metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jmds_prepare.core.two_source import TwoSourceArtifactNames

__all__ = ["TwoSourceArtifactNames", "TwoSourceMetadataLayout"]


@dataclass(frozen=True)
class TwoSourceMetadataLayout:
    """Immutable mapping from ``output_root`` to the six published artifacts."""

    output_root: Path
    names: TwoSourceArtifactNames

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def manifests_dir(self) -> Path:
        return self.output_root / "manifests"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def pristine_metadata_csv(self) -> Path:
        return self.manifests_dir / self.names.pristine_manifest

    @property
    def generated_metadata_csv(self) -> Path:
        return self.manifests_dir / self.names.generated_manifest

    @property
    def pristine_metadata_profile_json(self) -> Path:
        return self.reports_dir / self.names.pristine_profile

    @property
    def generated_metadata_profile_json(self) -> Path:
        return self.reports_dir / self.names.generated_profile

    @property
    def summary_json(self) -> Path:
        return self.reports_dir / self.names.summary

    @property
    def provenance_json(self) -> Path:
        return self.reports_dir / self.names.provenance
