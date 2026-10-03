from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .profiles.english import ENGLISH_PROFILE


_PROTOCOL_FILES = tuple(
    ENGLISH_PROFILE.jmds_protocol_filename(split)
    for split in ENGLISH_PROFILE.protocol_splits
)
_SUPPORTED_SPLITS = frozenset(ENGLISH_PROFILE.splits)


@dataclass(frozen=True)
class PreparationConfig:
    jmds_root: Path
    data_root: Path
    zenodo_record_id: int = ENGLISH_PROFILE.zenodo_record_id
    supported_splits: tuple[str, ...] = ENGLISH_PROFILE.splits
    sample_rate: int = 16_000
    asvspoof_protocol_root: Path | None = None

    @classmethod
    def load(cls, path: Path) -> PreparationConfig:
        with path.open(encoding="utf-8") as config_file:
            values: Any = yaml.safe_load(config_file)

        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")

        values = dict(values)
        if "jmds_root" in values:
            values["jmds_root"] = Path(values["jmds_root"])
        if "data_root" in values:
            values["data_root"] = Path(values["data_root"])
        if values.get("asvspoof_protocol_root") is not None:
            values["asvspoof_protocol_root"] = Path(
                values["asvspoof_protocol_root"]
            )
        if "supported_splits" in values:
            values["supported_splits"] = tuple(values["supported_splits"])
        return cls(**values)

    def validate(self) -> None:
        if self.supported_splits != ENGLISH_PROFILE.splits:
            raise ValueError(
                "supported_splits must be exactly "
                f"{', '.join(ENGLISH_PROFILE.splits)} for this phase; "
                f"received: {', '.join(self.supported_splits)}"
            )
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if not self.jmds_root.is_dir():
            raise FileNotFoundError(f"JMDS root does not exist: {self.jmds_root}")

        protocols_root = self.jmds_root / "cm_protocols"
        for filename in _PROTOCOL_FILES:
            protocol_path = protocols_root / filename
            if not protocol_path.is_file():
                raise FileNotFoundError(
                    f"Required JMDS protocol is missing: {protocol_path}"
                )

        generated_root = ENGLISH_PROFILE.generated_root(self.jmds_root)
        if not generated_root.is_dir():
            raise FileNotFoundError(
                f"Required generated English directory is missing: {generated_root}"
            )
