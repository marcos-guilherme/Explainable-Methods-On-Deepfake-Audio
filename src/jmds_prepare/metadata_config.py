"""Standalone YAML configuration for Portuguese metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .metadata_config_common import (
    reject_nested_output,
    scalar_path,
    validate_jmds_protocols,
)
from .profiles.portuguese import PortugueseProfile

_REQUIRED_KEYS = frozenset({"jmds_root", "coraa_root", "output_root"})


@dataclass(frozen=True)
class PortugueseMetadataConfig:
    jmds_root: Path
    coraa_root: Path
    output_root: Path

    @classmethod
    def load(cls, path: Path) -> PortugueseMetadataConfig:
        with Path(path).open(encoding="utf-8") as config_file:
            try:
                values: Any = yaml.safe_load(config_file)
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid YAML configuration: {error}") from error

        if not isinstance(values, dict):
            raise ValueError("Configuration must be a YAML mapping")

        unknown = set(values) - _REQUIRED_KEYS
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configuration key(s): {joined}")

        missing = _REQUIRED_KEYS - set(values)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Missing required configuration key(s): {joined}")

        return cls(
            jmds_root=scalar_path(values["jmds_root"], "jmds_root"),
            coraa_root=scalar_path(values["coraa_root"], "coraa_root"),
            output_root=scalar_path(values["output_root"], "output_root"),
        )

    def validate(self, profile: PortugueseProfile) -> None:
        jmds_root = self.jmds_root.resolve()
        coraa_root = self.coraa_root.resolve()
        output_root = self.output_root.resolve()

        if output_root == jmds_root:
            raise ValueError("output_root must not equal jmds_root")
        if output_root == coraa_root:
            raise ValueError("output_root must not equal coraa_root")
        reject_nested_output(
            output_root=output_root,
            forbidden_roots=[
                (jmds_root, "jmds_root"),
                (coraa_root, "coraa_root"),
            ],
        )

        if not jmds_root.is_dir():
            raise FileNotFoundError(f"JMDS root does not exist: {self.jmds_root}")

        validate_jmds_protocols(jmds_root, profile)

        if not coraa_root.is_dir():
            raise FileNotFoundError(f"CORAA root does not exist: {self.coraa_root}")

        for split in profile.coraa_splits:
            metadata_path = profile.coraa_metadata_path(self.coraa_root, split)
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Required CORAA metadata is missing: {metadata_path}"
                )
