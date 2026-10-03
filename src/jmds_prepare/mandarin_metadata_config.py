"""Standalone YAML configuration for Mandarin metadata extraction."""

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
from .profiles.mandarin import MandarinProfile

_REQUIRED_KEYS = frozenset({"jmds_root", "aishell_archive", "output_root"})


@dataclass(frozen=True)
class MandarinMetadataConfig:
    jmds_root: Path
    aishell_archive: Path
    output_root: Path

    @classmethod
    def load(cls, path: Path) -> MandarinMetadataConfig:
        with Path(path).open(encoding="utf-8") as config_file:
            try:
                values: Any = yaml.safe_load(config_file)
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid YAML configuration: {error}") from error

        if not isinstance(values, dict):
            raise TypeError("Configuration must be a YAML mapping")

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
            aishell_archive=scalar_path(values["aishell_archive"], "aishell_archive"),
            output_root=scalar_path(values["output_root"], "output_root"),
        )

    def validate(self, profile: MandarinProfile) -> None:
        jmds_root = self.jmds_root.resolve()
        aishell_archive = self.aishell_archive.resolve()
        output_root = self.output_root.resolve()
        archive_parent = aishell_archive.parent.resolve()

        if output_root == jmds_root:
            raise ValueError("output_root must not equal jmds_root")
        if output_root == aishell_archive:
            raise ValueError("output_root must not equal aishell_archive")
        reject_nested_output(
            output_root=output_root,
            forbidden_roots=[
                (jmds_root, "jmds_root"),
                (archive_parent, "aishell_archive parent"),
            ],
        )

        if not jmds_root.is_dir():
            raise FileNotFoundError(f"JMDS root does not exist: {self.jmds_root}")

        validate_jmds_protocols(jmds_root, profile)

        if not aishell_archive.is_file():
            raise FileNotFoundError(
                f"aishell_archive is missing or not a file: {self.aishell_archive}"
            )
