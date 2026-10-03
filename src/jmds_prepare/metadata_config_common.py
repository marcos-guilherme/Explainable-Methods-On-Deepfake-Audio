"""Shared validation helpers for two-source metadata YAML configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class JmdsProtocolProfile(Protocol):
    protocol_splits: tuple[str, ...]

    def jmds_protocol_path(self, jmds_root: Path, split: str) -> Path: ...


def scalar_path(value: Any, field_name: str) -> Path:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    return Path(value)


def reject_nested_output(
    *,
    output_root: Path,
    forbidden_roots: Sequence[tuple[Path, str]],
) -> None:
    for root, label in forbidden_roots:
        if output_root == root or output_root.is_relative_to(root):
            raise ValueError(f"output_root must not be nested inside {label}")


def validate_jmds_protocols(jmds_root: Path, profile: JmdsProtocolProfile) -> None:
    for split in profile.protocol_splits:
        protocol_path = profile.jmds_protocol_path(jmds_root, split)
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"Required JMDS protocol is missing: {protocol_path}"
            )
