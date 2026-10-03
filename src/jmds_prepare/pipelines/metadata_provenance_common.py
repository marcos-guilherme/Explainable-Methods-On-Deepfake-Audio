"""Shared provenance helpers for two-source metadata pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def build_input_evidence(
    input_paths: Mapping[str, Path],
    sha256_file: Callable[[Path], str],
) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for key, path in sorted(input_paths.items())
    ]
