"""Filesystem path helpers shared by every layer."""

from __future__ import annotations

from pathlib import Path


def existing_ancestor(path: Path) -> Path:
    """Return the nearest existing ancestor of ``path`` (itself if it exists)."""
    current = path.resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing ancestor for data root: {path}")
        current = parent
    return current
