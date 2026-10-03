"""Content-idempotent publication of artifact sets."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

_COMPARE_BLOCK_SIZE = 1024 * 1024


class DestinationConflictError(FileExistsError):
    """Raised when a destination exists with different content."""


def publish_artifact_set_idempotent(artifacts: Mapping[Path, bytes]) -> None:
    """Publish bytes to destinations without overwriting divergent content."""
    if not artifacts:
        return

    destinations = _resolve_unique_destinations(artifacts)
    _preflight_destinations(destinations)

    missing = {
        destination: content
        for destination, content in destinations.items()
        if not destination.exists()
    }
    if not missing:
        return

    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for destination, content in missing.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _temporary_path(destination.parent, destination.name)
            with temporary.open("wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporaries[destination] = temporary

        for destination, temporary in temporaries.items():
            os.link(temporary, destination)
            published.append((temporary, destination))
    except BaseException:
        for temporary, destination in published:
            if _same_file(temporary, destination):
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _resolve_unique_destinations(
    artifacts: Mapping[Path, bytes],
) -> dict[Path, bytes]:
    destinations: dict[Path, bytes] = {}
    for destination, content in artifacts.items():
        resolved = destination.resolve()
        if resolved in destinations:
            if destinations[resolved] != content:
                raise ValueError(
                    "Duplicate resolved destination with conflicting content: "
                    f"{resolved}"
                )
            raise ValueError(
                f"Duplicate resolved destination: {resolved}"
            )
        destinations[resolved] = content
    return destinations


def _preflight_destinations(destinations: Mapping[Path, bytes]) -> None:
    for destination, content in destinations.items():
        if not destination.exists():
            continue
        if not _file_matches_bytes(destination, content):
            raise DestinationConflictError(
                f"Destination exists with different content: {destination}"
            )


def _file_matches_bytes(path: Path, expected: bytes) -> bool:
    if path.stat().st_size != len(expected):
        return False
    view = memoryview(expected)
    offset = 0
    with path.open("rb") as existing:
        while offset < len(expected):
            block = existing.read(_COMPARE_BLOCK_SIZE)
            if not block:
                return False
            if block != view[offset : offset + len(block)]:
                return False
            offset += len(block)
    return offset == len(expected)


def _temporary_path(directory: Path, artifact_name: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{artifact_name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False
