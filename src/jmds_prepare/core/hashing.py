"""Streaming file digests shared by every preparation stage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Callable

DEFAULT_CHUNK_SIZE = 1024 * 1024


def file_digest(
    path: Path | str, algorithm: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """Return the hex digest of the exact bytes of ``path``."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the SHA-256 hex digest of the exact bytes of ``path``."""
    return file_digest(path, "sha256", chunk_size=chunk_size)


def md5_file(path: Path | str, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the MD5 hex digest of the exact bytes of ``path``."""
    return file_digest(path, "md5", chunk_size=chunk_size)


def sha256_file_preserving_times(
    path: Path, *, hasher: Callable[[Path], str] = sha256_file
) -> str:
    """Hash ``path`` with ``hasher`` and restore its access/modification times.

    Reading a file may bump its access time; source audio must look untouched
    after verification, so both timestamps are restored even if hashing fails.
    """
    source_stat = path.stat()
    try:
        return hasher(path)
    finally:
        os.utime(
            path,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )


def copy_with_sha256(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Stream ``source`` into ``destination`` and return the copied SHA-256."""
    digest = hashlib.sha256()
    while chunk := source.read(chunk_size):
        destination.write(chunk)
        digest.update(chunk)
    return digest.hexdigest()
