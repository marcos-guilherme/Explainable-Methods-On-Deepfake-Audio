"""Resumable, size- and checksum-verified downloads of remote archive files."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from ..core.hashing import file_digest, md5_file


CHUNK_SIZE = 8 * 1024 * 1024
CHECKSUM_LABELS = {"md5": "MD5", "sha256": "SHA-256"}

ORIGIN_DOWNLOADED = "downloaded"
ORIGIN_RESUMED = "resumed_download"
ORIGIN_ADOPTED_FINAL = "adopted_existing_final"
ORIGIN_PROMOTED_PARTIAL = "promoted_partial"
ORIGINS = (ORIGIN_DOWNLOADED, ORIGIN_RESUMED, ORIGIN_ADOPTED_FINAL, ORIGIN_PROMOTED_PARTIAL)


class RemoteFile(Protocol):
    """Remote file identity required by :func:`fetch_verified`."""

    @property
    def name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def download_url(self) -> str: ...


class VerifiableFile(Protocol):
    """Remote file description required by :func:`download_verified`.

    ``checksum`` is the expected hexadecimal MD5 of the complete file.
    """

    @property
    def name(self) -> str: ...

    @property
    def size(self) -> int: ...

    @property
    def checksum(self) -> str: ...

    @property
    def download_url(self) -> str: ...


@dataclass(frozen=True)
class TransportPolicy:
    """Response requirements beyond status, range and size checks.

    ``require_https`` rejects any non-HTTPS request URL, redirect hop or final
    URL. ``identity_encoding`` requests and requires unencoded bytes, so the
    bytes counted against the expected size are exactly the stored bytes.
    """

    require_https: bool = False
    identity_encoding: bool = False


LEGACY_TRANSPORT = TransportPolicy()
STRICT_TRANSPORT = TransportPolicy(require_https=True, identity_encoding=True)


@dataclass(frozen=True)
class FetchResult:
    path: Path
    origin: str


class ChecksumError(ValueError):
    """Raised when a downloaded file does not match its expected checksum."""


class DownloadSizeError(ValueError):
    """Raised when a downloaded file does not match its expected size."""


class DownloadResponseError(ValueError):
    """Raised when a download response cannot be safely written."""


def download_verified(
    file: VerifiableFile,
    destination: Path,
    *,
    session: Any = None,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
) -> Path:
    return fetch_verified(
        file,
        destination,
        checksum=file.checksum,
        algorithm="md5",
        session=session,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
    ).path


def fetch_verified(
    file: RemoteFile,
    destination: Path,
    *,
    checksum: str | None,
    algorithm: str,
    session: Any = None,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    transport: TransportPolicy = LEGACY_TRANSPORT,
    before_transfer: Callable[[int], None] | None = None,
) -> FetchResult:
    """Download ``file`` into ``destination`` verifying exact size and checksum.

    ``checksum=None`` verifies only the exact size; callers that need a digest
    of such a file must compute it themselves. ``before_transfer`` receives the
    bytes still to be written and may raise to stop before any network access.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if algorithm not in CHECKSUM_LABELS:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")
    destination.mkdir(parents=True, exist_ok=True)
    partial_path = destination / f"{file.name}.partial"
    final_path = destination / file.name
    if final_path.exists():
        if _matches(final_path, file.size, checksum, algorithm):
            partial_path.unlink(missing_ok=True)
            return FetchResult(final_path, ORIGIN_ADOPTED_FINAL)
        final_path.rename(_invalid_archive_path(final_path))

    if partial_path.exists():
        partial_size = partial_path.stat().st_size
        if partial_size > file.size:
            partial_path.rename(_invalid_archive_path(partial_path))
        elif partial_size == file.size:
            if _checksum_matches(partial_path, checksum, algorithm):
                _promote(partial_path, final_path)
                return FetchResult(final_path, ORIGIN_PROMOTED_PARTIAL)
            partial_path.rename(_invalid_archive_path(partial_path))

    if before_transfer is not None:
        existing = partial_path.stat().st_size if partial_path.exists() else 0
        before_transfer(file.size - existing)

    client = session or requests
    mode = "wb"
    for attempt in range(max_attempts):
        try:
            mode = _download_once(file, partial_path, client, transport)
        except requests.exceptions.RequestException:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(retry_delay * (2**attempt))
            continue

        actual_size = partial_path.stat().st_size
        if actual_size == file.size:
            break
        raise DownloadSizeError(
            f"Downloaded size for {file.name} is {actual_size}; "
            f"expected {file.size}"
        )
    else:  # pragma: no cover - the bounded loop always exits above
        raise AssertionError("download retry loop exhausted unexpectedly")

    if checksum is not None:
        actual_checksum = _digest(partial_path, algorithm)
        if actual_checksum.lower() != checksum.lower():
            raise ChecksumError(
                f"{CHECKSUM_LABELS[algorithm]} mismatch for {file.name}: "
                f"got {actual_checksum}, expected {checksum}"
            )

    _promote(partial_path, final_path)
    return FetchResult(final_path, ORIGIN_RESUMED if mode == "ab" else ORIGIN_DOWNLOADED)


def _download_once(
    file: RemoteFile,
    partial_path: Path,
    client: Any,
    transport: TransportPolicy = LEGACY_TRANSPORT,
) -> str:
    existing_size = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    if transport.identity_encoding:
        headers["Accept-Encoding"] = "identity"
    response = client.get(
        file.download_url,
        headers=headers,
        stream=True,
        timeout=30,
    )
    try:
        response.raise_for_status()
        _check_transport(response, file, transport)
        mode = _download_mode(response, file, existing_size)
        written_size = existing_size if mode == "ab" else 0
        with partial_path.open(mode) as output:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                remaining = file.size - written_size
                if len(chunk) > remaining:
                    if remaining:
                        output.write(chunk[:remaining])
                    raise DownloadSizeError(
                        f"Download stream for {file.name} would exceed "
                        f"expected size {file.size}"
                    )
                output.write(chunk)
                written_size += len(chunk)
        return mode
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _check_transport(response: Any, file: RemoteFile, transport: TransportPolicy) -> None:
    if transport.require_https:
        chain = [file.download_url]
        chain += [getattr(hop, "url", None) for hop in getattr(response, "history", ())]
        chain.append(getattr(response, "url", None))
        for url in chain:
            if url is not None and urlparse(url).scheme != "https":
                raise DownloadResponseError(
                    f"Non-HTTPS URL in the request/redirect chain for {file.name}: "
                    f"{url!r}"
                )
    if transport.identity_encoding:
        encoding = response.headers.get("Content-Encoding", "").strip().lower()
        if encoding not in ("", "identity"):
            raise DownloadResponseError(
                f"Unexpected Content-Encoding for {file.name}: {encoding!r}; "
                "expected identity"
            )


def _content_length(response: Any, file: RemoteFile) -> int | None:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    if not str(value).strip().isdigit():
        raise DownloadResponseError(
            f"Invalid Content-Length for {file.name}: {value!r}"
        )
    return int(value)


def _download_mode(
    response: Any, file: RemoteFile, existing_size: int
) -> str:
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        valid = match is not None
        if match is not None:
            start, end, total = (int(value) for value in match.groups())
            valid = (
                start == existing_size
                and start <= end
                and end < total
                and total == file.size
            )
            content_length = _content_length(response, file)
            if content_length is not None:
                valid = valid and content_length == end - start + 1
        if not valid:
            raise DownloadResponseError(
                f"Invalid Content-Range for {file.name}: {content_range!r}; "
                f"expected offset {existing_size} within total {file.size}"
            )
        return "ab" if existing_size else "wb"
    if response.status_code == 200:
        content_length = _content_length(response, file)
        if content_length is not None and content_length != file.size:
            raise DownloadResponseError(
                f"Invalid Content-Length for {file.name}: "
                f"{response.headers.get('Content-Length')!r}; expected {file.size}"
            )
        return "wb"
    raise DownloadResponseError(
        f"Unexpected download status for {file.name}: {response.status_code}"
    )


def _promote(partial_path: Path, final_path: Path) -> None:
    with partial_path.open("ab") as handle:
        os.fsync(handle.fileno())
    partial_path.replace(final_path)


def _matches_record(path: Path, file: VerifiableFile) -> bool:
    return _matches(path, file.size, file.checksum, "md5")


def _matches(path: Path, size: int, checksum: str | None, algorithm: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size and (
            _checksum_matches(path, checksum, algorithm)
        )
    except OSError:
        return False


def _checksum_matches(path: Path, checksum: str | None, algorithm: str) -> bool:
    return checksum is None or _digest(path, algorithm).lower() == checksum.lower()


def _digest(path: Path, algorithm: str) -> str:
    if algorithm == "md5":
        return _md5(path)
    return file_digest(path, algorithm, chunk_size=CHUNK_SIZE)


def _md5(path: Path) -> str:
    return md5_file(path, chunk_size=CHUNK_SIZE)


def _invalid_archive_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.invalid")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid.{suffix}")
        suffix += 1
    return candidate
