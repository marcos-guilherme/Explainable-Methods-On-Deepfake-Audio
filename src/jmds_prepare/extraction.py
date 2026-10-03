from __future__ import annotations

import os
import tarfile
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

from .core.hashing import copy_with_sha256, sha256_file
from .storage.ledger import (
    LEDGER_COLUMNS as LEDGER_COLUMNS,
    ExtractionLedgerEntry as ExtractionLedgerEntry,
    read_extraction_ledger as read_extraction_ledger,
    update_extraction_ledger as update_extraction_ledger,
)


@dataclass(frozen=True)
class ExtractionReport:
    extracted_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    rejected_unsafe_members: tuple[str, ...]
    members: tuple["ExtractedMember", ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class ArchiveInventory:
    flac_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    rejected_unsafe_members: tuple[str, ...]
    members: tuple["ArchiveMember", ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class ArchiveMember:
    utt_id: str
    member_name: str
    member_size: int


@dataclass(frozen=True)
class ExtractedMember:
    utt_id: str
    member_name: str
    member_size: int
    sha256_extracted: str


class DestinationConflictError(FileExistsError):
    """Raised when an extracted file would overwrite different content."""


def _is_unsafe_member_path(name: str) -> bool:
    if "\\" in name:
        return True

    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    return (
        posix_path.is_absolute()
        or ".." in posix_path.parts
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    )


def _copy_with_hash(source: BinaryIO, destination: BinaryIO) -> str:
    return copy_with_sha256(source, destination)


def _hash_file(path: Path) -> str:
    return sha256_file(path)


def _extract_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    utt_id: str,
    output: Path,
) -> str:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{utt_id}.flac"
    temporary_path: Path | None = None

    try:
        source = archive.extractfile(member)
        if source is None:
            raise tarfile.ExtractError(
                f"Could not read regular TAR member {member.name!r}"
            )

        with closing(source):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output,
                prefix=f".{utt_id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                source_hash = _copy_with_hash(source, temporary)

        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            if _hash_file(destination) == source_hash:
                temporary_path.unlink()
                temporary_path = None
                return source_hash
            raise DestinationConflictError(
                f"Destination for {utt_id!r} exists with different content: "
                f"{destination}"
            )
        temporary_path.unlink()
        temporary_path = None
        return source_hash
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _inventory_members(
    archive: tarfile.TarFile,
) -> tuple[dict[str, list[tarfile.TarInfo]], list[str]]:
    members_by_id: dict[str, list[tarfile.TarInfo]] = {}
    unsafe_members: list[str] = []
    for member in archive:
        if not member.isreg() or PurePosixPath(member.name).suffix != ".flac":
            continue
        if _is_unsafe_member_path(member.name):
            unsafe_members.append(member.name)
            continue
        utt_id = PurePosixPath(member.name).stem
        members_by_id.setdefault(utt_id, []).append(member)
    return members_by_id, unsafe_members


def inventory_archive(archive: Path) -> ArchiveInventory:
    """Inventory safe regular FLAC members without extracting any data."""
    with tarfile.open(Path(archive), mode="r:*") as tar:
        members_by_id, unsafe_members = _inventory_members(tar)
    duplicate_ids = tuple(
        sorted(
            utt_id
            for utt_id, members in members_by_id.items()
            if len(members) > 1
        )
    )
    return ArchiveInventory(
        flac_ids=tuple(sorted(members_by_id)),
        duplicate_ids=duplicate_ids,
        rejected_unsafe_members=tuple(unsafe_members),
        members=tuple(
            ArchiveMember(utt_id, members[0].name, members[0].size)
            for utt_id, members in sorted(members_by_id.items())
            if len(members) == 1
        ),
    )


def extract_selected(
    archive: Path, wanted_ids: set[str], output: Path
) -> ExtractionReport:
    """Safely extract requested regular-file members from a TAR archive."""
    archive = Path(archive)
    output = Path(output)
    with tarfile.open(archive, mode="r:*") as tar:
        all_members, unsafe_members = _inventory_members(tar)
        members_by_id = {
            utt_id: members
            for utt_id, members in all_members.items()
            if utt_id in wanted_ids
        }

        duplicate_ids = {
            utt_id
            for utt_id, members in members_by_id.items()
            if len(members) > 1
        }
        extractable_ids = set(members_by_id) - duplicate_ids

        extracted_members = []
        for utt_id in sorted(extractable_ids):
            member = members_by_id[utt_id][0]
            digest = _extract_member(tar, member, utt_id, output)
            extracted_members.append(
                ExtractedMember(utt_id, member.name, member.size, digest)
            )

    present_ids = set(members_by_id)
    return ExtractionReport(
        extracted_ids=tuple(sorted(extractable_ids)),
        missing_ids=tuple(sorted(wanted_ids - present_ids)),
        duplicate_ids=tuple(sorted(duplicate_ids)),
        rejected_unsafe_members=tuple(unsafe_members),
        members=tuple(extracted_members),
    )
