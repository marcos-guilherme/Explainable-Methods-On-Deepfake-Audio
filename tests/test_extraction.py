import io
import os
import tarfile
from pathlib import Path

import pytest

from jmds_prepare.extraction import (
    ArchiveInventory,
    DestinationConflictError,
    extract_selected,
    inventory_archive,
)


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_extracts_only_requested_regular_flac_and_reports_missing_and_unsafe(
    tmp_path,
):
    archive = tmp_path / "audio.tar"
    _write_tar(
        archive,
        [
            ("nested/T_0001.flac", b"selected"),
            ("nested/T_0002.flac", b"not selected"),
            ("../unsafe.flac", b"unsafe"),
            ("/absolute.flac", b"absolute"),
        ],
    )
    output = tmp_path / "raw" / "train"

    report = extract_selected(archive, {"T_0001", "T_9999"}, output)

    assert report.extracted_ids == ("T_0001",)
    assert report.missing_ids == ("T_9999",)
    assert report.duplicate_ids == ()
    assert report.rejected_unsafe_members == (
        "../unsafe.flac",
        "/absolute.flac",
    )
    assert (output / "T_0001.flac").read_bytes() == b"selected"
    assert not (output / "T_0002.flac").exists()
    assert not (tmp_path / "unsafe.flac").exists()


def test_inventory_reports_safe_flacs_duplicates_and_unsafe_members(tmp_path):
    archive = tmp_path / "inventory.tar"
    _write_tar(
        archive,
        [
            ("first/T_0001.flac", b"one"),
            ("second/T_0001.flac", b"duplicate"),
            ("nested/T_0002.flac", b"two"),
            ("nested/readme.txt", b"ignore"),
            ("../unsafe.flac", b"unsafe"),
        ],
    )

    inventory = inventory_archive(archive)

    assert inventory == ArchiveInventory(
        flac_ids=("T_0001", "T_0002"),
        duplicate_ids=("T_0001",),
        rejected_unsafe_members=("../unsafe.flac",),
    )


def test_inventory_rejects_malformed_tar(tmp_path):
    archive = tmp_path / "broken.tar"
    archive.write_bytes(b"not a tar")

    with pytest.raises(tarfile.ReadError):
        inventory_archive(archive)


def test_rejects_windows_and_backslash_member_names(tmp_path):
    archive = tmp_path / "unsafe-names.tar"
    unsafe_names = [
        r"..\unsafe.flac",
        "C:/unsafe.flac",
        "//server/share/unsafe.flac",
        r"nested\T_0001.flac",
    ]
    _write_tar(archive, [(name, b"unsafe") for name in unsafe_names])
    output = tmp_path / "output"

    report = extract_selected(archive, {"T_0001"}, output)

    assert report.rejected_unsafe_members == tuple(unsafe_names)
    assert report.missing_ids == ("T_0001",)
    assert not (output / "T_0001.flac").exists()


def test_ignores_non_regular_members(tmp_path):
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as tar:
        directory = tarfile.TarInfo("T_directory.flac")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        link = tarfile.TarInfo("T_link.flac")
        link.type = tarfile.SYMTYPE
        link.linkname = "../target"
        tar.addfile(link)

    report = extract_selected(
        archive, {"T_directory", "T_link"}, tmp_path / "output"
    )

    assert report.extracted_ids == ()
    assert report.missing_ids == ("T_directory", "T_link")


def test_duplicate_requested_id_is_reported_and_not_extracted(tmp_path):
    archive = tmp_path / "duplicate.tar"
    _write_tar(
        archive,
        [
            ("first/T_0001.flac", b"first"),
            ("second/T_0001.flac", b"second"),
        ],
    )
    output = tmp_path / "output"

    report = extract_selected(archive, {"T_0001"}, output)

    assert report.extracted_ids == ()
    assert report.missing_ids == ()
    assert report.duplicate_ids == ("T_0001",)
    assert not (output / "T_0001.flac").exists()


def test_identical_existing_destination_is_idempotent(tmp_path):
    archive = tmp_path / "audio.tar"
    _write_tar(archive, [("T_0001.flac", b"same bytes")])
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "T_0001.flac"
    destination.write_bytes(b"same bytes")
    original_stat = destination.stat()

    report = extract_selected(archive, {"T_0001"}, output)

    assert report.extracted_ids == ("T_0001",)
    assert destination.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert destination.read_bytes() == b"same bytes"


def test_conflicting_existing_destination_fails_without_rewrite(tmp_path):
    archive = tmp_path / "audio.tar"
    _write_tar(archive, [("T_0001.flac", b"archive bytes")])
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "T_0001.flac"
    destination.write_bytes(b"existing bytes")

    with pytest.raises(DestinationConflictError, match="T_0001"):
        extract_selected(archive, {"T_0001"}, output)

    assert destination.read_bytes() == b"existing bytes"
    assert list(output.glob("*.tmp")) == []


def test_concurrent_destination_is_never_overwritten(tmp_path, monkeypatch):
    archive = tmp_path / "audio.tar"
    _write_tar(archive, [("T_0001.flac", b"archive bytes")])
    output = tmp_path / "output"
    destination = output / "T_0001.flac"
    real_link = os.link
    publication_attempted = False

    def race_link(source, target):
        nonlocal publication_attempted
        publication_attempted = True
        destination.write_bytes(b"concurrent bytes")
        real_link(source, target)

    monkeypatch.setattr(os, "link", race_link)

    with pytest.raises(DestinationConflictError, match="T_0001"):
        extract_selected(archive, {"T_0001"}, output)

    assert publication_attempted
    assert destination.read_bytes() == b"concurrent bytes"
    assert list(output.glob("*.tmp")) == []


def test_malformed_tar_fails(tmp_path):
    archive = tmp_path / "broken.tar"
    archive.write_bytes(b"not a tar archive")

    with pytest.raises(tarfile.ReadError):
        extract_selected(archive, {"T_0001"}, tmp_path / "output")


def test_copy_failure_removes_temporary_output(tmp_path, monkeypatch):
    archive = tmp_path / "audio.tar"
    _write_tar(archive, [("T_0001.flac", b"audio bytes")])
    output = tmp_path / "output"

    original_extractfile = tarfile.TarFile.extractfile

    class BrokenStream:
        def read(self, _size=-1):
            raise OSError("simulated read failure")

        def close(self):
            pass

    def broken_extractfile(self, member):
        original_extractfile(self, member).close()
        return BrokenStream()

    monkeypatch.setattr(tarfile.TarFile, "extractfile", broken_extractfile)

    with pytest.raises(OSError, match="simulated read failure"):
        extract_selected(archive, {"T_0001"}, output)

    assert not (output / "T_0001.flac").exists()
    assert list(output.glob("*.tmp")) == []
