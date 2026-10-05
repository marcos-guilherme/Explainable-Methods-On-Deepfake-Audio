"""Tests for the shared ``core`` and ``storage`` building blocks.

They pin the new reusable helpers (hashing, data layout, ledger, verified
download) and prove that the legacy modules keep exposing the very same
objects, so existing imports and monkeypatches keep working.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
from pathlib import Path

import pytest

from jmds_prepare import audio, audit, cli, extraction, manifest, protocols, zenodo
from jmds_prepare.core import hashing
from jmds_prepare.storage import download, ledger
from jmds_prepare.storage.layout import (
    ASVSPOOF5_SOURCE_SLUG,
    ENGLISH_LANGUAGE_SLUG,
    DataLayout,
    english_layout,
)

PAYLOAD = b"jmds-prepare" * 200_000


@pytest.fixture
def payload_file(tmp_path: Path) -> Path:
    path = tmp_path / "payload.bin"
    path.write_bytes(PAYLOAD)
    return path


def _freeze_times(path: Path) -> tuple[int, int]:
    os.utime(path, ns=(1_000_000_000_123_456_700, 1_100_000_000_987_654_300))
    stat = path.stat()
    return stat.st_atime_ns, stat.st_mtime_ns


# --- core.hashing -------------------------------------------------------------


def test_sha256_and_md5_match_hashlib(payload_file):
    assert hashing.sha256_file(payload_file) == hashlib.sha256(PAYLOAD).hexdigest()
    assert hashing.md5_file(payload_file) == hashlib.md5(PAYLOAD).hexdigest()
    assert hashing.sha256_file(str(payload_file), chunk_size=7) == (
        hashlib.sha256(PAYLOAD).hexdigest()
    )


def test_copy_with_sha256_copies_bytes_and_returns_digest():
    destination = io.BytesIO()

    digest = hashing.copy_with_sha256(io.BytesIO(PAYLOAD), destination)

    assert destination.getvalue() == PAYLOAD
    assert digest == hashlib.sha256(PAYLOAD).hexdigest()


def test_sha256_preserving_times_restores_access_and_modification_times(
    payload_file,
):
    times = _freeze_times(payload_file)

    digest = hashing.sha256_file_preserving_times(payload_file)

    stat = payload_file.stat()
    assert (stat.st_atime_ns, stat.st_mtime_ns) == times
    assert digest == hashlib.sha256(PAYLOAD).hexdigest()


def test_sha256_preserving_times_uses_the_given_hasher_and_restores_on_error(
    payload_file,
):
    times = _freeze_times(payload_file)

    def failing_hasher(path: Path) -> str:
        os.utime(path, ns=(1, 2))
        raise OSError("hash failed")

    with pytest.raises(OSError, match="hash failed"):
        hashing.sha256_file_preserving_times(payload_file, hasher=failing_hasher)

    stat = payload_file.stat()
    assert (stat.st_atime_ns, stat.st_mtime_ns) == times


@pytest.mark.parametrize(
    "helper",
    [
        audio._sha256,
        audio._sha256_preserving_times,
        manifest._sha256,
        manifest._sha256_preserving_times,
        audit._sha256_path,
        extraction._hash_file,
        protocols.protocol_file_sha256,
        cli._sha256_file,
    ],
)
def test_legacy_sha256_helpers_delegate_to_core(payload_file, helper):
    assert helper(payload_file) == hashing.sha256_file(payload_file)


def test_legacy_md5_helper_delegates_to_core(payload_file):
    assert zenodo._md5(payload_file) == hashing.md5_file(payload_file)


def test_legacy_copy_with_hash_delegates_to_core():
    destination = io.BytesIO()

    digest = extraction._copy_with_hash(io.BytesIO(PAYLOAD), destination)

    assert destination.getvalue() == PAYLOAD
    assert digest == hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.parametrize("module", [audio, manifest])
def test_legacy_preserving_wrappers_keep_times_and_module_hasher(
    payload_file, monkeypatch, module
):
    times = _freeze_times(payload_file)
    seen: list[Path] = []
    monkeypatch.setattr(
        module, "_sha256", lambda path: seen.append(path) or "patched"
    )

    assert module._sha256_preserving_times(payload_file) == "patched"

    assert seen == [payload_file]
    stat = payload_file.stat()
    assert (stat.st_atime_ns, stat.st_mtime_ns) == times


# --- storage.layout -----------------------------------------------------------


def test_english_layout_reproduces_the_current_english_paths(tmp_path):
    layout = english_layout(tmp_path)

    assert layout == DataLayout(tmp_path, "english", "asvspoof5")
    assert (ENGLISH_LANGUAGE_SLUG, ASVSPOOF5_SOURCE_SLUG) == ("english", "asvspoof5")
    assert layout.archive_dir == tmp_path / "archives"
    assert layout.raw_source_dir == tmp_path / "raw" / "asvspoof5"
    assert layout.raw_split_dir("train") == tmp_path / "raw" / "asvspoof5" / "train"
    assert layout.processed_language_dir == tmp_path / "processed" / "english"
    assert layout.processed_dir("dev", "pristine") == (
        tmp_path / "processed" / "english" / "dev" / "pristine"
    )
    assert layout.manifests_dir == tmp_path / "manifests"
    assert layout.raw_manifest_path == tmp_path / "manifests" / "english_raw.csv"
    assert layout.processed_manifest_path == (
        tmp_path / "manifests" / "english_processed.csv"
    )
    assert layout.extraction_ledger_path == (
        tmp_path / "manifests" / "extraction_ledger.csv"
    )
    assert layout.reports_dir == tmp_path / "reports"


def test_layout_derives_paths_from_its_slugs(tmp_path):
    layout = DataLayout(str(tmp_path), "portuguese", "other_source")

    assert layout.data_root == tmp_path
    assert layout.raw_split_dir("dev") == tmp_path / "raw" / "other_source" / "dev"
    assert layout.processed_dir("train", "generated") == (
        tmp_path / "processed" / "portuguese" / "train" / "generated"
    )
    assert layout.raw_manifest_path == tmp_path / "manifests" / "portuguese_raw.csv"


def test_layout_is_immutable(tmp_path):
    layout = english_layout(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.language_slug = "other"  # type: ignore[misc]


@pytest.mark.parametrize("slug", ["", "en/us", "en\\us", "..", " english"])
def test_layout_rejects_slugs_that_are_not_single_path_segments(tmp_path, slug):
    with pytest.raises(ValueError, match="slug"):
        DataLayout(tmp_path, slug, "asvspoof5")
    with pytest.raises(ValueError, match="slug"):
        DataLayout(tmp_path, "english", slug)


@pytest.mark.parametrize(
    "slug",
    [
        "D:",
        "en:x",
        "en.",
        "en ",
        "en\x00x",
        "English",
        "_english",
        "-english",
        "en.us",
        "en us",
        "inglês",
        "nul",
        "NUL",
        "con",
        "prn",
        "aux",
        "com1",
        "lpt9",
    ],
)
def test_layout_rejects_windows_unsafe_or_non_whitelisted_slugs(tmp_path, slug):
    with pytest.raises(ValueError, match="language_slug"):
        DataLayout(tmp_path, slug, "asvspoof5")
    with pytest.raises(ValueError, match="source_slug"):
        DataLayout(tmp_path, "english", slug)


@pytest.mark.parametrize("slug", ["english", "asvspoof5", "pt_br", "a-1", "0x"])
def test_layout_accepts_whitelisted_slugs(tmp_path, slug):
    layout = DataLayout(tmp_path, slug, slug)

    assert (layout.language_slug, layout.source_slug) == (slug, slug)


def test_slug_error_message_states_the_allowed_pattern(tmp_path):
    with pytest.raises(ValueError) as error:
        DataLayout(tmp_path, "D:", "asvspoof5")

    message = str(error.value)
    assert "'D:'" in message
    assert "lowercase" in message
    assert "reserved" in message


# --- storage.ledger and storage.download re-exports ---------------------------


@pytest.mark.parametrize(
    "name",
    [
        "LEDGER_COLUMNS",
        "ExtractionLedgerEntry",
        "read_extraction_ledger",
        "update_extraction_ledger",
    ],
)
def test_extraction_reexports_the_ledger_objects(name):
    assert getattr(extraction, name) is getattr(ledger, name)


@pytest.mark.parametrize(
    "name",
    [
        "CHUNK_SIZE",
        "ChecksumError",
        "DownloadSizeError",
        "DownloadResponseError",
        "download_verified",
        "_download_once",
        "_download_mode",
        "_matches_record",
        "_invalid_archive_path",
    ],
)
def test_zenodo_reexports_the_download_objects(name):
    assert getattr(zenodo, name) is getattr(download, name)


def test_cli_aliases_resolve_to_the_storage_implementations():
    assert cli.download_verified is download.download_verified
    assert cli.update_extraction_ledger is ledger.update_extraction_ledger
    assert cli.read_extraction_ledger is ledger.read_extraction_ledger
    assert cli.ExtractionLedgerEntry is ledger.ExtractionLedgerEntry
