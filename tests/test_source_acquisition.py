"""Catalog-driven, verifiable acquisition of external source datasets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests
import yaml

from jmds_prepare import source_acquisition
from jmds_prepare.core import atomic, paths
from jmds_prepare.storage import artifacts, download, estimates
from jmds_prepare.storage.artifacts import (
    AcquisitionLockedError,
    CatalogError,
    InsufficientSpaceError,
    ReceiptMismatchError,
    acquire_artifact,
    acquire_catalog,
    load_catalog,
    plan_acquisition,
    receipt_path,
)
from jmds_prepare.storage.download import (
    ChecksumError,
    DownloadResponseError,
    DownloadSizeError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = REPO_ROOT / "configs" / "sources"
REVISION = "719c91226a79f5f9a8984145f15f29626eabc29a"
FIXED_NOW = datetime(2026, 10, 1, 23, 35, 0, tzinfo=UTC)
Usage = namedtuple("Usage", "total used free")


class FakeResponse:
    def __init__(self, *, status_code=200, chunks=(), headers=None, url=None, history=()):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = headers or {}
        self.url = url
        self.history = list(history)

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self._chunks


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected network call to {url}")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def artifact_entry(name, content, *, official=True, **overrides):
    entry = {
        "name": name,
        "remote_path": f"folder/{name}",
        "url": f"https://files.example/{REVISION}/folder/{name}?download=true",
        "size": len(content),
        "checksum": {
            "algorithm": "sha256",
            "value": sha256(content) if official else None,
            "source": "test LFS oid" if official else None,
            "note": None if official else "No official checksum published.",
        },
    }
    entry.update(overrides)
    return entry


def catalog_payload(*entries):
    return {
        "schema_version": 1,
        "dataset": {
            "id": "demo-v1",
            "name": "Demo corpus",
            "version": "v1",
            "revision": REVISION,
            "official_page": "https://example.org/demo",
            "distribution_page": "https://files.example/demo",
            "license": {
                "id": "CC-BY-NC-ND-4.0",
                "url": "https://example.org/demo/LICENSE",
                "source": "https://example.org/demo",
            },
            "usage_policy": {
                "preserve_originals": True,
                "commercial_use_permitted": False,
                "adapted_material_sharing_permitted": False,
                "statement": "Noncommercial; adapted material is never shared.",
            },
        },
        "artifacts": list(entries),
    }


def write_catalog(tmp_path, payload, name="demo.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def demo_catalog(tmp_path, *entries):
    return load_catalog(write_catalog(tmp_path, catalog_payload(*entries)))


def ok(content):
    return FakeResponse(
        chunks=(content,), headers={"Content-Length": str(len(content))}
    )


def acquire(catalog, artifact, destination, session, **kwargs):
    return acquire_artifact(
        catalog,
        artifact,
        destination,
        session=session,
        retry_delay=0,
        now=lambda: FIXED_NOW,
        **kwargs,
    )


# --- schema ------------------------------------------------------------------


def test_aishell3_catalog_matches_official_source():
    catalog = load_catalog(SOURCES / "aishell3.yaml")

    assert catalog.id == "aishell3"
    assert catalog.official_page == "https://openslr.org/93/"
    assert catalog.license.id == "Apache-2.0"
    assert catalog.usage_policy.preserve_originals is True
    (artifact,) = catalog.artifacts
    assert artifact.name == "data_aishell3.tgz"
    assert artifact.url == "https://www.openslr.org/resources/93/data_aishell3.tgz"
    assert artifact.size == 19_057_141_777
    assert artifact.checksum.value is None
    assert artifact.checksum.source is None
    assert "no official checksum" in artifact.checksum.note.lower()


def test_coraa_catalog_pins_revision_license_and_checksums():
    catalog = load_catalog(SOURCES / "coraa-v1.1.yaml")

    assert catalog.id == "coraa-v1.1"
    assert catalog.version == "v1.1"
    assert catalog.revision == REVISION
    assert catalog.official_page == "https://github.com/nilc-nlp/CORAA"
    assert catalog.license.id == "CC-BY-NC-ND-4.0"
    assert catalog.license.source == (
        "https://github.com/nilc-nlp/CORAA/blob/main/LICENSE"
    )
    policy = catalog.usage_policy
    assert policy.preserve_originals is True
    assert policy.commercial_use_permitted is False
    assert policy.adapted_material_sharing_permitted is False

    expected = {
        "dev.zip": (1_269_183_231, "5945180bbac98943e3e047e99a53075c6ff4ef8ef2e32d261c3bffd34b89c720"),
        "test.zip": (2_421_277_906, "4dd6cb9e658c83e45496058dbae8f08ade55eafe1ae617094000f6ef042f7238"),
        "train.part1.rar": (10_737_418_240, "50d234f590a1027ff9c0314de3159671062f630a3546eb943529a114c97af1a2"),
        "train.part2.rar": (10_737_418_240, "748a41dadf6362060950a2ea1e4964985323e7e2e75d7027e1328159d387aaef"),
        "train.part3.rar": (10_737_418_240, "11f9dbb379fbbe835158ea23617e9123ef8a47ad5aaf3baef9ebc831aaa7415e"),
        "train.part4.rar": (10_737_418_240, "cc7b05626a4eef79a6c58dba821091a70207e3d6f10aa43154db048c76162bb5"),
        "train.part5.rar": (10_078_493_674, "90e967848bfef587d0d6378fe9f5f0b6ff772a1c3f44b275df75224ceaa9bbf8"),
        "metadata_train_final.csv": (76_480_333, None),
        "metadata_dev_final.csv": (1_434_289, None),
        "metadata_test_final.csv": (2_635_505, None),
    }
    assert {a.name: (a.size, a.checksum.value) for a in catalog.artifacts} == expected
    prefix = (
        "https://huggingface.co/datasets/gabrielrstan/CORAA-v1.1/resolve/"
        f"{REVISION}/"
    )
    for artifact in catalog.artifacts:
        assert artifact.url == f"{prefix}{artifact.remote_path}?download=true"
        assert artifact.checksum.algorithm == "sha256"
        if artifact.checksum.value is None:
            assert artifact.checksum.source is None
            assert artifact.checksum.note
        else:
            assert artifact.checksum.source
    part1 = next(a for a in catalog.artifacts if a.name == "train.part1.rar")
    assert part1.remote_path == "train_dividido/train.part1.rar"


def _mutate(payload, dotted, value):
    target = payload
    *parents, last = dotted.split(".")
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if value is _DELETE:
        del target[last]
    else:
        target[last] = value
    return payload


_DELETE = object()


@pytest.mark.parametrize(
    ("dotted", "value", "message"),
    [
        ("unexpected", 1, "unexpected"),
        ("dataset.extra", "x", "extra"),
        ("artifacts.0.mirror", "x", "mirror"),
        ("artifacts.0.checksum.extra", "x", "extra"),
        ("dataset.license", _DELETE, "license"),
        ("artifacts.0.checksum", _DELETE, "checksum"),
        ("artifacts.0.checksum.note", _DELETE, "note"),
        ("schema_version", 2, "schema_version"),
        ("dataset.id", "Bad Id", "id"),
        ("dataset.official_page", "http://example.org", "https"),
        ("artifacts.0.url", "ftp://files.example/a.zip", "https"),
        (
            "artifacts.0.url",
            f"https://user:secret@files.example/{REVISION}/folder/a.zip",
            "credentials",
        ),
        ("artifacts.0.remote_path", "folder/other.zip", "remote_path"),
        ("artifacts.0.size", True, "size"),
        ("artifacts.0.size", -1, "size"),
        ("artifacts.0.size", "10", "size"),
        ("artifacts.0.checksum.algorithm", "crc32", "algorithm"),
        ("artifacts.0.checksum.value", "abc", "sha256"),
        ("artifacts.0.checksum.source", None, "source"),
        ("dataset.usage_policy.preserve_originals", "yes", "preserve_originals"),
        ("artifacts", [], "artifacts"),
    ],
)
def test_catalog_schema_is_strict(tmp_path, dotted, value, message):
    payload = _mutate(catalog_payload(artifact_entry("a.zip", b"abc")), dotted, value)

    with pytest.raises(CatalogError, match=message):
        load_catalog(write_catalog(tmp_path, payload))


def test_catalog_requires_note_when_checksum_is_null(tmp_path):
    entry = artifact_entry("a.zip", b"abc", official=False)
    entry["checksum"]["note"] = None

    with pytest.raises(CatalogError, match="note"):
        load_catalog(write_catalog(tmp_path, catalog_payload(entry)))


def test_catalog_rejects_duplicate_yaml_keys(tmp_path):
    path = write_catalog(tmp_path, catalog_payload(artifact_entry("a.zip", b"abc")))
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("  name: Demo corpus\n", "  name: Demo corpus\n  name: Other\n"), encoding="utf-8")

    with pytest.raises(CatalogError, match="[Dd]uplicate"):
        load_catalog(path)


def test_catalog_rejects_duplicate_local_names(tmp_path):
    payload = catalog_payload(
        artifact_entry("a.zip", b"abc"), artifact_entry("A.ZIP", b"def")
    )

    with pytest.raises(CatalogError, match="[Dd]uplicate"):
        load_catalog(write_catalog(tmp_path, payload))


def test_catalog_requires_pinned_revision_in_every_url(tmp_path):
    entry = artifact_entry("a.zip", b"abc", url="https://files.example/main/a.zip")

    with pytest.raises(CatalogError, match="revision"):
        load_catalog(write_catalog(tmp_path, catalog_payload(entry)))


# --- path safety -------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../evil.zip",
        "sub/file.zip",
        "sub\\file.zip",
        "/abs.zip",
        "C:evil.zip",
        "file.zip:stream",
        "CON",
        "nul.txt",
        "trailing.",
        "trailing ",
        "file.zip.partial",
        "file.zip.receipt.json",
        "file.zip.invalid",
        "bad\x00name",
    ],
)
def test_local_names_are_plain_safe_filenames(tmp_path, name):
    entry = artifact_entry("placeholder.zip", b"abc")
    entry["name"] = name

    with pytest.raises(CatalogError, match="name"):
        load_catalog(write_catalog(tmp_path, catalog_payload(entry)))


@pytest.mark.parametrize(
    "remote_path",
    ["../escape.zip", "/root.zip", "a/../../b.zip", "a\\b.zip", "a//b.zip", "C:/x.zip"],
)
def test_remote_paths_reject_traversal(tmp_path, remote_path):
    entry = artifact_entry("a.zip", b"abc", remote_path=remote_path)

    with pytest.raises(CatalogError, match="remote_path"):
        load_catalog(write_catalog(tmp_path, catalog_payload(entry)))


# --- SHA-256 verification and receipts ---------------------------------------


def test_official_sha256_download_writes_atomic_receipt(tmp_path):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"

    receipt = acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))

    assert (destination / "a.zip").read_bytes() == content
    assert not (destination / "a.zip.partial").exists()
    stored = json.loads(receipt_path(destination, catalog.artifacts[0]).read_text("utf-8"))
    assert stored == receipt
    assert stored["catalog"]["dataset_id"] == "demo-v1"
    assert stored["catalog"]["version"] == "v1"
    assert stored["catalog"]["revision"] == REVISION
    assert stored["catalog"]["sha256"] == sha256(catalog.source_path.read_bytes())
    assert stored["official_page"] == "https://example.org/demo"
    assert stored["license"] == {
        "id": "CC-BY-NC-ND-4.0",
        "url": "https://example.org/demo/LICENSE",
        "source": "https://example.org/demo",
    }
    assert stored["artifact"]["url"] == catalog.artifacts[0].url
    assert stored["expected"] == {
        "size_bytes": len(content),
        "checksum_algorithm": "sha256",
        "checksum": sha256(content),
        "checksum_source": "test LFS oid",
        "checksum_note": None,
    }
    assert stored["observed"] == {"size_bytes": len(content), "sha256": sha256(content)}
    assert stored["origin"] == "downloaded"
    assert stored["official_checksum_verified"] is True
    assert stored["verified_at_utc"] == "2026-10-01T23:35:00Z"
    assert not list(destination.glob("*.tmp"))


def test_missing_official_checksum_records_observed_sha256_without_claim(tmp_path):
    content = b"no published checksum"
    catalog = demo_catalog(tmp_path, artifact_entry("a.tgz", content, official=False))
    destination = tmp_path / "dest"

    receipt = acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))

    assert receipt["expected"]["checksum"] is None
    assert receipt["expected"]["checksum_source"] is None
    assert receipt["expected"]["checksum_note"] == "No official checksum published."
    assert receipt["observed"]["sha256"] == sha256(content)
    assert receipt["official_checksum_verified"] is False


@pytest.mark.parametrize(
    ("state", "official", "origin", "headers"),
    [
        ("missing", True, "downloaded", [{"Accept-Encoding": "identity"}]),
        (
            "short_partial",
            True,
            "resumed_download",
            [{"Range": "bytes=10-", "Accept-Encoding": "identity"}],
        ),
        ("final", False, "adopted_existing_final", []),
        ("complete_partial", False, "promoted_partial", []),
    ],
)
def test_receipt_origin_reflects_the_real_state(tmp_path, state, official, origin, headers):
    content = b"resumable archive content"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content, official=official))
    destination = tmp_path / "dest"
    destination.mkdir()
    responses = []
    if state == "missing":
        responses.append(ok(content))
    elif state == "short_partial":
        (destination / "a.zip.partial").write_bytes(content[:10])
        responses.append(
            FakeResponse(
                status_code=206,
                chunks=(content[10:],),
                headers={"Content-Range": f"bytes 10-{len(content) - 1}/{len(content)}"},
            )
        )
    elif state == "final":
        (destination / "a.zip").write_bytes(content)
    else:
        (destination / "a.zip.partial").write_bytes(content)
    session = FakeSession(*responses)

    receipt = acquire(catalog, catalog.artifacts[0], destination, session)

    assert receipt["origin"] == origin
    assert receipt["official_checksum_verified"] is official
    assert receipt["observed"] == {"size_bytes": len(content), "sha256": sha256(content)}
    assert [kwargs["headers"] for _url, kwargs in session.calls] == headers
    assert (destination / "a.zip").read_bytes() == content


def test_partial_is_fsynced_before_promotion(tmp_path, monkeypatch):
    content = b"durable bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    events = []
    real_fsync, real_replace = download.os.fsync, Path.replace
    monkeypatch.setattr(
        download.os, "fsync", lambda fd: (events.append("fsync"), real_fsync(fd))[1]
    )
    monkeypatch.setattr(
        Path, "replace", lambda self, target: (events.append("replace"), real_replace(self, target))[1]
    )
    monkeypatch.setattr(artifacts, "write_json_atomic", lambda *_args: None)

    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))

    assert events == ["fsync", "replace"]


def test_sha256_mismatch_preserves_partial_and_writes_no_receipt(tmp_path):
    content = b"official archive bytes"
    entry = artifact_entry("a.zip", content)
    entry["checksum"]["value"] = "0" * 64
    catalog = demo_catalog(tmp_path, entry)
    destination = tmp_path / "dest"

    with pytest.raises(ChecksumError, match="SHA-256 mismatch for a.zip"):
        acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))

    assert (destination / "a.zip.partial").read_bytes() == content
    assert not (destination / "a.zip").exists()
    assert not receipt_path(destination, catalog.artifacts[0]).exists()


def test_failed_receipt_write_leaves_no_receipt_or_temp(tmp_path, monkeypatch):
    content = b"no published checksum"
    catalog = demo_catalog(tmp_path, artifact_entry("a.tgz", content, official=False))
    destination = tmp_path / "dest"

    def fail(*_args, **_kwargs):
        raise RuntimeError("disk failure")

    monkeypatch.setattr(atomic.json, "dump", fail)
    with pytest.raises(RuntimeError, match="disk failure"):
        acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))
    monkeypatch.undo()

    assert not receipt_path(destination, catalog.artifacts[0]).exists()
    assert not list(destination.glob(".*.tmp"))
    receipt = acquire(catalog, catalog.artifacts[0], destination, FakeSession())
    assert receipt["origin"] == "adopted_existing_final"


def test_legacy_md5_mismatch_message_is_unchanged(tmp_path):
    RecordFile = namedtuple("RecordFile", "name size checksum download_url")
    content = b"complete archive"
    record = RecordFile("x.tar", len(content), "0" * 32, "https://files.example/x.tar")

    with pytest.raises(ChecksumError) as error:
        download.download_verified(record, tmp_path, session=FakeSession(ok(content)))

    assert str(error.value) == (
        f"MD5 mismatch for x.tar: got {hashlib.md5(content).hexdigest()}, "
        f"expected {'0' * 32}"
    )


# --- transport ----------------------------------------------------------------


@pytest.mark.parametrize(
    "response_kwargs",
    [
        {"url": "http://files.example/a.zip"},
        {
            "url": "https://cdn.example/a.zip",
            "history": [FakeResponse(url="http://files.example/redirect")],
        },
    ],
)
def test_non_https_redirect_is_rejected_before_writing(tmp_path, response_kwargs):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    response = FakeResponse(chunks=(content,), **response_kwargs)

    with pytest.raises(DownloadResponseError, match="HTTPS"):
        acquire(catalog, catalog.artifacts[0], destination, FakeSession(response))

    assert not (destination / "a.zip.partial").exists()


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"Content-Encoding": "gzip"}, "Content-Encoding"),
        ({"Content-Length": "abc"}, "Content-Length"),
    ],
)
def test_unsafe_response_headers_are_rejected(tmp_path, headers, message):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    response = FakeResponse(chunks=(content,), headers=headers)

    with pytest.raises(DownloadResponseError, match=message):
        acquire(catalog, catalog.artifacts[0], destination, FakeSession(response))

    assert not (destination / "a.zip.partial").exists()


def test_retries_transient_errors_then_resumes(tmp_path):
    content = b"flaky network content"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    session = FakeSession(requests.exceptions.ConnectionError("reset"), ok(content))

    acquire(catalog, catalog.artifacts[0], destination, session, max_attempts=2)

    assert len(session.calls) == 2
    assert (destination / "a.zip").read_bytes() == content


# --- reruns and receipt validation ---------------------------------------------


def test_rerun_reuses_verified_final_and_receipt_without_network(tmp_path):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))
    receipt_file = receipt_path(destination, catalog.artifacts[0])
    before = receipt_file.read_bytes()

    receipt = acquire_artifact(
        catalog,
        catalog.artifacts[0],
        destination,
        session=FakeSession(),
        now=lambda: datetime(2030, 1, 1, tzinfo=UTC),
    )

    assert receipt_file.read_bytes() == before
    assert receipt["verified_at_utc"] == "2026-10-01T23:35:00Z"


def test_receipt_pins_unofficial_sha256_and_never_accepts_divergent_bytes(tmp_path):
    content = b"first observed bytes"
    other = b"other observed bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.tgz", content, official=False))
    destination = tmp_path / "dest"
    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))
    receipt_file = receipt_path(destination, catalog.artifacts[0])
    before = receipt_file.read_bytes()
    (destination / "a.tgz").write_bytes(other)

    with pytest.raises(ChecksumError, match="SHA-256 mismatch"):
        acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(other)))

    assert receipt_file.read_bytes() == before
    quarantined = list(destination.glob("a.tgz.invalid*"))
    assert [path.read_bytes() for path in quarantined] == [other]
    assert not (destination / "a.tgz").exists()


def _changed_license(payload):
    payload["dataset"]["license"]["url"] = "https://example.org/other-license"


def _changed_policy(payload):
    payload["dataset"]["usage_policy"]["adapted_material_sharing_permitted"] = True


def _changed_url(payload):
    payload["artifacts"][0]["url"] = f"https://mirror.example/{REVISION}/folder/a.zip"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_changed_license, "license"),
        (_changed_policy, "usage_policy"),
        (_changed_url, "artifact.url"),
        (None, "Unreadable"),
    ],
)
def test_divergent_or_invalid_receipt_refuses_without_overwriting(tmp_path, change, message):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))
    receipt_file = receipt_path(destination, catalog.artifacts[0])
    if change is None:
        receipt_file.write_text("{not json", encoding="utf-8")
        changed = catalog
    else:
        payload = catalog_payload(artifact_entry("a.zip", content))
        change(payload)
        changed = load_catalog(write_catalog(tmp_path, payload, name="changed.yaml"))
    before = receipt_file.read_bytes()

    with pytest.raises(ReceiptMismatchError, match=message):
        acquire(changed, changed.artifacts[0], destination, FakeSession())

    assert receipt_file.read_bytes() == before
    assert (destination / "a.zip").read_bytes() == content


# --- oversized / mismatched files --------------------------------------------


def test_oversized_partial_is_quarantined_before_fresh_download(tmp_path):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    destination.mkdir()
    (destination / "a.zip.partial").write_bytes(content + b"garbage")
    session = FakeSession(ok(content))

    acquire(catalog, catalog.artifacts[0], destination, session)

    assert session.calls[0][1]["headers"] == {"Accept-Encoding": "identity"}
    quarantined = list(destination.glob("a.zip.partial.invalid*"))
    assert [path.read_bytes() for path in quarantined] == [content + b"garbage"]


def test_mismatched_final_is_quarantined_not_overwritten(tmp_path):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    destination.mkdir()
    (destination / "a.zip").write_bytes(b"x" * len(content))

    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(content)))

    quarantined = list(destination.glob("a.zip.invalid*"))
    assert [path.read_bytes() for path in quarantined] == [b"x" * len(content)]
    assert (destination / "a.zip").read_bytes() == content


def test_stream_exceeding_expected_size_is_rejected(tmp_path):
    content = b"official archive bytes"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    session = FakeSession(FakeResponse(chunks=(content + b"extra",)))

    with pytest.raises(DownloadSizeError, match="exceed"):
        acquire(catalog, catalog.artifacts[0], destination, session)

    assert not receipt_path(destination, catalog.artifacts[0]).exists()


# --- disk guard, lock and catalog execution ------------------------------------


def test_plan_counts_conservative_remaining_bytes(tmp_path):
    done, short, complete, unreceipted, fresh = (
        b"done-bytes", b"partial-content", b"complete-partial", b"unreceipted", b"fresh"
    )
    catalog = demo_catalog(
        tmp_path,
        artifact_entry("done.zip", done),
        artifact_entry("short.zip", short),
        artifact_entry("complete.zip", complete),
        artifact_entry("unreceipted.zip", unreceipted),
        artifact_entry("fresh.zip", fresh),
    )
    destination = tmp_path / "dest"
    acquire(catalog, catalog.artifacts[0], destination, FakeSession(ok(done)))
    (destination / "short.zip.partial").write_bytes(short[:4])
    (destination / "complete.zip.partial").write_bytes(complete)
    (destination / "unreceipted.zip").write_bytes(unreceipted)

    plan = plan_acquisition(
        catalog,
        destination,
        minimum_free_bytes=100,
        disk_usage=lambda _path: Usage(10_000, 0, 5_000),
    )

    expected = [0, len(short) - 4, len(complete), len(unreceipted), len(fresh)]
    assert [status.remaining_bytes for status in plan.artifacts] == expected
    assert plan.remaining_bytes == sum(expected)
    assert plan.required_free_bytes == sum(expected) + 100
    assert plan.free_bytes == 5_000
    assert plan.sufficient_space is True


def test_catalog_refuses_before_downloading_when_space_is_short(tmp_path):
    content = b"0123456789"
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", content))
    destination = tmp_path / "dest"
    session = FakeSession()

    with pytest.raises(InsufficientSpaceError, match="free"):
        acquire_catalog(
            catalog,
            destination,
            minimum_free_bytes=5,
            session=session,
            disk_usage=lambda _path: Usage(100, 0, len(content) + 4),
        )

    assert session.calls == []
    assert not destination.exists()


def test_space_is_rechecked_before_each_artifact(tmp_path):
    first, second = b"first artifact", b"second artifact"
    catalog = demo_catalog(
        tmp_path, artifact_entry("b.zip", first), artifact_entry("a.zip", second)
    )
    destination = tmp_path / "dest"
    session = FakeSession(ok(first), ok(second))
    free = iter([1_000, 1_000, 1])

    with pytest.raises(InsufficientSpaceError, match="a.zip"):
        acquire_catalog(
            catalog,
            destination,
            minimum_free_bytes=0,
            session=session,
            disk_usage=lambda _path: Usage(0, 0, next(free)),
        )

    assert [url for url, _ in session.calls] == [catalog.artifacts[0].url]
    assert receipt_path(destination, catalog.artifacts[0]).is_file()
    assert not (destination / artifacts.LOCK_NAME).exists()


def test_catalog_downloads_sequentially_and_releases_lock(tmp_path):
    first, second = b"first artifact", b"second artifact"
    catalog = demo_catalog(
        tmp_path, artifact_entry("b.zip", first), artifact_entry("a.zip", second)
    )
    destination = tmp_path / "dest"
    session = FakeSession(ok(first), ok(second))

    receipts = acquire_catalog(
        catalog,
        destination,
        minimum_free_bytes=0,
        session=session,
        disk_usage=lambda _path: Usage(100, 0, len(first) + len(second)),
        retry_delay=0,
    )

    assert [url for url, _ in session.calls] == [a.url for a in catalog.artifacts]
    assert [r["artifact"]["name"] for r in receipts] == ["b.zip", "a.zip"]
    assert not (destination / artifacts.LOCK_NAME).exists()


def test_catalog_stops_at_first_error_and_releases_lock(tmp_path):
    first, second = b"first artifact", b"second artifact"
    bad = artifact_entry("b.zip", first)
    bad["checksum"]["value"] = "0" * 64
    catalog = demo_catalog(tmp_path, bad, artifact_entry("a.zip", second))
    destination = tmp_path / "dest"
    session = FakeSession(ok(first), ok(second))

    with pytest.raises(ChecksumError):
        acquire_catalog(
            catalog,
            destination,
            minimum_free_bytes=0,
            session=session,
            disk_usage=lambda _path: Usage(100, 0, 100),
        )

    assert [url for url, _ in session.calls] == [catalog.artifacts[0].url]
    assert not (destination / artifacts.LOCK_NAME).exists()


def test_existing_lock_refuses_concurrent_acquisition(tmp_path):
    catalog = demo_catalog(tmp_path, artifact_entry("a.zip", b"0123"))
    destination = tmp_path / "dest"
    destination.mkdir()
    lock = destination / artifacts.LOCK_NAME
    lock.write_text("pid=1\n", encoding="utf-8")
    session = FakeSession()

    with pytest.raises(AcquisitionLockedError, match="lock"):
        acquire_catalog(
            catalog,
            destination,
            minimum_free_bytes=0,
            session=session,
            disk_usage=lambda _path: Usage(100, 0, 100),
        )

    assert session.calls == []
    assert lock.read_text(encoding="utf-8") == "pid=1\n"


# --- CLI and layering ----------------------------------------------------------


def run_cli(tmp_path, catalog_path, *extra, session=None, free=10**6):
    lines: list[str] = []
    code = source_acquisition.main(
        [
            "--catalog", str(catalog_path),
            "--destination", str(tmp_path / "dest"),
            "--minimum-free-gb", "0",
            *extra,
        ],
        session=session or FakeSession(),
        disk_usage=lambda _path: Usage(0, 0, free),
        output=lines.append,
    )
    return code, "\n".join(lines)


def test_dry_run_reports_without_network_or_writes(tmp_path):
    path = write_catalog(tmp_path, catalog_payload(artifact_entry("a.zip", b"0123456789")))

    code, text = run_cli(tmp_path, path, "--dry-run", free=99)

    assert code == 0
    for expected in (
        "demo-v1",
        "CC-BY-NC-ND-4.0",
        "https://example.org/demo/LICENSE",
        "https://files.example/",
        "remaining_bytes: 10",
        "free_bytes: 99",
        "required_free_bytes: 10",
    ):
        assert expected in text
    assert not (tmp_path / "dest").exists()


def test_cli_happy_path_separates_official_from_observed_verification(tmp_path):
    official, observed = b"official bytes", b"observed bytes"
    path = write_catalog(
        tmp_path,
        catalog_payload(
            artifact_entry("a.zip", official),
            artifact_entry("b.csv", observed, official=False),
        ),
    )

    code, text = run_cli(
        tmp_path, path, session=FakeSession(ok(official), ok(observed))
    )

    assert code == 0
    assert "verified by official checksum: a.zip" in text
    assert "verified by exact size + observed SHA-256 (no official checksum): b.csv" in text


def _missing_catalog(tmp_path):
    return tmp_path / "absent.yaml", None


def _invalid_catalog(tmp_path):
    payload = catalog_payload(artifact_entry("a.zip", b"0123"))
    payload["dataset"]["id"] = "Bad Id"
    return write_catalog(tmp_path, payload), None


def _checksum_mismatch(tmp_path):
    entry = artifact_entry("a.zip", b"0123")
    entry["checksum"]["value"] = "0" * 64
    return write_catalog(tmp_path, catalog_payload(entry)), FakeSession(ok(b"0123"))


def _locked(tmp_path):
    (tmp_path / "dest").mkdir()
    (tmp_path / "dest" / artifacts.LOCK_NAME).write_text("pid=1\n", encoding="utf-8")
    return write_catalog(tmp_path, catalog_payload(artifact_entry("a.zip", b"0123"))), None


def _short_space(tmp_path):
    return write_catalog(tmp_path, catalog_payload(artifact_entry("a.zip", b"0123"))), None


@pytest.mark.parametrize(
    ("scenario", "free"),
    [
        (_missing_catalog, 10**6),
        (_invalid_catalog, 10**6),
        (_checksum_mismatch, 10**6),
        (_locked, 10**6),
        (_short_space, 1),
    ],
)
def test_cli_reports_predictable_errors_without_traceback(tmp_path, scenario, free):
    catalog_path, session = scenario(tmp_path)

    code, text = run_cli(tmp_path, catalog_path, session=session, free=free)

    assert code == 1
    assert text.splitlines()[-1].startswith("error: ")
    assert "Traceback" not in text


@pytest.mark.parametrize("value", ["-1", "inf", "nan"])
def test_cli_rejects_non_finite_or_negative_reserve(tmp_path, value):
    path = write_catalog(tmp_path, catalog_payload(artifact_entry("a.zip", b"0123")))

    with pytest.raises(SystemExit):
        source_acquisition.main(
            ["--catalog", str(path), "--destination", str(tmp_path), "--minimum-free-gb", value],
            session=FakeSession(),
            output=lambda _line: None,
        )


def test_acquisition_layer_does_not_load_audio_profiles_or_english_cli():
    script = (
        "import json, sys\n"
        "import jmds_prepare.source_acquisition\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith("
        "('soundfile', 'jmds_prepare.profiles', 'jmds_prepare.cli')))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )

    assert json.loads(completed.stdout) == []
    assert paths.existing_ancestor.__module__ == "jmds_prepare.core.paths"
    assert estimates.existing_ancestor(Path(__file__)) == Path(__file__).resolve()
