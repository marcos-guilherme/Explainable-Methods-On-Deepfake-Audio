"""Tests for content-idempotent artifact-set publication."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jmds_prepare.core.publication import (
    DestinationConflictError,
    publish_artifact_set_idempotent,
)

CSV_BYTES = b"a,b\n1,2\n"
JSON_BYTES = b"{}\n"


def _artifact_paths(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "artifacts"
    return output / "data.csv", output / "meta.json"


def test_idempotent_publication_accepts_identical_content_twice(tmp_path):
    csv_path, json_path = _artifact_paths(tmp_path)
    artifacts = {csv_path: CSV_BYTES, json_path: JSON_BYTES}

    publish_artifact_set_idempotent(artifacts)
    publish_artifact_set_idempotent(artifacts)

    assert csv_path.read_bytes() == CSV_BYTES
    assert json_path.read_bytes() == JSON_BYTES


def test_divergent_existing_destination_prevents_publication(tmp_path):
    csv_path, json_path = _artifact_paths(tmp_path)
    csv_path.parent.mkdir(parents=True)
    csv_path.write_bytes(b"existing,csv\n9,9\n")

    with pytest.raises(DestinationConflictError, match="data.csv"):
        publish_artifact_set_idempotent(
            {csv_path: CSV_BYTES, json_path: JSON_BYTES}
        )

    assert csv_path.read_bytes() == b"existing,csv\n9,9\n"
    assert not json_path.exists()
    assert list(csv_path.parent.glob("*.tmp")) == []


def test_existing_destination_comparison_does_not_use_read_bytes(
    tmp_path, monkeypatch
):
    csv_path, _ = _artifact_paths(tmp_path)
    csv_path.parent.mkdir(parents=True)
    content = b"x" * (2 * 1024 * 1024)
    csv_path.write_bytes(content)

    def reject_read_bytes(_path):
        raise AssertionError("read_bytes loads the complete file")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    publish_artifact_set_idempotent({csv_path: content})


def test_partial_publication_rolls_back_newly_linked_destinations(
    tmp_path, monkeypatch
):
    csv_path, json_path = _artifact_paths(tmp_path)
    real_link = os.link
    calls = 0

    def fail_second_link(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second_link)

    with pytest.raises(OSError, match="publication failed"):
        publish_artifact_set_idempotent(
            {csv_path: CSV_BYTES, json_path: JSON_BYTES}
        )

    assert not csv_path.exists()
    assert not json_path.exists()
    assert not list(csv_path.parent.glob("*.tmp"))


def test_partial_publication_preserves_preexisting_identical_destinations(
    tmp_path, monkeypatch
):
    csv_path, json_path = _artifact_paths(tmp_path)
    csv_path.parent.mkdir(parents=True)
    csv_path.write_bytes(CSV_BYTES)
    real_link = os.link

    def fail_json_link(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
        if Path(destination) == json_path.resolve():
            raise OSError("publication failed")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_json_link)

    with pytest.raises(OSError, match="publication failed"):
        publish_artifact_set_idempotent(
            {csv_path: CSV_BYTES, json_path: JSON_BYTES}
        )

    assert csv_path.read_bytes() == CSV_BYTES
    assert not json_path.exists()
    assert not list(csv_path.parent.glob("*.tmp"))


def test_stale_temporary_does_not_block_publication(tmp_path):
    csv_path, json_path = _artifact_paths(tmp_path)
    csv_path.parent.mkdir(parents=True)
    stale = csv_path.parent / ".data.csv.stale.tmp"
    stale.write_bytes(b"leftover temporary\n")

    publish_artifact_set_idempotent(
        {csv_path: CSV_BYTES, json_path: JSON_BYTES}
    )

    assert csv_path.read_bytes() == CSV_BYTES
    assert json_path.read_bytes() == JSON_BYTES
    assert stale.read_bytes() == b"leftover temporary\n"


def test_duplicate_resolved_destinations_are_rejected(tmp_path):
    csv_path, _ = _artifact_paths(tmp_path)
    alias = csv_path.parent / ".." / csv_path.parent.name / csv_path.name

    with pytest.raises(ValueError, match="Duplicate resolved destination"):
        publish_artifact_set_idempotent(
            {
                csv_path: CSV_BYTES,
                alias: JSON_BYTES,
            }
        )

    assert not csv_path.exists()


def test_empty_artifact_set_is_a_no_op(tmp_path):
    publish_artifact_set_idempotent({})
    assert list(tmp_path.iterdir()) == []
