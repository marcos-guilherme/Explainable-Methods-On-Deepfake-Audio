from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.config import PreparationConfig
from jmds_prepare.manifest import (
    MANIFEST_COLUMNS,
    build_raw_manifest,
    write_manifest_atomic,
)
from jmds_prepare.protocols import (
    ASV_COLUMNS,
    CorrespondenceReport,
    JMDS_COLUMNS,
    protocol_fingerprint,
    validate_correspondence,
)


def _row(utt_id: str, label: str, *, split: str) -> dict[str, str]:
    prefix = "T" if split == "train" else "D"
    return {
        "spk_id": f"{prefix}_0001",
        "utt_id": utt_id,
        "gender": "F",
        "codec": "-",
        "attack_id": "pristine" if label == "pristine" else "A01",
        "label": label,
        "native": "yes",
        "language": "eng",
        "dataset": "ASVspoof2024",
    }


def _report(split: str, rows: list[dict[str, str]]) -> CorrespondenceReport:
    jmds = pd.DataFrame(rows, columns=JMDS_COLUMNS)
    asv_rows = [
        {
            "spk_id": row["spk_id"],
            "utt_id": row["utt_id"],
            "gender": row["gender"],
            "codec": row["codec"],
            "codec_q": "-",
            "codec_seed": "-",
            "attack_tag": "-",
            "attack_id": row["attack_id"],
            "key": "bonafide" if row["label"] == "pristine" else "spoof",
            "tmp": "-",
        }
        for row in rows
    ]
    asv = pd.DataFrame(asv_rows, columns=ASV_COLUMNS)
    return validate_correspondence(jmds, asv, split)


def _setup(
    tmp_path: Path,
    split_rows: dict[str, list[dict[str, str]]],
) -> tuple[PreparationConfig, list[CorrespondenceReport], dict[str, bytes]]:
    jmds_root = tmp_path / "jmds"
    protocols = jmds_root / "cm_protocols"
    protocols.mkdir(parents=True)
    data_root = tmp_path / "data"
    payloads: dict[str, bytes] = {}

    for split, rows in split_rows.items():
        pd.DataFrame(rows, columns=JMDS_COLUMNS).to_csv(
            protocols / f"open_v2_{split}.cm.csv", index=False
        )
        for row in rows:
            if row["label"] == "pristine":
                source = (
                    data_root
                    / "raw"
                    / "asvspoof5"
                    / split
                    / f"{row['utt_id']}.flac"
                )
            else:
                source = (
                    jmds_root
                    / "dataset"
                    / "English_ASVspoof2024_Generated"
                    / split
                    / "wav"
                    / f"{row['utt_id']}.wav"
                )
            source.parent.mkdir(parents=True, exist_ok=True)
            payload = f"audio:{split}:{row['utt_id']}".encode()
            source.write_bytes(payload)
            payloads[row["utt_id"]] = payload

    config = PreparationConfig(
        jmds_root=jmds_root,
        data_root=data_root,
        supported_splits=tuple(split_rows),
    )
    reports = [_report(split, rows) for split, rows in split_rows.items()]
    return config, reports, payloads


def test_builds_pristine_and_generated_rows_with_exact_schema_and_hashes(tmp_path):
    rows = [
        _row("T_0000000001", "pristine", split="train"),
        _row("T_0000000002", "generated", split="train"),
    ]
    config, reports, payloads = _setup(tmp_path, {"train": rows})

    manifest = build_raw_manifest(config, reports)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest["utt_id"].tolist() == ["T_0000000002", "T_0000000001"]
    by_id = manifest.set_index("utt_id")
    assert Path(by_id.loc["T_0000000001", "source_path"]) == (
        config.data_root
        / "raw"
        / "asvspoof5"
        / "train"
        / "T_0000000001.flac"
    )
    assert Path(by_id.loc["T_0000000002", "source_path"]) == (
        config.jmds_root
        / "dataset"
        / "English_ASVspoof2024_Generated"
        / "train"
        / "wav"
        / "T_0000000002.wav"
    )
    for utt_id, payload in payloads.items():
        assert by_id.loc[utt_id, "sha256_source"] == hashlib.sha256(
            payload
        ).hexdigest()
    assert set(manifest["processed_path"]) == {""}
    assert set(manifest["sha256_processed"]) == {""}
    assert set(manifest["metadata_source"]) == {"JMDS+ASVspoof5-verified"}


def test_build_filters_non_english_rows_before_duplicate_validation(tmp_path):
    rows = [_row("T_0000000001", "generated", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    protocol_path = (
        config.jmds_root / "cm_protocols" / "open_v2_train.cm.csv"
    )
    protocol = pd.read_csv(protocol_path, dtype=str)
    reused = protocol.copy()
    reused.loc[:, "language"] = "jpn"
    reused.loc[:, "dataset"] = "MLAAD"
    pd.concat([reused, protocol], ignore_index=True).to_csv(
        protocol_path, index=False
    )

    manifest = build_raw_manifest(config, reports)

    assert manifest["utt_id"].tolist() == ["T_0000000001"]
    assert manifest["language"].tolist() == ["eng"]


def test_orders_by_train_then_dev_label_and_utterance_id(tmp_path):
    train = [
        _row("T_0000000003", "pristine", split="train"),
        _row("T_0000000002", "generated", split="train"),
        _row("T_0000000001", "generated", split="train"),
    ]
    dev = [
        _row("D_0000000002", "pristine", split="dev"),
        _row("D_0000000001", "generated", split="dev"),
    ]
    config, reports, _ = _setup(tmp_path, {"dev": dev, "train": train})

    manifest = build_raw_manifest(config, reversed(reports))

    assert manifest["utt_id"].tolist() == [
        "T_0000000001",
        "T_0000000002",
        "T_0000000003",
        "D_0000000001",
        "D_0000000002",
    ]


def test_rejects_missing_source_file(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    source = (
        config.data_root
        / "raw"
        / "asvspoof5"
        / "train"
        / "T_0000000001.flac"
    )
    source.unlink()

    with pytest.raises(FileNotFoundError, match="T_0000000001"):
        build_raw_manifest(config, reports)


def test_rejects_duplicate_ids_globally(tmp_path):
    rows = [
        _row("T_0000000001", "pristine", split="train"),
        _row("T_0000000002", "generated", split="train"),
    ]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    protocol = config.jmds_root / "cm_protocols" / "open_v2_train.cm.csv"
    frame = pd.read_csv(protocol, dtype=str)
    frame.loc[1, "utt_id"] = "T_0000000001"
    frame.to_csv(protocol, index=False)

    with pytest.raises(ValueError, match="Duplicate.*T_0000000001"):
        build_raw_manifest(config, reports)


def test_rejects_duplicate_id_across_splits_in_manifest_control(
    tmp_path, monkeypatch
):
    train = [_row("T_0000000001", "pristine", split="train")]
    dev = [_row("D_0000000001", "pristine", split="dev")]
    config, reports, _ = _setup(tmp_path, {"train": train, "dev": dev})
    from jmds_prepare import manifest as manifest_module

    duplicated = pd.DataFrame(train, columns=JMDS_COLUMNS)
    reports[1].jmds_fingerprint = protocol_fingerprint(duplicated, "dev")
    monkeypatch.setattr(
        manifest_module,
        "read_jmds_protocol",
        lambda path, split, **kwargs: duplicated.copy(),
    )

    with pytest.raises(ValueError, match="Duplicate utt_id across manifest"):
        build_raw_manifest(config, reports)


def test_rejects_duplicated_resolved_paths(tmp_path, monkeypatch):
    rows = [
        _row("T_0000000001", "generated", split="train"),
        _row("T_0000000002", "generated", split="train"),
    ]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    from jmds_prepare import manifest as manifest_module

    shared = (
        config.jmds_root
        / "dataset"
        / "English_ASVspoof2024_Generated"
        / "train"
        / "wav"
        / "T_0000000001.wav"
    )
    expected_root = shared.parent
    monkeypatch.setattr(
        manifest_module, "_source_path", lambda *args: (shared, expected_root)
    )

    with pytest.raises(ValueError, match="Duplicate.*source path"):
        build_raw_manifest(config, reports)


@pytest.mark.parametrize("label", ["spoof", "bonafide", "unknown"])
def test_rejects_unsupported_labels(tmp_path, label):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    protocol = config.jmds_root / "cm_protocols" / "open_v2_train.cm.csv"
    frame = pd.read_csv(protocol, dtype=str)
    frame.loc[0, "label"] = label
    frame.to_csv(protocol, index=False)

    with pytest.raises(ValueError, match="Unsupported.*label"):
        build_raw_manifest(config, reports)


def test_rejects_unsupported_split(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    reports[0].split = "eval"

    with pytest.raises(ValueError, match="Unsupported.*eval"):
        build_raw_manifest(config, reports)


def test_rejects_report_with_correspondence_errors(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    reports[0].missing_ids = ("T_0000000001",)

    with pytest.raises(ValueError, match="Correspondence errors"):
        build_raw_manifest(config, reports)


def test_rejects_report_with_incomplete_matched_counts(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    reports[0].pristine_matched = 0

    with pytest.raises(ValueError, match="matched count"):
        build_raw_manifest(config, reports)


def test_rejects_legacy_report_without_fingerprints(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, _, _ = _setup(tmp_path, {"train": rows})
    legacy_report = CorrespondenceReport(
        "train",
        1,
        0,
        1,
        0,
        (),
        (),
        {},
    )

    with pytest.raises(ValueError, match="missing.*fingerprint.*JMDS.*ASV"):
        build_raw_manifest(config, [legacy_report])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("utt_id", "T_0000000009"),
        ("spk_id", "T_9999"),
        ("gender", "M"),
    ],
)
def test_rejects_stale_report_when_protocol_content_changes(
    tmp_path, field, value
):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    protocol = config.jmds_root / "cm_protocols" / "open_v2_train.cm.csv"
    frame = pd.read_csv(protocol, dtype=str)
    frame.loc[0, field] = value
    frame.to_csv(protocol, index=False)

    with pytest.raises(ValueError, match="(?i)stale.*fingerprint"):
        build_raw_manifest(config, reports)


def test_requires_one_report_for_each_configured_split(tmp_path):
    train = [_row("T_0000000001", "pristine", split="train")]
    dev = [_row("D_0000000001", "generated", split="dev")]
    config, reports, _ = _setup(tmp_path, {"train": train, "dev": dev})

    with pytest.raises(ValueError, match="Missing correspondence report.*dev"):
        build_raw_manifest(config, reports[:1])


def test_rejects_path_escape_before_reading_source(tmp_path, monkeypatch):
    rows = [_row("T_0000000001", "generated", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    from jmds_prepare import manifest as manifest_module

    escaped = config.jmds_root.parent / "escaped.wav"
    escaped.write_bytes(b"escape")
    expected_root = (
        config.jmds_root
        / "dataset"
        / "English_ASVspoof2024_Generated"
        / "train"
        / "wav"
    )
    monkeypatch.setattr(
        manifest_module, "_source_path", lambda *args: (escaped, expected_root)
    )

    with pytest.raises(ValueError, match="escapes expected root"):
        build_raw_manifest(config, reports)


def test_hashes_the_resolved_source_path(tmp_path, monkeypatch):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    from jmds_prepare import manifest as manifest_module

    root = config.data_root / "raw" / "asvspoof5" / "train"
    (root / "nested").mkdir()
    source_with_parent_segment = (
        root / "nested" / ".." / "T_0000000001.flac"
    )
    hashed_paths: list[Path] = []

    def record_hash(path: Path) -> str:
        hashed_paths.append(path)
        return "0" * 64

    monkeypatch.setattr(
        manifest_module,
        "_source_path",
        lambda *args: (source_with_parent_segment, root),
    )
    monkeypatch.setattr(manifest_module, "_sha256", record_hash)

    build_raw_manifest(config, reports)

    assert hashed_paths == [source_with_parent_segment.resolve()]


def test_atomic_write_flushes_syncs_and_replaces_destination(tmp_path, monkeypatch):
    destination = tmp_path / "manifests" / "raw.csv"
    destination.parent.mkdir()
    destination.write_text("old,data\n", encoding="utf-8")
    frame = pd.DataFrame([["T_1"]], columns=["utt_id"])
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)

    write_manifest_atomic(frame, destination)

    assert pd.read_csv(destination).to_dict("records") == [{"utt_id": "T_1"}]
    assert calls
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_serialized_csv_preserves_empty_processed_fields(tmp_path):
    rows = [_row("T_0000000001", "pristine", split="train")]
    config, reports, _ = _setup(tmp_path, {"train": rows})
    destination = tmp_path / "raw.csv"

    write_manifest_atomic(build_raw_manifest(config, reports), destination)

    with destination.open(newline="", encoding="utf-8") as manifest_file:
        serialized = next(csv.DictReader(manifest_file))
    assert serialized["processed_path"] == ""
    assert serialized["sha256_processed"] == ""


def test_atomic_write_removes_temp_when_csv_serialization_fails(
    tmp_path, monkeypatch
):
    destination = tmp_path / "raw.csv"
    destination.write_text("old,data\n", encoding="utf-8")
    frame = pd.DataFrame([["T_1"]], columns=["utt_id"])

    def fail_to_csv(*args, **kwargs) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(frame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="write failed"):
        write_manifest_atomic(frame, destination)

    assert destination.read_text(encoding="utf-8") == "old,data\n"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_removes_temp_when_fsync_fails(tmp_path, monkeypatch):
    destination = tmp_path / "raw.csv"
    destination.write_text("old,data\n", encoding="utf-8")
    frame = pd.DataFrame([["T_1"]], columns=["utt_id"])

    def fail_fsync(fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        write_manifest_atomic(frame, destination)

    assert destination.read_text(encoding="utf-8") == "old,data\n"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_atomic_write_removes_temp_file_when_replace_fails(tmp_path, monkeypatch):
    destination = tmp_path / "raw.csv"
    destination.write_text("old,data\n", encoding="utf-8")
    frame = pd.DataFrame([["T_1"]], columns=["utt_id"])

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_manifest_atomic(frame, destination)

    assert destination.read_text(encoding="utf-8") == "old,data\n"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []
