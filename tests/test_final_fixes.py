from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jmds_prepare import cli
from jmds_prepare.audit import (
    _publish_report_set,
    _summary_from_audited,
    audit_and_publish,
)
from jmds_prepare.config import PreparationConfig
from jmds_prepare.extraction import (
    LEDGER_COLUMNS,
    ExtractionLedgerEntry,
    read_extraction_ledger,
    update_extraction_ledger,
)
from jmds_prepare.manifest import MANIFEST_COLUMNS
from jmds_prepare.protocols import (
    ASV_COLUMNS,
    JMDS_COLUMNS,
    protocol_file_sha256,
    read_asvspoof_protocol,
    read_jmds_protocol,
    CorrespondenceReport,
)
from jmds_prepare.zenodo import RecordFile, get_record_metadata


def test_ledger_insert_replace_and_reject_duplicate_rows(tmp_path):
    ledger = tmp_path / "extraction_ledger.csv"
    first = ExtractionLedgerEntry(
        "T_0000000001", "train", "a.tar", "a" * 32, "x.flac", 3, "b" * 64
    )
    replacement = ExtractionLedgerEntry(
        "T_0000000001", "train", "b.tar", "c" * 32, "y.flac", 4, "d" * 64
    )

    update_extraction_ledger(ledger, [first])
    update_extraction_ledger(ledger, [replacement])

    frame = read_extraction_ledger(ledger)
    assert list(frame.columns) == LEDGER_COLUMNS
    assert frame.to_dict("records") == [replacement.as_dict()]
    pd.concat([frame, frame]).to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="Duplicate.*ledger"):
        read_extraction_ledger(ledger)


def test_resume_accepts_only_ledger_bound_flac(tmp_path):
    output = tmp_path / "raw" / "train"
    output.mkdir(parents=True)
    path = output / "T_0000000001.flac"
    sf.write(path, np.zeros(32), 16_000, format="FLAC")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = tmp_path / "ledger.csv"
    entry = ExtractionLedgerEntry(
        path.stem, "train", "a.tar", "a" * 32, "nested/x.flac",
        path.stat().st_size, digest,
    )
    update_extraction_ledger(ledger, [entry])

    assert cli._valid_extracted_ids(
        output, {path.stem}, ledger, {"a.tar": "a" * 32}, "train"
    ) == {path.stem}
    assert cli._valid_extracted_ids(
        output, {path.stem}, ledger, {"a.tar": "f" * 32}, "train"
    ) == set()
    path.write_bytes(b"arbitrary")
    assert cli._valid_extracted_ids(
        output, {path.stem}, ledger, {"a.tar": "a" * 32}, "train"
    ) == set()


def test_protocol_freshness_covers_full_file_bytes_and_eval_is_strict(tmp_path):
    jmds = tmp_path / "eval.csv"
    row = {
        "spk_id": "E_0001", "utt_id": "E_0000000001", "gender": "F",
        "codec": "-", "attack_id": "pristine", "label": "pristine",
        "native": "yes", "language": "eng", "dataset": "ASVspoof2024",
    }
    pd.DataFrame([row], columns=JMDS_COLUMNS).to_csv(jmds, index=False)
    first_hash = protocol_file_sha256(jmds)
    assert read_jmds_protocol(jmds, "eval")["utt_id"].tolist() == [row["utt_id"]]
    jmds.write_bytes(jmds.read_bytes() + b"\n")
    assert protocol_file_sha256(jmds) != first_hash

    asv = tmp_path / "eval.tsv"
    valid = "E_0001 E_0000000001 F - - - - pristine bonafide -"
    asv.write_text(f"{valid}\n{valid}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_asvspoof_protocol(asv, "eval")


def test_record_metadata_preserves_official_values_and_missing_reasons():
    payload = {
        "id": 42,
        "doi": "10.5281/zenodo.42",
        "metadata": {
            "license": {"id": "cc-by-4.0"},
            "version": "5",
            "publication_date": "2024-01-02",
        },
        "links": {"html": "https://zenodo.org/records/42"},
        "files": [
            {
                "key": name,
                "size": 1,
                "checksum": "md5:" + "0" * 32,
                "links": {"self": f"https://example/{name}"},
            }
            for name in cli.REQUIRED_ARCHIVES
        ],
    }

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    metadata = get_record_metadata(42, session=Session())
    assert metadata.record_id == 42
    assert metadata.doi == "10.5281/zenodo.42"
    assert metadata.license == "cc-by-4.0"
    assert metadata.missing_reasons == {}
    payload["metadata"].pop("version")
    assert get_record_metadata(42, session=Session()).version is None
    assert "version" in get_record_metadata(42, session=Session()).missing_reasons


def test_provenance_json_is_atomic_and_records_null_reason(tmp_path):
    config = PreparationConfig(tmp_path / "jmds", tmp_path / "data")
    correspondence = config.data_root / "reports" / "correspondence.json"
    correspondence.parent.mkdir(parents=True)
    correspondence.write_text('{"eval": {"status": "excluded"}}', encoding="utf-8")
    reports = {
        split: CorrespondenceReport(split, 0, 0, 0, 0, (), (), {})
        for split in ("train", "dev")
    }
    record = SimpleNamespace(
        record_id=42,
        doi="10.5281/zenodo.42",
        license="cc-by-4.0",
        version=None,
        publication_date="2024-01-02",
        official_url="https://zenodo.org/records/42",
        missing_reasons={"version": "field absent from Zenodo record API response"},
        files={
            "a.tar": RecordFile("a.tar", 3, "a" * 32, "https://example/a.tar")
        },
    )

    cli._write_provenance(config, record, reports)

    destination = config.data_root / "reports" / "provenance.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["zenodo"]["version"] is None
    assert payload["zenodo"]["missing_reasons"]["version"]
    assert payload["zenodo"]["files"][0]["md5"] == "a" * 32
    assert payload["manifest_field_origins"]["protocol_codec"].startswith("ASV")
    assert payload["manifest_field_origins"]["attack_id"].startswith("JMDS only")
    assert not list(destination.parent.glob(".provenance.json.*.tmp"))


def test_supported_splits_must_be_exactly_train_dev(tmp_path):
    config = PreparationConfig(
        jmds_root=tmp_path, data_root=tmp_path / "data",
        supported_splits=("train",),
    )
    with pytest.raises(ValueError, match="exactly.*train.*dev"):
        config.validate()


def test_new_manifest_schema_has_explicit_protocol_fields():
    assert "protocol_codec" in MANIFEST_COLUMNS
    assert "source_group_id" in MANIFEST_COLUMNS


def test_audit_publishes_three_artifacts_idempotently_with_progress(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(columns=MANIFEST_COLUMNS)
    progress: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "jmds_prepare.audit._build_audited_table",
        lambda _frame, callback=None: (
            callback(0, 0) if callback else None,
            pd.DataFrame(),
        )[1],
    )
    monkeypatch.setattr("jmds_prepare.audit._summary_from_audited", lambda _: {"ok": 1})
    monkeypatch.setattr("jmds_prepare.audit._baseline_from_audited", lambda _: {"ok": 2})

    callback = lambda done, total: progress.append((done, total))
    audit_and_publish(frame, tmp_path, progress=callback)
    audit_and_publish(frame, tmp_path, progress=callback)

    assert json.loads((tmp_path / "audit.json").read_text()) == {"ok": 1}
    assert json.loads((tmp_path / "technical_baseline.json").read_text()) == {"ok": 2}
    assert (tmp_path / "audit.csv").is_file()
    assert progress == [(0, 0), (0, 0)]
    (tmp_path / "audit.json").write_text('{"conflict": true}\n')
    with pytest.raises(FileExistsError, match="conflict"):
        audit_and_publish(frame, tmp_path)


def test_dry_run_storage_uses_real_headers_and_marks_missing_pristine(tmp_path):
    config = PreparationConfig(tmp_path / "jmds", tmp_path / "data")
    generated = (
        config.jmds_root
        / "dataset"
        / "English_ASVspoof2024_Generated"
        / "train"
        / "wav"
        / "T_0000000002.wav"
    )
    generated.parent.mkdir(parents=True)
    sf.write(generated, np.zeros(80), 8_000, subtype="PCM_16")
    pristine = (
        config.data_root / "raw" / "asvspoof5" / "train" / "T_0000000001.flac"
    )
    pristine.parent.mkdir(parents=True)
    sf.write(pristine, np.zeros(160), 16_000, format="FLAC")
    protocol = pd.DataFrame(
        [
            {"utt_id": pristine.stem, "label": "pristine"},
            {"utt_id": generated.stem, "label": "generated"},
            {"utt_id": "T_0000000003", "label": "pristine"},
        ]
    )

    estimate = cli._audio_storage_estimate(
        config, {"train": protocol, "dev": protocol.iloc[0:0]}
    )

    assert estimate["generated"]["files_measured"] == 1
    assert estimate["generated"]["processed_pcm16_bytes"] == 320
    assert estimate["pristine"]["files_measured"] == 1
    assert estimate["pristine"]["files_pending_extraction"] == 1
    assert estimate["pristine"]["status"] == "pending until extraction"


def test_fresh_acquisition_peak_includes_remaining_flac_bound_and_active_tar(
    tmp_path,
):
    files = {
        "a.tar": RecordFile("a.tar", 10, "a" * 32, "https://example/a"),
        "b.tar": RecordFile("b.tar", 20, "b" * 32, "https://example/b"),
        "c.tar": RecordFile("c.tar", 30, "c" * 32, "https://example/c"),
    }

    state = cli._acquisition_storage_state(
        files,
        tmp_path / "archives",
        tmp_path / "raw",
        tmp_path / "ledger.csv",
    )

    assert state["maximum_active_tar_working_set_bytes"] == 30
    assert state["conservative_total_acquisition_peak_bytes"] == 90


def test_partial_acquisition_peak_counts_flacs_partials_and_quarantines(tmp_path):
    files = {
        "a.tar": RecordFile("a.tar", 10, "a" * 32, "https://example/a"),
        "b.tar": RecordFile("b.tar", 20, "b" * 32, "https://example/b"),
        "c.tar": RecordFile("c.tar", 30, "c" * 32, "https://example/c"),
    }
    raw = tmp_path / "raw" / "train"
    raw.mkdir(parents=True)
    (raw / "T_0000000001.flac").write_bytes(b"x" * 10)
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / "b.tar.partial").write_bytes(b"x" * 4)
    (archives / "b.tar.invalid").write_bytes(b"x" * 6)
    (archives / "b.tar.invalid.1").write_bytes(b"x" * 7)
    (archives / "b.tar").write_bytes(b"x" * 3)
    ledger = tmp_path / "ledger.csv"
    update_extraction_ledger(
        ledger,
        [
            ExtractionLedgerEntry(
                "T_0000000001", "train", "a.tar", "a" * 32,
                "T_0000000001.flac", 10, "d" * 64,
            )
        ],
    )

    state = cli._acquisition_storage_state(files, archives, tmp_path / "raw", ledger)

    assert state["extracted_flac_bytes"] == 10
    assert state["current_archive_partial_quarantine_bytes"] == 20
    assert state["remaining_flac_upper_bound_bytes"] == 50
    assert state["largest_active_remaining_tar_bytes"] == 30
    assert state["maximum_active_tar_working_set_bytes"] == 50
    assert state["conservative_total_acquisition_peak_bytes"] == 110


def test_audit_summary_counts_formats_codecs_and_source_groups():
    audited = pd.DataFrame(
        [
            {
                "utt_id": "T_1", "split": "train", "label": "generated",
                "spk_id": "a", "attack_id": "A", "protocol_codec": "opus",
                "source_group_id": "g1", "source_format": "WAV",
                "source_subtype": "PCM_16", "output_format": "WAV",
                "output_subtype": "PCM_16", "sha256_source": "1",
                "sha256_processed": "2",
            },
            {
                "utt_id": "D_1", "split": "dev", "label": "generated",
                "spk_id": "b", "attack_id": "A", "protocol_codec": "opus",
                "source_group_id": "g1", "source_format": "FLAC",
                "source_subtype": "PCM_16", "output_format": "WAV",
                "output_subtype": "PCM_16", "sha256_source": "3",
                "sha256_processed": "4",
            },
            {
                "utt_id": "T_2", "split": "train", "label": "pristine",
                "spk_id": "c", "attack_id": "pristine", "protocol_codec": "",
                "source_group_id": "", "source_format": "FLAC",
                "source_subtype": "PCM_16", "output_format": "WAV",
                "output_subtype": "PCM_16", "sha256_source": "5",
                "sha256_processed": "6",
            },
        ]
    )

    summary = _summary_from_audited(audited)

    assert summary.counts["split_label_protocol_codec"][0] == {
        "split": "dev", "label": "generated", "protocol_codec": "opus", "count": 1
    }
    assert summary.counts["split_label_source_format_source_subtype"]
    assert summary.counts["split_label_output_format_output_subtype"]
    assert summary.cross_split_source_groups == [
        {"source_group_id": "g1", "splits": ["train", "dev"]}
    ]
    assert summary.duplicate_source_groups[0]["source_group_id"] == "g1"
    assert {sample["utt_id"] for sample in summary.duplicate_source_groups[0]["samples"]} == {
        "T_1", "D_1"
    }


def test_report_set_resumes_partial_identical_publication(tmp_path):
    samples = pd.DataFrame([{"utt_id": "T_1"}])
    summary = {"ok": 1}
    baseline = {"ok": 2}
    expected_summary = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    (tmp_path / "audit.json").write_bytes(expected_summary.encode("utf-8"))

    _publish_report_set(summary, samples, baseline, tmp_path)

    assert json.loads((tmp_path / "technical_baseline.json").read_text()) == baseline
    assert pd.read_csv(tmp_path / "audit.csv").to_dict("records") == [{"utt_id": "T_1"}]
    _publish_report_set(summary, samples, baseline, tmp_path)
