from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jmds_prepare.audio import AUDIO_METADATA_COLUMNS
from jmds_prepare.audit import (
    BASELINE_FEATURES,
    SILENCE_THRESHOLD,
    AuditSummary,
    BaselineReport,
    audit_manifest,
    fit_technical_baseline,
)
from jmds_prepare.manifest import MANIFEST_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    tmp_path: Path,
    *,
    utt_id: str,
    speaker: str,
    split: str,
    label: str,
    value: float,
    source_rate: int = 8_000,
    source_channels: int = 1,
) -> dict[str, object]:
    source = tmp_path / "source" / f"{utt_id}.wav"
    processed = tmp_path / "processed" / f"{utt_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    frames = 80 if split == "train" else 96
    samples = np.full((frames, source_channels), value, dtype=np.float64)
    samples[: frames // 4] = 0.0
    sf.write(source, samples, source_rate, subtype="PCM_16")
    sf.write(processed, samples[:, 0], 16_000, subtype="PCM_16")
    source_decoded, _ = sf.read(source, dtype="float64", always_2d=True)
    processed_decoded, _ = sf.read(processed, dtype="float64", always_2d=True)
    return {
        "utt_id": utt_id,
        "spk_id": speaker,
        "gender": "F",
        "language": "eng",
        "dataset": "ASVspoof2024",
        "split": split,
        "label": label,
        "attack_id": "pristine" if label == "pristine" else "A01",
        "source_path": str(source),
        "processed_path": str(processed),
        "sha256_source": _sha256(source),
        "sha256_processed": _sha256(processed),
        "metadata_source": "JMDS+ASVspoof5-verified",
        "source_sample_rate": source_rate,
        "source_channels": source_channels,
        "source_frames": frames,
        "source_duration_seconds": frames / source_rate,
        "source_peak": float(np.max(np.abs(source_decoded))),
        "source_rms": float(np.sqrt(np.mean(source_decoded**2))),
        "processed_sample_rate": 16_000,
        "processed_channels": 1,
        "processed_frames": frames,
        "processed_duration_seconds": frames / 16_000,
        "processed_peak": float(np.max(np.abs(processed_decoded))),
        "processed_rms": float(np.sqrt(np.mean(processed_decoded**2))),
    }


def _manifest(tmp_path: Path) -> pd.DataFrame:
    rows = [
        _row(
            tmp_path,
            utt_id="T_P1",
            speaker="tr_p1",
            split="train",
            label="pristine",
            value=0.05,
        ),
        _row(
            tmp_path,
            utt_id="T_P2",
            speaker="tr_p2",
            split="train",
            label="pristine",
            value=0.08,
        ),
        _row(
            tmp_path,
            utt_id="T_G1",
            speaker="tr_g1",
            split="train",
            label="generated",
            value=0.70,
            source_rate=22_050,
            source_channels=2,
        ),
        _row(
            tmp_path,
            utt_id="T_G2",
            speaker="tr_g2",
            split="train",
            label="generated",
            value=0.80,
            source_rate=22_050,
            source_channels=2,
        ),
        _row(
            tmp_path,
            utt_id="D_P1",
            speaker="dv_p1",
            split="dev",
            label="pristine",
            value=0.06,
        ),
        _row(
            tmp_path,
            utt_id="D_P2",
            speaker="dv_p2",
            split="dev",
            label="pristine",
            value=0.09,
        ),
        _row(
            tmp_path,
            utt_id="D_G1",
            speaker="dv_g1",
            split="dev",
            label="generated",
            value=0.72,
            source_rate=22_050,
            source_channels=2,
        ),
        _row(
            tmp_path,
            utt_id="D_G2",
            speaker="dv_g2",
            split="dev",
            label="generated",
            value=0.82,
            source_rate=22_050,
            source_channels=2,
        ),
    ]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS)


def _record(records: list[dict[str, object]], **expected: object) -> dict[str, object]:
    return next(
        record
        for record in records
        if all(record.get(field) == value for field, value in expected.items())
    )


def test_audit_reports_leakage_duplicates_counts_and_distributions(tmp_path):
    frame = _manifest(tmp_path)
    frame.loc[4, "spk_id"] = frame.loc[0, "spk_id"]
    frame.loc[1, "sha256_source"] = frame.loc[0, "sha256_source"]
    Path(frame.loc[1, "source_path"]).write_bytes(
        Path(frame.loc[0, "source_path"]).read_bytes()
    )
    frame.loc[1, "sha256_processed"] = frame.loc[0, "sha256_processed"]
    Path(frame.loc[1, "processed_path"]).write_bytes(
        Path(frame.loc[0, "processed_path"]).read_bytes()
    )
    frame.loc[1, AUDIO_METADATA_COLUMNS] = frame.loc[
        0, AUDIO_METADATA_COLUMNS
    ].to_numpy()
    output = tmp_path / "audit"

    summary = audit_manifest(frame, output)

    assert isinstance(summary, AuditSummary)
    assert summary.counts["split"] == [
        {"split": "dev", "count": 4},
        {"split": "train", "count": 4},
    ]
    assert _record(
        summary.counts["split_label"],
        split="train",
        label="generated",
    )["count"] == 2
    assert summary.cross_split_speakers == [
        {"left_split": "train", "right_split": "dev", "spk_id": "tr_p1"}
    ]
    duplicate = summary.duplicate_hashes["source"][0]
    assert duplicate["sha256"] == frame.loc[0, "sha256_source"]
    assert duplicate["samples"] == [
        {"utt_id": "T_P1", "split": "train"},
        {"utt_id": "T_P2", "split": "train"},
    ]
    assert summary.duplicate_hashes["processed"][0]["samples"] == duplicate["samples"]
    assert summary.silence_definition["threshold"] == SILENCE_THRESHOLD
    assert _record(
        summary.distributions["source_rms"],
        split="train",
        label="generated",
    )["count"] == 2
    assert _record(
        summary.distributions["processed_silence_ratio"],
        split="dev",
        label="pristine",
    )["mean"] == pytest.approx(0.25)

    payload = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    samples = pd.read_csv(output / "audit.csv")
    assert payload["sample_count"] == 8
    assert samples["utt_id"].tolist() == frame["utt_id"].tolist()
    assert "source_silence_ratio" in samples
    assert not list(output.glob(".*.tmp"))


def test_audit_rejects_missing_technical_values_before_publication(tmp_path):
    frame = _manifest(tmp_path)
    frame.loc[0, "source_rms"] = np.nan
    output = tmp_path / "audit"

    with pytest.raises(ValueError, match="source_rms.*diverges"):
        audit_manifest(frame, output)

    assert not output.exists()


def test_audit_uses_structured_group_records_without_delimiter_collisions(tmp_path):
    frame = _manifest(tmp_path)
    frame.loc[2, ["spk_id", "attack_id"]] = ["speaker|part", "attack"]
    frame.loc[3, ["spk_id", "attack_id"]] = ["speaker", "part|attack"]

    summary = audit_manifest(frame, tmp_path / "audit")

    records = summary.counts["speaker_attack"]
    assert _record(records, speaker="speaker|part", attack="attack")["count"] == 1
    assert _record(records, speaker="speaker", attack="part|attack")["count"] == 1
    assert len(
        [
            record
            for record in records
            if record["speaker"] in {"speaker|part", "speaker"}
        ]
    ) == 2


@pytest.mark.parametrize(
    ("entrypoint", "field"),
    [
        (audit_manifest, "source_rms"),
        (fit_technical_baseline, "processed_peak"),
    ],
)
def test_audit_entrypoints_reject_finite_stale_metadata(
    tmp_path, entrypoint, field
):
    frame = _manifest(tmp_path)
    frame.loc[0, field] = float(frame.loc[0, field]) + 0.01

    with pytest.raises(ValueError, match=f"{field}.*diverges"):
        if entrypoint is audit_manifest:
            entrypoint(frame, tmp_path / "audit")
        else:
            entrypoint(frame)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("missing", "missing"),
        ("source checksum", "(?i:source checksum)"),
        ("processed checksum", "sha256_processed diverges"),
    ],
)
def test_audit_hard_fails_before_publication_on_file_integrity_error(
    tmp_path, failure, expected
):
    frame = _manifest(tmp_path)
    if failure == "missing":
        Path(frame.loc[0, "processed_path"]).unlink()
    elif failure == "source checksum":
        Path(frame.loc[0, "source_path"]).write_bytes(b"tampered")
    else:
        Path(frame.loc[0, "processed_path"]).write_bytes(b"tampered")
    output = tmp_path / "audit"

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match=expected,
    ):
        audit_manifest(frame, output)

    assert not (output / "audit.json").exists()
    assert not (output / "audit.csv").exists()


def test_audit_publication_is_atomic_and_preserves_existing_reports(
    tmp_path, monkeypatch
):
    frame = _manifest(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    json_path = output / "audit.json"
    csv_path = output / "audit.csv"
    json_path.write_text("old-json", encoding="utf-8")
    csv_path.write_text("old-csv", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        audit_manifest(frame, output)

    assert json_path.read_text(encoding="utf-8") == "old-json"
    assert csv_path.read_text(encoding="utf-8") == "old-csv"

    json_path.unlink()
    csv_path.unlink()
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication failed")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", fail_second_link)
    with pytest.raises(OSError, match="publication failed"):
        audit_manifest(frame, output)

    assert not json_path.exists()
    assert not csv_path.exists()
    assert not list(output.glob(".*.tmp"))


def test_audit_rollback_preserves_concurrent_destination_replacement(
    tmp_path, monkeypatch
):
    frame = _manifest(tmp_path)
    output = tmp_path / "audit"
    json_path = output / "audit.json"
    real_link = os.link
    calls = 0

    def replace_first_then_fail(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            json_path.unlink()
            json_path.write_text("concurrent replacement", encoding="utf-8")
            raise OSError("publication failed")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", replace_first_then_fail)

    with pytest.raises(OSError, match="publication failed"):
        audit_manifest(frame, output)

    assert json_path.read_text(encoding="utf-8") == "concurrent replacement"
    assert not (output / "audit.csv").exists()
    assert not list(output.glob(".*.tmp"))


def test_baseline_reports_deterministic_metrics_and_coefficient_order(tmp_path):
    frame = _manifest(tmp_path)

    first = fit_technical_baseline(frame, seed=7)
    second = fit_technical_baseline(frame, seed=7)

    assert isinstance(first, BaselineReport)
    assert first == second
    assert first.feature_order == BASELINE_FEATURES
    assert [item["feature"] for item in first.coefficients] == BASELINE_FEATURES
    assert first.class_counts == {
        "train": {"generated": 2, "pristine": 2},
        "dev": {"generated": 2, "pristine": 2},
    }
    assert first.balanced_accuracy == pytest.approx(1.0)
    assert first.roc_auc == pytest.approx(1.0)
    assert first.f1 == pytest.approx(1.0)
    assert first.seed == 7
    assert first.convergence["converged"] is True
    assert first.convergence["iterations"] >= 1


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda frame: frame.__setitem__("source_rms", np.inf),
            "source_rms.*diverges",
        ),
        (
            lambda frame: frame.__setitem__(
                "spk_id",
                ["shared"] * 5 + ["dev-only"] * 3,
            ),
            "speaker leakage",
        ),
        (
            lambda frame: frame.drop(
                frame[(frame["split"] == "dev") & (frame["label"] == "generated")].index,
                inplace=True,
            ),
            "both classes",
        ),
        (
            lambda frame: frame.__setitem__(
                "utt_id", ["duplicate", "duplicate", *frame["utt_id"].tolist()[2:]]
            ),
            "Duplicate utt_id",
        ),
    ],
)
def test_baseline_refuses_invalid_inputs(tmp_path, mutation, expected):
    frame = _manifest(tmp_path)
    mutation(frame)

    with pytest.raises(ValueError, match=expected):
        fit_technical_baseline(frame)


def test_audit_accepts_only_processed_canonical_manifest(tmp_path):
    frame = _manifest(tmp_path).drop(columns=["processed_rms"])

    with pytest.raises(ValueError, match="processed canonical schema"):
        audit_manifest(frame, tmp_path / "audit")

