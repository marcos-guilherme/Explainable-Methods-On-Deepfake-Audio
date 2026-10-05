from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from jmds_prepare import audio as audio_module
from jmds_prepare.audio import (
    AUDIO_METADATA_COLUMNS,
    AudioMetadata,
    process_audio,
    process_manifest,
    verify_processed_manifest_row,
)
from jmds_prepare.manifest import MANIFEST_COLUMNS


def _write_audio(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "FLOAT",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate, subtype=subtype)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_row(
    source: Path,
    *,
    utt_id: str = "T_0000000001",
    split: str = "train",
    label: str = "pristine",
) -> dict[str, str]:
    return {
        "utt_id": utt_id,
        "spk_id": "T_0001",
        "gender": "F",
        "language": "eng",
        "dataset": "ASVspoof2024",
        "split": split,
        "label": label,
        "attack_id": "pristine" if label == "pristine" else "A01",
        "source_path": str(source),
        "processed_path": "",
        "sha256_source": _sha256(source),
        "sha256_processed": "",
        "metadata_source": "JMDS+ASVspoof5-verified",
    }


def test_process_audio_resamples_to_pcm16_mono_and_preserves_source(tmp_path):
    source = tmp_path / "raw" / "mono.wav"
    destination = tmp_path / "processed" / "mono.wav"
    samples = np.linspace(-0.8, 0.8, 801, dtype=np.float64)
    _write_audio(source, samples, 8_000)
    source_bytes = source.read_bytes()
    source_times = (source.stat().st_atime_ns, source.stat().st_mtime_ns)

    metadata = process_audio(source, destination, 16_000)

    assert isinstance(metadata, AudioMetadata)
    assert metadata.source_sample_rate == 8_000
    assert metadata.source_channels == 1
    assert metadata.source_frames == 801
    assert metadata.processed_sample_rate == 16_000
    assert metadata.processed_channels == 1
    assert metadata.processed_frames in {1_601, 1_602}
    assert abs(metadata.source_duration_seconds - metadata.processed_duration_seconds) <= (
        1 / 16_000
    )
    assert metadata.sha256_source == hashlib.sha256(source_bytes).hexdigest()
    assert metadata.sha256_processed == _sha256(destination)
    assert metadata.source_peak == pytest.approx(0.8)
    assert metadata.source_rms == pytest.approx(np.sqrt(np.mean(samples**2)))
    assert 0 < metadata.processed_peak <= 1.0
    assert metadata.processed_rms > 0

    info = sf.info(destination)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert (source.stat().st_atime_ns, source.stat().st_mtime_ns) == source_times
    assert source.read_bytes() == source_bytes


def test_process_audio_averages_channels_clips_and_is_idempotent(tmp_path):
    source = tmp_path / "stereo.wav"
    destination = tmp_path / "processed.wav"
    left = np.array([1.5, 0.5, -1.5, -0.5], dtype=np.float64)
    right = np.array([0.5, -0.5, -0.5, 0.5], dtype=np.float64)
    _write_audio(source, np.column_stack([left, right]), 22_050)

    first = process_audio(source, destination, 22_050)
    first_bytes = destination.read_bytes()
    first_mtime = destination.stat().st_mtime_ns
    decoded, rate = sf.read(destination, dtype="float64", always_2d=False)
    second = process_audio(source, destination, 22_050)

    assert rate == 22_050
    assert decoded == pytest.approx(np.array([1.0, 0.0, -1.0, 0.0]), abs=1 / 32_768)
    assert destination.read_bytes() == first_bytes
    assert destination.stat().st_mtime_ns == first_mtime
    assert second == first


def test_process_audio_averages_more_than_two_channels(tmp_path):
    source = tmp_path / "multichannel.wav"
    destination = tmp_path / "output.wav"
    samples = np.array(
        [
            [0.9, 0.3, -0.6],
            [-0.6, 0.0, 0.3],
        ],
        dtype=np.float64,
    )
    _write_audio(source, samples, 16_000)

    metadata = process_audio(source, destination, 16_000)
    decoded, _ = sf.read(destination, dtype="float64")

    assert metadata.source_channels == 3
    assert decoded == pytest.approx(np.array([0.2, -0.1]), abs=1 / 32_768)


def test_process_audio_handles_extreme_finite_values_without_stat_overflow(
    tmp_path,
):
    source = tmp_path / "extreme.wav"
    destination = tmp_path / "output.wav"
    maximum = np.finfo(np.float64).max
    samples = np.array(
        [[maximum, maximum, maximum], [-maximum, -maximum, -maximum]],
        dtype=np.float64,
    )
    _write_audio(source, samples, 16_000, subtype="DOUBLE")

    metadata = process_audio(source, destination, 16_000)
    decoded, _ = sf.read(destination, dtype="float64")

    assert metadata.source_peak == maximum
    assert np.isfinite(metadata.source_rms)
    assert metadata.source_rms == pytest.approx(maximum)
    assert np.isfinite(decoded).all()
    assert decoded == pytest.approx(np.array([1.0, -1.0]), abs=1 / 32_768)


@pytest.mark.parametrize("sample_rate", [0, -1, 16_000.5, True])
def test_process_audio_rejects_invalid_output_sample_rate(tmp_path, sample_rate):
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(8), 8_000)

    with pytest.raises(ValueError, match="sample_rate"):
        process_audio(source, tmp_path / "output.wav", sample_rate)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("empty", "empty"),
        ("nonfinite", "non-finite"),
        ("not_audio", "read audio"),
        ("missing", "read audio"),
    ],
)
def test_process_audio_rejects_invalid_sources(tmp_path, kind, expected):
    source = tmp_path / f"{kind}.wav"
    if kind == "empty":
        _write_audio(source, np.empty(0), 8_000)
    elif kind == "nonfinite":
        _write_audio(source, np.array([0.0, np.nan]), 8_000)
    elif kind == "not_audio":
        source.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        process_audio(source, tmp_path / "output.wav", 16_000)

    assert not (tmp_path / "output.wav").exists()


def test_process_audio_cleans_temp_when_atomic_link_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    destination = tmp_path / "nested" / "output.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)

    def fail_link(source_path: Path, destination_path: Path) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(OSError, match="link failed"):
        process_audio(source, destination, 16_000)

    assert not destination.exists()
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_process_audio_never_overwrites_concurrent_destination(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.wav"
    destination = tmp_path / "output.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    real_link = os.link

    def race_link(candidate: Path, target: Path) -> None:
        _write_audio(destination, np.zeros(160), 16_000, subtype="PCM_16")
        real_link(candidate, target)

    monkeypatch.setattr(os, "link", race_link)

    with pytest.raises(FileExistsError, match="conflict"):
        process_audio(source, destination, 16_000)

    decoded, _ = sf.read(destination, dtype="float64")
    assert np.count_nonzero(decoded) == 0
    assert list(tmp_path.glob(f".{destination.name}.*.tmp.wav")) == []


@pytest.mark.parametrize("failure", ["write", "fsync", "validation"])
def test_process_audio_cleans_temp_on_publication_preparation_failures(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source.wav"
    destination = tmp_path / "output.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)

    if failure == "write":
        monkeypatch.setattr(
            sf.SoundFile,
            "write",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif failure == "fsync":
        monkeypatch.setattr(
            os,
            "fsync",
            lambda *args: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    else:
        monkeypatch.setattr(
            audio_module,
            "_validate_output",
            lambda *args: (_ for _ in ()).throw(ValueError("validation failed")),
        )

    with pytest.raises((OSError, ValueError), match=failure):
        process_audio(source, destination, 16_000)

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp.wav")) == []


def test_process_audio_rejects_conflicting_existing_output(tmp_path):
    source = tmp_path / "source.wav"
    destination = tmp_path / "output.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    _write_audio(destination, np.zeros(160), 16_000, subtype="PCM_16")
    existing = destination.read_bytes()

    with pytest.raises(FileExistsError, match="conflict"):
        process_audio(source, destination, 16_000)

    assert destination.read_bytes() == existing


def test_process_manifest_populates_validated_metadata_and_writes_atomically(
    tmp_path,
):
    data_root = tmp_path / "data"
    pristine = data_root / "raw" / "asvspoof5" / "train" / "T_1.wav"
    generated = tmp_path / "jmds" / "generated" / "D_2.wav"
    _write_audio(pristine, np.linspace(-0.5, 0.5, 80), 8_000)
    _write_audio(generated, np.column_stack([np.ones(100), np.zeros(100)]), 22_050)
    manifest = pd.DataFrame(
        [
            _manifest_row(pristine, utt_id="T_1"),
            _manifest_row(
                generated,
                utt_id="D_2",
                split="dev",
                label="generated",
            ),
        ],
        columns=MANIFEST_COLUMNS,
    )
    manifest_path = data_root / "manifests" / "english_processed.csv"

    result = process_manifest(manifest, data_root, manifest_path, sample_rate=16_000)

    assert list(result.columns) == MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
    assert result["processed_path"].tolist() == [
        str(data_root / "processed" / "english" / "train" / "pristine" / "T_1.wav"),
        str(data_root / "processed" / "english" / "dev" / "generated" / "D_2.wav"),
    ]
    assert result["sha256_processed"].tolist() == [
        _sha256(Path(path)) for path in result["processed_path"]
    ]
    assert set(result["processed_sample_rate"]) == {16_000}
    assert set(result["processed_channels"]) == {1}
    pd.testing.assert_frame_equal(
        pd.read_csv(manifest_path).fillna(""),
        result.fillna(""),
        check_dtype=False,
        check_exact=False,
    )


def test_process_manifest_is_idempotent_with_processed_manifest(tmp_path):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    raw = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    manifest_path = tmp_path / "processed.csv"
    first = process_manifest(raw, data_root, manifest_path)
    output = Path(first.loc[0, "processed_path"])
    output_mtime = output.stat().st_mtime_ns

    second = process_manifest(first, data_root, manifest_path)

    pd.testing.assert_frame_equal(second, first)
    assert output.stat().st_mtime_ns == output_mtime


@pytest.mark.parametrize("collision", ["source", "output", "directory"])
def test_process_manifest_rejects_csv_destination_collisions(
    tmp_path, collision
):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    frame = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    output = (
        data_root
        / "processed"
        / "english"
        / "train"
        / "pristine"
        / "T_0000000001.wav"
    )
    if collision == "source":
        destination = source
    elif collision == "output":
        destination = output
    else:
        destination = output.parent
        destination.mkdir(parents=True)

    with pytest.raises(ValueError, match="manifest destination.*collid"):
        process_manifest(frame, data_root, destination)

    assert source.is_file()
    if collision != "output":
        assert not output.exists()


def test_process_manifest_rejects_nonexistent_ancestor_of_planned_outputs(
    tmp_path,
):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    frame = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    destination = data_root / "processed"

    assert not destination.exists()
    with pytest.raises(ValueError, match="manifest destination.*ancestor"):
        process_manifest(frame, data_root, destination)

    assert not destination.exists()
    assert not (data_root / "processed").exists()


@pytest.mark.parametrize("field", ["processed_path", "sha256_processed"])
def test_raw_manifest_rejects_prepopulated_processed_fields(tmp_path, field):
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    frame = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    frame.loc[0, field] = "already-populated"

    with pytest.raises(ValueError, match="raw manifest.*empty"):
        process_manifest(frame, tmp_path / "data", tmp_path / "manifest.csv")


@pytest.mark.parametrize(
    "field",
    ["processed_path", "sha256_processed", *AUDIO_METADATA_COLUMNS],
)
def test_processed_manifest_rejects_every_tampered_calculated_field(
    tmp_path, field
):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.linspace(-0.5, 0.5, 80), 8_000)
    raw = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    valid = process_manifest(
        raw,
        data_root,
        tmp_path / "first.csv",
        sample_rate=16_000,
    )
    tampered = valid.copy()
    if field == "processed_path":
        tampered.loc[0, field] = str(tmp_path / "other.wav")
    elif field == "sha256_processed":
        tampered.loc[0, field] = "0" * 64
    elif "sample_rate" in field or "channels" in field or "frames" in field:
        tampered.loc[0, field] = int(tampered.loc[0, field]) + 1
    else:
        tampered.loc[0, field] = float(tampered.loc[0, field]) + 0.01

    with pytest.raises(ValueError, match=f"{field}.*diverg"):
        process_manifest(
            tampered,
            data_root,
            tmp_path / "second.csv",
            sample_rate=16_000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata_source", "unverified"),
        ("dataset", "OtherDataset"),
        ("spk_id", ""),
        ("gender", "X"),
        ("attack_id", "A01"),
        ("language", "jpn"),
        ("split", "eval"),
        ("label", "spoof"),
    ],
)
def test_process_manifest_rejects_noncanonical_fields(
    tmp_path, field, value
):
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    frame = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    frame.loc[0, field] = value

    with pytest.raises(ValueError, match=field):
        process_manifest(frame, tmp_path / "data", tmp_path / "manifest.csv")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda frame: frame.assign(utt_id="../escape"), "escapes"),
        (lambda frame: pd.concat([frame, frame], ignore_index=True), "Duplicate"),
        (lambda frame: frame.assign(split="eval"), "split"),
        (lambda frame: frame.assign(label="spoof"), "label"),
        (lambda frame: frame.assign(language="jpn"), "English"),
        (lambda frame: frame.assign(sha256_source="0" * 64), "checksum"),
    ],
)
def test_process_manifest_rejects_invalid_rows_before_writing(
    tmp_path, mutate, expected
):
    source = tmp_path / "source.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    frame = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    manifest_path = tmp_path / "manifest.csv"

    with pytest.raises(ValueError, match=expected):
        process_manifest(mutate(frame), tmp_path / "data", manifest_path)

    assert not manifest_path.exists()


def test_process_manifest_rejects_duplicate_destination_paths(tmp_path):
    source_a = tmp_path / "a.wav"
    source_b = tmp_path / "b.wav"
    _write_audio(source_a, np.ones(80) * 0.1, 8_000)
    _write_audio(source_b, np.ones(80) * 0.2, 8_000)
    rows = [
        _manifest_row(source_a),
        _manifest_row(source_b, utt_id="t_0000000001"),
    ]
    rows[1]["sha256_source"] = _sha256(source_b)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    with pytest.raises(ValueError, match="Duplicate destination"):
        process_manifest(frame, tmp_path / "data", tmp_path / "manifest.csv")


def test_process_manifest_preserves_existing_manifest_when_audio_fails(tmp_path):
    source = tmp_path / "source.wav"
    invalid = tmp_path / "invalid.wav"
    _write_audio(source, np.ones(80) * 0.25, 8_000)
    invalid.write_text("not audio", encoding="utf-8")
    rows = [
        _manifest_row(source, utt_id="T_1"),
        _manifest_row(invalid, utt_id="T_2"),
    ]
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path = tmp_path / "processed.csv"
    manifest_path.write_text("old,manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="read audio"):
        process_manifest(frame, tmp_path / "data", manifest_path)

    assert manifest_path.read_text(encoding="utf-8") == "old,manifest\n"


@pytest.mark.parametrize("field", AUDIO_METADATA_COLUMNS)
def test_verify_processed_manifest_row_rejects_each_finite_metadata_tamper(
    tmp_path, field
):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.linspace(-0.5, 0.5, 80), 8_000)
    raw = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    processed = process_manifest(
        raw,
        data_root,
        tmp_path / "processed.csv",
        sample_rate=16_000,
    )
    tampered = processed.iloc[0].copy()
    if "sample_rate" in field or "channels" in field or "frames" in field:
        tampered[field] = int(tampered[field]) + 1
    else:
        tampered[field] = float(tampered[field]) + 0.01

    with pytest.raises(ValueError, match=f"{field}.*diverges"):
        verify_processed_manifest_row(tampered)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("source_path", "missing-source.wav", "Source audio is missing"),
        ("processed_path", "missing-processed.wav", "Processed audio is missing"),
        ("sha256_source", "0" * 64, "sha256_source diverges"),
        ("sha256_processed", "0" * 64, "sha256_processed diverges"),
    ],
)
def test_verify_processed_manifest_row_checks_paths_and_checksums(
    tmp_path, field, value, expected
):
    data_root = tmp_path / "data"
    source = tmp_path / "source.wav"
    _write_audio(source, np.linspace(-0.5, 0.5, 80), 8_000)
    raw = pd.DataFrame([_manifest_row(source)], columns=MANIFEST_COLUMNS)
    processed = process_manifest(
        raw,
        data_root,
        tmp_path / "processed.csv",
        sample_rate=16_000,
    )
    tampered = processed.iloc[0].copy()
    tampered[field] = value

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        verify_processed_manifest_row(tampered)
