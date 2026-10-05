"""Tests for brspeech_xai.data loaders and registry."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from brspeech_xai.data import (
    XAI_SAMPLE_COLUMNS,
    build_balanced_split,
    decode_audio,
    list_dataset_kinds,
)

# Canonical schema mirrored from jmds_prepare.core.xai_sample (no runtime import).
_EXPECTED_XAI_COLUMNS = [
    "sample_id",
    "language",
    "role",
    "label",
    "corpus",
    "native_split",
    "original_ref",
    "processed_path",
    "speaker_id",
    "group_id",
    "attack_id",
    "sha256_source",
    "sha256_processed",
    "selection_seed",
    "selection_rank",
    "selection_reason",
    "selection_source",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_wav(path: Path, *, sr: int = 16000, n_samples: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = np.linspace(-0.1, 0.1, n_samples, dtype=np.float32)
    sf.write(path, sig, sr, subtype="PCM_16")


def _valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "sample_id": "eng-train-bonafide-00001",
        "language": "eng",
        "role": "train",
        "label": "0",
        "corpus": "ASVspoof2024",
        "native_split": "train",
        "original_ref": "raw/asvspoof5/train/T_0000000001.flac",
        "processed_path": "wav/bonafide_01.wav",
        "speaker_id": "T_0001",
        "group_id": "",
        "attack_id": "",
        "sha256_source": "a" * 64,
        "sha256_processed": "",
        "selection_seed": "42",
        "selection_rank": "0",
        "selection_reason": "paired_train_quota",
        "selection_source": "jmds_protocol:open_v2_train.cm.csv",
    }
    row.update(overrides)
    return row


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    *,
    audio_dir: Path | None = None,
    compute_hashes: bool = True,
    bundle_root: Path | None = None,
    create_receipt: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = path.parent
    bundle_root = bundle_root or manifest_dir
    if create_receipt:
        (bundle_root / "bundle_receipt.json").write_text("{}", encoding="utf-8")
    prepared: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        rel = item["processed_path"]
        wav_path = (
            (bundle_root / rel).resolve()
            if not Path(rel).is_absolute()
            else Path(rel)
        )
        if audio_dir is not None:
            src = audio_dir / Path(rel).name
            _write_wav(src)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(src.read_bytes())
        if compute_hashes and wav_path.is_file():
            item["sha256_processed"] = _sha256_file(wav_path)
        prepared.append(item)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=XAI_SAMPLE_COLUMNS)
        writer.writeheader()
        for item in prepared:
            writer.writerow(item)


def _balanced_train_rows(prefix: str = "eng", *, n_per_class: int = 3) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, tag in ((0, "bonafide"), (1, "spoof")):
        for rank in range(n_per_class):
            rows.append(
                _valid_row(
                    sample_id=f"{prefix}-train-{tag}-{rank:05d}",
                    role="train",
                    label=str(label),
                    processed_path=f"wav/{tag}_{rank:02d}.wav",
                    selection_rank=str(rank),
                    attack_id="" if label == 0 else f"A{rank:03d}",
                )
            )
    return rows


def test_xai_sample_columns_match_jmds_prepare_contract():
    assert XAI_SAMPLE_COLUMNS == _EXPECTED_XAI_COLUMNS


def test_decode_audio_from_bytes():
    sig = np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, 16000, format="WAV")
    wav, sr = decode_audio({"bytes": buf.getvalue(), "path": None})
    assert sr == 16000
    assert wav.shape[0] == 8000


def test_registry_exposes_known_kinds():
    kinds = list_dataset_kinds()
    assert "hf_brspeech" in kinds
    assert "local_manifest" in kinds


def test_unknown_dataset_kind_raises():
    with pytest.raises(ValueError, match="dataset_kind desconhecido"):
        build_balanced_split(
            "AKCIT-Deepfake/BRSpeech-DF",
            "train",
            2,
            dataset_kind="not_a_real_kind",
        )


@patch("brspeech_xai.data._collect_stream")
def test_hf_legacy_default_kind_uses_stream_backend(mock_stream):
    wav = np.zeros(8, dtype=np.float32)
    prov = {"config": "default", "shard": None, "row_index": None, "label": 0}
    mock_stream.return_value = ([wav], [16000], [0], [prov])

    audios, srs, labels, provenance = build_balanced_split(
        "AKCIT-Deepfake/BRSpeech-DF",
        "train",
        1,
        loader="stream",
        seed=7,
    )

    mock_stream.assert_called_once_with("AKCIT-Deepfake/BRSpeech-DF", "train", 1)
    assert len(audios) == 1
    assert srs == [16000]
    assert labels == [0]
    assert provenance[0]["config"] == "default"


@patch("brspeech_xai.data._collect_stream")
def test_hf_legacy_explicit_kind_matches_default(mock_stream):
    wav = np.zeros(8, dtype=np.float32)
    mock_stream.return_value = ([wav], [16000], [0], [{"config": "default", "shard": None,
                                                        "row_index": None, "label": 0}])

    build_balanced_split(
        "AKCIT-Deepfake/BRSpeech-DF",
        "train",
        1,
        loader="stream",
        seed=7,
        dataset_kind="hf_brspeech",
    )

    mock_stream.assert_called_once()


def test_local_manifest_loads_role_filters_and_quotas(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    _write_manifest(manifest, _balanced_train_rows(n_per_class=4), audio_dir=tmp_path / "audio")

    audios, srs, labels, prov = build_balanced_split(
        "",
        "train",
        2,
        dataset_kind="local_manifest",
        manifest_path=str(manifest),
        seed=11,
    )

    assert len(audios) == 4
    assert len(srs) == 4
    assert labels.count(0) == 2
    assert labels.count(1) == 2
    assert all(isinstance(a, np.ndarray) and a.dtype == np.float32 for a in audios)
    assert {p["sample_id"] for p in prov} == {
        "eng-train-bonafide-00000",
        "eng-train-bonafide-00001",
        "eng-train-spoof-00000",
        "eng-train-spoof-00001",
    }
    for entry in prov:
        assert entry["role"] == "train"
        assert entry["manifest_path"] == str(manifest.resolve())
        assert "manifest_row" in entry
        assert entry["selection_seed"] == 42
        assert entry["label"] in (0, 1)
        assert entry["native_split"] == "train"
        assert entry["sha256_source"] == "a" * 64
        assert len(entry["sha256_processed"]) == 64


def test_local_manifest_selection_is_deterministic_by_rank(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=3)
    rows[2]["selection_rank"] = "5"
    rows[5]["selection_rank"] = "5"
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")

    _, _, _, prov_a = build_balanced_split(
        "", "train", 2, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=0,
    )
    _, _, _, prov_b = build_balanced_split(
        "", "train", 2, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=0,
    )
    assert [p["sample_id"] for p in prov_a] == [p["sample_id"] for p in prov_b]


def test_local_manifest_resolves_relative_paths_from_bundle_root(tmp_path):
    root = tmp_path / "bundle"
    manifest = root / "manifests" / "eng" / "xai_samples.csv"
    rel = "data/eng/train/bonafide/clip.wav"
    rows = [
        _valid_row(sample_id="eng-train-bonafide-00001", processed_path=rel),
        _valid_row(
            sample_id="eng-train-spoof-00001",
            label="1",
            processed_path="data/eng/train/spoof/clip.wav",
            attack_id="A001",
            selection_rank="1",
        ),
    ]
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio", bundle_root=root)

    audios, srs, _, prov = build_balanced_split(
        "", "train", 1, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=3,
    )
    assert len(audios) == 2
    assert all(sr == 16000 for sr in srs)
    assert {Path(item["processed_path"]).resolve() for item in prov} == {
        (root / row["processed_path"]).resolve() for row in rows
    }


def test_local_manifest_rejects_relative_paths_without_bundle_receipt(tmp_path):
    manifest = tmp_path / "manifests" / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    _write_manifest(
        manifest,
        rows,
        audio_dir=tmp_path / "audio",
        create_receipt=False,
    )

    with pytest.raises(ValueError, match="bundle_receipt.json"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=3,
        )


def test_local_manifest_rejects_relative_path_traversal(tmp_path):
    root = tmp_path / "bundle"
    manifest = root / "manifests" / "eng" / "xai_samples.csv"
    outside = tmp_path / "outside"
    bonafide = outside / "bonafide.wav"
    spoof = outside / "spoof.wav"
    _write_wav(bonafide)
    _write_wav(spoof)
    rows = [
        _valid_row(
            sample_id="eng-train-bonafide-00001",
            processed_path="../outside/bonafide.wav",
            sha256_processed=_sha256_file(bonafide),
        ),
        _valid_row(
            sample_id="eng-train-spoof-00001",
            label="1",
            processed_path="../outside/spoof.wav",
            sha256_processed=_sha256_file(spoof),
            attack_id="A001",
            selection_rank="1",
        ),
    ]
    _write_manifest(manifest, rows, compute_hashes=False, bundle_root=root)

    with pytest.raises(ValueError, match="bundle root"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=3,
        )


def test_local_manifest_preserves_absolute_processed_path(tmp_path):
    bonafide = tmp_path / "abs_bonafide.wav"
    spoof = tmp_path / "abs_spoof.wav"
    _write_wav(bonafide)
    _write_wav(spoof)
    manifest = tmp_path / "xai_samples.csv"
    rows = [
        _valid_row(
            sample_id="eng-train-bonafide-00001",
            processed_path=str(bonafide.resolve()),
            sha256_processed=_sha256_file(bonafide),
        ),
        _valid_row(
            sample_id="eng-train-spoof-00001",
            label="1",
            processed_path=str(spoof.resolve()),
            sha256_processed=_sha256_file(spoof),
            attack_id="A001",
            selection_rank="1",
        ),
    ]
    _write_manifest(
        manifest,
        rows,
        compute_hashes=False,
        create_receipt=False,
    )

    _, _, _, provenance = build_balanced_split(
        "", "train", 1, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=1,
    )

    assert {Path(item["processed_path"]) for item in provenance} == {
        bonafide.resolve(),
        spoof.resolve(),
    }


def test_local_manifest_rejects_duplicate_processed_path(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    rows[1]["processed_path"] = rows[0]["processed_path"]
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")

    with pytest.raises(ValueError, match="processed_path duplicado"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_n_per_class_zero_uses_balanced_all(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=3)
    rows.pop()  # remove one spoof -> 3 bonafide, 2 spoof
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")

    _, _, labels, _ = build_balanced_split(
        "", "train", 0, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=5,
    )
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_local_manifest_rejects_wrong_column_order(tmp_path):
    manifest = tmp_path / "bad.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(reversed(XAI_SAMPLE_COLUMNS)))
        writer.writerow(["x"] * len(XAI_SAMPLE_COLUMNS))
    with pytest.raises(ValueError, match="Column order"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_duplicate_sample_id(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=2)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    with pytest.raises(ValueError, match="sample_id duplicado"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_missing_file(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    for row in rows:
        row["sha256_processed"] = "b" * 64
    _write_manifest(manifest, rows, compute_hashes=False)
    with pytest.raises(ValueError, match="arquivo ausente"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_tampered_hash(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    tampered = []
    for row in rows:
        item = dict(row)
        item["sha256_processed"] = "c" * 64
        tampered.append(item)
    _write_manifest(manifest, tampered, compute_hashes=False)
    with pytest.raises(ValueError, match="sha256_processed"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_noncanonical_wav(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    bad_wav = manifest.parent / rows[0]["processed_path"]
    stereo = np.zeros((800, 2), dtype=np.float32)
    sf.write(bad_wav, stereo, 44100, subtype="PCM_16")
    digest = _sha256_file(bad_wav)
    good_wav = manifest.parent / rows[1]["processed_path"]
    updated = [
        dict(rows[0], sha256_processed=digest),
        dict(rows[1], sha256_processed=_sha256_file(good_wav)),
    ]
    _write_manifest(manifest, updated, compute_hashes=False)
    with pytest.raises(ValueError, match="16 kHz mono"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


@patch("brspeech_xai.data._collect_stream")
@pytest.mark.parametrize("bad_n", [0, None, -1])
def test_hf_brspeech_rejects_non_positive_n_per_class(mock_stream, bad_n):
    with pytest.raises(ValueError, match="n_per_class"):
        build_balanced_split(
            "AKCIT-Deepfake/BRSpeech-DF",
            "train",
            bad_n,
            dataset_kind="hf_brspeech",
        )
    mock_stream.assert_not_called()


def test_local_manifest_rejects_invalid_language_on_any_row(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    rows.append(
        _valid_row(
            sample_id="eng-calibration-bonafide-00001",
            role="calibration",
            processed_path="wav/calib_bonafide.wav",
            selection_rank="9",
        )
    )
    rows[-1]["language"] = "deu"
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    with pytest.raises(ValueError, match="Unsupported language"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_uppercase_hash_on_unselected_row(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    rows.append(
        _valid_row(
            sample_id="eng-test-bonafide-00001",
            role="test",
            processed_path="wav/test_bonafide.wav",
            selection_rank="9",
            sha256_source="A" * 64,
        )
    )
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    with pytest.raises(ValueError, match="sha256_source"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_resolves_bundle_root_relative_path_outside_manifest_dir(tmp_path):
    # Layout publicado: manifests/eng/ e data/ sob a mesma raiz.
    root = tmp_path / "published"
    data_dir = root / "data"
    manifest = root / "manifests" / "eng" / "xai_samples.csv"
    bonafide = data_dir / "bonafide.wav"
    spoof = data_dir / "spoof.wav"
    _write_wav(bonafide)
    _write_wav(spoof)
    rows = [
        _valid_row(
            sample_id="eng-train-bonafide-00001",
            processed_path="data/bonafide.wav",
            sha256_processed=_sha256_file(bonafide),
        ),
        _valid_row(
            sample_id="eng-train-spoof-00001",
            label="1",
            processed_path="data/spoof.wav",
            sha256_processed=_sha256_file(spoof),
            attack_id="A001",
            selection_rank="1",
        ),
    ]
    _write_manifest(manifest, rows, compute_hashes=False, bundle_root=root)
    _, _, _, prov = build_balanced_split(
        "", "train", 1, dataset_kind="local_manifest",
        manifest_path=str(manifest), seed=3,
    )
    resolved = {Path(p["processed_path"]).resolve() for p in prov}
    assert resolved == {bonafide.resolve(), spoof.resolve()}


def test_local_manifest_rejects_pcm24_wav(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    bad_wav = manifest.parent / rows[0]["processed_path"]
    good_wav = manifest.parent / rows[1]["processed_path"]
    sig = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    sf.write(bad_wav, sig, 16000, subtype="PCM_24")
    updated = [
        dict(rows[0], sha256_processed=_sha256_file(bad_wav)),
        dict(rows[1], sha256_processed=_sha256_file(good_wav)),
    ]
    _write_manifest(manifest, updated, compute_hashes=False)
    with pytest.raises(ValueError, match="PCM_16"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_float_wav(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = _balanced_train_rows(n_per_class=1)
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    bad_wav = manifest.parent / rows[0]["processed_path"]
    good_wav = manifest.parent / rows[1]["processed_path"]
    sig = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
    sf.write(bad_wav, sig, 16000, subtype="FLOAT")
    updated = [
        dict(rows[0], sha256_processed=_sha256_file(bad_wav)),
        dict(rows[1], sha256_processed=_sha256_file(good_wav)),
    ]
    _write_manifest(manifest, updated, compute_hashes=False)
    with pytest.raises(ValueError, match="PCM_16"):
        build_balanced_split(
            "", "train", 1, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )


def test_local_manifest_rejects_insufficient_class(tmp_path):
    manifest = tmp_path / "xai_samples.csv"
    rows = [_valid_row(sample_id="eng-train-bonafide-00001")]
    _write_manifest(manifest, rows, audio_dir=tmp_path / "audio")
    with pytest.raises(ValueError, match="amostras insuficientes"):
        build_balanced_split(
            "", "train", 2, dataset_kind="local_manifest",
            manifest_path=str(manifest), seed=1,
        )
