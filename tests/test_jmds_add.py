from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE, MandarinProfile
from jmds_prepare.sources.jmds import JMDS_COLUMNS
from jmds_prepare.sources.jmds_add import (
    _validate_resolved_audio_paths,
    read_mandarin_generated,
)

FIXTURES = Path(__file__).parent / "fixtures"
JMDS_FIXTURE = FIXTURES / "jmds_add_metadata.csv"


def _profile_with_train_count(count: int) -> MandarinProfile:
    return replace(
        MANDARIN_PROFILE,
        expected_generated_counts={"train": count, "dev": 7497, "eval": 9999},
    )


def _assert_mandarin_generated_train_output(
    frame: pd.DataFrame, *, jmds_root: Path
) -> None:
    assert len(frame) == 2
    assert set(frame["language"]) == {"zho"}
    assert set(frame["label"]) == {"generated"}
    assert set(frame["dataset"]) == {"ADD"}
    assert frame["utt_id"].tolist() == ["T_0000123456", "T_0000654321"]
    assert frame["utt_id"].is_unique
    assert frame["audio_path"].map(Path).map(Path.is_file).all()
    for field, expected in [
        ("attack_id", "unk"),
        ("gender", "unk"),
        ("spk_id", "unk"),
        ("codec", "-"),
        ("native", "yes"),
    ]:
        assert (frame[field] == expected).all()
    expected_paths = {
        str(MANDARIN_PROFILE.generated_wav_path(jmds_root, "train", utt_id))
        for utt_id in ("T_0000123456", "T_0000654321")
    }
    assert set(frame["audio_path"]) == expected_paths


@pytest.fixture
def jmds_root(tmp_path: Path) -> Path:
    root = tmp_path / "jmds"
    for utt_id in ("T_0000123456", "T_0000654321"):
        wav_path = MANDARIN_PROFILE.generated_wav_path(root, "train", utt_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFFxxxxWAVEfmt ")
    return root


def test_reads_mandarin_generated_subset_with_audio_paths(jmds_root: Path):
    profile = _profile_with_train_count(2)
    frame = read_mandarin_generated(
        JMDS_FIXTURE,
        split="train",
        jmds_root=jmds_root,
        profile=profile,
    )

    assert list(frame.columns) == [
        *JMDS_COLUMNS,
        "split",
        "metadata_source_file",
        "metadata_source_row",
        "audio_path",
    ]
    _assert_mandarin_generated_train_output(frame, jmds_root=jmds_root)
    assert frame["metadata_source_row"].tolist() == ["2", "3"]
    assert frame["metadata_source_file"].tolist() == [
        JMDS_FIXTURE.name,
        JMDS_FIXTURE.name,
    ]
    generated_root = (
        jmds_root
        / "dataset"
        / MANDARIN_PROFILE.generated_dir_name
        / "train"
        / MANDARIN_PROFILE.generated_audio_subdir
    )
    assert all(
        Path(path).is_relative_to(generated_root) for path in frame["audio_path"]
    )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (JMDS_COLUMNS[:-1], "missing columns"),
        (JMDS_COLUMNS + ["duration"], "extra columns"),
    ],
)
def test_rejects_missing_or_extra_jmds_columns(
    tmp_path, jmds_root, columns, message
):
    path = tmp_path / "jmds.csv"
    pd.DataFrame(columns=columns).to_csv(path, index=False)
    profile = _profile_with_train_count(0)

    with pytest.raises(ValueError, match=message):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


def test_rejects_reordered_jmds_columns(tmp_path, jmds_root):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    reordered = [JMDS_COLUMNS[1], JMDS_COLUMNS[0], *JMDS_COLUMNS[2:]]
    path = tmp_path / "jmds.csv"
    protocol[reordered].to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match="required order"):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


def test_excludes_pristine_aishell3_rows(jmds_root: Path):
    profile = _profile_with_train_count(2)
    frame = read_mandarin_generated(
        JMDS_FIXTURE,
        split="train",
        jmds_root=jmds_root,
        profile=profile,
    )

    assert "AISHELL3" not in frame["dataset"].tolist()
    assert "pristine" not in frame["label"].tolist()


def test_accepts_cross_corpus_duplicate_ids_outside_mandarin_generated_subset(
    tmp_path, jmds_root
):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    extra_rows = pd.DataFrame(
        [
            {
                "spk_id": "T_9999",
                "utt_id": "T_0000000099",
                "gender": "F",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
            {
                "spk_id": "T_9998",
                "utt_id": "T_0000000099",
                "gender": "M",
                "codec": "-",
                "attack_id": "A02",
                "label": "generated",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            },
        ]
    )
    protocol = pd.concat([protocol, extra_rows], ignore_index=True)
    path = tmp_path / "jmds_cross_corpus_duplicates.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    frame = read_mandarin_generated(
        path, split="train", jmds_root=jmds_root, profile=profile
    )

    _assert_mandarin_generated_train_output(frame, jmds_root=jmds_root)


def test_accepts_irrelevant_row_with_nonstandard_split_prefix_id(
    tmp_path, jmds_root
):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    extra_rows = pd.DataFrame(
        [
            {
                "spk_id": "U_0001",
                "utt_id": "U_0000000099",
                "gender": "F",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "eng",
                "dataset": "ASVspoof2024",
            }
        ]
    )
    protocol = pd.concat([protocol, extra_rows], ignore_index=True)
    path = tmp_path / "jmds_nonstandard_prefix.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    frame = read_mandarin_generated(
        path, split="train", jmds_root=jmds_root, profile=profile
    )

    _assert_mandarin_generated_train_output(frame, jmds_root=jmds_root)


def test_accepts_por_generated_mlaad_outside_mandarin_subset(tmp_path, jmds_root):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    extra_rows = pd.DataFrame(
        [
            {
                "spk_id": "T_9999",
                "utt_id": "T_0000000099",
                "gender": "F",
                "codec": "-",
                "attack_id": "A01",
                "label": "generated",
                "native": "yes",
                "language": "por",
                "dataset": "MLAAD",
            }
        ]
    )
    protocol = pd.concat([protocol, extra_rows], ignore_index=True)
    path = tmp_path / "jmds_por_mlaad.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    frame = read_mandarin_generated(
        path, split="train", jmds_root=jmds_root, profile=profile
    )

    _assert_mandarin_generated_train_output(frame, jmds_root=jmds_root)


def test_rejects_duplicate_jmds_utterance_ids(tmp_path, jmds_root):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[1, "utt_id"] = protocol.loc[0, "utt_id"]
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


def test_rejects_malformed_jmds_utterance_ids(tmp_path, jmds_root):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[0, "utt_id"] = "bad-id"
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match="Malformed.*utt_id"):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


@pytest.mark.parametrize("split", ["test", "unsupported"])
def test_rejects_unsupported_jmds_splits(jmds_root, split):
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match=split):
        read_mandarin_generated(
            JMDS_FIXTURE, split=split, jmds_root=jmds_root, profile=profile
        )


def test_rejects_unexpected_generated_count(tmp_path, jmds_root):
    profile = _profile_with_train_count(99)

    with pytest.raises(ValueError, match=r"(?i)expected 99"):
        read_mandarin_generated(
            JMDS_FIXTURE, split="train", jmds_root=jmds_root, profile=profile
        )


def test_rejects_missing_generated_wav(tmp_path, jmds_root):
    profile = _profile_with_train_count(2)
    missing = MANDARIN_PROFILE.generated_wav_path(
        jmds_root, "train", "T_0000654321"
    )
    missing.unlink()

    with pytest.raises(ValueError, match=r"(?i)missing.*wav"):
        read_mandarin_generated(
            JMDS_FIXTURE, split="train", jmds_root=jmds_root, profile=profile
        )


def test_validate_resolved_audio_paths_rejects_outside_generated_split_root(
    tmp_path: Path,
):
    jmds_root = tmp_path / "jmds"
    outside_wav = jmds_root / "outside.wav"
    outside_wav.parent.mkdir(parents=True, exist_ok=True)
    outside_wav.write_bytes(b"RIFFxxxxWAVEfmt ")
    audio_paths = pd.Series([str(outside_wav)])

    with pytest.raises(ValueError, match="generated split root"):
        _validate_resolved_audio_paths(
            audio_paths,
            split="train",
            jmds_root=jmds_root,
            profile=MANDARIN_PROFILE,
        )


def test_validate_resolved_audio_paths_rejects_duplicate_resolved_paths(
    tmp_path: Path,
):
    jmds_root = tmp_path / "jmds"
    inside_wav = MANDARIN_PROFILE.generated_wav_path(
        jmds_root, "train", "T_0000123456"
    )
    inside_wav.parent.mkdir(parents=True, exist_ok=True)
    inside_wav.write_bytes(b"RIFFxxxxWAVEfmt ")
    path = str(inside_wav)
    audio_paths = pd.Series([path, path])

    with pytest.raises(ValueError, match="Duplicate.*audio_path"):
        _validate_resolved_audio_paths(
            audio_paths,
            split="train",
            jmds_root=jmds_root,
            profile=MANDARIN_PROFILE,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("language", "eng", r"(?i)Expected 2 Mandarin generated rows"),
        ("label", "pristine", "label"),
        ("dataset", "AISHELL3", "dataset"),
    ],
)
def test_rejects_wrong_values_in_mandarin_generated_subset(
    tmp_path, jmds_root, field, value, message
):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    protocol = protocol.loc[
        (protocol["language"] == "zho") & (protocol["label"] == "generated")
    ].copy()
    protocol.loc[0, field] = value
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match=message):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attack_id", "A01"),
        ("gender", "F"),
        ("spk_id", "T_0001"),
        ("codec", "mp3"),
        ("native", "no"),
    ],
)
def test_rejects_non_sparse_add_field_values(tmp_path, jmds_root, field, value):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    protocol = protocol.loc[
        (protocol["language"] == "zho") & (protocol["label"] == "generated")
    ].copy()
    protocol.loc[0, field] = value
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match=field):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )


def test_rejects_invalid_mandarin_zho_row_combinations(tmp_path, jmds_root):
    protocol = pd.read_csv(JMDS_FIXTURE, dtype=str, keep_default_na=False)
    protocol.loc[
        protocol["dataset"] == "AISHELL3", ["label", "dataset"]
    ] = ["generated", "WRONG"]
    path = tmp_path / "jmds.csv"
    protocol.to_csv(path, index=False)
    profile = _profile_with_train_count(2)

    with pytest.raises(ValueError, match="dataset"):
        read_mandarin_generated(
            path, split="train", jmds_root=jmds_root, profile=profile
        )
