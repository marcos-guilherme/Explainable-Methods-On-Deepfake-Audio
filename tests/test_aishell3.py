from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE
from jmds_prepare.sources.aishell3 import (
    AISHELL3_COLUMNS,
    read_aishell3_metadata,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aishell3_mini.tgz"


def test_read_aishell3_streams_without_extracting_wavs(tmp_path, monkeypatch):
    extracted: list[str] = []
    original_open = tarfile.open

    def tracking_open(path, mode="r"):
        archive = original_open(path, mode)
        original_extractfile = archive.extractfile

        def wrapped(name):
            if str(name).endswith(".wav"):
                extracted.append(str(name))
            return original_extractfile(name)

        archive.extractfile = wrapped  # type: ignore[method-assign]
        return archive

    monkeypatch.setattr(tarfile, "open", tracking_open)
    frame, reconciliation = read_aishell3_metadata(
        FIXTURE, profile=MANDARIN_PROFILE
    )
    assert extracted == []
    assert list(frame.columns) == [*AISHELL3_COLUMNS]
    assert set(frame["split"]) == {"train", "test"}
    assert frame["archive_member_path"].str.endswith(".wav").all()
    assert frame["archive_member_path"].str.contains("/wav/SSB").all()
    assert (frame["prosody_available"] == "yes").sum() == 2
    assert len(frame) == 4
    test_report = reconciliation.by_split["test"]
    assert test_report["content_line_count"] == 3
    assert test_report["wav_member_count"] == 3
    assert test_report["matched_count"] == 2
    assert test_report["content_without_wav"] == ["SSB00050005"]
    assert test_report["wav_without_content"] == ["SSB00059999"]
    assert reconciliation.official_test_sample_count == 23262
    assert reconciliation.observed_test_content_line_count == 3
    assert (
        "24,773" in reconciliation.divergence_explanation
        or "24773" in reconciliation.divergence_explanation
    )


def test_manifest_includes_only_matched_utterances():
    frame, _ = read_aishell3_metadata(FIXTURE, profile=MANDARIN_PROFILE)
    assert set(frame["utt_id"]) == {
        "SSB00050001",
        "SSB00050002",
        "SSB00050003",
        "SSB00050004",
    }
    train = frame[frame["split"] == "train"]
    assert train["speaker_id"].tolist() == ["SSB0005", "SSB0005"]
    assert train["gender"].tolist() == ["female", "female"]
    assert train["age"].tolist() == ["B", "B"]
    assert train["accent"].tolist() == ["north", "north"]
    assert train["pinyin"].tolist() == ["ni3 hao3", "xun4 lian4"]
    assert train["prosody_available"].tolist() == ["yes", "yes"]
    assert train["archive_member_path"].tolist() == [
        "train/wav/SSB0005/SSB00050001.wav",
        "train/wav/SSB0005/SSB00050002.wav",
    ]


def test_train_reconciliation_reports_content_without_wav():
    _, reconciliation = read_aishell3_metadata(FIXTURE, profile=MANDARIN_PROFILE)
    train_report = reconciliation.by_split["train"]
    assert train_report["content_line_count"] == 3
    assert train_report["matched_count"] == 2
    assert train_report["content_without_wav"] == ["SSB09990001"]


def test_content_normalizes_wav_suffix(tmp_path):
    archive = tmp_path / "content.wav.tar.gz"
    _write_mini_tar(
        archive,
        _base_members(
            train_content=b"SSB00050001.wav\thello world\n",
        ),
    )
    frame, _ = read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)
    assert frame.iloc[0]["utt_id"] == "SSB00050001"


def test_spk_info_ignores_comment_lines(tmp_path):
    archive = tmp_path / "comments.tar.gz"
    members = _base_members()
    members[0] = (
        "spk-info.txt",
        (
            b"# voice-file name; age group; gender; accent\n"
            b"SSB0005\tB\tfemale\tnorth\n"
        ),
    )
    _write_mini_tar(archive, members)
    frame, _ = read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)
    assert frame.iloc[0]["gender"] == "female"
    assert frame.iloc[0]["age"] == "B"


def test_label_file_uses_pipe_delimiter(tmp_path):
    archive = tmp_path / "label.tar.gz"
    members = _base_members(
        train_label=(
            b"# comment\n"
            b"SSB00050001|tone1 tone2 % phrase$\n"
        ),
    )
    _write_mini_tar(archive, members)
    frame, _ = read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)
    assert frame.iloc[0]["pinyin"] == "tone1 tone2 % phrase$"
    assert frame.iloc[0]["prosody_available"] == "yes"


def _write_mini_tar(path: Path, members: list[tuple[str, bytes]], *, gz: bool = True) -> None:
    mode = "w:gz" if gz else "w"
    with tarfile.open(path, mode) as archive:
        for name, content in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _base_members(
    *,
    train_content: bytes = b"SSB00050001.wav\thello\n",
    train_label: bytes = b"SSB00050001|ni3 hao3\n",
) -> list[tuple[str, bytes]]:
    return [
        (
            "spk-info.txt",
            b"SSB0005\tB\tfemale\tnorth\n",
        ),
        ("train/content.txt", train_content),
        ("test/content.txt", b"SSB00050003.wav\ttest\n"),
        ("train/label_train-set.txt", train_label),
        ("train/wav/SSB0005/SSB00050001.wav", b"RIFFxxxxWAVEfmt "),
        ("test/wav/SSB0005/SSB00050003.wav", b"RIFFzzzzWAVEfmt "),
    ]


@pytest.mark.parametrize(
    "extra_members",
    [
        [("train/wav/SSB0005/SSB00050001.wav", b"dup")],
        [("spk-info.txt", b"SSB0005\tB\tfemale\tnorth\n")],
        [("train/content.txt", b"SSB00050001.wav\tdup\n")],
    ],
)
def test_rejects_duplicate_tar_members(tmp_path, extra_members):
    archive = tmp_path / "dup.tar.gz"
    members = _base_members() + extra_members
    _write_mini_tar(archive, members)

    with pytest.raises(ValueError, match="Duplicate TAR member"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../train/wav/SSB0005/SSB00050001.wav",
        "train/wav/SSB0005/../SSB00050001.wav",
        "/train/wav/SSB0005/SSB00050001.wav",
        r"train\wav\SSB0005\SSB00050001.wav",
    ],
)
def test_rejects_unsafe_tar_member_paths(tmp_path, unsafe_name):
    archive = tmp_path / "unsafe.tar.gz"
    members = [
        (unsafe_name, b"RIFFxxxxWAVEfmt "),
        *_base_members(),
    ]
    _write_mini_tar(archive, members)

    with pytest.raises(ValueError, match="unsafe"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)


def test_rejects_non_regular_tar_members(tmp_path):
    archive = tmp_path / "nonregular.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in _base_members():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        link = tarfile.TarInfo("test/wav/SSB0005/SSB00050004.wav")
        link.type = tarfile.SYMTYPE
        link.linkname = "test/wav/SSB0005/SSB00050003.wav"
        tar.addfile(link)

    with pytest.raises(ValueError, match="non-regular"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)


def test_rejects_unknown_speaker_id(tmp_path):
    archive = tmp_path / "unknown_spk.tar.gz"
    members = _base_members(
        train_content=b"SSB99990001.wav\tunknown speaker\n",
    )
    members.append(("train/wav/SSB9999/SSB99990001.wav", b"RIFFxxxxWAVEfmt "))
    _write_mini_tar(archive, members)

    with pytest.raises(ValueError, match="unknown speaker"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)


def test_rejects_duplicate_utterance_ids_in_content(tmp_path):
    archive = tmp_path / "dup_utt.tar.gz"
    members = _base_members(
        train_content=(
            b"SSB00050001.wav\tfirst\n"
            b"SSB00050001.wav\tsecond\n"
        ),
    )
    _write_mini_tar(archive, members)

    with pytest.raises(ValueError, match="Duplicate.*utt_id"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)


def test_rejects_wav_speaker_folder_mismatch(tmp_path):
    archive = tmp_path / "mismatch.tar.gz"
    members = _base_members()
    members = [
        m for m in members if m[0] != "train/wav/SSB0005/SSB00050001.wav"
    ]
    members.append(("train/wav/SSB0999/SSB00050001.wav", b"RIFFxxxxWAVEfmt "))
    _write_mini_tar(archive, members)

    with pytest.raises(ValueError, match="speaker folder"):
        read_aishell3_metadata(archive, profile=MANDARIN_PROFILE)

