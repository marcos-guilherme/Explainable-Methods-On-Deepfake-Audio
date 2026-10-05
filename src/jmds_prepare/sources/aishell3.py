"""Streaming AISHELL-3 metadata adapter with metadata–WAV reconciliation."""

from __future__ import annotations

import re
import tarfile
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pandas as pd

from ..profiles.mandarin import MandarinProfile

AISHELL3_COLUMNS = (
    "utt_id",
    "split",
    "archive_member_path",
    "speaker_id",
    "gender",
    "age",
    "accent",
    "transcription",
    "pinyin",
    "prosody_available",
    "metadata_source_file",
    "metadata_source_row",
)

_OFFICIAL_TEST_SAMPLE_COUNT = 23262
_OBSERVED_REAL_TEST_CONTENT_LINE_COUNT = 24773

_WAV_MEMBER_RE = re.compile(
    r"^(?P<split>train|test)/wav/(?P<speaker>[^/]+)/(?P<utt_id>[^/]+)\.wav$"
)


@dataclass(frozen=True)
class AishellReconciliation:
    by_split: Mapping[str, dict[str, Any]]
    official_test_sample_count: int
    observed_test_content_line_count: int
    divergence_explanation: str


def read_aishell3_metadata(
    archive_path: Path,
    *,
    profile: MandarinProfile,
) -> tuple[pd.DataFrame, AishellReconciliation]:
    """Stream ``archive_path`` once and build a reconciled AISHELL-3 manifest."""
    archive_path = Path(archive_path)
    expected_text_members = set(profile.aishell_archive_members.values())
    text_bodies: dict[str, str] = {}
    wav_by_split: dict[str, dict[str, tuple[str, int]]] = {
        split: {} for split in profile.aishell_splits
    }
    seen_members: set[str] = set()

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name
            if name in seen_members:
                raise ValueError(f"Duplicate TAR member: {name!r}")
            seen_members.add(name)

            if _is_unsafe_member_path(name):
                raise ValueError(f"unsafe TAR member path: {name!r}")

            if not member.isreg():
                if _is_tracked_member(name, expected_text_members):
                    raise ValueError(f"non-regular TAR member: {name!r}")
                wav_match = _WAV_MEMBER_RE.match(name)
                if wav_match is not None:
                    raise ValueError(f"non-regular TAR member: {name!r}")
                continue

            if name in expected_text_members:
                text_bodies[name] = _read_text_member(archive, member)
                continue

            wav_match = _WAV_MEMBER_RE.match(name)
            if wav_match is not None:
                split = wav_match.group("split")
                speaker = wav_match.group("speaker")
                utt_id = wav_match.group("utt_id")
                if speaker != utt_id[:7]:
                    raise ValueError(
                        "AISHELL wav speaker folder "
                        f"{speaker!r} does not match utt_id prefix "
                        f"{utt_id[:7]!r} for member {name!r}"
                    )
                split_wavs = wav_by_split[split]
                if utt_id in split_wavs:
                    raise ValueError(
                        f"Duplicate AISHELL wav utt_id {utt_id!r} in split {split!r}"
                    )
                split_wavs[utt_id] = (name, member.size)

    _require_text_members(text_bodies, expected_text_members)
    speakers = _parse_spk_info(
        text_bodies[profile.aishell_archive_members["spk-info"]]
    )
    labels = _parse_label_file(
        text_bodies[profile.aishell_archive_members["train/label"]]
    )

    manifest_rows: list[dict[str, str]] = []
    by_split: dict[str, dict[str, Any]] = {}

    for split in profile.aishell_splits:
        content_key = f"{split}/content"
        content_path = profile.aishell_archive_members[content_key]
        content_rows = _parse_content(text_bodies[content_path], content_path)
        split_wavs = wav_by_split[split]

        content_ids = [utt_id for utt_id, _, _ in content_rows]
        wav_ids = set(split_wavs)
        matched_ids = [utt_id for utt_id in content_ids if utt_id in wav_ids]
        content_without_wav = [
            utt_id for utt_id in content_ids if utt_id not in wav_ids
        ]
        wav_without_content = sorted(wav_ids - set(content_ids))

        by_split[split] = {
            "content_line_count": len(content_rows),
            "wav_member_count": len(split_wavs),
            "matched_count": len(matched_ids),
            "content_without_wav": content_without_wav,
            "wav_without_content": wav_without_content,
        }

        for utt_id, transcription, source_row in content_rows:
            if utt_id not in wav_ids:
                continue
            speaker_id = utt_id[:7]
            if speaker_id not in speakers:
                raise ValueError(f"unknown speaker_id {speaker_id!r} for {utt_id!r}")
            gender, age_group, accent = speakers[speaker_id]
            archive_member_path, _size = split_wavs[utt_id]
            label = labels.get(utt_id, "")
            manifest_rows.append(
                {
                    "utt_id": utt_id,
                    "split": split,
                    "archive_member_path": archive_member_path,
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "age": age_group,
                    "accent": accent,
                    "transcription": transcription,
                    "pinyin": label,
                    "prosody_available": "yes" if utt_id in labels else "no",
                    "metadata_source_file": PurePosixPath(content_path).name,
                    "metadata_source_row": str(source_row),
                }
            )

    frame = pd.DataFrame(manifest_rows, columns=list(AISHELL3_COLUMNS))
    observed_test_content_line_count = by_split["test"]["content_line_count"]
    reconciliation = AishellReconciliation(
        by_split=by_split,
        official_test_sample_count=_OFFICIAL_TEST_SAMPLE_COUNT,
        observed_test_content_line_count=observed_test_content_line_count,
        divergence_explanation=_build_divergence_explanation(
            observed_test_content_line_count
        ),
    )
    return frame, reconciliation


def _is_unsafe_member_path(name: str) -> bool:
    if "\\" in name:
        return True
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    return (
        posix_path.is_absolute()
        or ".." in posix_path.parts
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    )


def _is_tracked_member(name: str, expected_text_members: set[str]) -> bool:
    return name in expected_text_members


def _read_text_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read regular TAR member {member.name!r}")
    with closing(source):
        return source.read().decode("utf-8")


def _require_text_members(
    text_bodies: Mapping[str, str], expected: set[str]
) -> None:
    missing = sorted(expected - set(text_bodies))
    if missing:
        raise ValueError(
            "Missing AISHELL-3 metadata member(s): "
            + ", ".join(missing)
        )


def _parse_spk_info(text: str) -> dict[str, tuple[str, str, str]]:
    speakers: dict[str, tuple[str, str, str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 4:
            raise ValueError("Invalid AISHELL-3 spk-info.txt row")
        speaker_id, age_group, gender, accent = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
        )
        if speaker_id in speakers:
            raise ValueError(f"Duplicate AISHELL-3 speaker_id {speaker_id!r}")
        speakers[speaker_id] = (gender, age_group, accent)
    return speakers


def _normalize_content_utt_id(raw_id: str) -> str:
    if raw_id.endswith(".wav"):
        return raw_id[: -len(".wav")]
    return raw_id


def _parse_content(text: str, source_path: str) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split("\t", maxsplit=1)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid AISHELL-3 content row in {source_path!r} at line {line_no}"
            )
        utt_id = _normalize_content_utt_id(parts[0])
        transcription = parts[1]
        if utt_id in seen:
            raise ValueError(f"Duplicate AISHELL-3 utt_id value(s): {utt_id}")
        seen.add(utt_id)
        rows.append((utt_id, transcription, line_no))
    return rows


def _parse_label_file(text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("|", maxsplit=1)
        if len(parts) < 2:
            raise ValueError(
                "Invalid AISHELL-3 label row at "
                f"line {line_no} in label_train-set.txt"
            )
        utt_id, label = parts[0], parts[1]
        if utt_id in labels:
            raise ValueError(f"Duplicate AISHELL-3 label utt_id {utt_id!r}")
        labels[utt_id] = label
    return labels


def _build_divergence_explanation(observed_test_content_line_count: int) -> str:
    return (
        "AISHELL-3 test split reconciliation: observed test/content.txt line "
        f"count is {observed_test_content_line_count:,} in this archive scan; "
        f"the official AISHELL-3 publication reports {_OFFICIAL_TEST_SAMPLE_COUNT:,} "
        "test samples. Full data_aishell3.tgz archives typically show "
        f"{_OBSERVED_REAL_TEST_CONTENT_LINE_COUNT:,} content lines versus "
        f"{_OFFICIAL_TEST_SAMPLE_COUNT:,} official test samples. Compare "
        "content_without_wav and wav_without_content per split before interpreting "
        "test cardinality."
    )
