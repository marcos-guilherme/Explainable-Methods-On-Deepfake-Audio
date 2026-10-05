"""Tests for XAI dataset orchestration, validation and publication."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from jmds_prepare.core.hashing import sha256_file, sha256_file_preserving_times
from jmds_prepare.core.publication import DestinationConflictError, publish_artifact_set_idempotent
from jmds_prepare.core.xai_sample import XAI_SAMPLE_COLUMNS, XaiSample
from jmds_prepare.pipelines.xai_audit import audit_processed_sample, summarize_sample_audits
from jmds_prepare.pipelines.xai_dataset import (
    CANONICAL_SAMPLE_RATE,
    DatasetValidationError,
    SelectedValidationError,
    XaiDatasetDeps,
    build_selection_report,
    build_xai_provenance,
    build_xai_dataset,
    require_canonical_sample_rate,
    serialize_xai_samples_csv,
    serialize_json,
    validate_selected_before_materialization,
    validate_xai_sample_set,
)
from jmds_prepare.pipelines.xai_materialization import (
    CORAA_TRAIN_RAR_VOLUME_NAMES,
    ArchiveSpec,
    materialize_selected,
    standard_archive_spec,
)
from jmds_prepare.pipelines.xai_selection import (
    SelectedCandidate,
    SelectionCandidate,
    select_language_candidates,
)
from jmds_prepare.profiles.xai import (
    ENGLISH_XAI_PROFILE,
    MANDARIN_XAI_PROFILE,
    PORTUGUESE_XAI_PROFILE,
    SHARED_SPLIT_DISJOINT_FIELDS,
    XaiLanguageProfile,
)
from jmds_prepare.storage.xai_layout import XaiLayout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    frames: int = 400,
    value: float = 0.25,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.full(frames, value, dtype=np.float64)
    samples[: frames // 4] = 0.0
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members:
            archive.writestr(name, content)


def _tiny_eng_profile(*, paired: bool = True) -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="eng",
        language_slug="english",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"ASVspoof2024": 0, "JMDS": 1},
        corpus_native_split_roles={
            "ASVspoof2024": {"train": ("train",), "dev": ("calibration", "test")},
            "JMDS": {"train": ("train",), "dev": ("calibration", "test")},
        },
        shared_split_disjoint_fields={
            "ASVspoof2024": {"dev": ("speaker_id", "group_id")},
            "JMDS": {"dev": ("speaker_id", "group_id")},
        },
        paired_selection=paired,
    )


def _tiny_por_profile() -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="por",
        language_slug="portuguese",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"CORAA": 0, "MLAAD": 1},
        corpus_native_split_roles={
            "CORAA": {
                "train": ("train",),
                "dev": ("calibration",),
                "test": ("test",),
            },
            "MLAAD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={},
        paired_selection=False,
    )


def _selected(
    candidate: SelectionCandidate,
    *,
    role: str = "train",
    rank: int = 0,
    reason: str = "fixture",
) -> SelectedCandidate:
    return SelectedCandidate(
        candidate=candidate,
        role=role,
        selection_seed=7,
        selection_rank=rank,
        selection_reason=reason,
        selection_source="fixture.csv",
    )


def _local_candidate(
    tmp_path: Path,
    *,
    language: str = "por",
    label: int = 1,
    corpus: str = "MLAAD",
    native_split: str = "train",
    candidate_id: str = "utt-001",
    speaker_id: str | None = "spk-1",
) -> SelectionCandidate:
    source = tmp_path / "sources" / f"{candidate_id}.wav"
    _write_wav(source, sample_rate=8_000)
    return SelectionCandidate(
        candidate_id=candidate_id,
        language=language,
        label=label,
        corpus=corpus,
        native_split=native_split,
        original_ref=str(source),
        local_source_path=str(source),
        speaker_id=speaker_id,
        group_id=None,
        attack_id="A01" if label == 1 else None,
        metadata_source="fixture",
        materialization_mode="process_source",
    )


def _build_por_candidate_pool(tmp_path: Path) -> tuple[SelectionCandidate, ...]:
    candidates: list[SelectionCandidate] = []
    for index in range(1, 5):
        candidates.append(
            _local_candidate(
                tmp_path,
                candidate_id=f"mlaad-{index:03d}",
                label=1,
                corpus="MLAAD",
                native_split="train",
                speaker_id=f"spk-{index}",
            )
        )
        candidates.append(
            _local_candidate(
                tmp_path,
                candidate_id=f"mlaad-{index:03d}-dev",
                label=1,
                corpus="MLAAD",
                native_split="dev",
                speaker_id=f"spk-dev-{index}",
            )
        )
        candidates.append(
            _local_candidate(
                tmp_path,
                candidate_id=f"mlaad-{index:03d}-eval",
                label=1,
                corpus="MLAAD",
                native_split="eval",
                speaker_id=f"spk-eval-{index}",
            )
        )
    for index in range(1, 5):
        member = f"audio/train/file_{index}.wav"
        candidates.append(
            SelectionCandidate(
                candidate_id=member,
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref=member,
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="coraa.csv",
                materialization_mode="extract_archive",
            )
        )
        candidates.append(
            SelectionCandidate(
                candidate_id=f"dev/file_{index}.wav",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="dev",
                original_ref=f"dev/file_{index}.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="coraa.csv",
                materialization_mode="extract_archive",
            )
        )
        candidates.append(
            SelectionCandidate(
                candidate_id=f"test/file_{index}.wav",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="test",
                original_ref=f"test/file_{index}.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="coraa.csv",
                materialization_mode="extract_archive",
            )
        )
    return tuple(candidates)


def _wav_bytes(tmp_path: Path, *, value: float = 0.2) -> bytes:
    path = tmp_path / f"fixture-{value}.wav"
    _write_wav(path, sample_rate=8_000, value=value)
    return path.read_bytes()


def _write_coraa_volumes(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    part1 = directory / CORAA_TRAIN_RAR_VOLUME_NAMES[0]
    for name in CORAA_TRAIN_RAR_VOLUME_NAMES:
        (directory / name).write_bytes(f"fake-{name}".encode())
    return part1


def _por_archive_specs(tmp_path: Path) -> dict[tuple[str, str], ArchiveSpec]:
    train_part1 = _write_coraa_volumes(tmp_path / "coraa")
    dev_zip = tmp_path / "coraa" / "dev.zip"
    test_zip = tmp_path / "coraa" / "test.zip"
    _write_zip(
        dev_zip,
        [
            (f"dev/file_{index}.wav", _wav_bytes(tmp_path, value=0.2 + index * 0.01))
            for index in range(1, 5)
        ],
    )
    _write_zip(
        test_zip,
        [
            (f"test/file_{index}.wav", _wav_bytes(tmp_path, value=0.3 + index * 0.01))
            for index in range(1, 5)
        ],
    )
    return {
        ("CORAA", "train"): standard_archive_spec(
            corpus="CORAA",
            native_split="train",
            archive_path=train_part1,
        ),
        ("CORAA", "dev"): standard_archive_spec(
            corpus="CORAA",
            native_split="dev",
            archive_path=dev_zip,
        ),
        ("CORAA", "test"): standard_archive_spec(
            corpus="CORAA",
            native_split="test",
            archive_path=test_zip,
        ),
    }


class _FakeUnrarRunner:
    def __init__(self, *, list_output: str = "", member_bytes: dict[str, bytes] | None = None) -> None:
        self._list_output = list_output
        self._member_bytes = member_bytes or {}

    def __call__(self, args: list[str], **_: Any):
        command = args[1] if len(args) > 1 else ""
        if command == "lb":
            class Result:
                returncode = 0
                stdout = self._list_output
                stderr = ""

            return Result()
        if command == "x":
            destination = Path(str(args[-1]).rstrip("\\/"))
            list_path = Path(next(arg for arg in args if arg.startswith("@"))[1:])
            for member in list_path.read_text(encoding="utf-8").splitlines():
                if not member.strip():
                    continue
                payload = self._member_bytes.get(member, b"fallback-wav")
                target = destination.joinpath(*PurePosixPath(member).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()
        raise AssertionError(f"unexpected unrar args: {args}")


def _default_deps(**overrides: Any) -> XaiDatasetDeps:
    defaults = {
        "select_candidates": select_language_candidates,
        "materialize_selected": materialize_selected,
        "audit_sample": audit_processed_sample,
        "summarize_audits": summarize_sample_audits,
        "publish_artifacts": publish_artifact_set_idempotent,
        "sha256_file": sha256_file,
    }
    defaults.update(overrides)
    return XaiDatasetDeps(**defaults)


def test_serialize_xai_samples_csv_is_deterministic_and_canonical():
    sample = XaiSample(
        sample_id="por-train-bonafide-c1",
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        native_split="train",
        original_ref="audio/a.wav",
        processed_path="/data/por/train/bonafide/a.wav",
        speaker_id=None,
        group_id=None,
        attack_id=None,
        sha256_source="a" * 64,
        sha256_processed="b" * 64,
        selection_seed=7,
        selection_rank=0,
        selection_reason="unpaired_class_quota",
        selection_source="coraa.csv",
    )
    first = serialize_xai_samples_csv((sample,))
    second = serialize_xai_samples_csv((sample,))
    assert first == second
    assert first.decode("utf-8").endswith("\n")
    assert first.decode("utf-8").count("\n") == 2
    header, row = first.decode("utf-8").splitlines()
    assert header.split(",") == XAI_SAMPLE_COLUMNS
    XaiSample.from_dict(dict(zip(XAI_SAMPLE_COLUMNS, row.split(","))), enforce_column_order=True)


def test_build_xai_dataset_publishes_four_artifacts_for_portuguese(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    candidates = _build_por_candidate_pool(tmp_path)
    archive_specs = _por_archive_specs(tmp_path)
    manifest = tmp_path / "coraa.csv"
    manifest.write_text("file_path,split\n", encoding="utf-8")
    list_output = "\n".join(
        candidate.original_ref
        for candidate in candidates
        if candidate.materialization_mode == "extract_archive"
    )
    runner = _FakeUnrarRunner(
        list_output=list_output,
        member_bytes={
            candidate.original_ref: _wav_bytes(tmp_path, value=0.15)
            for candidate in candidates
            if candidate.materialization_mode == "extract_archive"
            and candidate.native_split == "train"
        },
    )

    build_xai_dataset(
        candidates,
        profile=profile,
        seed=7,
        layout=layout,
        archive_specs=archive_specs,
        sample_rate=16_000,
        input_paths={"pristine_manifest": manifest},
        deps=_default_deps(),
        unrar_executable="UnRAR.exe",
        archive_runner=runner,
    )

    manifest_path = layout.samples_csv("por")
    selection_report = layout.selection_report_json("por")
    audio_audit = layout.audio_audit_json("por")
    provenance = layout.provenance_json("por")
    for path in (manifest_path, selection_report, audio_audit, provenance):
        assert path.is_file(), path

    samples_text = manifest_path.read_text(encoding="utf-8")
    assert samples_text.endswith("\n")
    rows = samples_text.strip().splitlines()
    assert rows[0].split(",") == XAI_SAMPLE_COLUMNS
    assert len(rows) - 1 == sum(profile.per_class_targets.values()) * 2

    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert provenance_payload["limitations"]["paired_samples"] is False
    assert provenance_payload["limitations"]["class_corpus_confound"] is True
    assert provenance_payload["limitations"]["coraa"]["local_only_no_redistribution"] is True
    assert provenance_payload["limitations"]["coraa"]["license"] == "CC-BY-NC-ND-4.0"
    assert "timestamp" not in json.dumps(provenance_payload)


def test_build_xai_dataset_rerun_is_byte_identical(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    candidates = _build_por_candidate_pool(tmp_path)
    archive_specs = _por_archive_specs(tmp_path)
    manifest = tmp_path / "coraa.csv"
    manifest.write_text("file_path,split\n", encoding="utf-8")
    list_output = "\n".join(
        candidate.original_ref
        for candidate in candidates
        if candidate.materialization_mode == "extract_archive"
    )
    runner = _FakeUnrarRunner(
        list_output=list_output,
        member_bytes={
            candidate.original_ref: _wav_bytes(tmp_path, value=0.15)
            for candidate in candidates
            if candidate.materialization_mode == "extract_archive"
            and candidate.native_split == "train"
        },
    )
    kwargs = dict(
        candidates=candidates,
        profile=profile,
        seed=7,
        layout=layout,
        archive_specs=archive_specs,
        sample_rate=16_000,
        input_paths={"pristine_manifest": manifest},
        deps=_default_deps(),
        unrar_executable="UnRAR.exe",
        archive_runner=runner,
    )
    build_xai_dataset(**kwargs)
    first = {
        path: path.read_bytes()
        for path in (
            layout.samples_csv("por"),
            layout.selection_report_json("por"),
            layout.audio_audit_json("por"),
            layout.provenance_json("por"),
        )
    }
    build_xai_dataset(**kwargs)
    second = {
        path: path.read_bytes()
        for path in (
            layout.samples_csv("por"),
            layout.selection_report_json("por"),
            layout.audio_audit_json("por"),
            layout.provenance_json("por"),
        )
    }
    assert first == second


def test_build_xai_dataset_conflict_in_one_artifact_leaves_others_untouched(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    candidates = _build_por_candidate_pool(tmp_path)
    archive_specs = _por_archive_specs(tmp_path)
    manifest = tmp_path / "coraa.csv"
    manifest.write_text("file_path,split\n", encoding="utf-8")
    list_output = "\n".join(
        candidate.original_ref
        for candidate in candidates
        if candidate.materialization_mode == "extract_archive"
    )
    runner = _FakeUnrarRunner(
        list_output=list_output,
        member_bytes={
            candidate.original_ref: _wav_bytes(tmp_path, value=0.15)
            for candidate in candidates
            if candidate.materialization_mode == "extract_archive"
            and candidate.native_split == "train"
        },
    )
    kwargs = dict(
        candidates=candidates,
        profile=profile,
        seed=7,
        layout=layout,
        archive_specs=archive_specs,
        sample_rate=16_000,
        input_paths={"pristine_manifest": manifest},
        deps=_default_deps(),
        unrar_executable="UnRAR.exe",
        archive_runner=runner,
    )
    build_xai_dataset(**kwargs)
    manifest_path = layout.samples_csv("por")
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(b"conflicting,csv\n")

    with pytest.raises(DestinationConflictError):
        build_xai_dataset(**kwargs)

    assert manifest_path.read_bytes() == b"conflicting,csv\n"
    assert layout.selection_report_json("por").exists()
    assert layout.audio_audit_json("por").exists()
    assert layout.provenance_json("por").exists()


def test_validate_xai_sample_set_rejects_quota_mismatch(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    samples = _materialize_tiny_portuguese_dataset(tmp_path, layout)
    train_bonafide = next(
        sample for sample in samples if sample.role == "train" and sample.label == 0
    )
    broken = replace(train_bonafide, label=1)
    broken_index = samples.index(train_bonafide)
    with pytest.raises(DatasetValidationError, match="quota"):
        validate_xai_sample_set(
            tuple(
                broken if index == broken_index else sample
                for index, sample in enumerate(samples)
            ),
            profile=profile,
            layout=layout,
        )


def test_validate_xai_sample_set_rejects_disjoint_overlap(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_zho_profile()
    samples = _materialize_tiny_portuguese_dataset(
        tmp_path,
        layout,
        profile=profile,
        language="zho",
    )
    train_sample = next(sample for sample in samples if sample.role == "train" and sample.label == 0)
    calibration_index = next(
        index
        for index, sample in enumerate(samples)
        if sample.role == "calibration" and sample.label == 0
    )
    broken = replace(
        samples[calibration_index],
        speaker_id=train_sample.speaker_id,
    )
    with pytest.raises(DatasetValidationError, match="overlap"):
        validate_xai_sample_set(
            tuple(
                broken if index == calibration_index else sample
                for index, sample in enumerate(samples)
            ),
            profile=profile,
            layout=layout,
        )


def test_validate_xai_sample_set_rejects_hash_mismatch(tmp_path: Path, monkeypatch):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    samples = _materialize_tiny_portuguese_dataset(tmp_path, layout)

    def wrong_hash(_path: Path) -> str:
        return "c" * 64

    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_dataset.sha256_file_preserving_times",
        wrong_hash,
    )
    with pytest.raises(DatasetValidationError, match="sha256"):
        validate_xai_sample_set(samples, profile=profile, layout=layout)


def _materialize_tiny_portuguese_dataset(
    tmp_path: Path,
    layout: XaiLayout,
    *,
    profile: XaiLanguageProfile | None = None,
    language: str = "por",
    overlap_speaker: str | None = None,
) -> tuple[XaiSample, ...]:
    profile = profile or _tiny_por_profile()
    if language == "por":
        candidates = _build_por_candidate_pool(tmp_path)
        archive_specs = _por_archive_specs(tmp_path)
        list_output = "\n".join(
            candidate.original_ref
            for candidate in candidates
            if candidate.materialization_mode == "extract_archive"
        )
        runner = _FakeUnrarRunner(
            list_output=list_output,
            member_bytes={
                candidate.original_ref: _wav_bytes(tmp_path, value=0.15)
                for candidate in candidates
                if candidate.materialization_mode == "extract_archive"
                and candidate.native_split == "train"
            },
        )
        selected = select_language_candidates(candidates, profile, 7)
        return materialize_selected(
            selected,
            layout,
            archive_specs,
            archive_runner=runner,
            unrar_executable="UnRAR.exe",
        )

    candidates = _build_zho_candidate_pool(tmp_path)
    archive = tmp_path / "aishell.tgz"
    members = [
        (candidate.original_ref, _wav_bytes(tmp_path, value=0.18))
        for candidate in candidates
        if candidate.corpus == "AISHELL-3"
    ]
    _write_tar(archive, members)
    archive_specs = {
        ("AISHELL-3", "train"): standard_archive_spec(
            corpus="AISHELL-3",
            native_split="train",
            archive_path=archive,
        ),
        ("AISHELL-3", "test"): standard_archive_spec(
            corpus="AISHELL-3",
            native_split="test",
            archive_path=archive,
        ),
    }
    selected = select_language_candidates(candidates, profile, 7)
    return materialize_selected(selected, layout, archive_specs)


def _tiny_zho_profile() -> XaiLanguageProfile:
    return XaiLanguageProfile(
        language_code="zho",
        language_slug="mandarin",
        per_class_targets={"train": 2, "calibration": 1, "test": 1},
        corpus_labels={"AISHELL-3": 0, "ADD": 1},
        corpus_native_split_roles={
            "AISHELL-3": {"train": ("train", "calibration"), "test": ("test",)},
            "ADD": {
                "train": ("train",),
                "dev": ("calibration",),
                "eval": ("test",),
            },
        },
        shared_split_disjoint_fields={"AISHELL-3": {"train": ("speaker_id",)}},
        paired_selection=False,
    )


def _build_zho_candidate_pool(tmp_path: Path) -> tuple[SelectionCandidate, ...]:
    candidates: list[SelectionCandidate] = []
    speaker = "spk-aishell"
    for index in range(1, 6):
        candidates.append(
            SelectionCandidate(
                candidate_id=f"aishell-train-{index:03d}",
                language="zho",
                label=0,
                corpus="AISHELL-3",
                native_split="train",
                original_ref=f"train/wav/{speaker}/utt-{index:03d}.wav",
                local_source_path=None,
                speaker_id=speaker if index <= 3 else f"spk-{index}",
                group_id=None,
                attack_id=None,
                metadata_source="aishell.tgz",
                materialization_mode="extract_archive",
            )
        )
        candidates.append(
            SelectionCandidate(
                candidate_id=f"aishell-test-{index:03d}",
                language="zho",
                label=0,
                corpus="AISHELL-3",
                native_split="test",
                original_ref=f"test/wav/spk-{index}/utt-{index:03d}.wav",
                local_source_path=None,
                speaker_id=f"spk-test-{index}",
                group_id=None,
                attack_id=None,
                metadata_source="aishell.tgz",
                materialization_mode="extract_archive",
            )
        )
    for index in range(1, 6):
        candidates.append(
            _local_candidate(
                tmp_path,
                language="zho",
                label=1,
                corpus="ADD",
                native_split="train" if index <= 2 else ("dev" if index <= 3 else "eval"),
                candidate_id=f"add-{index:03d}",
                speaker_id=f"add-spk-{index}",
            )
        )
    return tuple(candidates)


def test_build_selection_report_records_paired_selection_without_utterance_pairing():
    profile = _tiny_eng_profile(paired=True)
    candidates = (
        _candidate_for_report("c1", label=0, speaker_id="S1"),
        _candidate_for_report("c2", label=1, speaker_id="S1"),
    )
    selected = (
        _selected(candidates[0], role="train", rank=0, reason="paired_speaker_quota"),
        _selected(candidates[1], role="train", rank=0, reason="paired_speaker_quota"),
    )
    report = build_selection_report(
        candidates=candidates,
        selected=selected,
        profile=profile,
        seed=11,
    )
    assert report["paired_selection"] is True
    assert report["paired_utterances"] is False
    assert report["seed"] == 11
    assert "targets" in report
    assert "disjoint_verifications" in report


def _candidate_for_report(
    candidate_id: str,
    *,
    label: int,
    speaker_id: str,
) -> SelectionCandidate:
    return SelectionCandidate(
        candidate_id=candidate_id,
        language="eng",
        label=label,
        corpus="ASVspoof2024" if label == 0 else "JMDS",
        native_split="train",
        original_ref=f"/ref/{candidate_id}",
        local_source_path=f"/proc/{candidate_id}.wav",
        speaker_id=speaker_id,
        group_id=None,
        attack_id=None if label == 0 else "A01",
        metadata_source="english.csv",
        materialization_mode="reuse_processed",
        expected_sha256_processed="b" * 64,
    )


def test_require_canonical_sample_rate_rejects_other_values():
    require_canonical_sample_rate(CANONICAL_SAMPLE_RATE)
    with pytest.raises(ValueError, match="16000"):
        require_canonical_sample_rate(8_000)


def test_build_xai_dataset_rejects_non_canonical_sample_rate(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_por_profile()
    with pytest.raises(ValueError, match="16000"):
        build_xai_dataset(
            (),
            profile=profile,
            seed=7,
            layout=layout,
            archive_specs={},
            sample_rate=8_000,
            input_paths={},
            deps=_default_deps(),
        )


def test_validate_selected_before_materialization_rejects_disjoint_overlap(
    tmp_path: Path,
):
    profile = _tiny_zho_profile()
    selected = select_language_candidates(_build_zho_candidate_pool(tmp_path), profile, 7)
    train_sample = next(item for item in selected if item.role == "train" and item.candidate.label == 0)
    calibration_index = next(
        index
        for index, item in enumerate(selected)
        if item.role == "calibration" and item.candidate.label == 0
    )
    broken_candidate = replace(
        selected[calibration_index].candidate,
        speaker_id=train_sample.candidate.speaker_id,
    )
    broken = replace(selected[calibration_index], candidate=broken_candidate)
    broken_selected = tuple(
        broken if index == calibration_index else item
        for index, item in enumerate(selected)
    )
    with pytest.raises(SelectedValidationError, match="disjoint overlap"):
        validate_selected_before_materialization(broken_selected, profile=profile)


def test_build_xai_dataset_skips_materialization_when_selected_invalid(
    tmp_path: Path,
):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_zho_profile()
    candidates = _build_zho_candidate_pool(tmp_path)
    selected = select_language_candidates(candidates, profile, 7)
    train_sample = next(item for item in selected if item.role == "train" and item.candidate.label == 0)
    calibration_index = next(
        index
        for index, item in enumerate(selected)
        if item.role == "calibration" and item.candidate.label == 0
    )
    broken_candidate = replace(
        selected[calibration_index].candidate,
        speaker_id=train_sample.candidate.speaker_id,
    )
    broken_selected = tuple(
        replace(selected[calibration_index], candidate=broken_candidate)
        if index == calibration_index
        else item
        for index, item in enumerate(selected)
    )

    def broken_select(_candidates, _profile, _seed):
        return broken_selected

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("materialize_selected must not run")

    deps = _default_deps(
        select_candidates=broken_select,
        materialize_selected=fail_materialize,
    )
    with pytest.raises(SelectedValidationError):
        build_xai_dataset(
            candidates,
            profile=profile,
            seed=7,
            layout=layout,
            archive_specs={},
            sample_rate=16_000,
            input_paths={},
            deps=deps,
        )
    assert not layout.data_dir.exists()


def test_build_xai_provenance_english_limitations(tmp_path: Path):
    profile = ENGLISH_XAI_PROFILE
    manifest = tmp_path / "english_processed.csv"
    manifest.write_text("sample\n", encoding="utf-8")
    payload = build_xai_provenance(
        profile=profile,
        seed=3,
        sample_rate=16_000,
        input_paths={"english_manifest": manifest},
        archive_specs={},
        layout=XaiLayout(tmp_path / "out"),
        sha256_file=lambda _path: "d" * 64,
    )
    assert payload["language"] == "eng"
    assert "speaker_balanced" not in payload["limitations"]
    assert payload["limitations"]["paired_by_speaker"] is True
    assert payload["limitations"]["paired_utterances"] is False
    assert "timestamp" not in json.dumps(payload)


def test_build_xai_provenance_mandarin_limitations(tmp_path: Path):
    profile = MANDARIN_XAI_PROFILE
    pristine = tmp_path / "aishell3_metadata.csv"
    archive = tmp_path / "data_aishell3.tgz"
    pristine.write_text("utt_id,split\n", encoding="utf-8")
    archive.write_bytes(b"fake-tar")
    payload = build_xai_provenance(
        profile=profile,
        seed=3,
        sample_rate=16_000,
        input_paths={
            "pristine_manifest": pristine,
            "aishell_archive": archive,
        },
        archive_specs={
            ("AISHELL-3", "train"): ArchiveSpec(
                "AISHELL-3", "train", "tar", (archive,)
            )
        },
        layout=XaiLayout(tmp_path / "out"),
        sha256_file=lambda _path: "d" * 64,
    )
    assert payload["limitations"]["paired_samples"] is False
    assert payload["limitations"]["class_corpus_confound"] is True
    assert payload["limitations"]["external_validation_under_corpus_shift"] is True
    assert payload["aishell3"]["pristine_candidate_source"] == (
        "reconciled_aishell3_metadata_manifest"
    )
    assert payload["aishell3"]["reconciliation_details"] == (
        "upstream_mandarin_metadata_provenance"
    )
    input_keys = {entry["key"] for entry in payload["inputs"]}
    assert {"pristine_manifest", "aishell_archive"}.issubset(input_keys)


def test_build_xai_provenance_portuguese_coraa_policy(tmp_path: Path):
    profile = PORTUGUESE_XAI_PROFILE
    manifest = tmp_path / "coraa.csv"
    manifest.write_text("file_path,split\n", encoding="utf-8")
    payload = build_xai_provenance(
        profile=profile,
        seed=3,
        sample_rate=16_000,
        input_paths={"pristine_manifest": manifest},
        archive_specs={},
        layout=XaiLayout(tmp_path / "out"),
        sha256_file=lambda _path: "d" * 64,
    )
    assert payload["limitations"]["coraa"]["local_only_no_redistribution"] is True
    assert payload["limitations"]["coraa"]["license"] == "CC-BY-NC-ND-4.0"


def _eng_reuse_candidate(
    tmp_path: Path,
    *,
    candidate_id: str,
    label: int,
    corpus: str,
    native_split: str,
    speaker_id: str,
    group_id: str | None = None,
) -> SelectionCandidate:
    source = tmp_path / "raw" / native_split / f"{candidate_id}.flac"
    processed = tmp_path / "processed" / native_split / f"{candidate_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(f"source-{candidate_id}".encode())
    _write_wav(processed, sample_rate=16_000, value=0.12)
    return SelectionCandidate(
        candidate_id=candidate_id,
        language="eng",
        label=label,
        corpus=corpus,
        native_split=native_split,
        original_ref=str(source),
        local_source_path=str(processed),
        speaker_id=speaker_id,
        group_id=group_id,
        attack_id=None if label == 0 else "A01",
        metadata_source="english_processed.csv",
        materialization_mode="reuse_processed",
        expected_sha256_source=_sha256(source),
        expected_sha256_processed=_sha256(processed),
    )


def _build_eng_candidate_pool(tmp_path: Path) -> tuple[SelectionCandidate, ...]:
    candidates: list[SelectionCandidate] = []
    for idx in range(4):
        speaker = f"S{idx // 2:02d}"
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"asv-{idx:02d}",
                label=0,
                corpus="ASVspoof2024",
                native_split="train",
                speaker_id=speaker,
            )
        )
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"jmds-{idx:02d}",
                label=1,
                corpus="JMDS",
                native_split="train",
                speaker_id=speaker,
            )
        )
    for speaker in ("C1", "C2"):
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"asv-cal-{speaker}",
                label=0,
                corpus="ASVspoof2024",
                native_split="dev",
                speaker_id=speaker,
                group_id=f"grp-cal-{speaker}",
            )
        )
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"jmds-cal-{speaker}",
                label=1,
                corpus="JMDS",
                native_split="dev",
                speaker_id=speaker,
                group_id=f"grp-cal-{speaker}",
            )
        )
    for speaker in ("T1", "T2"):
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"asv-test-{speaker}",
                label=0,
                corpus="ASVspoof2024",
                native_split="dev",
                speaker_id=speaker,
                group_id=f"grp-test-{speaker}",
            )
        )
        candidates.append(
            _eng_reuse_candidate(
                tmp_path,
                candidate_id=f"jmds-test-{speaker}",
                label=1,
                corpus="JMDS",
                native_split="dev",
                speaker_id=speaker,
                group_id=f"grp-test-{speaker}",
            )
        )
    return tuple(candidates)


def test_build_xai_dataset_eng_smoke_publishes_four_artifacts(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_eng_profile(paired=True)
    manifest = tmp_path / "english_processed.csv"
    manifest.write_text("sample\n", encoding="utf-8")
    build_xai_dataset(
        _build_eng_candidate_pool(tmp_path),
        profile=profile,
        seed=5,
        layout=layout,
        archive_specs={},
        sample_rate=16_000,
        input_paths={"english_manifest": manifest},
        deps=_default_deps(),
    )
    provenance = json.loads(
        layout.provenance_json("eng").read_text(encoding="utf-8")
    )
    assert layout.samples_csv("eng").is_file()
    assert provenance["limitations"]["paired_by_speaker"] is True
    assert provenance["limitations"]["paired_utterances"] is False
    assert "speaker_balanced" not in provenance["limitations"]


def test_build_xai_dataset_zho_smoke_publishes_four_artifacts(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    profile = _tiny_zho_profile()
    pristine = tmp_path / "aishell3_metadata.csv"
    pristine.write_text("placeholder\n", encoding="utf-8")
    build_xai_dataset(
        _build_zho_candidate_pool(tmp_path),
        profile=profile,
        seed=7,
        layout=layout,
        archive_specs=_zho_archive_specs(tmp_path),
        sample_rate=16_000,
        input_paths={
            "pristine_manifest": pristine,
            "aishell_archive": tmp_path / "aishell.tgz",
        },
        deps=_default_deps(),
    )
    provenance = json.loads(
        layout.provenance_json("zho").read_text(encoding="utf-8")
    )
    assert layout.samples_csv("zho").is_file()
    assert provenance["aishell3"]["pristine_candidate_source"] == (
        "reconciled_aishell3_metadata_manifest"
    )
    assert "pristine_manifest" in {entry["key"] for entry in provenance["inputs"]}


def _zho_archive_specs(tmp_path: Path) -> dict[tuple[str, str], ArchiveSpec]:
    archive = tmp_path / "aishell.tgz"
    members = [
        (candidate.original_ref, _wav_bytes(tmp_path, value=0.18))
        for candidate in _build_zho_candidate_pool(tmp_path)
        if candidate.corpus == "AISHELL-3"
    ]
    _write_tar(archive, members)
    return {
        ("AISHELL-3", "train"): standard_archive_spec(
            corpus="AISHELL-3",
            native_split="train",
            archive_path=archive,
        ),
        ("AISHELL-3", "test"): standard_archive_spec(
            corpus="AISHELL-3",
            native_split="test",
            archive_path=archive,
        ),
    }


