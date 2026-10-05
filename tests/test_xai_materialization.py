"""Tests for selective XAI materialization, archive extraction and audit."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from jmds_prepare.audio import process_audio
from jmds_prepare.core.hashing import sha256_file, sha256_file_preserving_times
from jmds_prepare.core.xai_sample import XaiSample
from jmds_prepare.pipelines.xai_audit import (
    SILENCE_THRESHOLD,
    audit_processed_sample,
    summarize_sample_audits,
)
from jmds_prepare.pipelines.xai_materialization import (
    CORAA_TRAIN_RAR_VOLUME_NAMES,
    ArchiveResolutionError,
    ArchiveSpec,
    DestinationConflictError,
    MaterializationError,
    PreflightError,
    derive_sample_id,
    inventory_archive_spec,
    materialize_selected,
    preflight_materialization,
    standard_archive_spec,
)
from jmds_prepare.pipelines.xai_selection import (
    SelectedCandidate,
    SelectionCandidate,
)
from jmds_prepare.storage.xai_layout import XaiLayout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
    frames: int = 400,
    value: float = 0.25,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        samples = np.full(frames, value, dtype=np.float64)
    else:
        samples = np.full((frames, channels), value, dtype=np.float64)
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


def _selected(
    candidate: SelectionCandidate,
    *,
    role: str = "train",
    rank: int = 0,
) -> SelectedCandidate:
    return SelectedCandidate(
        candidate=candidate,
        role=role,
        selection_seed=7,
        selection_rank=rank,
        selection_reason="fixture",
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
    role: str = "train",
    rank: int = 0,
    sample_rate: int = 8_000,
) -> SelectedCandidate:
    source = tmp_path / "sources" / f"{candidate_id}.wav"
    _write_wav(source, sample_rate=sample_rate)
    candidate = SelectionCandidate(
        candidate_id=candidate_id,
        language=language,
        label=label,
        corpus=corpus,
        native_split=native_split,
        original_ref=str(source),
        local_source_path=str(source),
        speaker_id="spk-1",
        group_id=None,
        attack_id="A01" if label == 1 else None,
        metadata_source="fixture",
        materialization_mode="process_source",
        expected_sha256_source=None,
        expected_sha256_processed=None,
    )
    return _selected(candidate, role=role, rank=rank)


def _english_reuse_candidate(
    tmp_path: Path,
    *,
    candidate_id: str = "T_0000000001",
    label: int = 0,
    role: str = "train",
    rank: int = 0,
) -> SelectedCandidate:
    source = tmp_path / "raw" / f"{candidate_id}.flac"
    processed = tmp_path / "processed" / f"{candidate_id}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"original-flac-bytes")
    _write_wav(processed, sample_rate=16_000, value=0.12)
    candidate = SelectionCandidate(
        candidate_id=candidate_id,
        language="eng",
        label=label,
        corpus="ASVspoof2024" if label == 0 else "JMDS",
        native_split="train",
        original_ref=str(source),
        local_source_path=str(processed),
        speaker_id="S_0001",
        group_id=None,
        attack_id=None if label == 0 else "A01",
        metadata_source="english_processed.csv",
        materialization_mode="reuse_processed",
        expected_sha256_source=_sha256(source),
        expected_sha256_processed=_sha256(processed),
    )
    return _selected(candidate, role=role, rank=rank)


class _FakeRunner:
    def __init__(
        self,
        *,
        list_output: str = "",
        list_outputs: dict[str, str] | None = None,
        returncode: int = 0,
        extract_bytes: bytes = b"",
        member_bytes: dict[str, bytes] | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self._list_output = list_output
        self._list_outputs = list_outputs
        self._returncode = returncode
        self._extract_bytes = extract_bytes
        self._member_bytes = member_bytes or {}
        self.extracted_paths: list[Path] = []
        self.listed_members: list[str] = []
        self.listed_members_by_archive: dict[Path, list[str]] = {}

    def __call__(
        self,
        args: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        **_: Any,
    ):
        return self.run(
            args,
            check=check,
            capture_output=capture_output,
            text=text,
        )

    def run(
        self,
        args: list[str],
        *,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        **_: Any,
    ):
        self.calls.append(list(args))
        if len(args) >= 2 and args[1] == "lb":
            archive_path = Path(args[-1])
            if self._list_outputs is None:
                stdout = (
                    self._list_output
                    if archive_path.name.lower().endswith(".part1.rar")
                    else ""
                )
            else:
                stdout = self._list_outputs.get(archive_path.name, "")
            return self._result(check=check, args=args, stdout=stdout)
        if len(args) >= 2 and args[1] == "x":
            destination_dir = Path(str(args[-1]).rstrip("\\/"))
            archive_path = Path(args[args.index("-idq") + 1])
            list_arg = next(arg for arg in args if arg.startswith("@"))
            list_path = Path(list_arg[1:])
            listed_members = [
                line.strip()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.listed_members.extend(listed_members)
            self.listed_members_by_archive[archive_path] = listed_members
            for member_name in listed_members:
                normalized_member_name = member_name.replace("\\", "/")
                destination = destination_dir.joinpath(*normalized_member_name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = self._member_bytes.get(member_name, self._extract_bytes)
                destination.write_bytes(payload)
                self.extracted_paths.append(destination)
            return self._result(check=check, args=args)

        return self._result(check=check, args=args)

    def _result(
        self,
        *,
        check: bool = False,
        args: list[str] | None = None,
        stdout: str | None = None,
    ):
        class _Result:
            def __init__(self, runner: _FakeRunner, stdout: str | None) -> None:
                self.returncode = runner._returncode
                self.stdout = runner._list_output if stdout is None else stdout
                self.stderr = ""

        result = _Result(self, stdout)
        if check and result.returncode != 0:
            raise RuntimeError(f"command failed: {args}")
        return result


def _write_coraa_train_volumes(directory: Path, *, include_all: bool = True) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    volumes: list[Path] = []
    for index, name in enumerate(CORAA_TRAIN_RAR_VOLUME_NAMES, start=1):
        path = directory / name
        if include_all or index == 1:
            path.write_bytes(f"fake-rar-{index}".encode())
        volumes.append(path)
    return tuple(volumes)


def test_standard_archive_spec_maps_known_corpora(tmp_path: Path):
    assert standard_archive_spec(
        corpus="AISHELL-3",
        native_split="train",
        archive_path=Path("aishell3.tgz"),
    ) == ArchiveSpec("AISHELL-3", "train", "tar", (Path("aishell3.tgz"),))
    assert standard_archive_spec(
        corpus="CORAA",
        native_split="dev",
        archive_path=Path("coraa-dev.zip"),
    ) == ArchiveSpec("CORAA", "dev", "zip", (Path("coraa-dev.zip"),))
    part1 = tmp_path / "train.part1.rar"
    part1.write_bytes(b"fake")
    spec = standard_archive_spec(
        corpus="CORAA",
        native_split="train",
        archive_path=part1,
    )
    assert spec.kind == "rar"
    assert spec.volume_paths == tuple(
        tmp_path / name for name in CORAA_TRAIN_RAR_VOLUME_NAMES
    )


def test_derive_sample_id_is_deterministic_and_filesystem_safe():
    first = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id="train/clips/0001.wav",
    )
    second = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id="train/clips/0001.wav",
    )
    shuffled = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id="train/clips/9999.wav",
    )

    assert first == second
    assert first != shuffled
    assert len(first) <= 200
    XaiLayout.validate_sample_id(first)


def test_derive_sample_id_avoids_slug_collisions():
    slash = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id="a/b",
    )
    hyphen = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id="a-b",
    )

    assert slash != hyphen
    assert len(slash) <= 200
    assert len(hyphen) <= 200


def test_derive_sample_id_truncates_long_tokens_safely():
    long_id = "x" * 500
    sample_id = derive_sample_id(
        language="por",
        role="train",
        label=0,
        corpus="CORAA",
        candidate_id=long_id,
    )

    assert len(sample_id) <= 200
    XaiLayout.validate_sample_id(sample_id)


def test_tar_inventory_resolves_requested_refs_only(tmp_path: Path):
    archive = tmp_path / "aishell.tgz"
    _write_tar(
        archive,
        [
            ("train/wav/spk1/utt-001.wav", b"one"),
            ("train/wav/spk1/utt-002.wav", b"two"),
            ("../unsafe.wav", b"unsafe"),
        ],
    )
    spec = ArchiveSpec("AISHELL-3", "train", "tar", (archive,))

    inventory = inventory_archive_spec(spec)

    assert inventory.resolve_ref("train/wav/spk1/utt-001.wav").member_name == (
        "train/wav/spk1/utt-001.wav"
    )
    assert inventory.duplicate_refs == ()
    assert "../unsafe.wav" in inventory.rejected_unsafe_members


def test_tar_preflight_rejects_duplicate_before_cache(tmp_path: Path):
    archive = tmp_path / "coraa.tar"
    _write_tar(
        archive,
        [
            ("train/duplicate.wav", b"dup1"),
            ("train/duplicate.wav", b"dup2"),
        ],
    )
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "train", "tar", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="dup",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/duplicate.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        ),
    ]

    with pytest.raises(PreflightError, match="Duplicate archive member"):
        preflight_materialization(selected, layout, {("CORAA", "train"): spec})

    assert not layout.staging_dir.exists()


def test_tar_preflight_rejects_missing_before_cache(tmp_path: Path):
    archive = tmp_path / "coraa.tar"
    _write_tar(archive, [("train/a.wav", b"a"), ("../escape.wav", b"escape")])
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "train", "tar", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="missing",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/missing.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        ),
    ]

    with pytest.raises(PreflightError, match="Missing archive member"):
        preflight_materialization(selected, layout, {("CORAA", "train"): spec})

    assert not layout.staging_dir.exists()


def test_zip_extracts_only_requested_members(tmp_path: Path):
    selected_wav = tmp_path / "selected.wav"
    ignored_wav = tmp_path / "ignored.wav"
    _write_wav(selected_wav, sample_rate=8_000, value=0.2)
    _write_wav(ignored_wav, sample_rate=8_000, value=0.3)
    archive = tmp_path / "coraa-dev.zip"
    _write_zip(
        archive,
        [
            ("dev/clips/001.wav", selected_wav.read_bytes()),
            ("dev/clips/002.wav", ignored_wav.read_bytes()),
        ],
    )
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "dev", "zip", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="dev",
                original_ref="dev/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    samples = materialize_selected(
        selected,
        layout,
        {("CORAA", "dev"): spec},
        sample_rate=16_000,
    )

    assert len(samples) == 1
    output = layout.sample_path("por", "train", 0, samples[0].sample_id)
    assert output.exists()
    assert not list(layout.staging_dir.glob("run-*"))
    assert not any(layout.data_dir.rglob("*.tmp*"))
    info = sf.info(output)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_rar_batch_extracts_two_members_with_one_list_and_one_extract(
    tmp_path: Path,
):
    volume_dir = tmp_path / "train_dividido"
    volumes = _write_coraa_train_volumes(volume_dir)
    first_wav = tmp_path / "first.wav"
    second_wav = tmp_path / "second.wav"
    _write_wav(first_wav, sample_rate=8_000, value=0.15)
    _write_wav(second_wav, sample_rate=8_000, value=0.20)
    runner = _FakeRunner(
        list_output="train/clips/001.wav\ntrain/clips/002.wav\n",
        member_bytes={
            "train/clips/001.wav": first_wav.read_bytes(),
            "train/clips/002.wav": second_wav.read_bytes(),
        },
    )
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    layout = XaiLayout(tmp_path / "out")

    def _archive_candidate(candidate_id: str, original_ref: str, rank: int) -> SelectedCandidate:
        return _selected(
            SelectionCandidate(
                candidate_id=candidate_id,
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref=original_ref,
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            ),
            rank=rank,
        )

    selected = [
        _archive_candidate("001", "train/clips/001.wav", 0),
        _archive_candidate("002", "train/clips/002.wav", 1),
    ]

    samples = materialize_selected(
        selected,
        layout,
        {("CORAA", "train"): spec},
        sample_rate=16_000,
        archive_runner=runner,
        unrar_executable="UnRAR",
    )

    assert len(samples) == 2
    list_calls = [call for call in runner.calls if len(call) >= 2 and call[1] == "lb"]
    extract_calls = [call for call in runner.calls if len(call) >= 2 and call[1] == "x"]
    assert len(list_calls) == 5
    assert len(extract_calls) == 1
    assert list_calls == [
        ["UnRAR", "lb", "-p-", str(volume)]
        for volume in volumes
    ]
    assert extract_calls[0][0] == "UnRAR"
    assert extract_calls[0][1] == "x"
    assert "-o-" in extract_calls[0]
    assert "-p-" in extract_calls[0]
    assert "-idq" in extract_calls[0]
    assert extract_calls[0][extract_calls[0].index("-idq") + 1] == str(volumes[0])
    listfile_arg = next(arg for arg in extract_calls[0] if arg.startswith("@"))
    assert listfile_arg.startswith("@")
    assert set(runner.listed_members) == {
        "train/clips/001.wav",
        "train/clips/002.wav",
    }
    staging_root = Path(str(extract_calls[0][-1]).rstrip("\\/"))
    assert sorted(
        path.relative_to(staging_root).as_posix() for path in runner.extracted_paths
    ) == ["train/clips/001.wav", "train/clips/002.wav"]
    assert not list(layout.staging_dir.glob("run-*"))


def test_rar_lists_all_volumes_once_and_extracts_only_required_volumes(
    tmp_path: Path,
):
    volumes = _write_coraa_train_volumes(tmp_path / "train_dividido")
    first_wav = tmp_path / "first.wav"
    second_wav = tmp_path / "second.wav"
    ignored_wav = tmp_path / "ignored.wav"
    _write_wav(first_wav, sample_rate=8_000, value=0.15)
    _write_wav(second_wav, sample_rate=8_000, value=0.20)
    _write_wav(ignored_wav, sample_rate=8_000, value=0.25)
    first_raw = r"train\CORAL\part1.wav"
    second_raw = r"train\CORAL\part2.wav"
    ignored_raw = r"train\CORAL\ignored.wav"
    runner = _FakeRunner(
        list_outputs={
            volumes[0].name: f"{first_raw}\n",
            volumes[1].name: f"{second_raw}\n",
            volumes[2].name: f"{ignored_raw}\n",
        },
        member_bytes={
            first_raw: first_wav.read_bytes(),
            second_raw: second_wav.read_bytes(),
            ignored_raw: ignored_wav.read_bytes(),
        },
    )
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    layout = XaiLayout(tmp_path / "out")

    def archive_candidate(candidate_id: str, original_ref: str, rank: int):
        return _selected(
            SelectionCandidate(
                candidate_id=candidate_id,
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref=original_ref,
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            ),
            rank=rank,
        )

    selected = [
        archive_candidate("part1", "train/CORAL/part1.wav", 0),
        archive_candidate("part2", "train/CORAL/part2.wav", 1),
    ]

    samples = materialize_selected(
        selected,
        layout,
        {("CORAA", "train"): spec},
        archive_runner=runner,
    )

    list_calls = [call for call in runner.calls if call[1] == "lb"]
    extract_calls = [call for call in runner.calls if call[1] == "x"]
    assert [Path(call[-1]) for call in list_calls] == list(volumes)
    assert [Path(call[call.index("-idq") + 1]) for call in extract_calls] == [
        volumes[0],
        volumes[1],
    ]
    assert runner.listed_members_by_archive == {
        volumes[0]: [first_raw],
        volumes[1]: [second_raw],
    }
    assert {sample.original_ref: sample.sha256_source for sample in samples} == {
        "train/CORAL/part1.wav": sha256_file(first_wav),
        "train/CORAL/part2.wav": sha256_file(second_wav),
    }
    assert all(Path(sample.processed_path).is_file() for sample in samples)
    assert not list(layout.staging_dir.glob("run-*"))


def test_rar_backslash_listing_resolves_posix_ref_and_extracts_raw_member(
    tmp_path: Path,
):
    volumes = _write_coraa_train_volumes(tmp_path / "train_dividido")
    source_wav = tmp_path / "source.wav"
    _write_wav(source_wav, sample_rate=8_000, value=0.15)
    raw_member_name = r"train\CORAL\1037_CO_bfamdl10.wav"
    normalized_ref = "train/CORAL/1037_CO_bfamdl10.wav"
    runner = _FakeRunner(
        list_output=f"{raw_member_name}\n",
        member_bytes={raw_member_name: source_wav.read_bytes()},
    )
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="1037_CO_bfamdl10",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref=normalized_ref,
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    samples = materialize_selected(
        selected,
        layout,
        {("CORAA", "train"): spec},
        archive_runner=runner,
    )

    assert runner.listed_members == [raw_member_name]
    extract_call = next(call for call in runner.calls if call[1] == "x")
    staging_root = Path(str(extract_call[-1]).rstrip("\\/"))
    assert [
        path.relative_to(staging_root).as_posix() for path in runner.extracted_paths
    ] == [normalized_ref]
    assert len(samples) == 1
    assert Path(samples[0].processed_path).is_file()


@pytest.mark.parametrize(
    "unsafe_member",
    [
        r"..\escape.wav",
        r"C:\escape.wav",
        r"\absolute\escape.wav",
    ],
)
def test_rar_inventory_rejects_unsafe_backslash_members_after_normalization(
    tmp_path: Path,
    unsafe_member: str,
):
    volumes = _write_coraa_train_volumes(tmp_path / "train_dividido")
    runner = _FakeRunner(list_outputs={volumes[1].name: f"{unsafe_member}\n"})
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)

    inventory = inventory_archive_spec(spec, archive_runner=runner)

    assert unsafe_member in inventory.rejected_unsafe_members
    assert not inventory.members_by_ref


def test_rar_inventory_detects_duplicates_after_path_normalization(tmp_path: Path):
    volumes = _write_coraa_train_volumes(tmp_path / "train_dividido")
    runner = _FakeRunner(
        list_outputs={
            volumes[0].name: "train/CORAL/sample.wav\n",
            volumes[1].name: "train\\CORAL\\sample.wav\n",
        }
    )
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)

    inventory = inventory_archive_spec(spec, archive_runner=runner)

    assert inventory.duplicate_refs == ("train/CORAL/sample.wav",)
    with pytest.raises(ArchiveResolutionError, match="Duplicate archive member"):
        inventory.resolve_ref("train/CORAL/sample.wav")


def test_rar_missing_sibling_volume_fails_before_runner(tmp_path: Path):
    volume_dir = tmp_path / "train_dividido"
    volumes = _write_coraa_train_volumes(volume_dir, include_all=False)
    runner = _FakeRunner()
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    with pytest.raises(PreflightError, match="Missing archive volume"):
        preflight_materialization(
            selected,
            layout,
            {("CORAA", "train"): spec},
            archive_runner=runner,
        )

    assert runner.calls == []
    assert not layout.staging_dir.exists()
    assert not layout.data_dir.exists()


def test_unrar_executable_missing_surfaces_preflight_error(tmp_path: Path):
    volume_dir = tmp_path / "train_dividido"
    volumes = _write_coraa_train_volumes(volume_dir)

    def missing_runner(*_args, **_kwargs):
        raise FileNotFoundError("UnRAR not installed")

    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    with pytest.raises(PreflightError, match="UnRAR executable not found"):
        preflight_materialization(
            selected,
            layout,
            {("CORAA", "train"): spec},
            archive_runner=missing_runner,
            unrar_executable="UnRAR.exe",
        )


def test_rar_preflight_surfaces_runner_listing_errors(tmp_path: Path):
    volume_dir = tmp_path / "train_dividido"
    volumes = _write_coraa_train_volumes(volume_dir)
    runner = _FakeRunner(returncode=1)
    spec = ArchiveSpec("CORAA", "train", "rar", volumes)
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    with pytest.raises(PreflightError, match="listing"):
        preflight_materialization(
            selected,
            layout,
            {("CORAA", "train"): spec},
            archive_runner=runner,
        )


def test_local_source_is_processed_to_pcm16_mono_16k(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_local_candidate(tmp_path, sample_rate=22_050)]

    samples = materialize_selected(selected, layout, {})

    assert len(samples) == 1
    sample = samples[0]
    path = Path(sample.processed_path)
    info = sf.info(path)
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert sample.sha256_processed == sha256_file(path)
    assert sample.sha256_source == sha256_file_preserving_times(
        Path(selected[0].candidate.local_source_path)
    )


def test_english_reuse_does_not_alter_processed_bytes_or_mtime(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_english_reuse_candidate(tmp_path)]
    processed = Path(selected[0].candidate.local_source_path)
    before_mtime_ns = processed.stat().st_mtime_ns
    before_bytes = processed.read_bytes()

    samples = materialize_selected(selected, layout, {})

    assert samples[0].processed_path == str(processed)
    assert processed.stat().st_mtime_ns == before_mtime_ns
    assert processed.read_bytes() == before_bytes
    assert not layout.data_dir.exists() or not any(layout.data_dir.rglob("*.wav"))


def test_hash_mismatch_fails_preflight_without_writes(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_english_reuse_candidate(tmp_path)]
    bad = replace(
        selected[0],
        candidate=replace(
            selected[0].candidate,
            expected_sha256_processed="0" * 64,
        ),
    )

    with pytest.raises(PreflightError, match="sha256_processed"):
        preflight_materialization([bad], layout, {})

    assert not layout.staging_dir.exists()


def test_content_idempotent_rerun_and_divergent_conflict(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_local_candidate(tmp_path, candidate_id="utt-a")]

    first = materialize_selected(selected, layout, {})
    second = materialize_selected(selected, layout, {})
    assert first == second

    path = Path(first[0].processed_path)
    path.write_bytes(b"tampered")
    with pytest.raises(DestinationConflictError):
        materialize_selected(selected, layout, {})


def test_transactional_batch_rollback_leaves_no_final_wavs_on_second_failure(
    tmp_path: Path,
    monkeypatch,
):
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _local_candidate(tmp_path, candidate_id="first"),
        _local_candidate(tmp_path, candidate_id="second", rank=1),
    ]
    calls = {"count": 0}

    def flaky_process_audio(source: Path, destination: Path, sample_rate: int):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated failure on second candidate")
        return process_audio(source, destination, sample_rate)

    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_materialization.process_audio",
        flaky_process_audio,
    )

    with pytest.raises(OSError, match="simulated failure on second candidate"):
        materialize_selected(selected, layout, {})

    assert calls["count"] == 2
    assert not layout.data_dir.exists() or not any(layout.data_dir.rglob("*.wav"))
    assert not list(layout.staging_dir.glob("run-*"))


def test_run_staging_is_removed_after_success(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_local_candidate(tmp_path, candidate_id="utt-ok")]

    materialize_selected(selected, layout, {})

    assert not list(layout.staging_dir.glob("run-*"))


def test_staging_and_archive_temporaries_are_cleaned_on_processing_error(
    tmp_path: Path,
    monkeypatch,
):
    selected_wav = tmp_path / "selected.wav"
    _write_wav(selected_wav, sample_rate=8_000, value=0.2)
    archive = tmp_path / "coraa-dev.zip"
    _write_zip(archive, [("dev/clips/001.wav", selected_wav.read_bytes())])
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "dev", "zip", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="dev",
                original_ref="dev/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]
    calls = {"count": 0}

    def broken_process_audio(source: Path, destination: Path, sample_rate: int):
        calls["count"] += 1
        assert source.is_file()
        raise OSError("simulated failure after staging")

    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_materialization.process_audio",
        broken_process_audio,
    )

    with pytest.raises(OSError, match="simulated failure after staging"):
        materialize_selected(
            selected,
            layout,
            {("CORAA", "dev"): spec},
            sample_rate=16_000,
        )

    assert calls["count"] == 1
    assert not list(layout.staging_dir.glob("run-*"))
    assert not list(layout.data_dir.rglob("*.tmp*")) if layout.data_dir.exists() else True


def test_no_writes_when_any_reference_in_batch_fails(tmp_path: Path):
    archive = tmp_path / "batch.tar"
    _write_tar(archive, [("train/good.wav", b"good")])
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "train", "tar", (archive,))
    good_local = tmp_path / "local.wav"
    _write_wav(good_local)
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="local",
                language="por",
                label=1,
                corpus="MLAAD",
                native_split="train",
                original_ref=str(good_local),
                local_source_path=str(good_local),
                speaker_id="spk",
                group_id=None,
                attack_id="A01",
                metadata_source="fixture",
                materialization_mode="process_source",
            )
        ),
        _selected(
            SelectionCandidate(
                candidate_id="missing",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="train/missing.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        ),
    ]

    with pytest.raises(PreflightError, match="missing"):
        materialize_selected(
            selected,
            layout,
            {("CORAA", "train"): spec},
        )

    assert not any(layout.data_dir.rglob("*.wav")) if layout.data_dir.exists() else True


def test_materialize_produces_valid_xai_samples_in_canonical_order(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _local_candidate(tmp_path, candidate_id="b", role="calibration", rank=1),
        _local_candidate(tmp_path, candidate_id="a", role="train", rank=0),
    ]

    samples = materialize_selected(selected, layout, {})

    assert [item.selection_rank for item in samples] == [0, 1]
    assert all(isinstance(item, XaiSample) for item in samples)
    assert samples[0].sample_id == derive_sample_id(
        language="por",
        role="train",
        label=1,
        corpus="MLAAD",
        candidate_id="a",
    )


def test_audit_reports_duration_rms_peak_and_silence(tmp_path: Path):
    wav = tmp_path / "sample.wav"
    _write_wav(wav, sample_rate=16_000, frames=800, value=0.5)

    audit = audit_processed_sample(wav, sample_id="sample-a")

    assert audit.sample_id == "sample-a"
    assert audit.duration_seconds == pytest.approx(800 / 16_000)
    assert audit.peak == pytest.approx(0.5)
    assert audit.rms > 0
    assert 0 <= audit.silence_fraction <= 1
    assert audit.silence_threshold == SILENCE_THRESHOLD


def test_audit_summary_is_json_safe_by_corpus_class_and_role(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _local_candidate(tmp_path, candidate_id="a", role="train", label=1),
        _local_candidate(tmp_path, candidate_id="b", role="train", label=1),
    ]
    samples = materialize_selected(selected, layout, {})
    audits = [
        audit_processed_sample(Path(sample.processed_path), sample_id=sample.sample_id)
        for sample in samples
    ]

    summary = summarize_sample_audits(samples, audits)

    assert summary["sample_count"] == 2
    assert summary["silence_definition"]["threshold"] == SILENCE_THRESHOLD
    group = next(
        item
        for item in summary["groups"]
        if item["corpus"] == "MLAAD" and item["label"] == 1 and item["role"] == "train"
    )
    assert group["count"] == 2
    assert group["duration_seconds"]["mean"] > 0


def test_preflight_rejects_duplicate_candidate_ids_sample_ids_and_destinations(
    tmp_path: Path,
):
    layout = XaiLayout(tmp_path / "out")
    first = _local_candidate(tmp_path, candidate_id="shared-id")
    second = _local_candidate(tmp_path, candidate_id="shared-id", rank=1)

    with pytest.raises(PreflightError, match="Duplicate candidate_id"):
        preflight_materialization([first, second], layout, {})


def test_preflight_rejects_unsafe_archive_ref_without_writing_cache(tmp_path: Path):
    archive = tmp_path / "unsafe.tar"
    _write_tar(archive, [("../escape.wav", b"unsafe")])
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "train", "tar", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="unsafe",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="train",
                original_ref="../escape.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    with pytest.raises(PreflightError, match="Unsafe original_ref|Missing archive member"):
        preflight_materialization(selected, layout, {("CORAA", "train"): spec})

    assert not layout.staging_dir.exists()
    assert not layout.data_dir.exists()


def test_reuse_processed_requires_existing_original_ref(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_english_reuse_candidate(tmp_path)]
    missing_source = replace(
        selected[0],
        candidate=replace(
            selected[0].candidate,
            original_ref=str(tmp_path / "raw" / "missing.flac"),
        ),
    )

    with pytest.raises(PreflightError, match="Missing original source"):
        preflight_materialization([missing_source], layout, {})


def test_reuse_processed_rejects_non_canonical_processed_format(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [_english_reuse_candidate(tmp_path)]
    processed = Path(selected[0].candidate.local_source_path)
    _write_wav(processed, sample_rate=8_000, value=0.12)

    with pytest.raises(PreflightError, match="Invalid reused processed audio"):
        preflight_materialization(selected, layout, {})


def test_selective_extraction_does_not_stage_ignored_members(
    tmp_path: Path,
    monkeypatch,
):
    selected_wav = tmp_path / "selected.wav"
    ignored_wav = tmp_path / "ignored.wav"
    _write_wav(selected_wav, sample_rate=8_000, value=0.2)
    _write_wav(ignored_wav, sample_rate=8_000, value=0.3)
    archive = tmp_path / "coraa-dev.zip"
    _write_zip(
        archive,
        [
            ("dev/clips/001.wav", selected_wav.read_bytes()),
            ("dev/clips/002.wav", ignored_wav.read_bytes()),
        ],
    )
    layout = XaiLayout(tmp_path / "out")
    spec = ArchiveSpec("CORAA", "dev", "zip", (archive,))
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="dev",
                original_ref="dev/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]
    observed_refs: list[str] = []
    import jmds_prepare.pipelines.xai_materialization as materialization_module

    original_extract = materialization_module._extract_archive_batch

    def spy_extract(spec, refs, inventory, staging_root, **kwargs):
        observed_refs.extend(refs)
        return original_extract(
            spec,
            refs,
            inventory,
            staging_root,
            **kwargs,
        )

    monkeypatch.setattr(
        materialization_module,
        "_extract_archive_batch",
        spy_extract,
    )
    materialize_selected(
        selected,
        layout,
        {("CORAA", "dev"): spec},
        sample_rate=16_000,
    )

    assert observed_refs == ["dev/clips/001.wav"]


def test_archive_resolution_error_when_spec_missing(tmp_path: Path):
    layout = XaiLayout(tmp_path / "out")
    selected = [
        _selected(
            SelectionCandidate(
                candidate_id="001",
                language="por",
                label=0,
                corpus="CORAA",
                native_split="dev",
                original_ref="dev/clips/001.wav",
                local_source_path=None,
                speaker_id=None,
                group_id=None,
                attack_id=None,
                metadata_source="fixture",
                materialization_mode="extract_archive",
            )
        )
    ]

    with pytest.raises(PreflightError, match="ArchiveSpec"):
        preflight_materialization(selected, layout, {})
