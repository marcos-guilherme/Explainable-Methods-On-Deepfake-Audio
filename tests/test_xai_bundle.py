"""Focused tests for the portable XAI VM bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from jmds_prepare import xai_bundle as cli
from jmds_prepare.core.xai_sample import XAI_SAMPLE_COLUMNS
from jmds_prepare.pipelines import xai_bundle as bundle_pipeline
from jmds_prepare.pipelines.xai_bundle import (
    BundleValidationError,
    build_xai_bundle,
    verify_xai_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0" * sample_width * channels * 32)


def _row(language: str, source: Path, *, sample_id: str | None = None) -> dict[str, str]:
    return {
        "sample_id": sample_id or f"{language}-sample",
        "language": language,
        "role": "test",
        "label": "0",
        "corpus": "fixture",
        "native_split": "test",
        "original_ref": f"original/{language}.wav",
        "processed_path": str(source),
        "speaker_id": "",
        "group_id": "",
        "attack_id": "",
        "sha256_source": "a" * 64,
        "sha256_processed": _sha256(source),
        "selection_seed": "7",
        "selection_rank": "0",
        "selection_reason": "fixture",
        "selection_source": "fixture.csv",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=XAI_SAMPLE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture_manifests(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    manifests: dict[str, Path] = {}
    sources: dict[str, Path] = {}
    for index, language in enumerate(("eng", "por", "zho")):
        source = tmp_path / "source" / f"{language}.wav"
        _write_wav(source)
        source.write_bytes(source.read_bytes() + (b"\0" * index))
        manifest = tmp_path / "input" / language / "xai_samples.csv"
        _write_manifest(manifest, [_row(language, source)])
        manifests[language] = manifest
        sources[language] = source
    return manifests, sources


def _load_receipt(bundle: Path) -> dict:
    return json.loads((bundle / "bundle_receipt.json").read_text(encoding="utf-8"))


def _write_receipt(bundle: Path, receipt: dict) -> None:
    (bundle / "bundle_receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )


def _refresh_manifest_receipt(
    bundle: Path,
    receipt: dict,
    *,
    language: str,
) -> None:
    relative = f"manifests/{language}/xai_samples.csv"
    manifest = bundle / relative
    for entry in receipt["files"]:
        if entry["path"] == relative:
            entry["sha256"] = _sha256(manifest)
            entry["size_bytes"] = manifest.stat().st_size
            break
    receipt["manifests"][language]["sha256"] = _sha256(manifest)
    receipt["manifests"][language]["size_bytes"] = manifest.stat().st_size
    receipt["counts"]["receipt_file_bytes"] = sum(
        entry["size_bytes"] for entry in receipt["files"]
    )


def test_build_rewrites_paths_preserves_fields_and_writes_verifiable_receipt(
    tmp_path: Path,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "xai-vm-bundle"

    result = build_xai_bundle(
        manifests,
        output,
        allow_noncanonical_counts=True,
    )

    assert result.file_count == 3
    assert result.output == output
    bundled_manifest = output / "manifests" / "eng" / "xai_samples.csv"
    with bundled_manifest.open(encoding="utf-8", newline="") as stream:
        bundled_row = next(csv.DictReader(stream))
    assert bundled_row["processed_path"] == "data/eng/test/bonafide/eng-sample.wav"
    assert bundled_row["original_ref"] == "original/eng.wav"
    assert bundled_row["sha256_processed"] == _sha256(
        output / bundled_row["processed_path"]
    )

    receipt = json.loads((output / "bundle_receipt.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "jmds_prepare.xai_bundle"
    assert receipt["schema_version"] == 1
    assert receipt["bundle_version"] == 1
    assert receipt["counts"]["samples_total"] == 3
    assert receipt["counts"]["samples_by_language"] == {"eng": 1, "por": 1, "zho": 1}
    assert {entry["path"] for entry in receipt["files"]} == {
        "data/eng/test/bonafide/eng-sample.wav",
        "data/por/test/bonafide/por-sample.wav",
        "data/zho/test/bonafide/zho-sample.wav",
        "manifests/eng/xai_samples.csv",
        "manifests/por/xai_samples.csv",
        "manifests/zho/xai_samples.csv",
    }
    verified = verify_xai_bundle(output, allow_noncanonical_counts=True)
    assert verified.file_count == 3


def test_build_dry_run_reports_audio_files_and_bytes_without_output(tmp_path: Path) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"

    result = build_xai_bundle(
        manifests,
        output,
        allow_noncanonical_counts=True,
        dry_run=True,
        archive=True,
    )

    assert result.file_count == 3
    assert result.total_bytes == sum(path.stat().st_size for path in sources.values())
    assert not output.exists()
    assert not output.with_name(f"{output.name}.tar.zst").exists()
    assert not output.with_name(f"{output.name}.tar.zst.sha256").exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_cli_build_archive_writes_valid_sidecar_and_extractable_bundle(
    tmp_path: Path,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "portable-bundle"
    args = [
        "build",
        "--eng-manifest",
        str(manifests["eng"]),
        "--por-manifest",
        str(manifests["por"]),
        "--zho-manifest",
        str(manifests["zho"]),
        "--output",
        str(output),
        "--allow-noncanonical-counts",
        "--archive",
    ]

    assert cli.main(args) == 0

    archive = tmp_path / "portable-bundle.tar.zst"
    sidecar = tmp_path / "portable-bundle.tar.zst.sha256"
    assert sidecar.read_text(encoding="ascii") == (
        f"{_sha256(archive)}  {archive.name}\n"
    )

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    member_names: list[str] = []
    member_file_names: list[str] = []
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    member_names.append(member.name)
                    if member.isfile():
                        member_file_names.append(member.name)
                    tar.extract(member, extracted)

    assert {Path(name).parts[0] for name in member_names} == {output.name}
    archived_files = {
        Path(name).relative_to(output.name).as_posix()
        for name in member_file_names
    }
    bundle_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert archived_files == bundle_files
    verified = verify_xai_bundle(
        extracted / output.name,
        allow_noncanonical_counts=True,
    )
    assert verified.file_count == 3


def test_archive_normalizes_recursive_tar_metadata(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    bundle = tmp_path / "portable-bundle"
    build_xai_bundle(
        manifests,
        bundle,
        allow_noncanonical_counts=True,
        archive=True,
    )

    archive = tmp_path / "portable-bundle.tar.zst"
    members: list[tarfile.TarInfo] = []
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                members.extend(tar)

    assert members
    assert all(member.uid == 0 and member.gid == 0 for member in members)
    assert all(member.uname == "" and member.gname == "" for member in members)
    assert all(member.mode == (0o755 if member.isdir() else 0o644) for member in members)


@pytest.mark.parametrize("existing_name", ["bundle.tar.zst", "bundle.tar.zst.sha256"])
def test_archive_build_refuses_existing_archive_outputs(
    tmp_path: Path,
    existing_name: str,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    existing = tmp_path / existing_name
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        build_xai_bundle(
            manifests,
            output,
            allow_noncanonical_counts=True,
            archive=True,
        )

    assert not output.exists()
    assert existing.read_bytes() == b"keep"


def test_archive_failure_leaves_bundle_but_no_partial_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"

    def fail_tar_open(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected archive failure")

    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_bundle.tarfile.open",
        fail_tar_open,
    )
    with pytest.raises(OSError, match="injected archive failure"):
        build_xai_bundle(
            manifests,
            output,
            allow_noncanonical_counts=True,
            archive=True,
        )

    assert output.is_dir()
    assert not (tmp_path / "bundle.tar.zst").exists()
    assert not (tmp_path / "bundle.tar.zst.sha256").exists()
    assert not list(tmp_path.glob(".bundle.tar.zst.tmp-*"))


def test_cli_archive_existing_bundle_writes_extractable_verified_archive(
    tmp_path: Path,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    bundle = tmp_path / "retryable-bundle"
    build_xai_bundle(manifests, bundle, allow_noncanonical_counts=True)

    assert cli.main(
        ["archive", str(bundle), "--allow-noncanonical-counts"]
    ) == 0

    archive = tmp_path / "retryable-bundle.tar.zst"
    sidecar = tmp_path / "retryable-bundle.tar.zst.sha256"
    assert sidecar.read_text(encoding="ascii") == (
        f"{_sha256(archive)}  {archive.name}\n"
    )
    extracted = tmp_path / "standalone-extracted"
    extracted.mkdir()
    with archive.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(extracted)
    verified = verify_xai_bundle(
        extracted / bundle.name,
        allow_noncanonical_counts=True,
    )
    assert verified.file_count == 3


@pytest.mark.parametrize("existing_name", ["bundle.tar.zst", "bundle.tar.zst.sha256"])
def test_archive_existing_bundle_refuses_existing_outputs(
    tmp_path: Path,
    existing_name: str,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    bundle = tmp_path / "bundle"
    build_xai_bundle(manifests, bundle, allow_noncanonical_counts=True)
    existing = tmp_path / existing_name
    existing.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        bundle_pipeline.archive_xai_bundle(
            bundle,
            allow_noncanonical_counts=True,
        )

    assert existing.read_bytes() == b"keep"


def test_cli_archive_rejects_invalid_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "invalid-bundle"
    bundle.mkdir()

    assert cli.main(
        ["archive", str(bundle), "--allow-noncanonical-counts"]
    ) == 1
    assert "error: Invalid bundle receipt" in capsys.readouterr().out
    assert not (tmp_path / "invalid-bundle.tar.zst").exists()


def test_cli_archive_reports_zstandard_failure_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_archive(*_args: object, **_kwargs: object) -> None:
        raise zstandard.ZstdError("injected zstandard failure")

    monkeypatch.setattr(cli, "archive_xai_bundle", fail_archive, raising=False)

    assert cli.main(["archive", str(tmp_path / "bundle")]) == 1
    assert capsys.readouterr().out == "error: injected zstandard failure\n"


def test_build_requires_exactly_three_language_manifests(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    del manifests["zho"]
    with pytest.raises(BundleValidationError, match="exactly eng, por, zho"):
        build_xai_bundle(
            manifests,
            tmp_path / "bundle",
            allow_noncanonical_counts=True,
        )


def test_build_requires_canonical_counts_unless_explicitly_allowed(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    with pytest.raises(BundleValidationError, match="6022"):
        build_xai_bundle(manifests, tmp_path / "bundle")


def test_build_accepts_relative_source_with_ancestor_receipt(tmp_path: Path) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    source_bundle = tmp_path / "source-bundle"
    relative = "data/eng/test/bonafide/eng-sample.wav"
    bundled_source = source_bundle / relative
    bundled_source.parent.mkdir(parents=True)
    bundled_source.write_bytes(sources["eng"].read_bytes())
    (source_bundle / "bundle_receipt.json").write_text("{}", encoding="utf-8")
    manifest = source_bundle / "manifests" / "eng" / "xai_samples.csv"
    _write_manifest(manifest, [_row("eng", bundled_source) | {"processed_path": relative}])
    manifests["eng"] = manifest

    result = build_xai_bundle(
        manifests,
        tmp_path / "output",
        allow_noncanonical_counts=True,
        dry_run=True,
    )

    assert result.file_count == 3


def test_build_rejects_relative_source_without_receipt(tmp_path: Path) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    relative = "source/eng.wav"
    _write_manifest(
        manifests["eng"],
        [_row("eng", sources["eng"]) | {"processed_path": relative}],
    )

    with pytest.raises(BundleValidationError, match="requires bundle_receipt.json"):
        build_xai_bundle(
            manifests,
            tmp_path / "output",
            allow_noncanonical_counts=True,
            dry_run=True,
        )


def test_build_rejects_relative_source_traversal_to_existing_file(
    tmp_path: Path,
) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    source_bundle = tmp_path / "source-bundle"
    source_bundle.mkdir()
    (source_bundle / "bundle_receipt.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.wav"
    outside.write_bytes(sources["eng"].read_bytes())
    manifest = source_bundle / "manifests" / "eng" / "xai_samples.csv"
    _write_manifest(
        manifest,
        [_row("eng", outside) | {"processed_path": "../outside.wav"}],
    )
    manifests["eng"] = manifest

    with pytest.raises(BundleValidationError, match="not confined"):
        build_xai_bundle(
            manifests,
            tmp_path / "output",
            allow_noncanonical_counts=True,
            dry_run=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_field", "Missing required columns"),
        ("language", "language mismatch"),
        ("duplicate_id", "Duplicate sample_id"),
        ("duplicate_destination", "Duplicate destination"),
        ("missing_source", "Missing processed file"),
        ("hash", "sha256_processed mismatch"),
        ("format", "PCM_16 mono 16 kHz"),
    ],
)
def test_build_rejects_invalid_manifests_and_sources(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    if mutation == "missing_field":
        rows = [_row("eng", sources["eng"])]
        rows[0].pop("corpus")
        with manifests["eng"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[c for c in XAI_SAMPLE_COLUMNS if c != "corpus"])
            writer.writeheader()
            writer.writerows(rows)
    elif mutation == "language":
        _write_manifest(manifests["eng"], [_row("por", sources["eng"])])
    elif mutation in {"duplicate_id", "duplicate_destination"}:
        first = _row("eng", sources["eng"])
        second_source = tmp_path / "source" / "eng-2.wav"
        _write_wav(second_source)
        second = _row(
            "eng",
            second_source,
            sample_id=first["sample_id"] if mutation == "duplicate_id" else "eng-other",
        )
        if mutation == "duplicate_destination":
            second["role"] = first["role"]
            second["label"] = first["label"]
            second["sample_id"] = first["sample_id"]
        _write_manifest(manifests["eng"], [first, second])
    elif mutation == "missing_source":
        sources["eng"].unlink()
    elif mutation == "hash":
        sources["eng"].write_bytes(sources["eng"].read_bytes() + b"changed")
    elif mutation == "format":
        _write_wav(sources["eng"], sample_rate=8_000)
        row = _row("eng", sources["eng"])
        _write_manifest(manifests["eng"], [row])

    with pytest.raises(BundleValidationError, match=message):
        build_xai_bundle(
            manifests,
            tmp_path / "bundle",
            allow_noncanonical_counts=True,
        )


def test_failed_build_cleans_staging_and_does_not_publish(tmp_path: Path) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    sources["zho"].write_bytes(b"not a wav")
    row = _row("zho", sources["zho"])
    _write_manifest(manifests["zho"], [row])
    output = tmp_path / "bundle"

    with pytest.raises(BundleValidationError):
        build_xai_bundle(manifests, output, allow_noncanonical_counts=True)

    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_copy_failure_cleans_sibling_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    real_copy = __import__("shutil").copyfile
    calls = 0

    def fail_second_copy(source: Path, destination: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture copy failure")
        return str(real_copy(source, destination))

    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_bundle.shutil.copyfile",
        fail_second_copy,
    )
    with pytest.raises(OSError, match="fixture copy failure"):
        build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_build_rejects_insufficient_disk_space_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_bundle.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(BundleValidationError, match="disk space"):
        build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_archive_build_reserves_space_for_bundle_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, sources = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    bundle_only_required = (
        sum(path.stat().st_size for path in sources.values())
        + sum(path.stat().st_size for path in manifests.values())
        + 4096
    )
    monkeypatch.setattr(
        "jmds_prepare.pipelines.xai_bundle.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=bundle_only_required),
    )

    with pytest.raises(BundleValidationError, match="disk space"):
        build_xai_bundle(
            manifests,
            output,
            allow_noncanonical_counts=True,
            archive=True,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_standalone_archive_checks_space_before_compression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    bundle = tmp_path / "bundle"
    build_xai_bundle(manifests, bundle, allow_noncanonical_counts=True)
    compressor_started = False

    class FailIfStarted:
        def __init__(self) -> None:
            nonlocal compressor_started
            compressor_started = True

    monkeypatch.setattr(bundle_pipeline.zstandard, "ZstdCompressor", FailIfStarted)
    monkeypatch.setattr(
        bundle_pipeline.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(BundleValidationError, match="disk space"):
        bundle_pipeline.archive_xai_bundle(
            bundle,
            allow_noncanonical_counts=True,
        )

    assert not compressor_started
    assert not (tmp_path / "bundle.tar.zst").exists()
    assert not list(tmp_path.glob(".bundle.tar.zst.tmp-*"))


def test_build_never_replaces_existing_output(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("size_hash", "Size mismatch"),
        ("traversal", "not confined"),
        ("duplicate", "Duplicate receipt path"),
        ("wav", "PCM_16 mono 16 kHz"),
    ],
)
def test_verify_rejects_receipt_manifest_and_audio_corruption(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    receipt_path = output / "bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audio_path = output / "data" / "eng" / "test" / "bonafide" / "eng-sample.wav"
    if corruption == "size_hash":
        audio_path.write_bytes(audio_path.read_bytes() + b"changed")
    elif corruption == "traversal":
        receipt["files"][0]["path"] = "../outside.wav"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif corruption == "duplicate":
        receipt["files"].append(dict(receipt["files"][0]))
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        _write_wav(audio_path, channels=2)
        for entry in receipt["files"]:
            if entry["path"] == "data/eng/test/bonafide/eng-sample.wav":
                entry["size_bytes"] = audio_path.stat().st_size
                entry["sha256"] = _sha256(audio_path)
        manifest = output / "manifests/eng/xai_samples.csv"
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["sha256_processed"] = _sha256(audio_path)
        _write_manifest(manifest, rows)
        _refresh_manifest_receipt(output, receipt, language="eng")
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(BundleValidationError, match=message):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_unknown_receipt_kind(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    receipt = _load_receipt(output)
    audio_entry = next(entry for entry in receipt["files"] if entry["kind"] == "audio")
    audio_entry["kind"] = "sidecar"
    _write_receipt(output, receipt)

    with pytest.raises(BundleValidationError, match="Unsupported receipt kind"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_receipt_listed_file_not_referenced_by_manifests(
    tmp_path: Path,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    extra = output / "extra.txt"
    extra.write_text("listed but unreferenced", encoding="utf-8")
    receipt = _load_receipt(output)
    receipt["files"].append(
        {
            "path": "extra.txt",
            "sha256": _sha256(extra),
            "size_bytes": extra.stat().st_size,
            "kind": "manifest",
            "language": "eng",
        }
    )
    receipt["counts"]["receipt_file_count"] += 1
    receipt["counts"]["receipt_file_bytes"] += extra.stat().st_size
    _write_receipt(output, receipt)

    with pytest.raises(BundleValidationError, match="Receipt paths mismatch"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_missing_bundled_file(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    (output / "data/eng/test/bonafide/eng-sample.wav").unlink()

    with pytest.raises(BundleValidationError, match="Missing bundled file"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_extra_unlisted_file(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    (output / "extra.txt").write_text("not in receipt", encoding="utf-8")

    with pytest.raises(BundleValidationError, match="Receipt file set mismatch"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_modified_bundled_manifest(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    manifest = output / "manifests/eng/xai_samples.csv"
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(BundleValidationError, match="Size mismatch"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("sample_count", "Manifest sample_count mismatch"),
        ("totals", "Receipt aggregate counts mismatch"),
    ],
)
def test_verify_rejects_mismatched_counts(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    receipt = _load_receipt(output)
    if field == "sample_count":
        receipt["manifests"]["eng"]["sample_count"] += 1
    else:
        receipt["counts"]["samples_total"] += 1
    _write_receipt(output, receipt)

    with pytest.raises(BundleValidationError, match=message):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_rejects_processed_path_outside_expected_layout(tmp_path: Path) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    manifest = output / "manifests/eng/xai_samples.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["processed_path"] = "data/por/test/bonafide/eng-sample.wav"
    _write_manifest(manifest, rows)
    receipt = _load_receipt(output)
    _refresh_manifest_receipt(output, receipt, language="eng")
    _write_receipt(output, receipt)

    with pytest.raises(BundleValidationError, match="Noncanonical processed_path"):
        verify_xai_bundle(output, allow_noncanonical_counts=True)


def test_verify_hashes_each_audio_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    build_xai_bundle(manifests, output, allow_noncanonical_counts=True)
    real_sha256_file = bundle_pipeline.sha256_file
    hashed_paths: list[Path] = []

    def counting_sha256_file(path: Path) -> str:
        hashed_paths.append(Path(path))
        return real_sha256_file(path)

    monkeypatch.setattr(bundle_pipeline, "sha256_file", counting_sha256_file)

    verify_xai_bundle(output, allow_noncanonical_counts=True)

    audio_paths = [path for path in hashed_paths if path.suffix == ".wav"]
    assert len(audio_paths) == 3
    assert len(set(audio_paths)) == 3


def test_cli_build_dry_run_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifests, _ = _fixture_manifests(tmp_path)
    output = tmp_path / "bundle"
    args = [
        "build",
        "--eng-manifest",
        str(manifests["eng"]),
        "--por-manifest",
        str(manifests["por"]),
        "--zho-manifest",
        str(manifests["zho"]),
        "--output",
        str(output),
        "--allow-noncanonical-counts",
        "--dry-run",
    ]
    assert cli.main(args) == 0
    assert "3 files" in capsys.readouterr().out
    assert not output.exists()

    args.remove("--dry-run")
    assert cli.main(args) == 0
    assert cli.main(["verify", str(output), "--allow-noncanonical-counts"]) == 0
