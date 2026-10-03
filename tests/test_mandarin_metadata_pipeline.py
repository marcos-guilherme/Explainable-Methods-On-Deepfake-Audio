"""Tests for Mandarin metadata config, layout and extraction pipeline."""

from __future__ import annotations

import ast
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from jmds_prepare.mandarin_metadata import (
    MandarinMetadataExtractionDeps,
    extract_mandarin_metadata,
)
from jmds_prepare.mandarin_metadata_config import MandarinMetadataConfig
from jmds_prepare.pipelines.mandarin_metadata_profiling import (
    build_aishell3_profile,
    build_jmds_add_profile,
    build_mandarin_summary,
)
from jmds_prepare.pipelines.mandarin_metadata_provenance import (
    build_mandarin_metadata_provenance,
)
from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE, MandarinProfile
from jmds_prepare.sources.aishell3 import read_aishell3_metadata
from jmds_prepare.sources.jmds_add import read_mandarin_generated
from jmds_prepare.storage.mandarin_metadata_layout import MandarinMetadataLayout

FIXTURES = Path(__file__).parent / "fixtures"
AISHELL_FIXTURE = FIXTURES / "aishell3_mini.tgz"
JMDS_FIXTURE = FIXTURES / "jmds_add_metadata.csv"
JMDS_HEADER = JMDS_FIXTURE.read_text(encoding="utf-8").splitlines()[0] + "\n"


def _test_profile() -> MandarinProfile:
    return replace(
        MANDARIN_PROFILE,
        expected_generated_counts={"train": 2, "dev": 0, "eval": 0},
    )


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    jmds_root = tmp_path / "jmds"
    archive_dir = tmp_path / "sources" / "aishell3"
    output_root = tmp_path / "output"
    archive_dir.mkdir(parents=True)
    aishell_archive = archive_dir / "data_aishell3.tgz"
    shutil.copy(AISHELL_FIXTURE, aishell_archive)

    for split in ("train", "dev", "eval"):
        protocol = jmds_root / "cm_protocols" / f"open_v2_{split}.cm.csv"
        protocol.parent.mkdir(parents=True, exist_ok=True)
        if split == "train":
            shutil.copy(JMDS_FIXTURE, protocol)
        else:
            protocol.write_text(JMDS_HEADER, encoding="utf-8")

    for utt_id in ("T_0000123456", "T_0000654321"):
        wav_path = MANDARIN_PROFILE.generated_wav_path(jmds_root, "train", utt_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"RIFFxxxxWAVEfmt ")

    output_root.mkdir(exist_ok=True)
    return jmds_root, aishell_archive, output_root


def _write_config(
    path: Path,
    *,
    jmds_root: Path,
    aishell_archive: Path,
    output_root: Path,
    extra: dict | None = None,
) -> None:
    payload = {
        "jmds_root": str(jmds_root),
        "aishell_archive": str(aishell_archive),
        "output_root": str(output_root),
    }
    if extra:
        payload.update(extra)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_loads_exact_schema(tmp_path):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )

    config = MandarinMetadataConfig.load(config_path)

    assert config.jmds_root == jmds_root
    assert config.aishell_archive == aishell_archive
    assert config.output_root == output_root


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"unknown_key": "x"}, "unknown"),
        ({}, "aishell_archive"),
    ],
)
def test_config_rejects_invalid_yaml(tmp_path, extra, match):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = {
        "jmds_root": str(jmds_root),
        "aishell_archive": str(aishell_archive),
        "output_root": str(output_root),
        **extra,
    }
    if "aishell_archive" in match:
        payload.pop("aishell_archive")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        MandarinMetadataConfig.load(config_path)


def test_config_rejects_non_scalar_paths(tmp_path):
    jmds_root, aishell_archive, _output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "jmds_root": str(jmds_root),
                "aishell_archive": str(aishell_archive),
                "output_root": ["not", "a", "path"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_root"):
        MandarinMetadataConfig.load(config_path)


def test_config_validate_rejects_missing_inputs_and_nested_output(tmp_path):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path / "valid")
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )
    config.validate(MANDARIN_PROFILE)

    missing_jmds, missing_archive, missing_output = _valid_roots(tmp_path / "missing")
    missing_protocol = missing_jmds / "cm_protocols" / "open_v2_train.cm.csv"
    missing_protocol.unlink()
    missing_config = MandarinMetadataConfig(
        jmds_root=missing_jmds,
        aishell_archive=missing_archive,
        output_root=missing_output,
    )
    with pytest.raises(FileNotFoundError, match="open_v2_train"):
        missing_config.validate(MANDARIN_PROFILE)

    nested_jmds, nested_archive, _nested_output = _valid_roots(tmp_path / "nested")
    nested_config = MandarinMetadataConfig(
        jmds_root=nested_jmds,
        aishell_archive=nested_archive,
        output_root=nested_jmds / "nested_output",
    )
    with pytest.raises(ValueError) as nested_error:
        nested_config.validate(MANDARIN_PROFILE)
    assert str(nested_error.value) == (
        "output_root must not be nested inside jmds_root"
    )

    equal_jmds, equal_archive, _equal_output = _valid_roots(tmp_path / "equal")
    equal_config = MandarinMetadataConfig(
        jmds_root=equal_jmds,
        aishell_archive=equal_archive,
        output_root=equal_jmds,
    )
    with pytest.raises(ValueError, match="equal"):
        equal_config.validate(MANDARIN_PROFILE)


def test_config_validate_rejects_equal_aishell_archive(tmp_path):
    jmds_root, aishell_archive, _output_root = _valid_roots(tmp_path / "equal-archive")
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=aishell_archive,
    )

    with pytest.raises(ValueError, match="equal"):
        config.validate(MANDARIN_PROFILE)


def test_config_validate_rejects_output_nested_in_archive_parent(tmp_path):
    jmds_root, aishell_archive, _output_root = _valid_roots(tmp_path / "nested-archive")
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=aishell_archive.parent / "nested_output",
    )

    with pytest.raises(ValueError) as nested_error:
        config.validate(MANDARIN_PROFILE)
    assert str(nested_error.value) == (
        "output_root must not be nested inside aishell_archive parent"
    )


def test_config_validate_rejects_missing_archive_file(tmp_path):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    aishell_archive.unlink()
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )

    with pytest.raises(FileNotFoundError, match="aishell_archive"):
        config.validate(MANDARIN_PROFILE)


def test_layout_exposes_six_artifact_paths(tmp_path):
    layout = MandarinMetadataLayout(tmp_path / "prepared")

    assert layout.aishell3_metadata_csv == (
        tmp_path / "prepared" / "manifests" / "aishell3_metadata.csv"
    )
    assert layout.jmds_add_generated_metadata_csv == (
        tmp_path / "prepared" / "manifests" / "jmds_add_generated_metadata.csv"
    )
    assert layout.aishell3_metadata_profile_json == (
        tmp_path / "prepared" / "reports" / "aishell3_metadata_profile.json"
    )
    assert layout.jmds_add_generated_metadata_profile_json == (
        tmp_path
        / "prepared"
        / "reports"
        / "jmds_add_generated_metadata_profile.json"
    )
    assert layout.mandarin_metadata_summary_json == (
        tmp_path / "prepared" / "reports" / "mandarin_metadata_summary.json"
    )
    assert layout.mandarin_metadata_provenance_json == (
        tmp_path / "prepared" / "reports" / "mandarin_metadata_provenance.json"
    )


def test_pipeline_publishes_six_artifacts_with_reconciliation(tmp_path):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    profile = _test_profile()
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )
    layout = MandarinMetadataLayout(output_root)
    published: dict = {}

    def fake_publish(artifacts):
        published["artifacts"] = dict(artifacts)

    messages: list[str] = []
    extract_mandarin_metadata(
        config,
        MandarinMetadataExtractionDeps(
            profile=profile,
            read_aishell3=read_aishell3_metadata,
            read_jmds_generated=read_mandarin_generated,
            build_aishell_profile=build_aishell3_profile,
            build_jmds_profile=build_jmds_add_profile,
            build_summary=build_mandarin_summary,
            build_provenance=build_mandarin_metadata_provenance,
            publish_artifacts=fake_publish,
            sha256_file=lambda _path: "deadbeef",
        ),
        output=messages.append,
    )

    artifacts = published["artifacts"]
    assert len(artifacts) == 6
    assert set(artifacts) == {
        layout.aishell3_metadata_csv,
        layout.jmds_add_generated_metadata_csv,
        layout.aishell3_metadata_profile_json,
        layout.jmds_add_generated_metadata_profile_json,
        layout.mandarin_metadata_summary_json,
        layout.mandarin_metadata_provenance_json,
    }

    profile_json = json.loads(
        artifacts[layout.aishell3_metadata_profile_json].decode("utf-8")
    )
    reconciliation = profile_json["reconciliation"]
    assert reconciliation["by_split"]["test"]["content_without_wav"] == [
        "SSB00050005"
    ]

    provenance = json.loads(
        artifacts[layout.mandarin_metadata_provenance_json].decode("utf-8")
    )
    assert provenance["excluded_jmds_pristine_count"] == 4410
    assert provenance["aishell3"]["license"] == "Apache-2.0"
    assert provenance["inputs"][0]["sha256"] == "deadbeef"

    assert any("AISHELL-3" in message for message in messages)
    assert any("JMDS/ADD" in message for message in messages)
    assert any("6" in message for message in messages)


def test_pipeline_rejects_generated_count_mismatch_before_publication(tmp_path):
    jmds_root, aishell_archive, output_root = _valid_roots(tmp_path)
    profile = replace(
        MANDARIN_PROFILE,
        expected_generated_counts={"train": 99, "dev": 0, "eval": 0},
    )
    config = MandarinMetadataConfig(
        jmds_root=jmds_root,
        aishell_archive=aishell_archive,
        output_root=output_root,
    )
    published = {"called": False}

    def fake_publish(_artifacts):
        published["called"] = True

    with pytest.raises(ValueError, match="Expected 99"):
        extract_mandarin_metadata(
            config,
            MandarinMetadataExtractionDeps(
                profile=profile,
                read_aishell3=read_aishell3_metadata,
                read_jmds_generated=read_mandarin_generated,
                build_aishell_profile=build_aishell3_profile,
                build_jmds_profile=build_jmds_add_profile,
                build_summary=build_mandarin_summary,
                build_provenance=build_mandarin_metadata_provenance,
                publish_artifacts=fake_publish,
                sha256_file=lambda _path: "deadbeef",
            ),
        )

    assert published["called"] is False


def test_mandarin_metadata_module_has_no_cli_or_english_imports():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "jmds_prepare"
        / "mandarin_metadata.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "jmds_prepare.cli" not in imports
    assert "jmds_prepare.profiles.english" not in imports


def test_metadata_extraction_still_has_no_language_literals():
    source = Path("src/jmds_prepare/pipelines/metadata_extraction.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in {"por", "zho"}:
            raise AssertionError("metadata_extraction must not branch on language codes")
