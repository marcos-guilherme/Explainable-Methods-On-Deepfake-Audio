"""Tests for Portuguese metadata config, layout and extraction pipeline."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from jmds_prepare.metadata_config import PortugueseMetadataConfig
from jmds_prepare.pipelines.metadata_extraction import (
    MetadataExtractionDeps,
    extract_portuguese_metadata,
)
from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE, PortugueseProfile
from jmds_prepare.sources.coraa import CORAA_COLUMNS
from jmds_prepare.sources.jmds import JMDS_COLUMNS
from jmds_prepare.storage.metadata_layout import PortugueseMetadataLayout

_CORAA_USAGE_STATEMENT = (
    "CC BY-NC-ND 4.0: noncommercial use only. Originals are preserved "
    "byte-identical. Adapted material may be produced and reproduced for "
    "this research but must not be shared; no publication or redistribution "
    "of derived audio, features or subsets."
)


def _test_profile() -> PortugueseProfile:
    return replace(
        PORTUGUESE_PROFILE,
        expected_coraa_counts={"train": 3, "dev": 0, "test": 0},
        expected_generated_counts={"train": 2, "dev": 0, "eval": 0},
    )


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    jmds_root = tmp_path / "jmds"
    coraa_root = tmp_path / "coraa"
    output_root = tmp_path / "output"
    for split in ("train", "dev", "eval"):
        protocol = jmds_root / "cm_protocols" / f"open_v2_{split}.cm.csv"
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text("protocol\n", encoding="utf-8")
    for split, filename in PORTUGUESE_PROFILE.coraa_metadata_files.items():
        metadata = coraa_root / filename
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("metadata\n", encoding="utf-8")
    output_root.mkdir(exist_ok=True)
    return jmds_root, coraa_root, output_root


def _write_config(
    path: Path,
    *,
    jmds_root: Path,
    coraa_root: Path,
    output_root: Path,
    extra: dict | None = None,
) -> None:
    payload = {
        "jmds_root": str(jmds_root),
        "coraa_root": str(coraa_root),
        "output_root": str(output_root),
    }
    if extra:
        payload.update(extra)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_loads_exact_schema(tmp_path):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=output_root,
    )

    config = PortugueseMetadataConfig.load(config_path)

    assert config.jmds_root == jmds_root
    assert config.coraa_root == coraa_root
    assert config.output_root == output_root


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"unknown_key": "x"}, "unknown"),
        ({}, "coraa_root"),
    ],
)
def test_config_rejects_invalid_yaml(tmp_path, extra, match):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = {
        "jmds_root": str(jmds_root),
        "coraa_root": str(coraa_root),
        "output_root": str(output_root),
        **extra,
    }
    if "coraa_root" in match:
        payload.pop("coraa_root")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        PortugueseMetadataConfig.load(config_path)


def test_config_rejects_non_scalar_paths(tmp_path):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "jmds_root": str(jmds_root),
                "coraa_root": str(coraa_root),
                "output_root": ["not", "a", "path"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_root"):
        PortugueseMetadataConfig.load(config_path)


def test_config_validate_rejects_missing_inputs_and_nested_output(tmp_path):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path / "valid")
    config = PortugueseMetadataConfig(
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=output_root,
    )
    config.validate(PORTUGUESE_PROFILE)

    missing_jmds, missing_coraa, missing_output = _valid_roots(tmp_path / "missing")
    missing_protocol = missing_jmds / "cm_protocols" / "open_v2_train.cm.csv"
    missing_protocol.unlink()
    missing_config = PortugueseMetadataConfig(
        jmds_root=missing_jmds,
        coraa_root=missing_coraa,
        output_root=missing_output,
    )
    with pytest.raises(FileNotFoundError, match="open_v2_train"):
        missing_config.validate(PORTUGUESE_PROFILE)

    nested_jmds, nested_coraa, _nested_output = _valid_roots(tmp_path / "nested")
    nested_config = PortugueseMetadataConfig(
        jmds_root=nested_jmds,
        coraa_root=nested_coraa,
        output_root=nested_jmds / "nested_output",
    )
    with pytest.raises(ValueError) as nested_error:
        nested_config.validate(PORTUGUESE_PROFILE)
    assert str(nested_error.value) == (
        "output_root must not be nested inside jmds_root"
    )

    equal_jmds, equal_coraa, _equal_output = _valid_roots(tmp_path / "equal")
    equal_config = PortugueseMetadataConfig(
        jmds_root=equal_jmds,
        coraa_root=equal_coraa,
        output_root=equal_jmds,
    )
    with pytest.raises(ValueError, match="equal"):
        equal_config.validate(PORTUGUESE_PROFILE)


def test_config_validate_rejects_equal_coraa_root(tmp_path):
    jmds_root, coraa_root, _output_root = _valid_roots(tmp_path / "equal-coraa")
    config = PortugueseMetadataConfig(
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=coraa_root,
    )

    with pytest.raises(ValueError, match="equal"):
        config.validate(PORTUGUESE_PROFILE)


def test_config_validate_rejects_output_nested_in_coraa_root(tmp_path):
    jmds_root, coraa_root, _output_root = _valid_roots(tmp_path / "nested-coraa")
    config = PortugueseMetadataConfig(
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=coraa_root / "nested_output",
    )

    with pytest.raises(ValueError) as nested_error:
        config.validate(PORTUGUESE_PROFILE)
    assert str(nested_error.value) == (
        "output_root must not be nested inside coraa_root"
    )


def test_config_load_rejects_syntactically_invalid_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("jmds_root: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML"):
        PortugueseMetadataConfig.load(config_path)


def test_layout_exposes_six_artifact_paths(tmp_path):
    layout = PortugueseMetadataLayout(tmp_path / "prepared")

    assert layout.coraa_metadata_csv == (
        tmp_path / "prepared" / "manifests" / "coraa_metadata.csv"
    )
    assert layout.jmds_mlaad_generated_metadata_csv == (
        tmp_path / "prepared" / "manifests" / "jmds_mlaad_generated_metadata.csv"
    )
    assert layout.coraa_metadata_profile_json == (
        tmp_path / "prepared" / "reports" / "coraa_metadata_profile.json"
    )
    assert layout.jmds_mlaad_generated_metadata_profile_json == (
        tmp_path
        / "prepared"
        / "reports"
        / "jmds_mlaad_generated_metadata_profile.json"
    )
    assert layout.portuguese_metadata_summary_json == (
        tmp_path / "prepared" / "reports" / "portuguese_metadata_summary.json"
    )
    assert layout.portuguese_metadata_provenance_json == (
        tmp_path / "prepared" / "reports" / "portuguese_metadata_provenance.json"
    )


def test_pipeline_concatenates_splits_and_publishes_once(tmp_path):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path)
    profile = _test_profile()
    config = PortugueseMetadataConfig(
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=output_root,
    )
    layout = PortugueseMetadataLayout(output_root)

    coraa_calls: list[tuple] = []
    jmds_calls: list[tuple] = []
    published: dict = {}

    def fake_read_coraa(path, *, split, profile):
        coraa_calls.append((path, split))
        assert split in profile.coraa_splits
        count = profile.expected_coraa_counts[split]
        columns = [*CORAA_COLUMNS, "split", "metadata_source_file", "metadata_source_row"]
        if count == 0:
            return pd.DataFrame(columns=columns)
        row = {column: f"{split}-{column}" for column in CORAA_COLUMNS}
        row.update(
            split=split,
            metadata_source_file=Path(path).name,
            metadata_source_row="2",
        )
        return pd.DataFrame([row] * count)

    def fake_read_jmds(path, *, split, jmds_root, profile):
        jmds_calls.append((path, split, jmds_root))
        count = profile.expected_generated_counts[split]
        columns = [
            *JMDS_COLUMNS,
            "split",
            "metadata_source_file",
            "metadata_source_row",
            "audio_path",
        ]
        if count == 0:
            return pd.DataFrame(columns=columns)
        row = {column: f"{split}-{column}" for column in JMDS_COLUMNS}
        row.update(
            split=split,
            metadata_source_file=Path(path).name,
            metadata_source_row="2",
            audio_path=f"/wav/{split}.wav",
        )
        return pd.DataFrame([row] * count)

    def fake_publish(artifacts):
        published["artifacts"] = dict(artifacts)

    messages: list[str] = []
    extract_portuguese_metadata(
        config,
        MetadataExtractionDeps(
            profile=profile,
            read_coraa=fake_read_coraa,
            read_jmds_generated=fake_read_jmds,
            build_coraa_profile=lambda frame: {"source": "CORAA", "sample_count": len(frame)},
            build_jmds_profile=lambda frame: {
                "source": "JMDS_MLAAD",
                "sample_count": len(frame),
            },
            build_summary=lambda coraa, jmds: {
                "comparison_limitations": {"paired_samples": False},
                "sources": {
                    "CORAA": {"sample_count": len(coraa)},
                    "JMDS_MLAAD": {"sample_count": len(jmds)},
                },
            },
            build_provenance=lambda **kwargs: {
                "excluded_jmds_pristine_count": profile.excluded_jmds_pristine_count,
                "coraa": {
                    "usage_policy": {
                        "preserve_originals": True,
                        "commercial_use_permitted": False,
                        "adapted_material_sharing_permitted": False,
                        "statement": _CORAA_USAGE_STATEMENT,
                    }
                },
                "inputs": sorted(kwargs["input_paths"]),
            },
            publish_artifacts=fake_publish,
            sha256_file=lambda _path: "deadbeef",
        ),
        output=messages.append,
    )

    assert [split for _, split in coraa_calls] == list(profile.coraa_splits)
    assert [split for _, split, _ in jmds_calls] == list(profile.protocol_splits)
    artifacts = published["artifacts"]
    assert len(artifacts) == 6
    assert set(artifacts) == {
        layout.coraa_metadata_csv,
        layout.jmds_mlaad_generated_metadata_csv,
        layout.coraa_metadata_profile_json,
        layout.jmds_mlaad_generated_metadata_profile_json,
        layout.portuguese_metadata_summary_json,
        layout.portuguese_metadata_provenance_json,
    }
    assert any("CORAA" in message and "3" in message for message in messages)
    assert any("JMDS" in message and "2" in message for message in messages)
    assert any("6" in message for message in messages)

    coraa_csv = artifacts[layout.coraa_metadata_csv].decode("utf-8")
    assert coraa_csv.endswith("\n")
    assert "\r\n" not in coraa_csv
    provenance = json.loads(
        artifacts[layout.portuguese_metadata_provenance_json].decode("utf-8")
    )
    assert provenance["excluded_jmds_pristine_count"] == 1000
    assert provenance["coraa"]["usage_policy"]["preserve_originals"] is True
    assert provenance["coraa"]["usage_policy"]["commercial_use_permitted"] is False
    assert (
        provenance["coraa"]["usage_policy"]["adapted_material_sharing_permitted"]
        is False
    )
    assert provenance["coraa"]["usage_policy"]["statement"] == _CORAA_USAGE_STATEMENT


def test_pipeline_rejects_count_mismatch_before_publication(tmp_path):
    jmds_root, coraa_root, output_root = _valid_roots(tmp_path)
    profile = _test_profile()
    config = PortugueseMetadataConfig(
        jmds_root=jmds_root,
        coraa_root=coraa_root,
        output_root=output_root,
    )

    traceability = ("split", "metadata_source_file", "metadata_source_row")
    empty_coraa = pd.DataFrame(columns=[*CORAA_COLUMNS, *traceability])
    empty_jmds = pd.DataFrame(
        columns=[*JMDS_COLUMNS, *traceability, "audio_path"]
    )

    def short_coraa(_path, *, split, profile):
        if split != "train":
            return empty_coraa.copy()
        return pd.DataFrame({column: ["x"] for column in CORAA_COLUMNS}).assign(
            split="train",
            metadata_source_file="metadata_train_final.csv",
            metadata_source_row="2",
        )

    published = {"called": False}

    def fake_publish(_artifacts):
        published["called"] = True

    with pytest.raises(ValueError, match="Expected 3"):
        extract_portuguese_metadata(
            config,
            MetadataExtractionDeps(
                profile=profile,
                read_coraa=short_coraa,
                read_jmds_generated=lambda *_a, **_k: empty_jmds.copy(),
                build_coraa_profile=lambda _f: {},
                build_jmds_profile=lambda _f: {},
                build_summary=lambda *_a: {},
                build_provenance=lambda **_k: {},
                publish_artifacts=fake_publish,
                sha256_file=lambda _p: "x",
            ),
        )

    assert published["called"] is False


def test_pipeline_json_uses_allow_nan_false(tmp_path):
    from jmds_prepare.pipelines.metadata_extraction import _serialize_json

    with pytest.raises(ValueError, match="nan"):
        _serialize_json({"bad": float("nan")})


def test_extract_two_source_metadata_has_no_language_literals():
    source = Path("src/jmds_prepare/pipelines/metadata_extraction.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in {"por", "zho"}:
            raise AssertionError("metadata_extraction must not branch on language codes")
    assert "def extract_two_source_metadata" in source


def test_metadata_extraction_has_no_cli_or_english_imports():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "jmds_prepare"
        / "pipelines"
        / "metadata_extraction.py"
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
    assert '"por"' not in module_path.read_text(encoding="utf-8")
