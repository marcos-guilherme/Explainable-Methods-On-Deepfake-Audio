"""Tests for the English profile and the per-source protocol/metadata modules.

They pin the canonical profile values, the legacy names that existing imports
and monkeypatches depend on, and the import layers of ``profiles``/``sources``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from jmds_prepare import cli, manifest, protocols, zenodo
from jmds_prepare.config import PreparationConfig
from jmds_prepare.profiles.english import ENGLISH_PROFILE, EvalExclusionPolicy
from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE
from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE
from jmds_prepare.sources import asvspoof
from jmds_prepare.sources import jmds_common
from jmds_prepare.sources import zenodo as zenodo_source

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "jmds_prepare"


# --- profiles.english ---------------------------------------------------------


def test_english_profile_holds_the_canonical_english_values(tmp_path):
    profile = ENGLISH_PROFILE

    assert profile.language_code == "eng"
    assert profile.language_slug == "english"
    assert profile.dataset == "ASVspoof2024"
    assert profile.generated_dir_name == "English_ASVspoof2024_Generated"
    assert profile.source_slug == "asvspoof5"
    assert profile.metadata_source == "JMDS+ASVspoof5-verified"
    assert profile.splits == ("train", "dev")
    assert profile.protocol_splits == ("train", "dev", "eval")
    assert dict(profile.asv_protocol_files) == {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }
    assert dict(profile.expected_pristine) == {"train": 18_797, "dev": 15_667}
    assert dict(profile.expected_generated) == {"train": 65_424, "dev": 54_808}
    assert profile.zenodo_record_id == 14498691
    assert profile.required_archives == (
        "flac_T_aa.tar",
        "flac_T_ab.tar",
        "flac_T_ac.tar",
        "flac_T_ad.tar",
        "flac_T_ae.tar",
        "flac_D_aa.tar",
        "flac_D_ab.tar",
        "flac_D_ac.tar",
    )
    assert profile.eval_exclusion == EvalExclusionPolicy(
        split="eval",
        status="excluded_known_id_mismatch",
        reason=(
            "JMDS eval IDs do not exactly correspond to the official "
            "ASVspoof5 eval protocol"
        ),
    )
    assert profile.generated_root(tmp_path) == (
        tmp_path / "dataset" / "English_ASVspoof2024_Generated"
    )


def test_english_profile_is_deeply_immutable():
    profile = ENGLISH_PROFILE

    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.language_code = "por"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.eval_exclusion.status = "included"  # type: ignore[misc]
    for mapping in (
        profile.asv_protocol_files,
        profile.expected_pristine,
        profile.expected_generated,
    ):
        with pytest.raises(TypeError):
            mapping["train"] = "changed"  # type: ignore[index]
    assert isinstance(profile.required_archives, tuple)


def test_config_default_record_id_is_unchanged(tmp_path):
    config = PreparationConfig(jmds_root=tmp_path, data_root=tmp_path)

    assert config.zenodo_record_id == 14498691


def test_manifest_asv_evidence_only_accepts_english_splits(tmp_path):
    config = PreparationConfig(
        jmds_root=tmp_path, data_root=tmp_path, asvspoof_protocol_root=tmp_path
    )

    with pytest.raises(KeyError):
        manifest._asv_evidence_by_id(config, "eval")


# --- legacy aliases -----------------------------------------------------------


def test_cli_expected_counts_and_protocol_files_are_patchable_dict_copies():
    for alias, canonical in (
        (cli.EXPECTED_PRISTINE, ENGLISH_PROFILE.expected_pristine),
        (cli.EXPECTED_GENERATED, ENGLISH_PROFILE.expected_generated),
        (cli.ASV_PROTOCOL_FILES, ENGLISH_PROFILE.asv_protocol_files),
    ):
        assert type(alias) is dict
        assert alias == dict(canonical)


def test_legacy_record_names_are_the_source_objects():
    assert cli.get_record_files is zenodo.get_record_files
    assert zenodo.get_record_files is zenodo_source.get_record_files
    assert cli._ORIGINAL_GET_RECORD_FILES is zenodo_source.get_record_files
    assert cli.REQUIRED_ARCHIVES is zenodo.REQUIRED_ARCHIVES
    assert zenodo.REQUIRED_ARCHIVES is ENGLISH_PROFILE.required_archives
    assert zenodo_source.REQUIRED_ARCHIVES is ENGLISH_PROFILE.required_archives


def test_legacy_correspondence_report_is_the_source_class():
    assert cli.CorrespondenceReport is protocols.CorrespondenceReport
    assert protocols.CorrespondenceReport is asvspoof.CorrespondenceReport


# --- import layers ------------------------------------------------------------


def _package_imports(path: Path) -> set[str]:
    """Return the ``jmds_prepare`` modules imported by ``path`` (absolute names)."""
    package_parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
    package_parts = package_parts[:-1]
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(
                alias.name for alias in node.names
                if alias.name.startswith("jmds_prepare")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join([*parts, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            if module.startswith("jmds_prepare"):
                imported.add(module)
    return imported


def test_profiles_layer_depends_only_on_itself():
    files = sorted((PACKAGE_ROOT / "profiles").glob("*.py"))
    assert {path.name for path in files} >= {
        "__init__.py",
        "english.py",
        "mandarin.py",
        "portuguese.py",
    }
    allowed = ("jmds_prepare.profiles", "jmds_prepare.core")
    for path in files:
        for module in _package_imports(path):
            assert module.startswith(allowed), (path, module)


def test_portuguese_profile_is_exported_from_profiles_package():
    from jmds_prepare.profiles import PORTUGUESE_PROFILE as exported

    assert exported is PORTUGUESE_PROFILE
    assert exported.generated_dir_name == "Portugese_MLAAD_Generated"


def test_mandarin_profile_is_exported_from_profiles_package():
    from jmds_prepare.profiles import MANDARIN_PROFILE as exported

    assert exported is MANDARIN_PROFILE
    assert exported.generated_dir_name == "Chinese_ADD_Generated"


def test_sources_layer_depends_only_on_core_profiles_and_sources():
    files = sorted((PACKAGE_ROOT / "sources").glob("*.py"))
    assert {path.name for path in files} >= {
        "__init__.py",
        "jmds.py",
        "jmds_common.py",
        "asvspoof.py",
        "zenodo.py",
    }
    allowed = ("jmds_prepare.core", "jmds_prepare.profiles", "jmds_prepare.sources")
    for path in files:
        for module in _package_imports(path):
            assert module.startswith(allowed), (path, module)

    jmds_imports = _package_imports(PACKAGE_ROOT / "sources" / "jmds.py")
    assert not jmds_imports & {
        "jmds_prepare.sources.asvspoof",
        "jmds_prepare.sources.zenodo",
    }
    assert "jmds_prepare.sources.jmds_common" in jmds_imports


def test_jmds_common_exposes_shared_validators():
    assert callable(jmds_common.validate_schema)
    assert callable(jmds_common.validate_split)
    assert callable(jmds_common.validate_unique_ids)
    assert callable(jmds_common.validate_utterance_ids)


def test_importing_profiles_and_sources_loads_no_legacy_or_pipeline_module():
    script = (
        "import json, sys\n"
        "import jmds_prepare.profiles.english\n"
        "import jmds_prepare.profiles.mandarin\n"
        "import jmds_prepare.profiles.portuguese\n"
        "import jmds_prepare.sources.jmds\n"
        "import jmds_prepare.sources.jmds_common\n"
        "import jmds_prepare.sources.asvspoof\n"
        "import jmds_prepare.sources.zenodo\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('jmds_prepare'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))

    assert not loaded & {
        "jmds_prepare.cli",
        "jmds_prepare.config",
        "jmds_prepare.manifest",
        "jmds_prepare.protocols",
        "jmds_prepare.zenodo",
        "jmds_prepare.audio",
        "jmds_prepare.audit",
        "jmds_prepare.extraction",
        "jmds_prepare.storage",
    }
