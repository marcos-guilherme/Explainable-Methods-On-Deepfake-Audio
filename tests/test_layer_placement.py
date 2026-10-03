"""Where canonical implementations live and how the cli aliases them.

Reusable primitives belong to ``core``/``storage``; ``pipelines`` keeps only
orchestration. The private ``jmds_prepare.cli`` aliases fall into two groups:
*injected* aliases are read by a cli wrapper on every call (patching them in
``cli`` propagates), while *import-only* aliases exist solely so old imports
keep working (to change behaviour, patch the module that calls them).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from jmds_prepare import cli
from jmds_prepare.config import PreparationConfig
from jmds_prepare.core import atomic
from jmds_prepare.pipelines import acquisition, acquisition_plan, common, validation
from jmds_prepare.sources import zenodo as zenodo_source
from jmds_prepare.storage import audio_validation, estimates

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "jmds_prepare"

CANONICAL = {
    "write_json_atomic": (atomic, "jmds_prepare.core.atomic"),
    "is_valid_flac": (audio_validation, "jmds_prepare.storage.audio_validation"),
    "acquisition_storage_state": (estimates, "jmds_prepare.storage.estimates"),
    "audio_storage_estimate": (estimates, "jmds_prepare.storage.estimates"),
    "bulk_generated_pcm16_bytes": (estimates, "jmds_prepare.storage.estimates"),
    "measure_audio_header": (estimates, "jmds_prepare.storage.estimates"),
    "existing_ancestor": (estimates, "jmds_prepare.storage.estimates"),
}

# cli alias -> canonical object. Patching these in ``cli`` changes no command.
IMPORT_ONLY_ALIASES = {
    "_write_json_atomic": atomic.write_json_atomic,
    "_bulk_generated_pcm16_bytes": estimates.bulk_generated_pcm16_bytes,
    "_measure_audio_header": estimates.measure_audio_header,
    "_existing_ancestor": estimates.existing_ancestor,
    "_asvspoof_root": validation.asvspoof_root,
    "_archive_split": acquisition.archive_split,
    "_missing_inventory_message": acquisition.missing_inventory_message,
}

# cli alias -> canonical object. A cli wrapper injects the current value.
INJECTED_ALIASES = {
    "_is_valid_flac": audio_validation.is_valid_flac,
    "_validate_inventory": acquisition.validate_inventory,
    "_audio_storage_estimate": estimates.audio_storage_estimate,
}


def _config(tmp_path: Path) -> PreparationConfig:
    return PreparationConfig(
        jmds_root=tmp_path / "jmds",
        data_root=tmp_path / "data",
        asvspoof_protocol_root=tmp_path / "asv",
    )


def _package_imports(path: Path) -> set[str]:
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


# --- canonical placement ----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CANONICAL))
def test_primitives_are_defined_in_core_or_storage(name):
    module, qualified = CANONICAL[name]
    implementation = getattr(module, name)

    assert implementation.__module__ == qualified


def test_pipelines_reexport_rather_than_redefine_primitives():
    assert common.write_json_atomic is atomic.write_json_atomic
    for name in (
        "acquisition_storage_state",
        "audio_storage_estimate",
        "bulk_generated_pcm16_bytes",
        "measure_audio_header",
        "existing_ancestor",
    ):
        assert getattr(acquisition_plan, name) is getattr(estimates, name), name
    assert not hasattr(acquisition, "is_valid_flac")

    for path in sorted((PACKAGE_ROOT / "pipelines").glob("*.py")):
        defined = {
            node.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
        }
        assert not defined & set(CANONICAL), path


@pytest.mark.parametrize(
    ("layer", "allowed"),
    [
        ("core", ("jmds_prepare.core",)),
        (
            "storage",
            ("jmds_prepare.core", "jmds_prepare.profiles", "jmds_prepare.storage"),
        ),
    ],
)
def test_core_and_storage_keep_their_import_layers(layer, allowed):
    for path in sorted((PACKAGE_ROOT / layer).glob("*.py")):
        for module in _package_imports(path):
            assert module.startswith(allowed), (path, module)


# --- cli alias classification ---------------------------------------------------


def test_cli_private_aliases_are_exactly_the_documented_groups():
    discovered = {
        name
        for name, value in vars(cli).items()
        if name.startswith("_")
        and not name.startswith("__")
        and inspect.isfunction(value)
        and value.__module__ != cli.__name__
    }
    # Identity marker for the "replaced record provider" rule, not an alias.
    discovered.remove("_ORIGINAL_GET_RECORD_FILES")
    assert cli._ORIGINAL_GET_RECORD_FILES is zenodo_source.get_record_files

    assert not IMPORT_ONLY_ALIASES.keys() & INJECTED_ALIASES.keys()
    assert discovered == IMPORT_ONLY_ALIASES.keys() | INJECTED_ALIASES.keys()
    for name, canonical in {**IMPORT_ONLY_ALIASES, **INJECTED_ALIASES}.items():
        assert vars(cli)[name] is canonical, name


def test_import_only_alias_patch_must_target_the_calling_module(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    reports = {
        split: cli.CorrespondenceReport(split, 1, 1, 1, 1, (), (), {})
        for split in ("train", "dev")
    }
    monkeypatch.setattr(cli, "_require_protocol_paths", lambda _config: None)
    monkeypatch.setattr(cli, "_compute_correspondence", lambda _config: reports)
    monkeypatch.setattr(cli, "_compute_eval_evidence", lambda _config: {})
    destination = config.data_root / "reports" / "correspondence.json"

    monkeypatch.setattr(
        cli,
        "_write_json_atomic",
        lambda *_: pytest.fail("import-only alias must not be called"),
    )
    cli.validate_protocols(config, output=lambda _message: None)
    assert destination.is_file()

    written: list[Path] = []
    monkeypatch.setattr(
        validation, "write_json_atomic", lambda _payload, path: written.append(path)
    )
    cli.validate_protocols(config, output=lambda _message: None)
    assert written == [destination]
