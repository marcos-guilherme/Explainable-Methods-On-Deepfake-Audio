"""Tests for the ``jmds_prepare.pipelines`` orchestration layer.

They pin the package layout and import layers, and prove that every thin
``jmds_prepare.cli`` wrapper injects the *current* values of the monkeypatchable
cli names into the pipelines on each call.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from jmds_prepare import cli
from jmds_prepare.config import PreparationConfig
from jmds_prepare.pipelines import (
    acquisition,
    acquisition_plan,
    auditing,
    processing,
    validation,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "jmds_prepare"
PIPELINES_ROOT = PACKAGE_ROOT / "pipelines"


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
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package_parts[: len(package_parts) - node.level + 1]
                module = ".".join([*parts, *([node.module] if node.module else [])])
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)
            else:
                imported.add(node.module or "")
    return imported


def _recorder(store: dict, result=None):
    def record(*args, **kwargs):
        store["args"] = args
        store["kwargs"] = kwargs
        return result

    return record


# --- layout and layers ---------------------------------------------------------


def test_pipelines_package_has_focused_modules():
    names = {path.name for path in PIPELINES_ROOT.glob("*.py")}

    assert names >= {
        "__init__.py",
        "validation.py",
        "acquisition.py",
        "processing.py",
        "auditing.py",
        "metadata_extraction.py",
        "metadata_profiling.py",
        "metadata_provenance.py",
        "mandarin_metadata_profiling.py",
        "mandarin_metadata_provenance.py",
    }


def test_pipelines_never_import_the_cli():
    for path in sorted(PIPELINES_ROOT.glob("*.py")):
        imports = _package_imports(path)
        assert not any(
            module == "jmds_prepare.cli" or module.startswith("jmds_prepare.cli.")
            for module in imports
        ), path


@pytest.mark.parametrize(
    "relative",
    [
        "core",
        "storage",
        "sources",
        "profiles",
        "audio.py",
        "audit.py",
        "config.py",
        "extraction.py",
        "manifest.py",
        "protocols.py",
        "zenodo.py",
    ],
)
def test_lower_layers_never_import_pipelines(relative):
    target = PACKAGE_ROOT / relative
    files = sorted(target.glob("*.py")) if target.is_dir() else [target]
    assert files
    for path in files:
        assert not any(
            module.startswith("jmds_prepare.pipelines")
            for module in _package_imports(path)
        ), path


def test_importing_pipelines_does_not_load_the_cli():
    script = (
        "import json, sys\n"
        "import jmds_prepare.pipelines.validation\n"
        "import jmds_prepare.pipelines.acquisition\n"
        "import jmds_prepare.pipelines.acquisition_plan\n"
        "import jmds_prepare.pipelines.processing\n"
        "import jmds_prepare.pipelines.auditing\n"
        "import jmds_prepare.pipelines.provenance\n"
        "import jmds_prepare.pipelines.metadata_extraction\n"
        "import jmds_prepare.pipelines.metadata_profiling\n"
        "import jmds_prepare.pipelines.metadata_provenance\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('jmds_prepare'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "jmds_prepare.cli" not in set(json.loads(completed.stdout))


def test_cli_no_longer_carries_pipeline_implementation_dependencies():
    imports = _package_imports(PACKAGE_ROOT / "cli.py")

    assert not imports & {
        "math",
        "wave",
        "tempfile",
        "json",
        "concurrent.futures",
        "numpy",
        "soundfile",
    }
    assert any(module.startswith("jmds_prepare.pipelines") for module in imports)


# --- delegation and monkeypatch propagation --------------------------------------


def test_process_english_injects_current_process_manifest(tmp_path, monkeypatch):
    config = _config(tmp_path)
    sentinel = object()
    output = lambda _message: None
    seen: dict = {}
    monkeypatch.setattr(cli, "process_manifest", sentinel)
    monkeypatch.setattr(processing, "process_english", _recorder(seen))

    cli.process_english(config, output=output)

    assert seen["args"] == (config,)
    assert seen["kwargs"] == {"process_manifest": sentinel, "output": output}


def test_audit_english_injects_current_audit_and_publish(tmp_path, monkeypatch):
    config = _config(tmp_path)
    sentinel = object()
    output = lambda _message: None
    seen: dict = {}
    monkeypatch.setattr(cli, "audit_and_publish", sentinel)
    monkeypatch.setattr(auditing, "audit_english", _recorder(seen))

    cli.audit_english(config, output=output)

    assert seen["args"] == (config,)
    assert seen["kwargs"] == {"audit_and_publish": sentinel, "output": output}


@pytest.fixture
def protocol_sentinels(monkeypatch):
    names = {
        "read_jmds_protocol": object(),
        "read_asvspoof_protocol": object(),
        "ASV_PROTOCOL_FILES": {"train": "patched.tsv"},
        "EXPECTED_PRISTINE": {"train": 7},
        "EXPECTED_GENERATED": {"train": 8},
    }
    for name, value in names.items():
        monkeypatch.setattr(cli, name, value)
    return names


def _assert_sources(sources, sentinels):
    assert isinstance(sources, validation.ProtocolSources)
    assert sources.read_jmds_protocol is sentinels["read_jmds_protocol"]
    assert sources.read_asvspoof_protocol is sentinels["read_asvspoof_protocol"]
    assert sources.asv_protocol_files is sentinels["ASV_PROTOCOL_FILES"]
    assert sources.expected_pristine is sentinels["EXPECTED_PRISTINE"]
    assert sources.expected_generated is sentinels["EXPECTED_GENERATED"]


@pytest.mark.parametrize(
    ("cli_name", "pipeline_name"),
    [
        ("validate_protocols", "validate_protocols"),
        ("_load_current_correspondence", "load_current_correspondence"),
    ],
)
def test_validation_wrappers_inject_current_steps(
    tmp_path, monkeypatch, protocol_sentinels, cli_name, pipeline_name
):
    config = _config(tmp_path)
    steps = {
        "_require_protocol_paths": object(),
        "_compute_correspondence": object(),
        "_compute_eval_evidence": object(),
    }
    for name, value in steps.items():
        monkeypatch.setattr(cli, name, value)
    result = object()
    seen: dict = {}
    monkeypatch.setattr(validation, pipeline_name, _recorder(seen, result))

    if cli_name == "validate_protocols":
        output = lambda _message: None
        returned = cli.validate_protocols(config, output=output)
        assert seen["kwargs"] == {"output": output}
    else:
        returned = cli._load_current_correspondence(config)
        assert seen["kwargs"] == {}

    assert returned is result
    passed_config, passed_steps = seen["args"]
    assert passed_config is config
    assert isinstance(passed_steps, validation.ValidationSteps)
    _assert_sources(passed_steps.sources, protocol_sentinels)
    assert passed_steps.require_protocol_paths is steps["_require_protocol_paths"]
    assert passed_steps.compute_correspondence is steps["_compute_correspondence"]
    assert passed_steps.compute_eval_evidence is steps["_compute_eval_evidence"]


@pytest.mark.parametrize(
    ("cli_name", "pipeline_name"),
    [
        ("_compute_correspondence", "compute_correspondence"),
        ("_compute_eval_evidence", "compute_eval_evidence"),
    ],
)
def test_protocol_helpers_inject_current_sources(
    tmp_path, monkeypatch, protocol_sentinels, cli_name, pipeline_name
):
    config = _config(tmp_path)
    result = object()
    seen: dict = {}
    monkeypatch.setattr(validation, pipeline_name, _recorder(seen, result))

    assert getattr(cli, cli_name)(config) is result
    assert seen["args"][0] is config
    _assert_sources(seen["args"][1], protocol_sentinels)


def test_require_protocol_paths_uses_current_protocol_files(
    tmp_path, monkeypatch, protocol_sentinels
):
    config = _config(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(validation, "require_protocol_paths", _recorder(seen))

    cli._require_protocol_paths(config)

    assert seen["args"] == (config, protocol_sentinels["ASV_PROTOCOL_FILES"])


ACQUISITION_NAMES = {
    "required_archives": "REQUIRED_ARCHIVES",
    "load_current_correspondence": "_load_current_correspondence",
    "record_for_acquisition": "_record_for_acquisition",
    "print_dry_run": "_print_dry_run",
    "resume_valid_ids": "_resume_valid_ids",
    "download_verified": "download_verified",
    "inventory_archive": "inventory_archive",
    "validate_inventory": "_validate_inventory",
    "extract_selected": "extract_selected",
    "validate_extraction": "_validate_extraction",
    "sha256_file": "_sha256_file",
    "update_extraction_ledger": "update_extraction_ledger",
    "validate_expected_pristine": "_validate_expected_pristine",
    "build_raw_manifest": "build_raw_manifest",
    "write_manifest_atomic": "write_manifest_atomic",
    "write_provenance": "_write_provenance",
}


@pytest.mark.parametrize("dry_run", [True, False])
def test_acquire_english_injects_current_collaborators(
    tmp_path, monkeypatch, dry_run
):
    config = _config(tmp_path)
    sentinels = {name: object() for name in ACQUISITION_NAMES.values()}
    for name, value in sentinels.items():
        monkeypatch.setattr(cli, name, value)
    output = lambda _message: None
    seen: dict = {}
    monkeypatch.setattr(acquisition, "acquire_english", _recorder(seen))

    cli.acquire_english(config, dry_run=dry_run, output=output)

    passed_config, deps = seen["args"]
    assert passed_config is config
    assert seen["kwargs"] == {"dry_run": dry_run, "output": output}
    assert isinstance(deps, acquisition.AcquisitionDeps)
    for field, cli_name in ACQUISITION_NAMES.items():
        assert getattr(deps, field) is sentinels[cli_name], field


def test_record_for_acquisition_injects_current_providers(monkeypatch):
    files_provider = object()
    metadata_provider = object()
    result = object()
    seen: dict = {}
    monkeypatch.setattr(cli, "get_record_files", files_provider)
    monkeypatch.setattr(cli, "get_record_metadata", metadata_provider)
    monkeypatch.setattr(acquisition, "record_for_acquisition", _recorder(seen, result))

    assert cli._record_for_acquisition(7) is result
    assert seen["args"] == (7,)
    assert seen["kwargs"] == {
        "get_record_files": files_provider,
        "get_record_metadata": metadata_provider,
        "original_get_record_files": cli._ORIGINAL_GET_RECORD_FILES,
    }


def test_record_for_acquisition_pipeline_is_usable_without_the_cli():
    original = object()
    replaced_files = {"x.tar": "sentinel"}

    replaced = acquisition.record_for_acquisition(
        3,
        get_record_files=lambda _id: replaced_files,
        get_record_metadata=lambda _id: pytest.fail("metadata must be unused"),
        original_get_record_files=original,
    )
    metadata = object()
    official = acquisition.record_for_acquisition(
        3,
        get_record_files=original,
        get_record_metadata=lambda _id: metadata,
        original_get_record_files=original,
    )

    assert replaced.files is replaced_files
    assert replaced.doi is None
    assert official is metadata


def test_valid_extracted_ids_injects_current_ledger_reader_hasher_and_flac_check(
    tmp_path, monkeypatch
):
    reader = object()
    hasher = object()
    flac_check = object()
    result = {"T_1"}
    seen: dict = {}
    monkeypatch.setattr(cli, "read_extraction_ledger", reader)
    monkeypatch.setattr(cli, "_sha256_file", hasher)
    monkeypatch.setattr(cli, "_is_valid_flac", flac_check)
    monkeypatch.setattr(acquisition, "valid_extracted_ids", _recorder(seen, result))

    returned = cli._valid_extracted_ids(
        tmp_path, {"T_1"}, tmp_path / "ledger.csv", {"a.tar": "a"}, "train"
    )

    assert returned is result
    assert seen["args"] == (
        tmp_path, {"T_1"}, tmp_path / "ledger.csv", {"a.tar": "a"}, "train"
    )
    assert seen["kwargs"] == {
        "read_extraction_ledger": reader,
        "sha256_file": hasher,
        "is_valid_flac": flac_check,
    }


def test_valid_extracted_ids_honours_a_patched_cli_flac_check(
    tmp_path, monkeypatch
):
    checked: list[str] = []

    def fake_check(path):
        checked.append(path.name)
        return path.name == "T_1.flac"

    monkeypatch.setattr(cli, "_is_valid_flac", fake_check)

    assert cli._valid_extracted_ids(tmp_path, {"T_1", "T_2"}) == {"T_1"}
    assert sorted(checked) == ["T_1.flac", "T_2.flac"]


def test_resume_valid_ids_resolves_current_validator(tmp_path, monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(
        cli,
        "_valid_extracted_ids",
        lambda output, targets: calls.append((output, targets)) or {"T_1"},
    )

    assert cli._resume_valid_ids(
        tmp_path, {"T_1"}, tmp_path / "ledger.csv", {}, "train"
    ) == {"T_1"}
    assert calls == [(tmp_path, {"T_1"})]


def test_validate_extraction_resolves_current_validator(tmp_path, monkeypatch):
    report = cli.ExtractionReport(("T_1",), (), (), ())
    monkeypatch.setattr(cli, "_valid_extracted_ids", lambda output, targets: set())

    with pytest.raises(ValueError, match="failed validation"):
        cli._validate_extraction(report, {"T_1"}, tmp_path)


def test_validate_expected_pristine_uses_current_counts_and_validator(
    tmp_path, monkeypatch
):
    for split in ("train", "dev"):
        (tmp_path / split).mkdir()
        (tmp_path / split / f"{split}_1.flac").write_bytes(b"x")
    targets = {"train": {"train_1"}, "dev": {"dev_1"}}
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 1, "dev": 1})
    monkeypatch.setattr(
        cli, "_valid_extracted_ids", lambda output, wanted: set(wanted)
    )

    cli._validate_expected_pristine(tmp_path, targets)

    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 2, "dev": 1})
    with pytest.raises(ValueError, match="target count"):
        cli._validate_expected_pristine(tmp_path, targets)


def test_print_dry_run_injects_current_storage_helpers(tmp_path, monkeypatch):
    config = _config(tmp_path)
    sentinels = {
        "REQUIRED_ARCHIVES": object(),
        "_acquisition_storage_state": object(),
        "_audio_storage_estimate": object(),
        "_valid_extracted_ids": object(),
    }
    for name, value in sentinels.items():
        monkeypatch.setattr(cli, name, value)
    output = lambda _message: None
    seen: dict = {}
    monkeypatch.setattr(acquisition_plan, "print_dry_run", _recorder(seen))

    cli._print_dry_run(
        config,
        {},
        {},
        {},
        archive_root=tmp_path / "a",
        raw_root=tmp_path / "r",
        output=output,
    )

    assert seen["args"] == (config, {}, {}, {})
    deps = seen["kwargs"].pop("deps")
    assert seen["kwargs"] == {
        "archive_root": tmp_path / "a",
        "raw_root": tmp_path / "r",
        "output": output,
    }
    assert isinstance(deps, acquisition_plan.DryRunDeps)
    assert deps.required_archives is sentinels["REQUIRED_ARCHIVES"]
    assert deps.acquisition_storage_state is sentinels["_acquisition_storage_state"]
    assert deps.audio_storage_estimate is sentinels["_audio_storage_estimate"]
    assert deps.valid_extracted_ids is sentinels["_valid_extracted_ids"]


def test_acquisition_storage_state_injects_current_ledger_reader(
    tmp_path, monkeypatch
):
    reader = object()
    result = {"ok": 1}
    seen: dict = {}
    monkeypatch.setattr(cli, "read_extraction_ledger", reader)
    monkeypatch.setattr(
        acquisition_plan, "acquisition_storage_state", _recorder(seen, result)
    )

    assert cli._acquisition_storage_state({}, tmp_path, tmp_path, tmp_path) is result
    assert seen["args"] == ({}, tmp_path, tmp_path, tmp_path)
    assert seen["kwargs"] == {"read_extraction_ledger": reader}


# --- pipelines are callable with explicit dependencies ---------------------------


def test_processing_pipeline_runs_with_explicit_callback(tmp_path):
    config = _config(tmp_path)
    raw = config.data_root / "manifests" / "english_raw.csv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame({"raw": ["1"]}).to_csv(raw, index=False)
    calls: list[tuple] = []
    messages: list[str] = []

    processing.process_english(
        config,
        process_manifest=lambda *args: calls.append(args) or pd.DataFrame({"p": [1]}),
        output=messages.append,
    )

    destination = config.data_root / "manifests" / "english_processed.csv"
    assert calls[0][1:] == (config.data_root, destination, config.sample_rate)
    assert messages == [f"processed manifest: {destination} (1 rows)"]


def test_auditing_pipeline_runs_with_explicit_callback(tmp_path):
    config = _config(tmp_path)
    processed = config.data_root / "manifests" / "english_processed.csv"
    processed.parent.mkdir(parents=True)
    pd.DataFrame({"utt_id": ["T_1"]}).to_csv(processed, index=False)
    messages: list[str] = []

    def fake_audit(frame, reports_dir, *, progress):
        progress(2, 3)

    auditing.audit_english(
        config, audit_and_publish=fake_audit, output=messages.append
    )

    reports = config.data_root / "reports"
    assert messages == [
        "audit progress: 2/3",
        f"audit reports: {reports}",
        f"technical baseline: {reports / 'technical_baseline.json'}",
    ]


def test_validation_pipeline_runs_with_explicit_steps(tmp_path):
    config = _config(tmp_path)
    reports = {
        split: cli.CorrespondenceReport(split, 1, 2, 1, 2, (), (), {})
        for split in ("train", "dev")
    }
    required: list[PreparationConfig] = []
    steps = validation.ValidationSteps(
        sources=SimpleNamespace(),
        require_protocol_paths=required.append,
        compute_correspondence=lambda _config: reports,
        compute_eval_evidence=lambda _config: {"status": "excluded"},
    )
    messages: list[str] = []

    assert validation.validate_protocols(
        config, steps, output=messages.append
    ) is reports

    payload = json.loads(
        (config.data_root / "reports" / "correspondence.json").read_text(
            encoding="utf-8"
        )
    )
    assert required == [config]
    assert payload["eval"] == {"status": "excluded"}
    assert messages == [
        "train: pristine matched 1; generated matched 2",
        "dev: pristine matched 1; generated matched 2",
        "eval: excluded (known ID mismatch)",
    ]
