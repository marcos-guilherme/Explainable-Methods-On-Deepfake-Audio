"""Characterization tests for the public contracts of ``jmds_prepare``.

They freeze today's CLI surface, artifact schemas, English layout and the
``jmds_prepare.cli`` aliases so that a modular refactor cannot silently change
them. None of them touch real datasets or the network.
"""

from __future__ import annotations

import hashlib
import inspect
import socket
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import soundfile as sf

from jmds_prepare import audio, audit, cli, extraction, manifest, zenodo
from jmds_prepare.audio import AUDIO_METADATA_COLUMNS
from jmds_prepare.config import PreparationConfig
from jmds_prepare.extraction import (
    LEDGER_COLUMNS,
    ArchiveInventory,
    ExtractionLedgerEntry,
    ExtractionReport,
    read_extraction_ledger,
    update_extraction_ledger,
)
from jmds_prepare.manifest import (
    MANIFEST_COLUMNS,
    validate_manifest_rows,
    write_manifest_atomic,
)
from jmds_prepare.protocols import ASV_COLUMNS, JMDS_COLUMNS, read_jmds_protocol
from jmds_prepare.zenodo import REQUIRED_ARCHIVES, RecordFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "validate-protocols",
    "acquire-english",
    "process-english",
    "audit-english",
)


def _config(tmp_path: Path) -> PreparationConfig:
    return PreparationConfig(
        jmds_root=tmp_path / "jmds",
        data_root=tmp_path / "data",
        asvspoof_protocol_root=tmp_path / "asv",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- (1) CLI parser and entry point -------------------------------------------


def test_console_script_points_to_cli_main():
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["scripts"] == {
        "jmds-prepare": "jmds_prepare.cli:main"
    }
    assert callable(cli.main)


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    """Run ``cli.main`` with config loading and every command handler stubbed."""
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    loaded: list[Path] = []
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        cli.PreparationConfig, "load", lambda path: loaded.append(path) or config
    )
    monkeypatch.setattr(cli.PreparationConfig, "validate", lambda _self: None)
    for handler in (
        "validate_protocols",
        "acquire_english",
        "process_english",
        "audit_english",
    ):
        monkeypatch.setattr(
            cli,
            handler,
            lambda *args, _name=handler, **kwargs: calls.append(
                (_name, args, kwargs)
            ),
        )
    return config, config_path, loaded, calls


@pytest.mark.parametrize(
    ("command", "handler", "expected_kwargs"),
    [
        ("validate-protocols", "validate_protocols", {}),
        ("acquire-english", "acquire_english", {"dry_run": False}),
        ("process-english", "process_english", {}),
        ("audit-english", "audit_english", {}),
    ],
)
def test_each_subcommand_dispatches_to_its_handler(
    dispatch, command, handler, expected_kwargs
):
    config, config_path, loaded, calls = dispatch

    assert cli.main([command, "--config", str(config_path)]) == 0

    assert loaded == [config_path]
    assert calls == [(handler, (config,), expected_kwargs)]


def test_dry_run_is_an_acquire_only_flag(dispatch):
    config, config_path, _, calls = dispatch

    cli.main(["acquire-english", "--config", str(config_path), "--dry-run"])
    assert calls == [("acquire_english", (config,), {"dry_run": True})]

    for command in COMMANDS:
        if command == "acquire-english":
            continue
        with pytest.raises(SystemExit) as error:
            cli.main([command, "--config", str(config_path), "--dry-run"])
        assert error.value.code == 2


@pytest.mark.parametrize("command", COMMANDS)
def test_config_flag_is_required_for_every_subcommand(dispatch, command):
    _, _, loaded, calls = dispatch

    with pytest.raises(SystemExit) as error:
        cli.main([command])

    assert error.value.code == 2
    assert loaded == []
    assert calls == []


def test_a_subcommand_is_required_and_unknown_ones_are_rejected(dispatch):
    _, config_path, _, calls = dispatch

    for argv in ([], ["--config", str(config_path)], ["acquire"]):
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
    assert calls == []


def test_help_exposes_program_name_and_the_four_subcommands(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "jmds-prepare" in help_text
    for command in COMMANDS:
        assert command in help_text

    for command in COMMANDS:
        with pytest.raises(SystemExit) as sub_error:
            cli.main([command, "--help"])
        assert sub_error.value.code == 0
        sub_help = capsys.readouterr().out
        assert "--config" in sub_help
        assert ("--dry-run" in sub_help) is (command == "acquire-english")


# --- (2) Public schemas -------------------------------------------------------


def test_public_column_orders_are_frozen():
    assert MANIFEST_COLUMNS == [
        "utt_id",
        "spk_id",
        "gender",
        "language",
        "dataset",
        "split",
        "label",
        "attack_id",
        "protocol_codec",
        "source_group_id",
        "source_path",
        "processed_path",
        "sha256_source",
        "sha256_processed",
        "metadata_source",
    ]
    assert AUDIO_METADATA_COLUMNS == [
        "source_sample_rate",
        "source_channels",
        "source_frames",
        "source_duration_seconds",
        "source_peak",
        "source_rms",
        "processed_sample_rate",
        "processed_channels",
        "processed_frames",
        "processed_duration_seconds",
        "processed_peak",
        "processed_rms",
    ]
    assert LEDGER_COLUMNS == [
        "utt_id",
        "split",
        "archive_name",
        "archive_md5",
        "member_name",
        "member_size",
        "sha256_extracted",
    ]
    assert JMDS_COLUMNS == [
        "spk_id",
        "utt_id",
        "gender",
        "codec",
        "attack_id",
        "label",
        "native",
        "language",
        "dataset",
    ]
    assert ASV_COLUMNS == [
        "spk_id",
        "utt_id",
        "gender",
        "codec",
        "codec_q",
        "codec_seed",
        "attack_tag",
        "attack_id",
        "key",
        "tmp",
    ]


def test_schemas_have_no_duplicate_columns_and_manifest_metadata_are_disjoint():
    for columns in (
        MANIFEST_COLUMNS,
        AUDIO_METADATA_COLUMNS,
        LEDGER_COLUMNS,
        JMDS_COLUMNS,
        ASV_COLUMNS,
    ):
        assert len(columns) == len(set(columns))
    assert not set(MANIFEST_COLUMNS) & set(AUDIO_METADATA_COLUMNS)


def test_readers_reject_reordered_columns(tmp_path):
    jmds = tmp_path / "jmds.csv"
    pd.DataFrame(columns=list(reversed(JMDS_COLUMNS))).to_csv(jmds, index=False)
    with pytest.raises(ValueError, match="required order"):
        read_jmds_protocol(jmds, "train")

    ledger = tmp_path / "ledger.csv"
    pd.DataFrame(columns=list(reversed(LEDGER_COLUMNS))).to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="schema is invalid"):
        read_extraction_ledger(ledger)

    with pytest.raises(ValueError, match="canonical schema"):
        validate_manifest_rows(pd.DataFrame(columns=list(reversed(MANIFEST_COLUMNS))))


def test_written_artifacts_use_the_public_column_order(tmp_path):
    ledger = tmp_path / "manifests" / "extraction_ledger.csv"
    update_extraction_ledger(
        ledger,
        [
            ExtractionLedgerEntry(
                "T_0000000001", "train", "a.tar", "a" * 32, "x.flac", 3, "b" * 64
            )
        ],
    )
    assert ledger.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        LEDGER_COLUMNS
    )

    raw = tmp_path / "manifests" / "english_raw.csv"
    write_manifest_atomic(pd.DataFrame(columns=MANIFEST_COLUMNS), raw)
    assert raw.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        MANIFEST_COLUMNS
    )


# --- (3) English layout -------------------------------------------------------


def _raw_manifest_row(source: Path) -> dict[str, str]:
    return {
        "utt_id": "T_0000000001",
        "spk_id": "T_0001",
        "gender": "F",
        "language": "eng",
        "dataset": "ASVspoof2024",
        "split": "train",
        "label": "pristine",
        "attack_id": "pristine",
        "protocol_codec": "",
        "source_group_id": "",
        "source_path": str(source),
        "processed_path": "",
        "sha256_source": _sha256(source),
        "sha256_processed": "",
        "metadata_source": "JMDS+ASVspoof5-verified",
    }


def test_process_english_layout_names_and_columns_are_stable(tmp_path):
    config = _config(tmp_path)
    source = config.data_root / "raw" / "asvspoof5" / "train" / "T_0000000001.flac"
    source.parent.mkdir(parents=True)
    sf.write(source, np.linspace(-0.5, 0.5, 800), 8_000, format="FLAC")
    raw_path = config.data_root / "manifests" / "english_raw.csv"
    write_manifest_atomic(
        pd.DataFrame([_raw_manifest_row(source)], columns=MANIFEST_COLUMNS),
        raw_path,
    )
    messages: list[str] = []

    cli.process_english(config, output=messages.append)

    processed_wav = (
        config.data_root
        / "processed"
        / "english"
        / "train"
        / "pristine"
        / "T_0000000001.wav"
    )
    processed_path = config.data_root / "manifests" / "english_processed.csv"
    assert processed_wav.is_file()
    assert processed_path.is_file()
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    assert list(processed.columns) == MANIFEST_COLUMNS + AUDIO_METADATA_COLUMNS
    assert Path(processed.loc[0, "processed_path"]) == processed_wav
    assert Path(processed.loc[0, "source_path"]) == source
    assert processed.loc[0, "processed_sample_rate"] == "16000"
    assert messages == [f"processed manifest: {processed_path} (1 rows)"]
    assert sf.info(processed_wav).subtype == "PCM_16"
    assert raw_path.is_file()


def test_audit_english_reads_processed_manifest_and_writes_reports_dir(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    processed_path = config.data_root / "manifests" / "english_processed.csv"
    processed_path.parent.mkdir(parents=True)
    pd.DataFrame({"utt_id": ["T_1"]}).to_csv(processed_path, index=False)
    received: list[tuple[pd.DataFrame, Path]] = []

    def fake_audit(frame, reports_dir, *, progress):
        received.append((frame, reports_dir))
        progress(1, 2)

    monkeypatch.setattr(cli, "audit_and_publish", fake_audit)
    messages: list[str] = []

    cli.audit_english(config, output=messages.append)

    reports_dir = config.data_root / "reports"
    assert received[0][0]["utt_id"].tolist() == ["T_1"]
    assert received[0][1] == reports_dir
    assert messages == [
        "audit progress: 1/2",
        f"audit reports: {reports_dir}",
        f"technical baseline: {reports_dir / 'technical_baseline.json'}",
    ]


# --- (4) Monkeypatchable aliases in jmds_prepare.cli -------------------------


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a regression lets acquisition tests reach the network."""

    def blocked(*_args, **_kwargs):
        raise AssertionError("network access attempted in an offline contract test")

    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("get_record_files", zenodo),
        ("get_record_metadata", zenodo),
        ("download_verified", zenodo),
        ("inventory_archive", extraction),
        ("extract_selected", extraction),
        ("update_extraction_ledger", extraction),
        ("read_extraction_ledger", extraction),
        ("build_raw_manifest", manifest),
        ("write_manifest_atomic", manifest),
        ("process_manifest", audio),
        ("audit_and_publish", audit),
    ],
)
def test_cli_binds_collaborators_as_module_level_aliases(name, module):
    assert vars(cli)[name] is getattr(module, name)


def test_acquisition_resolves_collaborators_through_cli_namespace(
    tmp_path, monkeypatch, no_network
):
    config = _config(tmp_path)
    ids = {"train": "T_0000000001", "dev": "D_0000000001"}
    protocol_rows = {
        split: pd.DataFrame(
            {"utt_id": [utt_id], "label": ["pristine"]},
        )
        for split, utt_id in ids.items()
    }
    files = {
        name: RecordFile(name, 10, f"{index:032x}", f"https://example.test/{name}")
        for index, name in enumerate(REQUIRED_ARCHIVES)
    }
    calls: list[str] = []
    raw_manifest = pd.DataFrame({"sentinel": [1]})

    monkeypatch.setattr(
        cli, "_load_current_correspondence", lambda _: ({}, protocol_rows)
    )
    monkeypatch.setattr(cli, "EXPECTED_PRISTINE", {"train": 1, "dev": 1})
    monkeypatch.setattr(
        cli, "get_record_files", lambda record_id: calls.append("get_record_files") or files
    )

    def download(file, destination):
        calls.append("download_verified")
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / file.name
        archive.write_bytes(b"tar")
        return archive

    def inventory(archive):
        calls.append("inventory_archive")
        split_id = ids["train" if archive.name.startswith("flac_T_") else "dev"]
        return ArchiveInventory((split_id,), (), ())

    def extract(archive, wanted_ids, output):
        calls.append("extract_selected")
        output.mkdir(parents=True, exist_ok=True)
        selected = tuple(sorted(wanted_ids))
        for utt_id in selected:
            sf.write(output / f"{utt_id}.flac", np.zeros(32), 16_000, format="FLAC")
        return ExtractionReport(selected, (), (), ())

    monkeypatch.setattr(cli, "download_verified", download)
    monkeypatch.setattr(cli, "inventory_archive", inventory)
    monkeypatch.setattr(cli, "extract_selected", extract)
    monkeypatch.setattr(
        cli,
        "update_extraction_ledger",
        lambda *_: calls.append("update_extraction_ledger"),
    )
    monkeypatch.setattr(
        cli,
        "build_raw_manifest",
        lambda *_: calls.append("build_raw_manifest") or raw_manifest,
    )
    written: list[Path] = []
    monkeypatch.setattr(
        cli,
        "write_manifest_atomic",
        lambda frame, path: calls.append("write_manifest_atomic")
        or written.append(path),
    )

    cli.acquire_english(config, dry_run=False, output=lambda _message: None)

    for name in (
        "get_record_files",
        "download_verified",
        "inventory_archive",
        "extract_selected",
        "update_extraction_ledger",
        "build_raw_manifest",
        "write_manifest_atomic",
    ):
        assert name in calls, f"cli.{name} was not used through the cli namespace"
    assert written == [config.data_root / "manifests" / "english_raw.csv"]


def test_replacing_get_record_files_replaces_the_metadata_provider(
    monkeypatch, no_network
):
    files = {"x.tar": "sentinel"}
    monkeypatch.setattr(cli, "get_record_files", lambda record_id: files)
    monkeypatch.setattr(
        cli, "get_record_metadata", lambda _id: pytest.fail("metadata must be unused")
    )

    record = cli._record_for_acquisition(7)

    assert record.record_id == 7
    assert record.files is files
    assert set(record.missing_reasons) == {
        "doi",
        "license",
        "version",
        "publication_date",
        "official_url",
    }


def test_untouched_get_record_files_uses_record_metadata(monkeypatch, no_network):
    sentinel = object()
    monkeypatch.setattr(cli, "get_record_metadata", lambda record_id: sentinel)

    assert cli._record_for_acquisition(7) is sentinel


def test_process_and_audit_resolve_collaborators_through_cli_namespace(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    manifests = config.data_root / "manifests"
    manifests.mkdir(parents=True)
    pd.DataFrame({"raw": ["1"]}).to_csv(manifests / "english_raw.csv", index=False)
    pd.DataFrame({"processed": ["1"]}).to_csv(
        manifests / "english_processed.csv", index=False
    )
    seen: dict[str, tuple] = {}
    monkeypatch.setattr(
        cli,
        "process_manifest",
        lambda *args: seen.setdefault("process_manifest", args)
        and pd.DataFrame({"processed": ["1"]}),
    )
    monkeypatch.setattr(
        cli,
        "audit_and_publish",
        lambda *args, **kwargs: seen.setdefault("audit_and_publish", args),
    )

    cli.process_english(config, output=lambda _message: None)
    cli.audit_english(config, output=lambda _message: None)

    assert seen["process_manifest"][1:] == (
        config.data_root,
        manifests / "english_processed.csv",
        config.sample_rate,
    )
    assert seen["audit_and_publish"][1] == config.data_root / "reports"


# --- (5) audit_manifest versus audit_and_publish -----------------------------


def test_audit_entry_points_keep_distinct_signatures():
    legacy = inspect.signature(audit.audit_manifest)
    combined = inspect.signature(audit.audit_and_publish)

    assert list(legacy.parameters) == ["frame", "output_dir"]
    assert list(combined.parameters) == ["frame", "output_dir", "progress"]
    progress = combined.parameters["progress"]
    assert progress.kind is inspect.Parameter.KEYWORD_ONLY
    assert progress.default is None
