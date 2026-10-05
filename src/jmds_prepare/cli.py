"""Command-line entry point and compatibility namespace for ``jmds-prepare``.

Orchestration lives in :mod:`jmds_prepare.pipelines`. Names in this module
fall into two groups:

* Injected collaborators. The wrappers below read these globals on every call
  and pass them to the pipelines, so monkeypatching them here redirects the
  commands: ``ASV_PROTOCOL_FILES``, ``EXPECTED_PRISTINE``,
  ``EXPECTED_GENERATED``, ``REQUIRED_ARCHIVES``, ``read_jmds_protocol``,
  ``read_asvspoof_protocol``, ``get_record_files``, ``get_record_metadata``,
  ``download_verified``, ``inventory_archive``, ``extract_selected``,
  ``read_extraction_ledger``, ``update_extraction_ledger``,
  ``build_raw_manifest``, ``write_manifest_atomic``, ``process_manifest``,
  ``audit_and_publish``, ``sha256_file``, every private wrapper defined here,
  and the aliases ``_is_valid_flac``, ``_validate_inventory`` and
  ``_audio_storage_estimate``.
* Import-only aliases. Everything else (schemas, classes, ``english_layout``,
  ``protocol_fingerprint``, ``validate_correspondence``, ``_write_json_atomic``,
  ``_asvspoof_root``, ``_archive_split``, ``_missing_inventory_message``,
  ``_bulk_generated_pcm16_bytes``, ``_measure_audio_header``,
  ``_existing_ancestor``) exists only so old imports keep working. Patching
  them here has no effect; patch the module that calls them instead.
"""

from __future__ import annotations

import argparse
import shutil as shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .audio import process_manifest as process_manifest
from .audit import audit_and_publish as audit_and_publish
from .config import PreparationConfig
from .core import atomic as _atomic
from .core.hashing import sha256_file as sha256_file
from .extraction import (
    ArchiveInventory as ArchiveInventory,
    ExtractionLedgerEntry as ExtractionLedgerEntry,
    ExtractionReport as ExtractionReport,
    extract_selected as extract_selected,
    inventory_archive as inventory_archive,
    read_extraction_ledger as read_extraction_ledger,
    update_extraction_ledger as update_extraction_ledger,
)
from .manifest import (
    build_raw_manifest as build_raw_manifest,
    write_manifest_atomic as write_manifest_atomic,
)
from .pipelines import acquisition as _acquisition
from .pipelines import acquisition_plan as _acquisition_plan
from .pipelines import auditing as _auditing
from .pipelines import processing as _processing
from .pipelines import provenance as _provenance
from .pipelines import validation as _validation
from .pipelines.common import Output as Output
from .profiles.english import ENGLISH_PROFILE as ENGLISH_PROFILE
from .protocols import (
    ASV_COLUMNS as ASV_COLUMNS,
    JMDS_COLUMNS as JMDS_COLUMNS,
    CorrespondenceReport as CorrespondenceReport,
    protocol_file_sha256 as protocol_file_sha256,
    protocol_fingerprint as protocol_fingerprint,
    read_asvspoof_protocol as read_asvspoof_protocol,
    read_jmds_protocol as read_jmds_protocol,
    validate_correspondence as validate_correspondence,
)
from .storage import audio_validation as _audio_validation
from .storage import estimates as _estimates
from .storage.layout import english_layout as english_layout
from .zenodo import (
    REQUIRED_ARCHIVES as REQUIRED_ARCHIVES,
    download_verified as download_verified,
    get_record_files as get_record_files,
    get_record_metadata as get_record_metadata,
)

# Mutable module-level copies: tests and callers monkeypatch these cli names.
ASV_PROTOCOL_FILES = dict(ENGLISH_PROFILE.asv_protocol_files)
EXPECTED_PRISTINE = dict(ENGLISH_PROFILE.expected_pristine)
EXPECTED_GENERATED = dict(ENGLISH_PROFILE.expected_generated)

_ORIGINAL_GET_RECORD_FILES = get_record_files

# Injected aliases: read by the wrappers below on every call.
_is_valid_flac = _audio_validation.is_valid_flac
_validate_inventory = _acquisition.validate_inventory
_audio_storage_estimate = _estimates.audio_storage_estimate

# Import-only aliases: patch the calling module, not these names.
_asvspoof_root = _validation.asvspoof_root
_missing_inventory_message = _acquisition.missing_inventory_message
_archive_split = _acquisition.archive_split
_bulk_generated_pcm16_bytes = _estimates.bulk_generated_pcm16_bytes
_measure_audio_header = _estimates.measure_audio_header
_existing_ancestor = _estimates.existing_ancestor
_write_json_atomic = _atomic.write_json_atomic


# --- dependency snapshots of the current cli globals -----------------------------


def _protocol_sources() -> _validation.ProtocolSources:
    return _validation.ProtocolSources(
        read_jmds_protocol=read_jmds_protocol,
        read_asvspoof_protocol=read_asvspoof_protocol,
        asv_protocol_files=ASV_PROTOCOL_FILES,
        expected_pristine=EXPECTED_PRISTINE,
        expected_generated=EXPECTED_GENERATED,
    )


def _validation_steps() -> _validation.ValidationSteps:
    return _validation.ValidationSteps(
        sources=_protocol_sources(),
        require_protocol_paths=_require_protocol_paths,
        compute_correspondence=_compute_correspondence,
        compute_eval_evidence=_compute_eval_evidence,
    )


def _acquisition_deps() -> _acquisition.AcquisitionDeps:
    return _acquisition.AcquisitionDeps(
        required_archives=REQUIRED_ARCHIVES,
        load_current_correspondence=_load_current_correspondence,
        record_for_acquisition=_record_for_acquisition,
        print_dry_run=_print_dry_run,
        resume_valid_ids=_resume_valid_ids,
        download_verified=download_verified,
        inventory_archive=inventory_archive,
        validate_inventory=_validate_inventory,
        extract_selected=extract_selected,
        validate_extraction=_validate_extraction,
        sha256_file=_sha256_file,
        update_extraction_ledger=update_extraction_ledger,
        validate_expected_pristine=_validate_expected_pristine,
        build_raw_manifest=build_raw_manifest,
        write_manifest_atomic=write_manifest_atomic,
        write_provenance=_write_provenance,
    )


def _dry_run_deps() -> _acquisition_plan.DryRunDeps:
    return _acquisition_plan.DryRunDeps(
        required_archives=REQUIRED_ARCHIVES,
        acquisition_storage_state=_acquisition_storage_state,
        audio_storage_estimate=_audio_storage_estimate,
        valid_extracted_ids=_valid_extracted_ids,
    )


# --- commands ------------------------------------------------------------------


def validate_protocols(
    config: PreparationConfig, *, output: Output = print
) -> dict[str, CorrespondenceReport]:
    """Validate train/dev correspondence and publish its current evidence."""
    return _validation.validate_protocols(
        config, _validation_steps(), output=output
    )


def acquire_english(
    config: PreparationConfig,
    *,
    dry_run: bool,
    output: Output = print,
) -> None:
    """Plan or acquire the ASVspoof5 pristine English train/dev audio."""
    _acquisition.acquire_english(
        config, _acquisition_deps(), dry_run=dry_run, output=output
    )


def process_english(
    config: PreparationConfig, *, output: Output = print
) -> None:
    """Process the canonical raw manifest to deterministic WAV files."""
    _processing.process_english(
        config, process_manifest=process_manifest, output=output
    )


def audit_english(
    config: PreparationConfig, *, output: Output = print
) -> None:
    """Audit the processed manifest and publish the technical baseline."""
    _auditing.audit_english(
        config, audit_and_publish=audit_and_publish, output=output
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jmds-prepare",
        description="Prepare the English JMDS/ASVspoof5 data reproducibly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate-protocols",
        "acquire-english",
        "process-english",
        "audit-english",
    ):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to the preparation YAML configuration.",
        )
        if command == "acquire-english":
            command_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Print the acquisition plan without mutating data.",
            )

    arguments = parser.parse_args(argv)
    config = PreparationConfig.load(arguments.config)
    config.validate()
    if arguments.command == "validate-protocols":
        validate_protocols(config)
    elif arguments.command == "acquire-english":
        acquire_english(config, dry_run=arguments.dry_run)
    elif arguments.command == "process-english":
        process_english(config)
    elif arguments.command == "audit-english":
        audit_english(config)
    else:  # pragma: no cover - argparse enforces the command choices.
        parser.error(f"unknown command: {arguments.command}")
    return 0


# --- compatibility wrappers for the historical private helpers -----------------


def _compute_correspondence(
    config: PreparationConfig,
) -> dict[str, CorrespondenceReport]:
    return _validation.compute_correspondence(config, _protocol_sources())


def _load_current_correspondence(
    config: PreparationConfig,
) -> tuple[
    dict[str, CorrespondenceReport],
    dict[str, pd.DataFrame],
]:
    return _validation.load_current_correspondence(config, _validation_steps())


def _compute_eval_evidence(config: PreparationConfig) -> dict[str, Any]:
    return _validation.compute_eval_evidence(config, _protocol_sources())


def _require_protocol_paths(config: PreparationConfig) -> None:
    _validation.require_protocol_paths(config, ASV_PROTOCOL_FILES)


def _print_dry_run(
    config: PreparationConfig,
    files: dict[str, Any],
    targets: dict[str, set[str]],
    protocols: dict[str, pd.DataFrame],
    *,
    archive_root: Path,
    raw_root: Path,
    output: Output,
) -> None:
    _acquisition_plan.print_dry_run(
        config,
        files,
        targets,
        protocols,
        archive_root=archive_root,
        raw_root=raw_root,
        output=output,
        deps=_dry_run_deps(),
    )


def _acquisition_storage_state(
    files: dict[str, Any],
    archive_root: Path,
    raw_root: Path,
    ledger_path: Path,
) -> dict[str, int]:
    return _acquisition_plan.acquisition_storage_state(
        files,
        archive_root,
        raw_root,
        ledger_path,
        read_extraction_ledger=read_extraction_ledger,
    )


def _valid_extracted_ids(
    output: Path,
    targets: set[str],
    ledger_path: Path | None = None,
    archive_md5s: dict[str, str] | None = None,
    split: str | None = None,
) -> set[str]:
    return _acquisition.valid_extracted_ids(
        output,
        targets,
        ledger_path,
        archive_md5s,
        split,
        read_extraction_ledger=read_extraction_ledger,
        sha256_file=_sha256_file,
        is_valid_flac=_is_valid_flac,
    )


def _resume_valid_ids(*args: Any) -> set[str]:
    return _acquisition.resume_valid_ids(_valid_extracted_ids, *args)


def _validate_extraction(
    report: ExtractionReport,
    wanted_ids: set[str],
    output: Path,
) -> None:
    _acquisition.validate_extraction(
        report, wanted_ids, output, valid_extracted_ids=_valid_extracted_ids
    )


def _validate_expected_pristine(
    raw_root: Path, targets: dict[str, set[str]]
) -> None:
    _acquisition.validate_expected_pristine(
        raw_root,
        targets,
        expected_pristine=EXPECTED_PRISTINE,
        valid_extracted_ids=_valid_extracted_ids,
    )


def _record_for_acquisition(record_id: int) -> Any:
    return _acquisition.record_for_acquisition(
        record_id,
        get_record_files=get_record_files,
        get_record_metadata=get_record_metadata,
        original_get_record_files=_ORIGINAL_GET_RECORD_FILES,
    )


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _write_provenance(
    config: PreparationConfig,
    record: Any,
    reports: dict[str, CorrespondenceReport],
) -> None:
    _provenance.write_provenance(config, record, reports)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
