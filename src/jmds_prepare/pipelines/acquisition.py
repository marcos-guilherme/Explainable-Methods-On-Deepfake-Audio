"""Selective pristine acquisition: download, inventory, extract, verify, delete."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from ..config import PreparationConfig
from ..extraction import ArchiveInventory, ExtractionReport
from ..profiles.english import ENGLISH_PROFILE
from ..sources.asvspoof import CorrespondenceReport
from ..storage.layout import english_layout
from ..storage.ledger import ExtractionLedgerEntry
from .common import Output


@dataclass(frozen=True)
class AcquisitionDeps:
    """Collaborators of :func:`acquire_english`, resolved by the caller per call."""

    required_archives: Sequence[str]
    load_current_correspondence: Callable[
        [PreparationConfig],
        tuple[dict[str, CorrespondenceReport], dict[str, pd.DataFrame]],
    ]
    record_for_acquisition: Callable[[int], Any]
    print_dry_run: Callable[..., None]
    resume_valid_ids: Callable[..., set[str]]
    download_verified: Callable[[Any, Path], Path]
    inventory_archive: Callable[[Path], ArchiveInventory]
    validate_inventory: Callable[[ArchiveInventory], None]
    extract_selected: Callable[[Path, set[str], Path], ExtractionReport]
    validate_extraction: Callable[[ExtractionReport, set[str], Path], None]
    sha256_file: Callable[[Path], str]
    update_extraction_ledger: Callable[
        [Path, Iterable[ExtractionLedgerEntry]], Any
    ]
    validate_expected_pristine: Callable[[Path, dict[str, set[str]]], None]
    build_raw_manifest: Callable[..., pd.DataFrame]
    write_manifest_atomic: Callable[[pd.DataFrame, Path], None]
    write_provenance: Callable[
        [PreparationConfig, Any, dict[str, CorrespondenceReport]], None
    ]


def acquire_english(
    config: PreparationConfig,
    deps: AcquisitionDeps,
    *,
    dry_run: bool,
    output: Output,
) -> None:
    """Plan or acquire the ASVspoof5 pristine English train/dev audio."""
    reports, protocols = deps.load_current_correspondence(config)
    record = deps.record_for_acquisition(config.zenodo_record_id)
    files = record.files
    targets = {
        split: set(
            protocol.loc[protocol["label"] == "pristine", "utt_id"].astype(str)
        )
        for split, protocol in protocols.items()
    }
    layout = english_layout(config.data_root)
    raw_root = layout.raw_source_dir
    archive_root = layout.archive_dir
    ledger_path = layout.extraction_ledger_path
    archive_md5s = {name: file.checksum for name, file in files.items()}

    if dry_run:
        deps.print_dry_run(
            config,
            files,
            targets,
            protocols,
            archive_root=archive_root,
            raw_root=raw_root,
            output=output,
        )
        return

    required_archives = deps.required_archives
    valid_targets = {
        split: deps.resume_valid_ids(
            raw_root / split,
            split_targets,
            ledger_path,
            archive_md5s,
            split,
        )
        for split, split_targets in targets.items()
    }
    inventoried_targets = {
        split: set(valid_ids) for split, valid_ids in valid_targets.items()
    }
    for archive_name in required_archives:
        split = archive_split(archive_name)
        output_dir = raw_root / split
        outstanding = targets[split] - valid_targets[split]
        is_last = archive_name == required_archives[-1]
        globally_incomplete = any(
            split_targets - valid_targets[name]
            for name, split_targets in targets.items()
        )
        if not outstanding and not (is_last and globally_incomplete):
            retained = archive_root / archive_name
            if retained.exists():
                deps.download_verified(files[archive_name], archive_root).unlink()
                output(f"{archive_name}: verified retained archive deleted")
                continue
            output(f"{archive_name}: skipped; no outstanding {split} IDs")
            continue

        archive = deps.download_verified(files[archive_name], archive_root)
        inventory = deps.inventory_archive(archive)
        deps.validate_inventory(inventory)
        inventoried_targets[split].update(
            targets[split] & set(inventory.flac_ids)
        )
        expected_here = outstanding & set(inventory.flac_ids)
        report = deps.extract_selected(archive, expected_here, output_dir)
        deps.validate_extraction(report, expected_here, output_dir)
        details = {member.utt_id: member for member in report.members}
        inventory_details = {
            member.utt_id: member for member in inventory.members
        }
        ledger_entries = []
        for utt_id in sorted(report.extracted_ids):
            detail = details.get(utt_id)
            inventoried = inventory_details.get(utt_id)
            destination = output_dir / f"{utt_id}.flac"
            ledger_entries.append(
                ExtractionLedgerEntry(
                    utt_id=utt_id,
                    split=split,
                    archive_name=archive_name,
                    archive_md5=files[archive_name].checksum,
                    member_name=(
                        detail.member_name
                        if detail
                        else inventoried.member_name
                        if inventoried
                        else f"{utt_id}.flac"
                    ),
                    member_size=(
                        detail.member_size
                        if detail
                        else inventoried.member_size
                        if inventoried
                        else destination.stat().st_size
                    ),
                    sha256_extracted=(
                        detail.sha256_extracted
                        if detail
                        else deps.sha256_file(destination)
                    ),
                )
            )
        deps.update_extraction_ledger(ledger_path, ledger_entries)
        valid_targets[split].update(report.extracted_ids)
        if is_last:
            never_appeared = {
                name: targets[name] - inventoried_targets[name]
                for name in ENGLISH_PROFILE.splits
            }
            if any(never_appeared.values()):
                raise ValueError(missing_inventory_message(never_appeared))
            deps.validate_expected_pristine(raw_root, targets)
        archive.unlink()
        output(
            f"{archive_name}: extracted {len(report.extracted_ids)}; "
            "verified; archive deleted"
        )

    deps.validate_expected_pristine(raw_root, targets)
    manifest = deps.build_raw_manifest(config, reports.values())
    manifest_path = layout.raw_manifest_path
    deps.write_manifest_atomic(manifest, manifest_path)
    deps.write_provenance(config, record, reports)
    output(f"raw manifest: {manifest_path} ({len(manifest)} rows)")


def missing_inventory_message(
    missing_by_split: dict[str, set[str]],
) -> str:
    missing = [
        f"{split}:{utt_id}"
        for split in ENGLISH_PROFILE.splits
        for utt_id in sorted(missing_by_split[split])
    ]
    preview = ", ".join(missing[:10])
    remainder = len(missing) - min(len(missing), 10)
    suffix = f", ... and {remainder} more" if remainder else ""
    return (
        f"{len(missing)} target(s) never appeared in any archive inventory: "
        f"{preview}{suffix}; last TAR preserved for diagnosis"
    )


def record_for_acquisition(
    record_id: int,
    *,
    get_record_files: Callable[[int], dict[str, Any]],
    get_record_metadata: Callable[[int], Any],
    original_get_record_files: Callable[[int], dict[str, Any]],
) -> Any:
    """Return official record metadata, or files-only metadata when replaced.

    A caller that swapped ``get_record_files`` replaced the metadata provider,
    so the official metadata endpoint must not be consulted.
    """
    if get_record_files is not original_get_record_files:
        files = get_record_files(record_id)
        missing = "unavailable because record metadata provider was replaced"
        return SimpleNamespace(
            record_id=record_id,
            doi=None,
            license=None,
            version=None,
            publication_date=None,
            official_url=None,
            files=files,
            missing_reasons={
                name: missing
                for name in (
                    "doi",
                    "license",
                    "version",
                    "publication_date",
                    "official_url",
                )
            },
        )
    return get_record_metadata(record_id)


def archive_split(name: str) -> str:
    return ENGLISH_PROFILE.archive_split(name)


def valid_extracted_ids(
    output: Path,
    targets: set[str],
    ledger_path: Path | None = None,
    archive_md5s: Mapping[str, str] | None = None,
    split: str | None = None,
    *,
    read_extraction_ledger: Callable[[Path], pd.DataFrame],
    sha256_file: Callable[[Path], str],
    is_valid_flac: Callable[[Path], bool],
) -> set[str]:
    if ledger_path is None or archive_md5s is None or split is None:
        return {
            utt_id
            for utt_id in targets
            if is_valid_flac(output / f"{utt_id}.flac")
        }
    ledger = read_extraction_ledger(ledger_path)
    by_id = ledger.set_index("utt_id", drop=False) if not ledger.empty else None
    return {
        utt_id
        for utt_id in targets
        if by_id is not None
        and utt_id in by_id.index
        and (output / f"{utt_id}.flac").is_file()
        and by_id.loc[utt_id, "split"] == split
        and archive_md5s.get(by_id.loc[utt_id, "archive_name"], "").lower()
        == str(by_id.loc[utt_id, "archive_md5"]).lower()
        and sha256_file(output / f"{utt_id}.flac")
        == by_id.loc[utt_id, "sha256_extracted"]
        and is_valid_flac(output / f"{utt_id}.flac")
    }


def resume_valid_ids(
    valid_extracted_ids: Callable[..., set[str]], *args: Any
) -> set[str]:
    # Injected legacy validators may only accept (output, targets).
    if not _accepts(valid_extracted_ids, args):
        return valid_extracted_ids(args[0], args[1])
    return valid_extracted_ids(*args)


def _accepts(function: Callable[..., Any], args: tuple[Any, ...]) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(*args)
    except TypeError:
        return False
    return True


def validate_extraction(
    report: ExtractionReport,
    wanted_ids: set[str],
    output: Path,
    *,
    valid_extracted_ids: Callable[[Path, set[str]], set[str]],
) -> None:
    if report.duplicate_ids:
        raise ValueError(
            "Duplicate requested IDs in archive: "
            + ", ".join(report.duplicate_ids)
        )
    if report.rejected_unsafe_members:
        raise ValueError(
            "Unsafe TAR members rejected: "
            + ", ".join(report.rejected_unsafe_members)
        )
    extracted = set(report.extracted_ids)
    if extracted != wanted_ids:
        raise ValueError(
            "Extractor did not return the exact expected IDs: "
            f"expected={len(wanted_ids)}, extracted={len(extracted)}"
        )
    invalid = sorted(
        utt_id
        for utt_id in report.extracted_ids
        if utt_id not in valid_extracted_ids(output, {utt_id})
    )
    if invalid:
        raise ValueError(
            "Extracted audio failed validation: " + ", ".join(invalid)
        )


def validate_inventory(inventory: ArchiveInventory) -> None:
    if inventory.duplicate_ids:
        raise ValueError(
            "Duplicate FLAC IDs in archive inventory: "
            + ", ".join(inventory.duplicate_ids)
        )
    if inventory.rejected_unsafe_members:
        raise ValueError(
            "Unsafe FLAC members in archive inventory: "
            + ", ".join(inventory.rejected_unsafe_members)
        )


def validate_expected_pristine(
    raw_root: Path,
    targets: dict[str, set[str]],
    *,
    expected_pristine: Mapping[str, int],
    valid_extracted_ids: Callable[[Path, set[str]], set[str]],
) -> None:
    for split in ENGLISH_PROFILE.splits:
        expected_count = expected_pristine[split]
        expected_ids = targets[split]
        if len(expected_ids) != expected_count:
            raise ValueError(
                f"{split} pristine protocol target count is "
                f"{len(expected_ids)}; expected {expected_count}"
            )
        output = raw_root / split
        actual_ids = (
            {path.stem for path in output.glob("*.flac") if path.is_file()}
            if output.is_dir()
            else set()
        )
        if actual_ids != expected_ids:
            missing = len(expected_ids - actual_ids)
            extra = len(actual_ids - expected_ids)
            raise ValueError(
                f"{split} pristine extraction is incomplete: "
                f"expected={expected_count}, actual={len(actual_ids)}, "
                f"missing={missing}, extra={extra}"
            )
        invalid = expected_ids - valid_extracted_ids(output, expected_ids)
        if invalid:
            raise ValueError(
                f"{split} has {len(invalid)} empty or invalid pristine files"
            )
