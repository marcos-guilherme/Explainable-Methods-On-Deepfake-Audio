from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PreparationConfig
from .core.hashing import sha256_file, sha256_file_preserving_times
from .profiles.english import ENGLISH_PROFILE
from .protocols import (
    CorrespondenceReport,
    protocol_fingerprint,
    read_jmds_protocol,
)
from .storage.layout import DataLayout, english_layout


MANIFEST_COLUMNS = [
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

_SPLIT_ORDER = {split: index for index, split in enumerate(ENGLISH_PROFILE.splits)}
_SUPPORTED_LABELS = frozenset(ENGLISH_PROFILE.label_equivalence)
_METADATA_SOURCE = ENGLISH_PROFILE.metadata_source


def build_raw_manifest(
    config: PreparationConfig,
    reports: Iterable[CorrespondenceReport],
) -> pd.DataFrame:
    report_by_split = _validated_reports(config, reports)
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for split in _SPLIT_ORDER:
        if split not in report_by_split:
            continue
        protocol_path = ENGLISH_PROFILE.jmds_protocol_path(config.jmds_root, split)
        protocol = read_jmds_protocol(
            protocol_path, split, language=ENGLISH_PROFILE.language_code
        )
        asv_by_id = _asv_evidence_by_id(config, split)
        _validate_report(report_by_split[split], protocol)
        _validate_english(protocol)

        for row in protocol.to_dict("records"):
            utt_id = row["utt_id"]
            if utt_id in seen_ids:
                raise ValueError(f"Duplicate utt_id across manifest: {utt_id}")
            seen_ids.add(utt_id)

            source, expected_root = _source_path(config, split, row)
            resolved_root = expected_root.resolve()
            resolved_source = source.resolve()
            if not resolved_source.is_relative_to(resolved_root):
                raise ValueError(
                    f"Source path for {utt_id} escapes expected root: {source}"
                )
            if resolved_source in seen_paths:
                raise ValueError(f"Duplicate resolved source path: {resolved_source}")
            seen_paths.add(resolved_source)
            if not resolved_source.is_file():
                raise FileNotFoundError(
                    f"Source audio is missing for {utt_id}: {source}"
                )

            records.append(
                {
                    "utt_id": utt_id,
                    "spk_id": row["spk_id"],
                    "gender": row["gender"],
                    "language": row["language"],
                    "dataset": row["dataset"],
                    "split": split,
                    "label": row["label"],
                    "attack_id": row["attack_id"],
                    "protocol_codec": (
                        asv_by_id.get(utt_id, {}).get("codec", "")
                    ).replace("-", ""),
                    "source_group_id": (
                        asv_by_id.get(utt_id, {}).get("tmp", "")
                    ).replace("-", ""),
                    "source_path": str(source),
                    "processed_path": "",
                    "sha256_source": _sha256(resolved_source),
                    "sha256_processed": "",
                    "metadata_source": _METADATA_SOURCE,
                }
            )

    records.sort(
        key=lambda record: (
            _SPLIT_ORDER[record["split"]],
            record["label"],
            record["utt_id"],
        )
    )
    manifest = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    _validate_manifest_rows(
        manifest,
        additional_columns=(),
        verify_source_hashes=False,
    )
    return manifest


def validate_manifest_rows(
    frame: pd.DataFrame,
    *,
    additional_columns: Sequence[str] = (),
) -> None:
    """Validate the canonical manifest schema, provenance, paths and hashes."""
    _validate_manifest_rows(
        frame,
        additional_columns=additional_columns,
        verify_source_hashes=True,
    )


def _validate_manifest_rows(
    frame: pd.DataFrame,
    *,
    additional_columns: Sequence[str],
    verify_source_hashes: bool,
) -> None:
    expected_columns = MANIFEST_COLUMNS + list(additional_columns)
    if list(frame.columns) != expected_columns:
        raise ValueError("Manifest columns do not match the canonical schema")

    seen_ids: set[str] = set()
    seen_sources: set[Path] = set()
    for _, row in frame.iterrows():
        values: dict[str, str] = {}
        for column in MANIFEST_COLUMNS:
            value = row[column]
            if column in {"protocol_codec", "source_group_id"} and pd.isna(value):
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"Manifest field {column} must be a string")
            values[column] = value

        for column in (
            "utt_id",
            "spk_id",
            "gender",
            "language",
            "dataset",
            "split",
            "label",
            "attack_id",
            "source_path",
            "sha256_source",
            "metadata_source",
        ):
            if not values[column]:
                raise ValueError(f"Manifest field {column} must not be empty")

        utt_id = values["utt_id"]
        if utt_id in seen_ids:
            raise ValueError(f"Duplicate utt_id across manifest: {utt_id}")
        seen_ids.add(utt_id)

        if values["gender"] not in {"F", "M"}:
            raise ValueError(f"Invalid gender for {utt_id}: {values['gender']}")
        if values["language"] != ENGLISH_PROFILE.language_code:
            raise ValueError(
                f"Invalid language for English manifest row {utt_id}: "
                f"{values['language']}"
            )
        if values["dataset"] != ENGLISH_PROFILE.dataset:
            raise ValueError(f"Invalid dataset for {utt_id}: {values['dataset']}")
        if values["split"] not in _SPLIT_ORDER:
            raise ValueError(f"Invalid split for {utt_id}: {values['split']}")
        if values["label"] not in _SUPPORTED_LABELS:
            raise ValueError(f"Invalid label for {utt_id}: {values['label']}")
        if (
            values["label"] == "pristine"
            and values["attack_id"] != "pristine"
        ):
            raise ValueError(f"Invalid attack_id for pristine row {utt_id}")
        if (
            values["label"] == "generated"
            and values["attack_id"] == "pristine"
        ):
            raise ValueError(f"Invalid attack_id for generated row {utt_id}")
        if values["metadata_source"] != _METADATA_SOURCE:
            raise ValueError(
                f"Invalid metadata_source for {utt_id}: "
                f"{values['metadata_source']}"
            )

        source = Path(values["source_path"]).resolve()
        if source in seen_sources:
            raise ValueError(f"Duplicate resolved source path: {source}")
        seen_sources.add(source)
        if not source.is_file():
            raise FileNotFoundError(f"Source audio is missing for {utt_id}: {source}")
        if (
            verify_source_hashes
            and _sha256_preserving_times(source) != values["sha256_source"]
        ):
            raise ValueError(f"Source checksum mismatch for {utt_id}")

        processed_path = values["processed_path"]
        processed_hash = values["sha256_processed"]
        if bool(processed_path) != bool(processed_hash):
            raise ValueError(
                f"processed_path and sha256_processed must both be empty or "
                f"populated for {utt_id}"
            )


def write_manifest_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validated_reports(
    config: PreparationConfig,
    reports: Iterable[CorrespondenceReport],
) -> dict[str, CorrespondenceReport]:
    configured_splits = set(config.supported_splits)
    unsupported_configured = configured_splits - set(_SPLIT_ORDER)
    if unsupported_configured:
        names = ", ".join(sorted(unsupported_configured))
        raise ValueError(f"Unsupported split(s): {names}")

    by_split: dict[str, CorrespondenceReport] = {}
    for report in reports:
        if report.split not in _SPLIT_ORDER:
            raise ValueError(f"Unsupported correspondence report split: {report.split}")
        if report.split in by_split:
            raise ValueError(
                f"Duplicate correspondence report for split: {report.split}"
            )
        missing_fingerprints = [
            name
            for name, value in (
                ("JMDS", report.jmds_fingerprint),
                ("ASV", report.asv_fingerprint),
            )
            if not value
        ]
        if missing_fingerprints:
            names = ", ".join(missing_fingerprints)
            raise ValueError(
                f"Correspondence report for {report.split} is missing "
                f"protocol fingerprints: {names}"
            )
        report.raise_for_errors()
        if (
            report.pristine_matched != report.pristine_expected
            or report.generated_matched != report.generated_expected
        ):
            raise ValueError(f"Incomplete matched count for split: {report.split}")
        by_split[report.split] = report

    missing = configured_splits - set(by_split)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing correspondence report for split(s): {names}")
    unexpected = set(by_split) - configured_splits
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Unexpected correspondence report for split(s): {names}")
    return by_split


def _validate_report(
    report: CorrespondenceReport, protocol: pd.DataFrame
) -> None:
    current_fingerprint = protocol_fingerprint(protocol, report.split)
    if report.jmds_fingerprint != current_fingerprint:
        raise ValueError(
            f"Stale correspondence report fingerprint for {report.split}"
        )

    actual_pristine = int((protocol["label"] == "pristine").sum())
    actual_generated = int((protocol["label"] == "generated").sum())
    if (
        report.pristine_expected != actual_pristine
        or report.generated_expected != actual_generated
    ):
        raise ValueError(
            f"Correspondence report counts do not match protocol for {report.split}"
        )


def _validate_english(protocol: pd.DataFrame) -> None:
    unsupported = sorted(set(protocol["language"]) - {ENGLISH_PROFILE.language_code})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"English manifest received non-English language(s): {names}")


@lru_cache(maxsize=8)
def _english_layout(data_root: Path) -> DataLayout:
    return english_layout(data_root)


def _source_path(
    config: PreparationConfig,
    split: str,
    row: dict[str, Any],
) -> tuple[Path, Path]:
    label = row["label"]
    utt_id = row["utt_id"]
    if label == "pristine":
        root = _english_layout(config.data_root).raw_split_dir(split)
        return root / f"{utt_id}.flac", root
    if label == "generated":
        root = ENGLISH_PROFILE.generated_split_dir(config.jmds_root, split)
        return root / f"{utt_id}.wav", root
    raise ValueError(f"Unsupported manifest label: {label}")


def _asv_evidence_by_id(
    config: PreparationConfig, split: str
) -> dict[str, dict[str, str]]:
    if config.asvspoof_protocol_root is None:
        return {}
    from .protocols import read_asvspoof_protocol

    if split not in ENGLISH_PROFILE.splits:
        raise KeyError(split)
    filename = ENGLISH_PROFILE.asv_protocol_files[split]
    path = config.asvspoof_protocol_root / filename
    if not path.is_file():
        return {}
    frame = read_asvspoof_protocol(path, split)
    return {
        str(row["utt_id"]): {key: str(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    }


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _sha256_preserving_times(path: Path) -> str:
    return sha256_file_preserving_times(path, hasher=_sha256)
