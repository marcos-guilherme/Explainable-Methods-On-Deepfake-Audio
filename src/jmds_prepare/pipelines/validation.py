from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..config import PreparationConfig
from ..profiles.english import ENGLISH_PROFILE
from ..sources.asvspoof import (
    CorrespondenceReport,
    protocol_file_sha256,
    protocol_fingerprint,
    validate_correspondence,
)
from ..sources.jmds import JMDS_COLUMNS
from ..storage.layout import english_layout
from .common import Output, write_json_atomic


@dataclass(frozen=True)
class ProtocolSources:
    """Protocol readers and expected counts as currently bound by the caller."""

    read_jmds_protocol: Callable[..., pd.DataFrame]
    read_asvspoof_protocol: Callable[[Path, str], pd.DataFrame]
    asv_protocol_files: Mapping[str, str]
    expected_pristine: Mapping[str, int]
    expected_generated: Mapping[str, int]


@dataclass(frozen=True)
class ValidationSteps:
    """Replaceable protocol-validation steps used by the command flows."""

    sources: ProtocolSources
    require_protocol_paths: Callable[[PreparationConfig], None]
    compute_correspondence: Callable[
        [PreparationConfig], dict[str, CorrespondenceReport]
    ]
    compute_eval_evidence: Callable[[PreparationConfig], dict[str, Any]]


def validate_protocols(
    config: PreparationConfig,
    steps: ValidationSteps,
    *,
    output: Output,
) -> dict[str, CorrespondenceReport]:
    """Validate train/dev correspondence and publish its current evidence."""
    steps.require_protocol_paths(config)
    reports = steps.compute_correspondence(config)
    eval_evidence = steps.compute_eval_evidence(config)
    payload = {
        "jmds_root": str(config.jmds_root.resolve()),
        "asvspoof_protocol_root": str(
            asvspoof_root(config).resolve()
        ),
        "splits": {
            split: asdict(reports[split]) for split in ENGLISH_PROFILE.splits
        },
        "eval": eval_evidence,
        "freshness_scope": (
            "exact bytes and canonical full-protocol content for complete "
            "JMDS and ASV train/dev/eval, plus compared-subset fingerprints"
        ),
    }
    reports_dir = english_layout(config.data_root).reports_dir
    destination = reports_dir / "correspondence.json"
    write_json_atomic(payload, destination)
    for split in ENGLISH_PROFILE.splits:
        report = reports[split]
        output(
            f"{split}: pristine matched {report.pristine_matched}; "
            f"generated matched {report.generated_matched}"
        )
    output("eval: excluded (known ID mismatch)")
    return reports


def compute_correspondence(
    config: PreparationConfig,
    sources: ProtocolSources,
) -> dict[str, CorrespondenceReport]:
    root = asvspoof_root(config)
    reports: dict[str, CorrespondenceReport] = {}
    for split in ENGLISH_PROFILE.splits:
        jmds = sources.read_jmds_protocol(
            jmds_path := ENGLISH_PROFILE.jmds_protocol_path(config.jmds_root, split),
            split,
            language=ENGLISH_PROFILE.language_code,
        )
        jmds_full = (
            pd.read_csv(jmds_path, dtype=str, keep_default_na=False)
            if jmds_path.is_file()
            else jmds.copy()
        )
        asv = sources.read_asvspoof_protocol(
            root / sources.asv_protocol_files[split],
            split,
        )
        full_asv = asv.copy()
        asv_path = root / sources.asv_protocol_files[split]
        asv = asv.loc[asv["utt_id"].isin(jmds["utt_id"])].copy()
        report = validate_correspondence(jmds, asv, split)
        report.jmds_full_fingerprint = protocol_fingerprint(jmds_full, split)
        report.asv_full_fingerprint = protocol_fingerprint(full_asv, split)
        report.jmds_file_sha256 = (
            protocol_file_sha256(jmds_path) if jmds_path.is_file() else ""
        )
        report.asv_file_sha256 = (
            protocol_file_sha256(asv_path) if asv_path.is_file() else ""
        )
        report.raise_for_errors()
        expected = (
            sources.expected_pristine[split],
            sources.expected_generated[split],
        )
        actual = (report.pristine_matched, report.generated_matched)
        if actual != expected:
            raise ValueError(
                f"Unexpected matched counts for {split}: "
                f"pristine={actual[0]}, generated={actual[1]}; "
                f"expected pristine={expected[0]}, generated={expected[1]}"
            )
        reports[split] = report
    return reports


def load_current_correspondence(
    config: PreparationConfig,
    steps: ValidationSteps,
) -> tuple[
    dict[str, CorrespondenceReport],
    dict[str, pd.DataFrame],
]:
    reports_dir = english_layout(config.data_root).reports_dir
    report_path = reports_dir / "correspondence.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "Current correspondence report is required; run "
            f"validate-protocols first: {report_path}"
        )
    steps.require_protocol_paths(config)
    reports = steps.compute_correspondence(config)
    with report_path.open(encoding="utf-8") as report_file:
        stored = json.load(report_file)
    expected_roots = (
        str(config.jmds_root.resolve()),
        str(asvspoof_root(config).resolve()),
    )
    stored_roots = (
        stored.get("jmds_root"),
        stored.get("asvspoof_protocol_root"),
    )
    if stored_roots != expected_roots:
        raise ValueError("Correspondence report was created for different roots")
    stored_splits = stored.get("splits")
    expected_splits = json.loads(
        json.dumps(
            {
                split: asdict(report)
                for split, report in reports.items()
            }
        )
    )
    if stored_splits != expected_splits:
        raise ValueError(
            "Correspondence report is stale; run validate-protocols again"
        )
    current_eval = steps.compute_eval_evidence(config)
    if stored.get("eval") != current_eval:
        raise ValueError(
            "eval correspondence evidence is stale; "
            "run validate-protocols again"
        )

    protocols = {
        split: steps.sources.read_jmds_protocol(
            ENGLISH_PROFILE.jmds_protocol_path(config.jmds_root, split),
            split,
            language=ENGLISH_PROFILE.language_code,
        )
        for split in ENGLISH_PROFILE.splits
    }
    return reports, protocols


def compute_eval_evidence(
    config: PreparationConfig,
    sources: ProtocolSources,
) -> dict[str, Any]:
    eval_split = ENGLISH_PROFILE.eval_exclusion.split
    jmds_path = ENGLISH_PROFILE.jmds_protocol_path(config.jmds_root, eval_split)
    jmds_full = pd.read_csv(jmds_path, dtype=str, keep_default_na=False)
    if list(jmds_full.columns) != JMDS_COLUMNS:
        raise ValueError("Invalid JMDS eval schema")
    jmds = sources.read_jmds_protocol(
        jmds_path, eval_split, language=ENGLISH_PROFILE.language_code
    )
    asv_path = asvspoof_root(config) / sources.asv_protocol_files[eval_split]
    asv = sources.read_asvspoof_protocol(asv_path, eval_split)

    jmds_ids = set(jmds["utt_id"])
    asv_ids = set(asv["utt_id"])
    equivalence = ENGLISH_PROFILE.label_equivalence
    jmds_by_label = {
        label: set(jmds.loc[jmds["label"] == label, "utt_id"])
        for label in equivalence
    }
    asv_by_label = {
        key: set(asv.loc[asv["key"] == key, "utt_id"])
        for key in equivalence.values()
    }
    labels_correspond = all(
        jmds_by_label[label] == asv_by_label[key]
        for label, key in equivalence.items()
    )
    if jmds_ids == asv_ids and labels_correspond:
        raise ValueError(
            "Eval protocol ID sets and labels now correspond; "
            "review the dataset design before changing the eval policy"
        )
    return {
        "status": ENGLISH_PROFILE.eval_exclusion.status,
        "reason": ENGLISH_PROFILE.eval_exclusion.reason,
        "jmds_fingerprint": protocol_fingerprint(jmds, eval_split),
        "asv_fingerprint": protocol_fingerprint(asv, eval_split),
        "jmds_full_fingerprint": protocol_fingerprint(jmds_full, eval_split),
        "asv_full_fingerprint": protocol_fingerprint(asv, eval_split),
        "jmds_file_sha256": protocol_file_sha256(jmds_path),
        "asv_file_sha256": protocol_file_sha256(asv_path),
        "counts": {
            "jmds": {
                label: len(ids) for label, ids in jmds_by_label.items()
            },
            "asv": {
                label: len(ids) for label, ids in asv_by_label.items()
            },
        },
        "overlap_by_label": {
            f"{label}_{key}": len(jmds_by_label[label] & asv_by_label[key])
            for label, key in equivalence.items()
        },
        "id_overlap_total": len(jmds_ids & asv_ids),
        "jmds_only_count": len(jmds_ids - asv_ids),
        "asv_only_count": len(asv_ids - jmds_ids),
    }


def require_protocol_paths(
    config: PreparationConfig,
    asv_protocol_files: Mapping[str, str],
) -> None:
    root = asvspoof_root(config)
    for split in ENGLISH_PROFILE.protocol_splits:
        jmds = ENGLISH_PROFILE.jmds_protocol_path(config.jmds_root, split)
        if not jmds.is_file():
            raise FileNotFoundError(f"Required JMDS protocol is missing: {jmds}")
        asv = root / asv_protocol_files[split]
        if not asv.is_file():
            raise FileNotFoundError(
                f"Required ASVspoof5 protocol is missing: {asv}"
            )


def asvspoof_root(config: PreparationConfig) -> Path:
    if config.asvspoof_protocol_root is None:
        raise ValueError(
            "asvspoof_protocol_root is required for CLI protocol validation"
        )
    return config.asvspoof_protocol_root
