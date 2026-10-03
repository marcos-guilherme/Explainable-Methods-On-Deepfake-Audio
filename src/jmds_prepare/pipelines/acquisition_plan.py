"""Dry-run planning for the English acquisition."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from ..config import PreparationConfig
from ..profiles.english import ENGLISH_PROFILE
from ..storage.estimates import (
    acquisition_storage_state as acquisition_storage_state,
    audio_storage_estimate as audio_storage_estimate,
    bulk_generated_pcm16_bytes as bulk_generated_pcm16_bytes,
    existing_ancestor as existing_ancestor,
    measure_audio_header as measure_audio_header,
)
from ..storage.layout import english_layout
from .common import Output

PROCESSING_SPACE_NOTE = (
    "process-english checks free space for the missing PCM-16 outputs "
    "before writing any audio or manifest"
)


@dataclass(frozen=True)
class DryRunDeps:
    required_archives: Sequence[str]
    acquisition_storage_state: Callable[
        [dict[str, Any], Path, Path, Path], dict[str, int]
    ]
    audio_storage_estimate: Callable[
        [PreparationConfig, dict[str, pd.DataFrame]], dict[str, Any]
    ]
    valid_extracted_ids: Callable[..., set[str]]


def print_dry_run(
    config: PreparationConfig,
    files: dict[str, Any],
    targets: dict[str, set[str]],
    protocols: dict[str, pd.DataFrame],
    *,
    archive_root: Path,
    raw_root: Path,
    output: Output,
    deps: DryRunDeps,
) -> None:
    total = sum(files[name].size for name in deps.required_archives)
    storage = deps.acquisition_storage_state(
        files,
        archive_root,
        raw_root,
        english_layout(config.data_root).extraction_ledger_path,
    )
    audio = deps.audio_storage_estimate(config, protocols)
    maximum_temporary = max(
        storage["largest_active_remaining_tar_bytes"],
        audio["maximum_single_output_bytes"],
    )
    free = shutil.disk_usage(existing_ancestor(config.data_root)).free
    output("DRY RUN: no archives or audio will be created or deleted")
    output(f"archive output: {archive_root}")
    output(f"audio output: {raw_root}")
    for name in deps.required_archives:
        file = files[name]
        output(
            f"{name}: size={file.size} bytes md5={file.checksum} "
            f"path={archive_root / name}"
        )
    output(f"exact TAR download bytes: {total}")
    output(
        "maximum active TAR working set bytes: "
        f"{storage['maximum_active_tar_working_set_bytes']} "
        "(current archives/partials/quarantines + largest active remaining TAR)"
    )
    output(
        "conservative total acquisition peak bytes: "
        f"{storage['conservative_total_acquisition_peak_bytes']} "
        f"(extracted FLAC bytes={storage['extracted_flac_bytes']} + "
        "current archives/partials/quarantines="
        f"{storage['current_archive_partial_quarantine_bytes']} + "
        "remaining retained-FLAC upper bound="
        f"{storage['remaining_flac_upper_bound_bytes']} + "
        "largest active remaining TAR="
        f"{storage['largest_active_remaining_tar_bytes']})"
    )
    output(
        "generated processed PCM-16 payload bytes from current audio headers: "
        f"{audio['generated']['processed_pcm16_bytes']} "
        f"(files measured={audio['generated']['files_measured']}; "
        f"method={audio['generated'].get('measurement_method', 'per-file headers')})"
    )
    output(
        "pristine processed PCM-16 payload bytes from current audio headers: "
        f"{audio['pristine']['processed_pcm16_bytes']} "
        f"(files measured={audio['pristine']['files_measured']}; "
        f"files pending extraction={audio['pristine']['files_pending_extraction']}; "
        f"status={audio['pristine']['status']})"
    )
    output(
        "maximum temporary bytes: "
        f"{maximum_temporary} "
        "(max of largest TAR partial and largest calculated single PCM-16 output)"
    )
    output(PROCESSING_SPACE_NOTE)
    output(f"free space: {free}")
    for split in ENGLISH_PROFILE.splits:
        existing = deps.valid_extracted_ids(raw_root / split, targets[split])
        output(
            f"{split}: target={len(targets[split])} "
            f"existing_valid={len(existing)} "
            f"outstanding={len(targets[split] - existing)}"
        )
