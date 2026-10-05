"""Strict Mandarin JMDS/ADD generated metadata adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..profiles.mandarin import MandarinProfile
from .jmds import JMDS_COLUMNS
from .jmds_common import (
    validate_schema,
    validate_split,
    validate_unique_ids,
    validate_utterance_ids,
)

_TRACEABILITY_COLUMNS = (
    "split",
    "metadata_source_file",
    "metadata_source_row",
    "audio_path",
)

_ADD_SPARSE_FIELDS = {
    "attack_id": "unk",
    "gender": "unk",
    "spk_id": "unk",
    "codec": "-",
    "native": "yes",
}


def read_mandarin_generated(
    path: Path,
    *,
    split: str,
    jmds_root: Path,
    profile: MandarinProfile,
) -> pd.DataFrame:
    validate_split(split, profile.protocol_splits)
    protocol = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_schema(list(protocol.columns), JMDS_COLUMNS, source="JMDS")
    _validate_mandarin_row_combinations(protocol, profile)
    selected = _select_generated_subset(protocol, profile)
    validate_utterance_ids(
        selected,
        split=split,
        prefixes=profile.split_id_prefixes,
        source="JMDS",
    )
    validate_unique_ids(selected, source="JMDS")
    _validate_expected_count(selected, split=split, profile=profile)
    _validate_add_sparse_fields(selected)
    audio_paths = _resolve_audio_paths(
        selected, split=split, jmds_root=jmds_root, profile=profile
    )
    _validate_resolved_audio_paths(
        audio_paths, split=split, jmds_root=jmds_root, profile=profile
    )
    return _with_traceability(
        selected, path=path, split=split, audio_paths=audio_paths
    )


def _validate_mandarin_row_combinations(
    protocol: pd.DataFrame, profile: MandarinProfile
) -> None:
    mandarin = protocol.loc[protocol["language"] == profile.language_code]
    valid_generated = (mandarin["label"] == "generated") & (
        mandarin["dataset"] == profile.generated_dataset
    )
    valid_pristine = (mandarin["label"] == "pristine") & (
        mandarin["dataset"] == "AISHELL3"
    )
    invalid = mandarin.loc[~(valid_generated | valid_pristine)]
    if invalid.empty:
        return

    wrong_dataset = invalid.loc[
        (invalid["label"] == "generated")
        & (invalid["dataset"] != profile.generated_dataset)
    ]
    if not wrong_dataset.empty:
        unexpected = sorted(wrong_dataset["dataset"].unique())
        raise ValueError(
            "Unexpected Mandarin generated dataset value(s): "
            + ", ".join(unexpected)
        )

    wrong_label = invalid.loc[
        (invalid["dataset"] == profile.generated_dataset)
        & (invalid["label"] != "generated")
    ]
    if not wrong_label.empty:
        unexpected = sorted(wrong_label["label"].unique())
        raise ValueError(
            "Unexpected Mandarin generated label value(s): "
            + ", ".join(unexpected)
        )

    unexpected = sorted(invalid["label"].unique())
    raise ValueError(
        "Unexpected Mandarin generated label value(s): "
        + ", ".join(unexpected)
    )


def _select_generated_subset(
    protocol: pd.DataFrame, profile: MandarinProfile
) -> pd.DataFrame:
    return protocol.loc[
        (protocol["language"] == profile.language_code)
        & (protocol["label"] == "generated")
        & (protocol["dataset"] == profile.generated_dataset)
    ].copy()


def _validate_expected_count(
    selected: pd.DataFrame, *, split: str, profile: MandarinProfile
) -> None:
    expected = profile.expected_generated_counts[split]
    actual = len(selected)
    if actual != expected:
        raise ValueError(
            f"Expected {expected} Mandarin generated rows for {split}, got {actual}"
        )


def _validate_add_sparse_fields(selected: pd.DataFrame) -> None:
    for field, expected in _ADD_SPARSE_FIELDS.items():
        wrong = selected.loc[selected[field] != expected]
        if wrong.empty:
            continue
        unexpected = sorted(wrong[field].unique())
        raise ValueError(
            f"Unexpected ADD {field} value(s): {', '.join(unexpected)}"
        )


def _resolve_audio_paths(
    selected: pd.DataFrame,
    *,
    split: str,
    jmds_root: Path,
    profile: MandarinProfile,
) -> pd.Series:
    return selected["utt_id"].map(
        lambda utt_id: str(profile.generated_wav_path(jmds_root, split, utt_id))
    )


def _validate_resolved_audio_paths(
    audio_paths: pd.Series,
    *,
    split: str,
    jmds_root: Path,
    profile: MandarinProfile,
) -> None:
    paths = audio_paths.map(Path)
    missing = sorted(
        str(path)
        for path in paths
        if not path.is_file()
    )
    if missing:
        raise ValueError(
            "Missing JMDS generated wav file(s): " + ", ".join(missing)
        )

    generated_root = _generated_split_root(jmds_root, split=split, profile=profile)
    resolved = paths.map(lambda value: value.resolve())
    outside = [
        str(path)
        for path in resolved
        if not path.is_relative_to(generated_root)
    ]
    if outside:
        raise ValueError(
            "Resolved JMDS audio_path must remain under the generated split root"
        )

    duplicates = sorted(
        resolved.loc[resolved.duplicated(keep=False)].astype(str).unique()
    )
    if duplicates:
        raise ValueError(
            f"Duplicate JMDS audio_path value(s): {', '.join(duplicates)}"
        )


def _generated_split_root(
    jmds_root: Path, *, split: str, profile: MandarinProfile
) -> Path:
    return (
        Path(jmds_root).resolve()
        / "dataset"
        / profile.generated_dir_name
        / split
        / profile.generated_audio_subdir
    ).resolve()


def _with_traceability(
    frame: pd.DataFrame,
    *,
    path: Path,
    split: str,
    audio_paths: pd.Series,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["split"] = split
    enriched["metadata_source_file"] = Path(path).name
    enriched["metadata_source_row"] = (enriched.index + 2).astype(str)
    enriched["audio_path"] = audio_paths.to_numpy()
    return enriched[[*JMDS_COLUMNS, *_TRACEABILITY_COLUMNS]]
