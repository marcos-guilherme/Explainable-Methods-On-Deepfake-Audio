"""Every Mandarin-specific rule for AISHELL-3 and JMDS/ADD metadata, in one place."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from jmds_prepare.core.two_source import TwoSourceArtifactNames


@dataclass(frozen=True)
class MandarinProfile:
    """Immutable description of the Mandarin AISHELL-3 and JMDS/ADD subsets.

    Mappings are stored as read-only proxies over private copies and sequences
    as tuples, so neither the caller's inputs nor the shared instance can be
    mutated after construction.
    """

    language_code: str
    language_slug: str
    generated_dataset: str
    generated_dir_name: str
    generated_audio_subdir: str
    protocol_splits: tuple[str, ...]
    aishell_splits: tuple[str, ...]
    aishell_archive_members: Mapping[str, str]
    expected_generated_counts: Mapping[str, int]
    split_id_prefixes: Mapping[str, str]
    excluded_jmds_pristine_count: int
    artifact_names: TwoSourceArtifactNames

    def __post_init__(self) -> None:
        for name in ("protocol_splits", "aishell_splits"):
            object.__setattr__(self, name, _frozen_tuple(getattr(self, name)))
        for name in (
            "aishell_archive_members",
            "expected_generated_counts",
            "split_id_prefixes",
        ):
            object.__setattr__(
                self, name, MappingProxyType(dict(getattr(self, name)))
            )

    def jmds_protocol_path(self, jmds_root: Path, split: str) -> Path:
        if split not in self.protocol_splits:
            raise ValueError(f"Unsupported split: {split}")
        return Path(jmds_root) / "cm_protocols" / f"open_v2_{split}.cm.csv"

    def generated_wav_path(
        self, jmds_root: Path, split: str, utt_id: str
    ) -> Path:
        if split not in self.protocol_splits:
            raise ValueError(f"Unsupported split: {split}")
        return (
            Path(jmds_root)
            / "dataset"
            / self.generated_dir_name
            / split
            / self.generated_audio_subdir
            / f"{utt_id}.wav"
        )


def _frozen_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return values if isinstance(values, tuple) else tuple(values)


MANDARIN_ARTIFACT_NAMES = TwoSourceArtifactNames(
    pristine_manifest="aishell3_metadata.csv",
    generated_manifest="jmds_add_generated_metadata.csv",
    pristine_profile="aishell3_metadata_profile.json",
    generated_profile="jmds_add_generated_metadata_profile.json",
    summary="mandarin_metadata_summary.json",
    provenance="mandarin_metadata_provenance.json",
)


MANDARIN_PROFILE = MandarinProfile(
    language_code="zho",
    language_slug="mandarin",
    generated_dataset="ADD",
    generated_dir_name="Chinese_ADD_Generated",
    generated_audio_subdir="wav",
    protocol_splits=("train", "dev", "eval"),
    aishell_splits=("train", "test"),
    aishell_archive_members={
        "spk-info": "spk-info.txt",
        "train/content": "train/content.txt",
        "test/content": "test/content.txt",
        "train/label": "train/label_train-set.txt",
    },
    expected_generated_counts={"train": 7146, "dev": 7497, "eval": 9999},
    split_id_prefixes={"train": "T", "dev": "D", "eval": "E"},
    excluded_jmds_pristine_count=4410,
    artifact_names=MANDARIN_ARTIFACT_NAMES,
)
