"""Every Portuguese-specific rule for CORAA and JMDS/MLAAD metadata, in one place."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from jmds_prepare.core.two_source import TwoSourceArtifactNames


@dataclass(frozen=True)
class PortugueseProfile:
    """Immutable description of the Portuguese CORAA and JMDS/MLAAD subsets.

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
    coraa_splits: tuple[str, ...]
    coraa_metadata_files: Mapping[str, str]
    expected_coraa_counts: Mapping[str, int]
    expected_generated_counts: Mapping[str, int]
    split_id_prefixes: Mapping[str, str]
    accepted_labels: frozenset[str]
    jmds_release: str
    coraa_version: str
    coraa_revision: str
    coraa_license: str
    excluded_jmds_pristine_count: int
    coraa_usage_policy_statement: str
    artifact_names: TwoSourceArtifactNames

    def __post_init__(self) -> None:
        for name in ("protocol_splits", "coraa_splits"):
            object.__setattr__(self, name, _frozen_tuple(getattr(self, name)))
        for name in (
            "coraa_metadata_files",
            "expected_coraa_counts",
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

    def coraa_metadata_path(self, coraa_root: Path, split: str) -> Path:
        if split not in self.coraa_splits:
            raise ValueError(f"Unsupported split: {split}")
        return Path(coraa_root) / self.coraa_metadata_files[split]


def _frozen_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return values if isinstance(values, tuple) else tuple(values)


PORTUGUESE_ARTIFACT_NAMES = TwoSourceArtifactNames(
    pristine_manifest="coraa_metadata.csv",
    generated_manifest="jmds_mlaad_generated_metadata.csv",
    pristine_profile="coraa_metadata_profile.json",
    generated_profile="jmds_mlaad_generated_metadata_profile.json",
    summary="portuguese_metadata_summary.json",
    provenance="portuguese_metadata_provenance.json",
)


PORTUGUESE_PROFILE = PortugueseProfile(
    language_code="por",
    language_slug="portuguese",
    generated_dataset="MLAAD",
    generated_dir_name="Portugese_MLAAD_Generated",
    generated_audio_subdir="wav",
    protocol_splits=("train", "dev", "eval"),
    coraa_splits=("train", "dev", "test"),
    coraa_metadata_files={
        "train": "metadata_train_final.csv",
        "dev": "metadata_dev_final.csv",
        "test": "metadata_test_final.csv",
    },
    expected_coraa_counts={"train": 382_258, "dev": 7_522, "test": 12_676},
    expected_generated_counts={"train": 1_806, "dev": 602, "eval": 603},
    split_id_prefixes={"train": "T", "dev": "D", "eval": "E"},
    accepted_labels=frozenset({"pristine", "generated"}),
    jmds_release="v2",
    coraa_version="v1.1",
    coraa_revision="719c91226a79f5f9a8984145f15f29626eabc29a",
    coraa_license="CC-BY-NC-ND-4.0",
    excluded_jmds_pristine_count=1_000,
    coraa_usage_policy_statement=(
        "CC BY-NC-ND 4.0: noncommercial use only. Originals are preserved "
        "byte-identical. Adapted material may be produced and reproduced for "
        "this research but must not be shared; no publication or redistribution "
        "of derived audio, features or subsets."
    ),
    artifact_names=PORTUGUESE_ARTIFACT_NAMES,
)
