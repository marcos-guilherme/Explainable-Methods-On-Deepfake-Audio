"""Every English-specific rule of the JMDS/ASVspoof5 preparation, in one place."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class EvalExclusionPolicy:
    """Why a protocol split is kept out of the prepared data."""

    split: str
    status: str
    reason: str


@dataclass(frozen=True)
class EnglishProfile:
    """Immutable description of the English JMDS subset and its ASVspoof5 source.

    Mappings are stored as read-only proxies over private copies and sequences
    as tuples, so neither the caller's inputs nor the shared instance can be
    mutated after construction.
    """

    language_code: str
    language_slug: str
    dataset: str
    generated_dir_name: str
    source_slug: str
    metadata_source: str
    splits: tuple[str, ...]
    protocol_splits: tuple[str, ...]
    asv_protocol_files: Mapping[str, str]
    expected_pristine: Mapping[str, int]
    expected_generated: Mapping[str, int]
    zenodo_record_id: int
    required_archives: tuple[str, ...]
    eval_exclusion: EvalExclusionPolicy
    split_id_prefixes: Mapping[str, str]
    label_equivalence: Mapping[str, str]
    archive_prefix_splits: Mapping[str, str]
    generated_audio_subdir: str

    def __post_init__(self) -> None:
        for name in ("splits", "protocol_splits", "required_archives"):
            object.__setattr__(self, name, _frozen_tuple(getattr(self, name)))
        for name in (
            "asv_protocol_files",
            "expected_pristine",
            "expected_generated",
            "split_id_prefixes",
            "label_equivalence",
            "archive_prefix_splits",
        ):
            object.__setattr__(
                self, name, MappingProxyType(dict(getattr(self, name)))
            )

    def generated_root(self, jmds_root: Path) -> Path:
        """Directory holding the JMDS generated audio for this language."""
        return Path(jmds_root) / "dataset" / self.generated_dir_name

    def generated_split_dir(self, jmds_root: Path, split: str) -> Path:
        """Directory holding the JMDS generated WAV files of one split."""
        return self.generated_root(jmds_root) / split / self.generated_audio_subdir

    def jmds_protocol_filename(self, split: str) -> str:
        if split not in self.protocol_splits:
            raise ValueError(f"Unsupported split: {split}")
        return f"open_v2_{split}.cm.csv"

    def jmds_protocol_path(self, jmds_root: Path, split: str) -> Path:
        return Path(jmds_root) / "cm_protocols" / self.jmds_protocol_filename(split)

    def archive_split(self, name: str) -> str:
        """Split whose pristine FLACs an official ASVspoof5 TAR contains."""
        for prefix, split in self.archive_prefix_splits.items():
            if name.startswith(prefix):
                return split
        raise ValueError(f"Cannot determine split for archive: {name}")


def _frozen_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return values if isinstance(values, tuple) else tuple(values)


ENGLISH_PROFILE = EnglishProfile(
    language_code="eng",
    language_slug="english",
    dataset="ASVspoof2024",
    generated_dir_name="English_ASVspoof2024_Generated",
    source_slug="asvspoof5",
    metadata_source="JMDS+ASVspoof5-verified",
    splits=("train", "dev"),
    protocol_splits=("train", "dev", "eval"),
    asv_protocol_files={
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    },
    expected_pristine={"train": 18_797, "dev": 15_667},
    expected_generated={"train": 65_424, "dev": 54_808},
    zenodo_record_id=14498691,
    required_archives=(
        *(f"flac_T_a{suffix}.tar" for suffix in "abcde"),
        *(f"flac_D_a{suffix}.tar" for suffix in "abc"),
    ),
    eval_exclusion=EvalExclusionPolicy(
        split="eval",
        status="excluded_known_id_mismatch",
        reason=(
            "JMDS eval IDs do not exactly correspond to the official "
            "ASVspoof5 eval protocol"
        ),
    ),
    split_id_prefixes={"train": "T", "dev": "D", "eval": "E"},
    label_equivalence={"pristine": "bonafide", "generated": "spoof"},
    archive_prefix_splits={"flac_T_": "train", "flac_D_": "dev"},
    generated_audio_subdir="wav",
)
