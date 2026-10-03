"""ASVspoof5 protocol reader and its correspondence with the JMDS protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.hashing import sha256_file
from ..profiles.english import ENGLISH_PROFILE
from .jmds import _validate_split, _validate_unique_ids, _validate_utterance_ids


ASV_COLUMNS = [
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

_ASV_LABELS = frozenset(ENGLISH_PROFILE.label_equivalence.values())
_EQUIVALENT_LABEL = ENGLISH_PROFILE.label_equivalence


@dataclass
class CorrespondenceReport:
    split: str
    pristine_expected: int
    generated_expected: int
    pristine_matched: int
    generated_matched: int
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    metadata_mismatches: dict[str, dict[str, tuple[str, str]]]
    jmds_fingerprint: str = ""
    asv_fingerprint: str = ""
    jmds_full_fingerprint: str = ""
    asv_full_fingerprint: str = ""
    jmds_file_sha256: str = ""
    asv_file_sha256: str = ""

    def raise_for_errors(self) -> None:
        errors: list[str] = []
        if self.missing_ids:
            errors.append(f"missing IDs: {', '.join(self.missing_ids)}")
        if self.extra_ids:
            errors.append(f"extra IDs: {', '.join(self.extra_ids)}")
        for utt_id, mismatches in self.metadata_mismatches.items():
            fields = ", ".join(sorted(mismatches))
            errors.append(f"{utt_id} metadata mismatch: {fields}")
        if errors:
            raise ValueError(
                f"Correspondence errors for {self.split}: {'; '.join(errors)}"
            )


def read_asvspoof_protocol(path: Path, split: str) -> pd.DataFrame:
    _validate_split(split)
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as protocol_file:
        for line_number, line in enumerate(protocol_file, start=1):
            fields = line.split()
            if len(fields) != len(ASV_COLUMNS):
                raise ValueError(
                    f"ASVspoof row {line_number} must contain exactly "
                    f"{len(ASV_COLUMNS)} fields, found {len(fields)}"
                )
            rows.append(fields)

    protocol = pd.DataFrame(rows, columns=ASV_COLUMNS, dtype=str)
    _validate_unique_ids(protocol, "ASVspoof")
    _validate_utterance_ids(protocol, split, "ASVspoof")
    unsupported = sorted(set(protocol["key"]) - _ASV_LABELS)
    if unsupported:
        raise ValueError(f"Unsupported ASVspoof key(s): {', '.join(unsupported)}")
    return protocol


def validate_correspondence(
    jmds: pd.DataFrame, asv: pd.DataFrame, split: str
) -> CorrespondenceReport:
    _validate_split(split)
    languages = sorted(set(jmds["language"]) - {ENGLISH_PROFILE.language_code})
    if languages:
        raise ValueError(
            "English correspondence validation received non-English "
            f"language(s): {', '.join(languages)}"
        )

    jmds_by_id = jmds.set_index("utt_id", drop=False)
    asv_by_id = asv.set_index("utt_id", drop=False)
    jmds_ids = set(jmds_by_id.index)
    asv_ids = set(asv_by_id.index)
    shared_ids = sorted(jmds_ids & asv_ids)

    metadata_mismatches: dict[str, dict[str, tuple[str, str]]] = {}
    matched = {"pristine": 0, "generated": 0}
    for utt_id in shared_ids:
        jmds_row = jmds_by_id.loc[utt_id]
        asv_row = asv_by_id.loc[utt_id]
        row_mismatches: dict[str, tuple[str, str]] = {}

        for field in ("spk_id", "gender"):
            if jmds_row[field] != asv_row[field]:
                row_mismatches[field] = (jmds_row[field], asv_row[field])

        expected_asv_label = _EQUIVALENT_LABEL[jmds_row["label"]]
        if expected_asv_label == asv_row["key"]:
            matched[jmds_row["label"]] += 1
        else:
            row_mismatches["label"] = (
                jmds_row["label"],
                asv_row["key"],
            )

        if row_mismatches:
            metadata_mismatches[utt_id] = row_mismatches

    return CorrespondenceReport(
        split=split,
        jmds_fingerprint=protocol_fingerprint(jmds, split),
        asv_fingerprint=protocol_fingerprint(asv, split),
        pristine_expected=int((jmds["label"] == "pristine").sum()),
        generated_expected=int((jmds["label"] == "generated").sum()),
        pristine_matched=matched["pristine"],
        generated_matched=matched["generated"],
        missing_ids=tuple(sorted(jmds_ids - asv_ids)),
        extra_ids=tuple(sorted(asv_ids - jmds_ids)),
        metadata_mismatches=metadata_mismatches,
    )


def protocol_fingerprint(protocol: pd.DataFrame, split: str) -> str:
    columns = list(protocol.columns)
    canonical = protocol.sort_values(by=columns, kind="mergesort")
    payload = {
        "split": split,
        "columns": columns,
        "rows": canonical.loc[:, columns].astype(str).values.tolist(),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def protocol_file_sha256(path: Path) -> str:
    """Hash the exact protocol bytes, including ordering and line endings."""
    return sha256_file(path)
