"""Corpus-agnostic contract for one row of ``xai_samples.csv``."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SUPPORTED_LANGUAGES = ("eng", "por", "zho")
SUPPORTED_ROLES = ("train", "calibration", "test")
_SUPPORTED_LABELS = frozenset({0, 1})

XAI_SAMPLE_COLUMNS = [
    "sample_id",
    "language",
    "role",
    "label",
    "corpus",
    "native_split",
    "original_ref",
    "processed_path",
    "speaker_id",
    "group_id",
    "attack_id",
    "sha256_source",
    "sha256_processed",
    "selection_seed",
    "selection_rank",
    "selection_reason",
    "selection_source",
]

_TEXT_FIELDS = (
    "sample_id",
    "language",
    "role",
    "corpus",
    "native_split",
    "original_ref",
    "processed_path",
    "selection_reason",
    "selection_source",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_PATTERN = re.compile(r"^(0|[1-9]\d*)$")


@dataclass(frozen=True)
class XaiSample:
    """Immutable, validated description of one canonical XAI sample row.

    ``selection_rank`` is zero-based to align with Python enumeration during
    deterministic selection.
    """

    sample_id: str
    language: str
    role: str
    label: int
    corpus: str
    native_split: str
    original_ref: str
    processed_path: str
    speaker_id: str | None
    group_id: str | None
    attack_id: str | None
    sha256_source: str
    sha256_processed: str
    selection_seed: int
    selection_rank: int
    selection_reason: str
    selection_source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "speaker_id", _normalize_optional_text(self.speaker_id)
        )
        object.__setattr__(self, "group_id", _normalize_optional_text(self.group_id))
        object.__setattr__(self, "attack_id", _normalize_optional_text(self.attack_id))
        object.__setattr__(self, "label", _coerce_canonical_label(self.label))
        object.__setattr__(
            self,
            "selection_seed",
            _coerce_canonical_int(self.selection_seed, field="selection_seed"),
        )
        object.__setattr__(
            self,
            "selection_rank",
            _coerce_canonical_int(self.selection_rank, field="selection_rank"),
        )
        for field in _TEXT_FIELDS:
            _require_exact_str(getattr(self, field), field=field)
        object.__setattr__(
            self,
            "sha256_source",
            _coerce_canonical_sha256(self.sha256_source, field="sha256_source"),
        )
        object.__setattr__(
            self,
            "sha256_processed",
            _coerce_canonical_sha256(
                self.sha256_processed, field="sha256_processed"
            ),
        )

        _validate_sample(
            sample_id=self.sample_id,
            language=self.language,
            role=self.role,
            label=self.label,
            corpus=self.corpus,
            native_split=self.native_split,
            original_ref=self.original_ref,
            processed_path=self.processed_path,
            speaker_id=self.speaker_id,
            group_id=self.group_id,
            attack_id=self.attack_id,
            sha256_source=self.sha256_source,
            sha256_processed=self.sha256_processed,
            selection_seed=self.selection_seed,
            selection_rank=self.selection_rank,
            selection_reason=self.selection_reason,
            selection_source=self.selection_source,
        )

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, Any],
        *,
        enforce_column_order: bool = False,
    ) -> XaiSample:
        return cls(**cls._parsed_row(row, enforce_column_order=enforce_column_order))

    @classmethod
    def validate_dict(
        cls,
        row: Mapping[str, Any],
        *,
        enforce_column_order: bool = False,
    ) -> None:
        cls._parsed_row(row, enforce_column_order=enforce_column_order)

    @classmethod
    def _parsed_row(
        cls,
        row: Mapping[str, Any],
        *,
        enforce_column_order: bool,
    ) -> dict[str, Any]:
        if enforce_column_order:
            if tuple(row.keys()) != tuple(XAI_SAMPLE_COLUMNS):
                raise ValueError("Column order does not match the canonical schema")

        missing = [column for column in XAI_SAMPLE_COLUMNS if column not in row]
        if missing:
            raise ValueError(
                "Missing required columns: " + ", ".join(missing)
            )

        extra = [column for column in row if column not in XAI_SAMPLE_COLUMNS]
        if extra:
            raise ValueError(
                "Unexpected columns: " + ", ".join(sorted(extra))
            )

        return {
            "sample_id": str(row["sample_id"]),
            "language": str(row["language"]),
            "role": str(row["role"]),
            "label": _parse_label(row["label"]),
            "corpus": str(row["corpus"]),
            "native_split": str(row["native_split"]),
            "original_ref": str(row["original_ref"]),
            "processed_path": str(row["processed_path"]),
            "speaker_id": _parse_optional_text(row["speaker_id"]),
            "group_id": _parse_optional_text(row["group_id"]),
            "attack_id": _parse_optional_text(row["attack_id"]),
            "sha256_source": _parse_sha256(row["sha256_source"], field="sha256_source"),
            "sha256_processed": _parse_sha256(
                row["sha256_processed"], field="sha256_processed"
            ),
            "selection_seed": _parse_non_negative_int(
                row["selection_seed"], field="selection_seed"
            ),
            "selection_rank": _parse_non_negative_int(
                row["selection_rank"], field="selection_rank"
            ),
            "selection_reason": str(row["selection_reason"]),
            "selection_source": str(row["selection_source"]),
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "language": self.language,
            "role": self.role,
            "label": str(self.label),
            "corpus": self.corpus,
            "native_split": self.native_split,
            "original_ref": self.original_ref,
            "processed_path": self.processed_path,
            "speaker_id": self.speaker_id or "",
            "group_id": self.group_id or "",
            "attack_id": self.attack_id or "",
            "sha256_source": self.sha256_source,
            "sha256_processed": self.sha256_processed,
            "selection_seed": str(self.selection_seed),
            "selection_rank": str(self.selection_rank),
            "selection_reason": self.selection_reason,
            "selection_source": self.selection_source,
        }


def _validate_sample(
    *,
    sample_id: str,
    language: str,
    role: str,
    label: int,
    corpus: str,
    native_split: str,
    original_ref: str,
    processed_path: str,
    speaker_id: str | None,
    group_id: str | None,
    attack_id: str | None,
    sha256_source: str,
    sha256_processed: str,
    selection_seed: int,
    selection_rank: int,
    selection_reason: str,
    selection_source: str,
) -> None:
    _require_non_empty_text(sample_id, field="sample_id")
    _require_language(language)
    _require_role(role)
    if label not in _SUPPORTED_LABELS:
        raise ValueError(f"Unsupported label: {label}")
    _require_non_empty_text(corpus, field="corpus")
    _require_non_empty_text(native_split, field="native_split")
    _require_non_empty_text(original_ref, field="original_ref")
    _require_non_empty_text(processed_path, field="processed_path")
    _require_non_empty_text(selection_reason, field="selection_reason")
    _require_non_empty_text(selection_source, field="selection_source")
    if label == 0 and attack_id is not None:
        raise ValueError("attack_id must be empty for label 0")


def _require_exact_str(value: Any, *, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    _require_exact_str(value, field="optional text field")
    text = value.strip()
    return text or None


def _parse_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_canonical_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Unsupported label: {value}")
    if not isinstance(value, int):
        raise ValueError(f"Unsupported label: {value}")
    if value not in _SUPPORTED_LABELS:
        raise ValueError(f"Unsupported label: {value}")
    return value


def _coerce_canonical_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}: {value}")
    if isinstance(value, float):
        raise ValueError(f"Invalid {field}: {value}")
    if not isinstance(value, int):
        raise ValueError(f"Invalid {field}: {value}")
    if value < 0:
        raise ValueError(f"Invalid {field}: {value}")
    return value


def _require_non_empty_text(value: Any, *, field: str) -> None:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} must be non-empty")


def _require_language(value: Any) -> None:
    language = str(value)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")


def _require_role(value: Any) -> None:
    role = str(value)
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role: {role}")


def _parse_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Unsupported label: {value}")
    if isinstance(value, int):
        if value in _SUPPORTED_LABELS:
            return value
        raise ValueError(f"Unsupported label: {value}")
    if isinstance(value, float):
        raise ValueError(f"Unsupported label: {value}")
    if not isinstance(value, str):
        raise ValueError(f"Unsupported label: {value}")
    if value not in {"0", "1"}:
        raise ValueError(f"Unsupported label: {value}")
    return int(value)


def _parse_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}: {value}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"Invalid {field}: {value}")
        return value
    if isinstance(value, float):
        raise ValueError(f"Invalid {field}: {value}")
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field}: {value}")
    if not _DECIMAL_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value}")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"Invalid {field}: {value}")
    return parsed


def _parse_sha256(value: Any, *, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}: {value}")
    text = value if isinstance(value, str) else str(value)
    return _coerce_canonical_sha256(text, field=field)


def _coerce_canonical_sha256(value: Any, *, field: str) -> str:
    _require_exact_str(value, field=field)
    if value != value.lower():
        raise ValueError(f"Invalid {field}: {value}")
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value}")
    return value
