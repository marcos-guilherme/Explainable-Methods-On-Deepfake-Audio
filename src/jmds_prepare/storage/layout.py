"""Directory layout of the prepared data tree under ``data_root``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..profiles.english import ENGLISH_PROFILE

ENGLISH_LANGUAGE_SLUG = ENGLISH_PROFILE.language_slug
ASVSPOOF5_SOURCE_SLUG = ENGLISH_PROFILE.source_slug

_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"{device}{index}" for device in ("com", "lpt") for index in range(10)}
)


@dataclass(frozen=True)
class DataLayout:
    """Immutable mapping from one language/source pair to its on-disk paths.

    ``language_slug`` names the processed audio and manifests; ``source_slug``
    names the raw audio extracted from the upstream source. Both are checked
    against a whitelist (lowercase ASCII letters, digits, ``_`` and ``-``,
    starting with a letter or digit, never a reserved Windows device name), so
    each one is a single portable path segment under ``data_root``.

    ``split`` and ``label`` passed to :meth:`raw_split_dir` and
    :meth:`processed_dir` are joined as-is: callers must validate them first
    (the manifest and audio stages only accept their canonical splits and
    labels before deriving any path).
    """

    data_root: Path
    language_slug: str
    source_slug: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        for field_name in ("language_slug", "source_slug"):
            _validate_slug(field_name, getattr(self, field_name))

    @property
    def archive_dir(self) -> Path:
        return self.data_root / "archives"

    @property
    def raw_source_dir(self) -> Path:
        return self.data_root / "raw" / self.source_slug

    def raw_split_dir(self, split: str) -> Path:
        return self.raw_source_dir / split

    @property
    def processed_language_dir(self) -> Path:
        return self.data_root / "processed" / self.language_slug

    def processed_dir(self, split: str, label: str) -> Path:
        return self.processed_language_dir / split / label

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"

    @property
    def raw_manifest_path(self) -> Path:
        return self.manifests_dir / f"{self.language_slug}_raw.csv"

    @property
    def processed_manifest_path(self) -> Path:
        return self.manifests_dir / f"{self.language_slug}_processed.csv"

    @property
    def extraction_ledger_path(self) -> Path:
        return self.manifests_dir / "extraction_ledger.csv"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"


def english_layout(data_root: Path) -> DataLayout:
    """Return the layout of the English ASVspoof5 pristine preparation."""
    return DataLayout(data_root, ENGLISH_LANGUAGE_SLUG, ASVSPOOF5_SOURCE_SLUG)


def _validate_slug(field_name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or _SLUG_PATTERN.fullmatch(value) is None
        or value in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"{field_name} must match {_SLUG_PATTERN.pattern} (lowercase ASCII "
            "letters, digits, '_' or '-', starting with a letter or digit) and "
            f"must not be a reserved Windows device name; received {value!r}"
        )
