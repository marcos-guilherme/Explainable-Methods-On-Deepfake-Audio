"""Immutable, path-safe layout for the XAI selective materialization tree."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jmds_prepare.core.xai_sample import SUPPORTED_LANGUAGES, SUPPORTED_ROLES
from jmds_prepare.storage.layout import _SLUG_PATTERN, _WINDOWS_RESERVED_NAMES

_SUPPORTED_LABELS = frozenset({0, 1})
_LABEL_SLUGS = {0: "bonafide", 1: "spoof"}
_SAMPLE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_SAMPLE_ID_LENGTH = 200


@dataclass(frozen=True)
class XaiLayout:
    """Immutable mapping to the XAI output tree under ``output_root``.

    Processed samples live at
    ``output_root/data/<language>/<role>/<bonafide|spoof>/<sample_id>.wav``.
    Supporting directories ``manifests/``, ``reports/`` and ``staging/`` sit
    beside ``data/`` under the same root.
    """

    output_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def data_dir(self) -> Path:
        return self.output_root / "data"

    @property
    def manifests_dir(self) -> Path:
        return self.output_root / "manifests"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"

    @property
    def staging_dir(self) -> Path:
        return self.output_root / "staging"

    def samples_csv(self, language: str) -> Path:
        """Return ``manifests/<language>/xai_samples.csv``."""
        _require_language(language)
        path = self.manifests_dir / language / "xai_samples.csv"
        _assert_confined(path, root=self.output_root)
        return path

    def selection_report_json(self, language: str) -> Path:
        """Return ``reports/<language>/selection_report.json``."""
        _require_language(language)
        path = self.reports_dir / language / "selection_report.json"
        _assert_confined(path, root=self.output_root)
        return path

    def audio_audit_json(self, language: str) -> Path:
        """Return ``reports/<language>/audio_audit.json``."""
        _require_language(language)
        path = self.reports_dir / language / "audio_audit.json"
        _assert_confined(path, root=self.output_root)
        return path

    def provenance_json(self, language: str) -> Path:
        """Return ``reports/<language>/provenance.json``."""
        _require_language(language)
        path = self.reports_dir / language / "provenance.json"
        _assert_confined(path, root=self.output_root)
        return path

    def sample_path(
        self,
        language: str,
        role: str,
        label: int,
        sample_id: str,
    ) -> Path:
        _require_language(language)
        _require_role(role)
        _require_label(label)
        self.validate_sample_id(sample_id)
        label_slug = _LABEL_SLUGS[label]
        destination = (
            self.data_dir / language / role / label_slug / f"{sample_id}.wav"
        )
        _assert_confined(destination, root=self.data_dir)
        return destination

    @staticmethod
    def validate_sample_id(sample_id: str) -> None:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be non-empty")
        if (
            _SAMPLE_ID_PATTERN.fullmatch(sample_id) is None
            or sample_id in _WINDOWS_RESERVED_NAMES
            or len(sample_id) > _MAX_SAMPLE_ID_LENGTH
        ):
            raise ValueError(
                "sample_id must match "
                f"{_SAMPLE_ID_PATTERN.pattern}, must not be a reserved Windows "
                f"device name, and must be at most {_MAX_SAMPLE_ID_LENGTH} "
                f"characters; received {sample_id!r}"
            )


def _require_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")


def _require_role(role: str) -> None:
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Unsupported role: {role}")


def _require_label(label: int) -> None:
    if isinstance(label, bool) or not isinstance(label, int) or label not in _SUPPORTED_LABELS:
        raise ValueError(f"Unsupported label: {label}")


def _assert_confined(path: Path, *, root: Path) -> None:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes output root: {path}")


def slugify_token(value: str) -> str:
    """Return a filesystem-safe slug derived from ``value``."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized or _SLUG_PATTERN.fullmatch(normalized) is None:
        digest = _short_digest(value)
        return f"id-{digest}"
    if normalized in _WINDOWS_RESERVED_NAMES:
        return f"id-{_short_digest(value)}"
    return normalized[: _MAX_SAMPLE_ID_LENGTH]


def _short_digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:16]
