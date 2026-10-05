"""Immutable XAI selection profiles and per-language native-split mappings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from jmds_prepare.core.xai_sample import SUPPORTED_LANGUAGES, SUPPORTED_ROLES

SHARED_SPLIT_DISJOINT_FIELDS = frozenset({"speaker_id", "group_id"})

XAI_PER_CLASS_TARGETS = MappingProxyType(
    {
        "train": 1_806,
        "calibration": 602,
        "test": 603,
    }
)

XAI_ROWS_PER_LANGUAGE = sum(XAI_PER_CLASS_TARGETS.values()) * 2
XAI_TOTAL_ROWS = XAI_ROWS_PER_LANGUAGE * 3


def per_class_row_total(role: str) -> int:
    """Return bonafide+spoof rows for one experimental role."""
    if role not in XAI_PER_CLASS_TARGETS:
        raise ValueError(f"Unsupported role: {role}")
    return XAI_PER_CLASS_TARGETS[role] * 2


def per_language_row_total() -> int:
    """Return total rows for one language across both classes."""
    return XAI_ROWS_PER_LANGUAGE


def _frozen_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return values if isinstance(values, tuple) else tuple(values)


def _validate_string_keys(mapping: Mapping[Any, Any], *, context: str) -> None:
    for key in mapping:
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings")


def _readonly_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    _validate_string_keys(mapping, context="per_class_targets")
    return MappingProxyType(dict(mapping))


def _readonly_nested_roles(
    mapping: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    context: str,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    _validate_string_keys(mapping, context=context)
    frozen: dict[str, Mapping[str, tuple[str, ...]]] = {}
    for corpus, split_roles in mapping.items():
        split_context = f"{context}[{corpus!r}]"
        _validate_string_keys(split_roles, context=split_context)
        frozen[corpus] = MappingProxyType(
            {
                native_split: _frozen_tuple(roles)
                for native_split, roles in split_roles.items()
            }
        )
    return MappingProxyType(frozen)


def _validate_corpus_labels(
    corpus_labels: Mapping[str, int],
    corpus_native_split_roles: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> None:
    _validate_string_keys(corpus_labels, context="corpus_labels")
    if not corpus_labels:
        raise ValueError("corpus_labels must be non-empty")
    for corpus, label in corpus_labels.items():
        if not corpus.strip():
            raise ValueError("corpus_labels keys must be non-empty")
        if label not in {0, 1}:
            raise ValueError(f"corpus_labels[{corpus!r}] must be 0 or 1")
    unknown = set(corpus_native_split_roles) - set(corpus_labels)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"corpus_native_split_roles references unlabeled corpus: {joined}")
    unused = set(corpus_labels) - set(corpus_native_split_roles)
    if unused:
        joined = ", ".join(sorted(unused))
        raise ValueError(f"corpus_labels references unknown corpus: {joined}")
    labels_by_value = {0: [], 1: []}
    for corpus, label in corpus_labels.items():
        labels_by_value[label].append(corpus)
    for label, corpora in labels_by_value.items():
        if len(corpora) != 1:
            joined = ", ".join(sorted(corpora))
            raise ValueError(
                f"corpus_labels must map exactly one corpus to label {label}, "
                f"found: {joined}"
            )


def _validate_profile(
    *,
    language_code: str,
    language_slug: str,
    per_class_targets: Mapping[str, int],
    corpus_labels: Mapping[str, int],
    corpus_native_split_roles: Mapping[str, Mapping[str, tuple[str, ...]]],
    shared_split_disjoint_fields: Mapping[str, Mapping[str, tuple[str, ...]]],
    paired_selection: bool,
) -> None:
    if type(paired_selection) is not bool:
        raise ValueError("paired_selection must be a bool")
    if language_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language_code: {language_code}")
    if not language_slug.strip():
        raise ValueError("language_slug must be non-empty")

    if set(per_class_targets) != set(SUPPORTED_ROLES):
        raise ValueError(
            "per_class_targets must contain exactly the supported roles: "
            + ", ".join(SUPPORTED_ROLES)
        )
    for role in SUPPORTED_ROLES:
        count = per_class_targets[role]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"per_class_targets[{role!r}] must be a positive int")

    if not corpus_native_split_roles:
        raise ValueError("corpus_native_split_roles must be non-empty")
    _validate_corpus_labels(corpus_labels, corpus_native_split_roles)

    for corpus, split_roles in corpus_native_split_roles.items():
        if not corpus.strip():
            raise ValueError("corpus names must be non-empty")
        if not split_roles:
            raise ValueError(f"corpus {corpus!r} must define at least one native split")
        for native_split, roles in split_roles.items():
            if not native_split.strip():
                raise ValueError(f"native split names must be non-empty in {corpus!r}")
            if not roles:
                raise ValueError(
                    f"{corpus}/{native_split} must map to at least one role"
                )
            _validate_role_tuple(roles, context=f"{corpus}/{native_split}")
            if len(roles) >= 2:
                fields = shared_split_disjoint_fields.get(corpus, {}).get(
                    native_split, ()
                )
                if not fields:
                    raise ValueError(
                        f"{corpus}/{native_split} feeds multiple roles and must "
                        "declare shared_split_disjoint_fields"
                    )

    for corpus, split_fields in shared_split_disjoint_fields.items():
        if corpus not in corpus_native_split_roles:
            raise ValueError(
                f"shared_split_disjoint_fields references unknown corpus: {corpus}"
            )
        for native_split, fields in split_fields.items():
            if native_split not in corpus_native_split_roles[corpus]:
                raise ValueError(
                    "shared_split_disjoint_fields references unknown native split: "
                    f"{corpus}/{native_split}"
                )
            roles = corpus_native_split_roles[corpus][native_split]
            if len(roles) < 2:
                raise ValueError(
                    f"{corpus}/{native_split} feeds a single role and cannot declare "
                    "shared_split_disjoint_fields"
                )
            if not fields:
                raise ValueError(
                    f"{corpus}/{native_split} must declare at least one disjoint field"
                )
            if len(fields) != len(set(fields)):
                raise ValueError(
                    f"{corpus}/{native_split} declares duplicate disjoint fields"
                )
            unknown = set(fields) - SHARED_SPLIT_DISJOINT_FIELDS
            if unknown:
                joined = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Unsupported shared_split_disjoint_field(s): {joined}"
                )


def _validate_role_tuple(roles: tuple[str, ...], *, context: str) -> None:
    if len(roles) != len(set(roles)):
        raise ValueError(f"{context} declares duplicate roles")
    for role in roles:
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"Unsupported role in {context}: {role}")


@dataclass(frozen=True)
class XaiLanguageProfile:
    """Immutable description of deterministic XAI sample selection for one language."""

    language_code: str
    language_slug: str
    per_class_targets: Mapping[str, int]
    corpus_labels: Mapping[str, int]
    corpus_native_split_roles: Mapping[str, Mapping[str, tuple[str, ...]]]
    shared_split_disjoint_fields: Mapping[str, Mapping[str, tuple[str, ...]]]
    paired_selection: bool

    def __post_init__(self) -> None:
        per_class_targets = _readonly_mapping(self.per_class_targets)
        corpus_labels = _readonly_mapping(self.corpus_labels)
        corpus_native_split_roles = _readonly_nested_roles(
            self.corpus_native_split_roles,
            context="corpus_native_split_roles",
        )
        shared_split_disjoint_fields = _readonly_nested_roles(
            self.shared_split_disjoint_fields,
            context="shared_split_disjoint_fields",
        )
        _validate_profile(
            language_code=self.language_code,
            language_slug=self.language_slug,
            per_class_targets=per_class_targets,
            corpus_labels=corpus_labels,
            corpus_native_split_roles=corpus_native_split_roles,
            shared_split_disjoint_fields=shared_split_disjoint_fields,
            paired_selection=self.paired_selection,
        )
        object.__setattr__(self, "per_class_targets", per_class_targets)
        object.__setattr__(self, "corpus_labels", corpus_labels)
        object.__setattr__(
            self, "corpus_native_split_roles", corpus_native_split_roles
        )
        object.__setattr__(
            self, "shared_split_disjoint_fields", shared_split_disjoint_fields
        )

    def label_for_corpus(self, corpus: str) -> int:
        try:
            return self.corpus_labels[corpus]
        except KeyError as exc:
            raise ValueError(f"Unknown corpus: {corpus}") from exc

    def rows_per_language(self) -> int:
        return sum(self.per_class_targets.values()) * 2


ENGLISH_XAI_PROFILE = XaiLanguageProfile(
    language_code="eng",
    language_slug="english",
    per_class_targets=dict(XAI_PER_CLASS_TARGETS),
    corpus_labels={"ASVspoof2024": 0, "JMDS": 1},
    corpus_native_split_roles={
        "ASVspoof2024": {
            "train": ("train",),
            "dev": ("calibration", "test"),
        },
        "JMDS": {
            "train": ("train",),
            "dev": ("calibration", "test"),
        },
    },
    shared_split_disjoint_fields={
        "ASVspoof2024": {"dev": ("speaker_id", "group_id")},
        "JMDS": {"dev": ("speaker_id", "group_id")},
    },
    paired_selection=True,
)

PORTUGUESE_XAI_PROFILE = XaiLanguageProfile(
    language_code="por",
    language_slug="portuguese",
    per_class_targets=dict(XAI_PER_CLASS_TARGETS),
    corpus_labels={"CORAA": 0, "MLAAD": 1},
    corpus_native_split_roles={
        "CORAA": {
            "train": ("train",),
            "dev": ("calibration",),
            "test": ("test",),
        },
        "MLAAD": {
            "train": ("train",),
            "dev": ("calibration",),
            "eval": ("test",),
        },
    },
    shared_split_disjoint_fields={},
    paired_selection=False,
)

MANDARIN_XAI_PROFILE = XaiLanguageProfile(
    language_code="zho",
    language_slug="mandarin",
    per_class_targets=dict(XAI_PER_CLASS_TARGETS),
    corpus_labels={"AISHELL-3": 0, "ADD": 1},
    corpus_native_split_roles={
        "AISHELL-3": {
            "train": ("train", "calibration"),
            "test": ("test",),
        },
        "ADD": {
            "train": ("train",),
            "dev": ("calibration",),
            "eval": ("test",),
        },
    },
    shared_split_disjoint_fields={
        "AISHELL-3": {"train": ("speaker_id",)},
    },
    paired_selection=False,
)

XAI_PROFILE_BY_LANGUAGE = MappingProxyType(
    {
        "eng": ENGLISH_XAI_PROFILE,
        "por": PORTUGUESE_XAI_PROFILE,
        "zho": MANDARIN_XAI_PROFILE,
    }
)
