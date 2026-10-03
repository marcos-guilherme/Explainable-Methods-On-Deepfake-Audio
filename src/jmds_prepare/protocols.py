"""Compatibility facade over :mod:`jmds_prepare.sources.jmds` and ``.asvspoof``."""

from __future__ import annotations

from .sources.asvspoof import (
    ASV_COLUMNS as ASV_COLUMNS,
    CorrespondenceReport as CorrespondenceReport,
    _ASV_LABELS as _ASV_LABELS,
    _EQUIVALENT_LABEL as _EQUIVALENT_LABEL,
    protocol_file_sha256 as protocol_file_sha256,
    protocol_fingerprint as protocol_fingerprint,
    read_asvspoof_protocol as read_asvspoof_protocol,
    validate_correspondence as validate_correspondence,
)
from .sources.jmds import (
    JMDS_COLUMNS as JMDS_COLUMNS,
    _JMDS_LABELS as _JMDS_LABELS,
    _SPLIT_PREFIX as _SPLIT_PREFIX,
    _SUPPORTED_SPLITS as _SUPPORTED_SPLITS,
    _validate_split as _validate_split,
    _validate_unique_ids as _validate_unique_ids,
    _validate_utterance_ids as _validate_utterance_ids,
    read_jmds_protocol as read_jmds_protocol,
)
