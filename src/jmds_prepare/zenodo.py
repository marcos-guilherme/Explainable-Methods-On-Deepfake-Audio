"""Compatibility facade over :mod:`jmds_prepare.sources.zenodo` and downloads."""

from __future__ import annotations

from .sources.zenodo import (
    REQUIRED_ARCHIVES as REQUIRED_ARCHIVES,
    RecordFile as RecordFile,
    RecordMetadata as RecordMetadata,
    get_record_files as get_record_files,
    get_record_metadata as get_record_metadata,
)
from .storage.download import (
    CHUNK_SIZE as CHUNK_SIZE,
    ChecksumError as ChecksumError,
    DownloadResponseError as DownloadResponseError,
    DownloadSizeError as DownloadSizeError,
    _download_mode as _download_mode,
    _download_once as _download_once,
    _invalid_archive_path as _invalid_archive_path,
    _matches_record as _matches_record,
    _md5 as _md5,
    download_verified as download_verified,
)
