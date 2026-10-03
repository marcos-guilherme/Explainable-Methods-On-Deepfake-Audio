"""Official Zenodo record metadata for the ASVspoof5 archives.

Downloading the archives belongs to :mod:`jmds_prepare.storage.download`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from ..profiles.english import ENGLISH_PROFILE


REQUIRED_ARCHIVES = ENGLISH_PROFILE.required_archives


@dataclass(frozen=True)
class RecordFile:
    name: str
    size: int
    checksum: str
    download_url: str


@dataclass(frozen=True)
class RecordMetadata:
    record_id: int
    doi: str | None
    license: str | None
    version: str | None
    publication_date: str | None
    official_url: str | None
    files: dict[str, RecordFile]
    missing_reasons: dict[str, str]


def get_record_files(
    record_id: int, *, session: Any = None
) -> dict[str, RecordFile]:
    return get_record_metadata(record_id, session=session).files


def get_record_metadata(
    record_id: int, *, session: Any = None
) -> RecordMetadata:
    client = session or requests
    response = client.get(
        f"https://zenodo.org/api/records/{record_id}",
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    available = {item["key"]: item for item in payload["files"]}
    missing = [name for name in REQUIRED_ARCHIVES if name not in available]
    if missing:
        raise ValueError(
            "Zenodo record is missing required archive(s): " + ", ".join(missing)
        )

    files = {}
    for name in REQUIRED_ARCHIVES:
        metadata = available[name]
        checksum = metadata["checksum"]
        if not checksum.startswith("md5:"):
            raise ValueError(f"Unsupported checksum for {name}: {checksum}")
        files[name] = RecordFile(
            name=name,
            size=int(metadata["size"]),
            checksum=checksum.removeprefix("md5:"),
            download_url=metadata["links"]["self"],
        )
    metadata = payload.get("metadata") or {}
    license_value = metadata.get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("id") or license_value.get("title")
    values = {
        "doi": payload.get("doi"),
        "license": license_value,
        "version": metadata.get("version"),
        "publication_date": metadata.get("publication_date"),
        "official_url": (payload.get("links") or {}).get("html"),
    }
    missing_reasons = {
        name: "field absent from Zenodo record API response"
        for name, value in values.items()
        if value is None
    }
    return RecordMetadata(
        record_id=int(payload.get("id", record_id)),
        files=files,
        missing_reasons=missing_reasons,
        **values,
    )
