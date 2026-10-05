from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ..config import PreparationConfig
from ..sources.asvspoof import CorrespondenceReport
from ..storage.layout import english_layout
from .common import write_json_atomic


def write_provenance(
    config: PreparationConfig,
    record: Any,
    reports: dict[str, CorrespondenceReport],
) -> None:
    reports_dir = english_layout(config.data_root).reports_dir
    correspondence_path = reports_dir / "correspondence.json"
    if not correspondence_path.is_file():
        return
    with correspondence_path.open(encoding="utf-8") as source:
        correspondence = json.load(source)
    payload = {
        "zenodo": {
            "record_id": record.record_id,
            "doi": record.doi,
            "license": record.license,
            "version": record.version,
            "publication_date": record.publication_date,
            "official_url": record.official_url,
            "missing_reasons": record.missing_reasons,
            "files": [
                {
                    "name": file.name,
                    "size": file.size,
                    "md5": file.checksum,
                    "download_url": file.download_url,
                }
                for file in record.files.values()
            ],
        },
        "protocols": {
            "train": asdict(reports["train"]),
            "dev": asdict(reports["dev"]),
            "eval": correspondence["eval"],
        },
        "acquired_or_validated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "manifest_field_origins": {
            "utt_id": "JMDS and ASVspoof5 correspondence",
            "spk_id": "JMDS and ASVspoof5 correspondence",
            "gender": "JMDS and ASVspoof5 correspondence",
            "language": "JMDS",
            "dataset": "JMDS",
            "split": "protocol filename",
            "label": "JMDS mapped to ASVspoof5 key",
            "attack_id": (
                "JMDS only; ASVspoof5 attack_id is not asserted as "
                "sample-level validated provenance"
            ),
            "protocol_codec": "ASVspoof5 codec; empty if unavailable",
            "source_group_id": "ASVspoof5 tmp field; empty if unavailable",
            "source_path": "calculated",
            "processed_path": "calculated",
            "sha256_source": "calculated from exact source bytes",
            "sha256_processed": "calculated from exact processed bytes",
            "metadata_source": "calculated provenance label",
        },
    }
    write_json_atomic(payload, reports_dir / "provenance.json")
