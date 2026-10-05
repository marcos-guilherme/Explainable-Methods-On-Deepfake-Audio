"""Canonical CSV ledger of audio members extracted from source archives."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


LEDGER_COLUMNS = [
    "utt_id",
    "split",
    "archive_name",
    "archive_md5",
    "member_name",
    "member_size",
    "sha256_extracted",
]


@dataclass(frozen=True)
class ExtractionLedgerEntry:
    utt_id: str
    split: str
    archive_name: str
    archive_md5: str
    member_name: str
    member_size: int
    sha256_extracted: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def read_extraction_ledger(path: Path) -> "pd.DataFrame":
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != LEDGER_COLUMNS:
        raise ValueError("Extraction ledger schema is invalid")
    duplicates = frame["utt_id"].duplicated(keep=False)
    if duplicates.any():
        ids = ", ".join(sorted(frame.loc[duplicates, "utt_id"].unique()))
        raise ValueError(f"Duplicate utt_id rows in extraction ledger: {ids}")
    frame["member_size"] = frame["member_size"].astype(int)
    return frame


def update_extraction_ledger(
    path: Path, entries: list[ExtractionLedgerEntry]
) -> None:
    """Atomically upsert canonical ledger rows by utterance ID."""
    import pandas as pd

    path = Path(path)
    frame = read_extraction_ledger(path)
    updates = pd.DataFrame(
        [entry.as_dict() for entry in entries], columns=LEDGER_COLUMNS
    )
    if not updates.empty and updates["utt_id"].duplicated().any():
        raise ValueError("Duplicate utt_id rows in ledger update")
    if not updates.empty:
        frame = frame.loc[~frame["utt_id"].isin(updates["utt_id"])]
        frame = pd.concat([frame, updates], ignore_index=True)
    frame = frame.sort_values(["split", "utt_id"], kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
