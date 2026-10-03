from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import PreparationConfig
from ..storage.layout import english_layout
from .common import Output

ProcessManifest = Callable[[pd.DataFrame, Path, Path, int], pd.DataFrame]


def process_english(
    config: PreparationConfig,
    *,
    process_manifest: ProcessManifest,
    output: Output,
) -> None:
    """Process the canonical raw manifest to deterministic WAV files."""
    layout = english_layout(config.data_root)
    raw_path = layout.raw_manifest_path
    destination = layout.processed_manifest_path
    manifest = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    processed = process_manifest(
        manifest,
        config.data_root,
        destination,
        config.sample_rate,
    )
    output(f"processed manifest: {destination} ({len(processed)} rows)")
