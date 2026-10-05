from __future__ import annotations

from typing import Callable

import pandas as pd

from ..config import PreparationConfig
from ..storage.layout import english_layout
from .common import Output

# Signature of ``audit.audit_and_publish``; the legacy ``audit_manifest`` flow
# stays separate and is not orchestrated here.
AuditAndPublish = Callable[..., object]


def audit_english(
    config: PreparationConfig,
    *,
    audit_and_publish: AuditAndPublish,
    output: Output,
) -> None:
    """Audit the processed manifest and publish the technical baseline."""
    layout = english_layout(config.data_root)
    manifest_path = layout.processed_manifest_path
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    reports_dir = layout.reports_dir
    audit_and_publish(
        manifest,
        reports_dir,
        progress=lambda done, total: output(f"audit progress: {done}/{total}"),
    )
    baseline_path = reports_dir / "technical_baseline.json"
    output(f"audit reports: {reports_dir}")
    output(f"technical baseline: {baseline_path}")
