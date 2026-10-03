"""Standalone CLI for Mandarin AISHELL-3 and JMDS/ADD metadata extraction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .core.hashing import sha256_file
from .core.publication import publish_artifact_set_idempotent
from .mandarin_metadata_config import MandarinMetadataConfig
from .pipelines.common import Output
from .pipelines.mandarin_metadata_profiling import (
    build_aishell3_profile,
    build_jmds_add_profile,
    build_mandarin_summary,
)
from .pipelines.mandarin_metadata_provenance import build_mandarin_metadata_provenance
from .pipelines.metadata_extraction import (
    TwoSourceMetadataExtractionDeps,
    _validate_split_counts,
    extract_two_source_metadata,
)
from .profiles.mandarin import MANDARIN_PROFILE, MandarinProfile
from .sources.aishell3 import AishellReconciliation, read_aishell3_metadata
from .sources.jmds_add import read_mandarin_generated
from .storage.mandarin_metadata_layout import MandarinMetadataLayout


@dataclass(frozen=True)
class MandarinMetadataExtractionDeps:
    profile: MandarinProfile
    read_aishell3: Callable[..., tuple[pd.DataFrame, AishellReconciliation]]
    read_jmds_generated: Callable[..., pd.DataFrame]
    build_aishell_profile: Callable[..., dict[str, Any]]
    build_jmds_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_summary: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
    build_provenance: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    sha256_file: Callable[[Path], str]


def extract_mandarin_metadata(
    config: MandarinMetadataConfig,
    deps: MandarinMetadataExtractionDeps,
    *,
    output: Output = print,
) -> None:
    profile = deps.profile
    layout = MandarinMetadataLayout(config.output_root)
    reconciliation_holder: dict[str, AishellReconciliation] = {}
    jmds_input_paths: dict[str, Path] = {}

    def collect_pristine(cfg: MandarinMetadataConfig) -> pd.DataFrame:
        frame, reconciliation = deps.read_aishell3(
            cfg.aishell_archive, profile=profile
        )
        reconciliation_holder["value"] = reconciliation
        return frame

    def collect_generated(cfg: MandarinMetadataConfig) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split in profile.protocol_splits:
            protocol_path = profile.jmds_protocol_path(cfg.jmds_root, split)
            frames.append(
                deps.read_jmds_generated(
                    protocol_path,
                    split=split,
                    jmds_root=cfg.jmds_root,
                    profile=profile,
                )
            )
            jmds_input_paths[f"jmds_{split}"] = protocol_path
        return pd.concat(frames, ignore_index=True)

    def validate_generated_counts(frame: pd.DataFrame) -> None:
        _validate_split_counts(
            frame,
            split_column="split",
            expected_counts=profile.expected_generated_counts,
            source="JMDS/ADD",
        )

    def build_pristine_profile(frame: pd.DataFrame) -> dict[str, Any]:
        return deps.build_aishell_profile(
            frame, reconciliation=reconciliation_holder["value"]
        )

    def build_provenance(**kwargs: Any) -> dict[str, Any]:
        return deps.build_provenance(
            profile=profile,
            input_paths={
                "aishell_archive": config.aishell_archive,
                **jmds_input_paths,
            },
            sha256_file=deps.sha256_file,
            reconciliation=reconciliation_holder["value"],
            **kwargs,
        )

    def mandarin_output(message: str) -> None:
        if message.startswith("pristine: "):
            output(f"AISHELL-3: {message.removeprefix('pristine: ')}")
        elif message.startswith("generated: "):
            output(f"JMDS/ADD: {message.removeprefix('generated: ')}")
        else:
            output(message)

    extract_two_source_metadata(
        config,
        TwoSourceMetadataExtractionDeps(
            layout=layout,
            collect_pristine=collect_pristine,
            collect_generated=collect_generated,
            validate_pristine_counts=None,
            validate_generated_counts=validate_generated_counts,
            build_pristine_profile=build_pristine_profile,
            build_generated_profile=deps.build_jmds_profile,
            build_summary=deps.build_summary,
            build_provenance=build_provenance,
            publish_artifacts=deps.publish_artifacts,
            provenance_kwargs={},
        ),
        output=mandarin_output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mandarin-metadata",
        description=(
            "Extract and publish Mandarin AISHELL-3 and JMDS/ADD metadata "
            "without modifying source files."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Mandarin metadata YAML configuration.",
    )
    arguments = parser.parse_args(argv)

    try:
        config = MandarinMetadataConfig.load(arguments.config)
        config.validate(MANDARIN_PROFILE)
        extract_mandarin_metadata(config, _default_deps())
    except (OSError, ValueError) as error:
        message = " ".join(str(error).splitlines())
        print(f"error: {message}", file=sys.stdout)
        return 1
    return 0


def _default_deps() -> MandarinMetadataExtractionDeps:
    return MandarinMetadataExtractionDeps(
        profile=MANDARIN_PROFILE,
        read_aishell3=read_aishell3_metadata,
        read_jmds_generated=read_mandarin_generated,
        build_aishell_profile=build_aishell3_profile,
        build_jmds_profile=build_jmds_add_profile,
        build_summary=build_mandarin_summary,
        build_provenance=build_mandarin_metadata_provenance,
        publish_artifacts=publish_artifact_set_idempotent,
        sha256_file=sha256_file,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
