"""Dependency-injected orchestration for two-source metadata extraction."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..metadata_config import PortugueseMetadataConfig
from ..profiles.portuguese import PortugueseProfile
from ..storage.metadata_layout import PortugueseMetadataLayout
from ..storage.two_source_metadata_layout import TwoSourceMetadataLayout
from .common import Output


@dataclass(frozen=True)
class TwoSourceMetadataExtractionDeps:
    layout: TwoSourceMetadataLayout
    collect_pristine: Callable[[Any], pd.DataFrame]
    collect_generated: Callable[[Any], pd.DataFrame]
    validate_pristine_counts: Callable[[pd.DataFrame], None] | None
    validate_generated_counts: Callable[[pd.DataFrame], None]
    build_pristine_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_generated_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_summary: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
    build_provenance: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    provenance_kwargs: Mapping[str, Any]


@dataclass(frozen=True)
class MetadataExtractionDeps:
    profile: PortugueseProfile
    read_coraa: Callable[..., pd.DataFrame]
    read_jmds_generated: Callable[..., pd.DataFrame]
    build_coraa_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_jmds_profile: Callable[[pd.DataFrame], dict[str, Any]]
    build_summary: Callable[[pd.DataFrame, pd.DataFrame], dict[str, Any]]
    build_provenance: Callable[..., dict[str, Any]]
    publish_artifacts: Callable[[Mapping[Path, bytes]], None]
    sha256_file: Callable[[Path], str]


def extract_two_source_metadata(
    config: Any,
    deps: TwoSourceMetadataExtractionDeps,
    *,
    output: Output = print,
) -> None:
    pristine = deps.collect_pristine(config)
    if deps.validate_pristine_counts is not None:
        deps.validate_pristine_counts(pristine)

    generated = deps.collect_generated(config)
    deps.validate_generated_counts(generated)

    output(f"pristine: {len(pristine)} rows")
    output(f"generated: {len(generated)} rows")

    pristine_profile = deps.build_pristine_profile(pristine)
    generated_profile = deps.build_generated_profile(generated)
    summary = deps.build_summary(pristine, generated)
    provenance = deps.build_provenance(**deps.provenance_kwargs)

    layout = deps.layout
    artifacts = {
        layout.pristine_metadata_csv: _serialize_csv(pristine),
        layout.generated_metadata_csv: _serialize_csv(generated),
        layout.pristine_metadata_profile_json: _serialize_json(pristine_profile),
        layout.generated_metadata_profile_json: _serialize_json(generated_profile),
        layout.summary_json: _serialize_json(summary),
        layout.provenance_json: _serialize_json(provenance),
    }
    deps.publish_artifacts(artifacts)
    output(f"published {len(artifacts)} artifacts")


def extract_portuguese_metadata(
    config: PortugueseMetadataConfig,
    deps: MetadataExtractionDeps,
    *,
    output: Output = print,
) -> None:
    profile = deps.profile
    layout = PortugueseMetadataLayout(config.output_root)
    coraa_input_paths: dict[str, Path] = {}
    jmds_input_paths: dict[str, Path] = {}

    def collect_pristine(cfg: PortugueseMetadataConfig) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split in profile.coraa_splits:
            metadata_path = profile.coraa_metadata_path(cfg.coraa_root, split)
            frames.append(
                deps.read_coraa(metadata_path, split=split, profile=profile)
            )
            coraa_input_paths[f"coraa_{split}"] = metadata_path
        return pd.concat(frames, ignore_index=True)

    def collect_generated(cfg: PortugueseMetadataConfig) -> pd.DataFrame:
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

    def validate_pristine_counts(frame: pd.DataFrame) -> None:
        _validate_split_counts(
            frame,
            split_column="split",
            expected_counts=profile.expected_coraa_counts,
            source="CORAA",
        )

    def validate_generated_counts(frame: pd.DataFrame) -> None:
        _validate_split_counts(
            frame,
            split_column="split",
            expected_counts=profile.expected_generated_counts,
            source="JMDS/MLAAD",
        )

    def build_provenance(**kwargs: Any) -> dict[str, Any]:
        return deps.build_provenance(
            profile=profile,
            input_paths={**coraa_input_paths, **jmds_input_paths},
            sha256_file=deps.sha256_file,
            **kwargs,
        )

    def portuguese_output(message: str) -> None:
        if message.startswith("pristine: "):
            output(f"CORAA: {message.removeprefix('pristine: ')}")
        elif message.startswith("generated: "):
            output(f"JMDS/MLAAD: {message.removeprefix('generated: ')}")
        else:
            output(message)

    extract_two_source_metadata(
        config,
        TwoSourceMetadataExtractionDeps(
            layout=layout,
            collect_pristine=collect_pristine,
            collect_generated=collect_generated,
            validate_pristine_counts=validate_pristine_counts,
            validate_generated_counts=validate_generated_counts,
            build_pristine_profile=deps.build_coraa_profile,
            build_generated_profile=deps.build_jmds_profile,
            build_summary=deps.build_summary,
            build_provenance=build_provenance,
            publish_artifacts=deps.publish_artifacts,
            provenance_kwargs={},
        ),
        output=portuguese_output,
    )


def _validate_split_counts(
    frame: pd.DataFrame,
    *,
    split_column: str,
    expected_counts: Mapping[str, int],
    source: str,
) -> None:
    for split, expected in expected_counts.items():
        actual = int((frame[split_column] == split).sum())
        if actual != expected:
            raise ValueError(
                f"Expected {expected} {source} rows for {split}, got {actual}"
            )


def _serialize_csv(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _serialize_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
