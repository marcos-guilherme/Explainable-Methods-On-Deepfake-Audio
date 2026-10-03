"""Standalone CLI for Portuguese CORAA and JMDS/MLAAD metadata extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core.hashing import sha256_file
from .core.publication import publish_artifact_set_idempotent
from .metadata_config import PortugueseMetadataConfig
from .pipelines.metadata_extraction import MetadataExtractionDeps, extract_portuguese_metadata
from .pipelines.metadata_profiling import (
    build_coraa_profile,
    build_jmds_mlaad_profile,
    build_portuguese_summary,
)
from .pipelines.metadata_provenance import build_metadata_provenance
from .profiles.portuguese import PORTUGUESE_PROFILE
from .sources.coraa import read_coraa_metadata
from .sources.jmds_mlaad import read_portuguese_generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="portuguese-metadata",
        description=(
            "Extract and publish Portuguese CORAA and JMDS/MLAAD metadata "
            "without modifying source files."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Portuguese metadata YAML configuration.",
    )
    arguments = parser.parse_args(argv)

    try:
        config = PortugueseMetadataConfig.load(arguments.config)
        config.validate(PORTUGUESE_PROFILE)
        extract_portuguese_metadata(config, _default_deps())
    except (OSError, ValueError) as error:
        message = " ".join(str(error).splitlines())
        print(f"error: {message}", file=sys.stdout)
        return 1
    return 0


def _default_deps() -> MetadataExtractionDeps:
    return MetadataExtractionDeps(
        profile=PORTUGUESE_PROFILE,
        read_coraa=read_coraa_metadata,
        read_jmds_generated=read_portuguese_generated,
        build_coraa_profile=build_coraa_profile,
        build_jmds_profile=build_jmds_mlaad_profile,
        build_summary=build_portuguese_summary,
        build_provenance=build_metadata_provenance,
        publish_artifacts=publish_artifact_set_idempotent,
        sha256_file=sha256_file,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
