"""CLI for portable XAI VM bundle construction and verification."""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path

import zstandard

from jmds_prepare.pipelines.xai_bundle import (
    archive_xai_bundle,
    build_xai_bundle,
    verify_xai_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            result = build_xai_bundle(
                {
                    "eng": arguments.eng_manifest,
                    "por": arguments.por_manifest,
                    "zho": arguments.zho_manifest,
                },
                arguments.output,
                allow_noncanonical_counts=arguments.allow_noncanonical_counts,
                dry_run=arguments.dry_run,
                archive=arguments.archive,
            )
            prefix = "dry-run: " if result.dry_run else "built: "
            print(
                f"{prefix}{result.file_count} files, "
                f"{result.total_bytes} bytes -> {result.output}"
            )
        elif arguments.command == "archive":
            archive, sidecar = archive_xai_bundle(
                arguments.bundle,
                allow_noncanonical_counts=arguments.allow_noncanonical_counts,
            )
            print(f"archived: {arguments.bundle} -> {archive}, {sidecar}")
        else:
            result = verify_xai_bundle(
                arguments.bundle,
                allow_noncanonical_counts=arguments.allow_noncanonical_counts,
            )
            print(
                f"verified: {result.file_count} files, "
                f"{result.total_bytes} bytes in {result.output}"
            )
    except (OSError, ValueError, tarfile.TarError, zstandard.ZstdError) as exc:
        message = " ".join(str(exc).splitlines())
        print(f"error: {message}", file=sys.stdout)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jmds_prepare.xai_bundle",
        description="Build, archive, or verify a portable trilingual XAI data bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build a bundle directory.")
    build.add_argument("--eng-manifest", required=True, type=Path)
    build.add_argument("--por-manifest", required=True, type=Path)
    build.add_argument("--zho-manifest", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--allow-noncanonical-counts", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.add_argument(
        "--archive",
        action="store_true",
        help="Stream a sibling .tar.zst archive and SHA-256 sidecar.",
    )

    archive = commands.add_parser(
        "archive",
        help="Verify and archive an existing bundle directory.",
    )
    archive.add_argument("bundle", type=Path)
    archive.add_argument("--allow-noncanonical-counts", action="store_true")

    verify = commands.add_parser("verify", help="Verify a bundle directory.")
    verify.add_argument("bundle", type=Path)
    verify.add_argument("--allow-noncanonical-counts", action="store_true")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
