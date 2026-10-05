"""Download external source datasets described by a catalog, with receipts.

Separate from the English JMDS commands in :mod:`jmds_prepare.cli`::

    python -m jmds_prepare.source_acquisition --catalog configs/sources/aishell3.yaml \
        --destination E:/sources/aishell3 --dry-run
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import requests

from .storage.artifacts import (
    AcquisitionLockedError,
    AcquisitionPlan,
    CatalogError,
    DatasetCatalog,
    InsufficientSpaceError,
    ReceiptMismatchError,
    acquire_catalog,
    load_catalog,
    plan_acquisition,
)
from .storage.download import ChecksumError, DownloadResponseError, DownloadSizeError

BYTES_PER_GB = 10**9
DEFAULT_MINIMUM_FREE_GB = 10.0

# requests' RequestException is an OSError; the explicit ValueError subclasses
# are listed for readers even though ValueError already covers them.
EXPECTED_ERRORS = (
    CatalogError,
    ReceiptMismatchError,
    ChecksumError,
    DownloadSizeError,
    DownloadResponseError,
    InsufficientSpaceError,
    AcquisitionLockedError,
    OSError,
    ValueError,
)


def _non_negative_gb(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number of GB") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number of GB")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jmds_prepare.source_acquisition",
        description="Verifiable, resumable download of one source catalog.",
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--minimum-free-gb",
        type=_non_negative_gb,
        default=DEFAULT_MINIMUM_FREE_GB,
        help="free space (10^9 bytes) that must remain after all downloads",
    )
    return parser


def describe_plan(plan: AcquisitionPlan) -> list[str]:
    catalog = plan.catalog
    policy = catalog.usage_policy
    lines = [
        f"dataset: {catalog.id} ({catalog.name})",
        f"version: {catalog.version}",
        f"revision: {catalog.revision or 'none'}",
        f"official_page: {catalog.official_page}",
        f"distribution_page: {catalog.distribution_page or 'none'}",
        (
            f"license: {catalog.license.id} {catalog.license.url} "
            f"(declared at {catalog.license.source})"
        ),
        f"preserve_originals: {str(policy.preserve_originals).lower()}",
        f"commercial_use_permitted: {str(policy.commercial_use_permitted).lower()}",
        (
            "adapted_material_sharing_permitted: "
            f"{str(policy.adapted_material_sharing_permitted).lower()}"
        ),
        f"usage_policy: {' '.join(policy.statement.split())}",
        f"destination: {plan.destination}",
    ]
    for status in plan.artifacts:
        artifact = status.artifact
        checksum = artifact.checksum
        expected = (
            f"{checksum.algorithm}:{checksum.value} ({checksum.source})"
            if checksum.value is not None
            else f"none ({' '.join((checksum.note or '').split())})"
        )
        lines += [
            f"artifact: {artifact.name}",
            f"  url: {artifact.url}",
            f"  size_bytes: {artifact.size}",
            f"  expected_checksum: {expected}",
            f"  state: {status.state}",
            f"  remaining_bytes: {status.remaining_bytes}",
        ]
    lines += [
        f"remaining_bytes: {plan.remaining_bytes}",
        f"minimum_free_bytes: {plan.minimum_free_bytes}",
        f"required_free_bytes: {plan.required_free_bytes}",
        f"free_bytes: {plan.free_bytes}",
        f"space: {'sufficient' if plan.sufficient_space else 'INSUFFICIENT'}",
    ]
    return lines


def main(
    argv: Sequence[str] | None = None,
    *,
    session: Any = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    output: Callable[[str], None] = print,
) -> int:
    args = _parser().parse_args(argv)
    minimum_free_bytes = int(args.minimum_free_gb * BYTES_PER_GB)
    try:
        return _run(args, minimum_free_bytes, session, disk_usage, output)
    except EXPECTED_ERRORS as error:
        output(f"error: {error}")
        return 1


def _run(
    args: argparse.Namespace,
    minimum_free_bytes: int,
    session: Any,
    disk_usage: Callable[[Path], Any],
    output: Callable[[str], None],
) -> int:
    catalog = load_catalog(args.catalog)
    plan = plan_acquisition(
        catalog,
        args.destination,
        minimum_free_bytes=minimum_free_bytes,
        disk_usage=disk_usage,
    )
    for line in describe_plan(plan):
        output(line)
    if args.dry_run:
        output("mode: dry-run (no network access, nothing written)")
        return 0

    if session is None:
        with requests.Session() as owned_session:
            receipts = _acquire(catalog, args, minimum_free_bytes, owned_session, disk_usage, output)
    else:
        receipts = _acquire(catalog, args, minimum_free_bytes, session, disk_usage, output)
    official = [r["artifact"]["name"] for r in receipts if r["official_checksum_verified"]]
    observed = [r["artifact"]["name"] for r in receipts if not r["official_checksum_verified"]]
    output(f"complete: {len(receipts)} artifacts with receipts")
    output(f"verified by official checksum: {', '.join(official) or 'none'}")
    output(
        "verified by exact size + observed SHA-256 (no official checksum): "
        f"{', '.join(observed) or 'none'}"
    )
    return 0


def _acquire(
    catalog: DatasetCatalog,
    args: argparse.Namespace,
    minimum_free_bytes: int,
    session: Any,
    disk_usage: Callable[[Path], Any],
    output: Callable[[str], None],
) -> list[dict[str, Any]]:
    return acquire_catalog(
        catalog,
        args.destination,
        minimum_free_bytes=minimum_free_bytes,
        session=session,
        disk_usage=disk_usage,
        progress=output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
