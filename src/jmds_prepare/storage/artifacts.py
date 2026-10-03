"""Catalog-driven acquisition of external source datasets with receipts.

A catalog (strict YAML) names a dataset's official source, licence, pinned
version and the exact artifacts to fetch. Every accepted artifact gets an
atomic JSON receipt with its observed size and SHA-256. A SHA-256 observed
without an official checksum is recorded as observed only; on reruns the
receipt pins it, so divergent bytes are never silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from ..core.atomic import write_json_atomic
from ..core.hashing import sha256_file
from ..core.paths import existing_ancestor
from .download import CHUNK_SIZE, ORIGINS, STRICT_TRANSPORT, fetch_verified

CATALOG_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_SUFFIX = ".receipt.json"
LOCK_NAME = ".source-acquisition.lock"
DIGEST_LENGTHS = {"md5": 32, "sha256": 64}

_DATASET_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_FORBIDDEN_CHARACTERS = set('<>:"/\\|?*')
_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class CatalogError(ValueError):
    """Raised when a source catalog is malformed or unsafe."""


class ReceiptMismatchError(ValueError):
    """Raised when an existing receipt disagrees with the catalog or itself."""


class InsufficientSpaceError(RuntimeError):
    """Raised before any download when free space cannot hold the remainder."""


class AcquisitionLockedError(RuntimeError):
    """Raised when another acquisition already holds the destination lock."""


@dataclass(frozen=True)
class ChecksumSpec:
    algorithm: str
    value: str | None
    source: str | None
    note: str | None


@dataclass(frozen=True)
class LicenseSpec:
    id: str
    url: str
    source: str


@dataclass(frozen=True)
class UsagePolicy:
    preserve_originals: bool
    commercial_use_permitted: bool
    adapted_material_sharing_permitted: bool
    statement: str


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    remote_path: str
    url: str
    size: int
    checksum: ChecksumSpec

    @property
    def download_url(self) -> str:
        return self.url


@dataclass(frozen=True)
class DatasetCatalog:
    schema_version: int
    id: str
    name: str
    version: str
    revision: str | None
    official_page: str
    distribution_page: str | None
    license: LicenseSpec
    usage_policy: UsagePolicy
    artifacts: tuple[ArtifactSpec, ...]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class ArtifactStatus:
    artifact: ArtifactSpec
    state: str
    remaining_bytes: int


@dataclass(frozen=True)
class AcquisitionPlan:
    catalog: DatasetCatalog
    destination: Path
    artifacts: tuple[ArtifactStatus, ...]
    remaining_bytes: int
    minimum_free_bytes: int
    free_bytes: int

    @property
    def required_free_bytes(self) -> int:
        return self.remaining_bytes + self.minimum_free_bytes

    @property
    def sufficient_space(self) -> bool:
        return self.free_bytes >= self.required_free_bytes


# --- catalog loading -----------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CatalogError(f"Duplicate key {key!r} at {key_node.start_mark}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_catalog(path: Path | str) -> DatasetCatalog:
    source_path = Path(path)
    raw = source_path.read_bytes()
    try:
        document = yaml.load(raw.decode("utf-8"), Loader=_StrictLoader)
    except yaml.YAMLError as error:
        raise CatalogError(f"Invalid catalog YAML in {source_path}: {error}") from error

    root = _exact_mapping(document, "catalog", {"schema_version", "dataset", "artifacts"})
    if type(root["schema_version"]) is not int or (
        root["schema_version"] != CATALOG_SCHEMA_VERSION
    ):
        raise CatalogError(f"catalog.schema_version must be {CATALOG_SCHEMA_VERSION}")

    dataset = _exact_mapping(
        root["dataset"],
        "dataset",
        {
            "id", "name", "version", "revision", "official_page",
            "distribution_page", "license", "usage_policy",
        },
    )
    dataset_id = _text(dataset["id"], "dataset.id")
    if not _DATASET_ID.fullmatch(dataset_id):
        raise CatalogError(
            f"dataset.id must match {_DATASET_ID.pattern!r}: {dataset_id!r}"
        )
    revision = _optional(dataset["revision"], "dataset.revision", _text)
    if revision is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", revision):
        raise CatalogError(f"dataset.revision is not a plain identifier: {revision!r}")

    licence = _exact_mapping(dataset["license"], "dataset.license", {"id", "url", "source"})
    policy = _exact_mapping(
        dataset["usage_policy"],
        "dataset.usage_policy",
        {
            "preserve_originals", "commercial_use_permitted",
            "adapted_material_sharing_permitted", "statement",
        },
    )

    entries = root["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise CatalogError("artifacts must be a non-empty list")
    artifacts = tuple(
        _artifact(entry, f"artifacts[{index}]", revision)
        for index, entry in enumerate(entries)
    )
    seen: set[str] = set()
    for artifact in artifacts:
        key = artifact.name.casefold()
        if key in seen:
            raise CatalogError(f"Duplicate local artifact name: {artifact.name!r}")
        seen.add(key)

    return DatasetCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        id=dataset_id,
        name=_text(dataset["name"], "dataset.name"),
        version=_text(dataset["version"], "dataset.version"),
        revision=revision,
        official_page=_https(dataset["official_page"], "dataset.official_page"),
        distribution_page=_optional(
            dataset["distribution_page"], "dataset.distribution_page", _https
        ),
        license=LicenseSpec(
            id=_text(licence["id"], "dataset.license.id"),
            url=_https(licence["url"], "dataset.license.url"),
            source=_https(licence["source"], "dataset.license.source"),
        ),
        usage_policy=UsagePolicy(
            preserve_originals=_flag(policy, "preserve_originals"),
            commercial_use_permitted=_flag(policy, "commercial_use_permitted"),
            adapted_material_sharing_permitted=_flag(
                policy, "adapted_material_sharing_permitted"
            ),
            statement=_text(policy["statement"], "dataset.usage_policy.statement"),
        ),
        artifacts=artifacts,
        source_path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _artifact(entry: Any, where: str, revision: str | None) -> ArtifactSpec:
    fields = _exact_mapping(entry, where, {"name", "remote_path", "url", "size", "checksum"})
    name = _text(fields["name"], f"{where}.name")
    _require_safe_filename(name, f"{where}.name")
    if name.endswith((".partial", RECEIPT_SUFFIX)) or ".invalid" in name:
        raise CatalogError(
            f"{where}.name collides with acquisition bookkeeping files: {name!r}"
        )
    remote_path = _text(fields["remote_path"], f"{where}.remote_path")
    for segment in remote_path.split("/"):
        _require_safe_filename(segment, f"{where}.remote_path")
    url = _https(fields["url"], f"{where}.url")
    if revision is not None and f"/{revision}/" not in urlparse(url).path:
        raise CatalogError(f"{where}.url must contain the pinned revision {revision}")
    if not unquote(urlparse(url).path).endswith(f"/{remote_path}"):
        raise CatalogError(f"{where}.remote_path {remote_path!r} does not end the URL path")
    size = fields["size"]
    if type(size) is not int or size <= 0:
        raise CatalogError(f"{where}.size must be a positive integer")
    return ArtifactSpec(
        name=name,
        remote_path=remote_path,
        url=url,
        size=size,
        checksum=_checksum(fields["checksum"], f"{where}.checksum"),
    )


def _checksum(value: Any, where: str) -> ChecksumSpec:
    fields = _exact_mapping(value, where, {"algorithm", "value", "source", "note"})
    algorithm = fields["algorithm"]
    if algorithm not in DIGEST_LENGTHS:
        raise CatalogError(f"{where}.algorithm must be one of {sorted(DIGEST_LENGTHS)}")
    digest = _optional(fields["value"], f"{where}.value", _text)
    source = _optional(fields["source"], f"{where}.source", _text)
    note = _optional(fields["note"], f"{where}.note", _text)
    if digest is None:
        if source is not None:
            raise CatalogError(f"{where}.source must be null without a checksum value")
        if note is None:
            raise CatalogError(
                f"{where}.note must explain why no official checksum is given"
            )
    else:
        length = DIGEST_LENGTHS[algorithm]
        if not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", digest):
            raise CatalogError(
                f"{where}.value must be a {length}-character hexadecimal "
                f"{algorithm} digest"
            )
        digest = digest.lower()
        if source is None:
            raise CatalogError(f"{where}.source is required with a checksum value")
    return ChecksumSpec(algorithm=algorithm, value=digest, source=source, note=note)


def _exact_mapping(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{where} must be a mapping")
    missing = keys - value.keys()
    unexpected = value.keys() - keys
    if missing:
        raise CatalogError(f"{where}: missing keys {sorted(missing)}")
    if unexpected:
        raise CatalogError(f"{where}: unexpected keys {sorted(map(str, unexpected))}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{where} must be a non-empty string")
    return value


def _https(value: Any, where: str) -> str:
    text = _text(value, where)
    parsed = urlparse(text)
    if parsed.scheme != "https" or not parsed.hostname:
        raise CatalogError(f"{where} must be an https URL: {text!r}")
    if parsed.username is not None or parsed.password is not None:
        raise CatalogError(f"{where} must not contain credentials")
    return text


def _optional(value: Any, where: str, check: Callable[[Any, str], str]) -> str | None:
    return None if value is None else check(value, where)


def _flag(policy: dict[str, Any], key: str) -> bool:
    value = policy[key]
    if not isinstance(value, bool):
        raise CatalogError(f"dataset.usage_policy.{key} must be a boolean")
    return value


def _require_safe_filename(name: str, where: str) -> None:
    unsafe = (
        name in {"", ".", ".."}
        or len(name) > 200
        or name != name.strip()
        or name.endswith(".")
        or any(ord(character) < 32 for character in name)
        or bool(_FORBIDDEN_CHARACTERS & set(name))
        or name.split(".", 1)[0].upper() in _RESERVED_STEMS
    )
    if unsafe:
        raise CatalogError(f"{where} contains an unsafe path segment: {name!r}")


# --- receipts ------------------------------------------------------------------


def receipt_path(destination: Path, artifact: ArtifactSpec) -> Path:
    return destination / f"{artifact.name}{RECEIPT_SUFFIX}"


def build_receipt(
    catalog: DatasetCatalog,
    artifact: ArtifactSpec,
    *,
    observed_size: int,
    observed_sha256: str,
    origin: str,
    verified_at: datetime,
) -> dict[str, Any]:
    """Receipt for bytes already verified against the catalog by the caller."""
    if origin not in ORIGINS:
        raise ValueError(f"Unknown receipt origin: {origin!r}")
    checksum = artifact.checksum
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "catalog": {
            "file_name": catalog.source_path.name,
            "sha256": catalog.sha256,
            "schema_version": catalog.schema_version,
            "dataset_id": catalog.id,
            "dataset_name": catalog.name,
            "version": catalog.version,
            "revision": catalog.revision,
        },
        "official_page": catalog.official_page,
        "distribution_page": catalog.distribution_page,
        "license": asdict(catalog.license),
        "usage_policy": asdict(catalog.usage_policy),
        "artifact": {
            "name": artifact.name,
            "remote_path": artifact.remote_path,
            "url": artifact.url,
        },
        "expected": {
            "size_bytes": artifact.size,
            "checksum_algorithm": checksum.algorithm,
            "checksum": checksum.value,
            "checksum_source": checksum.source,
            "checksum_note": checksum.note,
        },
        "observed": {"size_bytes": observed_size, "sha256": observed_sha256},
        "origin": origin,
        "official_checksum_verified": checksum.value is not None,
        "verified_at_utc": verified_at.astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def _read_receipt(
    path: Path, catalog: DatasetCatalog, artifact: ArtifactSpec
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        identity = {
            "catalog.dataset_id": receipt["catalog"]["dataset_id"],
            "catalog.version": receipt["catalog"]["version"],
            "catalog.revision": receipt["catalog"]["revision"],
            "artifact.name": receipt["artifact"]["name"],
            "artifact.remote_path": receipt["artifact"]["remote_path"],
            "artifact.url": receipt["artifact"]["url"],
            "expected.size_bytes": receipt["expected"]["size_bytes"],
            "expected.checksum_algorithm": receipt["expected"]["checksum_algorithm"],
            "expected.checksum": receipt["expected"]["checksum"],
            "observed.size_bytes": receipt["observed"]["size_bytes"],
            "license": receipt["license"],
            "usage_policy": receipt["usage_policy"],
            "origin": receipt["origin"] in ORIGINS,
        }
        observed_sha256 = receipt["observed"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReceiptMismatchError(f"Unreadable receipt {path}: {error}") from error

    expected = {
        "catalog.dataset_id": catalog.id,
        "catalog.version": catalog.version,
        "catalog.revision": catalog.revision,
        "artifact.name": artifact.name,
        "artifact.remote_path": artifact.remote_path,
        "artifact.url": artifact.url,
        "expected.size_bytes": artifact.size,
        "expected.checksum_algorithm": artifact.checksum.algorithm,
        "expected.checksum": artifact.checksum.value,
        "observed.size_bytes": artifact.size,
        "license": asdict(catalog.license),
        "usage_policy": asdict(catalog.usage_policy),
        "origin": True,
    }
    differing = sorted(key for key in expected if identity[key] != expected[key])
    well_formed = isinstance(observed_sha256, str) and bool(
        re.fullmatch(r"[0-9a-f]{64}", observed_sha256)
    )
    contradicts_official = (
        artifact.checksum.algorithm == "sha256"
        and artifact.checksum.value is not None
        and observed_sha256 != artifact.checksum.value
    )
    if not well_formed or contradicts_official:
        differing.append("observed.sha256")
    if differing:
        raise ReceiptMismatchError(
            f"Existing receipt {path} disagrees with catalog {catalog.id} on "
            f"{', '.join(differing)}; refusing to overwrite it"
        )
    return receipt


# --- acquisition ---------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def acquire_artifact(
    catalog: DatasetCatalog,
    artifact: ArtifactSpec,
    destination: Path,
    *,
    session: Any = None,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    now: Callable[[], datetime] = _utc_now,
    before_transfer: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Fetch or reuse one artifact and return its (possibly existing) receipt.

    Without a receipt, an existing exact-size final whose catalog checksum is
    null is adopted with ``origin: adopted_existing_final`` and its observed
    SHA-256; it is never presented as officially verified.
    """
    path = receipt_path(destination, artifact)
    existing = _read_receipt(path, catalog, artifact) if path.exists() else None
    if artifact.checksum.value is not None:
        algorithm, checksum = artifact.checksum.algorithm, artifact.checksum.value
    elif existing is not None:
        algorithm, checksum = "sha256", existing["observed"]["sha256"]
    else:
        algorithm, checksum = "sha256", None

    result = fetch_verified(
        artifact,
        destination,
        checksum=checksum,
        algorithm=algorithm,
        session=session,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        transport=STRICT_TRANSPORT,
        before_transfer=before_transfer,
    )
    if existing is not None:
        return existing

    observed_sha256 = (
        checksum
        if algorithm == "sha256" and checksum is not None
        else sha256_file(result.path, chunk_size=CHUNK_SIZE)
    )
    receipt = build_receipt(
        catalog,
        artifact,
        observed_size=result.path.stat().st_size,
        observed_sha256=observed_sha256,
        origin=result.origin,
        verified_at=now(),
    )
    write_json_atomic(receipt, path)
    return receipt


def artifact_status(destination: Path, artifact: ArtifactSpec) -> ArtifactStatus:
    """Read-only, conservative state: bytes this run may still have to write.

    Only an exact-size final with a receipt counts as done. Anything whose
    checksum is unverified may be quarantined and fetched again in full.
    """
    final_path = destination / artifact.name
    partial_path = destination / f"{artifact.name}.partial"
    if final_path.is_file() and final_path.stat().st_size == artifact.size:
        if receipt_path(destination, artifact).is_file():
            return ArtifactStatus(artifact, "final with receipt (verified on run)", 0)
        return ArtifactStatus(
            artifact, "final without receipt (verified on run)", artifact.size
        )
    if final_path.exists():
        return ArtifactStatus(artifact, "invalid final (quarantined on run)", artifact.size)
    if partial_path.is_file():
        partial_size = partial_path.stat().st_size
        if partial_size < artifact.size:
            return ArtifactStatus(
                artifact, f"partial {partial_size} bytes", artifact.size - partial_size
            )
        if partial_size == artifact.size:
            return ArtifactStatus(
                artifact, "complete partial (verified on run)", artifact.size
            )
        return ArtifactStatus(
            artifact, "oversized partial (quarantined on run)", artifact.size
        )
    return ArtifactStatus(artifact, "missing", artifact.size)


def plan_acquisition(
    catalog: DatasetCatalog,
    destination: Path,
    *,
    minimum_free_bytes: int,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> AcquisitionPlan:
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes must not be negative")
    statuses = tuple(artifact_status(destination, artifact) for artifact in catalog.artifacts)
    return AcquisitionPlan(
        catalog=catalog,
        destination=destination,
        artifacts=statuses,
        remaining_bytes=sum(status.remaining_bytes for status in statuses),
        minimum_free_bytes=minimum_free_bytes,
        free_bytes=int(disk_usage(existing_ancestor(destination)).free),
    )


def acquire_catalog(
    catalog: DatasetCatalog,
    destination: Path,
    *,
    minimum_free_bytes: int,
    session: Any = None,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    now: Callable[[], datetime] = _utc_now,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Acquire every artifact sequentially under an exclusive destination lock.

    Free space is checked for the whole catalog before anything is created,
    then again before each transfer against that transfer plus the
    conservative remainder of later artifacts. The first error stops the run.
    """
    plan = plan_acquisition(
        catalog,
        destination,
        minimum_free_bytes=minimum_free_bytes,
        disk_usage=disk_usage,
    )
    if not plan.sufficient_space:
        raise InsufficientSpaceError(
            f"Refusing to download {catalog.id}: free {plan.free_bytes} bytes < "
            f"required {plan.required_free_bytes} bytes (remaining "
            f"{plan.remaining_bytes} + reserve {plan.minimum_free_bytes})"
        )

    receipts = []
    with destination_lock(destination):
        for index, artifact in enumerate(catalog.artifacts):
            later = catalog.artifacts[index + 1:]

            def guard(needed: int, artifact: ArtifactSpec = artifact,
                      later: tuple[ArtifactSpec, ...] = later) -> None:
                remainder = sum(
                    artifact_status(destination, item).remaining_bytes for item in later
                )
                required = needed + remainder + minimum_free_bytes
                free = int(disk_usage(existing_ancestor(destination)).free)
                if free < required:
                    raise InsufficientSpaceError(
                        f"Refusing to download {artifact.name}: free {free} bytes < "
                        f"required {required} bytes (this artifact {needed} + later "
                        f"{remainder} + reserve {minimum_free_bytes})"
                    )

            if progress is not None:
                progress(f"acquiring {artifact.name} ({artifact.size} bytes)")
            receipts.append(
                acquire_artifact(
                    catalog,
                    artifact,
                    destination,
                    session=session,
                    max_attempts=max_attempts,
                    retry_delay=retry_delay,
                    now=now,
                    before_transfer=guard,
                )
            )
            if progress is not None:
                receipt = receipts[-1]
                progress(
                    f"verified {artifact.name} origin={receipt['origin']} "
                    f"sha256={receipt['observed']['sha256']}"
                )
    return receipts


@contextmanager
def destination_lock(destination: Path) -> Iterator[Path]:
    """Hold an exclusive lock file in ``destination`` for one acquisition."""
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise AcquisitionLockedError(
            f"Destination lock {lock_path} exists: another acquisition may be "
            "running. If none is, delete the lock file and retry."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)
