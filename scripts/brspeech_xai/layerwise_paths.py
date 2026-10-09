"""Layout seguro e cache de embeddings da suíte layer-wise."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


LANGUAGES = frozenset({"eng", "por", "zho"})
CACHE_SCHEMA_VERSION = 1
RESERVED_ROOT_NAMESPACES = frozenset({"aggregates"})
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SIMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CACHE_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "profile",
        "checkpoint",
        "manifest_sha256",
        "language",
        "role",
        "shape",
        "dtype",
        "sample_ids",
        "sample_ids_sha256",
    }
)


def _simple_id(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or _SIMPLE_ID_RE.fullmatch(value) is None
    ):
        raise ValueError(f"{field} must be a non-empty simple identifier")
    return value


def _profile_id(value: str, *, field: str = "profile_id") -> str:
    value = _simple_id(value, field=field)
    if value.casefold() in RESERVED_ROOT_NAMESPACES:
        raise ValueError(f"{field} uses a reserved root namespace: {value!r}")
    return value


def _language(value: str, *, field: str = "language") -> str:
    if value not in LANGUAGES:
        raise ValueError(f"{field} must be one of {sorted(LANGUAGES)}")
    return value


def _layer(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 12:
        raise ValueError("layer must be an integer from 1 through 12")
    return value


def _manifest_hash(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("manifest_sha256 must be a 64-character hexadecimal digest")
    return value.lower()


@dataclass(frozen=True)
class LayerwiseSuitePaths:
    """Resolve artefatos sem criar diretórios nem normalizar IDs perigosos."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def profile(self, profile_id: str) -> Path:
        return self.root / _profile_id(profile_id)

    def embeddings(self, profile_id: str, language: str, role: str) -> Path:
        return (
            self.profile(profile_id)
            / "embeddings"
            / _language(language)
            / f"{_simple_id(role, field='role')}.npy"
        )

    def embedding_metadata(
        self,
        profile_id: str,
        language: str,
        role: str,
    ) -> Path:
        embedding = self.embeddings(profile_id, language, role)
        return embedding.with_name(f"{embedding.stem}.metadata.json")

    def probe(self, profile_id: str, layer: int, source: str) -> Path:
        return self._probe_dir(profile_id, layer, source) / "d_ad.joblib"

    def thresholds(self, profile_id: str, layer: int, source: str) -> Path:
        return self._probe_dir(profile_id, layer, source) / "thresholds.json"

    def _probe_dir(self, profile_id: str, layer: int, source: str) -> Path:
        return (
            self.profile(profile_id)
            / "layer_probes"
            / _language(source, field="source")
            / f"layer_{_layer(layer):02d}"
        )

    def cell(
        self,
        profile_id: str,
        layer: int,
        source: str,
        target: str,
    ) -> Path:
        return (
            self.profile(profile_id)
            / "cells"
            / f"layer_{_layer(layer):02d}"
            / (
                f"{_language(source, field='source')}"
                f"_to_{_language(target, field='target')}"
            )
        )

    def layer_xai(
        self,
        profile_id: str,
        layer: int,
        source: str,
        target: str,
    ) -> Path:
        return (
            self.profile(profile_id)
            / "layer_xai"
            / f"layer_{_layer(layer):02d}"
            / (
                f"{_language(source, field='source')}"
                f"_to_{_language(target, field='target')}"
            )
        )

    def final_decision_trace(self, profile_id: str) -> Path:
        return self.profile(profile_id) / "final_decision_trace"

    def final_trace(self, profile_id: str) -> Path:
        """Alias conciso para o diretório de rastreamento da decisão final."""
        return self.final_decision_trace(profile_id)

    def final_trace_cell(
        self,
        profile_id: str,
        source: str,
        target: str,
    ) -> Path:
        return (
            self.final_decision_trace(profile_id)
            / f"{_language(source, field='source')}_to_"
            f"{_language(target, field='target')}"
        )

    def suite_cohort(self, target: str) -> Path:
        return (
            self.root
            / ".stage-artifacts"
            / f"suite__cohort__{_language(target, field='target')}"
            / "cohort.parquet"
        )

    def suite_cohort_metadata(self, target: str) -> Path:
        cohort = self.suite_cohort(target)
        return cohort.with_name("cohort.metadata.json")

    def suite_aggregate(self, name: str) -> Path:
        return self.root / "aggregates" / _simple_id(name, field="aggregate name")

    def probe_stability_dir(self, profile_id: str) -> Path:
        return self.profile(profile_id) / "probe_stability"

    def probe_stability_seed_dir(self, profile_id: str, seed: int) -> Path:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        return self.probe_stability_dir(profile_id) / f"seed_{seed:03d}"

    def probe_stability_performance(self, profile_id: str, seed: int) -> Path:
        return (
            self.probe_stability_seed_dir(profile_id, seed)
            / "layerwise_performance.csv"
        )

    def probe_stability_summary(self, profile_id: str) -> Path:
        return self.probe_stability_dir(profile_id) / "probe_stability_by_layer.csv"

    def probe_stability_manifest(self, profile_id: str) -> Path:
        return self.probe_stability_dir(profile_id) / "probe_stability_manifest.json"

    def probe_stability_train_bootstrap(
        self, profile_id: str, seed: int, source: str
    ) -> Path:
        return (
            self.probe_stability_seed_dir(profile_id, seed)
            / f"train_bootstrap_{_language(source, field='source')}.json"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_active_generation(
    destination: str | Path,
    *,
    expected_role: str | None = None,
) -> Path:
    """Resolve e valida a geração imutável apontada por ``active.json``."""
    destination = Path(destination)
    pointer_path = destination / "active.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        generation = pointer["generation"]
        if (
            pointer.get("schema_version") != 1
            or not isinstance(generation, str)
            or not generation
            or any(character not in "0123456789abcdef" for character in generation)
        ):
            raise ValueError("invalid active generation pointer")
        resolved = destination / "generations" / generation
        manifest = json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
        role = manifest.get("role")
        if (
            manifest.get("schema_version") != 2
            or manifest.get("generation") != generation
            or not isinstance(role, str)
            or not role
            or (expected_role is not None and role != expected_role)
        ):
            raise ValueError("active generation manifest mismatch")
        if not resolved.is_dir():
            raise ValueError("active generation directory is missing")
        artifacts = manifest.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or any(
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256", "size_bytes"}
                or not isinstance(artifact.get("path"), str)
                or not artifact["path"]
                or Path(artifact["path"]).is_absolute()
                or ".." in Path(artifact["path"]).parts
                or not isinstance(artifact.get("sha256"), str)
                or _SHA256_RE.fullmatch(artifact["sha256"]) is None
                or isinstance(artifact.get("size_bytes"), bool)
                or not isinstance(artifact["size_bytes"], int)
                or artifact["size_bytes"] < 0
                or not (resolved / artifact["path"]).is_file()
                or (resolved / artifact["path"]).stat().st_size
                != artifact["size_bytes"]
                or _file_sha256(resolved / artifact["path"])
                != artifact["sha256"]
                for artifact in artifacts
            )
        ):
            raise ValueError("active generation has missing or invalid manifest artifacts")
        return resolved
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"no valid active generation at {destination}") from exc


def publish_generation(
    destination: str | Path,
    writer: Callable[[Path], None],
    *,
    role: str = "generic",
) -> None:
    """Publica uma geração imutável e troca o ponteiro ativo atomicamente."""
    destination = Path(destination)
    role = _simple_id(role, field="generation role")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    generations = destination / "generations"
    generations.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    temporary = generations / f".{token}.tmp"
    generation = generations / token
    pointer = destination / "active.json"
    pointer_tmp = destination / f".active.{token}.tmp"
    pointer_published = False
    temporary.mkdir()
    try:
        writer(temporary)
        artifact_paths = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generation": token,
                    "role": role,
                    "artifacts": [
                        {
                            "path": str(path.relative_to(temporary)).replace("\\", "/"),
                            "sha256": _file_sha256(path),
                            "size_bytes": path.stat().st_size,
                        }
                        for path in artifact_paths
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, generation)
        with pointer_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": 1, "generation": token},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pointer_tmp, pointer)
        pointer_published = True
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        pointer_tmp.unlink(missing_ok=True)
        if generation.exists() and not pointer_published:
            shutil.rmtree(generation, ignore_errors=True)


def embedding_cache_key(
    *,
    profile_id: str,
    checkpoint: str,
    manifest_sha256: str,
    language: str,
    role: str,
    seed: int,
    audio_config: Any,
) -> str:
    """Calcula a chave SHA-256 da configuração canônica de extração."""
    profile_id = _profile_id(profile_id)
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint must be a non-empty string")
    language = _language(language)
    role = _simple_id(role, field="role")
    if not is_dataclass(audio_config) or isinstance(audio_config, type):
        raise TypeError("audio_config must be a dataclass instance")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "profile": profile_id,
        "checkpoint": checkpoint,
        "manifest_sha256": _manifest_hash(manifest_sha256),
        "language": language,
        "role": role,
        "seed": seed,
        "audio": asdict(audio_config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


compute_embedding_cache_key = embedding_cache_key


def _validated_sample_ids(sample_ids: Sequence[str]) -> list[str]:
    if isinstance(sample_ids, (str, bytes)):
        raise ValueError("sample_ids must be an ordered sequence of unique strings")
    values = list(sample_ids)
    for sample_id in values:
        _simple_id(sample_id, field="sample_id")
    if len(set(values)) != len(values):
        raise ValueError("sample_ids must be unique")
    return values


def sample_id_catalog_hash(sample_ids: Sequence[str]) -> str:
    """Hash sensível à ordem do catálogo de ``sample_id``."""
    values = _validated_sample_ids(sample_ids)
    encoded = json.dumps(
        values,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PROBE_ALIGNMENT_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "language",
        "role",
        "shape",
        "dtype",
        "sample_ids",
        "sample_ids_sha256",
    }
)


def read_embedding_metadata_json(metadata_path: Path) -> dict[str, Any]:
    """Lê metadata de embedding; falha fechada se ausente ou ilegível."""
    path = Path(metadata_path)
    if not path.is_file():
        raise ValueError(f"embedding metadata is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"embedding metadata is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"embedding metadata must be a JSON object: {path}")
    return raw


def assert_catalog_aligned_with_embedding_metadata(
    *,
    profile_id: str,
    language: str,
    role: str,
    catalog_sample_ids: Sequence[str],
    array: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    """Exige ordem idêntica entre catálogo regenerado e ``sample_ids`` persistidos."""
    location = f"{_profile_id(profile_id)}/{_language(language)}/{_simple_id(role, field='role')}"
    if not isinstance(metadata, Mapping):
        raise ValueError(f"embedding metadata must be a mapping for {location}")
    missing = _PROBE_ALIGNMENT_METADATA_FIELDS - set(metadata)
    if missing:
        raise ValueError(
            f"embedding metadata lacks fields {sorted(missing)} for {location}"
        )
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"embedding metadata schema_version is invalid for {location}"
        )
    if metadata.get("profile") != _profile_id(profile_id):
        raise ValueError(
            f"embedding metadata profile mismatch for {location}: "
            f"expected {profile_id!r}, found {metadata.get('profile')!r}"
        )
    if metadata.get("language") != _language(language):
        raise ValueError(
            f"embedding metadata language mismatch for {location}: "
            f"expected {language!r}, found {metadata.get('language')!r}"
        )
    if metadata.get("role") != _simple_id(role, field="role"):
        raise ValueError(
            f"embedding metadata role mismatch for {location}: "
            f"expected {role!r}, found {metadata.get('role')!r}"
        )
    if metadata.get("dtype") != "float32":
        raise ValueError(
            f"embedding metadata dtype must be float32 for {location}"
        )
    catalog_ids = _validated_sample_ids(catalog_sample_ids)
    try:
        metadata_ids = _validated_sample_ids(metadata["sample_ids"])
    except ValueError as exc:
        raise ValueError(
            f"embedding metadata sample_ids are invalid for {location}"
        ) from exc
    if catalog_ids != metadata_ids:
        raise ValueError(
            f"catalog sample_id order does not match embedding metadata for "
            f"{location}"
        )
    expected_hash = sample_id_catalog_hash(catalog_ids)
    recorded_hash = metadata.get("sample_ids_sha256")
    if not isinstance(recorded_hash, str) or _SHA256_RE.fullmatch(recorded_hash) is None:
        raise ValueError(
            f"embedding metadata sample_ids_sha256 is invalid for {location}"
        )
    if recorded_hash.lower() != expected_hash:
        raise ValueError(
            f"catalog sample_ids_sha256 does not match embedding metadata for "
            f"{location}"
        )
    if metadata.get("sample_ids_sha256") != sample_id_catalog_hash(metadata_ids):
        raise ValueError(
            f"embedding metadata sample_ids_sha256 is inconsistent with "
            f"sample_ids for {location}"
        )
    _validate_embedding_array(array, expected_samples=len(catalog_ids))
    if metadata.get("shape") != list(array.shape):
        raise ValueError(
            f"embedding metadata shape does not match array for {location}: "
            f"metadata {metadata.get('shape')!r} vs array {list(array.shape)!r}"
        )


def _validate_embedding_array(
    array: np.ndarray,
    *,
    expected_samples: int | None = None,
) -> None:
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 3
        or array.shape[1] != 13
        or array.shape[2] <= 0
    ):
        raise ValueError("embeddings must have shape [N, 13, H] with H > 0")
    if expected_samples is not None and array.shape[0] != expected_samples:
        raise ValueError("embedding sample count does not match sample_id catalog")
    if array.dtype != np.float32:
        raise ValueError("embeddings must have dtype float32")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must contain only finite values")


def build_embedding_cache_metadata(
    *,
    cache_key: str,
    profile_id: str,
    checkpoint: str,
    manifest_sha256: str,
    language: str,
    role: str,
    array: np.ndarray,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Cria metadata versionada após validar array e catálogo."""
    ids = _validated_sample_ids(sample_ids)
    _validate_embedding_array(array, expected_samples=len(ids))
    if not isinstance(cache_key, str) or _SHA256_RE.fullmatch(cache_key) is None:
        raise ValueError("cache_key must be a SHA-256 digest")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint must be a non-empty string")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key.lower(),
        "profile": _profile_id(profile_id),
        "checkpoint": checkpoint,
        "manifest_sha256": _manifest_hash(manifest_sha256),
        "language": _language(language),
        "role": _simple_id(role, field="role"),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_ids": ids,
        "sample_ids_sha256": sample_id_catalog_hash(ids),
    }


def load_valid_embedding_cache(
    embedding_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_cache_key: str,
    expected_profile_id: str,
    expected_checkpoint: str,
    expected_manifest_sha256: str,
    expected_language: str,
    expected_role: str,
    expected_sample_ids: Sequence[str],
) -> np.ndarray | None:
    """Retorna o cache apenas quando todas as invariantes são satisfeitas."""
    try:
        if (
            not isinstance(expected_cache_key, str)
            or _SHA256_RE.fullmatch(expected_cache_key) is None
        ):
            return None
        expected_cache_key = expected_cache_key.lower()
        expected_profile_id = _profile_id(
            expected_profile_id,
            field="expected_profile_id",
        )
        expected_language = _language(
            expected_language,
            field="expected_language",
        )
        expected_role = _simple_id(expected_role, field="expected_role")
        ids = _validated_sample_ids(expected_sample_ids)
        metadata_raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if not isinstance(metadata_raw, dict):
            return None
        metadata = metadata_raw
        if (
            set(metadata) != _CACHE_METADATA_FIELDS
            or metadata.get("schema_version") != CACHE_SCHEMA_VERSION
            or metadata.get("cache_key") != expected_cache_key
            or metadata.get("profile") != expected_profile_id
            or metadata.get("checkpoint") != expected_checkpoint
            or metadata.get("manifest_sha256")
            != _manifest_hash(expected_manifest_sha256)
            or metadata.get("language") != expected_language
            or metadata.get("role") != expected_role
            or metadata.get("sample_ids") != ids
            or metadata.get("sample_ids_sha256") != sample_id_catalog_hash(ids)
            or metadata.get("dtype") != "float32"
            or not isinstance(metadata.get("checkpoint"), str)
            or not metadata["checkpoint"]
        ):
            return None
        if (
            _SHA256_RE.fullmatch(metadata["cache_key"]) is None
            or _profile_id(metadata["profile"], field="profile")
            != expected_profile_id
            or _language(metadata["language"]) != expected_language
            or _simple_id(metadata["role"], field="role") != expected_role
        ):
            return None
        _manifest_hash(metadata.get("manifest_sha256"))
        stored_ids = _validated_sample_ids(metadata["sample_ids"])
        if metadata["sample_ids_sha256"] != sample_id_catalog_hash(stored_ids):
            return None
        array = np.load(Path(embedding_path), allow_pickle=False)
        _validate_embedding_array(array, expected_samples=len(ids))
        if metadata.get("shape") != list(array.shape):
            return None
        return array
    except (KeyError, OSError, TypeError, ValueError):
        return None


def validate_embedding_cache(*args: Any, **kwargs: Any) -> bool:
    """Indica cache hit sem expor exceções de artefatos inválidos."""
    return load_valid_embedding_cache(*args, **kwargs) is not None


def write_embedding_cache(
    embedding_path: str | Path,
    metadata_path: str | Path,
    array: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Publica array e metadata atomicamente, com metadata por último."""
    embedding_path = Path(embedding_path)
    metadata_path = Path(metadata_path)
    if embedding_path == metadata_path:
        raise ValueError("embedding and metadata paths must differ")
    ids = _validated_sample_ids(metadata.get("sample_ids", []))
    _validate_embedding_array(array, expected_samples=len(ids))
    expected_metadata = build_embedding_cache_metadata(
        cache_key=metadata.get("cache_key"),
        profile_id=metadata.get("profile"),
        checkpoint=metadata.get("checkpoint"),
        manifest_sha256=metadata.get("manifest_sha256"),
        language=metadata.get("language"),
        role=metadata.get("role"),
        array=array,
        sample_ids=ids,
    )
    if metadata != expected_metadata:
        raise ValueError("metadata does not match embeddings and cache schema")

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    array_tmp = embedding_path.with_name(f".{embedding_path.name}.{token}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.{token}.tmp")
    array_backup = embedding_path.with_name(f".{embedding_path.name}.{token}.bak")
    metadata_backup = metadata_path.with_name(f".{metadata_path.name}.{token}.bak")
    had_array = embedding_path.is_file()
    had_metadata = metadata_path.is_file()
    try:
        with array_tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with metadata_tmp.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        if had_array:
            shutil.copy2(embedding_path, array_backup)
        if had_metadata:
            shutil.copy2(metadata_path, metadata_backup)
        try:
            os.replace(array_tmp, embedding_path)
            os.replace(metadata_tmp, metadata_path)
        except BaseException:
            if had_array:
                shutil.copy2(array_backup, embedding_path)
            else:
                embedding_path.unlink(missing_ok=True)
            if had_metadata:
                shutil.copy2(metadata_backup, metadata_path)
            else:
                metadata_path.unlink(missing_ok=True)
            raise
    finally:
        array_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        array_backup.unlink(missing_ok=True)
        metadata_backup.unlink(missing_ok=True)
