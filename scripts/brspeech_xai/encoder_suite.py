"""CLI and fail-closed resumable orchestration for the layer-wise XAI suite.

This module intentionally imports no torch, transformers, or task runners at
module import time. Real execution is connected through ``SuiteFactories``;
dry-runs only validate inputs and publish the complete execution plan.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import yaml

from .config import LayerwiseXaiConfig
from .xai_registry import get_encoder_spec, resolve_experiment


PLAN_SCHEMA_VERSION = 1
MARKER_SCHEMA_VERSION = 1
CONTRACT_VERSION = "encoder-suite-v2"
LANGUAGE_ORDER = ("eng", "por", "zho")
ROLE_ORDER = ("train", "calibration", "test")
LAYER_ORDER = tuple(range(1, 13))
PRIMARY_METHODS = (
    "layerwise_linear_probe",
    "attnlrp_time",
    "dft_lrp_frequency",
    "stdft_lrp_time_frequency",
    "final_decision_layer_trace",
)
_STAGE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
ARTIFACT_CONTRACTS: Mapping[str, frozenset[str]] = {
    "cohort": frozenset({"cohort", "metadata"}),
    "embedding": frozenset({"embeddings", "metadata"}),
    "probe": frozenset({"head", "thresholds"}),
    "cell": frozenset({"scores", "predictions", "metrics"}),
    "emergence": frozenset({"summary"}),
    "xai": frozenset({"active_pointer"}),
    "trace": frozenset({"active_pointer"}),
    "profile_aggregate": frozenset({"index"}),
    "suite_aggregate": frozenset({"index"}),
}


@dataclass(frozen=True)
class ValidatedInput:
    language: str
    config_path: Path
    config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    role_counts: Mapping[str, int]
    audio_sample_rate: int
    audio_num_samples: int


@dataclass(frozen=True)
class StageRequest:
    """One content-addressed unit passed to an injected suite runner."""

    stage_id: str
    kind: str
    profile: str | None
    checkpoint: str | None
    language: str | None
    role: str | None
    layer: int | None
    source: str | None
    target: str | None
    fingerprint: str
    upstream_fingerprints: tuple[str, ...]
    relevant_manifest_hashes: Mapping[str, str]
    relevant_config_hashes: Mapping[str, str]
    output_dir: Path
    output_root: Path
    seed: int
    device: str
    suite_config: LayerwiseXaiConfig
    inputs: Mapping[str, ValidatedInput]


class SuiteRunner(Protocol):
    def run_stage(
        self,
        request: StageRequest,
        encoder: object | None,
    ) -> Mapping[str, str | Path]: ...

    def validate_stage(
        self,
        request: StageRequest,
        artifacts: Mapping[str, Path],
    ) -> None: ...

    def cleanup_encoder(self, encoder: object | None) -> None: ...


@dataclass(frozen=True)
class SuiteFactories:
    encoder_factory: Callable[[str, object, str], object]
    runner_factory: Callable[[], SuiteRunner]


class ProductionSuiteRunner:
    """Thin lazy bridge from orchestration stages to Tasks 2-8."""

    def __init__(self) -> None:
        from .encoder_suite_runtime import ProductionStageAdapter

        self._adapter = ProductionStageAdapter()

    def run_stage(
        self,
        request: StageRequest,
        encoder: object | None,
    ) -> Mapping[str, str | Path]:
        return self._adapter.run_stage(request, encoder)

    def cleanup_encoder(self, encoder: object | None) -> None:
        self._adapter.cleanup_encoder(encoder)

    def validate_stage(
        self,
        request: StageRequest,
        artifacts: Mapping[str, Path],
    ) -> None:
        self._adapter.validate_stage(request, artifacts)


def _production_encoder_factory(
    profile: str,
    spec: object,
    device: str,
    *,
    num_samples: int = 64600,
) -> object:
    from .encoders.hf_ssl import HFSSLEmbedder

    resolved_device = device
    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = str(getattr(spec, "checkpoint"))
    encoder = HFSSLEmbedder(
        checkpoint=checkpoint,
        layer=-1,
        pooling="mean",
        device=resolved_device,
        attn_implementation="eager",
        num_samples=num_samples,
    )
    expected_layers = int(getattr(spec, "n_transformer_layers"))
    if encoder.n_transformer_layers != expected_layers:
        raise ValueError(
            f"profile {profile!r} expected {expected_layers} transformer layers; "
            f"checkpoint returned {encoder.n_transformer_layers}"
        )
    return encoder


def build_production_factories(*, num_samples: int = 64600) -> SuiteFactories:
    return SuiteFactories(
        encoder_factory=lambda profile, spec, device: _production_encoder_factory(
            profile, spec, device, num_samples=num_samples
        ),
        runner_factory=ProductionSuiteRunner,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run the resumable multilingual layer-wise XAI suite "
            "for a selected language subset."
        )
    )
    parser.add_argument("--eng-config", type=Path)
    parser.add_argument("--por-config", type=Path)
    parser.add_argument("--zho-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(LayerwiseXaiConfig().profiles),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xai-per-class", type=int, default=25)
    parser.add_argument("--stdft-examples-per-class", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--conservation-tolerance", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--classical-audit", action="store_true")
    return parser


def _selected_languages(
    config_paths: Mapping[str, Path | str | None],
) -> tuple[str, ...]:
    unknown = set(config_paths) - set(LANGUAGE_ORDER)
    if unknown:
        raise ValueError(f"unsupported language config(s): {sorted(unknown)}")
    selected = tuple(
        language
        for language in LANGUAGE_ORDER
        if config_paths.get(language) is not None
    )
    if not selected:
        raise ValueError("at least one language config is required")
    return selected


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _canonical_existing_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError(f"{field} must be a file: {candidate}")
    return candidate


def _manifest_from_config(config_path: Path) -> Path:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"config is unreadable: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"config data section is missing: {config_path}")
    if data.get("dataset_kind") != "local_manifest":
        raise ValueError(
            f"suite config must use dataset_kind=local_manifest: {config_path}"
        )
    value = data.get("manifest_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config manifest_path is missing: {config_path}")
    manifest = Path(value).expanduser()
    if not manifest.is_absolute():
        manifest = config_path.parent / manifest
    return _canonical_existing_file(manifest, field="manifest_path")


def _validate_manifest(
    manifest_path: Path, language: str
) -> tuple[dict[str, int], dict[str, dict[int, int]]]:
    """Validate identity/roles without decoding audio or trusting the filename."""
    counts = {role: 0 for role in ROLE_ORDER}
    class_counts = {role: {0: 0, 1: 0} for role in ROLE_ORDER}
    identities: set[str] = set()
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sample_id", "language", "role", "label", "processed_path"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"manifest lacks required columns {sorted(required)}: {manifest_path}"
                )
            for line_number, row in enumerate(reader, start=2):
                sample_id = row.get("sample_id", "")
                row_language = row.get("language")
                role = row.get("role")
                label = row.get("label")
                processed_path = row.get("processed_path", "")
                if not sample_id or sample_id in identities:
                    raise ValueError(
                        f"manifest sample_id is empty or duplicated at line {line_number}"
                    )
                identities.add(sample_id)
                if row_language != language:
                    raise ValueError(
                        f"manifest language mismatch at line {line_number}: "
                        f"expected {language!r}, got {row_language!r}"
                    )
                if role not in counts:
                    raise ValueError(
                        f"manifest role is invalid at line {line_number}: {role!r}"
                    )
                if label not in {"0", "1"}:
                    raise ValueError(
                        f"manifest label is invalid at line {line_number}: {label!r}"
                    )
                if not processed_path:
                    raise ValueError(
                        f"manifest processed_path is empty at line {line_number}"
                    )
                counts[role] += 1
                class_counts[role][int(label)] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"manifest is unreadable: {manifest_path}") from exc
    missing = [role for role, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"manifest is missing required roles {missing}: {manifest_path}")
    unbalanced = [
        role for role, labels in class_counts.items() if any(count == 0 for count in labels.values())
    ]
    if unbalanced:
        raise ValueError(
            f"manifest roles must contain both binary classes {unbalanced}: "
            f"{manifest_path}"
        )
    return counts, class_counts


def _validate_inputs(
    config_paths: Mapping[str, str | Path],
    selected_languages: tuple[str, ...],
) -> dict[str, ValidatedInput]:
    if (
        not selected_languages
        or tuple(
            language
            for language in LANGUAGE_ORDER
            if language in selected_languages
        )
        != selected_languages
    ):
        raise ValueError(
            f"selected languages must be a canonical non-empty subset of "
            f"{list(LANGUAGE_ORDER)}"
        )
    if set(config_paths) != set(selected_languages):
        raise ValueError(
            f"config languages must be exactly {list(selected_languages)}"
        )
    validated: dict[str, ValidatedInput] = {}
    for language in selected_languages:
        from .config import load_config

        config_path = _canonical_existing_file(
            config_paths[language], field=f"{language} config"
        )
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = raw.get("data") if isinstance(raw, dict) else None
        loaded_config = load_config(config_path)
        audio_sample_rate = int(loaded_config.audio.sample_rate)
        audio_num_samples = int(loaded_config.audio.num_samples)
        calibration_split = (
            data.get("calibration_split") if isinstance(data, dict) else None
        )
        calibration_quota = (
            data.get("n_calibration_per_class") if isinstance(data, dict) else None
        )
        if (
            not isinstance(calibration_split, str)
            or calibration_split.strip() != "calibration"
            or isinstance(calibration_quota, bool)
            or not isinstance(calibration_quota, int)
            or calibration_quota <= 0
        ):
            raise ValueError(
                f"scientific calibration for {language} requires explicit "
                "calibration_split='calibration' and a positive "
                "n_calibration_per_class; in-sample calibration is forbidden"
            )
        manifest_path = _manifest_from_config(config_path)
        role_counts, class_counts = _validate_manifest(manifest_path, language)
        available = min(class_counts["calibration"].values())
        if calibration_quota > available:
            raise ValueError(
                f"scientific calibration quota for {language} requests "
                f"{calibration_quota} per class but manifest provides only "
                f"{available}; use an explicit coherent calibration cohort"
            )
        validated[language] = ValidatedInput(
            language=language,
            config_path=config_path,
            config_sha256=_sha256_file(config_path),
            manifest_path=manifest_path,
            manifest_sha256=_sha256_file(manifest_path),
            role_counts=role_counts,
            audio_sample_rate=audio_sample_rate,
            audio_num_samples=audio_num_samples,
        )
    return validated


def _validate_waveform_compatibility(
    inputs: Mapping[str, ValidatedInput],
    selected_languages: tuple[str, ...],
) -> None:
    parameters = {
        "audio.sample_rate": {
            language: item.audio_sample_rate for language, item in inputs.items()
        },
        "audio.num_samples": {
            language: item.audio_num_samples for language, item in inputs.items()
        },
    }
    mismatches = []
    for name, values in parameters.items():
        if len(set(values.values())) != 1:
            rendered = ", ".join(
                f"{language}={values[language]}" for language in selected_languages
            )
            mismatches.append(f"{name} ({rendered})")
    if mismatches:
        language_scope = ", ".join(selected_languages)
        raise ValueError(
            f"waveform configuration mismatch across {language_scope}: "
            + "; ".join(mismatches)
        )


def _resolved_methods(config: LayerwiseXaiConfig) -> dict[str, dict[str, object]]:
    method_ids = list(PRIMARY_METHODS)
    if config.classical_audit:
        method_ids.append("classical_band_audit")
    resolved: dict[str, dict[str, object]] = {}
    for profile in config.profiles:
        experiment = resolve_experiment(profile, method_ids)
        resolved[profile] = {
            "checkpoint": experiment.encoder.checkpoint,
            "encoder": experiment.encoder.encoder,
            "family": experiment.encoder.family,
            "layers": list(experiment.encoder.layer_indices),
            "methods": [method.method_id for method in experiment.methods],
        }
    return resolved


def _suite_config_hash(
    suite_config: LayerwiseXaiConfig,
    *,
    seed: int,
    device: str,
) -> str:
    encoder_profiles = {}
    for profile in suite_config.profiles:
        spec = get_encoder_spec(profile)
        encoder_profiles[profile] = {
            "profile_id": spec.profile_id,
            "encoder": spec.encoder,
            "checkpoint": spec.checkpoint,
            "family": spec.family,
            "n_transformer_layers": spec.n_transformer_layers,
            "layer_indices": list(spec.layer_indices),
            "capabilities": sorted(spec.capabilities),
            "attention_rule": spec.attention_rule,
            "xai_dtype": spec.xai_dtype,
        }
    return _canonical_hash(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "suite_config": asdict(suite_config),
            "encoder_profiles": encoder_profiles,
            "seed": seed,
            "device": device,
        }
    )


def _build_plan(
    *,
    suite_config: LayerwiseXaiConfig,
    inputs: Mapping[str, ValidatedInput],
    selected_languages: tuple[str, ...],
    seed: int,
    device: str,
) -> dict[str, object]:
    profiles = len(suite_config.profiles)
    layers = len(LAYER_ORDER)
    sources = len(selected_languages)
    targets = len(selected_languages)
    cells = profiles * layers * sources * targets
    cohort_size = 2 * suite_config.xai_per_class
    stdft_size = 2 * suite_config.stdft_examples_per_class
    resolved = _resolved_methods(suite_config)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "config_hash": _suite_config_hash(
            suite_config, seed=seed, device=device
        ),
        "seed": seed,
        "device": device,
        "suite_config": asdict(suite_config),
        "profiles": resolved,
        "layers": list(LAYER_ORDER),
        "languages": {
            language: {
                "role": "source_and_target",
                "roles": list(ROLE_ORDER),
            }
            for language in selected_languages
        },
        "counts": {
            "profiles": profiles,
            "layers_per_profile": layers,
            "cells_per_layer": sources * targets,
            "probes": profiles * layers * sources,
            "cells": cells,
        },
        "estimated_work": {
            "explanation_backprops": cells * cohort_size,
            "bias_zeroed_certificate_backprops": cells,
            "bias_zeroed_certificate_semantics": (
                "one additional bias-zeroed model/rule validation per cell"
            ),
            "bias_inclusive_gaps": cells * cohort_size,
            "bias_inclusive_gap_extra_backprops": 0,
            "bias_inclusive_gap_semantics": (
                "diagnostic computed from each explanation backprop; no extra backward"
            ),
            "stdft_examples": cells * stdft_size,
            "final_trace_backprops": profiles * sources * targets * cohort_size,
        },
        "embedding_bytes": {
            "value": None,
            "formula": "sum(N_role) * 13 * H_profile * 4 bytes",
            "declared_n_by_language_role": {
                language: dict(item.role_counts) for language, item in inputs.items()
            },
            "hidden_size": None,
            "reason": (
                "encoder hidden size H is not loaded during planning; bytes remain "
                "unknown until the injected encoder adapter validates its checkpoint"
            ),
        },
        "inputs": {
            language: {
                "config_path": str(item.config_path),
                "config_sha256": item.config_sha256,
                "manifest_path": str(item.manifest_path),
                "manifest_sha256": item.manifest_sha256,
                "role_counts": dict(item.role_counts),
            }
            for language, item in inputs.items()
        },
        "dependencies": {
            "cohort": ["manifest"],
            "embedding": ["manifest", "checkpoint"],
            "probe": ["source/train embedding", "source/calibration embedding"],
            "cell": ["probe", "target/test embedding"],
            "emergence": ["all layers of source->target cells"],
            "xai": ["cell", "target cohort", "target/test embedding"],
            "final_trace": ["layer-12 probe", "layer-12 XAI cohort"],
            "profile_aggregate": ["emergence", "xai", "final_trace"],
            "suite_aggregate": ["all profile aggregates"],
        },
        "planned_steps": [
            "resolve registry and methods",
            "validate configs and manifests by content",
            "publish fixed target cohorts before scores",
            "load and freeze one encoder",
            "extract/cache three roles per selected language",
            "fit probes and score layer-wise matrix",
            "summarize emergence",
            "run layer-wise XAI",
            "trace final decision",
            "release encoder resources",
            "aggregate suite orchestration artifacts",
        ],
    }


def _request_fingerprint(
    *,
    stage_id: str,
    kind: str,
    profile: str | None,
    checkpoint: str | None,
    manifest_hashes: Mapping[str, str],
    input_config_hashes: Mapping[str, str],
    config_hash: str,
    upstream: Sequence[str],
) -> str:
    return _canonical_hash(
        {
            "schema_version": MARKER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "stage_id": stage_id,
            "kind": kind,
            "profile": profile,
            "checkpoint": checkpoint,
            "manifest_hashes": dict(sorted(manifest_hashes.items())),
            "input_config_hashes": dict(sorted(input_config_hashes.items())),
            "config_hash": config_hash,
            "upstream_fingerprints": list(upstream),
        }
    )


def _stage_output_dir(root: Path, stage_id: str) -> Path:
    return root / ".stage-artifacts" / _STAGE_ID_RE.sub("__", stage_id)


def _make_request(
    *,
    output: Path,
    suite_config: LayerwiseXaiConfig,
    inputs: Mapping[str, ValidatedInput],
    seed: int,
    device: str,
    config_hash: str,
    stage_id: str,
    kind: str,
    profile: str | None = None,
    checkpoint: str | None = None,
    language: str | None = None,
    role: str | None = None,
    layer: int | None = None,
    source: str | None = None,
    target: str | None = None,
    relevant_languages: Sequence[str] = (),
    upstream: Sequence[StageRequest] = (),
) -> StageRequest:
    manifest_hashes = {
        language_id: inputs[language_id].manifest_sha256
        for language_id in relevant_languages
    }
    input_config_hashes = {
        language_id: inputs[language_id].config_sha256
        for language_id in relevant_languages
    }
    upstream_fingerprints = tuple(request.fingerprint for request in upstream)
    fingerprint = _request_fingerprint(
        stage_id=stage_id,
        kind=kind,
        profile=profile,
        checkpoint=checkpoint,
        manifest_hashes=manifest_hashes,
        input_config_hashes=input_config_hashes,
        config_hash=config_hash,
        upstream=upstream_fingerprints,
    )
    return StageRequest(
        stage_id=stage_id,
        kind=kind,
        profile=profile,
        checkpoint=checkpoint,
        language=language,
        role=role,
        layer=layer,
        source=source,
        target=target,
        fingerprint=fingerprint,
        upstream_fingerprints=upstream_fingerprints,
        relevant_manifest_hashes=manifest_hashes,
        relevant_config_hashes=input_config_hashes,
        output_dir=_stage_output_dir(output, stage_id),
        output_root=output,
        seed=seed,
        device=device,
        suite_config=suite_config,
        inputs=inputs,
    )


def _build_stage_graph(
    *,
    output: Path,
    suite_config: LayerwiseXaiConfig,
    inputs: Mapping[str, ValidatedInput],
    selected_languages: tuple[str, ...],
    seed: int,
    device: str,
    config_hash: str,
) -> tuple[
    tuple[StageRequest, ...],
    dict[str, tuple[StageRequest, ...]],
    tuple[StageRequest, ...],
    tuple[StageRequest, ...],
]:
    cohorts = tuple(
        _make_request(
            output=output,
            suite_config=suite_config,
            inputs=inputs,
            seed=seed,
            device=device,
            config_hash=config_hash,
            stage_id=f"suite:cohort:{target}",
            kind="cohort",
            target=target,
            relevant_languages=(target,),
        )
        for target in selected_languages
    )
    cohort_map = {request.target: request for request in cohorts}
    core_by_profile: dict[str, tuple[StageRequest, ...]] = {}
    profile_aggregates: list[StageRequest] = []
    for profile in suite_config.profiles:
        spec = get_encoder_spec(profile)
        embeddings: dict[tuple[str, str], StageRequest] = {}
        ordered_core: list[StageRequest] = []
        for language in selected_languages:
            for role in ROLE_ORDER:
                request = _make_request(
                    output=output,
                    suite_config=suite_config,
                    inputs=inputs,
                    seed=seed,
                    device=device,
                    config_hash=config_hash,
                    stage_id=f"{profile}:embedding:{language}:{role}",
                    kind="embedding",
                    profile=profile,
                    checkpoint=spec.checkpoint,
                    language=language,
                    role=role,
                    relevant_languages=(language,),
                )
                embeddings[(language, role)] = request
                ordered_core.append(request)

        probes: dict[tuple[int, str], StageRequest] = {}
        cells: dict[tuple[int, str, str], StageRequest] = {}
        for layer in LAYER_ORDER:
            for source in selected_languages:
                probe = _make_request(
                    output=output,
                    suite_config=suite_config,
                    inputs=inputs,
                    seed=seed,
                    device=device,
                    config_hash=config_hash,
                    stage_id=f"{profile}:probe:{layer:02d}:{source}",
                    kind="probe",
                    profile=profile,
                    checkpoint=spec.checkpoint,
                    layer=layer,
                    source=source,
                    relevant_languages=(source,),
                    upstream=(
                        embeddings[(source, "train")],
                        embeddings[(source, "calibration")],
                    ),
                )
                probes[(layer, source)] = probe
                ordered_core.append(probe)
                for target in selected_languages:
                    cell = _make_request(
                        output=output,
                        suite_config=suite_config,
                        inputs=inputs,
                        seed=seed,
                        device=device,
                        config_hash=config_hash,
                        stage_id=f"{profile}:cell:{layer:02d}:{source}:{target}",
                        kind="cell",
                        profile=profile,
                        checkpoint=spec.checkpoint,
                        layer=layer,
                        source=source,
                        target=target,
                        relevant_languages=(source, target),
                        upstream=(probe, embeddings[(target, "test")]),
                    )
                    cells[(layer, source, target)] = cell
                    ordered_core.append(cell)

        emergence: dict[tuple[str, str], StageRequest] = {}
        for source in selected_languages:
            for target in selected_languages:
                request = _make_request(
                    output=output,
                    suite_config=suite_config,
                    inputs=inputs,
                    seed=seed,
                    device=device,
                    config_hash=config_hash,
                    stage_id=f"{profile}:emergence:{source}:{target}",
                    kind="emergence",
                    profile=profile,
                    checkpoint=spec.checkpoint,
                    source=source,
                    target=target,
                    relevant_languages=(source, target),
                    upstream=tuple(
                        cells[(layer, source, target)] for layer in LAYER_ORDER
                    ),
                )
                emergence[(source, target)] = request
                ordered_core.append(request)

        xai: dict[tuple[int, str, str], StageRequest] = {}
        for layer in LAYER_ORDER:
            for source in selected_languages:
                for target in selected_languages:
                    request = _make_request(
                        output=output,
                        suite_config=suite_config,
                        inputs=inputs,
                        seed=seed,
                        device=device,
                        config_hash=config_hash,
                        stage_id=f"{profile}:xai:{layer:02d}:{source}:{target}",
                        kind="xai",
                        profile=profile,
                        checkpoint=spec.checkpoint,
                        layer=layer,
                        source=source,
                        target=target,
                        relevant_languages=(source, target),
                        upstream=(
                            cells[(layer, source, target)],
                            cohort_map[target],
                        ),
                    )
                    xai[(layer, source, target)] = request
                    ordered_core.append(request)

        traces: dict[tuple[str, str], StageRequest] = {}
        final_layer = LAYER_ORDER[-1]
        for source in selected_languages:
            for target in selected_languages:
                request = _make_request(
                    output=output,
                    suite_config=suite_config,
                    inputs=inputs,
                    seed=seed,
                    device=device,
                    config_hash=config_hash,
                    stage_id=f"{profile}:trace:{source}:{target}",
                    kind="trace",
                    profile=profile,
                    checkpoint=spec.checkpoint,
                    source=source,
                    target=target,
                    relevant_languages=(source, target),
                    upstream=(
                        probes[(final_layer, source)],
                        xai[(final_layer, source, target)],
                    ),
                )
                traces[(source, target)] = request
                ordered_core.append(request)
        core_by_profile[profile] = tuple(ordered_core)
        aggregate = _make_request(
            output=output,
            suite_config=suite_config,
            inputs=inputs,
            seed=seed,
            device=device,
            config_hash=config_hash,
            stage_id=f"{profile}:profile_aggregate",
            kind="profile_aggregate",
            profile=profile,
            checkpoint=spec.checkpoint,
            relevant_languages=selected_languages,
            upstream=(
                *emergence.values(),
                *xai.values(),
                *traces.values(),
            ),
        )
        profile_aggregates.append(aggregate)

    suite_aggregate = _make_request(
        output=output,
        suite_config=suite_config,
        inputs=inputs,
        seed=seed,
        device=device,
        config_hash=config_hash,
        stage_id="suite:aggregate",
        kind="suite_aggregate",
        relevant_languages=selected_languages,
        upstream=tuple(profile_aggregates),
    )
    return (
        cohorts,
        core_by_profile,
        tuple(profile_aggregates),
        (suite_aggregate,),
    )


def _marker_path(root: Path, stage_id: str) -> Path:
    return root / ".state" / f"{_STAGE_ID_RE.sub('__', stage_id)}.json"


def _marker_valid(
    root: Path,
    request: StageRequest,
    runner: SuiteRunner | None = None,
) -> bool:
    marker_path = _marker_path(root, request.stage_id)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            not isinstance(marker, dict)
            or set(marker) != {
                "schema_version",
                "contract_version",
                "stage_id",
                "profile",
                "fingerprint",
                "artifacts",
            }
            or marker["schema_version"] != MARKER_SCHEMA_VERSION
            or marker["contract_version"] != CONTRACT_VERSION
            or marker["stage_id"] != request.stage_id
            or marker["profile"] != request.profile
            or marker["fingerprint"] != request.fingerprint
            or not isinstance(marker["artifacts"], list)
            or not marker["artifacts"]
        ):
            return False
        resolved: dict[str, Path] = {}
        for artifact in marker["artifacts"]:
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"name", "path", "sha256"}
                or not isinstance(artifact["name"], str)
                or not isinstance(artifact["path"], str)
                or not isinstance(artifact["sha256"], str)
            ):
                return False
            path = Path(artifact["path"])
            canonical_root = root.resolve()
            if (
                not path.is_absolute()
                or not path.is_file()
                or not path.resolve().is_relative_to(canonical_root)
                or _sha256_file(path) != artifact["sha256"]
            ):
                return False
            resolved[artifact["name"]] = path.resolve()
        if {artifact["name"] for artifact in marker["artifacts"]} != set(
            ARTIFACT_CONTRACTS[request.kind]
        ):
            return False
        validator = None if runner is None else getattr(runner, "validate_stage", None)
        if callable(validator):
            validator(request, resolved)
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        return False


def _publish_marker(
    root: Path,
    request: StageRequest,
    artifacts: Mapping[str, str | Path],
    runner: SuiteRunner,
) -> None:
    expected = ARTIFACT_CONTRACTS.get(request.kind)
    if expected is None or set(artifacts) != set(expected):
        raise ValueError(
            f"artifact contract for {request.kind!r} requires "
            f"{sorted(expected or ())}; received {sorted(artifacts)}"
        )
    resolved = {
        name: Path(path).expanduser().resolve(strict=True)
        for name, path in artifacts.items()
    }
    paths = tuple(resolved.values())
    canonical_root = root.resolve()
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(
            f"stage {request.stage_id!r} must return at least one existing file artifact"
        )
    if len(set(paths)) != len(paths):
        raise ValueError(f"stage {request.stage_id!r} returned duplicate artifacts")
    if any(not path.is_relative_to(canonical_root) for path in paths):
        raise ValueError(
            f"stage {request.stage_id!r} returned an artifact outside the suite root"
        )
    validator = getattr(runner, "validate_stage", None)
    if callable(validator):
        validator(request, resolved)
    _atomic_json(
        _marker_path(root, request.stage_id),
        {
            "schema_version": MARKER_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "stage_id": request.stage_id,
            "profile": request.profile,
            "fingerprint": request.fingerprint,
            "artifacts": [
                {
                    "name": name,
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for name, path in sorted(resolved.items())
            ],
        },
    )


def _execute_stage(
    *,
    root: Path,
    runner: SuiteRunner,
    request: StageRequest,
    encoder: object | None,
    force: bool,
) -> bool:
    if not force and _marker_valid(root, request, runner):
        return False
    artifacts = runner.run_stage(request, encoder)
    if not isinstance(artifacts, Mapping):
        raise ValueError(
            f"runner must return a named artifact mapping for {request.stage_id}"
        )
    _publish_marker(root, request, artifacts, runner)
    return True


def _freeze_encoder(encoder: object) -> None:
    model = getattr(encoder, "_model", encoder)
    eval_fn = getattr(model, "eval", None)
    freeze_fn = getattr(model, "requires_grad_", None)
    if not callable(eval_fn) or not callable(freeze_fn):
        raise ValueError(
            "encoder model must expose eval() and requires_grad_(False)"
        )
    eval_fn()
    freeze_fn(False)


def _default_cleanup(runner: SuiteRunner, encoder: object | None) -> None:
    cleanup = getattr(runner, "cleanup_encoder", None)
    if callable(cleanup):
        cleanup(encoder)
    del encoder
    gc.collect()


def run_encoder_suite(
    *,
    eng_config: str | Path | None = None,
    por_config: str | Path | None = None,
    zho_config: str | Path | None = None,
    output: str | Path,
    suite_config: LayerwiseXaiConfig | None = None,
    seed: int = 42,
    device: str = "auto",
    dry_run: bool = False,
    force: bool = False,
    factories: SuiteFactories | None = None,
) -> dict[str, object]:
    """Validate, plan, and optionally execute the suite through injected adapters."""
    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "run_status.json"
    current_stage = "initialize"
    current_profile: str | None = None
    executed = 0
    skipped = 0
    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "running",
            "stage": current_stage,
            "profile": None,
        },
    )
    try:
        current_stage = "validate_parameters"
        if suite_config is None:
            suite_config = LayerwiseXaiConfig()
        if not isinstance(suite_config, LayerwiseXaiConfig):
            raise TypeError("suite_config must be a LayerwiseXaiConfig")
        seed = _strict_int(seed, field="seed")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        if type(dry_run) is not bool or type(force) is not bool:
            raise TypeError("dry_run and force must be strict booleans")

        current_stage = "validate_inputs"
        config_paths = {
            "eng": eng_config,
            "por": por_config,
            "zho": zho_config,
        }
        selected_languages = _selected_languages(config_paths)
        inputs = _validate_inputs(
            {language: config_paths[language] for language in selected_languages},
            selected_languages,
        )
        _validate_waveform_compatibility(inputs, selected_languages)
        current_stage = "build_execution_plan"
        plan = _build_plan(
            suite_config=suite_config,
            inputs=inputs,
            selected_languages=selected_languages,
            seed=seed,
            device=device,
        )
        plan_path = output_root / "execution_plan.json"
        _atomic_json(plan_path, plan)
        if dry_run:
            status = {
                "schema_version": 1,
                "status": "complete",
                "mode": "dry-run",
                "stage": None,
                "profile": None,
                "config_hash": plan["config_hash"],
            }
            _atomic_json(status_path, status)
            return {
                "status": "dry-run",
                "execution_plan": plan_path,
                "counts": plan["counts"],
            }

        current_stage = "build_stage_graph"
        config_hash = str(plan["config_hash"])
        cohorts, core_by_profile, profile_aggregates, suite_aggregates = (
            _build_stage_graph(
                output=output_root,
                suite_config=suite_config,
                inputs=inputs,
                selected_languages=selected_languages,
                seed=seed,
                device=device,
                config_hash=config_hash,
            )
        )
        if factories is None:
            current_stage = "build_production_factories"
            factories = build_production_factories(
                num_samples=inputs[selected_languages[0]].audio_num_samples
            )
        if not isinstance(factories, SuiteFactories):
            raise TypeError("factories must be a SuiteFactories instance")
        current_stage = "runner_factory"
        runner = factories.runner_factory()
        if not callable(getattr(runner, "run_stage", None)):
            raise ValueError("runner_factory must return an object with run_stage()")

        for request in cohorts:
            current_stage = request.stage_id
            current_profile = None
            changed = _execute_stage(
                root=output_root,
                runner=runner,
                request=request,
                encoder=None,
                force=force,
            )
            executed += int(changed)
            skipped += int(not changed)

        for profile, aggregate in zip(suite_config.profiles, profile_aggregates):
            current_profile = profile
            core = core_by_profile[profile]
            needs_encoder = force or any(
                not _marker_valid(output_root, request, runner) for request in core
            )
            encoder: object | None = None
            if needs_encoder:
                spec = get_encoder_spec(profile)
                current_stage = f"{profile}:encoder_load"
                try:
                    encoder = factories.encoder_factory(profile, spec, device)
                    _freeze_encoder(encoder)
                    for request in core:
                        current_stage = request.stage_id
                        changed = _execute_stage(
                            root=output_root,
                            runner=runner,
                            request=request,
                            encoder=encoder,
                            force=force,
                        )
                        executed += int(changed)
                        skipped += int(not changed)
                finally:
                    _default_cleanup(runner, encoder)
                    encoder = None
            else:
                skipped += len(core)

            current_stage = aggregate.stage_id
            changed = _execute_stage(
                root=output_root,
                runner=runner,
                request=aggregate,
                encoder=None,
                force=force,
            )
            executed += int(changed)
            skipped += int(not changed)

        for request in suite_aggregates:
            current_stage = request.stage_id
            current_profile = None
            changed = _execute_stage(
                root=output_root,
                runner=runner,
                request=request,
                encoder=None,
                force=force,
            )
            executed += int(changed)
            skipped += int(not changed)
    except BaseException as exc:
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "status": "failed",
                "stage": current_stage,
                "profile": current_profile,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    status = {
        "schema_version": 1,
        "status": "complete",
        "stage": None,
        "profile": None,
        "executed_stages": executed,
        "skipped_stages": skipped,
        "config_hash": config_hash,
    }
    _atomic_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "run_status.json"
    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "status": "running",
            "stage": "validate_suite_config",
            "profile": None,
        },
    )
    try:
        suite_config = LayerwiseXaiConfig(
            profiles=tuple(args.profiles),
            xai_per_class=args.xai_per_class,
            stdft_examples_per_class=args.stdft_examples_per_class,
            bootstrap_samples=args.bootstrap_samples,
            conservation_tolerance=args.conservation_tolerance,
            classical_audit=args.classical_audit,
        )
        result = run_encoder_suite(
            eng_config=args.eng_config,
            por_config=args.por_config,
            zho_config=args.zho_config,
            output=args.output,
            suite_config=suite_config,
            seed=args.seed,
            device=args.device,
            dry_run=args.dry_run,
            force=args.force,
        )
    except Exception as exc:
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            current = {}
        if current.get("status") == "running":
            _atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "stage": current.get("stage", "validate_suite_config"),
                    "profile": current.get("profile"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        print(f"encoder suite failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
