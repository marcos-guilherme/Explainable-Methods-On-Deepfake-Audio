"""Intervention-based DFT-LRP faithfulness for the layer-12 diagonal XAI cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .bands import mel_band_edges
from .dft_lrp import rfft_frequencies, stdft_lrp
from .encoder_suite import LANGUAGE_ORDER
from .final_trace import _port_head_for_dtype, _task7_cohort, _xai_torch_dtype
from .layerwise_paths import LayerwiseSuitePaths, publish_generation, resolve_active_generation
from .layerwise_xai import select_fixed_cohort
from .lrp_detector import (
    SSLDetectorAD,
    port_logistic_head,
    relevance_for_clip,
    verify_score_equivalence,
)
from .xai_registry import get_encoder_spec

_DEV = Path(__file__).resolve().parents[1] / "dev"
if str(_DEV) not in sys.path:
    sys.path.insert(0, str(_DEV))

import faithfulness_bands as fb  # noqa: E402

FINAL_LAYER = 12
DEFAULT_FAITHFULNESS_PER_CLASS = 8
DEFAULT_KS = tuple(fb._KS_DEFAULT)
_CALIBRATION_EQUIVALENCE_MAX_SAMPLES = 32
_FINGERPRINT_PROTOCOL = "layerwise_dft_lrp_faithfulness_v3"
_FAITHFULNESS_LIMITATIONS = (
    "intervention_filter_confound",
    "stft_band_pass_stop_artifact",
    "rms_matching_energy_control",
    "no_causal_world_claim",
    "layer12_diagonal_only",
    "no_cross_layer_fidelity",
    "no_off_diagonal_transfer_claim",
)
_SCOPE_PT = (
    "Fidelidade por intervenção STFT nas bandas ranqueadas pela relevância DFT-LRP "
    "recomputada (AttnLRP → STDFT) para o probe diagonal da camada 12 "
    "(source=target), sobre subamostra estratificada da coorte XAI fixa da Task 7."
)
_DISCLAIMER = (
    "Intervention-based faithfulness on recomputed layer-12 diagonal explanations; "
    "not classical D_ad runs, not cross-layer, not off-diagonal transfer."
)


def load_authoritative_xai_cohort(
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
) -> pd.DataFrame:
    """Fail-closed load of the persisted Task 7 cohort for one XAI cell."""
    if layer != FINAL_LAYER:
        raise ValueError(f"faithfulness is scoped to layer {FINAL_LAYER} only")
    if source != target:
        raise ValueError("faithfulness is scoped to diagonal source=target cells only")
    return _task7_cohort(paths, profile, layer, source, target)


def select_faithfulness_cohort(
    cohort: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Balanced subsample of the authoritative XAI cohort (deterministic by seed)."""
    if per_class <= 0:
        return cohort.reset_index(drop=True)
    catalog = cohort.loc[:, ["sample_id", "y_true", "processed_path"]].rename(
        columns={"y_true": "label"}
    )
    selected = select_fixed_cohort(catalog, per_class, seed)
    return selected.rename(columns={"label": "y_true"}).reset_index(drop=True)


def load_frequency_edges(suite_root: Path, language: str) -> np.ndarray:
    """Read mel band edges from the language config recorded in execution_plan.json."""
    return load_frequency_band_config(suite_root, language)[1]


def load_frequency_band_config(
    suite_root: Path, language: str
) -> tuple[dict[str, float | int], np.ndarray]:
    """Return canonical band config identity and mel edges for a suite language."""
    from .config import load_config

    plan_path = suite_root / "execution_plan.json"
    if not plan_path.is_file():
        raise ValueError(f"execution_plan.json is missing under {suite_root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict) or language not in inputs:
        raise ValueError(f"execution_plan lacks inputs for language {language!r}")
    config_path = Path(str(inputs[language]["config_path"])).expanduser()
    if not config_path.is_file():
        raise ValueError(f"config_path is missing for {language}: {config_path}")
    bands = load_config(config_path).bands
    identity = {
        "config_path": str(config_path.resolve()),
        "n_bands": int(bands.n_bands),
        "f_min": float(bands.f_min),
        "f_max": float(bands.f_max),
    }
    edges = mel_band_edges(bands.n_bands, bands.f_min, bands.f_max)
    return identity, edges


def active_xai_generation_id(
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
) -> str:
    """Authoritative active Task 7 generation token for resume fingerprinting."""
    pointer_path = paths.layer_xai(profile, layer, source, target) / "active.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        generation = pointer["generation"]
        if (
            pointer.get("schema_version") != 1
            or not isinstance(generation, str)
            or not generation
        ):
            raise ValueError("invalid active generation pointer")
        return generation
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"no valid active XAI generation for {profile}/layer_{layer:02d}/"
            f"{source}->{target}"
        ) from exc


def validate_faithfulness_params(
    *,
    cohort: pd.DataFrame,
    per_class: int,
    ks: Sequence[int],
    n_random: int,
    batch_size: int,
    eps: float,
    stdft_win: int,
    stdft_hop: int,
    n_bands: int,
) -> tuple[int, ...]:
    """Fail-closed validation of CLI/runtime numeric parameters."""
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 0:
        raise ValueError("per_class must be a non-negative integer")
    counts = cohort["y_true"].value_counts()
    for label in (0, 1):
        available = int(counts.get(label, 0))
        if per_class == 0:
            if available == 0:
                raise ValueError(f"cohort lacks class {label} samples")
            continue
        if available < per_class:
            raise ValueError(
                f"class {label} has only {available} samples; per_class={per_class}"
            )
    if not ks:
        raise ValueError("ks must be a non-empty sequence")
    normalized_ks: list[int] = []
    seen: set[int] = set()
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("each k must be an integer")
        if k in seen:
            raise ValueError(f"duplicate k in ks: {k}")
        seen.add(k)
        if k < 0 or k > n_bands:
            raise ValueError(f"k={k} must satisfy 0 <= k <= n_bands ({n_bands})")
        if k > 0 and 2 * k > n_bands:
            raise ValueError(
                f"k={k} exceeds half the band count; top-k and bottom-k would overlap"
            )
        normalized_ks.append(k)
    if isinstance(n_random, bool) or not isinstance(n_random, int) or n_random <= 0:
        raise ValueError("n_random must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        isinstance(stdft_win, bool)
        or not isinstance(stdft_win, int)
        or stdft_win <= 0
    ):
        raise ValueError("stdft_win must be a positive integer")
    if (
        isinstance(stdft_hop, bool)
        or not isinstance(stdft_hop, int)
        or stdft_hop <= 0
        or stdft_hop > stdft_win
    ):
        raise ValueError("stdft_hop must be a positive integer not greater than stdft_win")
    return tuple(normalized_ks)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_artifact_identity(
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
) -> dict[str, str]:
    """Content hashes of persisted probe/threshold artifacts for resume safety."""
    probe_path = paths.probe(profile, layer, source)
    threshold_path = paths.thresholds(profile, layer, source)
    if not probe_path.is_file():
        raise ValueError(f"probe artifact is missing: {probe_path}")
    if not threshold_path.is_file():
        raise ValueError(f"threshold artifact is missing: {threshold_path}")
    return {
        "d_ad_joblib_sha256": _sha256_file(probe_path),
        "thresholds_json_sha256": _sha256_file(threshold_path),
    }


def calibration_layer_embeddings(
    paths: LayerwiseSuitePaths,
    profile: str,
    source: str,
    layer: int,
    *,
    seed: int,
    max_samples: int = _CALIBRATION_EQUIVALENCE_MAX_SAMPLES,
) -> tuple[np.ndarray, int]:
    """Deterministic finite subset of calibration embeddings at ``layer``."""
    array_path = paths.embeddings(profile, source, "calibration")
    if not array_path.is_file():
        raise ValueError(f"calibration embeddings are missing: {array_path}")
    array = np.load(array_path, allow_pickle=False)
    if (
        not isinstance(array, np.ndarray)
        or array.ndim != 3
        or array.shape[1] <= layer
        or array.shape[0] == 0
    ):
        raise ValueError(
            f"calibration embeddings have incompatible shape for layer {layer}: "
            f"{array_path}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"calibration embeddings must be finite: {array_path}")
    total = int(array.shape[0])
    count = min(max_samples, total)
    digest = hashlib.sha256(
        f"{seed}|{profile}|{source}|calibration|layer_{layer:02d}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big") % (2**31))
    if count >= total:
        indices = np.arange(total, dtype=int)
    else:
        indices = np.sort(rng.choice(total, size=count, replace=False))
    vectors = np.asarray(array[indices, layer, :], dtype=np.float64)
    if not np.isfinite(vectors).all():
        raise ValueError("calibration subset contains non-finite vectors")
    return vectors, int(indices.size)


def build_faithfulness_fingerprint(
    *,
    checkpoint: str,
    layer: int,
    source: str,
    target: str,
    ks: Sequence[int],
    n_random: int,
    seed: int,
    per_class: int,
    rms_match: bool,
    stdft_win: int,
    stdft_hop: int,
    eps: float,
    sample_ids: Sequence[str],
    xai_generation_id: str,
    frequency_band_config: Mapping[str, float | int],
    probe_artifacts: Mapping[str, str],
) -> str:
    payload = {
        "checkpoint": checkpoint,
        "layer": layer,
        "source": source,
        "target": target,
        "ks": list(ks),
        "n_random": n_random,
        "seed": seed,
        "per_class": per_class,
        "rms_match": rms_match,
        "stdft_win": stdft_win,
        "stdft_hop": stdft_hop,
        "eps": float(eps),
        "sample_ids": list(sample_ids),
        "xai_generation_id": xai_generation_id,
        "frequency_band_config": dict(frequency_band_config),
        "probe_artifacts": dict(probe_artifacts),
        "protocol": _FINGERPRINT_PROTOCOL,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def should_skip_completed_run(destination: Path, fingerprint: str) -> bool:
    try:
        generation = resolve_active_generation(destination, expected_role="layer_faithfulness")
        manifest_path = generation / "faithfulness_run_manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("fingerprint") == fingerprint
    except ValueError:
        return False


def score_ssl_input_waves(
    model: SSLDetectorAD,
    waves: Sequence[np.ndarray],
    device: str,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Batch logits for perturbed input-value waveforms."""
    import torch

    logits: list[float] = []
    dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
        torch.float32,
    )
    with torch.no_grad():
        for start in range(0, len(waves), batch_size):
            chunk = waves[start : start + batch_size]
            tensor = torch.stack(
                [
                    torch.as_tensor(np.asarray(wave, dtype=np.float64), dtype=dtype, device=device)
                    for wave in chunk
                ]
            )
            logits.extend(model(tensor).detach().cpu().numpy().ravel().tolist())
    return np.asarray(logits, dtype=np.float64)


def process_ssl_intervention_clip(
    model: SSLDetectorAD,
    r_tf: np.ndarray,
    x_time: np.ndarray,
    logit: float,
    freqs: np.ndarray,
    edges: np.ndarray,
    win: int,
    hop: int,
    ks: Sequence[int],
    n_random: int,
    rng: np.random.Generator,
    device: str,
    *,
    batch_size: int = 16,
    rms_match: bool = True,
) -> list[tuple[str, int, float]]:
    """Same intervention schedule as ``faithfulness_bands.process_clip`` for SSLDetectorAD."""

    def score_fn(_model, wave_batch, _device):
        return score_ssl_input_waves(
            _model, wave_batch, _device, batch_size=batch_size
        )

    original_score = fb.score_waves
    try:
        fb.score_waves = score_fn  # type: ignore[assignment]
        return fb.process_clip(
            model,
            r_tf,
            x_time,
            logit,
            freqs,
            edges,
            win,
            hop,
            list(ks),
            n_random,
            rng,
            device,
            rms_match=rms_match,
        )
    finally:
        fb.score_waves = original_score  # type: ignore[assignment]


def build_ssl_detector(
    *,
    profile: str,
    layer: int,
    source: str,
    paths: LayerwiseSuitePaths,
    encoder: object,
    device: str,
    seed: int = 42,
    head_loader: Callable = joblib.load,
    port_head_fn: Callable = port_logistic_head,
    encoder_spec_fn: Callable = get_encoder_spec,
    verify_equivalence_fn: Callable = verify_score_equivalence,
):
    from .attnlrp import ensure_ssl_encoder_attnlrp, patch_ssl_encoder_for_attnlrp

    import torch

    spec = encoder_spec_fn(profile)
    capabilities = frozenset(getattr(spec, "capabilities", ()))
    attention_rule = str(getattr(spec, "attention_rule", ""))
    xai_dtype = _xai_torch_dtype(spec)
    torch_encoder = getattr(encoder, "_model", encoder)
    processor = getattr(encoder, "_processor", None)
    if not isinstance(torch_encoder, torch.nn.Module):
        raise ValueError("encoder must be a torch.nn.Module or HFSSLEmbedder")
    ensure_ssl_encoder_attnlrp(
        torch_encoder,
        attention_rule=attention_rule,
        capabilities=capabilities,
        patch_fn=patch_ssl_encoder_for_attnlrp,
    )
    torch_encoder.to(device=device, dtype=xai_dtype)
    torch_encoder.eval()
    probe_identity = probe_artifact_identity(paths, profile, layer, source)
    head = head_loader(paths.probe(profile, layer, source))
    weight, bias = _port_head_for_dtype(port_head_fn, head, xai_dtype)
    calibration_vectors, calibration_n = calibration_layer_embeddings(
        paths, profile, source, layer, seed=seed
    )
    max_equivalence_error = float(
        verify_equivalence_fn(
            head,
            calibration_vectors,
            np.asarray(weight),
            float(bias),
        )
    )
    model = SSLDetectorAD(
        torch_encoder, layer, np.asarray(weight), float(bias)
    ).to(device=device, dtype=xai_dtype)
    model.eval()
    diagnostics = {
        "probe_artifacts": probe_identity,
        "calibration_equivalence_n": calibration_n,
        "max_score_equivalence_error": max_equivalence_error,
    }
    return model, processor, spec, diagnostics


def build_cell_audio_loader(embedder: object) -> Callable[[Path], tuple[np.ndarray, int]]:
    """One preprocessed loader per cell, sharing the already-built embedder."""
    import soundfile as sf

    from .encoder_suite_runtime import build_preprocessed_audio_loader

    return build_preprocessed_audio_loader(
        embedder,
        read_audio=lambda audio_path: sf.read(
            audio_path, dtype="float32", always_2d=False
        ),
    )


def _assert_finite_csv_rows(rows: Sequence[Mapping[str, object]], *, context: str) -> None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, str)):
                continue
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite {key} in {context}")


def _json_safe_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
    return json.loads(encoded)


def release_profile_encoder(encoder: object | None, *, device: str) -> None:
    """Move a shared profile encoder off GPU and release CUDA cache when possible."""
    import torch

    if encoder is None:
        return
    torch_module = getattr(encoder, "_model", encoder)
    if isinstance(torch_module, torch.nn.Module):
        torch_module.to("cpu")
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_faithfulness_cell(
    *,
    suite_root: Path,
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
    cohort: pd.DataFrame,
    ks: Sequence[int],
    n_random: int,
    seed: int,
    per_class: int,
    rms_match: bool,
    stdft_win: int,
    stdft_hop: int,
    eps: float,
    device: str,
    batch_size: int,
    skip_if_complete: bool,
    encoder: object | None = None,
    relevance_fn: Callable | None = None,
    stdft_fn: Callable | None = None,
    build_detector_fn: Callable | None = None,
    audio_loader: Callable | None = None,
) -> dict[str, object]:
    if build_detector_fn is None:
        build_detector_fn = build_ssl_detector
    if relevance_fn is None:
        relevance_fn = relevance_for_clip
    if stdft_fn is None:
        stdft_fn = stdft_lrp
    if layer != FINAL_LAYER or source != target:
        raise ValueError("only layer-12 diagonal cells are supported")
    spec = get_encoder_spec(profile)
    band_config, edges = load_frequency_band_config(suite_root, source)
    n_bands = len(edges) - 1
    ks = validate_faithfulness_params(
        cohort=cohort,
        per_class=per_class,
        ks=ks,
        n_random=n_random,
        batch_size=batch_size,
        eps=eps,
        stdft_win=stdft_win,
        stdft_hop=stdft_hop,
        n_bands=n_bands,
    )
    selected = select_faithfulness_cohort(cohort, per_class=per_class, seed=seed)
    sample_ids = selected["sample_id"].tolist()
    xai_generation_id = active_xai_generation_id(
        paths, profile, layer, source, target
    )
    probe_artifacts = probe_artifact_identity(paths, profile, layer, source)
    freqs = rfft_frequencies(stdft_win, 16000)
    destination = paths.layer_faithfulness_cell(profile, layer, source, target)
    fingerprint = build_faithfulness_fingerprint(
        checkpoint=spec.checkpoint,
        layer=layer,
        source=source,
        target=target,
        ks=ks,
        n_random=n_random,
        seed=seed,
        per_class=per_class,
        rms_match=rms_match,
        stdft_win=stdft_win,
        stdft_hop=stdft_hop,
        eps=eps,
        sample_ids=sample_ids,
        xai_generation_id=xai_generation_id,
        frequency_band_config=band_config,
        probe_artifacts=probe_artifacts,
    )
    if skip_if_complete and should_skip_completed_run(destination, fingerprint):
        return {"status": "skipped", "profile": profile, "source": source, "n_clips": 0}

    if encoder is None:
        from .encoder_suite import _production_encoder_factory

        encoder = _production_encoder_factory(profile, spec, device)
    detector_out = build_detector_fn(
        profile=profile,
        layer=layer,
        source=source,
        paths=paths,
        encoder=encoder,
        device=device,
        seed=seed,
    )
    if isinstance(detector_out, tuple) and len(detector_out) == 4:
        model, processor, _spec, detector_diagnostics = detector_out
    else:
        model, processor, _spec = detector_out
        detector_diagnostics = {}
    if processor is None:
        raise ValueError("encoder is missing a HuggingFace processor")
    if audio_loader is None:
        audio_loader = build_cell_audio_loader(encoder)

    rng = np.random.default_rng(seed)
    classes = ["spoof", "bonafide"]
    rows: list[dict[str, object]] = []
    p_orig_per_clip: dict[int, float] = {}

    for clip_index, row in enumerate(selected.itertuples(index=False)):
        path = Path(row.processed_path)
        waveform, rate = audio_loader(path)
        if rate != 16000:
            raise ValueError(f"processed audio must be 16 kHz for {row.sample_id}")
        x_time, r_time, logit = relevance_fn(model, processor, waveform, device)
        _times, _freqs, r_tf, _spectrum = stdft_fn(
            x_time,
            r_time,
            16000,
            win_size=stdft_win,
            hop=stdft_hop,
            eps=eps,
        )
        pred_spoof = logit > 0
        true_cls = "spoof" if int(row.y_true) == 1 else "bonafide"
        p_orig = float(fb._sigmoid(logit if pred_spoof else -logit))
        p_orig_per_clip[clip_index] = p_orig
        for condition, k, p_pred in process_ssl_intervention_clip(
            model,
            r_tf,
            x_time,
            logit,
            freqs,
            edges,
            stdft_win,
            stdft_hop,
            ks,
            n_random,
            rng,
            device,
            batch_size=batch_size,
            rms_match=rms_match,
        ):
            rows.append(
                {
                    "index": clip_index,
                    "sample_id": row.sample_id,
                    "true_class": true_cls,
                    "pred_class": "spoof" if pred_spoof else "bonafide",
                    "condition": condition,
                    "k": k,
                    "p_pred": p_pred,
                }
            )

    manifest = _json_safe_manifest(
        {
            "schema_version": 3,
            "fingerprint": fingerprint,
            "profile": profile,
            "checkpoint": spec.checkpoint,
            "layer": layer,
            "source": source,
            "target": target,
            "ks": list(ks),
            "n_random": n_random,
            "seed": seed,
            "per_class": per_class,
            "rms_match": rms_match,
            "stdft_win": stdft_win,
            "stdft_hop": stdft_hop,
            "eps": float(eps),
            "sample_ids": sample_ids,
            "xai_generation_id": xai_generation_id,
            "frequency_band_config": band_config,
            "probe_artifacts": probe_artifacts,
            "calibration_equivalence_n": detector_diagnostics.get(
                "calibration_equivalence_n"
            ),
            "max_score_equivalence_error": detector_diagnostics.get(
                "max_score_equivalence_error"
            ),
            "scope": _SCOPE_PT,
            "disclaimer": _DISCLAIMER,
            "limitations": list(_FAITHFULNESS_LIMITATIONS),
            "device": device,
            "batch_size": batch_size,
        }
    )

    def publish_cell_artifacts(directory: Path) -> None:
        _assert_finite_csv_rows(rows, context="clip_interventions")
        clip_path = directory / "clip_interventions.csv"
        with clip_path.open("w", newline="") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "sample_id",
                    "true_class",
                    "pred_class",
                    "condition",
                    "k",
                    "p_pred",
                ],
            )
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        agg = fb._aggregate(rows, list(ks), classes)
        fb._write_curves_csv(directory / "curves.csv", agg, list(ks), classes)
        fb._write_aopc_csv(
            directory / "aopc.csv", rows, list(ks), classes, p_orig_per_clip
        )
        fb._write_paired_comparisons_csv(
            directory / "paired_comparisons.csv",
            rows,
            list(ks),
            classes,
            p_orig_per_clip,
            seed=seed,
        )
        (directory / "faithfulness_run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    publish_generation(destination, publish_cell_artifacts, role="layer_faithfulness")
    return {
        "status": "published",
        "profile": profile,
        "source": source,
        "target": target,
        "layer": layer,
        "n_clips": len(selected),
    }


def _discover_profiles(
    paths: LayerwiseSuitePaths,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested:
        return tuple(requested)
    discovered = sorted(
        child.name
        for child in paths.root.iterdir()
        if child.is_dir()
        and (child / "embeddings").is_dir()
        and child.name not in {"aggregates", ".stage-artifacts", ".state"}
    )
    if not discovered:
        raise ValueError(f"no profile directories with embeddings under {paths.root}")
    return tuple(discovered)


def run_layerwise_faithfulness(
    suite_root: Path | str,
    profiles: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    *,
    layer: int = FINAL_LAYER,
    per_class: int = DEFAULT_FAITHFULNESS_PER_CLASS,
    ks: Sequence[int] = DEFAULT_KS,
    n_random: int = 5,
    seed: int = 42,
    rms_match: bool = True,
    stdft_win: int = 512,
    stdft_hop: int = 128,
    eps: float = 1e-9,
    device: str = "auto",
    batch_size: int = 16,
    skip_if_complete: bool = True,
    run_cell_fn: Callable | None = None,
    **cell_overrides,
) -> list[dict[str, object]]:
    root = Path(suite_root)
    paths = LayerwiseSuitePaths(root)
    selected_profiles = _discover_profiles(paths, profiles)
    selected_languages = tuple(languages) if languages is not None else LANGUAGE_ORDER
    for language in selected_languages:
        if language not in LANGUAGE_ORDER:
            raise ValueError(f"unknown language: {language!r}")

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    cell_runner = run_cell_fn or (
        lambda **kwargs: run_faithfulness_cell(
            suite_root=root,
            paths=paths,
            skip_if_complete=skip_if_complete,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )
    )

    suite_results: list[dict[str, object]] = []
    for profile in selected_profiles:
        profile_results: list[dict[str, object]] = []
        shared_encoder: object | None = None
        shared_audio_loader: Callable | None = None
        try:
            if run_cell_fn is None:
                from .encoder_suite import _production_encoder_factory

                spec = get_encoder_spec(profile)
                shared_encoder = _production_encoder_factory(profile, spec, device)
                shared_audio_loader = build_cell_audio_loader(shared_encoder)
            for language in selected_languages:
                cohort = load_authoritative_xai_cohort(
                    paths, profile, layer, language, language
                )
                cell_kwargs = dict(cell_overrides)
                if shared_encoder is not None:
                    cell_kwargs.setdefault("encoder", shared_encoder)
                if shared_audio_loader is not None:
                    cell_kwargs.setdefault("audio_loader", shared_audio_loader)
                profile_results.append(
                    cell_runner(
                        profile=profile,
                        layer=layer,
                        source=language,
                        target=language,
                        cohort=cohort,
                        ks=ks,
                        n_random=n_random,
                        seed=seed,
                        per_class=per_class,
                        rms_match=rms_match,
                        stdft_win=stdft_win,
                        stdft_hop=stdft_hop,
                        eps=eps,
                        **cell_kwargs,
                    )
                )
        finally:
            release_profile_encoder(shared_encoder, device=device)
            shared_encoder = None
            shared_audio_loader = None
        suite_results.append(
            {
                "suite_root": str(root),
                "profile": profile,
                "layer": layer,
                "languages": list(selected_languages),
                "cells": profile_results,
            }
        )
    return suite_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled intervention faithfulness for recomputed layer-12 diagonal "
            "DFT-LRP explanations (not classical D_ad runs)."
        )
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        action="append",
        required=True,
        help="Completed layer-wise suite root (repeat for each encoder run).",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Profile IDs (default: auto-detect under each suite root).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Target/source languages for diagonal cells (default: eng por zho).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_FAITHFULNESS_PER_CLASS,
        help="Deterministic balanced subsample per class from the XAI cohort.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--rms-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stdft-win", type=int, default=512)
    parser.add_argument("--stdft-hop", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when an active generation matches the fingerprint.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(args.per_class, bool) or args.per_class < 0:
        raise SystemExit("error: --per-class must be a non-negative integer")
    if args.n_random <= 0:
        raise SystemExit("error: --n-random must be positive")
    if args.batch_size <= 0:
        raise SystemExit("error: --batch-size must be positive")
    if not np.isfinite(args.eps) or args.eps <= 0:
        raise SystemExit("error: --eps must be finite and positive")
    if args.stdft_win <= 0 or args.stdft_hop <= 0 or args.stdft_hop > args.stdft_win:
        raise SystemExit("error: invalid --stdft-win/--stdft-hop")
    if not args.ks:
        raise SystemExit("error: --ks must be non-empty")
    results = []
    for suite_root in args.suite_root:
        results.extend(
            run_layerwise_faithfulness(
                suite_root,
                profiles=args.profiles,
                languages=args.languages,
                per_class=args.per_class,
                ks=tuple(args.ks),
                n_random=args.n_random,
                seed=args.seed,
                rms_match=args.rms_match,
                stdft_win=args.stdft_win,
                stdft_hop=args.stdft_hop,
                eps=args.eps,
                device=args.device,
                batch_size=args.batch_size,
                skip_if_complete=not args.force,
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
