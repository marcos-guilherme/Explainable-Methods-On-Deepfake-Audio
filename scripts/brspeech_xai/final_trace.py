"""Rastreamento da mesma decisão final através dos hidden states do encoder."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from .attnlrp import ensure_ssl_encoder_attnlrp, patch_ssl_encoder_for_attnlrp
from .layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
)
from .lrp_detector import SSLDetectorAD, apply_head_to_hidden, port_logistic_head
from .xai_registry import get_encoder_spec


@dataclass(frozen=True)
class FinalDecisionTrace:
    """Métricas de uma amostra para a decisão definida pela head final."""

    layer_indices: tuple[int, ...]
    signed_sum: np.ndarray
    absolute_mass: np.ndarray
    temporal_entropy: np.ndarray
    adjacent_similarity: np.ndarray
    adjacent_l1_change: np.ndarray
    temporal_mass: tuple[np.ndarray, ...]
    valid_lengths: tuple[int, ...]
    transition_lengths: tuple[int, ...]
    interpolated: tuple[bool, ...]
    zero_mass: tuple[bool, ...]
    logit: float
    profile: str = ""
    source: str = ""
    target: str = ""
    sample_id: str = ""


def _positive_layer_count(expected_layers: int) -> int:
    if (
        isinstance(expected_layers, bool)
        or not isinstance(expected_layers, Integral)
        or int(expected_layers) < 1
    ):
        raise ValueError("expected_layers must be a positive integer")
    return int(expected_layers)


def _normalized_mass(vector: np.ndarray, eps: float) -> tuple[np.ndarray, bool]:
    total = float(vector.sum())
    if total <= eps:
        return np.zeros_like(vector), True
    return vector / total, False


def _interpolate(vector: np.ndarray, length: int) -> np.ndarray:
    if vector.size == length:
        return vector.copy()
    source = np.linspace(0.0, 1.0, num=vector.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    return np.interp(target, source, vector)


def _transition(
    previous: np.ndarray,
    current: np.ndarray,
    eps: float,
) -> tuple[float, float, int, bool]:
    length = min(previous.size, current.size)
    if length <= 0:
        raise ValueError("adjacent layers must have a positive valid temporal length")
    adjusted = previous.size != current.size
    left = _interpolate(previous, length)
    right = _interpolate(current, length)
    left, left_zero = _normalized_mass(left, eps)
    right, right_zero = _normalized_mass(right, eps)
    if left_zero and right_zero:
        similarity = 1.0
    elif left_zero or right_zero:
        similarity = 0.0
    else:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        similarity = float(np.dot(left, right) / max(denominator, eps))
    l1_change = float(np.abs(left - right).sum())
    return similarity, l1_change, length, adjusted


def trace_final_decision(
    model: SSLDetectorAD,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    profile: str = "",
    source: str = "",
    target: str = "",
    sample_id: str = "",
    expected_layers: int = 12,
    eps: float = 1e-12,
    attention_rule: str = "cp_lrp",
    capabilities: Sequence[str] = ("attnlrp_cp",),
) -> FinalDecisionTrace:
    """Executa um forward e um backward do logit final e resume cada camada."""
    expected_layers = _positive_layer_count(expected_layers)
    if not isinstance(model, SSLDetectorAD):
        raise ValueError("model must be an SSLDetectorAD with the transplanted final head")
    if model.layer != expected_layers:
        raise ValueError(
            f"model must use the final layer {expected_layers}; received {model.layer}"
        )
    ensure_ssl_encoder_attnlrp(
        model.encoder,
        attention_rule=attention_rule,
        capabilities=capabilities,
        patch_fn=None,
    )
    model.encoder.eval()
    model.encoder.requires_grad_(False)
    if (
        not isinstance(input_values, torch.Tensor)
        or input_values.ndim != 2
        or input_values.shape[0] != 1
        or input_values.shape[1] == 0
    ):
        raise ValueError("input_values must have batch=1 and a positive temporal length")
    if not torch.isfinite(input_values).all():
        raise ValueError("input_values must contain only finite values")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if attention_mask is not None:
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.ndim != 2
            or attention_mask.shape != input_values.shape
            or not torch.all((attention_mask == 0) | (attention_mask == 1))
        ):
            raise ValueError("attention_mask must be binary and match input_values")

    values = input_values.detach().clone().requires_grad_(True)
    mask = None if attention_mask is None else attention_mask.to(values.device)
    model.zero_grad(set_to_none=True)
    kwargs = {"output_hidden_states": True}
    if mask is not None:
        kwargs["attention_mask"] = mask
    output = model.encoder(values, **kwargs)
    hidden_states = getattr(output, "hidden_states", None)
    expected_states = expected_layers + 1
    if hidden_states is None or len(hidden_states) != expected_states:
        actual = 0 if hidden_states is None else len(hidden_states)
        raise ValueError(
            f"encoder hidden states must contain exactly {expected_states} tensors; "
            f"received {actual}"
        )

    retained = tuple(hidden_states[1:])
    frame_masks: list[torch.Tensor] = []
    for index, hidden in enumerate(retained, start=1):
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] == 0
            or hidden.shape[2] == 0
        ):
            raise ValueError(f"hidden state {index} has an incompatible shape")
        if not hidden.requires_grad:
            raise ValueError(f"hidden state {index} does not require gradients")
        if not torch.isfinite(hidden).all():
            raise ValueError(f"hidden state {index} contains non-finite values")
        hidden.retain_grad()
        frame_masks.append(
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if mask is None
            else model._hidden_frame_mask(hidden, mask)
        )

    final_hidden = retained[-1]
    logits = apply_head_to_hidden(
        model,
        final_hidden,
        attention_mask=mask,
    )
    if logits.shape != (1,) or not torch.isfinite(logits).all():
        raise ValueError("final head must produce one finite logit")
    logits[0].backward()

    signed: list[float] = []
    absolute: list[float] = []
    entropies: list[float] = []
    masses: list[np.ndarray] = []
    zero_mass: list[bool] = []
    valid_lengths: list[int] = []
    for index, (hidden, frame_mask) in enumerate(
        zip(retained, frame_masks), start=1
    ):
        gradient = hidden.grad
        if gradient is None or gradient.shape != hidden.shape:
            raise ValueError(f"hidden state {index} gradient is absent or incompatible")
        if not torch.isfinite(gradient).all():
            raise ValueError(f"hidden state {index} gradient contains non-finite values")
        relevance = hidden * gradient
        relevance = relevance * frame_mask.unsqueeze(-1).to(relevance.dtype)
        temporal = relevance.abs().sum(dim=-1)[0][frame_mask[0]]
        if temporal.numel() == 0:
            raise ValueError(f"hidden state {index} has no valid frames")
        total = temporal.sum()
        probabilities = temporal / total.clamp_min(eps)
        entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
        metrics = (relevance.sum(), total, entropy)
        if not all(torch.isfinite(metric) for metric in metrics):
            raise ValueError(f"hidden state {index} produced non-finite metrics")
        mass = temporal.detach().cpu().numpy().astype(np.float64, copy=False)
        masses.append(mass)
        signed.append(float(metrics[0].detach().cpu()))
        absolute.append(float(metrics[1].detach().cpu()))
        entropies.append(float(metrics[2].detach().cpu()))
        zero_mass.append(float(metrics[1].detach().cpu()) <= eps)
        valid_lengths.append(int(temporal.numel()))

    similarities: list[float] = []
    l1_changes: list[float] = []
    transition_lengths: list[int] = []
    interpolated: list[bool] = []
    for previous, current in zip(masses, masses[1:]):
        similarity, l1_change, length, adjusted = _transition(
            previous, current, eps
        )
        similarities.append(similarity)
        l1_changes.append(l1_change)
        transition_lengths.append(length)
        interpolated.append(adjusted)

    arrays = [
        np.asarray(signed, dtype=np.float64),
        np.asarray(absolute, dtype=np.float64),
        np.asarray(entropies, dtype=np.float64),
        np.asarray(similarities, dtype=np.float64),
        np.asarray(l1_changes, dtype=np.float64),
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("final trace contains non-finite metrics")
    return FinalDecisionTrace(
        layer_indices=tuple(range(1, expected_layers + 1)),
        signed_sum=arrays[0],
        absolute_mass=arrays[1],
        temporal_entropy=arrays[2],
        adjacent_similarity=arrays[3],
        adjacent_l1_change=arrays[4],
        temporal_mass=tuple(masses),
        valid_lengths=tuple(valid_lengths),
        transition_lengths=tuple(transition_lengths),
        interpolated=tuple(interpolated),
        zero_mass=tuple(zero_mass),
        logit=float(logits[0].detach().cpu()),
        profile=profile,
        source=source,
        target=target,
        sample_id=sample_id,
    )


def layer_transition_metrics(
    traces: Sequence[FinalDecisionTrace],
) -> pd.DataFrame:
    """Expande traces identificados em uma linha por transição adjacente."""
    values = list(traces)
    identities: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, object]] = []
    for trace in values:
        if not isinstance(trace, FinalDecisionTrace):
            raise ValueError("traces must contain FinalDecisionTrace instances")
        identity = (trace.profile, trace.source, trace.target, trace.sample_id)
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError("every trace must have a complete explicit identity")
        if identity in identities:
            raise ValueError(f"duplicate trace identity: {identity}")
        identities.add(identity)
        layers = len(trace.layer_indices)
        if (
            layers < 1
            or trace.layer_indices != tuple(range(1, layers + 1))
            or any(
                len(value) != layers
                for value in (
                    trace.signed_sum,
                    trace.absolute_mass,
                    trace.temporal_entropy,
                    trace.temporal_mass,
                    trace.valid_lengths,
                    trace.zero_mass,
                )
            )
            or any(
                len(value) != layers - 1
                for value in (
                    trace.adjacent_similarity,
                    trace.adjacent_l1_change,
                    trace.transition_lengths,
                    trace.interpolated,
                )
            )
        ):
            raise ValueError(f"trace shapes are incompatible for identity {identity}")
        for offset in range(1, layers):
            row = {
                "profile": trace.profile,
                "source": trace.source,
                "target": trace.target,
                "sample_id": trace.sample_id,
                "previous_layer": trace.layer_indices[offset - 1],
                "current_layer": trace.layer_indices[offset],
                "similarity": float(trace.adjacent_similarity[offset - 1]),
                "normalized_l1_change": float(
                    trace.adjacent_l1_change[offset - 1]
                ),
                "absolute_mass": float(trace.absolute_mass[offset]),
                "temporal_entropy": float(trace.temporal_entropy[offset]),
                "signed_sum": float(trace.signed_sum[offset]),
                "logit": float(trace.logit),
                "previous_valid_length": trace.valid_lengths[offset - 1],
                "current_valid_length": trace.valid_lengths[offset],
                "comparison_length": trace.transition_lengths[offset - 1],
                "interpolated": trace.interpolated[offset - 1],
                "previous_zero_mass": trace.zero_mass[offset - 1],
                "current_zero_mass": trace.zero_mass[offset],
            }
            numeric = [
                row["similarity"],
                row["normalized_l1_change"],
                row["absolute_mass"],
                row["temporal_entropy"],
                row["signed_sum"],
                row["logit"],
            ]
            if not np.isfinite(numeric).all():
                raise ValueError(f"trace metrics are non-finite for identity {identity}")
            rows.append(row)
    return pd.DataFrame(rows)


def _task7_cohort(
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
) -> pd.DataFrame:
    generation = resolve_active_generation(
        paths.layer_xai(profile, layer, source, target)
    )
    try:
        frame = pd.read_parquet(generation / "sample_relevance.parquet")
    except Exception as exc:
        raise ValueError("Task 7 cohort artifact is unreadable") from exc
    required = {"sample_id", "y_true", "processed_path"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Task 7 cohort artifact lacks authoritative fields: "
            f"{sorted(required - set(frame.columns))}"
        )
    ids = tuple(frame["sample_id"].tolist())
    if (
        not ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Task 7 cohort IDs are invalid")
    labels = frame["y_true"].to_numpy()
    if labels.dtype.kind not in {"i", "u"} or not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Task 7 cohort y_true must contain binary integers")
    processed_paths = frame["processed_path"].tolist()
    if any(
        not isinstance(path, (str, Path)) or not str(path)
        for path in processed_paths
    ):
        raise ValueError("Task 7 cohort processed_path values are invalid")
    normalized = frame.loc[:, ["sample_id", "y_true", "processed_path"]].copy()
    normalized["processed_path"] = normalized["processed_path"].map(str)
    return normalized.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _cohort_mapping(frame: pd.DataFrame) -> dict[str, tuple[int, str]]:
    return {
        row.sample_id: (int(row.y_true), str(row.processed_path))
        for row in frame.itertuples(index=False)
    }


def run_final_decision_trace_cell(
    *,
    profile: str,
    source: str,
    target: str,
    cohort: pd.DataFrame,
    paths: LayerwiseSuitePaths,
    encoder: torch.nn.Module,
    processor: object,
    audio_loader: Callable,
    expected_layers: int = 12,
    sample_rate: int = 16000,
    device: str | torch.device = "cpu",
    head_loader: Callable = joblib.load,
    port_head_fn: Callable = port_logistic_head,
    encoder_spec_fn: Callable = get_encoder_spec,
    patch_encoder_fn: Callable = patch_ssl_encoder_for_attnlrp,
) -> pd.DataFrame:
    """Publish one physical generation for one ``profile/source/target`` trace."""
    expected_layers = _positive_layer_count(expected_layers)
    if not isinstance(paths, LayerwiseSuitePaths):
        raise TypeError("paths must be a LayerwiseSuitePaths instance")
    if not isinstance(cohort, pd.DataFrame):
        raise ValueError("cohort must be a pandas DataFrame")
    required = {"sample_id", "y_true", "processed_path"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"cohort lacks required columns: {sorted(required - set(cohort.columns))}")
    ids = cohort["sample_id"].tolist()
    if (
        not ids
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("cohort sample_id values must be non-empty and unique")
    labels = cohort["y_true"].to_numpy()
    if labels.dtype.kind not in {"i", "u"} or not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("cohort y_true must contain binary integers")
    if not isinstance(encoder, torch.nn.Module):
        raise ValueError("encoder must be a torch module")

    spec = encoder_spec_fn(profile)
    capabilities = frozenset(getattr(spec, "capabilities", ()))
    attention_rule = str(getattr(spec, "attention_rule", ""))
    ensure_ssl_encoder_attnlrp(
        encoder,
        attention_rule=attention_rule,
        capabilities=capabilities,
        patch_fn=patch_encoder_fn,
    )
    encoder.to(device)
    encoder.eval()
    head = head_loader(paths.probe(profile, expected_layers, source))
    weight, bias = port_head_fn(head)
    model = SSLDetectorAD(
        encoder, expected_layers, np.asarray(weight), float(bias)
    ).to(device)
    model.eval()

    traces: list[FinalDecisionTrace] = []
    for row in cohort.itertuples(index=False):
        path = Path(row.processed_path)
        if not path.is_file():
            raise ValueError(f"processed_path is invalid for {row.sample_id}: {path}")
        loaded = audio_loader(path)
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ValueError("audio_loader must return (waveform, sample_rate)")
        waveform, actual_rate = loaded
        waveform = np.asarray(waveform)
        if (
            waveform.ndim != 1
            or waveform.size == 0
            or not np.isfinite(waveform).all()
            or actual_rate != sample_rate
        ):
            raise ValueError(
                f"canonical audio for {row.sample_id} must be finite mono "
                f"at target sample rate {sample_rate} Hz"
            )
        processed = processor(
            [waveform],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        if not isinstance(processed, Mapping) or "input_values" not in processed:
            raise ValueError("processor must return input_values")
        input_values = torch.as_tensor(processed["input_values"]).to(device)
        attention_mask = processed.get("attention_mask")
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask).to(device)
        traces.append(
            trace_final_decision(
                model,
                input_values,
                attention_mask=attention_mask,
                profile=profile,
                source=source,
                target=target,
                sample_id=row.sample_id,
                expected_layers=expected_layers,
                attention_rule=attention_rule,
                capabilities=capabilities,
            )
        )

    transitions = layer_transition_metrics(traces)
    layer_rows: list[dict[str, object]] = []
    for trace in traces:
        for offset, layer in enumerate(trace.layer_indices):
            layer_rows.append(
                {
                    "profile": trace.profile,
                    "source": trace.source,
                    "target": trace.target,
                    "sample_id": trace.sample_id,
                    "layer": layer,
                    "signed_sum": float(trace.signed_sum[offset]),
                    "absolute_mass": float(trace.absolute_mass[offset]),
                    "temporal_entropy": float(trace.temporal_entropy[offset]),
                    "temporal_mass": trace.temporal_mass[offset].tolist(),
                    "valid_length": trace.valid_lengths[offset],
                    "zero_mass": trace.zero_mass[offset],
                    "logit": trace.logit,
                }
            )
    layer_frame = pd.DataFrame(layer_rows)

    def write(directory: Path) -> None:
        layer_frame.to_parquet(directory / "layer_relevance.parquet", index=False)
        transitions.to_csv(directory / "layer_transition_metrics.csv", index=False)

    publish_generation(
        paths.final_trace_cell(profile, source, target),
        write,
        role="final_trace",
    )
    return transitions


def run_final_decision_trace(
    *,
    profiles: Sequence[str],
    sources: Sequence[str],
    targets: Sequence[str],
    paths: LayerwiseSuitePaths,
    encoder_factory: Callable,
    audio_loader: Callable,
    expected_layers: int = 12,
    sample_rate: int = 16000,
    device: str | torch.device = "cpu",
    head_loader: Callable = joblib.load,
    port_head_fn: Callable = port_logistic_head,
    encoder_spec_fn: Callable = get_encoder_spec,
    patch_encoder_fn: Callable = patch_ssl_encoder_for_attnlrp,
) -> pd.DataFrame:
    """Reusa a coorte publicada na Task 7 e a head final de cada fonte."""
    expected_layers = _positive_layer_count(expected_layers)
    if not isinstance(paths, LayerwiseSuitePaths):
        raise TypeError("paths must be a LayerwiseSuitePaths instance")
    profiles = tuple(profiles)
    sources = tuple(sources)
    targets = tuple(targets)
    if not profiles or not sources or not targets:
        raise ValueError("profiles, sources, and targets must be non-empty")
    if any(len(values) != len(set(values)) for values in (profiles, sources, targets)):
        raise ValueError("profiles, sources, and targets must not contain duplicates")

    persisted_cohorts: dict[str, pd.DataFrame] = {}
    for profile in profiles:
        for source in sources:
            for target in targets:
                candidate = _task7_cohort(
                    paths, profile, expected_layers, source, target
                )
                if target not in persisted_cohorts:
                    persisted_cohorts[target] = candidate
                    continue
                authority = _cohort_mapping(persisted_cohorts[target])
                observed = _cohort_mapping(candidate)
                if set(authority) != set(observed):
                    raise ValueError(
                        f"Task 7 cohort IDs diverge for {profile}/{source}->{target}"
                    )
                for sample_id in sorted(authority):
                    expected_label, expected_path = authority[sample_id]
                    actual_label, actual_path = observed[sample_id]
                    if actual_label != expected_label:
                        raise ValueError(
                            f"Task 7 cohort y_true diverges for sample_id={sample_id!r}"
                        )
                    if actual_path != expected_path:
                        raise ValueError(
                            f"Task 7 cohort processed_path diverges for "
                            f"sample_id={sample_id!r}"
                        )

    all_traces: list[FinalDecisionTrace] = []
    for profile in profiles:
        spec = encoder_spec_fn(profile)
        capabilities = frozenset(getattr(spec, "capabilities", ()))
        attention_rule = str(getattr(spec, "attention_rule", ""))
        encoder, processor = encoder_factory(profile)
        if not isinstance(encoder, torch.nn.Module):
            raise ValueError("encoder_factory must return a torch encoder")
        ensure_ssl_encoder_attnlrp(
            encoder,
            attention_rule=attention_rule,
            capabilities=capabilities,
            patch_fn=patch_encoder_fn,
        )
        encoder.to(device)
        encoder.eval()

        profile_traces: list[FinalDecisionTrace] = []
        for source in sources:
            head = head_loader(paths.probe(profile, expected_layers, source))
            weight, bias = port_head_fn(head)
            model = SSLDetectorAD(
                encoder, expected_layers, np.asarray(weight), float(bias)
            ).to(device)
            model.eval()
            for target in targets:
                for row in persisted_cohorts[target].itertuples(index=False):
                    path = Path(row.processed_path)
                    if not path.is_file():
                        raise ValueError(
                            f"processed_path is invalid for {row.sample_id}: {path}"
                        )
                    loaded = audio_loader(path)
                    if not isinstance(loaded, tuple) or len(loaded) != 2:
                        raise ValueError("audio_loader must return (waveform, sample_rate)")
                    waveform, actual_rate = loaded
                    waveform = np.asarray(waveform)
                    if (
                        waveform.ndim != 1
                        or waveform.size == 0
                        or not np.isfinite(waveform).all()
                        or actual_rate != sample_rate
                    ):
                        raise ValueError(
                            f"canonical audio for {row.sample_id} must be finite mono "
                            f"at {sample_rate} Hz"
                        )
                    processed = processor(
                        [waveform],
                        sampling_rate=sample_rate,
                        return_tensors="pt",
                    )
                    if not isinstance(processed, Mapping) or "input_values" not in processed:
                        raise ValueError("processor must return input_values")
                    input_values = torch.as_tensor(processed["input_values"]).to(device)
                    attention_mask = processed.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = torch.as_tensor(attention_mask).to(device)
                    profile_traces.append(
                        trace_final_decision(
                            model,
                            input_values,
                            attention_mask=attention_mask,
                            profile=profile,
                            source=source,
                            target=target,
                            sample_id=row.sample_id,
                            expected_layers=expected_layers,
                            attention_rule=attention_rule,
                            capabilities=capabilities,
                        )
                    )

        transitions = layer_transition_metrics(profile_traces)
        layer_rows: list[dict[str, object]] = []
        for trace in profile_traces:
            for offset, layer in enumerate(trace.layer_indices):
                layer_rows.append(
                    {
                        "profile": trace.profile,
                        "source": trace.source,
                        "target": trace.target,
                        "sample_id": trace.sample_id,
                        "layer": layer,
                        "signed_sum": float(trace.signed_sum[offset]),
                        "absolute_mass": float(trace.absolute_mass[offset]),
                        "temporal_entropy": float(trace.temporal_entropy[offset]),
                        "temporal_mass": trace.temporal_mass[offset].tolist(),
                        "valid_length": trace.valid_lengths[offset],
                        "zero_mass": trace.zero_mass[offset],
                        "logit": trace.logit,
                    }
                )
        layer_frame = pd.DataFrame(layer_rows)

        def write(directory: Path) -> None:
            layer_frame.to_parquet(directory / "layer_relevance.parquet", index=False)
            transitions.to_csv(
                directory / "layer_transition_metrics.csv", index=False
            )

        publish_generation(paths.final_trace(profile), write, role="final_trace_profile")
        all_traces.extend(profile_traces)
    return layer_transition_metrics(all_traces)
