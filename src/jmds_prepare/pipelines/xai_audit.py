"""Audit helpers for materialized XAI samples without publication."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from jmds_prepare.core.xai_sample import XaiSample

SILENCE_THRESHOLD = 1e-4
_EXPECTED_SAMPLE_RATE = 16_000
_EXPECTED_CHANNELS = 1
_EXPECTED_SUBTYPE = "PCM_16"


@dataclass(frozen=True)
class XaiSampleAudit:
    """Measured technical properties of one processed XAI WAV."""

    sample_id: str
    duration_seconds: float
    rms: float
    peak: float
    silence_fraction: float
    silence_threshold: float = SILENCE_THRESHOLD


def audit_processed_sample(path: Path, *, sample_id: str) -> XaiSampleAudit:
    """Measure one processed WAV and verify the canonical PCM16 mono 16 kHz format."""
    path = Path(path)
    _validate_processed_format(path)
    samples, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"Processed audio must be mono: {path}")
    mono = samples[:, 0]
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    silence_fraction = (
        float(np.mean(np.abs(mono) <= SILENCE_THRESHOLD)) if mono.size else 1.0
    )
    return XaiSampleAudit(
        sample_id=sample_id,
        duration_seconds=mono.size / sample_rate,
        rms=rms,
        peak=peak,
        silence_fraction=silence_fraction,
    )


def summarize_sample_audits(
    samples: tuple[XaiSample, ...] | list[XaiSample],
    audits: tuple[XaiSampleAudit, ...] | list[XaiSampleAudit],
) -> dict[str, Any]:
    """Return a JSON-safe summary grouped by corpus, label and role."""
    if len(samples) != len(audits):
        raise ValueError("samples and audits must have the same length")

    grouped: dict[tuple[str, int, str], list[XaiSampleAudit]] = {}
    for sample, audit in zip(samples, audits, strict=True):
        if sample.sample_id != audit.sample_id:
            raise ValueError(
                f"Audit sample_id mismatch: {audit.sample_id} != {sample.sample_id}"
            )
        key = (sample.corpus, sample.label, sample.role)
        grouped.setdefault(key, []).append(audit)

    groups: list[dict[str, Any]] = []
    for (corpus, label, role), items in sorted(grouped.items()):
        groups.append(
            {
                "corpus": corpus,
                "label": label,
                "role": role,
                "count": len(items),
                "duration_seconds": _numeric_summary(
                    [item.duration_seconds for item in items]
                ),
                "rms": _numeric_summary([item.rms for item in items]),
                "peak": _numeric_summary([item.peak for item in items]),
                "silence_fraction": _numeric_summary(
                    [item.silence_fraction for item in items]
                ),
            }
        )

    return {
        "sample_count": len(samples),
        "silence_definition": {
            "threshold": SILENCE_THRESHOLD,
            "description": (
                "Fraction of samples whose absolute amplitude is less than or "
                "equal to the threshold."
            ),
        },
        "groups": groups,
    }


def validate_processed_wav_format(path: Path) -> None:
    """Verify one WAV matches the canonical PCM16 mono 16 kHz contract."""
    _validate_processed_format(path)


def _validate_processed_format(path: Path) -> None:
    try:
        info = sf.info(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not inspect processed audio: {path}") from exc
    if (
        info.format != "WAV"
        or info.subtype != _EXPECTED_SUBTYPE
        or info.channels != _EXPECTED_CHANNELS
        or info.samplerate != _EXPECTED_SAMPLE_RATE
    ):
        raise ValueError(
            "Processed audio must be WAV PCM_16 mono 16 kHz: "
            f"{path} ({info.format}/{info.subtype}, "
            f"{info.channels}ch @ {info.samplerate} Hz)"
        )


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(finite),
        "max": max(finite),
        "mean": float(np.mean(finite)),
    }
