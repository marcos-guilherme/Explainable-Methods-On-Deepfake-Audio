"""Bootstrap estratificado e índices pré-registrados de emergência layer-wise."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapAucCI:
    """Resumo explícito do bootstrap percentil da ROC-AUC."""

    mean: float
    lower: float
    upper: float
    n_requested: int
    n_effective: int
    confidence: float


def _finite_1d(values: object, *, name: str, non_empty: bool = True) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if non_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_probability(value: object, *, name: str, inclusive_one: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    upper_valid = result <= 1.0 if inclusive_one else result < 1.0
    if not np.isfinite(result) or result <= 0.0 or not upper_valid:
        operator = "<=" if inclusive_one else "<"
        raise ValueError(f"{name} must satisfy 0 < {name} {operator} 1")
    return result


def _validated_chance(chance: object) -> float:
    if isinstance(chance, bool) or not isinstance(chance, Real):
        raise ValueError("chance must be a finite number")
    result = float(chance)
    if not np.isfinite(result):
        raise ValueError("chance must be finite")
    return result


def _validated_layer_indices(
    layer_indices: Sequence[int] | None,
    *,
    length: int,
) -> tuple[int, ...]:
    if layer_indices is None:
        if length > 12:
            raise ValueError("implicit layer indices must remain within 1 through 12")
        return tuple(range(1, length + 1))
    if isinstance(layer_indices, (str, bytes)):
        raise ValueError("layer_indices length must match the values")
    indices = tuple(layer_indices)
    if len(indices) != length:
        raise ValueError("layer_indices length must match the values")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in indices):
        raise ValueError("layer_indices must contain integers")
    normalized = tuple(int(value) for value in indices)
    if any(value < 1 or value > 12 for value in normalized):
        raise ValueError("layer_indices must contain layers from 1 through 12")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("layer_indices must be unique and strictly increasing")
    return normalized


def discriminative_onset(
    lower_ci: object,
    *,
    consecutive: int = 2,
    chance: float = 0.5,
    layer_indices: Sequence[int] | None = None,
) -> int | None:
    """Return the first layer starting a complete strict-above-chance run."""
    lower = _finite_1d(lower_ci, name="lower_ci")
    if isinstance(consecutive, bool) or not isinstance(consecutive, Integral):
        raise ValueError("consecutive must be an integer >= 1")
    consecutive = int(consecutive)
    if consecutive < 1:
        raise ValueError("consecutive must be >= 1")
    chance = _validated_chance(chance)
    indices = _validated_layer_indices(layer_indices, length=len(lower))

    above = lower > chance
    if consecutive > len(above):
        return None
    for start in range(len(above) - consecutive + 1):
        if bool(np.all(above[start : start + consecutive])):
            return indices[start]
    return None


def consolidation_layer(
    auc: object,
    *,
    fraction: float = 0.95,
    chance: float = 0.5,
    layer_indices: Sequence[int] | None = None,
) -> int | None:
    """Return the first layer sustaining the target fraction of final AUC gain."""
    curve = _finite_1d(auc, name="auc")
    fraction = _validated_probability(fraction, name="fraction", inclusive_one=True)
    chance = _validated_chance(chance)
    indices = _validated_layer_indices(layer_indices, length=len(curve))
    if curve[-1] <= chance:
        return None

    target = chance + fraction * (curve[-1] - chance)
    for position, value in enumerate(curve):
        if value >= target and bool(np.all(curve[position:] >= target)):
            return indices[position]
    return None


def bootstrap_auc_ci(
    scores: object,
    labels: object,
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapAucCI:
    """Compute a seeded class-stratified percentile interval for ROC-AUC."""
    score_array = _finite_1d(scores, name="scores")
    label_array = _finite_1d(labels, name="labels")
    if len(score_array) != len(label_array):
        raise ValueError("scores and labels must have the same length")
    if not set(np.unique(label_array).tolist()).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    if set(np.unique(label_array).tolist()) != {0.0, 1.0}:
        raise ValueError("labels must contain both classes")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral):
        raise ValueError("n_bootstrap must be a positive integer")
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    confidence = _validated_probability(
        confidence,
        name="confidence",
        inclusive_one=False,
    )

    labels_int = label_array.astype(np.int64)
    class_zero = np.flatnonzero(labels_int == 0)
    class_one = np.flatnonzero(labels_int == 1)
    rng = np.random.default_rng(int(seed))
    auc_values = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled_zero = rng.choice(class_zero, size=len(class_zero), replace=True)
        sampled_one = rng.choice(class_one, size=len(class_one), replace=True)
        sampled = np.concatenate((sampled_zero, sampled_one))
        auc_values[index] = roc_auc_score(labels_int[sampled], score_array[sampled])

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(auc_values, [alpha, 1.0 - alpha])
    return BootstrapAucCI(
        mean=float(np.mean(auc_values)),
        lower=float(lower),
        upper=float(upper),
        n_requested=n_bootstrap,
        n_effective=int(len(auc_values)),
        confidence=confidence,
    )


def _identity_seed(
    base_seed: int,
    profile: object,
    source: object,
    target: object,
    layer: int,
) -> int:
    payload = json.dumps(
        [int(base_seed), str(profile), str(source), str(target), int(layer)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_summary_parameters(
    *,
    layer_indices: Sequence[int],
    n_bootstrap: int,
    seed: int,
    confidence: float,
    chance: float,
    fraction: float,
    consecutive: int,
) -> tuple[tuple[int, ...], int, int, float, float, float, int]:
    try:
        number_of_layers = len(layer_indices)
    except TypeError as exc:
        raise ValueError("layer_indices must be a non-empty sequence") from exc
    if number_of_layers == 0:
        raise ValueError("layer_indices must be non-empty")
    indices = _validated_layer_indices(layer_indices, length=number_of_layers)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral):
        raise ValueError("n_bootstrap must be a positive integer")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be > 0")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    confidence = _validated_probability(
        confidence,
        name="confidence",
        inclusive_one=False,
    )
    chance = _validated_chance(chance)
    fraction = _validated_probability(fraction, name="fraction", inclusive_one=True)
    if isinstance(consecutive, bool) or not isinstance(consecutive, Integral):
        raise ValueError("consecutive must be an integer >= 1")
    if int(consecutive) < 1:
        raise ValueError("consecutive must be >= 1")
    return (
        indices,
        int(n_bootstrap),
        int(seed),
        confidence,
        chance,
        fraction,
        int(consecutive),
    )


def _validated_precomputed_intervals(group: pd.DataFrame) -> np.ndarray:
    lower = _finite_1d(group["auc_ci_lower"].to_numpy(), name="auc_ci_lower")
    upper = _finite_1d(group["auc_ci_upper"].to_numpy(), name="auc_ci_upper")
    if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(lower > upper):
        raise ValueError("confidence interval limits must satisfy 0 <= lower <= upper <= 1")
    return lower


def summarize_emergence(
    performance: pd.DataFrame,
    *,
    layer_indices: Sequence[int] = tuple(range(1, 13)),
    n_bootstrap: int,
    seed: int = 42,
    confidence: float = 0.95,
    chance: float = 0.5,
    fraction: float = 0.95,
    consecutive: int = 2,
) -> pd.DataFrame:
    """Summarize each exact ``(profile, source, target)`` layer-wise cell.

    Input rows must provide either ``scores`` and ``labels`` arrays or ready
    ``auc_ci_lower`` and ``auc_ci_upper`` limits.
    """
    if not isinstance(performance, pd.DataFrame):
        raise TypeError("performance must be a pandas DataFrame")
    required = {
        "profile",
        "layer",
        "source",
        "target",
        "auc",
        "corpus_shift",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"performance is missing required columns: {sorted(missing)}")
    (
        layers,
        n_bootstrap,
        seed,
        confidence,
        chance,
        fraction,
        consecutive,
    ) = _validate_summary_parameters(
        layer_indices=layer_indices,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
        chance=chance,
        fraction=fraction,
        consecutive=consecutive,
    )
    has_samples = {"scores", "labels"}.issubset(performance.columns)
    has_intervals = {"auc_ci_lower", "auc_ci_upper"}.issubset(performance.columns)
    if not has_samples and not has_intervals:
        raise ValueError(
            "performance must contain scores/labels or precomputed confidence intervals"
        )

    rows: list[dict[str, object]] = []
    keys = ["profile", "source", "target"]
    grouped = performance.groupby(keys, sort=True, dropna=False)
    for (profile, source, target), unsorted_group in grouped:
        raw_layers = unsorted_group["layer"].tolist()
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in raw_layers
        ):
            raise ValueError("performance layer values must be integers")
        group = unsorted_group.sort_values("layer", kind="stable").reset_index(drop=True)
        observed_layers = tuple(group["layer"].tolist())
        if len(group) != len(layers) or observed_layers != layers:
            raise ValueError(
                f"cell {(profile, source, target)!r} must contain exactly one row "
                f"for each expected layer {layers}"
            )

        auc = _finite_1d(group["auc"].to_numpy(), name="auc")
        if np.any(auc < 0.0) or np.any(auc > 1.0):
            raise ValueError("auc values must be between 0 and 1")
        shift_values = group["corpus_shift"].tolist()
        if (
            any(not isinstance(value, (bool, np.bool_)) for value in shift_values)
            or len(set(bool(value) for value in shift_values)) != 1
        ):
            raise ValueError("corpus_shift must be a consistent boolean within each cell")
        corpus_shift = bool(shift_values[0])
        if corpus_shift != (source != target):
            raise ValueError("corpus_shift is inconsistent with source and target")

        if has_samples:
            lower_values = []
            for row in group.itertuples(index=False):
                interval = bootstrap_auc_ci(
                    row.scores,
                    row.labels,
                    n_bootstrap=n_bootstrap,
                    seed=_identity_seed(seed, profile, source, target, int(row.layer)),
                    confidence=confidence,
                )
                lower_values.append(interval.lower)
            lower = np.asarray(lower_values, dtype=np.float64)
        else:
            lower = _validated_precomputed_intervals(group)

        onset = discriminative_onset(
            lower,
            consecutive=consecutive,
            chance=chance,
            layer_indices=layers,
        )
        consolidation = consolidation_layer(
            auc,
            fraction=fraction,
            chance=chance,
            layer_indices=layers,
        )
        rows.append(
            {
                "profile": profile,
                "source": source,
                "target": target,
                "onset": onset,
                "consolidation": consolidation,
                "final_auc": float(auc[-1]),
                "chance": chance,
                "fraction": fraction,
                "consecutive": consecutive,
                "bootstrap_confidence": confidence,
                "bootstrap_n": n_bootstrap,
                "corpus_shift": corpus_shift,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["onset"] = pd.array(result["onset"], dtype="Int64")
        result["consolidation"] = pd.array(result["consolidation"], dtype="Int64")
    return result
