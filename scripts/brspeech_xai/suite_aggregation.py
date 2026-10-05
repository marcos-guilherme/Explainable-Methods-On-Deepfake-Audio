"""Fail-closed scientific aggregation for the layer-wise encoder suite."""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .layerwise_paths import LayerwiseSuitePaths, publish_generation, resolve_active_generation


LANGUAGES = ("eng", "por", "zho")
LAYERS = tuple(range(1, 13))
AGGREGATE_SCHEMA_VERSION = 1
TABLE_NAMES = (
    "layerwise_performance.csv",
    "emergence_layers.csv",
    "language_shift_by_layer.csv",
    "encoder_relevance_agreement.csv",
    "spectral_divergence_by_layer.csv",
    "final_decision_reorganization.csv",
)
CLAIMS = {
    "diagonal": "within-language evaluation",
    "external_validation": "off-diagonal external validation under corpus shift",
    "causal_limit": "language differences are not identified as causal language effects",
    "signed_relevance": "directional relevance; never used as a probability distribution",
    "absolute_normalized": "non-negative per-sample band mass normalized to sum to one",
}
_TABLE_SCIENTIFIC_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "layerwise_performance.csv": {
        "statistic": "per_cell_roc_auc_and_accuracy",
        "unit": "roc_auc_and_fraction",
    },
    "emergence_layers.csv": {
        "statistic": "discriminative_onset_and_consolidation_layer",
        "unit": "layer_index_and_roc_auc",
    },
    "language_shift_by_layer.csv": {
        "statistic": "target_diagonal_referenced_auc_difference",
        "unit": "roc_auc",
    },
    "encoder_relevance_agreement.csv": {
        "statistic": "mean_of_per_sample_encoder_agreement",
        "unit": "metric_specific",
    },
    "spectral_divergence_by_layer.csv": {
        "statistic": "mean_distribution_language_shift",
        "unit": "metric_specific",
    },
    "final_decision_reorganization.csv": {
        "statistic": "transition_metric_mean",
        "unit": "metric_specific",
    },
}


def _table_uncertainty_metadata(name: str) -> dict[str, object]:
    if name == "emergence_layers.csv":
        return {
            "method": "stratified_bootstrap_auc_percentile",
            "confidence_column": "bootstrap_confidence",
            "bootstrap": True,
        }
    if name in {
        "encoder_relevance_agreement.csv",
        "final_decision_reorganization.csv",
    }:
        return {
            "method": "normal_95_sem",
            "confidence": 0.95,
            "bootstrap": False,
        }
    return {"method": "none", "bootstrap": False}


def _table_scientific_metadata(name: str) -> dict[str, object]:
    if name not in _TABLE_SCIENTIFIC_CONTRACTS:
        raise ValueError(f"unknown aggregate table {name!r}")
    return {
        **_table_uncertainty_metadata(name),
        **_TABLE_SCIENTIFIC_CONTRACTS[name],
    }


def _annotate_emergence_scientific_contract(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = _table_scientific_metadata("emergence_layers.csv")
    annotated = frame.copy()
    annotated["ci_method"] = metadata["method"]
    annotated["ci_confidence"] = annotated["bootstrap_confidence"]
    annotated["statistic"] = metadata["statistic"]
    annotated["unit"] = metadata["unit"]
    return annotated


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector with at least two bins")
    return array


def _edge_map(
    mapping: Mapping[str, object], target: str, profiles: Sequence[str]
) -> dict[str, np.ndarray]:
    if target not in mapping:
        raise ValueError(f"band edges are missing target {target!r}")
    raw = mapping[target]
    if isinstance(raw, Mapping):
        if set(raw) != set(profiles):
            raise ValueError(f"band edges for {target!r} must cover every profile")
        result = {
            profile: _finite_vector(raw[profile], name="band edges")
            for profile in profiles
        }
    else:
        edges = _finite_vector(raw, name="band edges")
        result = {profile: edges for profile in profiles}
    reference = result[profiles[0]]
    if (
        reference[0] < 0
        or np.any(np.diff(reference) <= 0)
        or any(
            edges.shape != reference.shape or not np.array_equal(edges, reference)
            for edges in result.values()
        )
    ):
        raise ValueError(f"mismatched or invalid band edges for target {target!r}")
    return result


def _cosine(left: np.ndarray, right: np.ndarray, *, name: str) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError(f"{name} has zero or invalid mass")
    value = float(np.dot(left, right) / denominator)
    if not np.isfinite(value):
        raise ValueError(f"{name} produced a non-finite cosine")
    return value


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _rank(left), _rank(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        raise ValueError("signed relevance is insufficient for Spearman correlation")
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("signed relevance produced non-finite Spearman correlation")
    return value


def _probability(value: np.ndarray, *, name: str) -> np.ndarray:
    if np.any(value < 0):
        raise ValueError(f"{name} must be non-negative")
    total = float(value.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"{name} has zero or invalid mass")
    normalized = value / total
    if not np.isclose(normalized.sum(), 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"{name} does not sum to one")
    return normalized


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = _probability(left, name="absolute normalized relevance")
    right = _probability(right, name="absolute normalized relevance")
    midpoint = (left + right) / 2.0

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    value = 0.5 * (kl(left) + kl(right))
    if not np.isfinite(value):
        raise ValueError("absolute normalized relevance produced non-finite Jensen-Shannon")
    return value


def _validate_relevance_samples(samples: pd.DataFrame) -> pd.DataFrame:
    required = {
        "profile",
        "source",
        "target",
        "layer",
        "sample_id",
        "y_true",
        "prediction",
        "band_signed",
        "band_abs_normalized",
    }
    if not isinstance(samples, pd.DataFrame) or not required.issubset(samples.columns):
        raise ValueError(f"relevance samples lack columns {sorted(required - set(samples.columns))}")
    identity = ["profile", "source", "target", "layer", "sample_id"]
    if samples.empty or samples.duplicated(identity).any():
        raise ValueError("relevance samples contain an empty or duplicate complete identity")
    if any(samples[column].isna().any() for column in identity):
        raise ValueError("relevance samples contain incomplete identities")
    if not set(samples["y_true"].tolist()).issubset({0, 1}) or not set(
        samples["prediction"].tolist()
    ).issubset({0, 1}):
        raise ValueError("relevance classes must be binary")
    return samples


def encoder_relevance_agreement(
    samples: pd.DataFrame,
    *,
    band_edges_by_target: Mapping[str, object],
) -> pd.DataFrame:
    """Compare encoders by ID/true label without requiring equal predictions."""
    samples = _validate_relevance_samples(samples)
    profiles = tuple(sorted(samples["profile"].unique()))
    columns = [
        "profile_a",
        "profile_b",
        "source",
        "target",
        "layer",
        "conditioning",
        "class_value",
        "pred_a",
        "pred_b",
        "prediction_relation",
        "n",
        "spearman_signed_mean",
        "spearman_signed_ci_low",
        "spearman_signed_ci_high",
        "cosine_signed_mean",
        "cosine_signed_ci_low",
        "cosine_signed_ci_high",
        "cosine_absolute_normalized_mean",
        "cosine_absolute_normalized_ci_low",
        "cosine_absolute_normalized_ci_high",
        "jensen_shannon_absolute_normalized_mean",
        "jensen_shannon_absolute_normalized_ci_low",
        "jensen_shannon_absolute_normalized_ci_high",
        "ci_method",
        "ci_confidence",
        "statistic",
        "unit",
        "spearman_signed_unit",
        "cosine_signed_unit",
        "cosine_absolute_normalized_unit",
        "jensen_shannon_absolute_normalized_unit",
    ]
    if len(profiles) < 2:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_keys = ["source", "target", "layer"]
    expected_cells = set(
        samples.loc[samples["profile"] == profiles[0], group_keys]
        .itertuples(index=False, name=None)
    )
    for profile in profiles[1:]:
        observed = set(
            samples.loc[samples["profile"] == profile, group_keys]
            .itertuples(index=False, name=None)
        )
        if observed != expected_cells:
            raise ValueError("profiles do not contain the same relevance cells")
    for profile_a, profile_b in combinations(profiles, 2):
        for source, target, layer in sorted(expected_cells):
            _edge_map(band_edges_by_target, str(target), profiles)
            left = samples.loc[
                (samples["profile"] == profile_a)
                & (samples["source"] == source)
                & (samples["target"] == target)
                & (samples["layer"] == layer)
            ]
            right = samples.loc[
                (samples["profile"] == profile_b)
                & (samples["source"] == source)
                & (samples["target"] == target)
                & (samples["layer"] == layer)
            ]
            if set(left["sample_id"]) != set(right["sample_id"]):
                raise ValueError("encoder relevance sample_id sets do not match")
            paired = left.merge(
                right,
                on=["source", "target", "layer", "sample_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_a", "_b"),
                sort=True,
            )
            if not np.array_equal(
                paired["y_true_a"].to_numpy(),
                paired["y_true_b"].to_numpy(),
            ):
                raise ValueError("encoder relevance true labels do not match")
            paired["prediction_relation"] = np.where(
                (paired["prediction_a"] == 1) & (paired["prediction_b"] == 1),
                "both_spoof",
                np.where(
                    (paired["prediction_a"] == 0) & (paired["prediction_b"] == 0),
                    "both_bonafide",
                    "disagree",
                ),
            )
            groups = [
                (
                    "true_label",
                    int(class_value),
                    "all",
                    "all",
                    "all",
                    group,
                    True,
                )
                for class_value, group in paired.groupby("y_true_a", sort=True)
            ]
            groups.extend(
                (
                    "prediction_relation",
                    relation,
                    int(pred_a),
                    int(pred_b),
                    relation,
                    group,
                    False,
                )
                for (pred_a, pred_b, relation), group in paired.groupby(
                    ["prediction_a", "prediction_b", "prediction_relation"],
                    sort=True,
                )
            )
            for (
                conditioning,
                class_value,
                pred_a,
                pred_b,
                relation,
                group,
                require_two,
            ) in groups:
                if require_two and len(group) < 2:
                    raise ValueError(
                        "encoder agreement requires at least 2 samples "
                        "per true-label group"
                    )
                metrics = {
                    "spearman_signed": [],
                    "cosine_signed": [],
                    "cosine_absolute_normalized": [],
                    "jensen_shannon_absolute_normalized": [],
                }
                for item in group.itertuples(index=False):
                    sa = _finite_vector(item.band_signed_a, name="signed relevance")
                    sb = _finite_vector(item.band_signed_b, name="signed relevance")
                    aa = _finite_vector(
                        item.band_abs_normalized_a,
                        name="absolute normalized relevance",
                    )
                    ab = _finite_vector(
                        item.band_abs_normalized_b,
                        name="absolute normalized relevance",
                    )
                    if not (sa.shape == sb.shape == aa.shape == ab.shape):
                        raise ValueError("encoder relevance bin counts do not match")
                    edges = _edge_map(
                        band_edges_by_target, str(target), profiles
                    )[profile_a]
                    if len(edges) != len(sa) + 1:
                        raise ValueError("band edges do not match relevance bins")
                    aa = _probability(aa, name="absolute normalized relevance")
                    ab = _probability(ab, name="absolute normalized relevance")
                    metrics["spearman_signed"].append(_spearman(sa, sb))
                    metrics["cosine_signed"].append(
                        _cosine(sa, sb, name="signed relevance")
                    )
                    metrics["cosine_absolute_normalized"].append(
                        _cosine(
                            aa,
                            ab,
                            name="absolute normalized relevance",
                        )
                    )
                    metrics["jensen_shannon_absolute_normalized"].append(
                        _jensen_shannon(aa, ab)
                    )
                row = {
                    "profile_a": profile_a,
                    "profile_b": profile_b,
                    "source": source,
                    "target": target,
                    "layer": int(layer),
                    "conditioning": conditioning,
                    "class_value": class_value,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "prediction_relation": relation,
                    "n": len(group),
                    "ci_method": "normal_95_sem",
                    "ci_confidence": 0.95,
                    "statistic": "mean_of_per_sample_encoder_agreement",
                    "unit": "metric_specific",
                    "spearman_signed_unit": "unitless",
                    "cosine_signed_unit": "unitless",
                    "cosine_absolute_normalized_unit": "unitless",
                    "jensen_shannon_absolute_normalized_unit": "bits",
                }
                for name, values in metrics.items():
                    mean, low, high = _mean_ci(pd.Series(values))
                    row[f"{name}_mean"] = mean
                    row[f"{name}_ci_low"] = low
                    row[f"{name}_ci_high"] = high
                rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(
        columns[:7], kind="mergesort"
    ).reset_index(drop=True)


def _mean_ci(values: pd.Series) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate metric values must be finite and non-empty")
    mean = float(array.mean())
    sem = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def final_decision_reorganization(
    transitions: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "profile",
        "source",
        "target",
        "sample_id",
        "previous_layer",
        "current_layer",
        "similarity",
        "normalized_l1_change",
        "absolute_mass",
        "temporal_entropy",
    }
    if not isinstance(transitions, pd.DataFrame) or not required.issubset(
        transitions.columns
    ):
        raise ValueError("transition table has an incompatible schema")
    label_key = ["profile", "source", "target", "sample_id"]
    if not {*label_key, "y_true", "prediction"}.issubset(labels.columns):
        raise ValueError("trace labels have an incompatible schema")
    if (
        labels.empty
        or labels.duplicated(label_key).any()
        or any(labels[column].isna().any() for column in label_key)
        or any(
            not isinstance(value, str) or not value
            for column in label_key
            for value in labels[column]
        )
    ):
        raise ValueError("trace labels contain incomplete or duplicate identities")
    if not set(labels["y_true"].tolist()).issubset({0, 1}) or not set(
        labels["prediction"].tolist()
    ).issubset({0, 1}):
        raise ValueError("trace labels must contain binary classes")
    key = ["profile", "source", "target", "sample_id"]
    if any(
        not isinstance(value, str) or not value
        for column in key
        for value in transitions[column]
    ):
        raise ValueError("transition identities must be complete")
    if transitions.duplicated(key + ["previous_layer", "current_layer"]).any():
        raise ValueError("transition table contains duplicate identities")
    expected = {(layer, layer + 1) for layer in range(1, 12)}
    for identity, group in transitions.groupby(key, sort=True):
        observed = set(
            group[["previous_layer", "current_layer"]].itertuples(
                index=False, name=None
            )
        )
        if len(group) != 11 or observed != expected:
            raise ValueError(f"trace identity {identity!r} must contain exactly 11 transitions")
    transition_identities = set(
        transitions[key].itertuples(index=False, name=None)
    )
    label_identities = set(labels[key].itertuples(index=False, name=None))
    if transition_identities != label_identities:
        raise ValueError("trace labels and transition identity sets do not match")
    joined = transitions.merge(
        labels,
        on=key,
        validate="many_to_one",
        how="left",
    )
    rows: list[dict[str, object]] = []
    base = ["profile", "source", "target", "previous_layer", "current_layer"]
    groupings = [("all", None), ("y_true", "y_true"), ("prediction", "prediction")]
    metrics = (
        "similarity",
        "normalized_l1_change",
        "absolute_mass",
        "temporal_entropy",
    )
    for conditioning, column in groupings:
        grouped = joined.groupby(
            base if column is None else base + [column], sort=True, dropna=False
        )
        for identity, group in grouped:
            identity = identity if isinstance(identity, tuple) else (identity,)
            class_value = "all" if column is None else int(identity[-1])
            fixed = identity if column is None else identity[:-1]
            row = dict(zip(base, fixed))
            row.update(
                {
                    "conditioning": conditioning,
                    "class_value": class_value,
                    "n": int(group["sample_id"].nunique()),
                    "ci_method": "normal_95_sem",
                    "ci_confidence": 0.95,
                    "statistic": "transition_metric_mean",
                    "unit": "metric_specific",
                    "similarity_unit": "unitless",
                    "normalized_l1_change_unit": "unitless_l1",
                    "absolute_mass_unit": "attribution_mass",
                    "temporal_entropy_unit": "nats",
                }
            )
            for metric in metrics:
                mean, low, high = _mean_ci(group[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        base + ["conditioning", "class_value"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_predictions_against_xai(
    predictions: pd.DataFrame,
    xai_samples: pd.DataFrame,
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    expected_n: int,
) -> pd.DataFrame:
    required_predictions = {"sample_id", "y_true", "score", "prediction"}
    required_xai = {
        "profile",
        "layer",
        "source",
        "target",
        "sample_id",
        "y_true",
        "prediction",
    }
    if (
        not isinstance(predictions, pd.DataFrame)
        or not required_predictions.issubset(predictions.columns)
        or not isinstance(xai_samples, pd.DataFrame)
        or not required_xai.issubset(xai_samples.columns)
    ):
        raise ValueError("prediction or XAI sample schema is incompatible")
    expected_identity = (profile, layer, source, target)
    observed_identity = set(
        xai_samples[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    )
    if observed_identity != {expected_identity}:
        raise ValueError("XAI samples do not match the requested cell identity")
    prediction_ids = predictions["sample_id"].tolist()
    xai_ids = xai_samples["sample_id"].tolist()
    if (
        isinstance(expected_n, bool)
        or not isinstance(expected_n, (int, np.integer))
        or expected_n < 1
        or len(predictions) != expected_n
        or len(xai_samples) != expected_n
        or any(not isinstance(value, str) or not value for value in prediction_ids)
        or any(not isinstance(value, str) or not value for value in xai_ids)
        or len(prediction_ids) != len(set(prediction_ids))
        or len(xai_ids) != len(set(xai_ids))
        or set(prediction_ids) != set(xai_ids)
    ):
        raise ValueError(
            "predictions must have unique non-empty IDs and exactly match the XAI cohort"
        )
    aligned = predictions.set_index("sample_id").loc[xai_ids].reset_index()
    if (
        not set(aligned["y_true"].tolist()).issubset({0, 1})
        or not set(aligned["prediction"].tolist()).issubset({0, 1})
        or not np.isfinite(aligned["score"].to_numpy(dtype=np.float64)).all()
        or
        not np.array_equal(
            aligned["y_true"].to_numpy(), xai_samples["y_true"].to_numpy()
        )
        or not np.array_equal(
            aligned["prediction"].to_numpy(),
            xai_samples["prediction"].to_numpy(),
        )
    ):
        raise ValueError("prediction labels do not match the XAI cohort by sample_id")
    return aligned


def spectral_divergence_by_layer(
    samples: pd.DataFrame,
    *,
    band_edges_by_target: Mapping[str, object],
) -> pd.DataFrame:
    """Compare mean target-language spectra against the source diagonal."""
    samples = _validate_relevance_samples(samples)
    rows: list[dict[str, object]] = []
    base = ["profile", "source", "layer", "y_true"]
    profiles = tuple(sorted(samples["profile"].unique()))
    for identity, group in samples.groupby(base, sort=True):
        profile, source, layer, true_label = identity
        diagonal = group.loc[group["target"] == source]
        if diagonal.empty:
            raise ValueError(
                f"spectral language comparison lacks diagonal target {source!r}"
            )
        reference_edges = _edge_map(
            band_edges_by_target, str(source), profiles
        )[str(profile)]
        reference_values = np.stack(
            [
                _probability(
                    _finite_vector(value, name="absolute normalized relevance"),
                    name="absolute normalized relevance",
                )
                for value in diagonal["band_abs_normalized"]
            ]
        )
        reference_mean = _probability(
            reference_values.mean(axis=0),
            name="mean diagonal absolute normalized relevance",
        )
        if len(reference_edges) != len(reference_mean) + 1:
            raise ValueError("diagonal band edges do not match relevance bins")
        for target, comparison in group.loc[group["target"] != source].groupby(
            "target", sort=True
        ):
            comparison_edges = _edge_map(
                band_edges_by_target, str(target), profiles
            )[str(profile)]
            if not np.array_equal(reference_edges, comparison_edges):
                raise ValueError("language targets use mismatched band edges")
            comparison_values = np.stack(
                [
                    _probability(
                        _finite_vector(value, name="absolute normalized relevance"),
                        name="absolute normalized relevance",
                    )
                    for value in comparison["band_abs_normalized"]
                ]
            )
            comparison_mean = _probability(
                comparison_values.mean(axis=0),
                name="mean off-diagonal absolute normalized relevance",
            )
            cosine = _cosine(
                reference_mean,
                comparison_mean,
                name="mean target-language relevance",
            )
            rows.append(
                {
                    "profile": profile,
                    "source": source,
                    "layer": int(layer),
                    "true_label": int(true_label),
                    "reference_target": source,
                    "comparison_target": target,
                    "n_reference": len(diagonal),
                    "n_comparison": len(comparison),
                    "jensen_shannon": _jensen_shannon(
                        reference_mean, comparison_mean
                    ),
                    "cosine_similarity": cosine,
                    "cosine_distance": 1.0 - cosine,
                    "corpus_shift": True,
                    "statistic": "mean_distribution_language_shift",
                    "unit": "metric_specific",
                }
            )
    columns = [
        "profile",
        "source",
        "layer",
        "true_label",
        "reference_target",
        "comparison_target",
        "n_reference",
        "n_comparison",
        "jensen_shannon",
        "cosine_similarity",
        "cosine_distance",
        "corpus_shift",
        "statistic",
        "unit",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        columns[:6], kind="mergesort"
    ).reset_index(drop=True)


def language_shift_by_layer(performance: pd.DataFrame) -> pd.DataFrame:
    required = {"profile", "layer", "source", "target", "auc"}
    if not required.issubset(performance.columns):
        raise ValueError("performance table has an incompatible schema")
    keys = ["profile", "layer", "source", "target"]
    if performance.duplicated(keys).any():
        raise ValueError("performance contains duplicate cells")
    diagonal = performance.loc[
        performance["source"] == performance["target"],
        ["profile", "layer", "target", "auc"],
    ].rename(columns={"auc": "target_diagonal_auc"})
    result = performance.merge(
        diagonal,
        on=["profile", "layer", "target"],
        how="left",
        validate="many_to_one",
    )
    if result["target_diagonal_auc"].isna().any():
        raise ValueError("performance lacks a target diagonal baseline")
    result["delta_auc"] = result["auc"] - result["target_diagonal_auc"]
    result["diagonal"] = result["source"] == result["target"]
    result["external_validation"] = ~result["diagonal"]
    result["corpus_shift"] = ~result["diagonal"]
    return result[
        keys
        + [
            "auc",
            "target_diagonal_auc",
            "delta_auc",
            "diagonal",
            "external_validation",
            "corpus_shift",
        ]
    ].sort_values(keys, kind="mergesort").reset_index(drop=True)


def run_classical_audit(
    *,
    enabled: bool,
    output_dir: Path,
    adapter: Callable[..., Mapping[str, Path]] | None,
) -> list[dict[str, object]]:
    if type(enabled) is not bool:
        raise TypeError("enabled must be a strict boolean")
    if not enabled:
        return []
    if not callable(adapter):
        raise ValueError("classical audit requires a production adapter")
    records = []
    for language in LANGUAGES:
        cell_dir = Path(output_dir) / "classical_audit" / f"{language}_to_{language}"
        artifacts = adapter(
            source=language,
            target=language,
            layer=12,
            output_dir=cell_dir,
        )
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError("classical audit adapter returned no artifacts")
        resolved = {}
        for name, raw_path in artifacts.items():
            path = Path(raw_path).resolve(strict=True)
            if not path.is_file() or not path.is_relative_to(Path(output_dir).resolve()):
                raise ValueError("classical audit artifact is invalid or outside aggregates")
            resolved[name] = {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        records.append(
            {
                "source": language,
                "target": language,
                "layer": 12,
                "artifacts": resolved,
            }
        )
    return records


def _validated_marker(root: Path, stage_id: str, expected_names: set[str]) -> dict:
    path = root / ".state" / f"{stage_id.replace(':', '__')}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("contract_version") != "encoder-suite-v2"
            or payload.get("stage_id") != stage_id
            or not isinstance(payload.get("fingerprint"), str)
        ):
            raise ValueError("marker schema or identity mismatch")
        artifacts = payload["artifacts"]
        if {item["name"] for item in artifacts} != expected_names:
            raise ValueError("marker artifact contract mismatch")
        canonical = root.resolve()
        for item in artifacts:
            artifact = Path(item["path"]).resolve(strict=True)
            if (
                not artifact.is_file()
                or not artifact.is_relative_to(canonical)
                or _sha256(artifact) != item["sha256"]
            ):
                raise ValueError("marker artifact hash mismatch")
        return payload
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid upstream marker for {stage_id}") from exc


def _load_authoritative_cohorts(
    root: str | Path,
    *,
    targets: Sequence[str] = LANGUAGES,
) -> dict[str, pd.DataFrame]:
    root = Path(root)
    paths = LayerwiseSuitePaths(root)
    cohorts: dict[str, pd.DataFrame] = {}
    for target in targets:
        _validated_marker(
            root,
            f"suite:cohort:{target}",
            {"cohort", "metadata"},
        )
        cohort_path = paths.suite_cohort(target)
        metadata_path = paths.suite_cohort_metadata(target)
        try:
            cohort = pd.read_parquet(cohort_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"authoritative suite cohort is unreadable for target {target!r}"
            ) from exc
        required = {"sample_id", "label", "processed_path"}
        ids = cohort["sample_id"].tolist() if "sample_id" in cohort else []
        if (
            not required.issubset(cohort.columns)
            or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids))
            or not set(cohort["label"].tolist()).issubset({0, 1})
            or any(
                not isinstance(value, (str, Path)) or not str(value)
                for value in cohort["processed_path"]
            )
            or metadata.get("schema_version") != 1
            or metadata.get("target") != target
            or metadata.get("sample_ids") != ids
        ):
            raise ValueError(
                f"authoritative suite cohort schema/identity is invalid for {target!r}"
            )
        normalized = cohort.loc[
            :, ["sample_id", "label", "processed_path"]
        ].copy()
        normalized["processed_path"] = normalized["processed_path"].map(str)
        cohorts[target] = normalized
    return cohorts


def _cohort_identity_map(
    frame: pd.DataFrame,
    *,
    label_column: str,
) -> dict[str, tuple[int, str]]:
    labels = frame[label_column].to_numpy()
    paths = frame["processed_path"].tolist()
    if (
        labels.dtype.kind not in {"i", "u"}
        or not set(labels.tolist()).issubset({0, 1})
        or any(
            not isinstance(value, (str, Path)) or not str(value)
            for value in paths
        )
    ):
        raise ValueError("cohort labels or processed_path values are invalid")
    return {
        str(row.sample_id): (
            int(getattr(row, label_column)),
            str(row.processed_path),
        )
        for row in frame.itertuples(index=False)
    }


def _validate_cell_against_authoritative_cohort(
    predictions: pd.DataFrame,
    xai_samples: pd.DataFrame,
    authority: pd.DataFrame,
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
) -> pd.DataFrame:
    prediction_columns = {
        "sample_id",
        "y_true",
        "processed_path",
        "score",
        "prediction",
    }
    xai_columns = {
        "profile",
        "source",
        "target",
        "layer",
        "sample_id",
        "y_true",
        "processed_path",
        "prediction",
    }
    if (
        not prediction_columns.issubset(predictions.columns)
        or not xai_columns.issubset(xai_samples.columns)
    ):
        raise ValueError("cell artifacts lack authoritative cohort columns")
    observed_identity = set(
        xai_samples[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    )
    if observed_identity != {(profile, layer, source, target)}:
        raise ValueError("cell XAI identity does not match its planned cell")
    for name, frame in (("predictions", predictions), ("sample_relevance", xai_samples)):
        ids = frame["sample_id"].tolist()
        if (
            not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"{name} has invalid or duplicate sample_id values")
    authoritative = _cohort_identity_map(authority, label_column="label")
    prediction_by_id = predictions.set_index("sample_id", drop=False)
    missing = set(authoritative).difference(prediction_by_id.index)
    if missing:
        raise ValueError(
            "predictions are missing authoritative suite cohort IDs: "
            f"{sorted(missing)}"
        )
    filtered_predictions = prediction_by_id.loc[list(authoritative)].reset_index(
        drop=True
    )
    predicted = _cohort_identity_map(
        filtered_predictions, label_column="y_true"
    )
    explained = _cohort_identity_map(xai_samples, label_column="y_true")
    if predicted != authoritative or explained != authoritative:
        raise ValueError(
            "predictions filtered to the cohort and sample_relevance must match "
            "the authoritative suite cohort"
        )
    prediction_by_id = filtered_predictions.set_index("sample_id")["prediction"]
    xai_by_id = xai_samples.set_index("sample_id")["prediction"]
    if not prediction_by_id.sort_index().equals(xai_by_id.sort_index()):
        raise ValueError("predictions and sample_relevance classes diverge by sample_id")
    return filtered_predictions


def _read_inputs(
    root: Path,
    profiles: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = LayerwiseSuitePaths(root)
    authoritative_cohorts = _load_authoritative_cohorts(root)
    performance_rows, emergence_frames, sample_frames, transition_frames = [], [], [], []
    expected_cells = {
        (profile, layer, source, target)
        for profile in profiles
        for layer in LAYERS
        for source in LANGUAGES
        for target in LANGUAGES
    }
    for profile, layer, source, target in sorted(expected_cells):
        _validated_marker(
            root,
            f"{profile}:cell:{layer:02d}:{source}:{target}",
            {"scores", "predictions", "metrics"},
        )
        metrics_path = paths.cell(profile, layer, source, target) / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        auc = metrics.get("threshold_free", {}).get("roc_auc")
        accuracy = metrics.get("fixed_threshold", {}).get("accuracy")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            for value in (auc, accuracy)
        ):
            raise ValueError("cell metrics contain invalid AUC or accuracy")
        predictions = pd.read_parquet(
            paths.cell(profile, layer, source, target) / "predictions.parquet"
        )
        _validated_marker(
            root,
            f"{profile}:xai:{layer:02d}:{source}:{target}",
            {"active_pointer"},
        )
        generation = resolve_active_generation(
            paths.layer_xai(profile, layer, source, target),
            expected_role="layer_xai",
        )
        samples = pd.read_parquet(generation / "sample_relevance.parquet")
        _validate_cell_against_authoritative_cohort(
            predictions,
            samples,
            authoritative_cohorts[target],
            profile=profile,
            layer=layer,
            source=source,
            target=target,
        )
        performance_rows.append(
            {
                "profile": profile,
                "layer": layer,
                "source": source,
                "target": target,
                "n": len(predictions),
                "auc": float(auc),
                "accuracy": float(accuracy),
                "diagonal": source == target,
                "external_validation": source != target,
                "corpus_shift": source != target,
            }
        )
        sample_frames.append(samples)
    for profile in profiles:
        for source in LANGUAGES:
            for target in LANGUAGES:
                _validated_marker(
                    root,
                    f"{profile}:emergence:{source}:{target}",
                    {"summary"},
                )
                emergence_frames.append(
                    pd.read_csv(
                        root
                        / ".stage-artifacts"
                        / f"{profile}__emergence__{source}__{target}"
                        / "emergence.csv"
                    )
                )
                _validated_marker(
                    root,
                    f"{profile}:trace:{source}:{target}",
                    {"active_pointer"},
                )
                generation = resolve_active_generation(
                    paths.final_trace_cell(profile, source, target),
                    expected_role="final_trace",
                )
                transitions = pd.read_csv(generation / "layer_transition_metrics.csv")
                labels = pd.read_parquet(
                    resolve_active_generation(
                        paths.layer_xai(profile, max(LAYERS), source, target),
                        expected_role="layer_xai",
                    )
                    / "sample_relevance.parquet"
                )[
                    [
                        "profile",
                        "source",
                        "target",
                        "sample_id",
                        "y_true",
                        "prediction",
                    ]
                ]
                transition_frames.append(
                    final_decision_reorganization(transitions, labels)
                )
    performance = pd.DataFrame(performance_rows)
    observed = set(
        performance[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected_cells:
        raise ValueError("suite performance cell set is incomplete")
    emergence = pd.concat(emergence_frames, ignore_index=True)
    emergence_identity = ["profile", "source", "target"]
    expected_emergence = {
        (profile, source, target)
        for profile in profiles
        for source in LANGUAGES
        for target in LANGUAGES
    }
    if (
        emergence.duplicated(emergence_identity).any()
        or set(
            emergence[emergence_identity].itertuples(index=False, name=None)
        )
        != expected_emergence
    ):
        raise ValueError("emergence cell set is duplicate or incomplete")

    samples = _validate_relevance_samples(
        pd.concat(sample_frames, ignore_index=True)
    )
    observed_xai = set(
        samples[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    )
    if observed_xai != expected_cells:
        raise ValueError("XAI cell set is incomplete")
    return (
        performance,
        emergence,
        samples,
        pd.concat(transition_frames, ignore_index=True),
    )


def _plot_tables(tables: Mapping[str, pd.DataFrame], directory: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    statuses = []

    def save(name: str, figure) -> None:
        outputs = []
        for suffix in ("png", "pdf"):
            path = directory / f"{name}.{suffix}"
            figure.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
            outputs.append(path.name)
        plt.close(figure)
        statuses.append({"name": name, "status": "complete", "files": outputs})

    performance = tables["layerwise_performance.csv"]
    pivot = performance.pivot_table(
        index=["profile", "source", "target"], columns="layer", values="auc"
    )
    figure, axis = plt.subplots(
        figsize=(8, max(3, 0.22 * len(pivot))), constrained_layout=True
    )
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xlabel("Layer")
    axis.set_ylabel("Encoder / source→target")
    axis.set_yticks(range(len(pivot)), ["/".join(map(str, item)) for item in pivot.index])
    axis.set_xticks(range(len(pivot.columns)), pivot.columns)
    figure.colorbar(image, ax=axis, label="ROC-AUC")
    save("encoder_layer_language_heatmap", figure)

    shift = tables["language_shift_by_layer.csv"]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for diagonal, group in shift.groupby("diagonal", sort=False):
        curve = group.groupby("layer")["auc"].mean()
        axis.plot(curve.index, curve.values, marker="o", label="diagonal" if diagonal else "off-diagonal")
    axis.set(xlabel="Layer", ylabel="Mean ROC-AUC")
    axis.legend()
    save("diagonal_offdiagonal_by_layer", figure)

    for table_name, metric, figure_name in (
        ("encoder_relevance_agreement.csv", "spearman_signed_mean", "relevance_agreement_by_layer"),
        ("spectral_divergence_by_layer.csv", "jensen_shannon", "spectral_divergence_by_layer"),
        ("final_decision_reorganization.csv", "normalized_l1_change_mean", "final_reorganization_by_layer"),
    ):
        frame = tables[table_name]
        if table_name == "encoder_relevance_agreement.csv" and frame.empty:
            statuses.append(
                {
                    "name": figure_name,
                    "status": "skipped",
                    "reason": "not_applicable_less_than_two_profiles",
                }
            )
            continue
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if not frame.empty:
            layer = "current_layer" if "current_layer" in frame else "layer"
            curve = frame.groupby(layer)[metric].mean()
            axis.plot(curve.index, curve.values, marker="o")
        axis.set(xlabel="Layer", ylabel=metric)
        save(figure_name, figure)
    return statuses


def _table_manifest_record(
    name: str,
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, object]:
    metadata = _table_scientific_metadata(name)
    method = metadata["method"]
    if name == "encoder_relevance_agreement.csv" and frame.empty:
        return {
            "name": name,
            "columns": list(frame.columns),
            "rows": 0,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "status": "not_applicable_less_than_two_profiles",
            "uncertainty": metadata,
        }
    if method != "none":
        required = {"ci_method", "ci_confidence", "statistic", "unit"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(
                f"scientific contract for {name} lacks IC/statistic/unit columns"
            )
        expected_statistic = metadata["statistic"]
        expected_unit = metadata["unit"]
        if (
            set(frame["ci_method"].tolist()) != {method}
            or set(frame["statistic"].tolist()) != {expected_statistic}
            or set(frame["unit"].tolist()) != {expected_unit}
        ):
            raise ValueError(f"scientific contract for {name} is inconsistent")
        confidence = frame["ci_confidence"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(confidence).all()
            or len(set(confidence.tolist())) != 1
            or not np.all((confidence > 0) & (confidence < 1))
        ):
            raise ValueError(f"scientific contract for {name} has invalid confidence")
        metadata["confidence"] = float(confidence[0])
        if method == "stratified_bootstrap_auc_percentile":
            if (
                "bootstrap_confidence" not in frame.columns
                or not np.allclose(
                    confidence,
                    frame["bootstrap_confidence"].to_numpy(dtype=np.float64),
                )
            ):
                raise ValueError(
                    f"scientific contract for {name} has divergent bootstrap confidence"
                )
    else:
        metadata["confidence"] = None
        if name == "spectral_divergence_by_layer.csv":
            required = {"statistic", "unit"}
            if (
                not required.issubset(frame.columns)
                or frame.empty
                or set(frame["statistic"].tolist()) != {metadata["statistic"]}
                or set(frame["unit"].tolist()) != {metadata["unit"]}
            ):
                raise ValueError(
                    f"scientific contract for {name} is inconsistent"
                )
    return {
        "name": name,
        "columns": list(frame.columns),
        "rows": len(frame),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "uncertainty": metadata,
    }


def aggregate_suite(
    *,
    root: str | Path,
    profiles: Sequence[str],
    config_hash: str,
    upstream_fingerprints: Sequence[str],
    inputs: Mapping[str, object],
    classical_audit: bool,
    classical_audit_adapter: Callable[..., Mapping[str, Path]] | None = None,
    plotter: Callable[[Mapping[str, pd.DataFrame], Path], list[dict]] = _plot_tables,
) -> Path:
    """Resolve validated Task 4-9 outputs and atomically publish suite aggregates."""
    root = Path(root).resolve()
    profiles = tuple(profiles)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be non-empty and unique")
    performance, emergence, samples, reorganization = _read_inputs(root, profiles)
    from .bands import mel_band_edges
    from .config import load_config

    edges: dict[str, np.ndarray] = {}
    for language in LANGUAGES:
        item = inputs[language]
        cfg = load_config(Path(getattr(item, "config_path")))
        edges[language] = mel_band_edges(
            cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max
        )
    emergence = _annotate_emergence_scientific_contract(emergence)
    tables = {
        "layerwise_performance.csv": performance.sort_values(
            ["profile", "layer", "source", "target"], kind="mergesort"
        ),
        "emergence_layers.csv": emergence.sort_values(
            ["profile", "source", "target"], kind="mergesort"
        ),
        "language_shift_by_layer.csv": language_shift_by_layer(performance),
        "encoder_relevance_agreement.csv": encoder_relevance_agreement(
            samples, band_edges_by_target=edges
        ),
        "spectral_divergence_by_layer.csv": spectral_divergence_by_layer(
            samples, band_edges_by_target=edges
        ),
        "final_decision_reorganization.csv": reorganization.sort_values(
            [
                "profile",
                "source",
                "target",
                "previous_layer",
                "current_layer",
                "conditioning",
                "class_value",
            ],
            kind="mergesort",
        ),
    }
    destination = root / "aggregates"

    def write(directory: Path) -> None:
        for name, frame in tables.items():
            frame.to_csv(directory / name, index=False)
        try:
            figures = plotter(tables, directory)
        except Exception as exc:
            figures = [
                {
                    "name": "optional_plotting",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
        audit = run_classical_audit(
            enabled=classical_audit,
            output_dir=directory,
            adapter=classical_audit_adapter,
        )
        table_records = []
        for name in TABLE_NAMES:
            path = directory / name
            table_records.append(_table_manifest_record(name, tables[name], path))
        (directory / "aggregate_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": AGGREGATE_SCHEMA_VERSION,
                    "config_hash": config_hash,
                    "upstream_fingerprints": list(upstream_fingerprints),
                    "profiles": list(profiles),
                    "claims": CLAIMS,
                    "tables": table_records,
                    "figures": figures,
                    "classical_audit": {
                        "enabled": classical_audit,
                        "scope": "three diagonals at layer 12 only",
                        "runs": audit,
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    publish_generation(destination, write, role="suite_aggregates")
    resolve_active_generation(destination, expected_role="suite_aggregates")
    return destination / "active.json"
