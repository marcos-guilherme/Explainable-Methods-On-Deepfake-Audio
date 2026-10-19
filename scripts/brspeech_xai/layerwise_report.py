"""Read-only discovery and validation for layer-wise XAI report sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

import numpy as np
import pandas as pd

from .bands import F_MAX, F_MIN, N_BANDS, mel_band_edges
from .encoder_suite import LANGUAGE_ORDER, LAYER_ORDER, ROLE_ORDER
from .layerwise_paths import LayerwiseSuitePaths, resolve_active_generation

FINAL_LAYER = LAYER_ORDER[-1]
# Pairs per class fixed by the faithfulness protocol (layerwise_faithfulness);
# the main faithfulness figure accepts only rows with this n_pairs.
FAITHFULNESS_PER_CLASS = 8

_LANGUAGE_SEGMENTS: dict[str, tuple[str, ...]] = {
    "eng": ("eng",),
    "por": ("por",),
    "zho": ("zho",),
    "eng-por": ("eng", "por"),
    "eng-zho": ("eng", "zho"),
    "por-zho": ("por", "zho"),
    "eng-por-zho": ("eng", "por", "zho"),
}

_RESULT_NAME = re.compile(
    r"^(?P<profile>[A-Za-z0-9._-]+)__"
    r"(?P<languages>eng-por-zho|eng-por|eng-zho|por-zho|eng|por|zho)__"
    r"(?P<protocol>layerwise_xai)__(?P<scope>pilot|full)$"
)


@dataclass(frozen=True)
class ExperimentIdentity:
    profile: str
    languages: tuple[str, ...]
    protocol: str
    scope: str


@dataclass(frozen=True)
class ReportSource:
    root: Path
    identity: ExperimentIdentity
    status: Mapping[str, object]
    plan: Mapping[str, object]
    aggregate_generation: Path
    aggregate_generation_id: str
    layer_xai_generations: Mapping[tuple[str, int, str, str], Path]
    final_trace_generations: Mapping[tuple[str, str, str], Path]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _deep_freeze_mapping(self.status))
        object.__setattr__(self, "plan", _deep_freeze_mapping(self.plan))
        object.__setattr__(
            self,
            "layer_xai_generations",
            MappingProxyType(dict(self.layer_xai_generations)),
        )
        object.__setattr__(
            self,
            "final_trace_generations",
            MappingProxyType(dict(self.final_trace_generations)),
        )


def _deep_freeze_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a mapping to freeze")
    return MappingProxyType(
        {key: _deep_freeze_value(item) for key, item in value.items()}
    )


def _deep_freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _deep_freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze_value(item) for item in value)
    return value


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _canonical_languages(plan: Mapping[str, object]) -> tuple[str, ...]:
    languages = plan.get("languages")
    if not isinstance(languages, dict):
        raise ValueError("execution plan languages must be an object")
    allowed = set(LANGUAGE_ORDER)
    unknown = sorted(
        language for language in languages if language not in allowed
    )
    if unknown:
        raise ValueError(
            "execution plan contains unknown languages: "
            + ", ".join(unknown)
        )
    return tuple(language for language in LANGUAGE_ORDER if language in languages)


def _plan_profiles(plan: Mapping[str, object]) -> tuple[str, ...]:
    profiles = plan.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("execution plan profiles must be a non-empty object")
    return tuple(sorted(str(profile) for profile in profiles))


def _stdft_examples_per_generation(plan: Mapping[str, object]) -> int:
    suite_config = plan.get("suite_config")
    if not isinstance(suite_config, dict):
        raise ValueError("execution plan suite_config must be an object")
    per_class = suite_config.get("stdft_examples_per_class")
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class <= 0:
        raise ValueError("execution plan stdft_examples_per_class must be positive")
    return 2 * per_class


def parse_result_dir_name(path: Path) -> ExperimentIdentity:
    match = _RESULT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"malformed result directory name: {path.name}")
    language_segment = match.group("languages")
    if language_segment not in _LANGUAGE_SEGMENTS:
        raise ValueError(f"malformed result directory name: {path.name}")
    languages = _LANGUAGE_SEGMENTS[language_segment]
    return ExperimentIdentity(
        profile=match.group("profile"),
        languages=languages,
        protocol=match.group("protocol"),
        scope=match.group("scope"),
    )


def validate_result_directory(root: Path) -> ReportSource:
    root = Path(root)
    identity = parse_result_dir_name(root)

    status_path = root / "run_status.json"
    plan_path = root / "execution_plan.json"
    if not status_path.is_file():
        raise ValueError(f"missing run status at {status_path}")
    if not plan_path.is_file():
        raise ValueError(f"missing execution plan at {plan_path}")

    status = _load_json_mapping(status_path)
    plan = _load_json_mapping(plan_path)

    if status.get("status") != "complete" or status.get("mode") == "dry-run":
        raise ValueError(
            "result directory must be a completed non-dry-run production run"
        )

    status_hash = status.get("config_hash")
    plan_hash = plan.get("config_hash")
    if not isinstance(status_hash, str) or not status_hash:
        raise ValueError("run status config_hash must be a non-empty string")
    if not isinstance(plan_hash, str) or not plan_hash:
        raise ValueError("execution plan config_hash must be a non-empty string")
    if status_hash != plan_hash:
        raise ValueError("run status and execution plan config_hash must match")

    profiles = _plan_profiles(plan)
    if len(profiles) != 1 or profiles[0] != identity.profile:
        raise ValueError(
            f"execution plan profile must match directory identity {identity.profile!r}"
        )

    plan_languages = _canonical_languages(plan)
    if plan_languages != identity.languages:
        raise ValueError(
            "execution plan languages must match directory identity "
            f"{identity.languages!r}, found {plan_languages!r}"
        )

    paths = LayerwiseSuitePaths(root)
    profile = identity.profile
    languages = identity.languages

    aggregate_generation = resolve_active_generation(
        root / "aggregates",
        expected_role="suite_aggregates",
    )
    aggregate_generation_id = aggregate_generation.name

    expected_stdft = _stdft_examples_per_generation(plan)
    layer_xai_generations: dict[tuple[str, int, str, str], Path] = {}
    for layer in LAYER_ORDER:
        for source in languages:
            for target in languages:
                destination = paths.layer_xai(profile, layer, source, target)
                generation = resolve_active_generation(
                    destination,
                    expected_role="layer_xai",
                )
                examples_dir = generation / "stdft_examples"
                observed = (
                    sorted(
                        path
                        for path in examples_dir.glob("*.npz")
                        if path.is_file()
                    )
                    if examples_dir.is_dir()
                    else []
                )
                if len(observed) != expected_stdft:
                    raise ValueError(
                        f"expected {expected_stdft} stdft_examples at "
                        f"{examples_dir}, found {len(observed)}"
                    )
                layer_xai_generations[(profile, layer, source, target)] = generation

    final_trace_generations: dict[tuple[str, str, str], Path] = {}
    for source in languages:
        for target in languages:
            destination = paths.final_trace_cell(profile, source, target)
            generation = resolve_active_generation(
                destination,
                expected_role="final_trace",
            )
            final_trace_generations[(profile, source, target)] = generation

    return ReportSource(
        root=root,
        identity=identity,
        status=status,
        plan=plan,
        aggregate_generation=aggregate_generation,
        aggregate_generation_id=aggregate_generation_id,
        layer_xai_generations=layer_xai_generations,
        final_trace_generations=final_trace_generations,
    )


# ---------------------------------------------------------------------------
# Scientific loading, report tables and figures
# ---------------------------------------------------------------------------

FIGURE_FAMILIES: tuple[str, ...] = (
    "performance_by_layer",
    "fixed_threshold_by_layer",
    "transfer_performance_heatmaps",
    "dft_relevance_heatmap",
    "class_relevance_by_layer",
    "decision_reorganization_by_layer",
    "stdft_examples",
    "conservation_diagnostics",
)
MAIN_RESULT_FIGURES: tuple[str, ...] = (
    "transfer_performance_heatmaps",
    "transfer_delta_roc_auc",
    "transfer_delta_mcc",
    "encoder_agreement_heatmaps",
    "dft_relevance_heatmap",
    "language_shift_intervals",
    "spectral_divergence_by_layer",
    "layer_faithfulness_intervals",
)
APPENDIX_FIGURES: tuple[str, ...] = (
    "performance_by_layer",
    "fixed_threshold_by_layer",
    "class_relevance_by_layer",
    "decision_reorganization_by_layer",
    "encoder_agreement_by_layer",
    "stdft_examples",
    "conservation_diagnostics",
)
COMPARISON_FAMILIES: tuple[str, ...] = (
    "encoder_agreement",
    "language_shift",
    "diagonal_vs_offdiagonal",
    "spectral_divergence",
)
_GROUP_LABELS_PT: Mapping[str, str] = MappingProxyType(
    {"all": "todos", "real": "real", "synthetic": "sintético"}
)


def _planned_figures(
    omitted_comparisons: Mapping[str, str],
    *,
    has_aligned_faithfulness: bool = False,
) -> tuple[str, ...]:
    """Base figure families plus comparison figures whose tables are complete."""
    delivered = tuple(
        name
        for name in COMPARISON_FAMILIES
        if name != "encoder_agreement" and name not in omitted_comparisons
    )
    additions: list[str] = []
    if "diagonal_vs_offdiagonal" not in omitted_comparisons:
        additions.extend(("transfer_delta_roc_auc", "transfer_delta_mcc"))
    if "encoder_agreement" not in omitted_comparisons:
        additions.extend(("encoder_agreement_heatmaps", "encoder_agreement_by_layer"))
    if "language_shift" not in omitted_comparisons:
        additions.append("language_shift_intervals")
    if "spectral_divergence" not in omitted_comparisons:
        additions.append("spectral_divergence_by_layer")
    if has_aligned_faithfulness:
        additions.append("layer_faithfulness_intervals")
    return tuple(dict.fromkeys(FIGURE_FAMILIES + delivered + tuple(additions)))
# Fixed, non-negotiable bootstrap contract shared by every comparison builder.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI_LEVEL = 0.95
_CI_PERCENTILES = (
    100.0 * (1.0 - BOOTSTRAP_CI_LEVEL) / 2.0,
    100.0 * (1.0 + BOOTSTRAP_CI_LEVEL) / 2.0,
)
# Minimum number of finite bootstrap draws required to report a percentile CI.
# Below this floor the interval is refused (NaN) rather than computed from an
# unknown, silently reduced sample.
_MIN_VALID_RESAMPLES = 2
# Jensen-Shannon distance is reported in bits (base-2 logarithm, range [0, 1]).
_JS_BASE = 2.0
# Comparison groups condition on the true class; "all" pools both classes but is
# still stratified by class during resampling.
_COMPARISON_GROUPS: tuple[tuple[str, object], ...] = (
    ("all", None),
    ("real", 0),
    ("synthetic", 1),
)
CLASS_NAMES: Mapping[int, str] = MappingProxyType({0: "real", 1: "synthetic"})
FALLBACK_BAND_ORIGIN = "fallback_default_mel_contract"
CONFIG_BAND_ORIGIN = "config"

_MEASURES = ("signed", "absolute_normalized")
_AGGREGATE_PERFORMANCE_COLUMNS = frozenset(
    {"profile", "layer", "source", "target", "n", "auc", "accuracy"}
)
_AGGREGATE_EMERGENCE_COLUMNS = frozenset(
    {"profile", "source", "target", "onset", "consolidation", "final_auc"}
)
_METRIC_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("threshold_free", ("roc_auc", "average_precision", "eer_diagnostic")),
    ("fixed_threshold", ("threshold", "accuracy", "mcc", "tpr", "fpr", "fnr")),
)
_BAND_SUMMARY_COLUMNS = frozenset(
    {
        "profile",
        "layer",
        "source",
        "target",
        "grouping",
        "group",
        "measure",
        "band",
        "n",
        "mean",
        "ci_low",
        "ci_high",
    }
)
_SAMPLE_COLUMNS = frozenset(
    {
        "profile",
        "layer",
        "source",
        "target",
        "sample_id",
        "y_true",
        "prediction",
        "score",
        "score_recompute_absolute_error",
        "band_signed",
        "band_abs_normalized",
    }
)
_VALIDATION_KIND = "bias_zeroed_model_rule_check"
_CONSERVATION_FIELDS = (
    "bias_zeroed_validation_residual",
    "max_bias_inclusive_attribution_gap",
    "max_score_recompute_absolute_error",
    "score_recompute_rtol",
    "score_recompute_atol",
    "max_dft_residual",
    "max_stdft_conservation_relative_error",
    "tolerance",
)
_STDFT_KEYS = frozenset(
    {
        "sample_id",
        "times",
        "freqs",
        "relevance",
        "spectrum",
        "relevance_time_sum",
        "relevance_tf_sum",
        "conservation_absolute_error",
        "conservation_relative_error",
        "conservation_tolerance",
    }
)
_TRANSITION_COLUMNS = frozenset(
    {
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
)
_TRANSITION_METRICS = (
    "similarity",
    "normalized_l1_change",
    "absolute_mass",
    "temporal_entropy",
)
_LEAD_COLUMNS = ("model", "source", "target", "language")
_PER_SAMPLE_KEYS = ("model", "source", "target", "layer", "sample_id")
_EXPECTED_TRANSITIONS = frozenset((layer, layer + 1) for layer in range(1, 12))


@dataclass(frozen=True)
class BandEdges:
    """Frequency-band edges used by one model on one target language."""

    model: str
    language: str
    edges_hz: tuple[float, ...]
    origin: str
    reason: str
    n_bands: int
    f_min: float
    f_max: float
    config_path: str | None = None
    recorded_sha256: str | None = None
    observed_sha256: str | None = None


@dataclass(frozen=True, eq=False)
class LoadedReportSource:
    """Scientific artifacts of one validated source, loaded read-only."""

    source: ReportSource
    performance: pd.DataFrame
    emergence: pd.DataFrame
    band_relevance: pd.DataFrame
    class_relevance: pd.DataFrame
    cohort: pd.DataFrame
    conservation: pd.DataFrame
    transitions: pd.DataFrame
    stdft_examples: pd.DataFrame
    stdft_payloads: Mapping[tuple, Mapping[str, np.ndarray]]
    probe_stability: pd.DataFrame
    layer_faithfulness: pd.DataFrame
    cell_predictions: pd.DataFrame
    sample_relevance: pd.DataFrame
    band_edges: tuple[BandEdges, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stdft_payloads",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(payload))
                    for key, payload in self.stdft_payloads.items()
                }
            ),
        )
        object.__setattr__(self, "band_edges", tuple(self.band_edges))


@dataclass(frozen=True, eq=False)
class ReportTables:
    """Deterministic, identity-labelled tables shared by every report output."""

    models: tuple[str, ...]
    languages: tuple[str, ...]
    performance: pd.DataFrame
    emergence: pd.DataFrame
    band_relevance: pd.DataFrame
    class_relevance: pd.DataFrame
    transitions: pd.DataFrame
    conservation: pd.DataFrame
    cohort: pd.DataFrame
    stdft_examples: pd.DataFrame
    stdft_payloads: Mapping[tuple, Mapping[str, np.ndarray]]
    band_edges: pd.DataFrame
    band_edge_provenance: pd.DataFrame
    transfer_selection: pd.DataFrame
    xai_performance_association: pd.DataFrame
    planned_figures: tuple[str, ...]
    omitted_comparisons: Mapping[str, str]
    probe_stability: pd.DataFrame
    layer_faithfulness: pd.DataFrame
    cell_predictions: pd.DataFrame
    sample_relevance: pd.DataFrame
    encoder_agreement: pd.DataFrame
    language_shift: pd.DataFrame
    diagonal_vs_offdiagonal: pd.DataFrame
    spectral_divergence: pd.DataFrame
    comparison_completeness: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stdft_payloads", MappingProxyType(dict(self.stdft_payloads))
        )
        object.__setattr__(
            self,
            "omitted_comparisons",
            MappingProxyType(dict(self.omitted_comparisons)),
        )
        object.__setattr__(
            self,
            "comparison_completeness",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(info))
                    for name, info in self.comparison_completeness.items()
                }
            ),
        )


@dataclass(frozen=True)
class FigureRecord:
    """Scientific description of one emitted figure family."""

    name: str
    files: tuple[str, ...]
    title: str
    caption: str
    metric: str
    units: str
    transformation: str
    models: tuple[str, ...]
    languages: tuple[str, ...]
    cells: tuple[tuple[str, str, str], ...]


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_csv(path: Path, required: frozenset[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing {path.name} in {path.parent}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"{path.name} is unreadable: {exc}") from exc
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    return frame


def _identified(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    result = frame.drop(columns=[c for c in ("profile",) if c in frame.columns])
    result = result.copy()
    result.insert(0, "model", model)
    result["language"] = result["target"]
    rest = [column for column in result.columns if column not in _LEAD_COLUMNS]
    return result[list(_LEAD_COLUMNS) + rest]


def _ordered(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def _expect_identity(
    frame: pd.DataFrame,
    expected: Mapping[str, object],
    context: str,
) -> None:
    for column, value in expected.items():
        if column not in frame.columns or set(frame[column].tolist()) != {value}:
            raise ValueError(
                f"{context}: {column} identity must be {value!r}"
            )


def _fallback_band_edges(
    model: str,
    language: str,
    reason: str,
    *,
    config_path: str | None = None,
    recorded_sha256: str | None = None,
    observed_sha256: str | None = None,
) -> BandEdges:
    edges = mel_band_edges()
    return BandEdges(
        model=model,
        language=language,
        edges_hz=tuple(float(value) for value in edges),
        origin=FALLBACK_BAND_ORIGIN,
        reason=reason,
        n_bands=N_BANDS,
        f_min=float(F_MIN),
        f_max=float(F_MAX),
        config_path=config_path,
        recorded_sha256=recorded_sha256,
        observed_sha256=observed_sha256,
    )


def _resolve_band_edges(source: ReportSource, language: str) -> BandEdges:
    model = source.identity.profile
    inputs = source.plan.get("inputs")
    entry = inputs.get(language) if isinstance(inputs, Mapping) else None
    if not isinstance(entry, Mapping):
        return _fallback_band_edges(model, language, "no_config_recorded")
    raw_path = entry.get("config_path")
    recorded_hash = entry.get("config_sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(recorded_hash, str)
        or not recorded_hash
    ):
        return _fallback_band_edges(model, language, "no_config_recorded")
    config_path = Path(raw_path)
    recorded = {"config_path": raw_path, "recorded_sha256": recorded_hash}
    if not config_path.is_file():
        return _fallback_band_edges(model, language, "config_file_missing", **recorded)
    observed_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    recorded["observed_sha256"] = observed_hash
    if observed_hash != recorded_hash:
        return _fallback_band_edges(
            model, language, "config_sha256_mismatch", **recorded
        )
    try:
        from .config import load_config

        bands = load_config(config_path).bands
        n_bands, f_min, f_max = int(bands.n_bands), float(bands.f_min), float(bands.f_max)
        edges = mel_band_edges(n_bands, f_min, f_max)
        if (
            n_bands < 1
            or edges.shape != (n_bands + 1,)
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("invalid band grid")
    except Exception:
        return _fallback_band_edges(model, language, "config_unreadable", **recorded)
    return BandEdges(
        model=model,
        language=language,
        edges_hz=tuple(float(value) for value in edges),
        origin=CONFIG_BAND_ORIGIN,
        reason="config_sha256_verified",
        n_bands=n_bands,
        f_min=f_min,
        f_max=f_max,
        **recorded,
    )


def _check_persisted_band_counts(
    source: ReportSource, band_edges: Mapping[str, BandEdges]
) -> None:
    """Fail before heavy loading when the resolved grid cannot fit the artifacts.

    The persisted band count is read from each cell's `dft_band_relevance.csv`
    and compared with the grid resolved for the target language, so a missing,
    edited or inconsistent config fails with the evidence needed to audit it.
    """
    for (profile, layer, src, tgt), generation in sorted(
        source.layer_xai_generations.items()
    ):
        edges = band_edges[tgt]
        path = generation / "dft_band_relevance.csv"
        try:
            persisted = int(pd.read_csv(path, usecols=["band"])["band"].nunique())
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(
                f"{profile}/layer_{layer:02d}/{src}->{tgt}: cannot read the persisted "
                f"band count from {path.name}"
            ) from exc
        if persisted == edges.n_bands:
            continue
        if edges.origin == FALLBACK_BAND_ORIGIN:
            grid = (
                f"fallback default mel contract n_bands={edges.n_bands} "
                f"({edges.f_min:g}-{edges.f_max:g} Hz)"
            )
        else:
            grid = f"recorded config grid n_bands={edges.n_bands}"
        recorded = (
            f"config_path={edges.config_path}; recorded sha256="
            f"{edges.recorded_sha256}; observed sha256={edges.observed_sha256}"
            if edges.config_path
            else "config_path=not recorded in the execution plan"
        )
        raise ValueError(
            f"{profile}/layer_{layer:02d}/{src}->{tgt}: persisted n_bands={persisted} "
            f"does not match the {grid} for target {tgt!r} "
            f"(reason {edges.reason}; {recorded})"
        )


def _validate_sample_id_column(frame: pd.DataFrame, context: str) -> None:
    if "sample_id" not in frame.columns:
        raise ValueError(f"{context}: cell predictions lack sample_id")
    ids = frame["sample_id"].tolist()
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"{context}: invalid or duplicate sample_id in predictions")


def _read_and_validate_cell_predictions(
    path: Path, row: Mapping[str, object], expected_n: int, cell: str
) -> pd.DataFrame:
    """Recompute count-based metrics from persisted scores and return per-sample rows."""
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"cell predictions are unreadable for {cell}") from exc
    missing = {"sample_id", "y_true", "score", "prediction"} - set(frame.columns)
    if missing:
        raise ValueError(f"cell predictions lack columns {sorted(missing)} for {cell}")
    if len(frame) != expected_n:
        raise ValueError(
            f"aggregate n {expected_n} disagrees with persisted predictions "
            f"({len(frame)} rows) for {cell}"
        )
    _validate_sample_id_column(frame, cell)
    y_true = frame["y_true"].to_numpy()
    prediction = frame["prediction"].to_numpy()
    score = frame["score"].to_numpy(dtype=np.float64)
    if (
        not set(y_true.tolist()).issubset({0, 1})
        or not set(prediction.tolist()).issubset({0, 1})
        or not np.isfinite(score).all()
        or set(y_true.tolist()) != {0, 1}
    ):
        raise ValueError(
            f"cell predictions need finite scores and both binary classes for {cell}"
        )
    y_true = y_true.astype(np.int64)
    prediction = prediction.astype(np.int64)
    threshold = float(row["threshold"])
    if not np.array_equal(prediction, (score >= threshold).astype(np.int64)):
        raise ValueError(
            f"persisted predictions disagree with the fixed threshold for {cell}"
        )
    tp = int(((prediction == 1) & (y_true == 1)).sum())
    tn = int(((prediction == 0) & (y_true == 0)).sum())
    fp = int(((prediction == 1) & (y_true == 0)).sum())
    fn = int(((prediction == 0) & (y_true == 1)).sum())
    positives, negatives = tp + fn, tn + fp
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    derived = {
        "accuracy": (tp + tn) / len(frame),
        "tpr": tp / positives,
        "fpr": fp / negatives,
        "fnr": fn / positives,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }
    for name, value in derived.items():
        if not np.isclose(float(row[name]), value, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"cell metric {name} disagrees with persisted predictions for {cell}"
            )
    from sklearn.metrics import roc_auc_score

    if not np.isclose(
        float(row["roc_auc"]), float(roc_auc_score(y_true, score)), rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"cell metric roc_auc disagrees with persisted predictions for {cell}"
        )
    return frame.assign(threshold=threshold)[
        ["sample_id", "y_true", "score", "prediction", "threshold"]
    ]


def _check_metrics_against_predictions(
    path: Path, row: Mapping[str, object], expected_n: int, cell: str
) -> None:
    _read_and_validate_cell_predictions(path, row, expected_n, cell)


def _validate_prediction_xai_alignment(
    context: str,
    predictions: pd.DataFrame,
    xai: pd.DataFrame,
) -> None:
    prediction_by_id = predictions.set_index("sample_id", drop=False)
    missing = set(xai["sample_id"]) - set(prediction_by_id.index)
    if missing:
        raise ValueError(
            f"{context}: sample_relevance sample_id values absent from predictions: "
            f"{sorted(missing)}"
        )
    xai_y_true = xai.set_index("sample_id")["y_true"]
    reference = prediction_by_id.loc[xai_y_true.index, "y_true"]
    if not np.array_equal(
        xai_y_true.to_numpy(dtype=np.int64),
        reference.to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            f"{context}: predictions and sample_relevance disagree on y_true "
            "for shared sample_id"
        )


def _xai_cohort_signature(frame: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    ordered = frame.sort_values("sample_id", kind="mergesort")
    return tuple(
        (str(row.sample_id), int(row.y_true))
        for row in ordered.itertuples(index=False)
    )


def _check_paired_xai_across_models(sample_relevance: pd.DataFrame) -> None:
    """Require identical XAI cohorts across models for the same source/target/layer."""
    keys = ["source", "target", "layer"]
    for _, group in sample_relevance.groupby(keys, sort=True):
        reference: tuple[tuple[str, int], ...] | None = None
        reference_model: str | None = None
        for model, part in group.groupby("model", sort=True):
            signature = _xai_cohort_signature(part)
            if reference is None:
                reference = signature
                reference_model = str(model)
                continue
            if signature != reference:
                src, tgt, layer = (
                    str(part["source"].iloc[0]),
                    str(part["target"].iloc[0]),
                    int(part["layer"].iloc[0]),
                )
                raise ValueError(
                    f"XAI cohort for pairing differs across models at layer {layer} "
                    f"({src}->{tgt}): {reference_model!r} vs {model!r}"
                )


_MARKER_KEYS = frozenset(
    {"schema_version", "contract_version", "stage_id", "profile", "fingerprint", "artifacts"}
)
_MARKER_ARTIFACT_FILES = {
    "scores": "scores.npy",
    "predictions": "predictions.parquet",
    "metrics": "metrics.json",
}
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_ABSOLUTE_RECORDED_PATH_RE = re.compile(r"(/|[A-Za-z]:/)")


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cell_marker(
    root: Path,
    paths: LayerwiseSuitePaths,
    profile: str,
    layer: int,
    source: str,
    target: str,
    cell: str,
) -> None:
    """Validate a cell `.state` marker so a moved or renamed bundle stays readable.

    The marker records absolute paths from the machine and root name that produced
    it. Those prefixes are not trusted: each artifact is resolved from the current
    root through `LayerwiseSuitePaths.cell`, the recorded path must end in the same
    canonical relative suffix, and the current file must match the recorded SHA-256.
    """
    stage_id = f"{profile}:cell:{layer:02d}:{source}:{target}"
    marker_path = root / ".state" / f"{stage_id.replace(':', '__')}.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cell marker is missing or unreadable for {cell}") from exc
    if not isinstance(marker, dict) or set(marker) != _MARKER_KEYS:
        raise ValueError(f"cell marker has an unexpected schema for {cell}")
    schema_version = marker["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != 1
        or marker["contract_version"] != "encoder-suite-v2"
        or marker["stage_id"] != stage_id
        or marker["profile"] != profile
    ):
        raise ValueError(f"cell marker identity or contract mismatch for {cell}")
    fingerprint = marker["fingerprint"]
    if not isinstance(fingerprint, str) or _HEX64_RE.fullmatch(fingerprint) is None:
        raise ValueError(f"cell marker fingerprint is not a SHA-256 digest for {cell}")
    artifacts = marker["artifacts"]
    if not isinstance(artifacts, list) or any(
        not isinstance(item, dict)
        or set(item) != {"name", "path", "sha256"}
        or not all(isinstance(item[key], str) for key in item)
        for item in artifacts
    ):
        raise ValueError(f"cell marker artifacts are malformed for {cell}")
    names = [item["name"] for item in artifacts]
    if len(names) != len(set(names)) or set(names) != set(_MARKER_ARTIFACT_FILES):
        raise ValueError(f"cell marker artifact names mismatch for {cell}")

    cell_dir = paths.cell(profile, layer, source, target)
    canonical_root = root.resolve()
    for item in artifacts:
        name = item["name"]
        expected = cell_dir / _MARKER_ARTIFACT_FILES[name]
        suffix = "/".join(expected.relative_to(root).parts)
        recorded = item["path"].replace("\\", "/")
        if (
            _ABSOLUTE_RECORDED_PATH_RE.match(recorded) is None
            or ".." in recorded.split("/")
            or not recorded.endswith("/" + suffix)
        ):
            raise ValueError(
                f"cell marker {name} artifact path {item['path']!r} does not end "
                f"with the canonical cell path {suffix!r} for {cell}"
            )
        if (
            not expected.is_file()
            or not expected.resolve().is_relative_to(canonical_root)
        ):
            raise ValueError(
                f"cell marker {name} artifact file is missing from the current "
                f"root for {cell}"
            )
        if _HEX64_RE.fullmatch(item["sha256"]) is None or (
            _sha256_of(expected) != item["sha256"]
        ):
            raise ValueError(
                f"cell marker {name} artifact sha256 does not match the current "
                f"file for {cell}"
            )


_PROBE_STABILITY_COLUMNS = frozenset(
    {
        "profile",
        "layer",
        "source",
        "target",
        "n_seeds",
        "roc_auc_mean",
        "roc_auc_std",
        "roc_auc_min",
        "roc_auc_max",
        "mcc_mean",
        "mcc_std",
        "mcc_min",
        "mcc_max",
    }
)


def _load_probe_stability_summary(
    source: ReportSource, paths: LayerwiseSuitePaths
) -> pd.DataFrame:
    profile = source.identity.profile
    summary_path = paths.probe_stability_summary(profile)
    if not summary_path.is_file():
        return pd.DataFrame(columns=sorted(_PROBE_STABILITY_COLUMNS))
    frame = _read_csv(summary_path, _PROBE_STABILITY_COLUMNS)
    if frame.empty:
        return frame
    if (frame["profile"] != profile).any():
        raise ValueError(
            "probe_stability_by_layer.csv profile does not match report source"
        )
    return _ordered(
        _identified(frame, profile),
        ["model", "source", "target", "layer"],
    )


_LAYER_FAITHFULNESS_COLUMNS = frozenset(
    {
        "model",
        "source",
        "target",
        "layer",
        "k",
        "true_class",
        "comparison",
        "mean_difference",
        "ci_low",
        "ci_high",
        "n_pairs",
        "wilcoxon_p",
        "status",
    }
)


def _load_layer_faithfulness_summaries(
    source: ReportSource, paths: LayerwiseSuitePaths
) -> pd.DataFrame:
    profile = source.identity.profile
    rows: list[dict[str, object]] = []
    for language in source.identity.languages:
        destination = paths.layer_faithfulness_cell(
            profile, FINAL_LAYER, language, language
        )
        if not (destination / "active.json").is_file():
            continue
        try:
            generation = resolve_active_generation(
                destination, expected_role="layer_faithfulness"
            )
        except ValueError:
            continue
        paired_path = generation / "paired_comparisons.csv"
        if not paired_path.is_file():
            continue
        frame = _read_csv(
            paired_path,
            frozenset(
                {
                    "level",
                    "k",
                    "true_class",
                    "comparison",
                    "mean_difference",
                    "ci_low",
                    "ci_high",
                    "n_pairs",
                    "wilcoxon_p",
                    "status",
                }
            ),
        )
        aggregate = frame.loc[frame["level"] == "aggregate"].copy()
        if aggregate.empty:
            continue
        aggregate = aggregate.assign(
            source=language,
            layer=FINAL_LAYER,
            target=language,
        )
        rows.extend(aggregate.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=sorted(_LAYER_FAITHFULNESS_COLUMNS))
    result = _identified(pd.DataFrame(rows), profile)
    if (result["source"] != result["target"]).any():
        raise ValueError("layer faithfulness summaries must be diagonal cells only")
    if (result["layer"] != FINAL_LAYER).any():
        raise ValueError("layer faithfulness summaries must refer to the final layer")
    return _ordered(
        result,
        ["model", "source", "target", "layer", "k", "true_class", "comparison"],
    )


def _load_performance(
    source: ReportSource, paths: LayerwiseSuitePaths
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = source.identity.profile
    languages = source.identity.languages
    frame = _read_csv(
        source.aggregate_generation / "layerwise_performance.csv",
        _AGGREGATE_PERFORMANCE_COLUMNS,
    )
    frame = frame[frame["profile"] == profile]
    keys = ["layer", "source", "target"]
    if frame.duplicated(keys).any():
        raise ValueError("layerwise_performance.csv contains duplicate cells")
    expected = {
        (layer, src, tgt)
        for layer in LAYER_ORDER
        for src in languages
        for tgt in languages
    }
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            "layerwise_performance.csv cell set is incomplete or unexpected"
        )
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for item in frame.sort_values(keys, kind="mergesort").itertuples(index=False):
        layer, src, tgt = int(item.layer), str(item.source), str(item.target)
        cell = f"{profile}/layer_{layer:02d}/{src}->{tgt}"
        cell_dir = paths.cell(profile, layer, src, tgt)
        _validate_cell_marker(source.root, paths, profile, layer, src, tgt, cell)
        metrics_path = cell_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cell metrics are unreadable for {cell}") from exc
        if not isinstance(metrics, dict):
            raise ValueError(f"cell metrics must be an object for {cell}")
        row: dict[str, object] = {
            "layer": layer,
            "source": src,
            "target": tgt,
            "n": int(item.n),
        }
        for section, names in _METRIC_SECTIONS:
            values = metrics.get(section)
            if not isinstance(values, dict):
                raise ValueError(f"cell metrics lack {section} for {cell}")
            for name in names:
                if not _is_finite_number(values.get(name)):
                    raise ValueError(
                        f"cell metrics lack finite {section}.{name} for {cell}"
                    )
                row[name] = float(values[name])
        if not np.isclose(row["roc_auc"], float(item.auc), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"cell metrics roc_auc disagrees with aggregate for {cell}"
            )
        if not np.isclose(row["accuracy"], float(item.accuracy), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"cell metrics accuracy disagrees with aggregate for {cell}"
            )
        validated = _read_and_validate_cell_predictions(
            cell_dir / "predictions.parquet", row, int(item.n), cell
        )
        prediction_frames.append(
            validated.assign(layer=layer, source=src, target=tgt)
        )
        rows.append(row)
    performance = _ordered(
        _identified(pd.DataFrame(rows), profile),
        ["model", "source", "target", "layer"],
    )
    cell_predictions = _ordered(
        _identified(pd.concat(prediction_frames, ignore_index=True), profile),
        list(_PER_SAMPLE_KEYS),
    )
    if cell_predictions.duplicated(list(_PER_SAMPLE_KEYS)).any():
        raise ValueError("duplicate per-sample prediction identity within source")
    return performance, cell_predictions


def _load_emergence(source: ReportSource) -> pd.DataFrame:
    profile = source.identity.profile
    languages = source.identity.languages
    frame = _read_csv(
        source.aggregate_generation / "emergence_layers.csv",
        _AGGREGATE_EMERGENCE_COLUMNS,
    )
    frame = frame[frame["profile"] == profile]
    keys = ["source", "target"]
    if frame.duplicated(keys).any():
        raise ValueError("emergence_layers.csv contains duplicate cells")
    expected = {(src, tgt) for src in languages for tgt in languages}
    if set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("emergence_layers.csv cell set is incomplete or unexpected")
    frame = frame.copy()
    for column in ("onset", "consolidation"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    return _ordered(_identified(frame, profile), ["model", "source", "target"])


def _band_vectors(
    samples: pd.DataFrame, column: str, n_bands: int, context: str
) -> np.ndarray:
    try:
        values = np.stack(
            [np.asarray(value, dtype=np.float64) for value in samples[column]]
        )
    except ValueError as exc:
        raise ValueError(f"{context}: {column} has ragged band vectors") from exc
    if (
        values.ndim != 2
        or values.shape[1] != n_bands
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"{context}: {column} must hold {n_bands} finite band values per sample"
        )
    return values


def _check_band_indices(frame: pd.DataFrame, n_bands: int, context: str) -> None:
    if sorted(frame["band"].tolist()) != list(range(1, n_bands + 1)):
        raise ValueError(
            f"{context}: band indices must be exactly 1..{n_bands} for the edges"
        )


def _group_name(grouping: str, group: object) -> str:
    """Name a summary group by what it conditions on (true or predicted class)."""
    name = CLASS_NAMES[int(group)]
    return name if grouping == "y_true" else f"predicted_{name}"


def _expected_summaries(
    samples: pd.DataFrame, signed: np.ndarray, absolute: np.ndarray
) -> dict[tuple[str, object, str], tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Recompute every persisted summary row from per-sample band vectors."""
    selectors: list[tuple[str, object, np.ndarray]] = [
        ("all", "all", np.ones(len(samples), dtype=bool))
    ]
    for column in ("y_true", "prediction"):
        values = samples[column].to_numpy().astype(np.int64)
        for group in sorted(set(values.tolist())):
            selectors.append((column, group, values == group))
    expected = {}
    for grouping, group, mask in selectors:
        for measure, matrix in (("signed", signed), ("absolute_normalized", absolute)):
            part = matrix[mask]
            n = len(part)
            mean = part.mean(axis=0)
            sem = (
                part.std(axis=0, ddof=1) / math.sqrt(n)
                if n > 1
                else np.zeros(part.shape[1])
            )
            expected[(grouping, group, measure)] = (
                n,
                mean,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
            )
    return expected


def _check_summaries(
    frame: pd.DataFrame,
    expected: Mapping[tuple[str, object, str], tuple],
    groupings: tuple[str, ...],
    n_bands: int,
    context: str,
) -> None:
    observed: dict[tuple[str, object, str], pd.DataFrame] = {}
    for (grouping, group, measure), part in frame.groupby(
        ["grouping", "group", "measure"], sort=True
    ):
        key_group = "all" if grouping == "all" else int(group)
        observed[(grouping, key_group, measure)] = part.sort_values("band")
    for grouping in groupings:
        seen = {key[1] for key in observed if key[0] == grouping}
        wanted = {key[1] for key in expected if key[0] == grouping}
        if seen != wanted:
            raise ValueError(
                f"{context}: summary grouping '{grouping}' groups {sorted(seen, key=str)} "
                f"do not match sample_relevance groups {sorted(wanted, key=str)}"
            )
    for key, (n, mean, low, high) in expected.items():
        grouping, group, measure = key
        if grouping not in groupings:
            continue
        label = f"summary grouping '{grouping}' group {group} measure {measure}"
        part = observed.get(key)
        if part is None:
            raise ValueError(f"{context}: {label} is missing")
        if part["band"].tolist() != list(range(1, n_bands + 1)):
            raise ValueError(f"{context}: {label} has invalid band indices")
        if not (part["n"] == n).all():
            raise ValueError(f"{context}: {label}: n disagrees with sample_relevance")
        for column, reference in (("mean", mean), ("ci_low", low), ("ci_high", high)):
            if not np.allclose(
                part[column].to_numpy(dtype=np.float64), reference, rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    f"{context}: {label}: {column} disagrees with sample_relevance"
                )


def _load_layer_xai_cell(
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    generation: Path,
    band_edges: BandEdges,
) -> dict[str, object]:
    context = f"{profile}/layer_{layer:02d}/{source}->{target}"
    identity = {"profile": profile, "layer": layer, "source": source, "target": target}
    n_bands = len(band_edges.edges_hz) - 1

    overall = _read_csv(generation / "dft_band_relevance.csv", _BAND_SUMMARY_COLUMNS)
    _expect_identity(overall, identity, context)
    if set(overall["grouping"]) != {"all"} or set(overall["group"].astype(str)) != {"all"}:
        raise ValueError(f"{context}: dft_band_relevance.csv must hold only the 'all' group")
    if set(overall["measure"]) != set(_MEASURES):
        raise ValueError(f"{context}: dft_band_relevance.csv measures are incomplete")
    for measure in _MEASURES:
        _check_band_indices(overall[overall["measure"] == measure], n_bands, context)

    byclass = _read_csv(
        generation / "dft_band_relevance_byclass.csv", _BAND_SUMMARY_COLUMNS
    )
    _expect_identity(byclass, identity, context)
    if set(byclass["grouping"]) != {"y_true", "prediction"}:
        raise ValueError(
            f"{context}: by-class summary must hold grouping 'y_true' and 'prediction'"
        )
    for _, part in byclass.groupby(["grouping", "group", "measure"], sort=True):
        _check_band_indices(part, n_bands, context)

    sample_path = generation / "sample_relevance.parquet"
    if not sample_path.is_file():
        raise ValueError(f"missing sample_relevance.parquet in {generation}")
    try:
        samples = pd.read_parquet(sample_path)
    except Exception as exc:
        raise ValueError(f"{context}: sample_relevance.parquet is unreadable") from exc
    missing = _SAMPLE_COLUMNS - set(samples.columns)
    if missing:
        raise ValueError(f"{context}: sample_relevance lacks columns {sorted(missing)}")
    _expect_identity(samples, identity, context)
    ids = samples["sample_id"].tolist()
    if (
        not ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or set(samples["y_true"].tolist()) != {0, 1}
        or not set(samples["prediction"].tolist()).issubset({0, 1})
    ):
        raise ValueError(
            f"{context}: sample_relevance has invalid sample IDs or labels "
            "(both y_true classes are required)"
        )
    samples = samples.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    signed = _band_vectors(samples, "band_signed", n_bands, context)
    absolute = _band_vectors(samples, "band_abs_normalized", n_bands, context)
    expected_summaries = _expected_summaries(samples, signed, absolute)
    _check_summaries(overall, expected_summaries, ("all",), n_bands, context)
    _check_summaries(
        byclass, expected_summaries, ("y_true", "prediction"), n_bands, context
    )

    score = samples["score"].to_numpy(dtype=np.float64)
    score_error = samples["score_recompute_absolute_error"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(score).all()
        or not np.isfinite(score_error).all()
        or np.any(score_error < 0)
    ):
        raise ValueError(f"{context}: score recompute errors must be finite and non-negative")

    conservation_path = generation / "attnlrp_conservation.json"
    try:
        conservation = json.loads(conservation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{context}: attnlrp_conservation.json is unreadable") from exc
    if not isinstance(conservation, dict) or any(
        conservation.get(key) != value for key, value in identity.items()
    ):
        raise ValueError(f"{context}: conservation identity does not match the cell")
    for field in _CONSERVATION_FIELDS:
        if not _is_finite_number(conservation.get(field)):
            raise ValueError(f"{context}: conservation field {field} must be finite")
    if conservation["tolerance"] <= 0 or conservation.get("n") != len(samples):
        raise ValueError(f"{context}: conservation tolerance or sample count is invalid")
    if conservation.get("validation_kind") != _VALIDATION_KIND:
        raise ValueError(
            f"{context}: validation_kind must be {_VALIDATION_KIND!r}, "
            f"found {conservation.get('validation_kind')!r}"
        )
    validation_sample_id = conservation.get("validation_sample_id")
    if (
        not isinstance(validation_sample_id, str)
        or validation_sample_id not in set(samples["sample_id"])
    ):
        raise ValueError(
            f"{context}: validation_sample_id {validation_sample_id!r} "
            "is not a sample of the XAI cohort"
        )
    rtol = float(conservation["score_recompute_rtol"])
    atol = float(conservation["score_recompute_atol"])
    if rtol <= 0 or atol <= 0:
        raise ValueError(f"{context}: score recompute rtol/atol must be positive")
    if not np.isclose(
        float(conservation["max_score_recompute_absolute_error"]),
        float(score_error.max()),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{context}: max_score_recompute_absolute_error disagrees with sample_relevance"
        )
    max_score_ratio = float((score_error / (atol + rtol * np.abs(score))).max())

    classes = dict(zip(samples["sample_id"], samples["y_true"].astype(int)))
    examples_dir = generation / "stdft_examples"
    index_rows: list[dict[str, object]] = []
    payloads: dict[tuple, dict[str, np.ndarray]] = {}
    for path in sorted(examples_dir.glob("*.npz")):
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if not _STDFT_KEYS.issubset(data.files):
                raise ValueError(f"{context}: STDFT example {path.name} lacks required arrays")
            sample_id = str(data["sample_id"].item())
            arrays = {
                name: np.array(data[name], dtype=np.float64)
                for name in ("times", "freqs", "relevance", "spectrum")
            }
            scalars = {
                name: float(data[name])
                for name in (
                    "relevance_time_sum",
                    "relevance_tf_sum",
                    "conservation_absolute_error",
                    "conservation_relative_error",
                    "conservation_tolerance",
                )
            }
        if sample_id not in classes:
            raise ValueError(f"{context}: STDFT example {sample_id!r} is not in the cohort")
        shape = (len(arrays["times"]), len(arrays["freqs"]))
        if (
            arrays["times"].ndim != 1
            or arrays["freqs"].ndim != 1
            or arrays["relevance"].shape != shape
            or arrays["spectrum"].shape != shape
            or not all(np.isfinite(value).all() for value in arrays.values())
            or not all(math.isfinite(value) for value in scalars.values())
        ):
            raise ValueError(f"{context}: STDFT example {sample_id!r} is malformed")
        for value in arrays.values():
            value.setflags(write=False)
        y_true = classes[sample_id]
        key = (profile, source, target, layer, sample_id)
        if key in payloads:
            raise ValueError(f"{context}: duplicate STDFT example {sample_id!r}")
        payloads[key] = arrays
        index_rows.append(
            {
                "layer": layer,
                "y_true": y_true,
                "class_name": CLASS_NAMES[y_true],
                "sample_id": sample_id,
                "artifact": path.name,
                "n_times": shape[0],
                "n_freqs": shape[1],
                **scalars,
            }
        )

    overall_out = overall.drop(columns=["grouping", "group"])
    byclass_out = byclass.copy()
    byclass_out["class_name"] = [
        _group_name(grouping, group)
        for grouping, group in zip(byclass_out["grouping"], byclass_out["group"])
    ]
    cohort_row = {
        "layer": layer,
        "n": len(samples),
        "n_real": int((samples["y_true"] == 0).sum()),
        "n_synthetic": int((samples["y_true"] == 1).sum()),
    }
    conservation_row = {
        "layer": layer,
        "n": len(samples),
        "n_stdft": len(index_rows),
        "validation_kind": conservation["validation_kind"],
        "validation_sample_id": validation_sample_id,
        **{field: float(conservation[field]) for field in _CONSERVATION_FIELDS},
        "max_score_recompute_ratio": max_score_ratio,
    }
    return {
        "band": overall_out,
        "byclass": byclass_out,
        "cohort_ids": tuple(
            zip(samples["sample_id"].tolist(), samples["y_true"].astype(int).tolist())
        ),
        "cohort": cohort_row,
        "conservation": conservation_row,
        "sample_relevance": samples[
            [
                "sample_id",
                "y_true",
                "prediction",
                "score",
                "band_signed",
                "band_abs_normalized",
            ]
        ].copy(),
        "stdft_rows": index_rows,
        "stdft_payloads": payloads,
    }


def _load_transitions(
    *,
    profile: str,
    source: str,
    target: str,
    generation: Path,
) -> pd.DataFrame:
    context = f"{profile}/final_trace/{source}->{target}"
    frame = _read_csv(generation / "layer_transition_metrics.csv", _TRANSITION_COLUMNS)
    _expect_identity(
        frame, {"profile": profile, "source": source, "target": target}, context
    )
    if frame.empty or frame.duplicated(
        ["sample_id", "previous_layer", "current_layer"]
    ).any():
        raise ValueError(f"{context}: transition rows are empty or duplicated")
    numeric = frame[list(_TRANSITION_METRICS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{context}: transition metrics must be finite")
    for sample_id, group in frame.groupby("sample_id", sort=True):
        observed = set(
            group[["previous_layer", "current_layer"]].itertuples(
                index=False, name=None
            )
        )
        if len(group) != 11 or observed != _EXPECTED_TRANSITIONS:
            raise ValueError(
                f"{context}: sample {sample_id!r} must contain exactly "
                "11 transitions from 1->2 through 11->12"
            )
    columns = ["profile", "source", "target", "sample_id", "previous_layer",
               "current_layer", *_TRANSITION_METRICS]
    return frame[columns].copy()


def _class_counts(cohort: Sequence[tuple[str, int]]) -> tuple[int, int]:
    return (
        sum(1 for _, label in cohort if label == 0),
        sum(1 for _, label in cohort if label == 1),
    )


def _check_fixed_cohort(
    references: dict[str, tuple[int, str, tuple]],
    target: str,
    layer: int,
    source: str,
    cohort: tuple,
) -> None:
    """The XAI cohort is fixed per target: same IDs and labels in every layer."""
    reference = references.setdefault(target, (layer, source, cohort))
    ref_layer, ref_source, ref_cohort = reference
    if cohort != ref_cohort:
        raise ValueError(
            f"XAI cohort for target {target!r} differs at layer {layer} "
            f"({source}->{target}) from layer {ref_layer} ({ref_source}->{target}): "
            "sample_id/label pairs differ; class counts (real, synthetic) "
            f"{_class_counts(ref_cohort)} vs {_class_counts(cohort)}"
        )


def load_report_source(source: ReportSource) -> LoadedReportSource:
    """Read persisted scientific artifacts from a validated source without writing."""
    if not isinstance(source, ReportSource):
        raise TypeError("source must be a validated ReportSource")
    profile = source.identity.profile
    languages = source.identity.languages
    paths = LayerwiseSuitePaths(source.root)
    band_edges = {
        language: _resolve_band_edges(source, language) for language in languages
    }
    _check_persisted_band_counts(source, band_edges)

    performance, cell_predictions = _load_performance(source, paths)
    probe_stability = _load_probe_stability_summary(source, paths)
    layer_faithfulness = _load_layer_faithfulness_summaries(source, paths)
    emergence = _load_emergence(source)

    band_frames, class_frames = [], []
    cohort_rows, conservation_rows, stdft_rows, sample_relevance_rows = [], [], [], []
    payloads: dict[tuple, dict[str, np.ndarray]] = {}
    reference_cohorts: dict[str, tuple[int, str, tuple]] = {}
    for (_, layer, src, tgt), generation in sorted(source.layer_xai_generations.items()):
        cell = _load_layer_xai_cell(
            profile=profile,
            layer=layer,
            source=src,
            target=tgt,
            generation=generation,
            band_edges=band_edges[tgt],
        )
        context = f"{profile}/layer_{layer:02d}/{src}->{tgt}"
        prediction_subset = cell_predictions.loc[
            (cell_predictions["layer"] == layer)
            & (cell_predictions["source"] == src)
            & (cell_predictions["target"] == tgt),
            ["sample_id", "y_true", "score", "prediction", "threshold"],
        ]
        _validate_prediction_xai_alignment(
            context, prediction_subset, cell["sample_relevance"]
        )
        _check_fixed_cohort(reference_cohorts, tgt, layer, src, cell["cohort_ids"])
        band_frames.append(cell["band"])
        class_frames.append(cell["byclass"])
        identity = {"source": src, "target": tgt}
        cohort_rows.append({**identity, **cell["cohort"]})
        conservation_rows.append({**identity, **cell["conservation"]})
        stdft_rows.extend({**identity, **row} for row in cell["stdft_rows"])
        sample_relevance_rows.extend(
            {"layer": layer, **identity, **row.to_dict()}
            for _, row in cell["sample_relevance"].iterrows()
        )
        payloads.update(cell["stdft_payloads"])

    transition_frames = [
        _load_transitions(profile=profile, source=src, target=tgt, generation=generation)
        for (_, src, tgt), generation in sorted(source.final_trace_generations.items())
    ]

    band = _identified(pd.concat(band_frames, ignore_index=True), profile)
    byclass = _identified(pd.concat(class_frames, ignore_index=True), profile)
    transitions = _identified(pd.concat(transition_frames, ignore_index=True), profile)
    stdft = _identified(pd.DataFrame(stdft_rows), profile)
    return LoadedReportSource(
        source=source,
        performance=performance,
        emergence=emergence,
        band_relevance=_ordered(
            band, ["model", "source", "target", "layer", "measure", "band"]
        ),
        class_relevance=_ordered(
            byclass,
            ["model", "source", "target", "grouping", "group", "layer", "measure", "band"],
        ),
        cohort=_ordered(
            _identified(pd.DataFrame(cohort_rows), profile),
            ["model", "source", "target", "layer"],
        ),
        conservation=_ordered(
            _identified(pd.DataFrame(conservation_rows), profile),
            ["model", "source", "target", "layer"],
        ),
        transitions=_ordered(
            transitions,
            ["model", "source", "target", "sample_id", "previous_layer"],
        ),
        stdft_examples=_ordered(
            stdft,
            ["model", "source", "target", "layer", "y_true", "sample_id"],
        ),
        stdft_payloads={
            key: payload for key, payload in payloads.items()
        },
        probe_stability=probe_stability,
        layer_faithfulness=layer_faithfulness,
        cell_predictions=cell_predictions,
        sample_relevance=_ordered(
            _identified(pd.DataFrame(sample_relevance_rows), profile),
            list(_PER_SAMPLE_KEYS),
        ),
        band_edges=tuple(band_edges[language] for language in languages),
    )


def _band_edge_tables(
    edges: Sequence[BandEdges],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    band_rows, provenance_rows = [], []
    for item in edges:
        values = np.asarray(item.edges_hz, dtype=np.float64)
        for band in range(1, len(values)):
            low, high = float(values[band - 1]), float(values[band])
            band_rows.append(
                {
                    "model": item.model,
                    "language": item.language,
                    "band": band,
                    "low_hz": low,
                    "high_hz": high,
                    "center_hz": (low + high) / 2.0,
                    "label": f"{low:.0f}\u2013{high:.0f} Hz",
                }
            )
        provenance_rows.append(
            {
                "model": item.model,
                "language": item.language,
                "origin": item.origin,
                "reason": item.reason,
                "n_bands": item.n_bands,
                "f_min": item.f_min,
                "f_max": item.f_max,
            }
        )
    return (
        _ordered(pd.DataFrame(band_rows), ["model", "language", "band"]),
        _ordered(pd.DataFrame(provenance_rows), ["model", "language"]),
    )


def _transition_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "source", "target", "language", "previous_layer", "current_layer"]
    rows = []
    for identity, group in transitions.groupby(keys, sort=True):
        row = dict(zip(keys, identity))
        n = int(group["sample_id"].nunique())
        row["transition"] = f"{row['previous_layer']}\u2192{row['current_layer']}"
        row["n"] = n
        for metric in _TRANSITION_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            mean = float(values.mean())
            sem = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = mean - 1.96 * sem
            row[f"{metric}_ci_high"] = mean + 1.96 * sem
        rows.append(row)
    return _ordered(
        pd.DataFrame(rows),
        ["model", "source", "target", "previous_layer"],
    )


def _omitted_comparisons(
    completeness: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    """Reasons for comparisons that are *not* delivered this edition.

    A comparison is removed from the omission map only once its table is complete
    against the expected grid. Until then it reports why it is missing: too few
    models/languages, or a table that is only partially populated relative to the
    actual model/language/source/target/layer availability.
    """
    omitted: dict[str, str] = {}
    for name, info in completeness.items():
        status = info.get("status")
        if status in ("single_model", "single_language"):
            omitted[name] = str(status)
        elif not info.get("complete", False):
            omitted[name] = "unavailable_incomplete_table"
    return omitted


def build_transfer_selection(performance: pd.DataFrame) -> pd.DataFrame:
    """Select one layer per model/source/target by ROC-AUC only.

    MCC is copied from that same selected row; it is never maximized
    independently. Ties in ROC-AUC resolve to the earliest layer.
    """
    columns = ["model", "source", "target", "selected_layer", "roc_auc", "mcc"]
    required = {"model", "source", "target", "layer", "roc_auc", "mcc"}
    if not required <= set(performance.columns):
        raise ValueError("performance lacks columns required for transfer reduction")
    ranked = performance.sort_values(
        ["model", "source", "target", "roc_auc", "layer"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby(
        ["model", "source", "target"], sort=True, as_index=False
    ).first()
    selected = selected.rename(columns={"layer": "selected_layer"})
    return selected.loc[:, columns].reset_index(drop=True)


def build_xai_performance_associations(
    performance: pd.DataFrame, band_relevance: pd.DataFrame
) -> pd.DataFrame:
    """Describe layer-wise ROC-AUC association with spectral concentration.

    Concentration is the maximum, across frequency bands, of the cohort mean
    absolute-normalized band mass. Spearman's rho is reported only when all
    twelve encoder layers are available and both series vary.
    """
    keys = ["model", "source", "target"]
    selected = band_relevance[
        band_relevance["measure"] == "absolute_normalized"
    ]
    concentration = (
        selected.groupby([*keys, "layer"], sort=True)["mean"]
        .max()
        .rename("spectral_concentration")
        .reset_index()
    )
    joined = performance[[*keys, "layer", "roc_auc"]].merge(
        concentration, on=[*keys, "layer"], how="inner", validate="one_to_one"
    )
    identities = (
        performance[keys].drop_duplicates().sort_values(keys).itertuples(
            index=False, name=None
        )
    )
    rows: list[dict[str, object]] = []
    for identity in identities:
        group = joined
        for key, value in zip(keys, identity):
            group = group[group[key] == value]
        group = group.sort_values("layer")
        layers = tuple(int(value) for value in group["layer"])
        row: dict[str, object] = {
            **dict(zip(keys, identity)),
            "n_layers": len(layers),
            "concentration_metric": "maximum_mean_band_mass",
            "association_method": "spearman_rank_correlation",
            "rho": np.nan,
            "status": "unavailable_requires_12_layers",
        }
        values = group[["roc_auc", "spectral_concentration"]].to_numpy(
            dtype=np.float64
        )
        if layers == LAYER_ORDER and np.isfinite(values).all():
            ranks = group[["roc_auc", "spectral_concentration"]].rank(
                method="average"
            )
            if all(ranks[column].nunique() > 1 for column in ranks):
                row["rho"] = float(
                    np.corrcoef(ranks["roc_auc"], ranks["spectral_concentration"])[
                        0, 1
                    ]
                )
                row["status"] = "available"
            else:
                row["status"] = "unavailable_constant_series"
        elif len(layers) == len(LAYER_ORDER):
            row["status"] = "unavailable_invalid_or_duplicate_layers"
        rows.append(row)
    return _ordered(pd.DataFrame(rows), keys)


# ---------------------------------------------------------------------------
# Cross-model / cross-language comparisons (pure, deterministic builders)
#
# Every builder is a pure function over the identity-labelled report tables.
# It reads nothing from disk, loads no model, and produces byte-identical output
# regardless of the order of its input rows. All uncertainty is quantified with
# the fixed bootstrap contract: BOOTSTRAP_RESAMPLES stratified resamples, base
# seed BOOTSTRAP_SEED (combined with a stable hash of the row identity so order
# cannot change the output), and a percentile BOOTSTRAP_CI_LEVEL interval. No
# p-values are produced anywhere.
# ---------------------------------------------------------------------------

_ENCODER_AGREEMENT_COLUMNS = (
    "model_a",
    "model_b",
    "source",
    "target",
    "layer",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
    "valid_bootstrap_n",
    "degenerate_n",
)
_LANGUAGE_SHIFT_COLUMNS = (
    "model",
    "source",
    "target_a",
    "target_b",
    "layer",
    "group",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n_a",
    "n_b",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
    "valid_bootstrap_n",
    "degenerate_n",
)
_DIAGONAL_COMPARISON_COLUMNS = (
    "model",
    "source",
    "target",
    "layer",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
    "valid_bootstrap_n",
    "degenerate_n",
)
_SPECTRAL_DIVERGENCE_COLUMNS = (
    "model",
    "language_a",
    "language_b",
    "layer",
    "group",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n_a",
    "n_b",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
    "valid_bootstrap_n",
    "degenerate_n",
)


def _empty_comparison(columns: Sequence[str]) -> pd.DataFrame:
    """A typed, empty comparison frame used for single model/language backends."""
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _finalize_comparison(
    rows: Sequence[Mapping[str, object]], columns: Sequence[str], keys: Sequence[str]
) -> pd.DataFrame:
    if not rows:
        return _empty_comparison(columns)
    frame = pd.DataFrame(list(rows))[list(columns)]
    return frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def _comparison_rng(*identity: object) -> np.random.Generator:
    """A generator whose seed is stably derived from the base seed and identity."""
    digest = hashlib.sha256(
        "||".join(str(part) for part in identity).encode("utf-8")
    ).digest()
    return np.random.default_rng([BOOTSTRAP_SEED, int.from_bytes(digest[:8], "big")])


def _resample_matrix(
    rng: np.random.Generator, labels: np.ndarray, n_resamples: int
) -> np.ndarray:
    """A ``(n_resamples, n)`` matrix of stratified bootstrap row positions.

    Each class is resampled with replacement to its original count, so every
    resample preserves the class balance. Columns are grouped by class; because
    all downstream statistics are permutation-invariant within a resample this
    grouping does not affect any result.
    """
    labels = np.asarray(labels)
    blocks = []
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        blocks.append(members[draws])
    return np.concatenate(blocks, axis=1)


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    numerator = (xm * ym).sum(axis=1)
    denominator = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    out = np.full(len(x), np.nan)
    nonzero = denominator > 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman's rho per row; NaN where either row is a constant series."""
    from scipy.stats import rankdata

    return _rowwise_pearson(rankdata(a, axis=1), rankdata(b, axis=1))


def _rowwise_kappa(pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    """Binary Cohen's kappa per row; NaN on a degenerate (p_e == 1) denominator."""
    n = pred_a.shape[1]
    agreement = (pred_a == pred_b).mean(axis=1)
    pa1 = pred_a.mean(axis=1)
    pb1 = pred_b.mean(axis=1)
    expected = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)
    out = np.full(pred_a.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = expected < 1.0
        out[valid] = (agreement[valid] - expected[valid]) / (1.0 - expected[valid])
    return out


def _rowwise_jensenshannon(p: np.ndarray, q: np.ndarray, base: float) -> np.ndarray:
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(a > 0, a * np.log(a / b), 0.0)
        return term.sum(axis=1)

    divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    divergence = np.clip(divergence, 0.0, None)
    return np.sqrt(divergence / math.log(base))


def _rowwise_auc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """ROC-AUC per row via average ranks (matches roc_auc_score)."""
    from scipy.stats import rankdata

    ranks = rankdata(scores, axis=1)
    positive = labels == 1
    n_pos = positive.sum(axis=1).astype(np.float64)
    n_neg = labels.shape[1] - n_pos
    out = np.full(labels.shape[0], np.nan)
    valid = (n_pos > 0) & (n_neg > 0)
    rank_sum = (ranks * positive).sum(axis=1)
    out[valid] = (
        rank_sum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0
    ) / (n_pos[valid] * n_neg[valid])
    return out


def _rowwise_mcc(labels: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row Matthews correlation and a boolean degenerate-denominator mask."""
    tp = ((prediction == 1) & (labels == 1)).sum(axis=1).astype(np.float64)
    tn = ((prediction == 0) & (labels == 0)).sum(axis=1).astype(np.float64)
    fp = ((prediction == 1) & (labels == 0)).sum(axis=1).astype(np.float64)
    fn = ((prediction == 0) & (labels == 1)).sum(axis=1).astype(np.float64)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    degenerate = denominator == 0
    out = np.zeros(labels.shape[0])
    nonzero = ~degenerate
    out[nonzero] = (tp[nonzero] * tn[nonzero] - fp[nonzero] * fn[nonzero]) / denominator[
        nonzero
    ]
    return out, degenerate


def _resampled_means(vectors: np.ndarray, index_matrix: np.ndarray) -> np.ndarray:
    """Mean band vector for every resample: ``(n_resamples, n_bands)``."""
    return vectors[index_matrix].mean(axis=1)


def _wasserstein_batch(
    support_a: np.ndarray,
    support_b: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
) -> np.ndarray:
    """1-D Wasserstein distance for many weight rows over fixed supports.

    Mirrors ``scipy.stats.wasserstein_distance`` (which normalizes the weights)
    but shares the support-dependent structure across all resample rows.
    """
    sorter_a = np.argsort(support_a, kind="mergesort")
    sorter_b = np.argsort(support_b, kind="mergesort")
    values_a = support_a[sorter_a]
    values_b = support_b[sorter_b]
    all_values = np.concatenate((values_a, values_b))
    all_values.sort(kind="mergesort")
    deltas = np.diff(all_values)
    a_indices = values_a.searchsorted(all_values[:-1], side="right")
    b_indices = values_b.searchsorted(all_values[:-1], side="right")

    def _cdf(weights: np.ndarray, sorter: np.ndarray, indices: np.ndarray) -> np.ndarray:
        ordered = weights[:, sorter]
        cumulative = np.concatenate(
            (np.zeros((ordered.shape[0], 1)), np.cumsum(ordered, axis=1)), axis=1
        )
        totals = cumulative[:, -1:]
        return cumulative[:, indices] / totals

    cdf_a = _cdf(weights_a, sorter_a, a_indices)
    cdf_b = _cdf(weights_b, sorter_b, b_indices)
    return (np.abs(cdf_a - cdf_b) * deltas).sum(axis=1)


def _bootstrap_ci(draws: np.ndarray) -> tuple[float, float, int, str]:
    """Percentile CI plus the finite-draw bookkeeping needed to trust it.

    Returns ``(ci_low, ci_high, valid_bootstrap_n, status)``. The bounds are
    never clamped to the point estimate. When fewer than ``_MIN_VALID_RESAMPLES``
    draws are finite the interval is refused (NaN) with an explicit status; when
    some but not all draws are finite the status flags the reduction rather than
    presenting a silent ``ok``.
    """
    draws = np.asarray(draws, dtype=np.float64)
    requested = int(draws.size)
    finite = draws[np.isfinite(draws)]
    valid_n = int(finite.size)
    if valid_n < _MIN_VALID_RESAMPLES:
        return float("nan"), float("nan"), valid_n, "insufficient_bootstrap"
    low, high = np.percentile(finite, _CI_PERCENTILES)
    status = "ok" if valid_n == requested else "reduced_bootstrap"
    return float(low), float(high), valid_n, status


def _as_distribution(values: object, context: str) -> np.ndarray:
    """Validate a normalized relevance vector, failing closed on zero vectors."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError(
            f"{context}: normalized relevance must be a finite non-negative vector"
        )
    total = float(arr.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"{context}: normalized relevance must sum to 1 (got {total:.6g}); "
            "zero or unnormalized vectors are rejected rather than fabricated"
        )
    return arr


def _distribution_matrix(frame: pd.DataFrame, context: str) -> np.ndarray:
    vectors = [
        _as_distribution(value, context) for value in frame["band_abs_normalized"]
    ]
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"{context}: relevance vectors have inconsistent band counts")
    return np.stack(vectors)


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho; NaN (constant-series) when either series does not vary."""
    from scipy.stats import rankdata

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via average ranks; matches sklearn.metrics.roc_auc_score."""
    from scipy.stats import rankdata

    ranks = rankdata(scores)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def _fast_mcc(labels: np.ndarray, prediction: np.ndarray) -> tuple[float, bool]:
    """Matthews correlation and whether its denominator was degenerate.

    A degenerate denominator (a prediction or truth vector with a single class)
    yields 0.0 as sklearn does, but the degeneracy flag is returned so callers
    can surface it instead of silently reporting a bare 0.0.
    """
    tp = float(((prediction == 1) & (labels == 1)).sum())
    tn = float(((prediction == 0) & (labels == 0)).sum())
    fp = float(((prediction == 1) & (labels == 0)).sum())
    fn = float(((prediction == 0) & (labels == 1)).sum())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0, True
    return (tp * tn - fp * fn) / denominator, False


def _aligned_pair(
    left: pd.DataFrame, right: pd.DataFrame, context: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort two cohorts by sample_id and require identical IDs and labels."""
    left = left.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    right = right.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if left["sample_id"].tolist() != right["sample_id"].tolist():
        raise ValueError(f"{context}: cohorts have non-identical sample IDs")
    if not np.array_equal(
        left["y_true"].to_numpy(dtype=np.int64),
        right["y_true"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"{context}: cohorts disagree on y_true labels")
    return left, right


def _cell_keys(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple]:
    unique = frame[list(columns)].drop_duplicates()
    return sorted(tuple(row) for row in unique.itertuples(index=False, name=None))


def _metadata() -> dict[str, object]:
    return {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "ci_level": BOOTSTRAP_CI_LEVEL,
    }


def build_encoder_agreement(
    cell_predictions: pd.DataFrame, sample_relevance: pd.DataFrame
) -> pd.DataFrame:
    """Agreement between every unordered model pair on each source/target/layer.

    Full predictions yield the score Spearman rho, the fraction of matching
    predictions and Cohen's kappa; the XAI cohort yields the mean per-sample
    cosine similarity and Jensen-Shannon distance between absolute-normalized
    relevance vectors. Each metric carries its own CI, n, unit and status.
    """
    from sklearn.metrics import cohen_kappa_score

    models = sorted(set(cell_predictions["model"]))
    if len(models) < 2:
        return _empty_comparison(_ENCODER_AGREEMENT_COLUMNS)

    rows: list[dict[str, object]] = []
    meta = _metadata()
    for index_a in range(len(models)):
        for index_b in range(index_a + 1, len(models)):
            model_a, model_b = models[index_a], models[index_b]

            def _row(src, tgt, layer, metric, unit, estimate, ci_low, ci_high, n,
                     status, valid_bootstrap_n=0, degenerate_n=0):
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "source": src,
                        "target": tgt,
                        "layer": int(layer),
                        "metric": metric,
                        "unit": unit,
                        "estimate": float(estimate),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "n": int(n),
                        "status": status,
                        "valid_bootstrap_n": int(valid_bootstrap_n),
                        "degenerate_n": int(degenerate_n),
                        **meta,
                    }
                )

            pred_a = cell_predictions[cell_predictions["model"] == model_a]
            pred_b = cell_predictions[cell_predictions["model"] == model_b]
            shared_pred = sorted(
                set(_cell_keys(pred_a, ("source", "target", "layer")))
                & set(_cell_keys(pred_b, ("source", "target", "layer")))
            )
            for source, target, layer in shared_pred:
                mask_a = (
                    (pred_a["source"] == source)
                    & (pred_a["target"] == target)
                    & (pred_a["layer"] == layer)
                )
                mask_b = (
                    (pred_b["source"] == source)
                    & (pred_b["target"] == target)
                    & (pred_b["layer"] == layer)
                )
                context = f"encoder_agreement {model_a}|{model_b} {source}->{target} L{layer}"
                left, right = _aligned_pair(pred_a[mask_a], pred_b[mask_b], context)
                labels = left["y_true"].to_numpy(dtype=np.int64)
                score_a = left["score"].to_numpy(dtype=np.float64)
                score_b = right["score"].to_numpy(dtype=np.float64)
                pred_la = left["prediction"].to_numpy(dtype=np.int64)
                pred_lb = right["prediction"].to_numpy(dtype=np.int64)
                n = len(labels)
                identity = (model_a, model_b, source, target, layer)
                matrix = _resample_matrix(
                    _comparison_rng(*identity, "predictions"), labels, BOOTSTRAP_RESAMPLES
                )

                rho = _spearman_rho(score_a, score_b)
                if math.isnan(rho):
                    _row(source, target, layer, "score_spearman", "rho",
                         float("nan"), float("nan"), float("nan"), n, "constant_series")
                else:
                    low, high, valid_n, status = _bootstrap_ci(
                        _rowwise_spearman(score_a[matrix], score_b[matrix])
                    )
                    _row(source, target, layer, "score_spearman", "rho", rho, low, high,
                         n, status, valid_n)

                match = (pred_la == pred_lb).astype(np.float64)
                low, high, valid_n, status = _bootstrap_ci(match[matrix].mean(axis=1))
                _row(source, target, layer, "prediction_agreement", "fraction",
                     float(match.mean()), low, high, n, status, valid_n)

                with np.errstate(invalid="ignore", divide="ignore"):
                    kappa = float(cohen_kappa_score(pred_la, pred_lb))
                if not math.isfinite(kappa):
                    _row(source, target, layer, "cohen_kappa", "kappa",
                         float("nan"), float("nan"), float("nan"), n, "degenerate")
                else:
                    draws = _rowwise_kappa(pred_la[matrix], pred_lb[matrix])
                    low, high, valid_n, status = _bootstrap_ci(draws)
                    degenerate_n = int(BOOTSTRAP_RESAMPLES - valid_n)
                    _row(source, target, layer, "cohen_kappa", "kappa", kappa, low, high,
                         n, status, valid_n, degenerate_n)

            rel_a = sample_relevance[sample_relevance["model"] == model_a]
            rel_b = sample_relevance[sample_relevance["model"] == model_b]
            shared_rel = sorted(
                set(_cell_keys(rel_a, ("source", "target", "layer")))
                & set(_cell_keys(rel_b, ("source", "target", "layer")))
            )
            for source, target, layer in shared_rel:
                mask_a = (
                    (rel_a["source"] == source)
                    & (rel_a["target"] == target)
                    & (rel_a["layer"] == layer)
                )
                mask_b = (
                    (rel_b["source"] == source)
                    & (rel_b["target"] == target)
                    & (rel_b["layer"] == layer)
                )
                context = (
                    f"encoder_agreement/xai {model_a}|{model_b} {source}->{target} L{layer}"
                )
                left, right = _aligned_pair(rel_a[mask_a], rel_b[mask_b], context)
                labels = left["y_true"].to_numpy(dtype=np.int64)
                matrix_a = _distribution_matrix(left, context)
                matrix_b = _distribution_matrix(right, context)
                n = len(labels)
                identity = (model_a, model_b, source, target, layer)
                if matrix_a.shape[1] != matrix_b.shape[1]:
                    # The two encoders report on different band grids, so their
                    # relevance vectors are not comparable element-wise. Record
                    # the fact explicitly rather than fabricating a distance.
                    for metric, unit in (
                        ("relevance_cosine_similarity", "cosine"),
                        ("relevance_jensen_shannon_distance", "bits"),
                    ):
                        _row(source, target, layer, metric, unit, float("nan"),
                             float("nan"), float("nan"), n, "incomparable_band_grid")
                    continue
                norms = np.linalg.norm(matrix_a, axis=1) * np.linalg.norm(matrix_b, axis=1)
                cosine = (matrix_a * matrix_b).sum(axis=1) / norms
                js = _rowwise_jensenshannon(matrix_a, matrix_b, _JS_BASE)
                matrix = _resample_matrix(
                    _comparison_rng(*identity, "relevance"), labels, BOOTSTRAP_RESAMPLES
                )
                for metric, unit, values in (
                    ("relevance_cosine_similarity", "cosine", cosine),
                    ("relevance_jensen_shannon_distance", "bits", js),
                ):
                    low, high, valid_n, status = _bootstrap_ci(values[matrix].mean(axis=1))
                    _row(source, target, layer, metric, unit,
                         float(values.mean()), low, high, n, status, valid_n)

    return _finalize_comparison(
        rows,
        _ENCODER_AGREEMENT_COLUMNS,
        ["model_a", "model_b", "source", "target", "layer", "metric"],
    )


def build_language_shift(sample_relevance: pd.DataFrame) -> pd.DataFrame:
    """Jensen-Shannon distance between a fixed model/source's relevance under two
    target languages, per layer and class group.

    Different target languages describe different test audio, so the two mean
    distributions are resampled independently (each stratified by class).
    """
    from scipy.spatial.distance import jensenshannon

    targets = sorted(set(sample_relevance["target"]))
    if len(targets) < 2:
        return _empty_comparison(_LANGUAGE_SHIFT_COLUMNS)

    rows: list[dict[str, object]] = []
    meta = _metadata()
    for model in sorted(set(sample_relevance["model"])):
        model_frame = sample_relevance[sample_relevance["model"] == model]
        for source in sorted(set(model_frame["source"])):
            source_frame = model_frame[model_frame["source"] == source]
            available_targets = sorted(set(source_frame["target"]))
            for index_a in range(len(available_targets)):
                for index_b in range(index_a + 1, len(available_targets)):
                    target_a = available_targets[index_a]
                    target_b = available_targets[index_b]
                    for layer in sorted(set(source_frame["layer"])):
                        layer_frame = source_frame[source_frame["layer"] == layer]
                        cell_a = layer_frame[layer_frame["target"] == target_a]
                        cell_b = layer_frame[layer_frame["target"] == target_b]
                        if cell_a.empty or cell_b.empty:
                            continue
                        for group, class_value in _COMPARISON_GROUPS:
                            sub_a = (
                                cell_a
                                if class_value is None
                                else cell_a[cell_a["y_true"] == class_value]
                            )
                            sub_b = (
                                cell_b
                                if class_value is None
                                else cell_b[cell_b["y_true"] == class_value]
                            )
                            if sub_a.empty or sub_b.empty:
                                continue
                            context = (
                                f"language_shift {model} {source}->"
                                f"{target_a}|{target_b} L{layer} {group}"
                            )
                            matrix_a = _distribution_matrix(
                                sub_a.sort_values("sample_id", kind="mergesort"), context
                            )
                            matrix_b = _distribution_matrix(
                                sub_b.sort_values("sample_id", kind="mergesort"), context
                            )
                            labels_a = (
                                sub_a.sort_values("sample_id", kind="mergesort")[
                                    "y_true"
                                ].to_numpy(dtype=np.int64)
                            )
                            labels_b = (
                                sub_b.sort_values("sample_id", kind="mergesort")[
                                    "y_true"
                                ].to_numpy(dtype=np.int64)
                            )

                            estimate = float(
                                jensenshannon(
                                    matrix_a.mean(axis=0),
                                    matrix_b.mean(axis=0),
                                    base=_JS_BASE,
                                )
                            )
                            rng = _comparison_rng(
                                model, source, target_a, target_b, layer, group
                            )
                            means_a = _resampled_means(
                                matrix_a,
                                _resample_matrix(rng, labels_a, BOOTSTRAP_RESAMPLES),
                            )
                            means_b = _resampled_means(
                                matrix_b,
                                _resample_matrix(rng, labels_b, BOOTSTRAP_RESAMPLES),
                            )
                            draws = _rowwise_jensenshannon(means_a, means_b, _JS_BASE)
                            low, high, valid_n, status = _bootstrap_ci(draws)
                            rows.append(
                                {
                                    "model": model,
                                    "source": source,
                                    "target_a": target_a,
                                    "target_b": target_b,
                                    "layer": int(layer),
                                    "group": group,
                                    "metric": "jensen_shannon_distance",
                                    "unit": "bits",
                                    "estimate": estimate,
                                    "ci_low": low,
                                    "ci_high": high,
                                    "n_a": int(len(matrix_a)),
                                    "n_b": int(len(matrix_b)),
                                    "status": status,
                                    "valid_bootstrap_n": int(valid_n),
                                    "degenerate_n": 0,
                                    **meta,
                                }
                            )
    return _finalize_comparison(
        rows,
        _LANGUAGE_SHIFT_COLUMNS,
        ["model", "source", "target_a", "target_b", "layer", "group"],
    )


def build_diagonal_vs_offdiagonal(cell_predictions: pd.DataFrame) -> pd.DataFrame:
    """Delta (offdiagonal minus diagonal) of ROC-AUC and MCC on the shared test
    set, per model/target/offdiagonal-source/layer.

    Both cells describe the same target audio, so the stratified bootstrap is
    paired: one resample of sample positions is applied to both cells. The IDs
    and labels must match exactly or the comparison fails closed.
    """
    targets = sorted(set(cell_predictions["target"]))
    if len(set(cell_predictions["source"]) | set(cell_predictions["target"])) < 2:
        return _empty_comparison(_DIAGONAL_COMPARISON_COLUMNS)

    rows: list[dict[str, object]] = []
    meta = _metadata()
    for model in sorted(set(cell_predictions["model"])):
        model_frame = cell_predictions[cell_predictions["model"] == model]
        for target in targets:
            to_target = model_frame[model_frame["target"] == target]
            diagonal = to_target[to_target["source"] == target]
            if diagonal.empty:
                continue
            offdiagonal_sources = sorted(
                set(to_target[to_target["source"] != target]["source"])
            )
            for source in offdiagonal_sources:
                offdiagonal = to_target[to_target["source"] == source]
                for layer in sorted(set(offdiagonal["layer"])):
                    diag_cell = diagonal[diagonal["layer"] == layer]
                    off_cell = offdiagonal[offdiagonal["layer"] == layer]
                    if diag_cell.empty or off_cell.empty:
                        continue
                    context = (
                        f"diagonal_vs_offdiagonal {model} {source}->{target} L{layer}"
                    )
                    diag_sorted, off_sorted = _aligned_pair(diag_cell, off_cell, context)
                    labels = diag_sorted["y_true"].to_numpy(dtype=np.int64)
                    diag_score = diag_sorted["score"].to_numpy(dtype=np.float64)
                    off_score = off_sorted["score"].to_numpy(dtype=np.float64)
                    diag_pred = diag_sorted["prediction"].to_numpy(dtype=np.int64)
                    off_pred = off_sorted["prediction"].to_numpy(dtype=np.int64)
                    n = len(labels)
                    identity = (model, source, target, layer)
                    matrix = _resample_matrix(
                        _comparison_rng(*identity), labels, BOOTSTRAP_RESAMPLES
                    )
                    labels_rows = labels[matrix]
                    auc_draws = _rowwise_auc(labels_rows, off_score[matrix]) - _rowwise_auc(
                        labels_rows, diag_score[matrix]
                    )
                    off_mcc_draws, off_mcc_deg = _rowwise_mcc(labels_rows, off_pred[matrix])
                    diag_mcc_draws, diag_mcc_deg = _rowwise_mcc(labels_rows, diag_pred[matrix])
                    mcc_draws = off_mcc_draws - diag_mcc_draws
                    mcc_degenerate = int((off_mcc_deg | diag_mcc_deg).sum())

                    off_mcc_point, off_point_deg = _fast_mcc(labels, off_pred)
                    diag_mcc_point, diag_point_deg = _fast_mcc(labels, diag_pred)

                    auc_low, auc_high, auc_valid, auc_status = _bootstrap_ci(auc_draws)
                    rows.append(
                        {
                            "model": model,
                            "source": source,
                            "target": target,
                            "layer": int(layer),
                            "metric": "delta_roc_auc",
                            "unit": "dimensionless",
                            "estimate": float(
                                _fast_auc(labels, off_score) - _fast_auc(labels, diag_score)
                            ),
                            "ci_low": float(auc_low),
                            "ci_high": float(auc_high),
                            "n": int(n),
                            "status": auc_status,
                            "valid_bootstrap_n": int(auc_valid),
                            "degenerate_n": 0,
                            **meta,
                        }
                    )

                    mcc_low, mcc_high, mcc_valid, mcc_boot_status = _bootstrap_ci(mcc_draws)
                    point_degenerate = off_point_deg or diag_point_deg
                    mcc_status = (
                        "degenerate_mcc_denominator"
                        if (point_degenerate or mcc_degenerate > 0)
                        else mcc_boot_status
                    )
                    rows.append(
                        {
                            "model": model,
                            "source": source,
                            "target": target,
                            "layer": int(layer),
                            "metric": "delta_mcc",
                            "unit": "dimensionless",
                            "estimate": float(off_mcc_point - diag_mcc_point),
                            "ci_low": float(mcc_low),
                            "ci_high": float(mcc_high),
                            "n": int(n),
                            "status": mcc_status,
                            "valid_bootstrap_n": int(mcc_valid),
                            "degenerate_n": mcc_degenerate,
                            **meta,
                        }
                    )
    return _finalize_comparison(
        rows,
        _DIAGONAL_COMPARISON_COLUMNS,
        ["model", "source", "target", "layer", "metric"],
    )


def build_spectral_divergence(
    sample_relevance: pd.DataFrame, band_edges: pd.DataFrame
) -> pd.DataFrame:
    """Wasserstein distance (in Hz) between two languages' diagonal relevance
    distributions, per model, layer and class group.

    The support of each distribution is that model/language's recorded band
    centers. Different languages describe different audio, so the two
    distributions are resampled independently (each stratified by class).
    """
    from scipy.stats import wasserstein_distance

    diagonal = sample_relevance[sample_relevance["source"] == sample_relevance["target"]]
    languages = sorted(set(diagonal["target"]))
    if len(languages) < 2:
        return _empty_comparison(_SPECTRAL_DIVERGENCE_COLUMNS)

    centers = {
        (str(model), str(language)): group.sort_values("band")["center_hz"].to_numpy(
            dtype=np.float64
        )
        for (model, language), group in band_edges.groupby(["model", "language"])
    }

    rows: list[dict[str, object]] = []
    meta = _metadata()
    for model in sorted(set(diagonal["model"])):
        model_frame = diagonal[diagonal["model"] == model]
        present = sorted(set(model_frame["target"]))
        for index_a in range(len(present)):
            for index_b in range(index_a + 1, len(present)):
                language_a = present[index_a]
                language_b = present[index_b]
                # Keys are stored as plain ``str`` so numpy string dtypes from the
                # grouped frame cannot miss an otherwise-equal center entry.
                support_a = centers.get((str(model), str(language_a)))
                support_b = centers.get((str(model), str(language_b)))
                if support_a is None or support_b is None:
                    raise ValueError(
                        f"spectral_divergence {model} {language_a}|{language_b}: "
                        "missing recorded band centers"
                    )
                for layer in sorted(set(model_frame["layer"])):
                    layer_frame = model_frame[model_frame["layer"] == layer]
                    cell_a = layer_frame[layer_frame["target"] == language_a]
                    cell_b = layer_frame[layer_frame["target"] == language_b]
                    if cell_a.empty or cell_b.empty:
                        continue
                    for group, class_value in _COMPARISON_GROUPS:
                        sub_a = (
                            cell_a
                            if class_value is None
                            else cell_a[cell_a["y_true"] == class_value]
                        )
                        sub_b = (
                            cell_b
                            if class_value is None
                            else cell_b[cell_b["y_true"] == class_value]
                        )
                        if sub_a.empty or sub_b.empty:
                            continue
                        context = (
                            f"spectral_divergence {model} {language_a}|{language_b} "
                            f"L{layer} {group}"
                        )
                        sorted_a = sub_a.sort_values("sample_id", kind="mergesort")
                        sorted_b = sub_b.sort_values("sample_id", kind="mergesort")
                        matrix_a = _distribution_matrix(sorted_a, context)
                        matrix_b = _distribution_matrix(sorted_b, context)
                        if matrix_a.shape[1] != len(support_a) or matrix_b.shape[1] != len(
                            support_b
                        ):
                            raise ValueError(
                                f"{context}: band vectors disagree with recorded centers"
                            )
                        labels_a = sorted_a["y_true"].to_numpy(dtype=np.int64)
                        labels_b = sorted_b["y_true"].to_numpy(dtype=np.int64)

                        estimate = float(
                            wasserstein_distance(
                                support_a,
                                support_b,
                                matrix_a.mean(axis=0),
                                matrix_b.mean(axis=0),
                            )
                        )
                        rng = _comparison_rng(
                            model, language_a, language_b, layer, group
                        )
                        means_a = _resampled_means(
                            matrix_a, _resample_matrix(rng, labels_a, BOOTSTRAP_RESAMPLES)
                        )
                        means_b = _resampled_means(
                            matrix_b, _resample_matrix(rng, labels_b, BOOTSTRAP_RESAMPLES)
                        )
                        draws = _wasserstein_batch(
                            support_a, support_b, means_a, means_b
                        )
                        low, high, valid_n, status = _bootstrap_ci(draws)
                        rows.append(
                            {
                                "model": model,
                                "language_a": language_a,
                                "language_b": language_b,
                                "layer": int(layer),
                                "group": group,
                                "metric": "wasserstein_distance",
                                "unit": "Hz",
                                "estimate": estimate,
                                "ci_low": low,
                                "ci_high": high,
                                "n_a": int(len(matrix_a)),
                                "n_b": int(len(matrix_b)),
                                "status": status,
                                "valid_bootstrap_n": int(valid_n),
                                "degenerate_n": 0,
                                **meta,
                            }
                        )
    return _finalize_comparison(
        rows,
        _SPECTRAL_DIVERGENCE_COLUMNS,
        ["model", "language_a", "language_b", "layer", "group"],
    )


# ---------------------------------------------------------------------------
# Completeness of comparison tables against the expected grid
# ---------------------------------------------------------------------------

_ENCODER_METRICS = (
    "cohen_kappa",
    "prediction_agreement",
    "relevance_cosine_similarity",
    "relevance_jensen_shannon_distance",
    "score_spearman",
)
_DIAGONAL_METRICS = ("delta_mcc", "delta_roc_auc")


def _expected_encoder_identities(models, languages, layers):
    pairs = [
        (models[i], models[j])
        for i in range(len(models))
        for j in range(i + 1, len(models))
    ]
    return {
        (a, b, source, target, layer, metric)
        for a, b in pairs
        for source in languages
        for target in languages
        for layer in layers
        for metric in _ENCODER_METRICS
    }


def _expected_language_shift_identities(models, languages, layers):
    target_pairs = [
        (languages[i], languages[j])
        for i in range(len(languages))
        for j in range(i + 1, len(languages))
    ]
    return {
        (model, source, target_a, target_b, layer, group)
        for model in models
        for source in languages
        for target_a, target_b in target_pairs
        for layer in layers
        for group, _ in _COMPARISON_GROUPS
    }


def _expected_diagonal_identities(models, languages, layers):
    return {
        (model, source, target, layer, metric)
        for model in models
        for target in languages
        for source in languages
        if source != target
        for layer in layers
        for metric in _DIAGONAL_METRICS
    }


def _expected_spectral_identities(models, languages, layers):
    language_pairs = [
        (languages[i], languages[j])
        for i in range(len(languages))
        for j in range(i + 1, len(languages))
    ]
    return {
        (model, language_a, language_b, layer, group)
        for model in models
        for language_a, language_b in language_pairs
        for layer in layers
        for group, _ in _COMPARISON_GROUPS
    }


_COMPLETENESS_SPECS: Mapping[str, tuple] = MappingProxyType(
    {
        "encoder_agreement": (
            _expected_encoder_identities,
            ("model_a", "model_b", "source", "target", "layer", "metric"),
        ),
        "language_shift": (
            _expected_language_shift_identities,
            ("model", "source", "target_a", "target_b", "layer", "group"),
        ),
        "diagonal_vs_offdiagonal": (
            _expected_diagonal_identities,
            ("model", "source", "target", "layer", "metric"),
        ),
        "spectral_divergence": (
            _expected_spectral_identities,
            ("model", "language_a", "language_b", "layer", "group"),
        ),
    }
)


def _observed_identities(table: pd.DataFrame, keys: Sequence[str]) -> set:
    if table is None or table.empty:
        return set()
    identities = set()
    for row in table[list(keys)].itertuples(index=False, name=None):
        identities.add(
            tuple(int(value) if key == "layer" else value for key, value in zip(keys, row))
        )
    return identities


def comparison_completeness(
    name: str,
    table: pd.DataFrame,
    models: Sequence[str],
    languages: Sequence[str],
    layers: Sequence[int],
) -> dict[str, object]:
    """Assess one comparison table against the grid implied by availability.

    The expected grid is the full combinatorial product of the actually observed
    models, languages and layers (plus the comparison's metric/group axes). A
    table is complete only when it covers that grid exactly; otherwise it is
    reported as ``unavailable_incomplete_table`` with the number and a sample of
    the missing identities so the gap can be diagnosed.
    """
    generator, keys = _COMPLETENESS_SPECS[name]
    models = sorted(str(model) for model in set(models))
    languages = sorted(str(language) for language in set(languages))
    layers = sorted(int(layer) for layer in set(layers))
    expected = generator(models, languages, layers)
    observed = _observed_identities(table, keys)
    missing = expected - observed
    extra = observed - expected
    complete = bool(expected) and not missing and not extra
    return {
        "status": "complete" if complete else "unavailable_incomplete_table",
        "complete": complete,
        "expected_rows": len(expected),
        "observed_rows": len(observed),
        "missing_rows": len(missing),
        "extra_rows": len(extra),
        "missing_examples": tuple(sorted(missing, key=lambda item: tuple(map(str, item)))[:10]),
        "expected": tuple(sorted(expected, key=lambda item: tuple(map(str, item)))),
    }


def build_report_tables(sources: Sequence[LoadedReportSource]) -> ReportTables:
    """Merge loaded sources into deterministic, identity-labelled report tables."""
    loaded = list(sources)
    if not loaded:
        raise ValueError("at least one loaded report source is required")
    if any(not isinstance(item, LoadedReportSource) for item in loaded):
        raise TypeError("sources must be LoadedReportSource instances")
    identities = [
        (
            item.source.identity.profile,
            item.source.identity.languages,
            item.source.identity.protocol,
            item.source.identity.scope,
        )
        for item in loaded
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate report source identity")

    def merged(name: str, keys: Sequence[str]) -> pd.DataFrame:
        frame = pd.concat([getattr(item, name) for item in loaded], ignore_index=True)
        return _ordered(frame, keys)

    performance = merged("performance", ["model", "source", "target", "layer"])
    if performance.duplicated(["model", "source", "target", "layer"]).any():
        raise ValueError("duplicate model/language cells across report sources")

    cell_predictions = merged("cell_predictions", list(_PER_SAMPLE_KEYS))
    if cell_predictions.duplicated(list(_PER_SAMPLE_KEYS)).any():
        raise ValueError("duplicate per-sample prediction identity across report sources")
    sample_relevance = merged("sample_relevance", list(_PER_SAMPLE_KEYS))
    if sample_relevance.duplicated(list(_PER_SAMPLE_KEYS)).any():
        raise ValueError("duplicate per-sample XAI identity across report sources")
    if len(models := tuple(sorted(set(performance["model"])))) > 1:
        _check_paired_xai_across_models(sample_relevance)

    band_edges, provenance = _band_edge_tables(
        [edge for item in loaded for edge in item.band_edges]
    )
    if band_edges.duplicated(["model", "language", "band"]).any():
        raise ValueError("duplicate band edges across report sources")
    edge_columns = band_edges.rename(
        columns={
            "low_hz": "band_low_hz",
            "high_hz": "band_high_hz",
            "center_hz": "band_center_hz",
            "label": "band_label",
        }
    )

    def with_bands(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
        joined = frame.merge(
            edge_columns, on=["model", "language", "band"], how="left", validate="many_to_one"
        )
        if joined["band_low_hz"].isna().any():
            raise ValueError("relevance bands lack recorded band edges")
        return _ordered(joined, keys)

    band_relevance = with_bands(
        merged("band_relevance", ["model", "source", "target", "layer", "measure", "band"]),
        ["model", "source", "target", "layer", "measure", "band"],
    )
    class_relevance = with_bands(
        merged(
            "class_relevance",
            ["model", "source", "target", "grouping", "group", "layer", "measure", "band"],
        ),
        ["model", "source", "target", "grouping", "group", "layer", "measure", "band"],
    )
    transitions = _transition_summary(
        pd.concat([item.transitions for item in loaded], ignore_index=True)
    )

    present = set(performance["source"]) | set(performance["target"])
    languages = tuple(language for language in LANGUAGE_ORDER if language in present)
    payloads: dict[tuple, Mapping[str, np.ndarray]] = {}
    for item in loaded:
        payloads.update(item.stdft_payloads)

    encoder_agreement = build_encoder_agreement(cell_predictions, sample_relevance)
    language_shift = build_language_shift(sample_relevance)
    diagonal_vs_offdiagonal = build_diagonal_vs_offdiagonal(cell_predictions)
    spectral_divergence = build_spectral_divergence(sample_relevance, band_edges)

    model_list = list(models)
    pred_languages = sorted(set(cell_predictions["source"]) | set(cell_predictions["target"]))
    pred_layers = sorted({int(value) for value in cell_predictions["layer"]})
    rel_languages = sorted(set(sample_relevance["source"]) | set(sample_relevance["target"]))
    rel_layers = sorted({int(value) for value in sample_relevance["layer"]})

    def _assess(name, table, langs, layers_list, has_inputs, insufficient_reason):
        if not has_inputs:
            return {
                "status": insufficient_reason,
                "complete": False,
                "expected_rows": 0,
                "observed_rows": int(len(table)),
                "missing_rows": 0,
                "extra_rows": 0,
                "missing_examples": (),
            }
        info = comparison_completeness(name, table, model_list, langs, layers_list)
        info.pop("expected", None)  # keep ReportTables lightweight
        return info

    comparison_completeness_map = {
        "encoder_agreement": _assess(
            "encoder_agreement", encoder_agreement, pred_languages, pred_layers,
            len(model_list) > 1, "single_model",
        ),
        "language_shift": _assess(
            "language_shift", language_shift, rel_languages, rel_layers,
            len(rel_languages) > 1, "single_language",
        ),
        "diagonal_vs_offdiagonal": _assess(
            "diagonal_vs_offdiagonal", diagonal_vs_offdiagonal, pred_languages, pred_layers,
            len(pred_languages) > 1, "single_language",
        ),
        "spectral_divergence": _assess(
            "spectral_divergence", spectral_divergence, rel_languages, rel_layers,
            len(rel_languages) > 1, "single_language",
        ),
    }
    layer_faithfulness = (
        merged(
            "layer_faithfulness",
            ["model", "source", "target", "layer", "k", "true_class", "comparison"],
        )
        if any(not item.layer_faithfulness.empty for item in loaded)
        else pd.DataFrame(columns=sorted(_LAYER_FAITHFULNESS_COLUMNS))
    )
    faithfulness_keys = ["model", "source", "target", "layer"]
    aligned_faithfulness = layer_faithfulness[
        layer_faithfulness["comparison"].isin(
            ("top_minus_random", "top_minus_bottom")
        )
    ].merge(
        performance[faithfulness_keys].drop_duplicates(),
        on=faithfulness_keys,
        how="inner",
    )
    has_aligned_faithfulness = not _displayable_faithfulness_rows(
        aligned_faithfulness
    ).empty
    return ReportTables(
        models=models,
        languages=languages,
        performance=performance,
        emergence=merged("emergence", ["model", "source", "target"]),
        band_relevance=band_relevance,
        class_relevance=class_relevance,
        transitions=transitions,
        conservation=merged("conservation", ["model", "source", "target", "layer"]),
        cohort=merged("cohort", ["model", "source", "target", "layer"]),
        stdft_examples=merged(
            "stdft_examples",
            ["model", "source", "target", "layer", "y_true", "sample_id"],
        ),
        stdft_payloads=payloads,
        band_edges=band_edges,
        band_edge_provenance=provenance,
        transfer_selection=build_transfer_selection(performance),
        xai_performance_association=build_xai_performance_associations(
            performance, band_relevance
        ),
        omitted_comparisons=(
            omitted := _omitted_comparisons(comparison_completeness_map)
        ),
        planned_figures=_planned_figures(
            omitted, has_aligned_faithfulness=has_aligned_faithfulness
        ),
        probe_stability=merged(
            "probe_stability",
            ["model", "source", "target", "layer"],
        )
        if any(not item.probe_stability.empty for item in loaded)
        else pd.DataFrame(columns=sorted(_PROBE_STABILITY_COLUMNS)),
        layer_faithfulness=layer_faithfulness,
        cell_predictions=cell_predictions,
        sample_relevance=sample_relevance,
        encoder_agreement=encoder_agreement,
        language_shift=language_shift,
        diagonal_vs_offdiagonal=diagonal_vs_offdiagonal,
        spectral_divergence=spectral_divergence,
        comparison_completeness=comparison_completeness_map,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_PDF_METADATA = {"Creator": "layerwise_report", "Producer": "matplotlib", "CreationDate": None}
_PNG_METADATA = {"Software": None}
_FIGURE_WIDTH = 7.0
_CONSERVATION_FLOOR = 1e-12


def selected_transfer_effects(
    tables: ReportTables, metric: str
) -> pd.DataFrame:
    selected = tables.transfer_selection.loc[
        tables.transfer_selection["source"] != tables.transfer_selection["target"],
        ["model", "source", "target", "selected_layer"],
    ]
    effects = tables.diagonal_vs_offdiagonal[
        tables.diagonal_vs_offdiagonal["metric"] == metric
    ]
    merged = effects.merge(
        selected,
        on=["model", "source", "target"],
        how="inner",
        validate="many_to_one",
    )
    return _ordered(
        merged[merged["layer"] == merged["selected_layer"]],
        ("model", "source", "target"),
    )


def _usable_claim_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    usable = frame.loc[frame["status"] == "ok"].copy()
    finite = np.isfinite(usable[list(columns)].to_numpy(dtype=np.float64)).all(axis=1)
    return usable.loc[finite]


def _pooled_layer_scope(
    frame: pd.DataFrame,
    columns: Sequence[str] = ("estimate",),
    **filters: object,
) -> str:
    """How many layers a pooled claim aggregates, from the same usable rows."""
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if selected.empty or "layer" not in selected:
        return ""
    layers = sorted(
        {int(value) for value in _usable_claim_rows(selected, columns)["layer"]}
    )
    if len(layers) > 1:
        return f"agregando as {len(layers)} camadas"
    if layers:
        return f"na camada {layers[0]}"
    return ""


def transfer_effect_claim(frame: pd.DataFrame) -> Mapping[str, float | int]:
    usable = _usable_claim_rows(frame, ("estimate", "ci_low", "ci_high"))
    estimates = usable["estimate"].to_numpy(dtype=np.float64)
    return {
        "n": int(estimates.size),
        "median": float(np.median(estimates)) if estimates.size else float("nan"),
        "negative_n": int((estimates < 0).sum()),
        "ci_below_zero_n": int((usable["ci_high"] < 0).sum()),
        "ci_above_zero_n": int((usable["ci_low"] > 0).sum()),
    }


def agreement_claim(frame: pd.DataFrame) -> Mapping[str, float | int]:
    selected = frame.loc[frame["metric"] == "prediction_agreement"]
    usable = _usable_claim_rows(selected, ("estimate",))
    estimates = usable["estimate"].to_numpy(dtype=np.float64)
    return {
        "n": int(estimates.size),
        "median": float(np.median(estimates)) if estimates.size else float("nan"),
        "minimum": float(np.min(estimates)) if estimates.size else float("nan"),
        "maximum": float(np.max(estimates)) if estimates.size else float("nan"),
    }


def faithfulness_claim(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "total", "ci_above_zero_n", "ci_below_zero_n"]
    usable = _usable_claim_rows(frame, ("ci_low", "ci_high"))
    rows = [
        {
            "model": model,
            "total": int(len(group)),
            "ci_above_zero_n": int((group["ci_low"] > 0).sum()),
            "ci_below_zero_n": int((group["ci_high"] < 0).sum()),
        }
        for model, group in usable.groupby("model", sort=True)
    ]
    return pd.DataFrame(rows, columns=columns)


def build_results_synthesis(tables: ReportTables) -> pd.DataFrame:
    """Build the four-question synthesis from the validated claim helpers."""
    columns = ["question", "evidence", "result", "caution"]
    rows: list[dict[str, str]] = []

    transfer_parts = []
    for metric, label in (("delta_roc_auc", "Δ ROC-AUC"), ("delta_mcc", "Δ MCC")):
        claim = transfer_effect_claim(
            tables.diagonal_vs_offdiagonal.loc[
                tables.diagonal_vs_offdiagonal["metric"] == metric
            ]
        )
        if claim["n"]:
            transfer_parts.append(
                f"{label}: {claim['negative_n']}/{claim['n']} estimativas negativas; "
                f"{claim['ci_below_zero_n']}/{claim['n']} ICs inteiramente abaixo de zero"
            )
    transfer_scope = _pooled_layer_scope(
        tables.diagonal_vs_offdiagonal[
            tables.diagonal_vs_offdiagonal["metric"].isin(("delta_roc_auc", "delta_mcc"))
        ],
        ("estimate", "ci_low", "ci_high"),
    )
    transfer_result = "; ".join(transfer_parts)
    if transfer_parts and transfer_scope:
        transfer_result = f"{transfer_result} ({transfer_scope})"
    rows.append(
        {
            "question": "Quanto o desempenho cai na transferência?",
            "evidence": "Diferença fora da diagonal menos diagonal no mesmo conjunto-alvo.",
            "result": transfer_result if transfer_parts else "Não disponível.",
            "caution": "O sinal descreve a mudança observada, não sua causa.",
        }
    )

    agreement = agreement_claim(tables.encoder_agreement)
    agreement_scope = _pooled_layer_scope(
        tables.encoder_agreement, metric="prediction_agreement"
    )
    agreement_result = (
        "Não disponível."
        if not agreement["n"]
        else (
            f"Mediana {_num(agreement['median'])}; faixa "
            f"{_num(agreement['minimum'])}–{_num(agreement['maximum'])} "
            f"em {agreement['n']} estimativas"
            + (f", {agreement_scope}." if agreement_scope else ".")
        )
    )
    rows.append(
        {
            "question": "Quanto os encoders concordam?",
            "evidence": "Fração de decisões iguais nos mesmos áudios e condições.",
            "result": agreement_result,
            "caution": "Acordo não indica qual modelo está correto.",
        }
    )

    has_shift = not _usable_claim_rows(
        tables.language_shift, ("estimate", "ci_low", "ci_high")
    ).empty
    has_divergence = not _usable_claim_rows(
        tables.spectral_divergence, ("estimate", "ci_low", "ci_high")
    ).empty
    shift_result = (
        "Mudança de relevância e divergência espectral observadas."
        if has_shift and has_divergence
        else "Não disponível."
    )
    rows.append(
        {
            "question": "Como a relevância muda entre corpora/idiomas?",
            "evidence": "Jensen–Shannon com probe fixo e Wasserstein entre sistemas diagonais.",
            "result": shift_result,
            "caution": "Corpus, áudio e idioma mudam juntos; não há isolamento causal.",
        }
    )

    faithfulness = faithfulness_claim(_aligned_layer_faithfulness(tables))
    if faithfulness.empty:
        faithfulness_result = "Não disponível."
    else:
        parts = []
        for row in faithfulness.itertuples(index=False):
            parts.append(
                f"{_model_name(row.model)}: {row.ci_above_zero_n}/{row.total} ICs "
                f"acima de zero e {row.ci_below_zero_n}/{row.total} abaixo de zero"
            )
        faithfulness_result = "; ".join(parts) + "."
    rows.append(
        {
            "question": "Remover as faixas mais relevantes altera mais o escore?",
            "evidence": "Top-minus-random e top-minus-bottom com ICs persistidos.",
            "result": faithfulness_result,
            "caution": "Intervenção STFT controlada; não implica causalidade no mundo real.",
        }
    )
    return pd.DataFrame(rows, columns=columns)


_SUPTITLE_LINE_CHARS = 100


def _wrap_long_suptitle(fig) -> None:
    suptitle = getattr(fig, "_suptitle", None)
    engine = fig.get_layout_engine()
    if suptitle is None or type(engine).__name__ != "ConstrainedLayoutEngine":
        return
    text = suptitle.get_text()
    if "\n" not in text and len(text) > _SUPTITLE_LINE_CHARS and " (" in text:
        heading, scope = text.rsplit(" (", 1)
        suptitle.set_text(f"{heading}\n({scope}")


def _save_report_figure(fig, name: str, figures_dir: Path) -> tuple[str, ...]:
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    _wrap_long_suptitle(fig)
    fig.savefig(figures_dir / f"{name}.pdf", metadata=_PDF_METADATA)
    fig.savefig(figures_dir / f"{name}.png", metadata=_PNG_METADATA)
    plt.close(fig)
    return (f"{name}.pdf", f"{name}.png")


def _cells(tables: ReportTables) -> tuple[tuple[str, str, str], ...]:
    frame = tables.performance[["model", "source", "target"]].drop_duplicates()
    return tuple(frame.itertuples(index=False, name=None))


SAMPLE_ID_DISPLAY_LENGTH = 24


def abbreviate_sample_id(sample_id: object, max_length: int = SAMPLE_ID_DISPLAY_LENGTH) -> str:
    """Short, deterministic display form that keeps the start and the end.

    IDs up to `max_length` characters are returned unchanged. Longer ones keep
    the first and last characters around a single ellipsis, so the dataset
    prefix and the hash-like suffix both stay recognisable. Tables and
    manifests always hold the full ID; this form is for titles and prose.
    """
    text = str(sample_id)
    if max_length < 5:
        raise ValueError("max_length must be at least 5")
    if len(text) <= max_length:
        return text
    keep = max_length - 1
    head = (keep + 1) // 2
    tail = keep - head
    return f"{text[:head]}\u2026{text[len(text) - tail:]}"


def _display_sample_ids(sample_ids: Sequence[object]) -> dict[str, str]:
    """Abbreviate IDs, lengthening only as much as needed to keep them distinct."""
    unique = sorted({str(item) for item in sample_ids})
    if not unique:
        return {}
    longest = max(len(item) for item in unique)
    limit = SAMPLE_ID_DISPLAY_LENGTH
    while True:
        mapping = {item: abbreviate_sample_id(item, limit) for item in unique}
        if len(set(mapping.values())) == len(unique) or limit >= longest:
            return mapping
        limit += 4


def _cell_label(cell: tuple[str, str, str]) -> str:
    return f"{cell[0]}, {cell[1]}\u2192{cell[2]}"


def _pair_label(cell: tuple[str, str, str]) -> str:
    return f"{cell[1]}\u2192{cell[2]}"


def _cells_for_model(
    tables: ReportTables, model: str
) -> tuple[tuple[str, str, str], ...]:
    return tuple(cell for cell in _cells(tables) if cell[0] == model)


def _summary_cells(
    tables: ReportTables,
) -> tuple[tuple[str, str, str], ...]:
    cells = _cells(tables)
    if len(cells) <= 9:
        return cells
    return tuple(cell for cell in cells if cell[1] == cell[2])


def _cell_rows(frame: pd.DataFrame, cell: tuple[str, str, str]) -> pd.DataFrame:
    return frame[
        (frame["model"] == cell[0])
        & (frame["source"] == cell[1])
        & (frame["target"] == cell[2])
    ]


def _band_range_text(tables: ReportTables) -> str:
    """The frequency span covered by the bands, taken from the resolved edges."""
    low = float(tables.band_edges["low_hz"].min())
    high = float(tables.band_edges["high_hz"].max())
    return f"{low:.0f}\u2013{high:.0f} Hz"


def _scope(tables: ReportTables) -> str:
    return f"{', '.join(tables.models)} \u00b7 {', '.join(tables.languages)}"


def _record(
    tables: ReportTables,
    name: str,
    files: tuple[str, ...],
    *,
    heading: str,
    description: str,
    metric: str,
    units: str,
    transformation: str,
    cells: tuple[tuple[str, str, str], ...] | None = None,
) -> FigureRecord:
    cells = cells if cells is not None else _cells(tables)
    pairs = "; ".join(_cell_label(cell) for cell in cells)
    return FigureRecord(
        name=name,
        files=files,
        title=f"{heading} ({_scope(tables)})",
        caption=(
            f"{description} Modelo: {', '.join(tables.models)}. "
            f"Idioma (treino\u2192avaliação): {pairs}. "
            f"Métrica: {metric}. Unidades: {units}. Transformação: {transformation}."
        ),
        metric=metric,
        units=units,
        transformation=transformation,
        models=tables.models,
        languages=tables.languages,
        cells=cells,
    )


_LAYER_AXIS_LABEL = "Camada do encoder (índice do bloco Transformer)"
_CLASS_NAMES_PT = {0: "real", 1: "sintético"}
# Canonical binary-class order mirrors data.LABEL_NAMES without importing the
# dataset module (and its optional I/O dependencies) into report generation.
_TRUE_CLASS_ORDER = ("bonafide", "spoof")
_TRUE_CLASS_NAMES_PT = MappingProxyType(
    dict(zip(_TRUE_CLASS_ORDER, (_CLASS_NAMES_PT[0], _CLASS_NAMES_PT[1])))
)
_MIN_BAND_TICK_SPACING_IN = 0.34
_HEATMAP_AXIS_SHARE = 0.55
_MULTI_PANEL_XLABEL_Y = 0.115


def _layer_axis(ax) -> None:
    _set_layer_ticks(ax, list(range(1, 13)))
    ax.set_xlim(0.6, 12.4)


def _set_layer_ticks(ax, ticks: Sequence[int]) -> None:
    """A tick per layer; beyond six layers, labels alternate back from the last."""
    ax.set_xticks(ticks)
    last = ticks[-1]
    ax.set_xticklabels(
        [
            str(tick) if len(ticks) <= 6 or (last - tick) % 2 == 0 else ""
            for tick in ticks
        ]
    )
    ax.tick_params(axis="x", labelsize=7)


def _band_tick_indices(n_bands: int, axis_width_in: float) -> list[int]:
    """Band indices to label: both ends always, evenly spread, never crowded.

    At most one label per `_MIN_BAND_TICK_SPACING_IN` inches of axis width, so
    the count adapts to the number of bands and to the panel size.
    """
    if n_bands <= 1:
        return [0]
    span = n_bands - 1
    gap = max(1, math.ceil(_MIN_BAND_TICK_SPACING_IN * span / max(axis_width_in, 1e-6)))
    count = span // gap + 1
    if gap == 1:
        return list(range(n_bands))
    count = max(count, 2)
    positions = np.linspace(0, span, count)
    return sorted({int(round(value)) for value in positions})


def _line_panels(
    plt,
    tables: ReportTables,
    panels: Sequence[tuple[str, str, float]],
    shape: tuple[int, int],
    height: float,
):
    if len(tables.models) > 1:
        fig, axes = plt.subplots(
            len(tables.models),
            len(panels),
            figsize=(_FIGURE_WIDTH, 2.25 * len(tables.models) + 1.0),
            sharex=True,
            squeeze=False,
            constrained_layout=False,
        )
        for row, model in enumerate(tables.models):
            for column, (metric, ylabel, scale) in enumerate(panels):
                ax = axes[row, column]
                for cell in _cells_for_model(tables, model):
                    rows = _cell_rows(tables.performance, cell)
                    ax.plot(
                        rows["layer"],
                        rows[metric] * scale,
                        marker="o",
                        markersize=2.5,
                        linewidth=1.0,
                        label=_pair_label(cell),
                    )
                if row == 0:
                    ax.set_title(ylabel, fontsize=8)
                if column == 0:
                    ax.set_ylabel(f"{model}\n{ylabel}", fontsize=7)
                _layer_axis(ax)
        fig.supxlabel(
            _LAYER_AXIS_LABEL,
            fontsize=8,
            y=_MULTI_PANEL_XLABEL_Y,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=3,
            fontsize=6.5,
            title="Idioma (treino\u2192avaliação)",
            title_fontsize=7,
        )
        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.90,
            bottom=0.24,
            hspace=0.28,
            wspace=0.30,
        )
        return fig, axes.ravel()

    fig, axes = plt.subplots(
        *shape,
        figsize=(_FIGURE_WIDTH, height),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    flat = axes.ravel()
    cells = _cells(tables)
    for ax, (column, ylabel, scale) in zip(flat, panels):
        for cell in cells:
            rows = _cell_rows(tables.performance, cell)
            ax.plot(
                rows["layer"],
                rows[column] * scale,
                marker="o",
                markersize=3,
                label=_cell_label(cell),
            )
        ax.set_ylabel(ylabel)
        _layer_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel(_LAYER_AXIS_LABEL)
    return fig, flat


def _figure_performance(plt, tables, figures_dir):
    name = "performance_by_layer"
    fig, axes = _line_panels(
        plt,
        tables,
        [
            ("roc_auc", "ROC-AUC (adimensional)", 1.0),
            ("eer_diagnostic", "EER diagnóstico (%)", 100.0),
        ],
        (2, 1),
        4.6,
    )
    for index in range(0, len(axes), 2):
        axes[index].axhline(
            0.5,
            color="0.4",
            linestyle="--",
            linewidth=0.8,
            label="acaso (0,5)" if len(tables.models) == 1 else None,
        )
    if len(tables.models) == 1:
        axes[0].legend(loc="best", ncol=1)
    record = _record(
        tables,
        name,
        (),
        heading="ROC-AUC e EER diagnóstico por camada do encoder",
        description="Discriminação sem limiar de um probe linear em cada camada do encoder, avaliada no conjunto de teste do idioma-alvo.",
        metric="ROC-AUC e taxa de erro igual (EER) diagnóstica",
        units="ROC-AUC adimensional (0\u20131); EER em porcentagem",
        transformation="escores do probe por camada no conjunto de teste do alvo; EER = fração \u00d7 100",
    )
    fig.suptitle(record.title, fontsize=9)
    files = _save_report_figure(fig, name, figures_dir)
    return _with_files(record, files)


def _with_files(record: FigureRecord, files: tuple[str, ...]) -> FigureRecord:
    return FigureRecord(**{**record.__dict__, "files": files})


def _figure_fixed_threshold(plt, tables, figures_dir):
    name = "fixed_threshold_by_layer"
    fig, axes = _line_panels(
        plt,
        tables,
        [
            ("accuracy", "Acurácia (fração)", 1.0),
            ("mcc", "MCC (adimensional, \u22121 a 1)", 1.0),
            ("tpr", "TPR: sintéticos\ndetectados\n(fração dos sintéticos)", 1.0),
            ("fpr", "FPR: reais marcados\ncomo sintéticos\n(fração dos reais)", 1.0),
        ],
        (2, 2),
        4.8,
    )
    for index in range(1, len(axes), 4):
        axes[index].axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    if len(tables.models) == 1:
        axes[0].legend(loc="best")
    record = _record(
        tables,
        name,
        (),
        heading="Métricas de limiar fixo por camada do encoder",
        description=(
            "Acurácia, coeficiente de correlação de Matthews, taxa de verdadeiros "
            "positivos e taxa de falsos positivos no limiar calibrado no idioma de "
            "origem. A classe positiva é sintético (rótulo 1): TPR é a fração das "
            "amostras sintéticas detectadas e FPR a fração das amostras reais "
            "marcadas como sintéticas."
        ),
        metric="acurácia, MCC, TPR e FPR no limiar fixo calibrado na origem (classe positiva = sintético)",
        units="frações (0\u20131), exceto MCC (adimensional, \u22121 a 1); TPR sobre os sintéticos, FPR sobre os reais",
        transformation="predições duras a partir do escore do probe \u2265 limiar fixo no conjunto de teste do alvo",
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_transfer_heatmaps(plt, tables, figures_dir):
    name = "transfer_performance_heatmaps"
    models = tables.models
    languages = tuple(
        language
        for language in ("eng", "por", "zho")
        if language in tables.languages
    )
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(_FIGURE_WIDTH, 5.4),
        squeeze=False,
        constrained_layout=True,
    )
    frame = tables.transfer_selection
    images = {}
    metrics = (
        ("roc_auc", "ROC-AUC (camada selecionada)", (0.0, 1.0), "viridis"),
        ("mcc", "MCC (mesma camada)", (-1.0, 1.0), "coolwarm"),
    )
    language_labels = [_language_name(item).capitalize() for item in languages]
    for row_index, (metric, metric_title, limits, cmap) in enumerate(metrics):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            model_rows = frame[frame["model"] == model]
            matrix = (
                model_rows.pivot(index="source", columns="target", values=metric)
                .reindex(index=languages, columns=languages)
                .to_numpy(dtype=np.float64)
            )
            selected_layers = (
                model_rows.pivot(
                    index="source", columns="target", values="selected_layer"
                )
                .reindex(index=languages, columns=languages)
                .to_numpy(dtype=np.float64)
            )
            image = ax.imshow(
                matrix,
                vmin=limits[0],
                vmax=limits[1],
                cmap=cmap,
                aspect="equal",
            )
            images[metric] = image
            ax.set_xticks(range(len(languages)))
            ax.set_yticks(range(len(languages)))
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(False)
            if row_index == len(metrics) - 1:
                ax.set_xticklabels(language_labels, rotation=30, ha="right")
                ax.set_xlabel("Idioma de avaliação", fontsize=8)
            else:
                ax.set_xticklabels([])
            if column_index == 0:
                ax.set_yticklabels(language_labels)
                ax.set_ylabel("Idioma de treino", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(f"{_model_name(model)}\n{metric_title}", fontsize=8)
            for source_index in range(len(languages)):
                for target_index in range(len(languages)):
                    value = matrix[source_index, target_index]
                    selected_layer = selected_layers[source_index, target_index]
                    if np.isfinite(value) and np.isfinite(selected_layer):
                        ax.text(
                            target_index,
                            source_index,
                            f"{_figure_number(value)}\nC{int(selected_layer)}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=(
                                "white"
                                if metric == "roc_auc" and value < 0.45
                                else "black"
                            ),
                        )
    for metric, image in images.items():
        row = 0 if metric == "roc_auc" else 1
        colorbar = fig.colorbar(image, ax=axes[row, :].tolist(), shrink=0.8)
        colorbar.set_label("ROC-AUC" if metric == "roc_auc" else "MCC", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Transferência por idioma de treino e avaliação",
        description=(
            "Para cada célula treino→avaliação, a camada é escolhida pela maior "
            "ROC-AUC. O painel de MCC reutiliza essa mesma camada."
        ),
        metric="ROC-AUC e MCC na camada selecionada por ROC-AUC",
        units="ROC-AUC (0–1) e MCC (−1 a 1), adimensionais",
        transformation=(
            "em cada célula origem→alvo, selecionar a camada de ROC-AUC máxima; "
            "mostrar ROC-AUC e MCC dessa mesma camada selecionada, sem maximizar "
            "o MCC independentemente"
        ),
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_transfer_effect(
    plt,
    tables: ReportTables,
    figures_dir: Path,
    *,
    metric: str,
):
    specifications = {
        "delta_roc_auc": (
            "transfer_delta_roc_auc",
            "Efeito de transferência em ROC-AUC",
            "Δ ROC-AUC (fora da diagonal − diagonal)",
            "ROC-AUC",
        ),
        "delta_mcc": (
            "transfer_delta_mcc",
            "Efeito de transferência em MCC",
            "Δ MCC (fora da diagonal − diagonal)",
            "MCC",
        ),
    }
    if metric not in specifications:
        raise ValueError(f"unsupported transfer effect metric: {metric}")
    name, heading, xlabel, metric_label = specifications[metric]
    frame = selected_transfer_effects(tables, metric)
    if frame.empty:
        raise ValueError(f"{name} figure requested but no selected effects exist")

    models = tuple(model for model in tables.models if model in set(frame["model"]))
    pairs = sorted(
        set(frame[["source", "target"]].itertuples(index=False, name=None))
    )
    pair_position = {pair: index for index, pair in enumerate(pairs)}
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(_FIGURE_WIDTH, 0.9 + 0.3 * len(pairs)),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for column, model in enumerate(models):
        ax = axes[0, column]
        model_rows = frame[frame["model"] == model].sort_values(
            ["source", "target"], kind="stable"
        )
        positions = np.array(
            [
                pair_position[(str(row.source), str(row.target))]
                for row in model_rows.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        estimates = model_rows["estimate"].to_numpy(dtype=np.float64)
        low = model_rows["ci_low"].to_numpy(dtype=np.float64)
        high = model_rows["ci_high"].to_numpy(dtype=np.float64)
        ordered_intervals = np.isfinite(low) & np.isfinite(high) & (low <= high)
        if ordered_intervals.any():
            ax.hlines(
                positions[ordered_intervals],
                low[ordered_intervals],
                high[ordered_intervals],
                linewidth=1.0,
            )
        finite_estimates = np.isfinite(estimates)
        if finite_estimates.any():
            ax.scatter(
                estimates[finite_estimates],
                positions[finite_estimates],
                marker="o",
                s=16,
                zorder=3,
            )
        ax.axvline(0.0, color="0.45", linestyle="--", linewidth=0.8)
        ax.set_title(_model_name(model), fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.locator_params(axis="x", nbins=4)
    first = axes[0, 0]
    first.set_yticks(range(len(pairs)))
    first.set_yticklabels([_transfer_pretty(source, target) for source, target in pairs])
    first.set_ylim(len(pairs) - 0.5, -0.5)
    fig.supxlabel(xlabel, fontsize=8)

    cells = tuple(
        frame[["model", "source", "target"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    record = _record(
        tables,
        name,
        (),
        heading=heading,
        description=(
            f"Efeito {metric_label} fora da diagonal menos o sistema diagonal "
            "correspondente, com estimativa pontual e intervalo bootstrap de 95%. "
            "A linha vertical marca efeito zero. Como as diferenças fora da "
            "diagonal combinam mudança de corpus e idioma, não identificam um "
            "efeito causal do idioma."
        ),
        metric=f"Δ {metric_label} fora da diagonal − diagonal",
        units="adimensional",
        transformation=(
            "uma linha por transferência fora da diagonal na camada selecionada "
            "pela ROC-AUC da própria célula; intervalo percentil bootstrap de 95%"
        ),
        cells=cells,
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_transfer_delta_roc_auc(plt, tables, figures_dir):
    return _figure_transfer_effect(
        plt, tables, figures_dir, metric="delta_roc_auc"
    )


def _figure_transfer_delta_mcc(plt, tables, figures_dir):
    return _figure_transfer_effect(plt, tables, figures_dir, metric="delta_mcc")


def _grid_shape(count: int, max_columns: int = 3) -> tuple[int, int]:
    columns = min(count, max_columns)
    return math.ceil(count / columns), columns


def _band_matrix(rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pivot = rows.pivot_table(index="layer", columns="band", values="mean", aggfunc="first")
    labels = (
        rows.drop_duplicates("band").sort_values("band")["band_center_hz"].tolist()
    )
    return pivot.to_numpy(dtype=np.float64), [f"{value:.0f}" for value in labels]


def _style_heatmap_axis(ax, labels: Sequence[str], columns: int) -> None:
    """Label the band axis with an adaptive subset that keeps both end bands."""
    axis_width = _FIGURE_WIDTH * _HEATMAP_AXIS_SHARE / columns
    shown = _band_tick_indices(len(labels), axis_width)
    ax.grid(False)
    ax.set_xticks(shown)
    ax.set_xticklabels([labels[index] for index in shown], rotation=60, ha="right")
    ax.set_yticks(range(12))
    ax.set_yticklabels(range(1, 13))


def _figure_heatmap(plt, tables, figures_dir):
    name = "dft_relevance_heatmap"
    cells = tuple(cell for cell in _cells(tables) if cell[1] == cell[2])
    nrows, ncols = _grid_shape(len(cells))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_FIGURE_WIDTH, 3.6 * nrows + 0.4),
        squeeze=False,
        constrained_layout=True,
    )
    selected = tables.band_relevance[tables.band_relevance["measure"] == "absolute_normalized"]
    matrices = {cell: _band_matrix(_cell_rows(selected, cell)) for cell in cells}
    vmax = max(float(matrix.max()) for matrix, _ in matrices.values())
    image = None
    for ax, cell in zip(axes.ravel(), cells):
        matrix, labels = matrices[cell]
        image = ax.imshow(
            matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax
        )
        _style_heatmap_axis(ax, labels, ncols)
        ax.set_title(_cell_label(cell), fontsize=8)
        ax.set_xlabel("Centro da banda (Hz, espaçamento mel)")
        ax.set_ylabel("Camada do encoder")
    for ax in axes.ravel()[len(cells):]:
        ax.set_visible(False)
    if len(tables.models) > 1:
        for ax in axes.ravel():
            ax.set_xlabel("")
        fig.supxlabel(
            "Centro da banda (Hz, espaçamento mel)", fontsize=8
        )
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        label="Relevância DFT absoluta normalizada média\n(fração da massa em banda)",
    )
    band_range = _band_range_text(tables)
    record = _record(
        tables,
        name,
        (),
        heading="Relevância DFT por camada e banda de frequência",
        description=(
            "Média por camada e banda de frequência da relevância AttnLRP absoluta "
            "normalizada após a camada de inspeção da DFT; a massa em banda de cada "
            f"amostra ({band_range}) soma 1, então os valores são frações da massa "
            "em banda e não do espectro inteiro."
        ),
        metric="relevância DFT absoluta normalizada média por banda",
        units=(
            f"fração da massa em banda por amostra ({band_range}; adimensional, soma 1 "
            "sobre as bandas); centro da banda em Hz"
        ),
        transformation="relevância AttnLRP \u2192 DFT \u2192 |R(f)| agregada em bandas de espaçamento mel \u2192 normalização por amostra \u2192 média sobre o conjunto fixo de amostras",
        cells=cells,
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_class_relevance(plt, tables, figures_dir):
    name = "class_relevance_by_layer"
    cells = _summary_cells(tables)
    if len(tables.models) > 1:
        nrows, ncols = _grid_shape(len(cells))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(_FIGURE_WIDTH, 2.45 * nrows + 0.7),
            squeeze=False,
            constrained_layout=True,
        )
        frame = tables.class_relevance
        frame = frame[
            (frame["grouping"] == "y_true")
            & (frame["measure"] == "absolute_normalized")
        ]
        prepared = {}
        for cell in cells:
            rows = _cell_rows(frame, cell)
            real, labels = _band_matrix(rows[rows["group"] == 0])
            synthetic, _ = _band_matrix(rows[rows["group"] == 1])
            prepared[cell] = (synthetic - real, labels)
        limit = (
            max(float(np.abs(matrix).max()) for matrix, _ in prepared.values())
            or 1.0
        )
        image = None
        for ax, cell in zip(axes.ravel(), cells):
            matrix, labels = prepared[cell]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            _style_heatmap_axis(ax, labels, ncols)
            ax.set_title(_cell_label(cell), fontsize=7)
            ax.set_xlabel("Centro da banda (Hz)")
            ax.set_ylabel("Camada do encoder")
        for ax in axes.ravel()[len(cells):]:
            ax.set_visible(False)
        for ax in axes.ravel():
            ax.set_xlabel("")
        fig.supxlabel("Centro da banda (Hz)", fontsize=8)
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.85,
            label="Sintético \u2212 real\n(fração da massa em banda)",
        )
        band_range = _band_range_text(tables)
        record = _record(
            tables,
            name,
            (),
            heading="Contraste de relevância entre amostras sintéticas e reais",
            description=(
                "Diferença entre a relevância DFT absoluta normalizada média das "
                "amostras sintéticas e reais nas avaliações dentro da própria "
                f"língua; a massa em banda de cada amostra ({band_range}) soma 1."
            ),
            metric="diferença sintético \u2212 real da relevância DFT absoluta normalizada média por banda",
            units=(
                f"pontos de fração da massa em banda ({band_range}; adimensional)"
            ),
            transformation="AttnLRP \u2192 DFT \u2192 bandas mel normalizadas em valor absoluto \u2192 média por classe verdadeira \u2192 sintético \u2212 real",
            cells=cells,
        )
        fig.suptitle(record.title, fontsize=9)
        return _with_files(
            record, _save_report_figure(fig, name, figures_dir)
        )

    fig, axes = plt.subplots(
        len(cells),
        3,
        figsize=(_FIGURE_WIDTH, 2.7 * len(cells) + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    frame = tables.class_relevance
    frame = frame[(frame["grouping"] == "y_true") & (frame["measure"] == "absolute_normalized")]
    prepared = {}
    for cell in cells:
        rows = _cell_rows(frame, cell)
        real, labels = _band_matrix(rows[rows["group"] == 0])
        synthetic, _ = _band_matrix(rows[rows["group"] == 1])
        prepared[cell] = (real, synthetic, labels)
    vmax = max(max(float(r.max()), float(s.max())) for r, s, _ in prepared.values())
    limit = max(float(np.abs(s - r).max()) for r, s, _ in prepared.values()) or 1.0
    level = diff = None
    titles = ("real", "sintético", "sintético \u2212 real")
    for row, cell in enumerate(cells):
        real, synthetic, labels = prepared[cell]
        for column, matrix in enumerate((real, synthetic, synthetic - real)):
            ax = axes[row, column]
            if column < 2:
                level = ax.imshow(
                    matrix, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax
                )
            else:
                diff = ax.imshow(
                    matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
                )
            _style_heatmap_axis(ax, labels, 3)
            ax.set_title(f"{titles[column]}\n{_cell_label(cell)}", fontsize=7)
            ax.set_xlabel("Centro da banda (Hz)")
            if column == 0:
                ax.set_ylabel("Camada do encoder")
    fig.colorbar(level, ax=axes[:, :2].ravel().tolist(), shrink=0.85,
                 label="Relevância absoluta normalizada média\n(fração da massa em banda)")
    fig.colorbar(diff, ax=axes[:, 2].ravel().tolist(), shrink=0.85,
                 label="Diferença\n(fração da massa em banda)")
    band_range = _band_range_text(tables)
    record = _record(
        tables,
        name,
        (),
        heading="Perfis de relevância de amostras reais e sintéticas por camada",
        description=(
            "Relevância DFT absoluta normalizada média por banda das amostras reais "
            "(bona fide) e sintéticas (spoof) do conjunto fixo, e sua diferença, por camada; "
            f"a massa em banda de cada amostra ({band_range}) soma 1."
        ),
        metric="relevância DFT absoluta normalizada média por banda, condicionada à classe verdadeira",
        units=(
            f"fração da massa em banda por amostra ({band_range}; adimensional); "
            "diferença em pontos de fração da massa em banda"
        ),
        transformation="AttnLRP \u2192 DFT \u2192 bandas mel normalizadas em valor absoluto, agrupadas pela classe verdadeira (0 = real, 1 = sintético)",
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_reorganization(plt, tables, figures_dir):
    name = "decision_reorganization_by_layer"
    panels = (
        ("similarity", "Similaridade cosseno\n(adimensional)"),
        ("normalized_l1_change", "Variação L1 normalizada\n(adimensional, 0\u20132)"),
    )
    if len(tables.models) > 1:
        fig, axes = plt.subplots(
            len(tables.models),
            len(panels),
            figsize=(_FIGURE_WIDTH, 2.25 * len(tables.models) + 1.0),
            sharex=True,
            squeeze=False,
            constrained_layout=False,
        )
        for row, model in enumerate(tables.models):
            for column, (metric, ylabel) in enumerate(panels):
                ax = axes[row, column]
                for cell in _cells_for_model(tables, model):
                    rows = _cell_rows(tables.transitions, cell)
                    x = rows["current_layer"].to_numpy()
                    line = ax.plot(
                        x,
                        rows[f"{metric}_mean"],
                        marker="o",
                        markersize=2.5,
                        linewidth=1.0,
                        label=_pair_label(cell),
                    )[0]
                    ax.fill_between(
                        x,
                        rows[f"{metric}_ci_low"],
                        rows[f"{metric}_ci_high"],
                        color=line.get_color(),
                        alpha=0.15,
                        linewidth=0,
                    )
                if row == 0:
                    ax.set_title(ylabel, fontsize=8)
                if column == 0:
                    ax.set_ylabel(f"{model}\n{ylabel}", fontsize=7)
        labels = _cell_rows(
            tables.transitions, _cells(tables)[0]
        )["transition"].tolist()
        for ax in axes[-1]:
            ax.set_xticks(range(2, 13))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        fig.supxlabel(
            "Transição entre camadas adjacentes (anterior \u2192 atual)",
            fontsize=8,
            y=_MULTI_PANEL_XLABEL_Y,
        )
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="outside lower center",
            ncol=3,
            fontsize=6.5,
            title="Idioma (treino\u2192avaliação)",
            title_fontsize=7,
        )
        fig.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.90,
            bottom=0.24,
            hspace=0.28,
            wspace=0.20,
        )
        record = _record(
            tables,
            name,
            (),
            heading="Reorganização da relevância da decisão final entre camadas",
            description="Média (com intervalo normal de 95% sobre as amostras) da similaridade e da variação L1 normalizada da massa temporal de relevância entre camadas adjacentes para a decisão final.",
            metric="similaridade cosseno e variação L1 normalizada da massa temporal de relevância entre camadas adjacentes",
            units="adimensional (cosseno 0\u20131 para massa não negativa; L1 0\u20132)",
            transformation="relevância AttnLRP com sinal do logit final \u2192 massa temporal absoluta por camada \u2192 normalização para massa unitária \u2192 comparação entre camadas adjacentes, média \u00b1 1,96 EPM sobre as amostras",
        )
        fig.suptitle(record.title, fontsize=9)
        return _with_files(
            record, _save_report_figure(fig, name, figures_dir)
        )

    fig, axes = plt.subplots(
        2, 1, figsize=(_FIGURE_WIDTH, 4.6), sharex=True, squeeze=False,
        constrained_layout=True,
    )
    flat = axes.ravel()
    for cell in _cells(tables):
        rows = _cell_rows(tables.transitions, cell)
        x = rows["current_layer"].to_numpy()
        for ax, (metric, ylabel) in zip(flat, panels):
            line = ax.plot(x, rows[f"{metric}_mean"], marker="o", markersize=3,
                           label=_cell_label(cell))[0]
            ax.fill_between(x, rows[f"{metric}_ci_low"], rows[f"{metric}_ci_high"],
                            color=line.get_color(), alpha=0.2, linewidth=0)
            ax.set_ylabel(ylabel)
    labels = (
        _cell_rows(tables.transitions, _cells(tables)[0])["transition"].tolist()
    )
    flat[1].set_xticks(range(2, 13))
    flat[1].set_xticklabels(labels, rotation=45, ha="right")
    flat[1].set_xlabel("Transição entre camadas adjacentes (anterior \u2192 atual)")
    flat[0].legend(loc="best")
    record = _record(
        tables,
        name,
        (),
        heading="Reorganização da relevância da decisão final entre camadas",
        description="Média (com intervalo normal de 95% sobre as amostras) da similaridade e da variação L1 normalizada da massa temporal de relevância entre camadas adjacentes para a decisão final.",
        metric="similaridade cosseno e variação L1 normalizada da massa temporal de relevância entre camadas adjacentes",
        units="adimensional (cosseno 0\u20131 para massa não negativa; L1 0\u20132)",
        transformation="relevância AttnLRP com sinal do logit final \u2192 massa temporal absoluta por camada \u2192 normalização para massa unitária \u2192 comparação entre camadas adjacentes, média \u00b1 1,96 EPM sobre as amostras",
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _representative_layers(layers: Sequence[int]) -> list[int]:
    ordered = sorted(set(int(layer) for layer in layers))
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return list(dict.fromkeys(picks))


def _figure_stdft(plt, tables, figures_dir):
    from matplotlib.colors import SymLogNorm

    name = "stdft_examples"
    cell = _cells(tables)[0]
    rows = _cell_rows(tables.stdft_examples, cell)
    layers = _representative_layers(rows["layer"])
    classes = (0, 1)
    common: dict[int, str | None] = {}
    for y_true in classes:
        sets = [
            set(rows[(rows["layer"] == layer) & (rows["y_true"] == y_true)]["sample_id"])
            for layer in layers
        ]
        shared = set.intersection(*sets) if sets else set()
        common[y_true] = min(shared) if shared else None

    def chosen(layer: int, y_true: int) -> str | None:
        if common[y_true] is not None:
            return common[y_true]
        part = rows[(rows["layer"] == layer) & (rows["y_true"] == y_true)]
        return str(part["sample_id"].iloc[0]) if len(part) else None

    selected = {
        (layer, y_true): chosen(layer, y_true) for layer in layers for y_true in classes
    }
    limit = 0.0
    absolute_values: list[np.ndarray] = []
    for (layer, _), sample_id in selected.items():
        if sample_id is not None:
            payload = tables.stdft_payloads[(*cell, layer, sample_id)]
            absolute = np.abs(payload["relevance"])
            absolute_values.append(absolute.ravel())
            limit = max(limit, float(absolute.max()))
    limit = limit or 1.0
    positive = np.concatenate(absolute_values)
    positive = positive[positive > 0.0]
    linear_threshold = (
        float(np.percentile(positive, 95.0))
        if positive.size
        else np.finfo(np.float64).eps
    )
    norm = SymLogNorm(
        linthresh=max(linear_threshold, np.finfo(np.float64).eps),
        linscale=1.0,
        vmin=-limit,
        vmax=limit,
        base=10,
    )
    display_ids = _display_sample_ids([item for item in selected.values() if item is not None])
    fig, axes = plt.subplots(
        2, len(layers), figsize=(_FIGURE_WIDTH, 4.8), squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, y_true in enumerate(classes):
        for column, layer in enumerate(layers):
            ax = axes[row, column]
            sample_id = selected[(layer, y_true)]
            ax.grid(False)
            if sample_id is None:
                ax.set_axis_off()
                continue
            payload = tables.stdft_payloads[(*cell, layer, sample_id)]
            image = ax.pcolormesh(
                payload["times"], payload["freqs"], payload["relevance"].T,
                shading="auto", cmap="RdBu_r", norm=norm,
            )
            ax.set_title(
                f"camada {layer} \u00b7 {_CLASS_NAMES_PT[y_true]}\n{display_ids[sample_id]}",
                fontsize=7,
            )
            if row == 1:
                ax.set_xlabel("Tempo (s)")
            if column == 0:
                ax.set_ylabel("Frequência (Hz)")
    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.85,
        label="Relevância STDFT com sinal\n(escala simétrica não linear)",
    )
    inner_ticks = sorted(
        value for value in {linear_threshold, 1.0} if 0.0 < value < limit
    )
    colorbar_ticks = [
        -limit,
        *(-value for value in reversed(inner_ticks)),
        0.0,
        *inner_ticks,
        limit,
    ]
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_ticklabels(
        [f"{value:.2g}".replace("-", "\N{MINUS SIGN}") for value in colorbar_ticks]
    )
    record = _record(
        tables,
        name,
        (),
        heading="Exemplos de relevância STDFT",
        description=(
            "Mapas representativos de relevância tempo\u2013frequência (janela de Hann de 512, "
            "passo 256) de uma amostra real e uma sintética do conjunto fixo na primeira, na "
            f"intermediária e na última camada; exibidos para {_cell_label(cell)}."
        ),
        metric="relevância STDFT-LRP com sinal por célula tempo\u2013frequência",
        units=(
            "relevância na escala do logit (com sinal); cores em escala simétrica "
            "não linear comum entre painéis; tempo em s, frequência em Hz"
        ),
        transformation="relevância AttnLRP no domínio do tempo \u2192 STDFT-LRP com janelamento (WOLA), conservando a relevância total",
        cells=(cell,),
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _joined(values: Sequence[object]) -> str:
    unique = sorted(set(values), key=str)
    if len(unique) == 1:
        return str(unique[0])
    if all(isinstance(value, (int, np.integer)) for value in unique):
        return f"{min(unique)}\u2013{max(unique)}"
    return ", ".join(str(value) for value in unique)


def _validation_sample_display(table: pd.DataFrame) -> str:
    """Abbreviated validation sample ID(s); the full IDs stay in the tables."""
    ids = [str(item) for item in table["validation_sample_id"]]
    shown = _display_sample_ids(ids)
    return ", ".join(shown[item] for item in sorted(set(ids)))


def build_conservation_series(tables: ReportTables) -> pd.DataFrame:
    """Per-layer worst case of each conservation check, with its own scope and limit.

    The checks cover different populations and obey different rules, so each one
    keeps its own panel, scope text and limit instead of sharing one tolerance.
    """
    table = tables.conservation
    tolerance = float(table["tolerance"].min())
    worst = table.groupby("layer", sort=True).max(numeric_only=True)
    scope_bias = (
        f"amostra única ({_joined(table['validation_kind'])}; amostra de "
        f"validação {_validation_sample_display(table)}), n=1"
    )
    scope_dft = f"conjunto fixo de amostras XAI (n={_joined(table['n'])})"
    scope_stdft = f"subconjunto STDFT (n={_joined(table['n_stdft'])})"
    scope_score = (
        f"conjunto fixo de amostras XAI (n={_joined(table['n'])}); aceito se "
        f"|\u0394p| \u2264 atol + rtol\u00b7|p| (rtol={_joined_sci(table['score_recompute_rtol'])}, "
        f"atol={_joined_sci(table['score_recompute_atol'])})"
    )
    tolerance_label = f"tolerância de conservação ({_sci(tolerance)})"
    panels = (
        (
            "bias_zeroed_single_sample",
            "bias_zeroed_validation_residual",
            "Verificação bias-zeroed do modelo/regra\n(resíduo relativo)",
            scope_bias,
            tolerance,
            tolerance_label,
        ),
        (
            "dft_xai_cohort",
            "max_dft_residual",
            "Conservação da DFT, máximo no conjunto fixo\n(resíduo relativo)",
            scope_dft,
            tolerance,
            tolerance_label,
        ),
        (
            "stdft_subset",
            "max_stdft_conservation_relative_error",
            "Conservação da STDFT, máximo no subconjunto\n(resíduo relativo)",
            scope_stdft,
            tolerance,
            tolerance_label,
        ),
        (
            "score_recompute_rtol_atol",
            "max_score_recompute_ratio",
            "Recálculo do score, máximo no conjunto fixo\n(|\u0394p| / (atol + rtol\u00b7|p|))",
            scope_score,
            1.0,
            "limite de aceitação rtol/atol (razão = 1)",
        ),
    )
    frames = []
    for panel, column, label, scope, limit, limit_label in panels:
        frames.append(
            pd.DataFrame(
                {
                    "panel": panel,
                    "layer": worst.index.to_numpy(),
                    "value": worst[column].to_numpy(dtype=np.float64),
                    "limit": limit,
                    "limit_label": limit_label,
                    "value_label": label,
                    "scope": scope,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _figure_conservation(plt, tables, figures_dir):
    name = "conservation_diagnostics"
    series = build_conservation_series(tables)
    panels = list(dict.fromkeys(series["panel"]))
    fig, axes = plt.subplots(
        2, 2, figsize=(_FIGURE_WIDTH, 5.6), sharex=True, squeeze=False,
        constrained_layout=True,
    )
    for ax, panel in zip(axes.ravel(), panels):
        rows = series[series["panel"] == panel]
        first = rows.iloc[0]
        ax.plot(
            rows["layer"],
            np.maximum(rows["value"].to_numpy(dtype=np.float64), _CONSERVATION_FLOOR),
            marker="o", markersize=3, label="máximo entre células",
        )
        ax.axhline(float(first["limit"]), color="0.2", linestyle="--", linewidth=1.0,
                   label=str(first["limit_label"]))
        ax.set_yscale("log")
        ax.set_ylabel(str(first["value_label"]), fontsize=7)
        ax.set_title(str(first["scope"]), fontsize=6.5)
        ax.legend(loc="best", fontsize=6)
        _layer_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel(_LAYER_AXIS_LABEL)
    scopes = " | ".join(
        f"{panel}: {series[series['panel'] == panel].iloc[0]['scope']}"
        for panel in panels
    )
    record = _record(
        tables,
        name,
        (),
        heading="Diagnósticos de conservação por camada",
        description=(
            "Quatro verificações com escopos distintos, cada uma contra o seu limite. "
            "A verificação bias-zeroed usa uma única amostra para validar a decomposição "
            "modelo/regra, não o conjunto inteiro; o resíduo da DFT cobre o conjunto "
            "fixo de amostras XAI; o resíduo "
            "da STDFT cobre apenas o subconjunto STDFT; o recálculo do score é aceito "
            f"pela regra rtol/atol, não pela tolerância de conservação. Escopos: {scopes}."
        ),
        metric="piores resíduos de conservação por camada e razão do recálculo do score",
        units=(
            "resíduos relativos adimensionais nas verificações bias-zeroed, DFT e STDFT "
            "(limite: tolerância de conservação); razão do recálculo do score |\u0394p| / "
            "(atol + rtol\u00b7|p|) (limite: 1); escala logarítmica"
        ),
        transformation=(
            "máximo sobre as amostras de cada escopo e sobre as células por camada; "
            f"valores abaixo de {_sci(_CONSERVATION_FLOOR)} são desenhados nesse piso"
        ),
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


_ENCODER_AGREEMENT_TRANSFORMATION = (
    "mediana por camada agregada sobre células treino→avaliação; faixa sombreada "
    "entre quartis 25 e 75 % dessas células (não reproduz ci_low/ci_high); "
    "intervalos bootstrap por linha permanecem no CSV, não nos resumos TeX compactos"
)
_BOOTSTRAP_CI_TRANSFORMATION = (
    "cada curva usa o ponto estimado persistido e a faixa sombreada reproduz "
    "o intervalo de confiança bootstrap de 95 % (ci_low–ci_high) da mesma linha "
    "da tabela, interrompida quando o IC não é finito; os resumos TeX agregam "
    "medianas e faixas entre linhas, sem repetir ci_low/ci_high linha a linha"
)
_COMPARISON_METRIC_LABELS_PT: Mapping[str, str] = MappingProxyType(
    {
        "score_spearman": "Spearman do score",
        "prediction_agreement": "Acordo de decisão",
        "cohen_kappa": "Kappa de Cohen",
        "relevance_cosine_similarity": "Similaridade cosseno",
        "relevance_jensen_shannon_distance": "Distância JS",
        "jensen_shannon_distance": "Distância JS",
        "delta_roc_auc": "Δ ROC-AUC",
        "delta_mcc": "Δ MCC",
        "wasserstein_distance": "Wasserstein (Hz)",
    }
)
_COMPARISON_LEGEND_KW = {
    "loc": "outside right center",
    "ncol": 1,
    "fontsize": 6.5,
    "frameon": False,
}
_PERF_ENCODER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("score_spearman", "Spearman do score", "ρ"),
    ("prediction_agreement", "Acordo de decisão", "fração"),
    ("cohen_kappa", "Kappa de Cohen", "κ"),
)
_XAI_ENCODER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("relevance_cosine_similarity", "Similaridade cosseno", "cosseno"),
    ("relevance_jensen_shannon_distance", "Distância JS", "bits"),
)
_ENCODER_HEATMAP_METRICS: tuple[
    tuple[str, str, tuple[float, float], str], ...
] = (
    ("prediction_agreement", "Acordo de decisão", (0.0, 1.0), "viridis"),
    ("cohen_kappa", "Kappa de Cohen", (-1.0, 1.0), "coolwarm"),
    ("score_spearman", "Spearman do score", (-1.0, 1.0), "coolwarm"),
    (
        "relevance_cosine_similarity",
        "Similaridade cosseno",
        (-1.0, 1.0),
        "coolwarm",
    ),
    (
        "relevance_jensen_shannon_distance",
        "Distância JS",
        (0.0, 1.0),
        "magma",
    ),
)


def _layer_axis_dynamic(ax, layers: Sequence[int]) -> None:
    ticks = sorted({int(value) for value in layers})
    if not ticks:
        return
    _set_layer_ticks(ax, ticks)
    ax.set_xlim(min(ticks) - 0.4, max(ticks) + 0.4)


def _median_iqr_by_layer(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.groupby([*keys, "layer"], sort=True)
    for index, group in grouped:
        if not isinstance(index, tuple):
            index = (index,)
        layer = int(index[-1])
        prefix = index[:-1]
        values = group["estimate"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                **dict(zip(keys, prefix)),
                "layer": layer,
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def _plot_median_band(ax, summary: pd.DataFrame, *, label: str | None = None) -> None:
    ordered = summary.sort_values("layer")
    layers = ordered["layer"].to_numpy(dtype=int)
    ax.plot(
        layers,
        ordered["median"],
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label=label,
    )
    ax.fill_between(layers, ordered["q25"], ordered["q75"], alpha=0.2)
    _layer_axis_dynamic(ax, layers)


def _comparison_metric_label(metric: str) -> str:
    return _COMPARISON_METRIC_LABELS_PT.get(metric, metric)


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _plot_persisted_bootstrap_ci_segmented(
    ax, series: pd.DataFrame, *, label: str | None = None
) -> None:
    ordered = series.sort_values("layer")
    layers = ordered["layer"].to_numpy(dtype=int)
    estimate = ordered["estimate"].to_numpy(dtype=np.float64)
    low = ordered["ci_low"].to_numpy(dtype=np.float64)
    high = ordered["ci_high"].to_numpy(dtype=np.float64)
    ci_finite = np.isfinite(estimate) & np.isfinite(low) & np.isfinite(high)
    label_used = False
    for start, end in _contiguous_true_runs(ci_finite):
        segment = slice(start, end)
        ax.plot(
            layers[segment],
            estimate[segment],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            label=label if not label_used else None,
        )
        ax.fill_between(
            layers[segment],
            low[segment],
            high[segment],
            alpha=0.22,
        )
        label_used = True
    estimate_only = np.isfinite(estimate) & ~ci_finite
    if estimate_only.any():
        ax.plot(
            layers[estimate_only],
            estimate[estimate_only],
            marker="x",
            linestyle="none",
            markersize=3.0,
            color="0.35",
        )
    if np.isfinite(estimate).any():
        _layer_axis_dynamic(ax, layers[np.isfinite(estimate)])


def _plot_persisted_bootstrap_ci(
    ax, series: pd.DataFrame, *, label: str | None = None
) -> None:
    _plot_persisted_bootstrap_ci_segmented(ax, series, label=label)


def _pair_pretty(model_a: str, model_b: str) -> str:
    return f"{_model_name(model_a)}|{_model_name(model_b)}"


def _transfer_pretty(source: str, target: str) -> str:
    return f"{_language_name(source)}\u2192{_language_name(target)}"


def _language_pair_pretty(language_a: str, language_b: str) -> str:
    return f"{_language_name(language_a)}\u2194{_language_name(language_b)}"


def _canonical_language_pairs(
    frame: pd.DataFrame, first: str, second: str
) -> tuple[tuple[str, str], ...]:
    present = {
        (str(row[0]), str(row[1]))
        for row in frame[[first, second]].drop_duplicates().itertuples(
            index=False, name=None
        )
    }
    return tuple(
        (language_a, language_b)
        for index, language_a in enumerate(LANGUAGE_ORDER)
        for language_b in LANGUAGE_ORDER[index + 1 :]
        if (language_a, language_b) in present
    )


def _finalize_comparison_line_figure(
    fig,
    axes,
    *,
    legend_title: str | None = None,
) -> None:
    flat = np.atleast_1d(axes).ravel()
    for axis in flat:
        if axis.get_visible():
            axis.set_xlabel("")
    handles: list[object] = []
    labels: list[str] = []
    seen: set[str] = set()
    for axis in flat:
        if not axis.get_visible():
            continue
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if not label or label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if handles:
        legend = fig.legend(
            handles,
            labels,
            title=legend_title,
            title_fontsize=7 if legend_title else None,
            **_COMPARISON_LEGEND_KW,
        )
        if legend_title:
            legend.get_title().set_fontsize(7)
    fig.supxlabel(_LAYER_AXIS_LABEL, fontsize=8)
    fig.set_layout_engine("constrained", h_pad=0.04, w_pad=0.04)


def encoder_agreement_heatmap_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse finite encoder-agreement estimates to layer medians."""
    keys = ("model_a", "model_b", "source", "target", "metric")
    finite = frame[np.isfinite(frame["estimate"].to_numpy(dtype=np.float64))]
    if finite.empty:
        return pd.DataFrame(columns=(*keys, "estimate"))
    return (
        finite.groupby(list(keys), sort=True, as_index=False)["estimate"]
        .median()
        .reset_index(drop=True)
    )


def _figure_encoder_agreement_heatmaps(
    plt, tables: ReportTables, figures_dir: Path
):
    name = "encoder_agreement_heatmaps"
    frame = encoder_agreement_heatmap_data(tables.encoder_agreement)
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    pairs = sorted(
        {
            (str(row.model_a), str(row.model_b))
            for row in frame[["model_a", "model_b"]].drop_duplicates().itertuples(
                index=False
            )
        }
    )
    languages = tuple(
        language
        for language in ("eng", "por", "zho")
        if language in set(frame["source"]) | set(frame["target"])
    )
    fig, axes = plt.subplots(
        len(_ENCODER_HEATMAP_METRICS),
        len(pairs),
        figsize=(_FIGURE_WIDTH, 2.25 * len(_ENCODER_HEATMAP_METRICS)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (metric, label, limits, cmap) in enumerate(
        _ENCODER_HEATMAP_METRICS
    ):
        for column_index, (model_a, model_b) in enumerate(pairs):
            ax = axes[row_index, column_index]
            rows = frame[
                (frame["model_a"] == model_a)
                & (frame["model_b"] == model_b)
                & (frame["metric"] == metric)
            ]
            matrix = (
                rows.pivot(index="source", columns="target", values="estimate")
                .reindex(index=languages, columns=languages)
                .to_numpy(dtype=np.float64)
            )
            image = ax.imshow(
                matrix,
                vmin=limits[0],
                vmax=limits[1],
                cmap=cmap,
                aspect="equal",
            )
            ax.set_xticks(range(len(languages)))
            ax.set_xticklabels(
                [_language_name(item).capitalize() for item in languages],
                rotation=30,
                ha="right",
            )
            ax.set_yticks(range(len(languages)))
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(False)
            if row_index == len(_ENCODER_HEATMAP_METRICS) - 1:
                ax.set_xlabel("Idioma de avaliação", fontsize=8)
            if column_index == 0:
                ax.set_yticklabels(
                    [_language_name(item).capitalize() for item in languages]
                )
                ax.set_ylabel("Idioma de treino", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(
                f"{_model_name(model_a)} × {_model_name(model_b)}\n{label}",
                fontsize=7,
            )
            midpoint = (limits[0] + limits[1]) / 2.0
            for source_index in range(len(languages)):
                for target_index in range(len(languages)):
                    value = matrix[source_index, target_index]
                    if np.isfinite(value):
                        ax.text(
                            target_index,
                            source_index,
                            _figure_number(value),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if value < midpoint else "black",
                        )
        colorbar = fig.colorbar(
            image, ax=axes[row_index, :].tolist(), shrink=0.85, aspect=12
        )
        colorbar.set_label(label, fontsize=7)
        colorbar.ax.tick_params(labelsize=7)
    layer_count = tables.encoder_agreement["layer"].nunique()
    record = _record(
        tables,
        name,
        (),
        heading="Acordo entre encoders por idioma",
        description=(
            "Mapas por par de modelos e métrica, com idioma de treino nas linhas "
            "e idioma de avaliação nas colunas. "
            f"Cada célula é a mediana entre {layer_count} camadas."
        ),
        metric="acordo entre encoders (desempenho e explicação)",
        units="fração, κ, ρ, cosseno e bits, cada métrica em sua própria escala",
        transformation=(
            f"mediana das estimativas finitas entre as {layer_count} camadas para cada par "
            "de modelos, idioma de treino, idioma de avaliação e métrica"
        ),
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_encoder_agreement_lines(
    plt, tables: ReportTables, figures_dir: Path, name: str
):
    frame = tables.encoder_agreement
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    pairs = sorted(
        {
            (str(row.model_a), str(row.model_b))
            for row in frame[["model_a", "model_b"]].drop_duplicates().itertuples(
                index=False
            )
        }
    )
    n_pairs = len(pairs)
    n_cols = len(_PERF_ENCODER_METRICS)
    fig, axes = plt.subplots(
        2 * n_pairs,
        n_cols,
        figsize=(_FIGURE_WIDTH, 2.1 * 2 * n_pairs + 0.8),
        squeeze=False,
    )
    for pair_index, (model_a, model_b) in enumerate(pairs):
        pair_frame = frame[
            (frame["model_a"] == model_a) & (frame["model_b"] == model_b)
        ]
        for _block_index, (metrics, row_block) in enumerate(
            ((_PERF_ENCODER_METRICS, 0), (_XAI_ENCODER_METRICS, 1))
        ):
            for metric_index, (metric, title, ylabel) in enumerate(metrics):
                ax = axes[2 * pair_index + row_block, metric_index]
                metric_frame = pair_frame[pair_frame["metric"] == metric]
                if metric_frame.empty:
                    ax.set_visible(False)
                    continue
                summary = _median_iqr_by_layer(
                    metric_frame, ("model_a", "model_b")
                )
                if summary.empty:
                    ax.set_visible(False)
                    continue
                _plot_median_band(ax, summary)
                if pair_index == 0:
                    ax.set_title(title, fontsize=8)
                if metric_index == 0:
                    ax.set_ylabel(
                        f"{_pair_pretty(model_a, model_b)}\n{ylabel}", fontsize=7
                    )
                else:
                    ax.set_ylabel(ylabel, fontsize=7)
            for extra_col in range(len(metrics), n_cols):
                axes[2 * pair_index + row_block, extra_col].set_visible(False)
    record = _record(
        tables,
        name,
        (),
        heading="Acordo entre encoders por camada",
        description=(
            "Pares de modelos com curvas separadas para acordos de desempenho "
            "(Spearman do score, fração de decisões iguais e kappa) e para "
            "explicações (similaridade cosseno e distância JS em escalas distintas). "
            f"Cada painel agrega células treino→avaliação transparentemente."
        ),
        metric="acordo entre encoders (desempenho e explicação)",
        units="ρ, fração, κ, cosseno e bits em eixos rotulados separadamente",
        transformation=_ENCODER_AGREEMENT_TRANSFORMATION,
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(fig, axes)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_encoder_agreement_by_layer(
    plt, tables: ReportTables, figures_dir: Path
):
    return _figure_encoder_agreement_lines(
        plt, tables, figures_dir, "encoder_agreement_by_layer"
    )


def _figure_language_shift(plt, tables: ReportTables, figures_dir: Path):
    name = "language_shift"
    frame = tables.language_shift
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    models = sorted(set(frame["model"]))
    sources = sorted(set(frame["source"]))
    n_rows = len(models)
    n_cols = len(sources)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(_FIGURE_WIDTH, 2.0 * n_rows + 0.8),
        squeeze=False,
        sharex=True,
    )
    for row, model in enumerate(models):
        for column, source in enumerate(sources):
            ax = axes[row, column]
            subset = frame[(frame["model"] == model) & (frame["source"] == source)]
            for (target_a, target_b), pair_frame in subset.groupby(
                ["target_a", "target_b"], sort=True
            ):
                for group in ("all", "real", "synthetic"):
                    group_frame = pair_frame[pair_frame["group"] == group]
                    if group_frame.empty:
                        continue
                    _plot_persisted_bootstrap_ci(
                        ax,
                        group_frame,
                        label=(
                            f"{_language_pair_pretty(target_a, target_b)} · "
                            f"{_GROUP_LABELS_PT.get(group, group)}"
                        ),
                    )
            if row == 0:
                ax.set_title(
                    f"Probe fixo · {_language_name(source)}", fontsize=8
                )
            if column == 0:
                ax.set_ylabel(f"{_model_name(model)}\nJS (bits)", fontsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Mudança de relevância entre idiomas-alvo",
        description=(
            "Distância de Jensen–Shannon entre distribuições de relevância sob o "
            "mesmo probe fixo e idioma de treino, comparando pares de idiomas-alvo "
            "ao longo das camadas."
        ),
        metric="distância JS entre idiomas-alvo",
        units="bits (logaritmo base 2)",
        transformation=_BOOTSTRAP_CI_TRANSFORMATION,
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(
        fig, axes, legend_title="Par de idiomas-alvo e grupo"
    )
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_language_shift_intervals(
    plt, tables: ReportTables, figures_dir: Path
) -> FigureRecord:
    name = "language_shift_intervals"
    frame = tables.language_shift
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    frame = frame[frame["group"] == "all"]
    present_models = set(frame["model"])
    present_sources = set(frame["source"])
    models = tuple(model for model in tables.models if model in present_models)
    sources = tuple(
        language for language in LANGUAGE_ORDER if language in present_sources
    )
    target_pairs = _canonical_language_pairs(frame, "target_a", "target_b")
    fig, axes = plt.subplots(
        len(models),
        len(sources),
        figsize=(_FIGURE_WIDTH, 2.0 * len(models) + 0.8),
        squeeze=False,
        sharex=True,
    )
    for row, model in enumerate(models):
        for column, source in enumerate(sources):
            ax = axes[row, column]
            subset = frame[(frame["model"] == model) & (frame["source"] == source)]
            for target_a, target_b in target_pairs:
                pair_frame = subset[
                    (subset["target_a"] == target_a)
                    & (subset["target_b"] == target_b)
                ]
                if pair_frame.empty:
                    continue
                ordered = pair_frame.sort_values("layer")
                layers = ordered["layer"].to_numpy(dtype=int)
                estimate = ordered["estimate"].to_numpy(dtype=np.float64)
                low = ordered["ci_low"].to_numpy(dtype=np.float64)
                high = ordered["ci_high"].to_numpy(dtype=np.float64)
                point_mask = np.isfinite(estimate)
                interval_mask = (
                    point_mask & np.isfinite(low) & np.isfinite(high)
                )
                label = _language_pair_pretty(str(target_a), str(target_b))
                point_color = None
                if point_mask.any():
                    points = ax.errorbar(
                        layers[point_mask],
                        estimate[point_mask],
                        fmt="o",
                        markersize=3.0,
                        linestyle="none",
                        label=label,
                    )
                    point_color = points.lines[0].get_color()
                    _layer_axis_dynamic(ax, layers[point_mask])
                if interval_mask.any():
                    ax.vlines(
                        layers[interval_mask],
                        low[interval_mask],
                        high[interval_mask],
                        linewidth=1.0,
                        color=point_color,
                    )
            if row == 0:
                ax.set_title(f"Probe fixo · {_language_name(source)}", fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{_model_name(model)}\nJS (bits)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Mudança de relevância com intervalos persistidos",
        description=(
            "Estimativas pontuais da distância de Jensen–Shannon para o grupo "
            "agregado, com uma haste por intervalo bootstrap persistido."
        ),
        metric="distância JS entre idiomas-alvo",
        units="bits (logaritmo base 2)",
        transformation=(
            "pontos = estimate e hastes = ci_low–ci_high persistidos para cada "
            "par e camada do grupo all; nenhuma distribuição ou densidade é "
            "inferida dos intervalos"
        ),
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(
        fig, axes, legend_title="Par de idiomas-alvo"
    )
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_diagonal_vs_offdiagonal(plt, tables: ReportTables, figures_dir: Path):
    name = "diagonal_vs_offdiagonal"
    frame = tables.diagonal_vs_offdiagonal
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    models = sorted(set(frame["model"]))
    metrics = (
        ("delta_roc_auc", "Δ ROC-AUC (fora − diag.)", "adimensional"),
        ("delta_mcc", "Δ MCC (fora − diag.)", "adimensional"),
    )
    fig, axes = plt.subplots(
        len(models),
        len(metrics),
        figsize=(_FIGURE_WIDTH, 2.2 * len(models) + 0.8),
        squeeze=False,
        sharex=True,
    )
    for row, model in enumerate(models):
        model_frame = frame[frame["model"] == model]
        for column, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row, column]
            metric_frame = model_frame[model_frame["metric"] == metric]
            for (source, target), group in metric_frame.groupby(
                ["source", "target"], sort=True
            ):
                _plot_persisted_bootstrap_ci(
                    ax,
                    group,
                    label=_transfer_pretty(str(source), str(target)),
                )
            ax.axhline(0.0, color="0.45", linewidth=0.8)
            if row == 0:
                ax.set_title(title, fontsize=8)
            if column == 0:
                ax.set_ylabel(f"{_model_name(model)}\n{ylabel}", fontsize=7)
            else:
                ax.set_ylabel(ylabel, fontsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Diagonal versus transferência fora da diagonal",
        description=(
            "Diferença (fora da diagonal menos diagonal) de ROC-AUC e MCC sobre "
            "o mesmo conjunto de teste, por camada, com referência em zero."
        ),
        metric="Δ desempenho off-diagonal − diagonal",
        units="ROC-AUC e MCC adimensionais",
        transformation=_BOOTSTRAP_CI_TRANSFORMATION,
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(
        fig, axes, legend_title="Transferência (treino→avaliação)"
    )
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_spectral_divergence(plt, tables: ReportTables, figures_dir: Path):
    name = "spectral_divergence"
    frame = tables.spectral_divergence
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    models = sorted(set(frame["model"]))
    pairs = sorted(
        {
            (str(row.language_a), str(row.language_b))
            for row in frame[["language_a", "language_b"]].drop_duplicates().itertuples(
                index=False
            )
        }
    )
    n_cols = len(pairs)
    fig, axes = plt.subplots(
        len(models),
        n_cols,
        figsize=(_FIGURE_WIDTH, 2.0 * len(models) + 0.6),
        squeeze=False,
        sharex=True,
    )
    axes = np.atleast_2d(axes)
    for row, model in enumerate(models):
        model_frame = frame[frame["model"] == model]
        for column, (language_a, language_b) in enumerate(pairs):
            ax = axes[row, column]
            pair_frame = model_frame[
                (model_frame["language_a"] == language_a)
                & (model_frame["language_b"] == language_b)
            ]
            for group in ("all", "real", "synthetic"):
                group_frame = pair_frame[pair_frame["group"] == group]
                if group_frame.empty:
                    continue
                _plot_persisted_bootstrap_ci(
                    ax,
                    group_frame,
                    label=_GROUP_LABELS_PT.get(group, group),
                )
            if row == 0:
                ax.set_title(
                    f"{_language_pair_pretty(language_a, language_b)} (diag.)",
                    fontsize=8,
                )
            if column == 0:
                ax.set_ylabel(f"{_model_name(model)}\nWasserstein (Hz)", fontsize=7)
            else:
                ax.set_ylabel("Wasserstein (Hz)", fontsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Divergência espectral entre idiomas (pares diagonais)",
        description=(
            "Distância de Wasserstein entre espectros de relevância em Hz, "
            "comparando pares de idiomas na diagonal treino=avaliação."
        ),
        metric="Wasserstein entre distribuições espectrais",
        units="Hz (suporte nos centros de banda registrados)",
        transformation=_BOOTSTRAP_CI_TRANSFORMATION,
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(fig, axes, legend_title="Grupo de classe")
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_spectral_divergence_by_layer(
    plt, tables: ReportTables, figures_dir: Path
) -> FigureRecord:
    name = "spectral_divergence_by_layer"
    frame = tables.spectral_divergence
    if frame.empty:
        raise ValueError(f"{name} figure requested but the comparison table is empty")
    frame = frame[frame["group"] == "all"]
    present_models = set(frame["model"])
    models = tuple(model for model in tables.models if model in present_models)
    pairs = _canonical_language_pairs(frame, "language_a", "language_b")
    fig, axes = plt.subplots(
        len(models),
        len(pairs),
        figsize=(_FIGURE_WIDTH, 2.0 * len(models) + 0.6),
        squeeze=False,
        sharex=True,
    )
    for row, model in enumerate(models):
        model_frame = frame[frame["model"] == model]
        for column, (language_a, language_b) in enumerate(pairs):
            ax = axes[row, column]
            pair_frame = model_frame[
                (model_frame["language_a"] == language_a)
                & (model_frame["language_b"] == language_b)
            ]
            _plot_persisted_bootstrap_ci(ax, pair_frame)
            if row == 0:
                ax.set_title(
                    f"{_language_pair_pretty(language_a, language_b)} (diag.)",
                    fontsize=8,
                )
            if column == 0:
                ax.set_ylabel(f"{_model_name(model)}\nWasserstein (Hz)", fontsize=7)
            else:
                ax.set_ylabel("Wasserstein (Hz)", fontsize=7)
    record = _record(
        tables,
        name,
        (),
        heading="Divergência espectral por camada",
        description=(
            "Distância de Wasserstein entre espectros de relevância dos sistemas "
            "diagonais, em pequenos múltiplos por modelo e par de idiomas."
        ),
        metric="Wasserstein entre distribuições espectrais",
        units="Hz (suporte nos centros de banda registrados)",
        transformation=(
            "camada no eixo x e estimate em Hz no eixo y para o grupo all; "
            "faixa = ci_low–ci_high bootstrap persistido da mesma linha"
        ),
    )
    fig.suptitle(record.title, fontsize=9)
    _finalize_comparison_line_figure(fig, axes)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


def _figure_layer_faithfulness_intervals(
    plt, tables: ReportTables, figures_dir: Path
) -> FigureRecord:
    name = "layer_faithfulness_intervals"
    aligned = _aligned_layer_faithfulness(tables)
    if aligned.empty:
        raise ValueError(f"{name} figure requested but the aligned table is empty")
    frame = _displayable_faithfulness_rows(aligned)
    comparisons = _FAITHFULNESS_CONTROLS
    if frame.empty or set(frame["comparison"]) != set(comparisons):
        raise ValueError(
            f"{name} requires diagonal layer {FINAL_LAYER}, "
            f"{FAITHFULNESS_PER_CLASS} samples per class, "
            "and both registered controls"
        )
    present_models = set(frame["model"])
    models = tuple(model for model in tables.models if model in present_models)
    present_languages = set(frame["source"])
    languages = tuple(
        language for language in LANGUAGE_ORDER if language in present_languages
    )
    classes = tuple(
        true_class
        for true_class in _TRUE_CLASS_ORDER
        if true_class in set(frame["true_class"])
    )
    if classes != _TRUE_CLASS_ORDER:
        raise ValueError(
            f"{name} requires canonical true classes "
            f"{', '.join(_TRUE_CLASS_ORDER)}"
        )
    ks = tuple(sorted({int(value) for value in frame["k"]}))
    incomplete = []
    required_controls = set(comparisons)
    for model in models:
        model_languages = sorted(
            set(frame.loc[frame["model"] == model, "source"]),
            key=LANGUAGE_ORDER.index,
        )
        for language in model_languages:
            for true_class in classes:
                for k_value in ks:
                    group = frame[
                        (frame["model"] == model)
                        & (frame["source"] == language)
                        & (frame["true_class"] == true_class)
                        & (frame["k"] == k_value)
                    ]
                    observed = set(group["comparison"])
                    if observed != required_controls:
                        missing = ", ".join(sorted(required_controls - observed))
                        incomplete.append(
                            f"model={model}, language={language}, "
                            f"true_class={true_class}, k={k_value}: "
                            f"{missing or 'unexpected controls'}"
                        )
    if incomplete:
        raise ValueError(
            f"{name} missing required controls for " + "; ".join(incomplete)
        )
    from matplotlib.lines import Line2D

    series = [(language, comparison) for language in languages for comparison in comparisons]
    spread = 0.72
    step = spread / max(len(series) - 1, 1)
    offsets = {
        item: (index - (len(series) - 1) / 2.0) * step
        for index, item in enumerate(series)
    }
    fig, axes = plt.subplots(
        len(classes),
        len(models),
        figsize=(_FIGURE_WIDTH, len(classes) * (0.5 + 0.32 * len(ks)) + 1.3),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    k_position = {k_value: index for index, k_value in enumerate(ks)}
    for row, true_class in enumerate(classes):
        for column, model in enumerate(models):
            ax = axes[row, column]
            subset = frame[
                (frame["model"] == model) & (frame["true_class"] == true_class)
            ]
            for language, comparison in series:
                ordered = subset[
                    (subset["source"] == language)
                    & (subset["comparison"] == comparison)
                ].sort_values("k")
                if ordered.empty:
                    continue
                estimate = ordered["mean_difference"].to_numpy(dtype=np.float64)
                low = ordered["ci_low"].to_numpy(dtype=np.float64)
                high = ordered["ci_high"].to_numpy(dtype=np.float64)
                positions = np.array(
                    [k_position[int(value)] for value in ordered["k"]],
                    dtype=np.float64,
                )
                mask = np.isfinite(estimate) & np.isfinite(low) & np.isfinite(high)
                if not mask.any():
                    continue
                color = _FAITHFULNESS_CONTROL_COLORS[comparison]
                y = positions[mask] + offsets[(language, comparison)]
                ax.hlines(y, low[mask], high[mask], color=color, linewidth=0.9)
                ax.plot(
                    estimate[mask],
                    y,
                    marker=_FAITHFULNESS_LANGUAGE_MARKERS[language],
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    linestyle="none",
                    markersize=4.0,
                    gid=f"{language}|{comparison}",
                )
            ax.axvline(0.0, color="0.45", linewidth=0.8)
            ax.set_yticks(range(len(ks)))
            ax.set_yticklabels([str(k_value) for k_value in ks])
            ax.set_ylim(len(ks) - 0.5, -0.5)
            ax.tick_params(axis="both", labelsize=7)
            ax.locator_params(axis="x", nbins=4)
            if row == 0:
                ax.set_title(_model_name(model), fontsize=8)
            if column == 0:
                ax.set_ylabel(
                    f"{_TRUE_CLASS_NAMES_PT[true_class]}\n$k$", fontsize=7
                )
            if row == len(classes) - 1:
                ax.set_xlabel("Diferença média pareada", fontsize=7)
    language_handles = [
        Line2D(
            [],
            [],
            marker=_FAITHFULNESS_LANGUAGE_MARKERS[language],
            color="0.35",
            linestyle="none",
            markersize=4.5,
            label=_language_name(language).capitalize(),
        )
        for language in languages
    ]
    control_handles = [
        Line2D(
            [],
            [],
            color=_FAITHFULNESS_CONTROL_COLORS[comparison],
            linewidth=1.6,
            label=_FAITHFULNESS_COMPARISON_PT[comparison],
        )
        for comparison in comparisons
    ]
    legend_style = {"fontsize": 7, "title_fontsize": 7, "frameon": False}
    fig.legend(
        handles=language_handles,
        loc="outside lower left",
        ncol=len(language_handles),
        title="Idioma (formato do marcador)",
        **legend_style,
    )
    fig.legend(
        handles=control_handles,
        loc="outside lower right",
        ncol=len(control_handles),
        title="Comparação (cor)",
        **legend_style,
    )
    cells = tuple(
        row
        for row in _cells(tables)
        if row[1] == row[2] and row[0] in set(frame["model"])
    )
    record = _record(
        tables,
        name,
        (),
        heading="Fidelidade por intervenção",
        description=(
            f"Camada {FINAL_LAYER}, escopo diagonal e "
            f"{_count_word_pt(FAITHFULNESS_PER_CLASS)} amostras por classe; "
            "diferenças pareadas top-minus-random e top-minus-bottom. Limite "
            "principal: a intervenção STFT controlada não implica causalidade "
            "no mundo real."
        ),
        metric="diferença média de comprehensividade",
        units="adimensional",
        transformation=(
            "colunas = modelos com dados alinhados; linhas = classes verdadeiras "
            "(real, sintético); k no eixo vertical e mean_difference no eixo "
            "horizontal; segmentos horizontais = ci_low–ci_high persistidos; "
            "formato dos marcadores = idioma, cor = controle e deslocamento vertical "
            "fixo por idioma e controle em cada k; sem valores-p"
        ),
        cells=cells,
    )
    fig.suptitle(record.title, fontsize=9)
    return _with_files(record, _save_report_figure(fig, name, figures_dir))


_FIGURE_BUILDERS = {
    "performance_by_layer": _figure_performance,
    "fixed_threshold_by_layer": _figure_fixed_threshold,
    "transfer_performance_heatmaps": _figure_transfer_heatmaps,
    "transfer_delta_roc_auc": _figure_transfer_delta_roc_auc,
    "transfer_delta_mcc": _figure_transfer_delta_mcc,
    "dft_relevance_heatmap": _figure_heatmap,
    "class_relevance_by_layer": _figure_class_relevance,
    "decision_reorganization_by_layer": _figure_reorganization,
    "stdft_examples": _figure_stdft,
    "conservation_diagnostics": _figure_conservation,
    "encoder_agreement_heatmaps": _figure_encoder_agreement_heatmaps,
    "encoder_agreement_by_layer": _figure_encoder_agreement_by_layer,
    "language_shift": _figure_language_shift,
    "language_shift_intervals": _figure_language_shift_intervals,
    "diagonal_vs_offdiagonal": _figure_diagonal_vs_offdiagonal,
    "spectral_divergence": _figure_spectral_divergence,
    "spectral_divergence_by_layer": _figure_spectral_divergence_by_layer,
    "layer_faithfulness_intervals": _figure_layer_faithfulness_intervals,
}


def render_report_figures(
    tables: ReportTables, figures_dir: str | Path
) -> tuple[FigureRecord, ...]:
    """Emit deterministic PDF and PNG files for every planned figure family."""
    if not isinstance(tables, ReportTables):
        raise TypeError("tables must be a ReportTables instance")
    import matplotlib as mpl

    from .plotting import set_plot_style

    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    records = []
    with mpl.rc_context():
        set_plot_style()
        for name in tables.planned_figures:
            builder = _FIGURE_BUILDERS.get(name)
            if builder is None:
                raise ValueError(f"unknown planned figure family: {name}")
            records.append(builder(plt, tables, figures_dir))
    return tuple(records)


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------

_LATEX_ESCAPES: Mapping[str, str] = MappingProxyType(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\u2192": r"$\rightarrow$",
        "\u2190": r"$\leftarrow$",
        "\u2194": r"$\leftrightarrow$\allowbreak{}",
        "\u2212": r"$-$",
        "\u2264": r"$\leq$",
        "\u2265": r"$\geq$",
        "\u2248": r"$\approx$",
        "\u2260": r"$\neq$",
        "\u0394": r"$\Delta$",
        "\u00b5": r"$\mu$",
        "\u03bc": r"$\mu$",
        "\u00b1": r"$\pm$",
        "\u00d7": r"$\times$",
        "\u00f7": r"$\div$",
        "\u00b0": r"$^\circ$",
        "\u00b7": r"\textperiodcentered{}",
        "\u2013": "--",
        "\u2014": "---",
        "\u2026": r"\ldots{}",
        "\u201c": "``",
        "\u201d": "''",
        "\u2018": "`",
        "\u2019": "'",
        "\u00a0": "~",
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
)
_MODEL_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "hubert_base": "HuBERT Base",
        "wavlm_base": "WavLM Base",
        "wav2vec2_base": "Wav2Vec2 Base",
    }
)
_LANGUAGE_NAMES: Mapping[str, str] = MappingProxyType(
    {"eng": "inglês", "por": "português", "zho": "mandarim"}
)
_COMPARISON_SECTION_TITLE = "Comparações entre modelos e idiomas"
_SECTION_TITLES_PT: tuple[str, ...] = (
    "Resumo executivo",
    "Como ler este relatório",
    "Resultados",
    "Limitações transversais",
    "Conclusão",
)
_APPENDIX_TITLES_PT: tuple[str, ...] = (
    "Dados e protocolo de avaliação",
    "Desempenho ao longo das camadas",
    "Emergência e reorganização da decisão",
    "Relevância em frequência ao longo das camadas",
    _COMPARISON_SECTION_TITLE,
    "Fidelidade das explicações",
    "Exemplos STDFT e conservação numérica",
)
_NOT_AVAILABLE = "n/d"
_FIT_WIDTH_OPEN = r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%"
_MANIFEST_NAME = "report_manifest.json"
_BUILD_SCRIPT_NAME = "build_local.ps1"
_PDFLATEX = "pdflatex -interaction=nonstopmode -halt-on-error report.tex"
_BUILD_LOCAL_PS1 = f"""$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath "report.pdf") {{
    Remove-Item -LiteralPath "report.pdf" -Force -ErrorAction Stop
}}
if (Test-Path -LiteralPath "report.pdf") {{
    throw "stale report.pdf could not be removed"
}}
{_PDFLATEX}
if ($LASTEXITCODE -ne 0) {{
    throw "pdflatex failed on the first pass (exit code $LASTEXITCODE)"
}}
{_PDFLATEX}
if ($LASTEXITCODE -ne 0) {{
    throw "pdflatex failed on the second pass (exit code $LASTEXITCODE)"
}}
if (-not (Test-Path "report.pdf")) {{
    throw "report.pdf was not produced"
}}
"""
_TABLE_CSVS: tuple[tuple[str, str], ...] = (
    ("performance_by_layer", "performance"),
    ("transfer_selected_layers", "transfer_selection"),
    ("xai_performance_association", "xai_performance_association"),
    ("emergence_layers", "emergence"),
    ("dft_band_relevance", "band_relevance"),
    ("class_relevance", "class_relevance"),
    ("decision_reorganization", "transitions"),
    ("conservation_by_layer", "conservation"),
    ("cohort_by_layer", "cohort"),
    ("stdft_examples", "stdft_examples"),
    ("band_edges", "band_edges"),
    ("band_edge_provenance", "band_edge_provenance"),
)


def _is_pdflatex_safe(char: str) -> bool:
    """Characters that `utf8`/`T1` pdflatex typesets without extra definitions."""
    return " " <= char <= "~" or ("\u00c0" <= char <= "\u00ff" and char.isalpha())


def _escape_char(char: str) -> str:
    mapped = _LATEX_ESCAPES.get(char)
    if mapped is not None:
        return mapped
    if _is_pdflatex_safe(char):
        return char
    return f"[U+{ord(char):04X}]"


def escape_latex(text: object) -> str:
    """Escape LaTeX text in one pass.

    Special characters are escaped, typographic symbols are mapped to safe
    commands, and anything pdflatex cannot typeset (other scripts, emoji,
    control characters) becomes an ASCII `[U+XXXX]` code point.
    """
    return "".join(_escape_char(char) for char in str(text))


def _assert_pdflatex_safe(text: str, name: str) -> None:
    unsafe = sorted(
        {char for char in text if char != "\n" and not _is_pdflatex_safe(char)}
    )
    if unsafe:
        listed = ", ".join(f"U+{ord(char):04X}" for char in unsafe)
        raise ValueError(
            f"{name} contains characters pdflatex cannot typeset: {listed}"
        )


def _tt(value: object) -> str:
    return rf"\texttt{{{escape_latex(value)}}}"


def _tt_breakable(value: object) -> str:
    """Monospace text that may wrap after `_`, `-` and the ellipsis.

    A monospace token has no hyphenation points, so long identifiers would
    otherwise overflow the margin.
    """
    escaped = escape_latex(value)
    for mark in (r"\_", "-", r"\ldots{}"):
        escaped = escaped.replace(mark, mark + r"\allowbreak{}")
    return rf"\texttt{{{escaped}}}"


def _csv_tex(name: str) -> str:
    return _tt_breakable(f"tables/{name}.csv").replace("/", r"/\allowbreak{}")


def _true_minus(text: str) -> str:
    if re.fullmatch(r"-0(\.0*)?", text):
        return text[1:]
    return text.replace("-", "\u2212")


def _num(value: object, digits: int = 3) -> str:
    return _true_minus(f"{float(value):.{digits}f}").replace(".", ",")


def _signed(value: object, digits: int = 3) -> str:
    text = _true_minus(f"{float(value):.{digits}f}")
    return (text if text.startswith("\u2212") else f"+{text}").replace(".", ",")


def _sci(value: object) -> str:
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return f"{_true_minus(mantissa)}e{_true_minus(exponent)}".replace(".", ",")


def _figure_number(value: object, digits: int = 2) -> str:
    return _true_minus(f"{float(value):.{digits}f}")


def _model_name(model: str) -> str:
    return _MODEL_NAMES.get(model, model)


def _language_name(language: str) -> str:
    return _LANGUAGE_NAMES.get(language, language)


def _join_pt(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def _label(*parts: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", "-".join(str(part) for part in parts))


def _cell_tex(cell: tuple[str, str, str]) -> str:
    model, source, target = cell
    return f"{_tt(model)}, {escape_latex(source)}$\\rightarrow${escape_latex(target)}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _write_tex(path: Path, text: str) -> None:
    text = text.replace("\u2212", r"\ensuremath{-}")
    _assert_pdflatex_safe(text, path.name)
    _write_text(path, text)


def _row(columns: Sequence[object]) -> str:
    return " & ".join(str(column) for column in columns) + r" \\"


def _itemize(items: Sequence[str]) -> list[str]:
    """An itemize environment, or nothing when there is nothing to list."""
    if not items:
        return []
    return [r"\begin{itemize}", *[rf"\item {item}" for item in items], r"\end{itemize}", ""]


def _joined_sci(values: Sequence[object]) -> str:
    unique = sorted({float(value) for value in values})
    if len(unique) == 1:
        return _sci(unique[0])
    return f"{_sci(unique[0])} a {_sci(unique[-1])}"


def _is_missing(value: object) -> bool:
    return value is None or bool(pd.isna(value))


def _joined_values(values: Sequence[object]) -> str:
    return _joined(list(values))


def _role_counts(
    sources: Sequence[LoadedReportSource],
) -> dict[tuple[str, str], dict[str, int]]:
    """Declared per-role audio counts by model and language, when recorded."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for item in sources:
        inputs = item.source.plan.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for language in item.source.identity.languages:
            entry = inputs.get(language)
            declared = entry.get("role_counts") if isinstance(entry, Mapping) else None
            if not isinstance(declared, Mapping):
                continue
            values = {role: declared.get(role) for role in ROLE_ORDER}
            if any(type(value) is not int or value < 0 for value in values.values()):
                continue
            counts[(item.source.identity.profile, language)] = values
    return counts


def _report_title(tables: ReportTables) -> str:
    models = [_model_name(model) for model in tables.models]
    languages = [_language_name(language) for language in tables.languages]
    if len(models) == 1 and len(languages) == 1:
        return f"Estudo de caso: {models[0]} em {languages[0]}"
    return (
        f"Relatório layer-wise de XAI: {_join_pt(models)} "
        f"({_join_pt(languages)})"
    )


def _figure_block(record: FigureRecord, tables: ReportTables) -> list[str]:
    pdf = next(name for name in record.files if name.endswith(".pdf"))
    caption = _caption_pt(record, tables)
    return [
        r"\begin{figure}[H]",
        r"\centering",
        rf"\includegraphics[width=\linewidth,height=\figuremaxheight,keepaspectratio]"
        rf"{{figures/{pdf}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{fig:{record.name}}}",
        r"\end{figure}",
        "",
    ]


def _caption_pt(record: FigureRecord, tables: ReportTables) -> str:
    band_range = escape_latex(_band_range_text(tables))
    if record.name == "performance_by_layer":
        body = (
            "ROC-AUC e EER diagnóstico do probe linear em cada camada. "
            "A linha tracejada marca o acaso para a ROC-AUC (0,5)."
        )
    elif record.name == "fixed_threshold_by_layer":
        body = (
            "Acurácia, MCC, TPR e FPR por camada no limiar calibrado na origem. "
            "A classe positiva é sintético; TPR mede sintéticos detectados e FPR, "
            "áudios reais marcados como sintéticos."
        )
    elif record.name == "transfer_performance_heatmaps":
        body = (
            "Desempenho na camada de maior ROC-AUC de cada combinação. "
            "As linhas indicam o idioma de treino e as colunas, o de avaliação; "
            "a parte superior mostra ROC-AUC e a inferior, o MCC da mesma camada."
        )
    elif record.name in ("transfer_delta_roc_auc", "transfer_delta_mcc"):
        metric = "ROC-AUC" if record.name.endswith("roc_auc") else "MCC"
        body = (
            f"Mudança de {metric} na transferência em relação ao sistema diagonal "
            "do mesmo conjunto-alvo. O ponto é a estimativa, o segmento é o "
            f"intervalo bootstrap de {_ci_percent_tex()} e a linha tracejada marca zero."
        )
    elif record.name == "encoder_agreement_heatmaps":
        body = (
            "Concordância entre encoders nos mesmos áudios e condições. "
            "As linhas indicam o idioma de treino e as colunas, o de avaliação; "
            "cada célula mostra a "
            f"mediana entre as {_layer_count(tables.encoder_agreement)} camadas, "
            "em escala própria para cada métrica."
        )
    elif record.name == "language_shift_intervals":
        body = (
            "Mudança da relevância entre conjuntos-alvo com probe fixo, medida "
            "pela distância Jensen--Shannon. As linhas separam os encoders, as "
            "colunas o idioma de treino, e as hastes mostram os intervalos bootstrap."
        )
    elif record.name == "spectral_divergence_by_layer":
        body = (
            "Deslocamento da massa de relevância em frequência entre sistemas "
            "diagonais, medido por Wasserstein em Hz. As colunas separam os pares "
            "de idiomas; a curva percorre as camadas e a faixa mostra o intervalo "
            "bootstrap."
        )
    elif record.name == "layer_faithfulness_intervals":
        body = (
            f"Alteração do escore após remover faixas na camada {FINAL_LAYER}. "
            "As colunas separam os encoders, as linhas as classes, o eixo "
            r"horizontal mostra a diferença média e o vertical, $k$. "
            "O formato do marcador identifica o idioma, as cores distinguem "
            "top-minus-random de top-minus-bottom e o deslocamento vertical separa "
            "os controles."
        )
    elif record.name == "dft_relevance_heatmap":
        body = (
            "Massa média de relevância absoluta normalizada por camada e faixa "
            f"mel ({band_range}). Em cada amostra, a massa distribuída entre as "
            "faixas soma 1."
        )
    elif record.name == "class_relevance_by_layer":
        body = (
            "Massa média de relevância por classe verdadeira e diferença "
            f"sintético $-$ real em cada camada e faixa ({band_range})."
        )
    elif record.name == "decision_reorganization_by_layer":
        body = (
            "Mudança da relevância temporal entre camadas adjacentes. As curvas "
            "mostram similaridade cosseno e variação L1 normalizada, com média e "
            "intervalo normal de 95\\% entre amostras."
        )
    elif record.name == "stdft_examples":
        body = (
            "Mapas STDFT-LRP de uma amostra real e uma sintética na primeira, "
            "na intermediária e na última camada. Todos usam a mesma escala "
            "simétrica de cor."
        )
    elif record.name == "conservation_diagnostics":
        body = _conservation_caption(tables)
    elif record.name == "diagonal_vs_offdiagonal":
        body = (
            r"$\Delta$ ROC-AUC e $\Delta$ MCC por camada, calculados como "
            "transferência menos sistema diagonal no mesmo conjunto-alvo. "
            f"As faixas mostram intervalos bootstrap de {_ci_percent_tex()}."
        )
    elif record.name == "encoder_agreement_by_layer":
        body = (
            "Acordo entre pares de encoders ao longo das camadas. As curvas "
            "mostram acordos de desempenho (Spearman do "
            "escore, fração de decisões iguais, kappa) e de explicação "
            "(cosseno e distância JS); a faixa mostra os quartis entre células."
        )
    elif record.name == "language_shift":
        body = (
            "Distância Jensen--Shannon entre pares de "
            "idiomas-alvo por camada, com probe fixo, para os grupos agregado, "
            "real e sintético."
        )
    elif record.name == "spectral_divergence":
        body = (
            "Distância de Wasserstein em frequência entre sistemas diagonais, "
            "por camada, para os grupos agregado, real e sintético."
        )
    else:
        body = escape_latex(record.title)
    return body


def _conservation_scopes(tables: ReportTables) -> dict[str, str]:
    table = tables.conservation
    cohort = escape_latex(_joined_values(table["n"]))
    subset = escape_latex(_joined_values(table["n_stdft"]))
    kind = _tt_breakable(_joined_values(table["validation_kind"]))
    sample = _tt_breakable(_validation_sample_display(table))
    rtol = _joined_sci(table["score_recompute_rtol"])
    atol = _joined_sci(table["score_recompute_atol"])
    return {
        "bias_zeroed_single_sample": (
            "resíduo relativo bias-zeroed, validação do modelo/regra em uma "
            f"única amostra (tipo {kind}; amostra {sample}; n=1)"
        ),
        "dft_xai_cohort": (
            f"resíduo relativo da DFT, máximo sobre o conjunto fixo de amostras XAI (n={cohort})"
        ),
        "stdft_subset": (
            "resíduo relativo da STDFT, máximo sobre o subconjunto STDFT "
            f"(n={subset})"
        ),
        "score_recompute_rtol_atol": (
            f"recálculo do score, máximo sobre o conjunto fixo de amostras XAI (n={cohort}); "
            f"aceito se {escape_latex('|Δp| ≤ atol + rtol·|p|')} "
            f"(rtol={rtol}, atol={atol})"
        ),
    }


def _conservation_caption(tables: ReportTables) -> str:
    return (
        "Verificações numéricas de conservação por camada, em escala logarítmica: "
        "resíduo bias-zeroed, resíduos da DFT e da STDFT e erro no recálculo do "
        "escore. Cada painel tem seu próprio limite de aceitação."
    )


def _has_comparison(tables: ReportTables, name: str) -> bool:
    return name not in tables.omitted_comparisons and not getattr(tables, name).empty


def _estimate_claim(frame: pd.DataFrame, **filters: object) -> Mapping[str, float | int]:
    """Median and range of usable estimates, the same rows the claim helpers keep."""
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    usable = _usable_claim_rows(selected, ("estimate",))
    estimates = usable["estimate"].to_numpy(dtype=np.float64)
    if not estimates.size:
        return {"n": 0}
    return {
        "n": int(estimates.size),
        "median": float(np.median(estimates)),
        "minimum": float(np.min(estimates)),
        "maximum": float(np.max(estimates)),
    }


def _median_by(
    frame: pd.DataFrame, keys: Sequence[str], **filters: object
) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    usable = _usable_claim_rows(selected, ("estimate",))
    return usable.groupby(list(keys), sort=True)["estimate"].median()


def _delta_claims(tables: ReportTables) -> dict[str, Mapping[str, float | int]]:
    if not _has_comparison(tables, "diagonal_vs_offdiagonal"):
        return {}
    frame = tables.diagonal_vs_offdiagonal
    claims = {
        metric: transfer_effect_claim(frame[frame["metric"] == metric])
        for metric in ("delta_roc_auc", "delta_mcc")
    }
    return {metric: claim for metric, claim in claims.items() if claim["n"]}


_DELTA_LABELS_TEX: Mapping[str, str] = MappingProxyType(
    {"delta_roc_auc": r"$\Delta$ ROC-AUC", "delta_mcc": r"$\Delta$ MCC"}
)


def _executive_comparison_items(tables: ReportTables) -> list[str]:
    """One short headline per comparison present; details live in the blocks."""
    items: list[str] = []
    deltas = _delta_claims(tables)
    if deltas:
        parts = [
            f"{_DELTA_LABELS_TEX[metric]} com {claim['negative_n']} de "
            f"{claim['n']} estimativas negativas (mediana {_signed(claim['median'])})"
            for metric, claim in deltas.items()
        ]
        scope = _pooled_layer_scope(
            tables.diagonal_vs_offdiagonal[
                tables.diagonal_vs_offdiagonal["metric"].isin(tuple(deltas))
            ],
            ("estimate", "ci_low", "ci_high"),
        )
        items.append(
            "Na comparação diagonal e fora da diagonal, no mesmo conjunto-alvo"
            + _scope_clause(scope)
            + ": "
            + _join_pt(parts)
            + r" (Seção~\ref{sec:resultado-transferencia})."
        )
    if _has_comparison(tables, "encoder_agreement"):
        claim = agreement_claim(tables.encoder_agreement)
        if claim["n"]:
            scope = _pooled_layer_scope(
                tables.encoder_agreement, metric="prediction_agreement"
            )
            items.append(
                f"No acordo entre encoders{_scope_clause(scope)}, a fração de "
                f"decisões iguais tem mediana {_num(claim['median'])}, com faixa "
                f"de {_num(claim['minimum'])} a {_num(claim['maximum'])} "
                r"(Seção~\ref{sec:resultado-acordo})."
            )
    if _has_comparison(tables, "language_shift"):
        claim = _estimate_claim(tables.language_shift, group="all")
        if claim["n"]:
            scope = _pooled_layer_scope(tables.language_shift, group="all")
            items.append(
                "Na mudança de relevância entre corpora/idiomas, com probe fixo"
                f"{_scope_clause(scope)}, a distância Jensen--Shannon mediana é "
                f"{_num(claim['median'])} bits "
                r"(Seção~\ref{sec:resultado-relevancia})."
            )
    if _has_comparison(tables, "spectral_divergence"):
        claim = _estimate_claim(tables.spectral_divergence, group="all")
        if claim["n"]:
            scope = _pooled_layer_scope(tables.spectral_divergence, group="all")
            items.append(
                "Na divergência espectral entre sistemas diagonais"
                f"{_scope_clause(scope)}, a distância de Wasserstein mediana é "
                f"{_num(claim['median'])} Hz "
                r"(Seção~\ref{sec:resultado-espectral})."
            )
    return items


def _scope_clause(scope: str) -> str:
    return f", {scope}" if scope else ""


def _executive_summary_section(tables: ReportTables) -> list[str]:
    selected = tables.transfer_selection
    best = selected.loc[selected["roc_auc"].idxmax()]
    off_diagonal = selected[selected["source"] != selected["target"]]
    scope = (
        "um estudo de caso diagonal, sem comparação de transferência"
        if off_diagonal.empty
        else (
            f"{len(off_diagonal)} células fora da diagonal, interpretadas como "
            "transferência sob mudança de corpus/idioma"
        )
    )
    available = int(
        (tables.xai_performance_association["status"] == "available").sum()
    )
    fidelity_stability = _executive_fidelity_stability_blurbs(tables)
    return [
        rf"\section{{{_SECTION_TITLES_PT[0]}}}",
        r"\label{sec:resumo}",
        f"O relatório reúne {len(tables.models)} modelo(s), "
        f"{len(tables.languages)} idioma(s) e {len(selected)} combinações de "
        f"treino e avaliação; contém {scope}.",
        "",
        f"A maior ROC-AUC após a redução por célula é {_num(best['roc_auc'])}, "
        f"na camada {int(best['selected_layer'])} de "
        f"{_cell_tex((best['model'], best['source'], best['target']))}; o MCC "
        f"reportado ({_num(best['mcc'])}) vem dessa mesma camada. A redução "
        "seleciona exclusivamente pela ROC-AUC e nunca maximiza o MCC em separado.",
        "",
        *_itemize(_executive_comparison_items(tables)),
        *fidelity_stability,
        *([""] if fidelity_stability else []),
        f"A associação XAI$\\leftrightarrow$desempenho está disponível em "
        f"{available} de {len(tables.xai_performance_association)} célula(s); "
        f"ela é descritiva, baseada em {len(LAYER_ORDER)} camadas e sem "
        "interpretação causal.",
        "",
    ]


def _shared_and_exceptions(
    values: Mapping[tuple[str, str], object],
) -> tuple[object, list[tuple[str, str]]]:
    """The most frequent value and the model/language pairs that differ from it."""
    tallies: dict[object, int] = {}
    for value in values.values():
        tallies[value] = tallies.get(value, 0) + 1
    shared = max(tallies, key=lambda value: tallies[value])
    return shared, [key for key, value in values.items() if value != shared]


def _role_count_text(declared: Mapping[str, int]) -> str:
    return (
        f"{sum(declared.values())} áudios selecionados no total "
        f"(treino {declared['train']}, calibração {declared['calibration']}, "
        f"teste {declared['test']}); a avaliação por camada usa o subconjunto de "
        f"teste ({declared['test']} áudios)"
    )


def _xai_count_text(n: int, n_real: int, n_synthetic: int) -> str:
    return (
        f"o conjunto fixo de amostras XAI tem {n} áudios ({n_real} reais e "
        f"{n_synthetic} sintéticos)"
    )


def _compact_protocol_counts(
    tables: ReportTables, sources: Sequence[LoadedReportSource]
) -> list[str]:
    """One shared count statement plus the model/language pairs that differ."""
    declared = {
        key: tuple(sorted(value.items()))
        for key, value in _role_counts(sources).items()
    }
    xai = {
        (row.model, row.target): (int(row.n), int(row.n_real), int(row.n_synthetic))
        for row in tables.cohort.drop_duplicates(["model", "target"]).itertuples(
            index=False
        )
    }
    parts: list[str] = []
    exceptions: list[str] = []
    if declared:
        shared, differing = _shared_and_exceptions(declared)
        parts.append(_role_count_text(dict(shared)))
        exceptions += [
            f"{_tt(model)}, {escape_latex(language)}: "
            f"{_role_count_text(dict(declared[(model, language)]))}"
            for model, language in differing
        ]
    elif sources:
        parts.append("as contagens por papel não estão registradas no plano de execução")
    if xai:
        shared, differing = _shared_and_exceptions(xai)
        parts.append(_xai_count_text(*shared))
        exceptions += [
            f"{_tt(model)}, {escape_latex(language)}: "
            f"{_xai_count_text(*xai[(model, language)])}"
            for model, language in differing
        ]
    if not parts:
        return []
    statement = "Em cada par modelo/idioma, " + "; ".join(parts) + "."
    if exceptions:
        return [statement + " Exceções:", "", *_itemize([f"{item}." for item in exceptions])]
    return [statement, ""]


def _ci_percent_tex() -> str:
    return f"{100 * BOOTSTRAP_CI_LEVEL:.0f}\\%"


_COUNT_WORDS_PT = (
    "zero", "um", "dois", "três", "quatro", "cinco",
    "seis", "sete", "oito", "nove", "dez",
)


def _count_word_pt(value: int) -> str:
    return _COUNT_WORDS_PT[value] if 0 <= value < len(_COUNT_WORDS_PT) else str(value)


def _thousands_tex(value: int) -> str:
    """Portuguese round thousands ("2~mil") without a digit-dot-digit group."""
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}~mil"
    return str(value)


def _layer_count(frame: pd.DataFrame) -> int:
    return int(frame["layer"].nunique()) if "layer" in frame else len(LAYER_ORDER)


def _glossary_items() -> list[str]:
    return [
        r"\textbf{Diagonal}: o probe é treinado e avaliado no mesmo idioma "
        "(treino = avaliação).",
        r"\textbf{Transferência}: célula fora da diagonal, em que o probe treinado "
        "em um idioma é avaliado em outro; descreve uma mudança conjunta de "
        "corpus/idioma.",
        r"\textbf{ROC-AUC}: qualidade da ordenação dos escores, sem limiar; 0,5 "
        "corresponde ao acaso e 1 à separação perfeita.",
        r"\textbf{MCC}: correlação de Matthews das decisões no limiar fixo calibrado "
        "no idioma de origem; vai de $-1$ a 1, com 0 próximo do acaso. A classe "
        "positiva é sintético.",
        r"\textbf{Intervalo bootstrap}: a unidade de reamostragem é o áudio; são "
        + _thousands_tex(BOOTSTRAP_RESAMPLES)
        + " reamostragens bootstrap estratificadas por classe, com semente base "
        f"{BOOTSTRAP_SEED} combinada com um hash estável da identidade da "
        f"comparação, e intervalo percentil de {_ci_percent_tex()}. Não são "
        "realizados testes de hipótese e não são produzidos valores-p.",
        r"\textbf{Relevância}: atribuição AttnLRP projetada em bandas mel por "
        "DFT-LRP; a massa absoluta é normalizada por amostra e descreve o probe, "
        "não uma pista acústica verificada.",
    ]


def _orientation_section(
    tables: ReportTables, sources: Sequence[LoadedReportSource]
) -> list[str]:
    n_cells = len(tables.transfer_selection)
    lines = [
        rf"\section{{{_SECTION_TITLES_PT[1]}}}",
        r"\label{sec:orientacao}",
        "O estudo pergunta como a decisão de um probe linear sobre representações "
        "de encoders de fala, que separa áudio real de sintético, se forma ao longo "
        "das camadas e o que as explicações XAI revelam sobre ela quando o idioma "
        f"de treino ou de avaliação muda. O desenho cruza {len(tables.models)} "
        f"encoder(s), {len(tables.languages)} idioma(s) de treino e "
        f"{len(tables.languages)} idioma(s) de avaliação, em {n_cells} "
        "combinações, com uma avaliação por camada.",
        "",
        *_inventory_lines(tables, sources),
        *_compact_protocol_counts(tables, sources),
        "Os mapas usam sempre a mesma ordem de idiomas, com o idioma de treino nas "
        "linhas e o idioma de avaliação nas colunas. Cada bloco de "
        r"Resultados (Seção~\ref{sec:resultados}) traz uma pergunta, a figura, "
        "até três achados, a interpretação, a principal cautela e o caminho até o "
        "detalhe completo nos apêndices e CSVs.",
        "",
        r"\subsection*{Glossário}",
        *_itemize(_glossary_items()),
    ]
    return lines


def _inventory_lines(
    tables: ReportTables, sources: Sequence[LoadedReportSource]
) -> list[str]:
    ordered = sorted(sources, key=_source_sort_key)
    rows = []
    for item in ordered:
        identity = item.source.identity
        layers = sorted({key[1] for key in item.source.layer_xai_generations})
        rows.append(
            _row(
                [
                    _tt(identity.profile),
                    escape_latex(", ".join(identity.languages)),
                    escape_latex(identity.scope),
                    f"{layers[0]}--{layers[-1]}",
                    _tt(str(item.source.status["config_hash"])[:12]),
                ]
            )
        )
    case_study = len(tables.models) == 1 and len(tables.languages) == 1
    lines = [
        "Este relatório é gerado de forma determinística a partir de "
        "artefatos persistidos de execuções concluídas. A geração é somente "
        "leitura: nenhum encoder é carregado, nenhuma explicação é recalculada "
        "e nenhuma GPU é necessária.",
        "",
    ]
    if case_study:
        lines.append(
            f"Esta edição é um estudo de caso de modelo único "
            f"({escape_latex(_model_name(tables.models[0]))}) e idioma único "
            f"({escape_latex(_language_name(tables.languages[0]))}). "
            "Portanto, afirmações comparativas entre modelos ou entre idiomas "
            "não estão disponíveis nesta edição."
        )
    else:
        availability = []
        if len(tables.models) == 1:
            availability.append("comparações entre modelos não estão disponíveis")
        if len(tables.languages) == 1:
            availability.append("comparações entre idiomas não estão disponíveis")
        qualifier = (
            " " + _join_pt(availability).capitalize() + "."
            if availability
            else ""
        )
        lines.append(
            f"Esta edição reúne {len(tables.models)} modelo(s) "
            f"({escape_latex(_join_pt([_model_name(m) for m in tables.models]))}) "
            f"e {len(tables.languages)} idioma(s) "
            f"({escape_latex(_join_pt([_language_name(x) for x in tables.languages]))}), "
            "com sínteses comparativas por célula de treino e avaliação."
            + qualifier
        )
    lines.append("")
    if rows:
        lines += [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Inventário dos experimentos consumidos; o hash da configuração mostra os 12 primeiros caracteres.}",
            r"\label{tab:inventario}",
            _FIT_WIDTH_OPEN,
            r"\begin{tabular}{lllll}",
            r"\toprule",
            _row(
                [
                    "Modelo",
                    "Idiomas",
                    "Escopo",
                    "Camadas",
                    "Hash da config.",
                ]
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    return lines


def _provenance(text: str) -> list[str]:
    return [rf"\textit{{Proveniência:}} {text}", ""]


def _appendix_section(index: int, label: str) -> list[str]:
    return [rf"\section{{{_APPENDIX_TITLES_PT[index]}}}", rf"\label{{{label}}}"]


def _protocol_appendix(
    tables: ReportTables, sources: Sequence[LoadedReportSource]
) -> list[str]:
    counts = _role_counts(sources)
    layers = sorted(set(tables.performance["layer"]))
    band_range = escape_latex(_band_range_text(tables))
    n_bands = escape_latex(_joined_values(tables.band_edge_provenance["n_bands"]))
    items = []
    for model in tables.models:
        for language in tables.languages:
            cohort = tables.cohort[
                (tables.cohort["model"] == model)
                & (tables.cohort["target"] == language)
            ]
            xai = None
            if not cohort.empty:
                row = cohort.iloc[0]
                xai = (
                    f"o conjunto fixo de amostras XAI tem {int(row['n'])} áudios "
                    f"({int(row['n_real'])} reais e {int(row['n_synthetic'])} "
                    "sintéticos)"
                )
            declared = counts.get((model, language))
            if declared is None and xai is None:
                continue
            prefix = f"{_tt(model)}, {escape_latex(language)}: "
            if declared is None:
                items.append(
                    prefix
                    + "contagens por papel não registradas no plano de "
                    f"execução; {xai}."
                )
                continue
            total = sum(declared.values())
            text = (
                f"{total} áudios selecionados no total "
                f"(treino {declared['train']}, calibração "
                f"{declared['calibration']}, teste {declared['test']}). "
                f"A avaliação por camada usa o subconjunto de teste "
                f"({declared['test']} áudios)"
            )
            items.append(prefix + text + (f"; {xai}." if xai else "."))
    provenance = tables.band_edge_provenance
    fallback = provenance[provenance["origin"] != CONFIG_BAND_ORIGIN]
    if fallback.empty:
        edges = (
            "As bordas das bandas foram lidas das configurações registradas "
            "e verificadas por SHA-256."
        )
    else:
        cases = "; ".join(
            f"{_tt(row.model)}, {escape_latex(row.language)} "
            f"({escape_latex(row.reason)})"
            for row in fallback.itertuples(index=False)
        )
        edges = (
            "Onde a configuração registrada não pôde ser verificada, as "
            f"bordas das bandas seguem o contrato mel padrão: {cases}."
        )
    counts_intro = (
        [
            "Contagens de áudios declaradas no plano de execução, separando o "
            "total selecionado do subconjunto de teste e do conjunto de "
            "explicação:",
            "",
        ]
        if items
        else []
    )
    return [
        *_appendix_section(0, "app:protocolo"),
        *_provenance(
            "planos de execução dos experimentos listados em "
            f"{_tt_breakable(_MANIFEST_NAME)}; bordas das bandas em "
            f"{_csv_tex('band_edges')}, com a origem de cada borda em "
            f"{_csv_tex('band_edge_provenance')}; conjunto fixo de amostras XAI "
            f"por camada em {_csv_tex('cohort_by_layer')}."
        ),
        *counts_intro,
        *_itemize(items),
        "Cada camada do encoder (blocos Transformer "
        f"{layers[0]} a {layers[-1]}) recebe um probe linear treinado no "
        "idioma de origem. O limiar de decisão é fixo e calibrado no idioma "
        "de origem com o papel de calibração; a classe positiva é sintético "
        "(rótulo 1). A explicação usa AttnLRP seguida de DFT-LRP em "
        f"{n_bands} bandas mel ({band_range}) e de STDFT-LRP; a relevância "
        "por banda é normalizada por amostra. " + edges,
        "",
    ]


def _performance_appendix(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    names = ("performance_by_layer", "fixed_threshold_by_layer")
    present = [name for name in names if name in figures]
    stability_tex = _probe_stability_tex(tables)
    stability_csv = (
        f"; variabilidade do probe em {_csv_tex('probe_stability_by_layer')}"
        if stability_tex.strip()
        else ""
    )
    lines = [
        *_appendix_section(1, "app:desempenho"),
        *_provenance(
            "métricas por camada (ROC-AUC, EER diagnóstico, acurácia, MCC, TPR "
            f"e FPR) em {_csv_tex('performance_by_layer')}; camada selecionada "
            f"por célula em {_csv_tex('transfer_selected_layers')}"
            + stability_csv
            + "."
        ),
    ]
    for name in present:
        lines += _figure_block(figures[name], tables)
    selected = tables.transfer_selection
    off_diagonal = selected[selected["source"] != selected["target"]]
    selected_metrics = tables.performance.merge(
        selected[["model", "source", "target", "selected_layer"]],
        on=["model", "source", "target"],
        how="inner",
        validate="many_to_one",
    )
    selected_metrics = selected_metrics[
        selected_metrics["layer"] == selected_metrics["selected_layer"]
    ]
    off_metrics = selected_metrics[
        selected_metrics["source"] != selected_metrics["target"]
    ]
    lines += [
        r"\subsection{Ranking sem limiar e calibração do limiar}",
        "ROC-AUC mede a ordenação dos escores sem escolher um limiar; MCC, TPR e FPR "
        "medem o comportamento no limiar calibrado na origem. Por isso, uma "
        "ROC-AUC preservada não implica MCC, TPR ou FPR preservados.",
        "",
    ]
    if off_diagonal.empty:
        lines += [
            "Há somente avaliação diagonal neste relatório; não há célula fora da "
            "diagonal para descrever transferência.",
            "",
        ]
    else:
        lines += [
            "As células fora da diagonal são descritas como transferência sob "
            "mudança de corpus/idioma, sem atribuir efeito causal ao idioma. "
            f"Nelas, a ROC-AUC selecionada varia de {_num(off_diagonal['roc_auc'].min())} "
            f"a {_num(off_diagonal['roc_auc'].max())}, enquanto o MCC da mesma "
            f"camada varia de {_num(off_diagonal['mcc'].min())} a "
            f"{_num(off_diagonal['mcc'].max())}, a TPR de "
            f"{_num(off_metrics['tpr'].min())} a {_num(off_metrics['tpr'].max())} "
            f"e a FPR de {_num(off_metrics['fpr'].min())} a "
            f"{_num(off_metrics['fpr'].max())}.",
            "",
        ]
    lines += [r"\input{tables/performance_by_layer.tex}", ""]
    if stability_tex.strip():
        lines += [
            r"\subsection{Variabilidade por reamostragem estratificada do treino do probe}",
            "Esta subseção resume re-treinos do probe linear após bootstrap "
            "estratificado com reposição do conjunto de treino de cada idioma-fonte "
            "(contagens por classe preservadas), com calibração e teste fixos, sobre "
            "os mesmos embeddings materializados. Trata-se de análise de sensibilidade "
            "à amostra finita de treino; "
            r"\textbf{não} mede estabilidade de atribuições XAI (AttnLRP/DFT-LRP), "
            "nem estabilidade ponta-a-ponta de sementes do pipeline completo, "
            "nem recomputa explicações. "
            "São três reamostragens por célula: indicam ordens de grandeza da "
            "variabilidade, não um intervalo de confiança estreito.",
            "",
            r"\input{tables/probe_stability_by_layer.tex}",
            "",
            _probe_stability_pattern_paragraph(
                probe_stability_aggregate_by_model(_aligned_probe_stability(tables))
            ),
            "",
        ]
    return lines


def _emergence_appendix(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    lines = [
        *_appendix_section(2, "app:emergencia"),
        *_provenance(
            "início (onset) e consolidação persistidos pelo agregador do "
            f"experimento, sem recálculo, em {_csv_tex('emergence_layers')}; "
            "similaridade e variação L1 entre camadas adjacentes, com o número "
            f"de amostras, em {_csv_tex('decision_reorganization')}."
        ),
    ]
    if tables.emergence.empty:
        lines += ["Não há estimativa de emergência nesta edição.", ""]
    else:
        total = len(tables.emergence)
        onset = _identified_layers_text(
            tables.emergence["onset"],
            total,
            found="o início estimado da emergência foi identificado",
            missing="o início da emergência não foi identificado",
        )
        consolidation = _identified_layers_text(
            tables.emergence["consolidation"],
            total,
            found="a consolidação foi identificada",
            missing="a consolidação não foi identificada",
        )
        lines += [
            f"Em resumo, {onset}; {consolidation}. Os valores por célula, com a "
            r"ROC-AUC final, estão na Tabela~\ref{tab:emergence}.",
            "",
            r"\input{tables/emergence_layers.tex}",
            "",
        ]
    lines += [r"\subsection{Reorganização da decisão entre camadas}"]
    if "decision_reorganization_by_layer" not in figures:
        return [*lines, "A figura de reorganização não foi gerada nesta edição.", ""]
    lows = pd.DataFrame(
        [
            _cell_rows(tables.transitions, cell)
            .sort_values("similarity_mean", kind="stable")
            .iloc[0]
            for cell in _cells(tables)
        ]
    )
    transitions = lows.groupby(
        ["previous_layer", "current_layer"], sort=True
    ).size()
    (previous, current), count = max(
        transitions.items(), key=lambda item: (item[1], -item[0][0])
    )
    low, high = _num(lows["similarity_mean"].min()), _num(lows["similarity_mean"].max())
    span = f"é {low}" if low == high else f"vai de {low} a {high}"
    lines += [
        f"Nas {len(lows)} célula(s), a menor similaridade cosseno média entre "
        f"camadas adjacentes {span}; a transição "
        f"{int(previous)}$\\rightarrow${int(current)} concentra esse mínimo em "
        f"{int(count)} delas.",
        "",
        *_figure_block(figures["decision_reorganization_by_layer"], tables),
    ]
    return lines


def _identified_layers_text(
    values: pd.Series, total: int, *, found: str, missing: str
) -> str:
    layers = [int(value) for value in values if not _is_missing(value)]
    if not layers:
        return missing if total == 1 else f"{missing} em nenhuma das {total} células"
    span = (
        f"na camada {layers[0]}"
        if min(layers) == max(layers)
        else f"entre as camadas {min(layers)} e {max(layers)}"
    )
    text = f"{found} em {len(layers)} de {total} célula(s), {span}"
    if len(layers) < total:
        text += f", e {missing} nas outras {total - len(layers)}"
    return text


def _relevance_appendix(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    names = ("class_relevance_by_layer",)
    lines = [
        *_appendix_section(3, "app:relevancia"),
        *_provenance(
            "relevância absoluta normalizada por célula, camada e banda mel, "
            f"no conjunto fixo de amostras XAI, em {_csv_tex('dft_band_relevance')} "
            f"e, por classe verdadeira, em {_csv_tex('class_relevance')}; "
            "associação entre concentração espectral e ROC-AUC em "
            f"{_csv_tex('xai_performance_association')}."
        ),
    ]
    for name in names:
        if name in figures:
            lines += _figure_block(figures[name], tables)
    lines += [
        r"\subsection{Associação descritiva entre XAI e desempenho}",
        "A concentração espectral de cada camada é definida como a maior massa "
        "média entre as bandas de relevância absoluta normalizada. Para cada "
        "modelo e célula treino$\\rightarrow$avaliação, calcula-se a correlação "
        "de postos de Spearman entre essa concentração e a ROC-AUC nas "
        f"{len(LAYER_ORDER)} camadas.",
        "",
    ]
    association = tables.xai_performance_association
    available = association[association["status"] == "available"]
    summary = (
        f"A associação está disponível em {len(available)} de {len(association)} "
        "célula(s)"
    )
    if not available.empty:
        rho = available["rho"].astype(float)
        summary += (
            f"; nelas, $\\rho$ tem mediana {_signed(rho.median())} e vai de "
            f"{_signed(rho.min())} a {_signed(rho.max())}"
        )
    lines += [
        summary + "; o CSV registra o motivo de cada indisponibilidade. "
        f"Esta análise é descritiva, usa apenas n={len(LAYER_ORDER)} camadas por "
        "célula quando disponível e não sustenta inferência nem interpretação "
        "causal.",
        "",
    ]
    return lines


def _comparisons_appendix(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    present = [name for name in COMPARISON_FAMILIES if _has_comparison(tables, name)]
    lines = _appendix_section(4, "app:comparacoes")
    if not present:
        return [
            *lines,
            *_provenance(
                "nesta edição não há comparação entre modelos ou idiomas; "
                "nenhum CSV de comparação foi gerado."
            ),
        ]
    lines += [
        *_provenance(
            "limites \\texttt{ci\\_low}--\\texttt{ci\\_high} de cada grupo, camada "
            "e célula em "
            + _join_pt([_csv_tex(name) for name in present])
            + r"; o contrato do intervalo bootstrap está no glossário "
            r"(Seção~\ref{sec:orientacao})."
        ),
        "As tabelas agregam medianas e faixas entre linhas; os valores por "
        "linha ficam nos CSV.",
        "",
    ]
    for name in present:
        lines += [rf"\input{{tables/{name}.tex}}", ""]
    if "diagonal_vs_offdiagonal" in figures:
        lines += [
            r"\subsection{Diagonal versus fora da diagonal por camada}",
            *_figure_block(figures["diagonal_vs_offdiagonal"], tables),
        ]
    if "encoder_agreement_by_layer" in figures:
        lines += [
            r"\subsection{Acordo entre encoders por camada}",
            *_figure_block(figures["encoder_agreement_by_layer"], tables),
        ]
    profiles = [
        name for name in ("language_shift", "spectral_divergence") if name in figures
    ]
    if profiles:
        lines += [
            r"\subsection{Perfis técnicos por grupo de classe}",
            r"\label{app:perfis-grupo-classe}",
            "Séries agregada, real e sintética da mudança de relevância e da "
            "divergência espectral; complementam as figuras principais.",
            "",
        ]
        for name in profiles:
            lines += _figure_block(figures[name], tables)
    return lines


def _faithfulness_appendix(tables: ReportTables) -> list[str]:
    lines = _appendix_section(5, "app:fidelidade")
    aligned = _aligned_layer_faithfulness(tables)
    if not _layer_faithfulness_tex(tables).strip():
        return [
            *lines,
            *_provenance(
                "nesta edição não há artefatos de intervenção sobre as faixas "
                "relevantes; nenhum CSV de fidelidade foi gerado."
            ),
        ]
    return [
        *lines,
        *_provenance(
            "diferenças pareadas por modelo, célula diagonal, classe e $k$, com "
            "intervalos bootstrap e número de pares, em "
            f"{_csv_tex('layer_faithfulness_summary')}."
        ),
        rf"\subsection{{Fidelidade por intervenção (camada {FINAL_LAYER}, diagonal)}}",
        "Comparações pareadas de comprehensividade (delete) entre faixas "
        r"top-$k$ (mais relevantes), bottom-$k$ (menos relevantes) e "
        "aleatórias com energia RMS igualada, com recomputação AttnLRP "
        r"$\rightarrow$ STDFT sobre o subconjunto fixo de amostras XAI. "
        "Valores positivos de top-minus-random ou top-minus-bottom "
        "sugerem, com cautela, que remover as faixas mais relevantes segundo o "
        "DFT-LRP altera mais o escore do probe do que remover os controles; a "
        "filtragem STFT e o reescalonamento RMS são fontes de variação "
        "introduzidas pelo procedimento e não autorizam alegação causal no "
        f"mundo real. Escopo restrito à camada~{FINAL_LAYER} com origem=destino; "
        "não extrapolar para outras camadas ou células fora da diagonal.",
        "",
        _faithfulness_scope_note(aligned),
        "",
        r"\input{tables/layer_faithfulness_summary.tex}",
        "",
        _faithfulness_pattern_paragraph(faithfulness_aggregate_counts(aligned)),
        "",
    ]


def _abbreviation_note(tables: ReportTables) -> list[str]:
    """Explain shortened sample IDs, only when some ID was actually shortened."""
    ids = [
        *map(str, tables.stdft_examples["sample_id"]),
        *map(str, tables.conservation["validation_sample_id"]),
    ]
    if all(len(item) <= SAMPLE_ID_DISPLAY_LENGTH for item in ids):
        return []
    return [
        "Os identificadores de amostra longos aparecem abreviados (início, "
        "reticências e fim); os identificadores completos estão nos CSV.",
        "",
    ]


def _stdft_conservation_appendix(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    lines = [
        *_appendix_section(6, "app:stdft-conservacao"),
        *_provenance(
            f"exemplos STDFT por célula, camada e classe em {_csv_tex('stdft_examples')}; "
            "verificações de conservação por célula e camada, com tolerâncias e "
            f"a amostra de validação, em {_csv_tex('conservation_by_layer')}."
        ),
        *_abbreviation_note(tables),
        r"\subsection{Exemplos tempo-frequência STDFT}",
    ]
    if "stdft_examples" in figures:
        examples = tables.stdft_examples
        n_cells = len(examples[["model", "source", "target"]].drop_duplicates())
        shown = figures["stdft_examples"].cells
        lines += [
            f"O conjunto contém {len(examples)} exemplos STDFT em {n_cells} "
            f"célula(s) e {examples['layer'].nunique()} camadas "
            f"({int((examples['y_true'] == 0).sum())} reais e "
            f"{int((examples['y_true'] == 1).sum())} sintéticos no total). "
            "A figura mostra amostras de "
            + _join_pt([_cell_tex(c) for c in shown])
            + ".",
            "",
            *_figure_block(figures["stdft_examples"], tables),
        ]
    else:
        lines += ["A figura de exemplos não foi gerada nesta edição.", ""]
    lines += [r"\subsection{Conservação e qualidade numérica}"]
    if "conservation_diagnostics" in figures:
        series = build_conservation_series(tables)
        scopes = _conservation_scopes(tables)
        tolerance = _sci(tables.conservation["tolerance"].min())
        items = []
        for panel in dict.fromkeys(series["panel"]):
            rows = series[series["panel"] == panel]
            worst = float(rows["value"].max())
            limit = float(rows["limit"].iloc[0])
            status = "dentro do limite" if worst <= limit else "acima do limite"
            scope = scopes[panel]
            items.append(
                f"{scope[0].upper()}{scope[1:]}: máximo {_sci(worst)} sobre as "
                f"camadas, limite {_sci(limit)} ({status})."
            )
        lines += [
            "As verificações cobrem populações diferentes e obedecem a limites "
            f"próprios: três usam a tolerância de conservação ({tolerance}) e o "
            "recálculo do score usa a razão rtol/atol, com limite 1.",
            "",
            *_itemize(items),
            *_figure_block(figures["conservation_diagnostics"], tables),
        ]
    if _conservation_tex(tables):
        lines += [r"\input{tables/conservation_by_layer.tex}", ""]
    return lines


def _appendices(
    tables: ReportTables,
    figures: Mapping[str, FigureRecord],
    sources: Sequence[LoadedReportSource],
) -> list[str]:
    """Appendices A–G: complete tables, detail figures and CSV provenance."""
    return [
        r"\clearpage",
        r"\appendix",
        r"\label{appendix:start}",
        "",
        *_protocol_appendix(tables, sources),
        *_performance_appendix(tables, figures),
        *_emergence_appendix(tables, figures),
        *_relevance_appendix(tables, figures),
        *_comparisons_appendix(tables, figures),
        *_faithfulness_appendix(tables),
        *_stdft_conservation_appendix(tables, figures),
    ]


def _result_block(
    tables: ReportTables,
    figures: Mapping[str, FigureRecord],
    *,
    key: str,
    heading: str,
    question: str,
    figure_names: Sequence[str],
    findings: Sequence[str],
    interpretation: str,
    caution: str,
    details: str,
    evidence_blocks: Sequence[str] = (),
) -> list[str]:
    if not 1 <= len(findings) <= 3:
        raise ValueError(f"result block {key} needs one to three findings")
    shown = [name for name in figure_names if name in figures]
    references = _join_pt([rf"Figura~\ref{{fig:{name}}}" for name in shown])
    lines = [
        rf"\subsection{{{heading}}}",
        rf"\label{{sec:resultado-{key}}}",
        rf"\textbf{{Pergunta:}} {question}"
        + (f" ({references})" if references else ""),
        "",
    ]
    for name in shown:
        lines += _figure_block(figures[name], tables)
    for evidence in evidence_blocks:
        if evidence:
            lines += [evidence, ""]
    return [
        *lines,
        r"\textbf{Resultado:}",
        *_itemize(findings),
        rf"\textbf{{Interpretação:}} {interpretation}",
        "",
        rf"\textbf{{Cautela:}} {caution}",
        "",
        rf"\textit{{Detalhe completo:}} {details}",
        "",
    ]


def _range_text(claim: Mapping[str, float | int], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"mediana {_num(claim['median'])}{suffix} em {claim['n']} estimativas, "
        f"com faixa de {_num(claim['minimum'])} a {_num(claim['maximum'])}{suffix}"
    )


def _extremes_text(medians: pd.Series, label, unit: str = "") -> str | None:
    """Range of group medians as a predicate, or None for a single group."""
    if len(medians) < 2:
        return None
    suffix = f" {unit}" if unit else ""
    low, high = medians.idxmin(), medians.idxmax()
    if _num(medians[low]) == _num(medians[high]):
        return f"é {_num(medians[low])}{suffix} em todos"
    return (
        f"vai de {_num(medians[low])}{suffix} ({escape_latex(label(low))}) a "
        f"{_num(medians[high])}{suffix} ({escape_latex(label(high))})"
    )


def _layer_trend(medians: pd.Series, unit: str) -> tuple[str | None, str]:
    """Median at the first and last layer, and the observed direction."""
    if len(medians) < 2:
        return None, ""
    first, last = medians.index.min(), medians.index.max()
    a, b = float(medians[first]), float(medians[last])
    finding = (
        f"Na camada {int(first)}, a mediana é {_num(a)} {unit}; na camada "
        f"{int(last)}, {_num(b)} {unit}."
    )
    relation = "maior" if b > a else "menor" if b < a else "igual"
    direction = (
        "Comparando apenas a primeira e a última camada, a mediana na última é "
        f"{relation} {'à' if relation == 'igual' else 'que a'} da primeira; a "
        "comparação não descreve as camadas intermediárias nem implica variação "
        "monotônica."
    )
    return finding, direction


def _selection_text(label: str, frame: pd.DataFrame) -> str:
    if len(frame) == 1:
        row = frame.iloc[0]
        return (
            f"{label}, a ROC-AUC selecionada é {_num(row['roc_auc'])} e o MCC da "
            f"mesma camada é {_num(row['mcc'])}."
        )
    return (
        f"{label} ({len(frame)}), a ROC-AUC selecionada tem mediana "
        f"{_num(frame['roc_auc'].median())} (de {_num(frame['roc_auc'].min())} a "
        f"{_num(frame['roc_auc'].max())}) e o MCC da mesma camada tem mediana "
        f"{_num(frame['mcc'].median())} (de {_num(frame['mcc'].min())} a "
        f"{_num(frame['mcc'].max())})."
    )


def _overview_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    selected = tables.transfer_selection
    diagonal = selected[selected["source"] == selected["target"]]
    off_diagonal = selected[selected["source"] != selected["target"]]
    best = selected.loc[selected["roc_auc"].idxmax()]
    findings = []
    if len(diagonal) > 1:
        findings.append(_selection_text("Nas células diagonais", diagonal))
    if not off_diagonal.empty:
        findings.append(_selection_text("Nas células de transferência", off_diagonal))
    findings.append(
        f"A maior ROC-AUC selecionada é {_num(best['roc_auc'])}, na camada "
        f"{int(best['selected_layer'])} de "
        f"{_cell_tex((best['model'], best['source'], best['target']))}; o MCC "
        f"dessa camada é {_num(best['mcc'])}."
    )
    interpretation = (
        "A ROC-AUC descreve a ordenação dos escores sem limiar, e o MCC descreve "
        "as decisões no limiar calibrado na origem; lidas na mesma camada, as duas "
        "medidas separam perda de ordenação de perda de calibração."
    )
    if not off_diagonal.empty and not diagonal.empty:
        gap = float(off_diagonal["roc_auc"].median() - diagonal["roc_auc"].median())
        position = "abaixo" if gap < 0 else "acima" if gap > 0 else "no mesmo nível"
        interpretation += (
            " A mediana da ROC-AUC de transferência fica "
            + (f"{_num(abs(gap))} {position} da" if gap else f"{position} da")
            + " mediana diagonal; o bloco seguinte compara cada transferência com "
            "o sistema diagonal do mesmo conjunto-alvo."
        )
    elif off_diagonal.empty:
        interpretation += (
            " Com apenas a diagonal, o mapa descreve a separação dentro de cada "
            "idioma, sem informação sobre transferência."
        )
    return _result_block(
        tables,
        figures,
        key="desempenho",
        heading="Visão geral do desempenho",
        question=(
            "quão bem o probe linear separa áudio real de sintético em cada "
            "combinação de idioma de treino e de avaliação?"
        ),
        figure_names=("transfer_performance_heatmaps",),
        findings=findings,
        interpretation=interpretation,
        caution=(
            "a camada de cada célula é escolhida pela maior ROC-AUC no próprio "
            "conjunto de teste, o que torna o valor exibido otimista; o MCC vem "
            "dessa camada e não é maximizado separadamente."
        ),
        details=(
            rf"Apêndice~\ref{{app:desempenho}}; {_csv_tex('transfer_selected_layers')} "
            f"e {_csv_tex('performance_by_layer')}."
        ),
    )


def _transfer_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    deltas = _delta_claims(tables)
    if not deltas:
        return []
    findings = [
        f"{claim['negative_n']} de {claim['n']} estimativas de "
        f"{_DELTA_LABELS_TEX[metric]} são negativas; {claim['ci_below_zero_n']} "
        f"de {claim['n']} intervalos ficam inteiramente abaixo de zero e "
        f"{claim['ci_above_zero_n']} de {claim['n']} intervalos ficam "
        f"inteiramente acima de zero (mediana {_signed(claim['median'])}, todas "
        "as camadas)."
        for metric, claim in deltas.items()
    ]
    selected = {
        metric: transfer_effect_claim(selected_transfer_effects(tables, metric))
        for metric in deltas
    }
    selected_parts = [
        f"{claim['negative_n']} de {claim['n']} {_DELTA_LABELS_TEX[metric]}"
        for metric, claim in selected.items()
        if claim["n"]
    ]
    if selected_parts:
        findings.append(
            "Na camada selecionada de cada célula, mostrada nas figuras, são "
            f"negativos {_join_pt(selected_parts)}."
        )
    interpretation = (
        r"Um $\Delta$ negativo indica desempenho menor fora da diagonal do que no "
        "sistema diagonal avaliado no mesmo conjunto-alvo."
    )
    shares = {
        metric: claim["negative_n"] / claim["n"] for metric, claim in deltas.items()
    }
    if max(shares.values()) <= 0.5:
        interpretation += (
            " Nenhuma das medidas tem maioria de estimativas negativas; neste "
            "conjunto, a transferência não se traduz em queda predominante."
        )
    elif len(shares) == 2:
        auc_share, mcc_share = shares["delta_roc_auc"], shares["delta_mcc"]
        if mcc_share > auc_share:
            interpretation += (
                " Quedas de MCC são proporcionalmente mais frequentes que quedas "
                "de ROC-AUC, o que sugere que parte da perda está no limiar "
                "calibrado na origem, e não apenas na ordenação dos escores."
            )
        elif mcc_share < auc_share:
            interpretation += (
                " Quedas de ROC-AUC são proporcionalmente mais frequentes que "
                "quedas de MCC, o que situa a perda já na ordenação dos escores."
            )
        else:
            interpretation += (
                " Quedas de ROC-AUC e de MCC têm a mesma frequência relativa."
            )
    return _result_block(
        tables,
        figures,
        key="transferencia",
        heading="Quanto o desempenho cai na transferência?",
        question=(
            "quanto muda o desempenho quando o probe treinado em um idioma é "
            "avaliado em outro, em relação ao sistema diagonal do mesmo "
            "conjunto-alvo?"
        ),
        figure_names=("transfer_delta_roc_auc", "transfer_delta_mcc"),
        findings=findings,
        interpretation=interpretation,
        caution=(
            r"cada $\Delta$ troca ao mesmo tempo o conjunto de treino do probe e o "
            "par corpus/idioma; o sinal descreve a direção da mudança observada, "
            "não sua causa. Os intervalos reamostram áudios do conjunto-alvo e não "
            "incluem a variação de novos treinos do probe."
        ),
        details=(
            r"Apêndice~\ref{app:comparacoes} (Tabela~\ref{tab:diagonal-off-summary}); "
            f"{_csv_tex('diagonal_vs_offdiagonal')}."
        ),
    )


def _agreement_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    if not _has_comparison(tables, "encoder_agreement"):
        return []
    frame = tables.encoder_agreement
    claim = agreement_claim(frame)
    if not claim["n"]:
        return []
    findings = [f"A fração de decisões iguais tem {_range_text(claim)}."]
    kappa = _estimate_claim(frame, metric="cohen_kappa")
    if kappa["n"]:
        findings.append(
            "O kappa de Cohen, que desconta o acordo esperado ao acaso, tem "
            f"{_range_text(kappa)}."
        )
    pairs = _extremes_text(
        _median_by(frame, ["model_a", "model_b"], metric="prediction_agreement"),
        lambda pair: _pair_pretty(*pair),
    )
    if pairs:
        findings.append(
            f"Por par de modelos, a mediana da fração de decisões iguais {pairs}."
        )
    by_layer = (
        r"; o perfil por camada está na Figura~\ref{fig:encoder_agreement_by_layer}"
        if "encoder_agreement_by_layer" in figures
        else ""
    )
    return _result_block(
        tables,
        figures,
        key="acordo",
        heading="Até que ponto os encoders concordam?",
        question=(
            "encoders diferentes, avaliados nos mesmos áudios e condições, tomam "
            "as mesmas decisões e apontam relevâncias parecidas?"
        ),
        figure_names=("encoder_agreement_heatmaps",),
        findings=findings,
        interpretation=(
            "A faixa mostra quanto o acordo varia entre células e camadas; a "
            "mediana, sozinha, esconde essa heterogeneidade. Quando dois modelos "
            "acertam a maior parte dos áudios, a fração de decisões iguais sobe "
            "mesmo sem estratégia comum, por isso o kappa complementa a leitura."
        ),
        caution=(
            "acordo alto não torna os modelos intercambiáveis e acordo baixo não "
            "aponta qual modelo está correto; cada métrica tem escala própria e não "
            "deve ser comparada com as demais."
        ),
        details=(
            r"Apêndice~\ref{app:comparacoes} (Tabela~\ref{tab:encoder-agreement-summary}"
            + by_layer
            + f"); {_csv_tex('encoder_agreement')}."
        ),
    )


def _profiles_detail(figures: Mapping[str, FigureRecord], name: str) -> str:
    return (
        r"séries real e sintética no Apêndice~\ref{app:perfis-grupo-classe}; "
        if name in figures
        else ""
    )


def frequency_profile_claims(tables: ReportTables) -> pd.DataFrame:
    """Return the recurring dominant band for each diagonal model/language cell."""
    columns = [
        "model",
        "language",
        "band",
        "band_label",
        "mean_mass",
        "layers_dominant",
        "n_layers",
    ]
    frame = tables.band_relevance
    selected = frame[
        (frame["source"] == frame["target"])
        & (frame["measure"] == "absolute_normalized")
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=columns)

    selected["language"] = selected["target"]
    identity = ["model", "language", "layer"]
    per_layer = (
        selected.sort_values(
            identity + ["mean", "band"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(identity)
        .loc[:, identity + ["band", "band_label"]]
    )
    dominance = (
        per_layer.groupby(
            ["model", "language", "band", "band_label"], as_index=False, sort=True
        )
        .size()
        .rename(columns={"size": "layers_dominant"})
    )
    mean_mass = (
        selected.groupby(["model", "language", "band"], as_index=False, sort=True)[
            "mean"
        ]
        .mean()
        .rename(columns={"mean": "mean_mass"})
    )
    layer_counts = (
        selected.groupby(["model", "language"], as_index=False, sort=True)["layer"]
        .nunique()
        .rename(columns={"layer": "n_layers"})
    )
    claims = dominance.merge(
        mean_mass, on=["model", "language", "band"], validate="one_to_one"
    ).merge(layer_counts, on=["model", "language"], validate="many_to_one")
    maximum = claims.groupby(["model", "language"])["layers_dominant"].transform(
        "max"
    )
    claims = claims[claims["layers_dominant"] == maximum].copy()
    model_order = {model: index for index, model in enumerate(tables.models)}
    language_order = {
        language: index for index, language in enumerate(tables.languages)
    }
    claims["_model_order"] = claims["model"].map(model_order)
    claims["_language_order"] = claims["language"].map(language_order)
    return (
        claims.sort_values(
            ["_model_order", "_language_order", "band"], kind="mergesort"
        )
        .drop(columns=["_model_order", "_language_order"])
        .reset_index(drop=True)
        .loc[:, columns]
    )


def _frequency_profile_table(claims: pd.DataFrame) -> str:
    body = []
    for (model, language), rows in claims.groupby(
        ["model", "language"], sort=False
    ):
        labels = [
            escape_latex(str(label)).replace("–", "--")
            for label in rows["band_label"]
        ]
        body.append(
            [
                escape_latex(_model_name(str(model))),
                escape_latex(_language_name(str(language))),
                _join_pt(labels),
                (
                    f"{int(rows['layers_dominant'].iloc[0])} de "
                    f"{int(rows['n_layers'].iloc[0])}"
                ),
            ]
        )
    caption = (
        "Faixas com maior massa média de relevância no maior número de camadas "
        "da Figura~\\ref{fig:dft_relevance_heatmap}. Empates aparecem como "
        "faixas co-dominantes."
    )
    return _comparison_compact_table(
        caption,
        "tab:frequency-profile-summary",
        ["Modelo", "Idioma", "Faixa(s) dominante(s)", "Camadas"],
        body,
    )


def _frequency_profile_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    claims = frequency_profile_claims(tables)
    if claims.empty or "dft_relevance_heatmap" not in figures:
        return []
    model_findings = []
    for model in tables.models:
        rows = claims[claims["model"] == model]
        if rows.empty:
            continue
        parts = []
        for language in tables.languages:
            language_rows = rows[rows["language"] == language]
            if language_rows.empty:
                continue
            labels = [
                escape_latex(str(label)).replace("–", "--")
                for label in language_rows["band_label"]
            ]
            layers_dominant = int(language_rows["layers_dominant"].iloc[0])
            n_layers = int(language_rows["n_layers"].iloc[0])
            parts.append(
                f"{escape_latex(_language_name(language))}: {_join_pt(labels)} em "
                f"{layers_dominant} de {n_layers} camadas"
            )
        model_findings.append(
            f"{escape_latex(_model_name(model))}: {_join_pt(parts)}"
        )
    findings = (
        [f"{finding}." for finding in model_findings]
        if len(model_findings) <= 3
        else [_join_pt(model_findings) + "."]
    )
    return _result_block(
        tables,
        figures,
        key="faixas-frequencia",
        heading="Em quais faixas a relevância se concentra para cada encoder e idioma?",
        question=(
            "em quais faixas de frequência a relevância do probe se concentra ao "
            "longo das camadas, para cada encoder e idioma?"
        ),
        figure_names=("dft_relevance_heatmap",),
        findings=findings,
        interpretation=(
            "A faixa mais frequentemente dominante é aquela com a maior massa "
            "média de relevância absoluta normalizada no maior número de camadas. "
            "Em caso de empate, todas as faixas co-dominantes são mantidas. Os "
            "itens resumem essa recorrência; o mapa preserva o perfil completo das "
            "24 bandas em cada uma das 12 camadas, inclusive concentrações "
            "secundárias e mudanças entre camadas."
        ),
        caution=(
            "a dominância é relativa à massa de relevância do probe no conjunto "
            "fixo de amostras XAI, não uma pista acústica causal. Diferenças entre "
            "os painéis refletem conjuntamente corpus, idioma, modelo e probe."
        ),
        details=(
            r"contraste entre áudio real e sintético no "
            r"Apêndice~\ref{app:relevancia} "
            r"(Figura~\ref{fig:class_relevance_by_layer}); "
            f"{_csv_tex('dft_band_relevance')}."
        ),
        evidence_blocks=(_frequency_profile_table(claims),),
    )


def _relevance_shift_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    if not _has_comparison(tables, "language_shift"):
        return []
    frame = tables.language_shift
    claim = _estimate_claim(frame, group="all")
    if not claim["n"]:
        return []
    findings = [
        "Com o probe fixo, a distância Jensen--Shannon do grupo agregado tem "
        f"{_range_text(claim, 'bits')}."
    ]
    models = _extremes_text(
        _median_by(frame, ["model"], group="all"), _model_name, "bits"
    )
    if models:
        findings.append(f"Por modelo, a mediana {models}.")
    trend, direction = _layer_trend(_median_by(frame, ["layer"], group="all"), "bits")
    if trend:
        findings.append(trend)
    return _result_block(
        tables,
        figures,
        key="relevancia",
        heading="Como a relevância muda entre corpora e idiomas?",
        question=(
            "com modelo, idioma de treino e probe fixos, quanto muda o perfil de "
            "relevância ao trocar o conjunto de avaliação?"
        ),
        figure_names=("language_shift_intervals",),
        findings=findings,
        interpretation=(
            "Distância maior indica perfis médios de relevância menos parecidos "
            "entre os conjuntos-alvo. Como o probe é o mesmo, a diferença vem dos "
            "áudios avaliados, e não de um novo treino. " + direction
        ).strip(),
        caution=(
            "corpus, áudio e idioma mudam juntos; a distância não isola um efeito "
            "do idioma e não mede desempenho."
        ),
        details=(
            _profiles_detail(figures, "language_shift")
            + r"Apêndice~\ref{app:comparacoes} (Tabela~\ref{tab:language-shift-summary}); "
            f"{_csv_tex('language_shift')}."
        ),
    )


def _spectral_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    if not _has_comparison(tables, "spectral_divergence"):
        return []
    frame = tables.spectral_divergence
    claim = _estimate_claim(frame, group="all")
    if not claim["n"]:
        return []
    findings = [
        "Entre sistemas diagonais, a distância de Wasserstein do grupo agregado "
        f"tem {_range_text(claim, 'Hz')}."
    ]
    pairs = _extremes_text(
        _median_by(frame, ["language_a", "language_b"], group="all"),
        lambda pair: _language_pair_pretty(*pair),
        "Hz",
    )
    if pairs:
        findings.append(f"Por par de idiomas, a mediana {pairs}.")
    trend, direction = _layer_trend(_median_by(frame, ["layer"], group="all"), "Hz")
    if trend:
        findings.append(trend)
    return _result_block(
        tables,
        figures,
        key="espectral",
        heading="Quanto a relevância se desloca em frequência?",
        question=(
            "entre sistemas diagonais de idiomas distintos, quanto se desloca a "
            "massa de relevância ao longo do eixo de frequência?"
        ),
        figure_names=("spectral_divergence_by_layer",),
        findings=findings,
        interpretation=(
            "O valor em Hz resume quanto a massa de relevância precisaria se "
            "deslocar no eixo de frequência para que os perfis médios coincidissem. "
            "Cada sistema usa o próprio probe e o próprio conjunto, então a "
            "distância reúne diferenças de treino e de dados. " + direction
        ).strip(),
        caution=(
            "distância maior não significa modelo melhor nem pior, não separa "
            "corpus de idioma e não identifica a causa do deslocamento."
        ),
        details=(
            _profiles_detail(figures, "spectral_divergence")
            + r"Apêndice~\ref{app:comparacoes} (Tabela~\ref{tab:spectral-divergence-summary}); "
            f"{_csv_tex('spectral_divergence')}."
        ),
    )


def _faithfulness_leader(claim: pd.DataFrame) -> str | None:
    """Model with the strictly largest share of usable intervals above zero."""
    usable = claim[claim["total"] > 0]
    if len(usable) < 2:
        return None
    shares = (usable["ci_above_zero_n"] / usable["total"]).to_numpy(dtype=float)
    order = np.argsort(-shares, kind="stable")
    if np.isclose(shares[order[0]], shares[order[1]]):
        return None
    return str(usable["model"].iloc[order[0]])


def _faithfulness_block(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    aligned = _aligned_layer_faithfulness(tables)
    claim = faithfulness_claim(aligned)
    if claim.empty:
        return []
    items = [
        f"{escape_latex(_model_name(row.model))}: {row.ci_above_zero_n} de "
        f"{row.total} intervalos inteiramente acima de zero e "
        f"{row.ci_below_zero_n} de {row.total} inteiramente abaixo de zero"
        for row in claim.itertuples(index=False)
    ]
    findings = (
        [f"{item}." for item in items]
        if len(items) <= 3
        else [_join_pt(items) + "."]
    )
    interpretation = (
        "Um intervalo inteiramente acima de zero sugere que, neste protocolo, "
        "remover as faixas mais relevantes segundo o DFT-LRP altera mais o escore "
        "do probe do que remover o controle (faixas aleatórias ou menos "
        "relevantes); a medida é a variação do escore, não uma mudança direta "
        "da classificação."
    )
    leader = _faithfulness_leader(claim)
    if leader is not None:
        interpretation += (
            f" {escape_latex(_model_name(leader))} tem a maior proporção de "
            "intervalos acima de zero, o que indica maior concordância entre "
            "explicação e intervenção nesse modelo, não superioridade do modelo."
        )
    layers = _join_pt([str(int(value)) for value in sorted(aligned["layer"].unique())])
    pairs = _join_pt([str(int(value)) for value in sorted(aligned["n_pairs"].unique())])
    return _result_block(
        tables,
        figures,
        key="fidelidade",
        heading="As faixas apontadas sustentam a decisão?",
        question=(
            "remover as faixas apontadas como mais relevantes altera mais o escore "
            "do que remover faixas de controle?"
        ),
        figure_names=("layer_faithfulness_intervals",),
        findings=findings,
        interpretation=interpretation,
        caution=(
            f"escopo restrito à camada {layers} das células diagonais, com {pairs} "
            "pares por classe e contagens sem correção para comparações múltiplas. "
            "A intervenção STFT controlada não implica causalidade no mundo real e "
            "não se estende a outras camadas nem à transferência."
        ),
        details=(
            r"Apêndice~\ref{app:fidelidade} (Tabela~\ref{tab:layer-faithfulness-summary}); "
            f"{_csv_tex('layer_faithfulness_summary')}."
        ),
    )


def _synthesis_scope_tex(tables: ReportTables) -> str:
    rows = len(build_results_synthesis(tables))
    return f"{_count_word_pt(rows)} perguntas comparativas"


def _main_results_section(
    tables: ReportTables, figures: Mapping[str, FigureRecord]
) -> list[str]:
    blocks = (
        _overview_block,
        _transfer_block,
        _agreement_block,
        _frequency_profile_block,
        _relevance_shift_block,
        _spectral_block,
        _faithfulness_block,
    )
    lines = [
        rf"\section{{{_SECTION_TITLES_PT[2]}}}",
        r"\label{sec:resultados}",
        "Os resultados vão do padrão global de desempenho para as explicações e, "
        "por fim, para a fidelidade das explicações. Cada bloco responde a uma "
        "pergunta; os valores citados são derivados das tabelas do bundle.",
        "",
    ]
    for block in blocks:
        lines += block(tables, figures)
    descriptive_sections = [
        r"visão geral do desempenho (Seção~\ref{sec:resultado-desempenho})"
    ]
    if (
        not frequency_profile_claims(tables).empty
        and "dft_relevance_heatmap" in figures
    ):
        descriptive_sections.append(
            r"perfil por faixa de frequência "
            r"(Seção~\ref{sec:resultado-faixas-frequencia})"
        )
    lines += [
        r"\subsection{Síntese}",
        r"A Tabela~\ref{tab:results-synthesis} resume as "
        f"{_synthesis_scope_tex(tables)}, uma por linha: transferência, acordo "
        "entre encoders, mudança de relevância (que reúne a distância "
        "Jensen--Shannon e a divergência espectral) e fidelidade, com a "
        "evidência usada, o resultado e a cautela correspondente. "
        f"{_join_pt(descriptive_sections).capitalize()} "
        + ("não têm linhas próprias." if len(descriptive_sections) > 1 else "não tem linha própria."),
        "",
        r"\input{tables/results_synthesis.tex}",
        "",
    ]
    return lines


def _cross_cutting_limitations_section(tables: ReportTables) -> list[str]:
    limitations = [
        "As explicações descrevem o probe linear sobre as representações do "
        "encoder e não constituem verdade de referência sobre as pistas acústicas "
        "do áudio sintético.",
        "Toda comparação entre idiomas, quando presente, reflete uma mudança "
        "conjunta de corpus/idioma: áudio, corpus e idioma mudam juntos, e as "
        "fontes de variação do procedimento não isolam o efeito do idioma.",
        f"Os intervalos percentis de {_ci_percent_tex()} quantificam apenas a "
        "variação obtida ao reamostrar os áudios observados; não incluem novas execuções do pipeline "
        "nem substituem testes de hipótese.",
        "O relatório não recalcula explicações, não seleciona amostras e não "
        "treina probes; todos os valores vêm dos artefatos listados em "
        f"{_tt_breakable(_MANIFEST_NAME)}.",
    ]
    return [
        rf"\section{{{_SECTION_TITLES_PT[3]}}}",
        r"\label{sec:limitacoes}",
        "As cautelas específicas acompanham cada resultado. Valem para todo o "
        "relatório:",
        "",
        *_itemize(limitations),
    ]


def _profile_diagonal_text(tables: ReportTables, model: str) -> str:
    selected = tables.transfer_selection
    rows = selected[
        (selected["model"] == model) & (selected["source"] == selected["target"])
    ]
    if rows.empty:
        return "Não há células diagonais disponíveis para este encoder."
    return (
        f"Nas {len(rows)} células diagonais, a ROC-AUC selecionada tem mediana "
        f"{_num(rows['roc_auc'].median())} e o MCC da mesma camada, "
        f"{_num(rows['mcc'].median())}."
    )


def _profile_layer_text(tables: ReportTables, model: str) -> str:
    selected = tables.transfer_selection
    diagonal_selected = selected[
        (selected["model"] == model) & (selected["source"] == selected["target"])
    ]
    performance = tables.performance
    diagonal = performance[
        (performance["model"] == model)
        & (performance["source"] == performance["target"])
    ]
    if diagonal.empty:
        return "Não há perfil diagonal por camada disponível para este encoder."

    by_layer = diagonal.groupby("layer", sort=True)["roc_auc"].median()
    first_layer, last_layer = int(by_layer.index.min()), int(by_layer.index.max())
    peak = float(by_layer.max())
    peak_layers = [
        str(int(layer))
        for layer, value in by_layer.items()
        if np.isclose(float(value), peak)
    ]
    peak_location = (
        f"na camada {peak_layers[0]}"
        if len(peak_layers) == 1
        else f"nas camadas {_join_pt(peak_layers)}"
    )
    selected_layers = sorted(
        {int(layer) for layer in diagonal_selected["selected_layer"]}
    )
    selected_text = (
        _join_pt([str(layer) for layer in selected_layers])
        if selected_layers
        else "não disponível"
    )
    return (
        f"As camadas escolhidas nas células diagonais são {selected_text}. "
        f"A mediana da ROC-AUC diagonal é {_num(by_layer.loc[first_layer])} na "
        f"camada {first_layer} e {_num(by_layer.loc[last_layer])} na camada "
        f"{last_layer}; o maior valor mediano, {_num(peak)}, ocorre "
        f"{peak_location}. "
        "A comparação entre extremos não implica evolução monotônica."
    )


def _profile_frequency_text(tables: ReportTables, model: str) -> str:
    claims = frequency_profile_claims(tables)
    rows = claims[claims["model"] == model]
    if rows.empty:
        return "Não há resumo de relevância por faixa disponível para este encoder."
    parts = []
    for language in tables.languages:
        language_rows = rows[rows["language"] == language]
        if language_rows.empty:
            continue
        labels = [
            escape_latex(str(label)).replace("–", "--")
            for label in language_rows["band_label"]
        ]
        parts.append(
            f"{escape_latex(_language_name(language))}: {_join_pt(labels)} em "
            f"{int(language_rows['layers_dominant'].iloc[0])} de "
            f"{int(language_rows['n_layers'].iloc[0])} camadas"
        )
    if not parts:
        return "Não há resumo de relevância por faixa disponível para este encoder."
    return "Faixas mais frequentemente dominantes: " + "; ".join(parts) + "."


def _profile_transfer_text(tables: ReportTables, model: str) -> str:
    parts = []
    for metric, label in (("delta_roc_auc", "ROC-AUC"), ("delta_mcc", "MCC")):
        rows = selected_transfer_effects(tables, metric)
        rows = rows[rows["model"] == model]
        claim = transfer_effect_claim(rows)
        if claim["n"]:
            parts.append(
                f"$\\Delta$ {label} com mediana {_signed(claim['median'])}; "
                f"{claim['negative_n']} de {claim['n']} estimativas negativas"
            )
    if not parts:
        return "Não há células fora da diagonal disponíveis para este encoder."
    return "; ".join(parts) + "."


def _profile_faithfulness_text(tables: ReportTables, model: str) -> str:
    claim = faithfulness_claim(
        _aligned_layer_faithfulness(tables).query("model == @model")
    )
    if claim.empty:
        return "Não há resultados de fidelidade alinhados para este encoder."
    row = claim.iloc[0]
    return (
        f"{int(row['ci_above_zero_n'])} de {int(row['total'])} intervalos ficam "
        "inteiramente acima de zero e "
        f"{int(row['ci_below_zero_n'])} de {int(row['total'])} ficam "
        "inteiramente abaixo de zero."
    )


def _encoder_profile_sections(tables: ReportTables) -> list[str]:
    """Parallel, data-derived profiles used in the report conclusion."""
    lines: list[str] = []
    for model in tables.models:
        lines += [
            rf"\subsection{{{escape_latex(_model_name(model))}}}",
            rf"\textbf{{Desempenho diagonal.}} {_profile_diagonal_text(tables, model)}",
            "",
            rf"\textbf{{Ao longo das camadas.}} {_profile_layer_text(tables, model)}",
            "",
            rf"\textbf{{Frequência.}} {_profile_frequency_text(tables, model)}",
            "",
            rf"\textbf{{Transferência.}} {_profile_transfer_text(tables, model)}",
            "",
            rf"\textbf{{Fidelidade.}} {_profile_faithfulness_text(tables, model)}",
            "",
        ]
    return lines


def _cross_encoder_conclusion(tables: ReportTables) -> list[str]:
    if len(tables.models) < 2:
        return [
            r"\subsection{Comparação entre encoders}",
            "Este relatório contém apenas um encoder; portanto, não sustenta uma "
            "comparação entre arquiteturas.",
            "",
        ]

    def displayed_leaders(values: pd.Series) -> tuple[list[str], str]:
        displayed_maximum = _num(values.max())
        leaders = [
            str(model)
            for model, value in values.items()
            if _num(value) == displayed_maximum
        ]
        return leaders, displayed_maximum

    diagonal = tables.transfer_selection[
        tables.transfer_selection["source"] == tables.transfer_selection["target"]
    ]
    performance_parts = []
    for metric, label in (("roc_auc", "ROC-AUC"), ("mcc", "MCC")):
        medians = diagonal.groupby("model", sort=False)[metric].median()
        if len(medians) < 2:
            continue
        leaders, value = displayed_leaders(medians)
        performance_parts.append(
            f"a maior mediana de {label} é {value}, em "
            f"{_join_pt([escape_latex(_model_name(model)) for model in leaders])}"
        )
    performance = (
        "Nas células diagonais, " + "; ".join(performance_parts) + "."
        if performance_parts
        else ""
    )

    diagonal_performance = tables.performance[
        tables.performance["source"] == tables.performance["target"]
    ]
    peaks = []
    for model in tables.models:
        rows = diagonal_performance[diagonal_performance["model"] == model]
        if rows.empty:
            continue
        by_layer = rows.groupby("layer", sort=True)["roc_auc"].median()
        peaks.append(
            f"{escape_latex(_model_name(model))}: camada {int(by_layer.idxmax())} "
            f"({_num(by_layer.max())})"
        )
    layer_text = (
        "A camada com maior mediana diagonal de ROC-AUC também muda: "
        + "; ".join(peaks)
        + "."
        if peaks
        else ""
    )

    frequency = frequency_profile_claims(tables)
    stable_frequency = []
    for model in tables.models:
        rows = frequency[frequency["model"] == model]
        if rows.empty or rows["language"].nunique() != len(tables.languages):
            continue
        one_per_language = rows.groupby("language").size().eq(1).all()
        same_band = rows["band_label"].nunique() == 1
        all_layers = (rows["layers_dominant"] == rows["n_layers"]).all()
        if one_per_language and same_band and all_layers:
            stable_frequency.append(
                f"{escape_latex(_model_name(model))} mantém "
                f"{escape_latex(str(rows['band_label'].iloc[0])).replace('–', '--')} "
                "como faixa dominante em todos os idiomas e camadas"
            )
    frequency_text = (
        "Na frequência, " + "; ".join(stable_frequency) + "."
        if stable_frequency
        else (
            "Os perfis de frequência mudam entre encoders e podem mudar entre "
            "idiomas."
        )
    )

    transfer_parts = []
    for metric, label in (("delta_roc_auc", "ROC-AUC"), ("delta_mcc", "MCC")):
        rows = selected_transfer_effects(tables, metric)
        medians = rows.groupby("model", sort=False)["estimate"].median()
        if len(medians) < 2:
            continue
        maximum = float(medians.max())
        displayed_maximum = _signed(maximum)
        leaders = [
            str(model)
            for model, value in medians.items()
            if _signed(value) == displayed_maximum
        ]
        transfer_parts.append(
            f"a maior mediana de $\\Delta$ {label} é {displayed_maximum}, em "
            f"{_join_pt([escape_latex(_model_name(model)) for model in leaders])}"
        )
    transfer_text = (
        "Na transferência, " + "; ".join(transfer_parts) + "."
        if transfer_parts
        else ""
    )

    faithfulness = faithfulness_claim(_aligned_layer_faithfulness(tables))
    faithfulness_text = ""
    leader = _faithfulness_leader(faithfulness)
    if leader is not None:
        row = faithfulness[faithfulness["model"] == leader].iloc[0]
        faithfulness_text = (
            f"Na fidelidade, {escape_latex(_model_name(leader))} tem a maior "
            "proporção de intervalos inteiramente acima de zero "
            f"({int(row['ci_above_zero_n'])} de {int(row['total'])}); isso indica "
            "maior concordância entre explicação e intervenção neste protocolo, "
            "não superioridade geral."
        )
    return [
        r"\subsection{Comparação entre encoders}",
        performance,
        layer_text,
        frequency_text,
        transfer_text,
        faithfulness_text,
        "Em conjunto, os resultados mostram associação entre a escolha do encoder "
        "e os padrões observados; não identificam um efeito causal do idioma nem "
        "permitem atribuir as diferenças a uma característica específica do "
        "pré-treino.",
        "",
    ]


def _short_conclusion_section(tables: ReportTables) -> list[str]:
    profile_intro = (
        "O perfil abaixo usa"
        if len(tables.models) == 1
        else f"Os {len(tables.models)} perfis abaixo usam"
    )
    lines = [
        rf"\section{{{_SECTION_TITLES_PT[4]}}}",
        r"\label{sec:conclusao}",
        f"{profile_intro} a mesma sequência de leitura para mostrar o "
        "que muda quando se troca o encoder: desempenho diagonal, comportamento "
        "ao longo das camadas, relevância em frequência, transferência e "
        "fidelidade.",
        "",
    ]
    lines += _encoder_profile_sections(tables)
    lines += _cross_encoder_conclusion(tables)
    lines += [
        r"As quatro perguntas comparativas e suas cautelas estão na "
        r"Tabela~\ref{tab:results-synthesis}.",
        "",
    ]
    return lines


def write_report_tex(
    path: str | Path,
    tables: ReportTables,
    figures: Sequence[FigureRecord],
    sources: Sequence[LoadedReportSource],
) -> None:
    """Write the Portuguese report, deriving every statement from loaded values."""
    if not isinstance(tables, ReportTables):
        raise TypeError("tables must be a ReportTables instance")
    available = {record.name: record for record in figures}
    title = escape_latex(_report_title(tables))
    lines = [
        r"\documentclass[a4paper,11pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[brazilian]{babel}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{subcaption}",
        r"\usepackage{xcolor}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue!50!black,"
        r"urlcolor=blue!50!black}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\setlength{\emergencystretch}{3em}",
        # Leaves room below tall multi-panel figures for the longest caption.
        r"\newlength{\figuremaxheight}",
        r"\setlength{\figuremaxheight}{0.7\textheight}",
        rf"\title{{{title}}}",
        r"\author{}",
        r"\date{}",
        "",
        r"\begin{document}",
        # The TOC is empty on the first pass; separate numbering keeps body pages stable.
        r"\pagenumbering{roman}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        r"\pagenumbering{arabic}",
        "",
    ]
    sections = (
        _executive_summary_section(tables),
        _orientation_section(tables, sources),
        _main_results_section(tables, available),
        _cross_cutting_limitations_section(tables),
        _short_conclusion_section(tables),
        _appendices(tables, available, sources),
    )
    for section in sections:
        lines += section
    lines += [r"\end{document}", ""]
    _write_tex(Path(path), "\n".join(lines))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _aligned_probe_stability(tables: ReportTables) -> pd.DataFrame:
    frame = tables.probe_stability
    if frame.empty:
        return frame
    keys = ["model", "source", "target", "layer"]
    perf_keys = tables.performance[keys].drop_duplicates()
    return frame.merge(perf_keys, on=keys, how="inner")


def _aligned_layer_faithfulness(tables: ReportTables) -> pd.DataFrame:
    frame = tables.layer_faithfulness
    if frame.empty:
        return frame
    keys = ["model", "source", "target", "layer"]
    perf_keys = tables.performance[keys].drop_duplicates()
    subset = frame[
        frame["comparison"].isin(("top_minus_random", "top_minus_bottom"))
    ].copy()
    return subset.merge(perf_keys, on=keys, how="inner")


_FAITHFULNESS_CONTROLS = ("top_minus_random", "top_minus_bottom")
_FAITHFULNESS_COMPARISON_PT: Mapping[str, str] = MappingProxyType(
    {
        "top_minus_random": "top-minus-random",
        "top_minus_bottom": "top-minus-bottom",
    }
)
_FAITHFULNESS_CONTROL_COLORS: Mapping[str, str] = MappingProxyType(
    {"top_minus_random": "#1f77b4", "top_minus_bottom": "#d62728"}
)
_FAITHFULNESS_LANGUAGE_MARKERS: Mapping[str, str] = MappingProxyType(
    {"eng": "o", "por": "s", "zho": "^"}
)


def _displayable_faithfulness_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    """Aligned rows the faithfulness figure can show: diagonal, final layer, full n."""
    if aligned.empty:
        return aligned
    return aligned[
        (aligned["source"] == aligned["target"])
        & (aligned["layer"] == FINAL_LAYER)
        & (aligned["n_pairs"] == FAITHFULNESS_PER_CLASS)
        & aligned["comparison"].isin(_FAITHFULNESS_CONTROLS)
    ].copy()


def displayable_layer_faithfulness(tables: ReportTables) -> pd.DataFrame:
    return _displayable_faithfulness_rows(_aligned_layer_faithfulness(tables))


def faithfulness_aggregate_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count aggregate faithfulness cells by mean sign and bootstrap CI placement."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "comparison",
                "total",
                "n_mean_positive",
                "n_ci_above_zero",
                "n_ci_below_zero",
            ]
        )
    evaluated = frame.loc[frame["status"] == "ok"].copy()
    if evaluated.empty:
        evaluated = frame.copy()
    rows: list[dict[str, object]] = []
    for (model, comparison), group in evaluated.groupby(
        ["model", "comparison"], sort=True
    ):
        mean_diff = group["mean_difference"].astype(float)
        ci_low = group["ci_low"].astype(float)
        ci_high = group["ci_high"].astype(float)
        rows.append(
            {
                "model": model,
                "comparison": comparison,
                "total": int(len(group)),
                "n_mean_positive": int((mean_diff > 0).sum()),
                "n_ci_above_zero": int((ci_low > 0).sum()),
                "n_ci_below_zero": int((ci_high < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def probe_stability_aggregate_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Median and maximum probe metric std per model across layer/source/target cells."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "roc_auc_std_median",
                "roc_auc_std_max",
                "mcc_std_median",
                "mcc_std_max",
            ]
        )
    rows: list[dict[str, object]] = []
    for model in sorted(frame["model"].unique()):
        subset = frame.loc[frame["model"] == model]
        rows.append(
            {
                "model": model,
                "roc_auc_std_median": float(subset["roc_auc_std"].median()),
                "roc_auc_std_max": float(subset["roc_auc_std"].max()),
                "mcc_std_median": float(subset["mcc_std"].median()),
                "mcc_std_max": float(subset["mcc_std"].max()),
            }
        )
    return pd.DataFrame(rows)


def _faithfulness_scope_note(aligned: pd.DataFrame) -> str:
    cells = aligned[["model", "source", "target"]].drop_duplicates()
    n_cells = len(cells)
    k_values = sorted({int(value) for value in aligned["k"].dropna().unique()})
    k_text = _join_pt([str(value) for value in k_values])
    return (
        f"Escopo: {n_cells} célula(s) diagonal(is) modelo$\\times$idioma, "
        "duas classes (bonafide e spoof), "
        f"valores de $k$ = {escape_latex(k_text)}, "
        f"{_count_word_pt(FAITHFULNESS_PER_CLASS)} amostras por classe no "
        "subconjunto XAI (contrato fixo do protocolo de fidelidade). "
        "Top-$k$ e bottom-$k$ referem-se às faixas mais e menos relevantes "
        "segundo o DFT-LRP; top-minus-random e top-minus-bottom são diferenças "
        "médias pareadas frente a controles aleatórios ou às faixas menos "
        "relevantes. "
        "As contagens abaixo são descritivas, sem correção por comparações "
        "múltiplas, e não substituem teste de hipótese confirmatório."
    )


def _faithfulness_pattern_paragraph(counts: pd.DataFrame) -> str:
    if counts.empty:
        return ""
    by_model: list[tuple[str, int, int, int, int]] = []
    for model in sorted(counts["model"].unique()):
        subset = counts.loc[counts["model"] == model]
        by_model.append(
            (
                model,
                int(subset["n_ci_above_zero"].sum()),
                int(subset["n_ci_below_zero"].sum()),
                int(subset["n_mean_positive"].sum()),
                int(subset["total"].sum()),
            )
        )
    by_model.sort(key=lambda item: (item[1], item[3]), reverse=True)
    leader = by_model[0]
    leader_name = escape_latex(_model_name(leader[0]))
    mixed = [
        escape_latex(_model_name(model))
        for model, above, below, _positive, _total in by_model
        if above > 0 and below > 0
    ]
    trailing = by_model[-1]
    lines = [
        "Leitura cautelosa dos totais: "
        f"{leader_name} concentra o maior número de células agregadas com "
        f"intervalo de bootstrap inteiramente acima de zero ({leader[1]} de "
        f"{leader[4]} avaliadas), seguido pelos demais modelos presentes nesta "
        "edição."
    ]
    if len(by_model) > 1 and leader[1] > trailing[1]:
        lines.append(
            f"Em contraste, {escape_latex(_model_name(trailing[0]))} apresenta "
            f"menos sinais estritamente positivos ({trailing[1]} células com IC "
            "acima de zero)."
        )
    if mixed:
        lines.append(
            "Resultados mistos (simultaneamente células com IC acima e abaixo "
            f"de zero) aparecem em {_join_pt(mixed)}, o que limita generalizações "
            "fortes entre arquiteturas."
        )
    lines.append(
        "Nenhum destes padrões implica causalidade no mundo real; a filtragem "
        "STFT e o reescalonamento RMS permanecem fontes de variação do "
        "procedimento."
    )
    return " ".join(lines)


def _probe_stability_pattern_paragraph(summary: pd.DataFrame) -> str:
    if summary.empty:
        return ""
    auc_calm = summary.loc[summary["roc_auc_std_median"].idxmin()]
    mcc_calm = summary.loc[summary["mcc_std_median"].idxmin()]
    auc_volatile = summary.loc[summary["roc_auc_std_max"].idxmax()]
    mcc_volatile = summary.loc[summary["mcc_std_max"].idxmax()]
    return (
        "Entre os modelos listados, a menor mediana de desvio-padrão da "
        f"ROC-AUC aparece em {escape_latex(_model_name(auc_calm['model']))} "
        f"({_num(auc_calm['roc_auc_std_median'])}); para o MCC, em "
        f"{escape_latex(_model_name(mcc_calm['model']))} "
        f"({_num(mcc_calm['mcc_std_median'])}). O maior desvio-padrão máximo "
        f"da ROC-AUC aparece em {escape_latex(_model_name(auc_volatile['model']))} "
        f"({_num(auc_volatile['roc_auc_std_max'])}); para o MCC, em "
        f"{escape_latex(_model_name(mcc_volatile['model']))} "
        f"({_num(mcc_volatile['mcc_std_max'])}). "
        "Com apenas três reamostragens bootstrap, estes números funcionam "
        "como verificação de sensibilidade à amostra finita de treino, não "
        "como intervalo de confiança preciso para o desempenho."
    )


def _executive_fidelity_stability_blurbs(tables: ReportTables) -> list[str]:
    blurbs: list[str] = []
    aligned_fidelity = _aligned_layer_faithfulness(tables)
    if not aligned_fidelity.empty:
        claim = faithfulness_claim(aligned_fidelity)
        model = _faithfulness_leader(claim)
        if model is not None:
            row = claim[claim["model"] == model].iloc[0]
            blurbs.append(
                f"Na fidelidade por intervenção (camada~{FINAL_LAYER}, diagonal), "
                f"{escape_latex(_model_name(model))} tem a maior proporção de "
                "intervalos bootstrap inteiramente acima de zero "
                f"({int(row['ci_above_zero_n'])} de {int(row['total'])}), "
                "sem implicar superioridade causal."
            )
    aligned_stability = _aligned_probe_stability(tables)
    if not aligned_stability.empty:
        summary = probe_stability_aggregate_by_model(aligned_stability)
        if not summary.empty:
            calm = summary.sort_values("roc_auc_std_median").iloc[0]
            blurbs.append(
                "Na sensibilidade do treino do probe (três bootstraps "
                "estratificados), "
                f"{escape_latex(_model_name(calm['model']))} mostra a mediana "
                "mais baixa de desvio-padrão da ROC-AUC "
                f"({_num(calm['roc_auc_std_median'])}), interpretada apenas "
                "como cheque de robustez, não como incerteza ponta-a-ponta."
            )
    return blurbs


def _layer_faithfulness_tex(tables: ReportTables) -> str:
    aligned = _aligned_layer_faithfulness(tables)
    if aligned.empty:
        return ""
    counts = faithfulness_aggregate_counts(aligned)
    body = [
        _row(
            [
                escape_latex(_model_name(row.model)),
                escape_latex(
                    _FAITHFULNESS_COMPARISON_PT.get(
                        str(row.comparison), str(row.comparison)
                    )
                ),
                int(row.total),
                int(row.n_mean_positive),
                int(row.n_ci_above_zero),
                int(row.n_ci_below_zero),
            ]
        )
        for row in counts.itertuples(index=False)
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Resumo descritivo da fidelidade por intervenção "
            rf"(camada~{FINAL_LAYER}, células diagonais): contagens sobre células agregadas "
            r"por modelo e comparação. Detalhe completo em "
            r"\texttt{layer\_faithfulness\_summary.csv}.}",
            r"\label{tab:layer-faithfulness-summary}",
            _FIT_WIDTH_OPEN,
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            _row(
                [
                    "Modelo",
                    "Comparação",
                    "Total",
                    r"$\Delta$ média $>0$",
                    "IC $>0$",
                    "IC $<0$",
                ]
            ),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )


def _num_std(value: object, n_seeds: object) -> str:
    if int(n_seeds) <= 1:
        return r"\multicolumn{1}{c}{--}"
    return _num(value)


def _probe_stability_tex(tables: ReportTables) -> str:
    aligned = _aligned_probe_stability(tables)
    if aligned.empty:
        return ""
    summary = probe_stability_aggregate_by_model(aligned)
    body = [
        _row(
            [
                escape_latex(_model_name(row.model)),
                _num(row.roc_auc_std_median),
                _num(row.roc_auc_std_max),
                _num(row.mcc_std_median),
                _num(row.mcc_std_max),
            ]
        )
        for row in summary.itertuples(index=False)
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Sensibilidade do treino do probe: mediana e máximo do "
            r"desvio-padrão de ROC-AUC e MCC sobre células camada$\times$origem"
            r"$\times$destino (três bootstraps estratificados). Detalhe completo "
            r"em \texttt{probe\_stability\_by\_layer.csv}.}",
            r"\label{tab:probe-stability-summary}",
            _FIT_WIDTH_OPEN,
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            _row(
                [
                    "Modelo",
                    r"Mediana ROC $\sigma$",
                    r"Máx. ROC $\sigma$",
                    r"Mediana MCC $\sigma$",
                    r"Máx. MCC $\sigma$",
                ]
            ),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )


def _performance_tex(tables: ReportTables) -> str:
    blocks: list[str] = []
    for cell in _cells(tables):
        rows = _cell_rows(tables.performance, cell)
        body = [
            _row(
                [
                    int(row.layer),
                    int(row.n),
                    _num(row.roc_auc),
                    _num(row.average_precision),
                    _num(row.eer_diagnostic * 100, 1),
                    _num(row.accuracy),
                    _num(row.mcc),
                    _num(row.tpr),
                    _num(row.fpr),
                ]
            )
            for row in rows.itertuples(index=False)
        ]
        blocks += [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{Métricas por camada: {_cell_tex(cell)}. ROC-AUC e AP "
            r"sem limiar; EER diagnóstico em \%; acurácia, MCC, TPR e FPR no "
            r"limiar fixo da origem (classe positiva: sintético).}",
            rf"\label{{tab:performance-{_label(*cell)}}}",
            _FIT_WIDTH_OPEN,
            r"\begin{tabular}{rrrrrrrrr}",
            r"\toprule",
            _row(
                [
                    "Camada",
                    "$n$",
                    "ROC-AUC",
                    "AP",
                    r"EER (\%)",
                    "Acurácia",
                    "MCC",
                    "TPR",
                    "FPR",
                ]
            ),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    return "\n".join(blocks)


def _emergence_tex(tables: ReportTables) -> str:
    body = []
    for row in tables.emergence.itertuples(index=False):
        body.append(
            _row(
                [
                    _tt(row.model),
                    escape_latex(row.source),
                    escape_latex(row.target),
                    _NOT_AVAILABLE if _is_missing(row.onset) else int(row.onset),
                    _NOT_AVAILABLE
                    if _is_missing(row.consolidation)
                    else int(row.consolidation),
                    _num(row.final_auc),
                ]
            )
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Camadas de início e consolidação da emergência, "
            r"conforme persistidas pelo agregador, e ROC-AUC final.}",
            r"\label{tab:emergence}",
            r"\begin{tabular}{lllrrr}",
            r"\toprule",
            _row(
                [
                    "Modelo",
                    "Origem",
                    "Alvo",
                    "Início",
                    "Consolidação",
                    "ROC-AUC final",
                ]
            ),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _conservation_tex(tables: ReportTables) -> str:
    series = build_conservation_series(tables)
    panels = list(dict.fromkeys(series["panel"]))
    wide = series.pivot(index="layer", columns="panel", values="value")
    limits = [
        float(series.loc[series["panel"] == panel, "limit"].iloc[0])
        for panel in panels
    ]
    body = [
        _row([int(layer), *[_sci(wide.loc[layer, panel]) for panel in panels]])
        for layer in wide.index
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Pior caso por camada de cada verificação de conservação "
            r"(máximo sobre as células). As três primeiras colunas são resíduos "
            r"relativos comparados à tolerância de conservação; a última é a "
            r"razão do recálculo do score, aceita até 1.}",
            r"\label{tab:conservation}",
            r"\begin{tabular}{rrrrr}",
            r"\toprule",
            _row(["Camada", "Bias-zeroed", "DFT", "STDFT", "Score (razão)"]),
            r"\midrule",
            *body,
            r"\midrule",
            _row(["Limite", *[_sci(limit) for limit in limits]]),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _comparison_status_summary(frame: pd.DataFrame) -> str:
    counts = frame.groupby("status", sort=True).size()
    return "; ".join(f"{status}: {int(count)}" for status, count in counts.items())


def _comparison_compact_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    body: Sequence[Sequence[object]],
) -> str:
    if not body:
        return ""
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            _FIT_WIDTH_OPEN,
            rf"\begin{{tabular}}{{{'l' * len(headers)}}}",
            r"\toprule",
            _row(headers),
            r"\midrule",
            *(_row(row) for row in body),
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )


def _encoder_agreement_tex(tables: ReportTables) -> str:
    frame = tables.encoder_agreement
    if frame.empty:
        return ""
    body = []
    for keys, group in frame.groupby(["model_a", "model_b", "metric"], sort=True):
        model_a, model_b, metric = keys
        estimates = group["estimate"].to_numpy(dtype=np.float64)
        estimates = estimates[np.isfinite(estimates)]
        body.append(
            [
                escape_latex(_pair_pretty(str(model_a), str(model_b))),
                escape_latex(_comparison_metric_label(str(metric))),
                int(len(group)),
                _num(float(np.median(estimates))) if estimates.size else _NOT_AVAILABLE,
                (
                    f"{_num(float(np.min(estimates)))}--{_num(float(np.max(estimates)))}"
                    if estimates.size
                    else _NOT_AVAILABLE
                ),
                escape_latex(_comparison_status_summary(group)),
            ]
        )
    return _comparison_compact_table(
        "Resumo do acordo entre encoders: mediana e extensão agregadas por par "
        "de modelos e métrica (limites por linha apenas no CSV). Detalhe "
        "completo em \\texttt{encoder\\_agreement.csv}.",
        "tab:encoder-agreement-summary",
        ["Par", "Métrica", "Linhas", "Mediana", "Faixa", "Status (contagens)"],
        body,
    )


def _language_shift_tex(tables: ReportTables) -> str:
    frame = tables.language_shift
    if frame.empty:
        return ""
    body = []
    for (model, group_name), group in frame.groupby(["model", "group"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=np.float64)
        estimates = estimates[np.isfinite(estimates)]
        body.append(
            [
                escape_latex(_model_name(str(model))),
                escape_latex(_GROUP_LABELS_PT.get(str(group_name), str(group_name))),
                int(len(group)),
                _num(float(np.median(estimates))) if estimates.size else _NOT_AVAILABLE,
                (
                    f"{_num(float(np.min(estimates)))}--{_num(float(np.max(estimates)))}"
                    if estimates.size
                    else _NOT_AVAILABLE
                ),
                escape_latex(_comparison_status_summary(group)),
            ]
        )
    return _comparison_compact_table(
        "Resumo compacto da mudança de relevância entre idiomas-alvo (JS): mediana "
        "e faixa agregadas sobre linhas da tabela por modelo e grupo (não repete "
        "os limites linha a linha). Detalhe completo em "
        "\\texttt{language\\_shift.csv}.",
        "tab:language-shift-summary",
        ["Modelo", "Grupo", "Linhas", "Mediana", "Faixa", "Status (contagens)"],
        body,
    )


def _diagonal_vs_offdiagonal_tex(tables: ReportTables) -> str:
    frame = tables.diagonal_vs_offdiagonal
    if frame.empty:
        return ""
    body = []
    for (model, metric), group in frame.groupby(["model", "metric"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=np.float64)
        estimates = estimates[np.isfinite(estimates)]
        body.append(
            [
                escape_latex(_model_name(str(model))),
                escape_latex(_comparison_metric_label(str(metric))),
                int(len(group)),
                _num(float(np.median(estimates))) if estimates.size else _NOT_AVAILABLE,
                (
                    f"{_num(float(np.min(estimates)))}--{_num(float(np.max(estimates)))}"
                    if estimates.size
                    else _NOT_AVAILABLE
                ),
                escape_latex(_comparison_status_summary(group)),
            ]
        )
    return _comparison_compact_table(
        "Resumo compacto diagonal versus fora da diagonal ($\\Delta$ ROC-AUC e "
        "$\\Delta$ MCC) por modelo e métrica: medianas e faixas agregadas, sem "
        "repetir os limites linha a linha. Detalhe completo em "
        "\\texttt{diagonal\\_vs\\_offdiagonal.csv}.",
        "tab:diagonal-off-summary",
        ["Modelo", "Métrica", "Linhas", "Mediana", "Faixa", "Status (contagens)"],
        body,
    )


def _spectral_divergence_tex(tables: ReportTables) -> str:
    frame = tables.spectral_divergence
    if frame.empty:
        return ""
    body = []
    for (model, group_name), group in frame.groupby(["model", "group"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=np.float64)
        estimates = estimates[np.isfinite(estimates)]
        body.append(
            [
                escape_latex(_model_name(str(model))),
                escape_latex(_GROUP_LABELS_PT.get(str(group_name), str(group_name))),
                int(len(group)),
                _num(float(np.median(estimates))) if estimates.size else _NOT_AVAILABLE,
                (
                    f"{_num(float(np.min(estimates)))}--{_num(float(np.max(estimates)))}"
                    if estimates.size
                    else _NOT_AVAILABLE
                ),
                escape_latex(_comparison_status_summary(group)),
            ]
        )
    return _comparison_compact_table(
        "Resumo compacto da divergência espectral (Wasserstein em Hz) por modelo "
        "e grupo: medianas e faixas agregadas (limites completos no CSV). "
        "Detalhe completo em \\texttt{spectral\\_divergence.csv}.",
        "tab:spectral-divergence-summary",
        ["Modelo", "Grupo", "Linhas", "Mediana (Hz)", "Faixa (Hz)", "Status (contagens)"],
        body,
    )


def _results_synthesis_tex(tables: ReportTables) -> str:
    synthesis = build_results_synthesis(tables)
    body = [
        _row(
            [
                escape_latex(row.question),
                escape_latex(row.evidence),
                escape_latex(row.result),
                escape_latex(row.caution),
            ]
        )
        for row in synthesis.itertuples(index=False)
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Síntese dos resultados verificados e dos limites de interpretação.}",
            r"\label{tab:results-synthesis}",
            _FIT_WIDTH_OPEN,
            r"\begin{tabular}{p{0.20\linewidth}p{0.25\linewidth}p{0.29\linewidth}p{0.22\linewidth}}",
            r"\toprule",
            _row(["Pergunta", "Evidência", "Resultado", "Cautela"]),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ]
    )


_COMPARISON_TEX_BUILDERS: Mapping[str, object] = MappingProxyType(
    {
        "encoder_agreement": _encoder_agreement_tex,
        "language_shift": _language_shift_tex,
        "diagonal_vs_offdiagonal": _diagonal_vs_offdiagonal_tex,
        "spectral_divergence": _spectral_divergence_tex,
    }
)


def write_report_tables(tables: ReportTables, tables_dir: str | Path) -> tuple[str, ...]:
    """Write the derived CSV tables and LaTeX table fragments; return file names."""
    if not isinstance(tables, ReportTables):
        raise TypeError("tables must be a ReportTables instance")
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, attribute in _TABLE_CSVS:
        frame = getattr(tables, attribute)
        frame.to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")
        written.append(f"{name}.csv")
    fragments = {
        "performance_by_layer": _performance_tex(tables),
        "emergence_layers": _emergence_tex(tables),
        "conservation_by_layer": _conservation_tex(tables),
    }
    stability_tex = _probe_stability_tex(tables)
    if stability_tex.strip():
        fragments["probe_stability_by_layer"] = stability_tex
    fidelity_tex = _layer_faithfulness_tex(tables)
    if fidelity_tex.strip():
        fragments["layer_faithfulness_summary"] = fidelity_tex
    synthesis = build_results_synthesis(tables)
    synthesis.to_csv(
        tables_dir / "results_synthesis.csv", index=False, lineterminator="\n"
    )
    written.append("results_synthesis.csv")
    fragments["results_synthesis"] = _results_synthesis_tex(tables)
    for name, text in fragments.items():
        if text:
            _write_tex(tables_dir / f"{name}.tex", text)
            written.append(f"{name}.tex")
    aligned = _aligned_probe_stability(tables)
    if not aligned.empty:
        aligned.to_csv(
            tables_dir / "probe_stability_by_layer.csv",
            index=False,
            lineterminator="\n",
        )
        written.append("probe_stability_by_layer.csv")
    aligned_fidelity = _aligned_layer_faithfulness(tables)
    if not aligned_fidelity.empty:
        aligned_fidelity.to_csv(
            tables_dir / "layer_faithfulness_summary.csv",
            index=False,
            lineterminator="\n",
        )
        written.append("layer_faithfulness_summary.csv")
    for name in COMPARISON_FAMILIES:
        frame = getattr(tables, name)
        if name in tables.omitted_comparisons or frame.empty:
            continue
        frame.to_csv(tables_dir / f"{name}.csv", index=False, lineterminator="\n")
        written.append(f"{name}.csv")
        builder = _COMPARISON_TEX_BUILDERS[name]
        text = builder(tables)
        if text:
            _write_tex(tables_dir / f"{name}.tex", text)
            written.append(f"{name}.tex")
    return tuple(sorted(written))


# ---------------------------------------------------------------------------
# Manifest and bundle
# ---------------------------------------------------------------------------


def _source_sort_key(item: LoadedReportSource) -> tuple:
    identity = item.source.identity
    return (identity.profile, identity.languages, identity.protocol, identity.scope)


def _consumed_artifacts(source: ReportSource) -> list[dict[str, str]]:
    root = source.root
    paths = LayerwiseSuitePaths(root)
    profile = source.identity.profile
    files = {root / "run_status.json", root / "execution_plan.json"}
    generations = [
        source.aggregate_generation,
        *source.layer_xai_generations.values(),
        *source.final_trace_generations.values(),
    ]
    for generation in generations:
        files.add(generation.parent.parent / "active.json")
        files.update(path for path in generation.rglob("*") if path.is_file())
    for _, layer, origin, target in source.layer_xai_generations:
        cell_dir = paths.cell(profile, layer, origin, target)
        files.update(
            cell_dir / name
            for name in ("scores.npy", "predictions.parquet", "metrics.json")
        )
        stage_id = f"{profile}:cell:{layer:02d}:{origin}:{target}"
        files.add(root / ".state" / f"{stage_id.replace(':', '__')}.json")
    consumed = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_of(path)}
        for path in files
    ]
    return sorted(consumed, key=lambda item: item["path"])


def _manifest_inputs(source: ReportSource) -> dict[str, dict[str, object]]:
    inputs = source.plan.get("inputs")
    declared: dict[str, dict[str, object]] = {}
    if not isinstance(inputs, Mapping):
        return declared
    for language in source.identity.languages:
        entry = inputs.get(language)
        if not isinstance(entry, Mapping):
            continue
        record: dict[str, object] = {}
        for key in ("config_sha256", "manifest_sha256"):
            if isinstance(entry.get(key), str):
                record[key] = entry[key]
        counts = entry.get("role_counts")
        if isinstance(counts, Mapping):
            record["role_counts"] = {
                str(role): counts[role] for role in sorted(counts)
            }
        declared[language] = record
    return declared


def _is_report_artifact(relative: PurePosixPath) -> bool:
    """Whether a bundle-relative path is a file this generator can produce."""
    if len(relative.parts) == 1:
        return relative.name in {"report.tex", _BUILD_SCRIPT_NAME}
    if len(relative.parts) == 2:
        suffixes = {
            "figures": {".pdf", ".png"},
            "tables": {".csv", ".tex"},
        }.get(relative.parts[0])
        return suffixes is not None and relative.suffix in suffixes
    return False


def _generated_file_entries(
    bundle_dir: Path, generated_files: Sequence[str] | None
) -> list[dict[str, str]]:
    if generated_files is None:
        relative = [
            PurePosixPath(path.relative_to(bundle_dir).as_posix())
            for path in bundle_dir.rglob("*")
            if path.is_file()
        ]
        relative = [item for item in relative if _is_report_artifact(item)]
    else:
        relative = [PurePosixPath(item) for item in generated_files]
    entries = []
    for item in sorted(set(relative)):
        if (
            not item.parts
            or item.is_absolute()
            or ".." in item.parts
            or "." in item.parts
            or "\\" in str(item)
            or str(item) == _MANIFEST_NAME
            or not (bundle_dir / Path(*item.parts)).is_file()
        ):
            raise ValueError(f"invalid generated file entry: {item}")
        entries.append(
            {
                "path": str(item),
                "sha256": _sha256_of(bundle_dir / Path(*item.parts)),
            }
        )
    return entries


def build_report_manifest(
    sources: Sequence[LoadedReportSource],
    bundle_dir: str | Path,
    generated_files: Sequence[str] | None = None,
) -> dict[str, object]:
    """Describe consumed sources and generated files with stable relative paths.

    `generated_files` lists the bundle-relative files the current generation
    produced. When omitted, only recognised report artifacts are scanned, so
    compilation leftovers (`report.aux`, `.log`, `.pdf`, ...) are never listed.
    """
    ordered = sorted(sources, key=_source_sort_key)
    bundle_dir = Path(bundle_dir)
    entries = []
    for item in ordered:
        source = item.source
        identity = source.identity
        profile = identity.profile
        entries.append(
            {
                "result_directory": source.root.name,
                "config_hash": source.status["config_hash"],
                "profile": profile,
                "languages": list(identity.languages),
                "scope": identity.scope,
                "layers": sorted({key[1] for key in source.layer_xai_generations}),
                "aggregate_generation": source.aggregate_generation_id,
                "layer_xai_generations": {
                    f"{profile}/layer_{layer:02d}/{origin}->{target}": generation.name
                    for (_, layer, origin, target), generation in sorted(
                        source.layer_xai_generations.items()
                    )
                },
                "final_trace_generations": {
                    f"{profile}/{origin}->{target}": generation.name
                    for (_, origin, target), generation in sorted(
                        source.final_trace_generations.items()
                    )
                },
                "inputs": _manifest_inputs(source),
                "band_edges": [
                    {
                        "language": edge.language,
                        "origin": edge.origin,
                        "reason": edge.reason,
                        "n_bands": edge.n_bands,
                        "f_min": edge.f_min,
                        "f_max": edge.f_max,
                    }
                    for edge in item.band_edges
                ],
                "consumed_artifacts": _consumed_artifacts(source),
            }
        )
    return {
        "schema_version": 1,
        "sources": entries,
        "generated_files": _generated_file_entries(bundle_dir, generated_files),
    }


def _manifest_text(manifest: Mapping[str, object]) -> str:
    return (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _check_output_directory(output: Path, roots: Sequence[Path]) -> None:
    resolved = output.resolve()
    for root in roots:
        source = root.resolve()
        if (
            resolved == source
            or resolved.is_relative_to(source)
            or source.is_relative_to(resolved)
        ):
            raise ValueError(
                f"output {output} must not overlap the read-only result "
                f"directory {root}"
            )
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"output {output} must be a directory")
    if any(output.iterdir()) and not (output / _MANIFEST_NAME).is_file():
        raise ValueError(
            f"output {output} is not empty and is not a previous report bundle"
        )
    _previous_bundle_files(output)


def _previous_bundle_files(output: Path) -> list[Path]:
    """Files listed by a previous manifest, proven to stay inside `output`.

    Every entry is validated before anything is deleted: it must be a non-empty
    relative path (no drive, UNC root, backslash or `..` component) and its
    resolved location, links included, must lie strictly inside the resolved
    output directory.
    """
    manifest_path = output / _MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    base = output.resolve()
    targets: list[Path] = []
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = [item["path"] for item in previous["generated_files"]]
        for raw in listed:
            if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
                raise ValueError(f"unsafe path {raw!r}")
            posix, windows = PurePosixPath(raw), PureWindowsPath(raw)
            if (
                not posix.parts
                or posix.is_absolute()
                or windows.drive
                or windows.root
                or any(part in {"", ".", ".."} for part in raw.split("/"))
            ):
                raise ValueError(f"unsafe path {raw!r}")
            target = output / Path(*posix.parts)
            resolved = target.resolve()
            if resolved == base or not resolved.is_relative_to(base):
                raise ValueError(f"path {raw!r} escapes the output directory")
            targets.append(target)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"previous report manifest at {manifest_path} is unreadable or "
            f"lists a path outside the output directory: {exc}"
        ) from exc
    return targets


def generate_report_bundle(
    result_dirs: Sequence[str | Path], output_dir: str | Path
) -> Path:
    """Validate explicit result directories and write a portable report bundle.

    Sources are read-only; nothing is written until every source has been
    validated and loaded.
    """
    roots = [Path(item) for item in result_dirs]
    if not roots:
        raise ValueError("at least one result directory is required")
    seen: set[ExperimentIdentity] = set()
    for root in roots:
        identity = parse_result_dir_name(root)
        if identity in seen:
            raise ValueError(f"duplicate result identity: {root.name}")
        seen.add(identity)
    output = Path(output_dir)
    _check_output_directory(output, roots)

    loaded = [
        load_report_source(validate_result_directory(root)) for root in roots
    ]
    tables = build_report_tables(loaded)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.new-", dir=output.parent)
    )
    try:
        # mkdtemp is always 0700; publish with the mode a plain mkdir would get.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(staging, 0o777 & ~umask)
        _build_bundle(staging, tables, loaded)
        _verify_staged_bundle(staging)
        _publish_bundle(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _build_bundle(
    staging: Path, tables: ReportTables, loaded: Sequence[LoadedReportSource]
) -> None:
    figures = render_report_figures(tables, staging / "figures")
    table_files = write_report_tables(tables, staging / "tables")
    write_report_tex(staging / "report.tex", tables, figures, loaded)
    _write_text(staging / _BUILD_SCRIPT_NAME, _BUILD_LOCAL_PS1)
    produced = [
        "report.tex",
        _BUILD_SCRIPT_NAME,
        *[f"figures/{name}" for record in figures for name in record.files],
        *[f"tables/{name}" for name in table_files],
    ]
    _write_text(
        staging / _MANIFEST_NAME,
        _manifest_text(build_report_manifest(loaded, staging, produced)),
    )


def _verify_staged_bundle(staging: Path) -> None:
    """Prove the staged bundle is complete before it can replace anything."""
    manifest = json.loads((staging / _MANIFEST_NAME).read_text(encoding="utf-8"))
    listed = {item["path"]: item["sha256"] for item in manifest["generated_files"]}
    required = {"report.tex", _BUILD_SCRIPT_NAME}
    if not required <= set(listed):
        raise ValueError("staged report bundle is missing required files")
    for relative, digest in listed.items():
        if _sha256_of(staging / Path(*PurePosixPath(relative).parts)) != digest:
            raise ValueError(f"staged report bundle file {relative} changed after listing")


def _publish_bundle(staging: Path, output: Path) -> None:
    """Replace `output` by `staging`, restoring the previous bundle on failure."""
    backup = None
    if output.exists():
        backup = output.parent / f".{output.name}.old-{uuid.uuid4().hex[:12]}"
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None:
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brspeech_xai.layerwise_report",
        description=(
            "Generate a portable Portuguese LaTeX report bundle from completed "
            "layer-wise XAI result directories. Read-only: no encoder is "
            "loaded and no GPU is used."
        ),
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "completed result directory named "
            "<model>__<languages>__layerwise_xai__<scope>; repeat for several"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="DIR",
        help="new or previously generated report bundle directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output = generate_report_bundle(args.result, args.output)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Report bundle written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
