"""TDD contract for read-only layer-wise report source validation."""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import brspeech_xai.layerwise_report as report_module
from brspeech_xai.encoder_suite import LANGUAGE_ORDER, LAYER_ORDER
from brspeech_xai.layerwise_paths import LayerwiseSuitePaths, publish_generation
from brspeech_xai.layerwise_report import (
    ExperimentIdentity,
    LoadedReportSource,
    ReportSource,
    abbreviate_sample_id,
    build_conservation_series,
    build_report_manifest,
    build_report_tables,
    escape_latex,
    faithfulness_aggregate_counts,
    generate_report_bundle,
    load_report_source,
    main,
    parse_result_dir_name,
    probe_stability_aggregate_by_model,
    render_report_figures,
    validate_result_directory,
    write_report_tables,
    write_report_tex,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _language_segment(languages: tuple[str, ...]) -> str:
    return "-".join(languages)


def _write_generation(
    destination: Path,
    *,
    role: str,
    stdft_count: int = 0,
    payload_name: str = "payload.json",
) -> None:
    def writer(directory: Path) -> None:
        (directory / payload_name).write_text(
            json.dumps({"schema_version": 1, "role": role}),
            encoding="utf-8",
        )
        if stdft_count:
            examples = directory / "stdft_examples"
            examples.mkdir()
            for index in range(stdft_count):
                np.savez(
                    examples / f"example_{index:02d}.npz",
                    sample_id=np.asarray(f"sample-{index}"),
                    value=np.asarray(index, dtype=np.float32),
                )

    publish_generation(destination, writer, role=role)


def write_minimal_result_root(
    tmp_path: Path,
    *,
    status: str = "complete",
    mode: str | None = None,
    dir_name: str = "hubert_base__eng__layerwise_xai__full",
    profile: str = "hubert_base",
    languages: tuple[str, ...] = ("eng",),
    stdft_examples_per_class: int = 1,
    plan_profile: str | None = None,
    plan_languages: tuple[str, ...] | None = None,
    config_hash: str = "abc123config",
    break_aggregate_pointer: bool = False,
    wrong_layer_xai_role: bool = False,
    missing_stdft: bool = False,
) -> Path:
    root = tmp_path / dir_name
    root.mkdir(parents=True, exist_ok=True)
    paths = LayerwiseSuitePaths(root)

    effective_profile = plan_profile if plan_profile is not None else profile
    effective_languages = (
        plan_languages if plan_languages is not None else languages
    )
    stdft_count = 0 if missing_stdft else 2 * stdft_examples_per_class

    plan = {
        "schema_version": 1,
        "config_hash": config_hash,
        "suite_config": {
            "profiles": (effective_profile,),
            "stdft_examples_per_class": stdft_examples_per_class,
        },
        "profiles": {
            effective_profile: {
                "checkpoint": "facebook/hubert-base-ls960",
                "encoder": "hubert",
            }
        },
        "layers": list(LAYER_ORDER),
        "languages": {
            language: {"role": "source_and_target"}
            for language in effective_languages
        },
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, sort_keys=True),
        encoding="utf-8",
    )

    run_status = {
        "schema_version": 1,
        "status": status,
        "config_hash": config_hash,
    }
    if mode is not None:
        run_status["mode"] = mode
    (root / "run_status.json").write_text(
        json.dumps(run_status, sort_keys=True),
        encoding="utf-8",
    )

    _write_generation(root / "aggregates", role="suite_aggregates")
    if break_aggregate_pointer:
        (root / "aggregates" / "active.json").write_text("{}", encoding="utf-8")

    layer_role = "wrong_role" if wrong_layer_xai_role else "layer_xai"
    for layer in LAYER_ORDER:
        for source in languages:
            for target in languages:
                destination = paths.layer_xai(
                    profile, layer, source, target
                )
                _write_generation(
                    destination,
                    role=layer_role,
                    stdft_count=stdft_count,
                )

    for source in languages:
        for target in languages:
            destination = paths.final_trace_cell(profile, source, target)
            _write_generation(destination, role="final_trace")

    return root


def test_parse_result_name_uses_model_language_protocol_and_scope():
    identity = parse_result_dir_name(
        Path("hubert_base__eng__layerwise_xai__full")
    )
    assert identity.profile == "hubert_base"
    assert identity.languages == ("eng",)
    assert identity.scope == "full"


@pytest.mark.parametrize(
    ("status", "mode"),
    [("failed", None), ("running", None), ("complete", "dry-run")],
)
def test_validation_rejects_non_complete_production_run(
    tmp_path, status, mode
):
    root = write_minimal_result_root(tmp_path, status=status, mode=mode)
    with pytest.raises(ValueError, match="completed non-dry-run"):
        validate_result_directory(root)


def test_parse_result_dir_name_rejects_malformed_name():
    with pytest.raises(ValueError, match="malformed result directory name"):
        parse_result_dir_name(Path("hubert_base_eng_layerwise_xai_full"))


_ALL_LANGUAGE_SUBSETS = (
    ("eng", ("eng",)),
    ("por", ("por",)),
    ("zho", ("zho",)),
    ("eng-por", ("eng", "por")),
    ("eng-zho", ("eng", "zho")),
    ("por-zho", ("por", "zho")),
    ("eng-por-zho", ("eng", "por", "zho")),
)


@pytest.mark.parametrize(("segment", "languages"), _ALL_LANGUAGE_SUBSETS)
@pytest.mark.parametrize("scope", ["pilot", "full"])
def test_parse_result_dir_name_accepts_every_non_empty_language_subset(
    segment, languages, scope
):
    identity = parse_result_dir_name(
        Path(f"wavlm_base__{segment}__layerwise_xai__{scope}")
    )
    assert identity.profile == "wavlm_base"
    assert identity.languages == languages
    assert identity.scope == scope


def test_language_subsets_match_the_suite_canonical_order():
    from itertools import combinations

    expected = {
        "-".join(subset): subset
        for size in range(1, len(LANGUAGE_ORDER) + 1)
        for subset in combinations(LANGUAGE_ORDER, size)
    }
    assert expected == dict(_ALL_LANGUAGE_SUBSETS)
    assert report_module._LANGUAGE_SEGMENTS == expected
    assert len(expected) == 7


@pytest.mark.parametrize(
    "segment",
    [
        "zho-eng",
        "por-eng",
        "zho-por",
        "eng-zho-por",
        "zho-por-eng",
        "eng--zho",
        "eng-zho-",
        "-eng-zho",
        "eng_zho",
        "eng-deu",
        "eng-eng",
        "engzho",
    ],
)
def test_parse_result_dir_name_rejects_non_canonical_language_segments(segment):
    with pytest.raises(ValueError, match="malformed result directory name"):
        parse_result_dir_name(Path(f"hubert_base__{segment}__layerwise_xai__full"))


def test_validation_accepts_an_eng_zho_result_directory(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        dir_name="hubert_base__eng-zho__layerwise_xai__full",
        languages=("eng", "zho"),
    )
    source = validate_result_directory(root)
    assert source.identity.languages == ("eng", "zho")
    assert source.identity.profile == "hubert_base"
    assert {key[2:] for key in source.layer_xai_generations} == {
        ("eng", "eng"),
        ("eng", "zho"),
        ("zho", "eng"),
        ("zho", "zho"),
    }


def test_validation_rejects_an_eng_zho_name_whose_plan_adds_por(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        dir_name="hubert_base__eng-zho__layerwise_xai__full",
        languages=("eng", "zho"),
        plan_languages=("eng", "por", "zho"),
    )
    with pytest.raises(ValueError, match="language"):
        validate_result_directory(root)


def test_report_bundle_is_generated_for_an_eng_zho_result(tmp_path):
    root = write_scientific_result_root(tmp_path, languages=("eng", "zho"))
    assert root.name == "hubert_base__eng-zho__layerwise_xai__full"
    output = tmp_path / "report"
    generate_report_bundle([root], output)
    tex = _read_text(output / "report.tex")
    assert "inglês" in tex and "mandarim" in tex and "português" not in tex
    manifest = json.loads(_read_text(output / "report_manifest.json"))
    assert manifest["sources"][0]["languages"] == ["eng", "zho"]


def test_validation_rejects_plan_profile_mismatch(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        plan_profile="wavlm_base",
    )
    with pytest.raises(ValueError, match="profile"):
        validate_result_directory(root)


def test_validation_rejects_plan_language_mismatch(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        plan_languages=("eng", "por"),
    )
    with pytest.raises(ValueError, match="language"):
        validate_result_directory(root)


def test_validation_rejects_unknown_plan_language(tmp_path):
    root = write_minimal_result_root(tmp_path)
    plan_path = root / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["languages"]["deu"] = {"role": "source_and_target"}
    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown language"):
        validate_result_directory(root)


def test_validation_rejects_broken_active_pointer(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        break_aggregate_pointer=True,
    )
    with pytest.raises(ValueError, match="active generation"):
        validate_result_directory(root)


def test_validation_rejects_wrong_generation_role(tmp_path):
    root = write_minimal_result_root(
        tmp_path,
        wrong_layer_xai_role=True,
    )
    with pytest.raises(ValueError, match="active generation"):
        validate_result_directory(root)


def test_validation_rejects_missing_stdft_examples(tmp_path):
    root = write_minimal_result_root(tmp_path, missing_stdft=True)
    with pytest.raises(ValueError, match="stdft_examples"):
        validate_result_directory(root)


def test_validation_accepts_complete_minimal_root(tmp_path):
    root = write_minimal_result_root(tmp_path)
    source = validate_result_directory(root)

    assert isinstance(source, ReportSource)
    assert source.root == root
    assert source.identity == ExperimentIdentity(
        profile="hubert_base",
        languages=("eng",),
        protocol="layerwise_xai",
        scope="full",
    )
    assert source.aggregate_generation_id
    assert source.layer_xai_generations[("hubert_base", 1, "eng", "eng")].is_dir()
    assert source.final_trace_generations[("hubert_base", "eng", "eng")].is_dir()


def test_validation_preserves_source_immutability(tmp_path):
    root = write_minimal_result_root(tmp_path)
    before = _hash_tree(root)

    validate_result_directory(root)

    after = _hash_tree(root)
    assert before == after


def test_report_source_exposes_immutable_mappings(tmp_path):
    root = write_minimal_result_root(tmp_path)
    source = validate_result_directory(root)

    with pytest.raises(TypeError):
        source.status["status"] = "failed"

    with pytest.raises(TypeError):
        source.plan["config_hash"] = "tampered"

    with pytest.raises(TypeError):
        source.plan["languages"]["eng"]["role"] = "tampered"

    with pytest.raises(TypeError):
        source.layer_xai_generations[("hubert_base", 1, "eng", "eng")] = Path(
            "/tmp/evil"
        )

    with pytest.raises(TypeError):
        source.final_trace_generations[("hubert_base", "eng", "eng")] = Path(
            "/tmp/evil"
        )


def test_validation_counts_only_npz_files_not_directories(tmp_path):
    root = write_minimal_result_root(tmp_path)
    paths = LayerwiseSuitePaths(root)
    generation = paths.layer_xai("hubert_base", 1, "eng", "eng") / "generations"
    generation_dir = next(generation.iterdir())
    examples = generation_dir / "stdft_examples"
    (examples / "decoy.npz").mkdir()

    source = validate_result_directory(root)

    assert len(list(examples.glob("*.npz"))) == 3
    assert sum(1 for path in examples.glob("*.npz") if path.is_file()) == 2
    assert source.identity.languages == ("eng",)


# ---------------------------------------------------------------------------
# Task 2: scientific loader, tables and figures (synthetic immutable results)
# ---------------------------------------------------------------------------

_COHORT = (("real-a", 0), ("real-b", 0), ("spoof-a", 1), ("spoof-b", 1))
_TRACE_SAMPLES = ("real-a", "spoof-a")
_EIGHT_FAMILIES = (
    "performance_by_layer",
    "fixed_threshold_by_layer",
    "transfer_performance_heatmaps",
    "dft_relevance_heatmap",
    "class_relevance_by_layer",
    "decision_reorganization_by_layer",
    "stdft_examples",
    "conservation_diagnostics",
)
_COMPARISON_FAMILIES = (
    "encoder_agreement",
    "language_shift",
    "diagonal_vs_offdiagonal",
    "spectral_divergence",
)


def _cell_seed(layer: int, source: str, target: str) -> int:
    return (
        1000 * layer
        + 10 * LANGUAGE_ORDER.index(source)
        + LANGUAGE_ORDER.index(target)
    )


_ROLE_COUNTS = {"train": 12, "calibration": 6, "test": 8}
_PREDICTION = {"real-a": 0, "real-b": 1, "spoof-a": 1, "spoof-b": 1}
_SCORE_RTOL = 5e-3
_SCORE_ATOL = 2e-6
_VALIDATION_KIND = "bias_zeroed_model_rule_check"


def _score_error(layer: int, index: int) -> float:
    return 2e-5 * layer * (index + 1)


def _sample_frame(
    profile: str,
    layer: int,
    source: str,
    target: str,
    n_bands: int,
    cohort: Sequence[tuple[str, int]] = _COHORT,
) -> pd.DataFrame:
    rng = np.random.default_rng(_cell_seed(layer, source, target))
    rows = []
    for index, (sample_id, y_true) in enumerate(cohort):
        prediction = _PREDICTION.get(sample_id, y_true)
        rows.append(
            {
                "profile": profile,
                "layer": layer,
                "source": source,
                "target": target,
                "sample_id": sample_id,
                "y_true": y_true,
                "processed_path": f"/synthetic/{sample_id}.wav",
                "score": 0.25 + 0.5 * prediction,
                "score_recompute_absolute_error": _score_error(layer, index),
                "prediction": prediction,
                "band_signed": rng.normal(size=n_bands).tolist(),
                "band_abs_normalized": rng.dirichlet(
                    np.ones(n_bands)
                ).tolist(),
            }
        )
    return pd.DataFrame(rows)


def _summary_rows(
    samples: pd.DataFrame,
    *,
    grouping: str,
    group_column: str | None,
) -> list[dict]:
    first = samples.iloc[0]
    groups = (
        [("all", samples)]
        if group_column is None
        else list(samples.groupby(group_column))
    )
    rows = []
    for group_value, frame in groups:
        for measure, column in (
            ("signed", "band_signed"),
            ("absolute_normalized", "band_abs_normalized"),
        ):
            values = np.stack(frame[column].to_numpy())
            n = len(values)
            mean = values.mean(axis=0)
            sem = (
                values.std(axis=0, ddof=1) / np.sqrt(n)
                if n > 1
                else np.zeros(values.shape[1])
            )
            for band, (center, error) in enumerate(zip(mean, sem), start=1):
                rows.append(
                    {
                        "profile": first["profile"],
                        "layer": int(first["layer"]),
                        "source": first["source"],
                        "target": first["target"],
                        "grouping": grouping,
                        "group": group_value,
                        "measure": measure,
                        "band": band,
                        "n": n,
                        "mean": float(center),
                        "ci_low": float(center - 1.96 * error),
                        "ci_high": float(center + 1.96 * error),
                    }
                )
    return rows


def _write_layer_xai_generation(
    destination: Path,
    *,
    profile: str,
    layer: int,
    source: str,
    target: str,
    n_bands: int,
    tolerance: float,
    shuffle: bool,
    cohort: Sequence[tuple[str, int]] = _COHORT,
    byclass_corruption: str | None = None,
    conservation_overrides: Mapping[str, object] | None = None,
) -> None:
    samples = _sample_frame(profile, layer, source, target, n_bands, cohort)
    overall = pd.DataFrame(
        _summary_rows(samples, grouping="all", group_column=None)
    )
    byclass = pd.DataFrame(
        _summary_rows(samples, grouping="y_true", group_column="y_true")
        + _summary_rows(
            samples, grouping="prediction", group_column="prediction"
        )
    )
    if byclass_corruption == "prediction_mean":
        mask = (
            (byclass["grouping"] == "prediction")
            & (byclass["measure"] == "absolute_normalized")
            & (byclass["band"] == 1)
        )
        byclass.loc[mask, "mean"] += 0.01
    elif byclass_corruption == "y_true_n":
        byclass.loc[byclass["grouping"] == "y_true", "n"] += 1
    elif byclass_corruption == "missing_prediction_group":
        byclass = byclass[
            ~((byclass["grouping"] == "prediction") & (byclass["group"] == 0))
        ]
    elif byclass_corruption == "y_true_ci":
        mask = (byclass["grouping"] == "y_true") & (byclass["group"] == 1)
        byclass.loc[mask, "ci_high"] += 0.1
    elif byclass_corruption == "all_ci":
        overall.loc[:, "ci_low"] -= 0.1
    elif byclass_corruption == "signed_group_mean":
        mask = (
            (byclass["grouping"] == "y_true")
            & (byclass["measure"] == "signed")
            & (byclass["group"] == 0)
            & (byclass["band"] == 3)
        )
        byclass.loc[mask, "mean"] -= 0.05
    elif byclass_corruption is not None:
        raise ValueError(byclass_corruption)
    if shuffle:
        overall = overall.iloc[::-1].reset_index(drop=True)
        byclass = byclass.iloc[::-1].reset_index(drop=True)
        samples = samples.iloc[::-1].reset_index(drop=True)

    def writer(directory: Path) -> None:
        samples.to_parquet(directory / "sample_relevance.parquet", index=False)
        overall.to_csv(directory / "dft_band_relevance.csv", index=False)
        byclass.to_csv(directory / "dft_band_relevance_byclass.csv", index=False)
        conservation = {
            "profile": profile,
            "layer": layer,
            "source": source,
            "target": target,
            "n": len(samples),
            "validation_kind": _VALIDATION_KIND,
            "validation_sample_id": str(cohort[0][0]),
            "bias_zeroed_validation_residual": 1e-6 * layer,
            "max_bias_inclusive_attribution_gap": 2e-4,
            "max_score_recompute_absolute_error": float(
                samples["score_recompute_absolute_error"].max()
            ),
            "score_recompute_rtol": _SCORE_RTOL,
            "score_recompute_atol": _SCORE_ATOL,
            "max_dft_residual": 1e-7 * layer,
            "max_stdft_conservation_relative_error": 5e-8 * layer,
            "tolerance": tolerance,
        }
        conservation.update(conservation_overrides or {})
        (directory / "attnlrp_conservation.json").write_text(
            json.dumps(conservation, sort_keys=True),
            encoding="utf-8",
        )
        examples = directory / "stdft_examples"
        examples.mkdir()
        order = list(cohort)
        if shuffle:
            order = order[::-1]
        for sample_id, _ in order:
            rng = np.random.default_rng(
                [_cell_seed(layer, source, target), len(sample_id), ord(sample_id[-1])]
            )
            artifact_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
            times = np.linspace(0.0, 1.0, 5)
            freqs = np.linspace(0.0, 8000.0, 9)
            relevance = rng.normal(size=(5, 9))
            np.savez(
                examples / f"{artifact_id}.npz",
                sample_id=np.asarray(sample_id),
                times=times,
                freqs=freqs,
                relevance=relevance,
                spectrum=np.abs(rng.normal(size=(5, 9))),
                relevance_time_sum=np.asarray(float(relevance.sum())),
                relevance_tf_sum=np.asarray(float(relevance.sum())),
                conservation_absolute_error=np.asarray(0.0),
                conservation_relative_error=np.asarray(1e-8),
                conservation_tolerance=np.asarray(tolerance),
            )

    publish_generation(destination, writer, role="layer_xai")


def _write_final_trace_generation(
    destination: Path,
    *,
    profile: str,
    source: str,
    target: str,
    transitions_per_sample: int,
    shuffle: bool,
) -> None:
    rows = []
    for sample_id in _TRACE_SAMPLES:
        for previous in range(1, transitions_per_sample + 1):
            rows.append(
                {
                    "profile": profile,
                    "source": source,
                    "target": target,
                    "sample_id": sample_id,
                    "previous_layer": previous,
                    "current_layer": previous + 1,
                    "similarity": 0.5 + 0.03 * previous,
                    "normalized_l1_change": 0.9 - 0.04 * previous,
                    "absolute_mass": 1.0 + previous,
                    "temporal_entropy": 2.0 + 0.1 * previous,
                    "signed_sum": 0.1 * previous,
                    "logit": 1.5,
                }
            )
    frame = pd.DataFrame(rows)
    if shuffle:
        frame = frame.iloc[::-1].reset_index(drop=True)

    def writer(directory: Path) -> None:
        frame.to_csv(directory / "layer_transition_metrics.csv", index=False)

    publish_generation(destination, writer, role="final_trace")


@functools.lru_cache(maxsize=None)
def _cell_artifacts(layer: int, source: str, target: str):
    """Self-consistent persisted predictions and the metrics derived from them."""
    from brspeech_xai.metrics import evaluate_at_threshold

    rng = np.random.default_rng(_cell_seed(layer, source, target) + 7)
    ids = [sample_id for sample_id, _ in _COHORT]
    labels = [label for _, label in _COHORT]
    for index in range(36):
        ids.append(f"fill-{index:02d}")
        labels.append(index % 2)
    y = np.asarray(labels)
    gain = 0.03 * layer + (0.0 if source == target else -0.01)
    scores = np.clip(0.5 + (y - 0.5) * gain + rng.normal(0.0, 0.2, len(y)), 0.0, 1.0)
    metrics = evaluate_at_threshold(scores, y, 0.5)
    predictions = pd.DataFrame(
        {
            "sample_id": ids,
            "y_true": y,
            "score": scores,
            "prediction": (scores >= 0.5).astype(int),
        }
    )
    return predictions, metrics


def _cell_metrics(layer: int, source: str, target: str) -> dict:
    return _cell_artifacts(layer, source, target)[1]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_cell_marker(
    root: Path,
    profile: str,
    layer: int,
    source: str,
    target: str,
    *,
    metrics_points_to: str = "metrics.json",
    recorded_prefix: str | None = None,
) -> Path:
    """Write a cell marker in the producer's schema.

    `recorded_prefix` replaces the machine/root part of the recorded path, as in a
    bundle published elsewhere and moved afterwards.
    """
    cell_dir = LayerwiseSuitePaths(root).cell(profile, layer, source, target)
    artifacts = []
    for name, file_name in (
        ("scores", "scores.npy"),
        ("predictions", "predictions.parquet"),
        ("metrics", metrics_points_to),
    ):
        path = cell_dir / file_name
        recorded = (
            str(path.resolve())
            if recorded_prefix is None
            else f"{recorded_prefix}/{path.relative_to(root).as_posix()}"
        )
        artifacts.append(
            {"name": name, "path": recorded, "sha256": _sha256_file(path)}
        )
    stage_id = f"{profile}:cell:{layer:02d}:{source}:{target}"
    state = root / ".state"
    state.mkdir(exist_ok=True)
    marker = state / f"{stage_id.replace(':', '__')}.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_version": "encoder-suite-v2",
                "stage_id": stage_id,
                "profile": profile,
                "fingerprint": hashlib.sha256(stage_id.encode()).hexdigest(),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return marker


def _rewrite_cell_metrics(root: Path, profile: str, layer: int, mutate) -> None:
    metrics_path = (
        LayerwiseSuitePaths(root).cell(profile, layer, "eng", "eng") / "metrics.json"
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    mutate(payload)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    _refresh_cell_marker(root, profile, layer, "eng", "eng")


def write_scientific_result_root(
    tmp_path: Path,
    *,
    profile: str = "hubert_base",
    languages: tuple[str, ...] = ("eng",),
    n_bands: int = 8,
    config_n_bands: int | None = None,
    with_config: bool = True,
    transitions_per_sample: int = 11,
    shuffle: bool = False,
    tolerance: float = 1e-3,
    base_dir: str = "results",
    cohort_overrides: Mapping[int, Sequence[tuple[str, int]]] | None = None,
    byclass_corruption: tuple[int, str] | None = None,
    conservation_overrides: Mapping[str, object] | None = None,
    aggregate_n_offset: int = 0,
    root_name: str | None = None,
    role_counts: Mapping[str, int] | None = _ROLE_COUNTS,
) -> Path:
    segment = "-".join(languages)
    root = tmp_path / base_dir / (
        root_name or f"{profile}__{segment}__layerwise_xai__full"
    )
    root.mkdir(parents=True)
    paths = LayerwiseSuitePaths(root)

    inputs: dict[str, dict] = {language: {} for language in languages}
    if role_counts is not None:
        for language in languages:
            inputs[language]["role_counts"] = dict(role_counts)
    if with_config:
        config_dir = tmp_path / base_dir / f"configs_{profile}"
        config_dir.mkdir(exist_ok=True)
        for language in languages:
            config_path = config_dir / f"{language}.yaml"
            config_path.write_text(
                "bands:\n"
                f"  n_bands: {config_n_bands or n_bands}\n"
                "  f_min: 20.0\n"
                "  f_max: 7900.0\n",
                encoding="utf-8",
            )
            inputs[language].update(
                {
                    "config_path": str(config_path),
                    "config_sha256": hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                }
            )

    plan = {
        "schema_version": 1,
        "config_hash": "scientific-config",
        "suite_config": {
            "profiles": [profile],
            "stdft_examples_per_class": 2,
        },
        "profiles": {profile: {"checkpoint": "synthetic", "encoder": "hubert"}},
        "layers": list(LAYER_ORDER),
        "languages": {
            language: {"role": "source_and_target"} for language in languages
        },
        "inputs": inputs,
    }
    (root / "execution_plan.json").write_text(
        json.dumps(plan, sort_keys=True), encoding="utf-8"
    )
    (root / "run_status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "config_hash": "scientific-config",
            }
        ),
        encoding="utf-8",
    )

    cells = [
        (layer, source, target)
        for layer in LAYER_ORDER
        for source in languages
        for target in languages
    ]
    performance = pd.DataFrame(
        [
            {
                "profile": profile,
                "layer": layer,
                "source": source,
                "target": target,
                "n": len(_cell_artifacts(layer, source, target)[0])
                + aggregate_n_offset,
                "auc": _cell_metrics(layer, source, target)["threshold_free"][
                    "roc_auc"
                ],
                "accuracy": _cell_metrics(layer, source, target)[
                    "fixed_threshold"
                ]["accuracy"],
                "diagonal": source == target,
                "external_validation": source != target,
                "corpus_shift": source != target,
            }
            for layer, source, target in cells
        ]
    )
    emergence = pd.DataFrame(
        [
            {
                "profile": profile,
                "source": source,
                "target": target,
                "onset": 2,
                "consolidation": 8,
                "final_auc": _cell_metrics(12, source, target)[
                    "threshold_free"
                ]["roc_auc"],
                "chance": 0.5,
                "fraction": 0.95,
                "consecutive": 2,
                "bootstrap_confidence": 0.95,
                "bootstrap_n": 100,
                "corpus_shift": source != target,
            }
            for source in languages
            for target in languages
        ]
    )
    if shuffle:
        performance = performance.iloc[::-1].reset_index(drop=True)
        emergence = emergence.iloc[::-1].reset_index(drop=True)

    def aggregate_writer(directory: Path) -> None:
        performance.to_csv(directory / "layerwise_performance.csv", index=False)
        emergence.to_csv(directory / "emergence_layers.csv", index=False)

    publish_generation(
        root / "aggregates", aggregate_writer, role="suite_aggregates"
    )

    for layer, source, target in cells:
        cell_dir = paths.cell(profile, layer, source, target)
        cell_dir.mkdir(parents=True)
        predictions, metrics = _cell_artifacts(layer, source, target)
        predictions.to_parquet(cell_dir / "predictions.parquet", index=False)
        np.save(
            cell_dir / "scores.npy",
            predictions["score"].to_numpy(dtype=np.float32),
        )
        (cell_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        _refresh_cell_marker(root, profile, layer, source, target)
        corruption = (
            byclass_corruption[1]
            if byclass_corruption is not None and byclass_corruption[0] == layer
            else None
        )
        _write_layer_xai_generation(
            paths.layer_xai(profile, layer, source, target),
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            n_bands=n_bands,
            tolerance=tolerance,
            shuffle=shuffle,
            cohort=(cohort_overrides or {}).get(layer, _COHORT),
            byclass_corruption=corruption,
            conservation_overrides=conservation_overrides,
        )
    for source in languages:
        for target in languages:
            _write_final_trace_generation(
                paths.final_trace_cell(profile, source, target),
                profile=profile,
                source=source,
                target=target,
                transitions_per_sample=transitions_per_sample,
                shuffle=shuffle,
            )
    return root


@pytest.fixture(scope="module")
def eng_root(tmp_path_factory):
    return write_scientific_result_root(tmp_path_factory.mktemp("eng_case"))


@pytest.fixture(scope="module")
def eng_loaded(eng_root):
    return load_report_source(validate_result_directory(eng_root))


@pytest.fixture(scope="module")
def eng_tables(eng_loaded):
    return build_report_tables([eng_loaded])


def test_loader_and_tables_cover_the_layerwise_scientific_contract(
    eng_loaded, eng_tables
):
    assert isinstance(eng_loaded, LoadedReportSource)
    tables = eng_tables
    assert set(tables.performance["layer"]) == set(range(1, 13))
    assert {
        "roc_auc",
        "eer_diagnostic",
        "accuracy",
        "mcc",
        "tpr",
        "fpr",
        "fnr",
    } <= set(tables.performance.columns)
    assert "average_precision" in tables.performance.columns
    assert tables.band_relevance["band"].nunique() == 8
    assert set(tables.band_relevance["measure"]) == {
        "signed",
        "absolute_normalized",
    }
    assert len(tables.emergence) == 1
    assert set(tables.transitions["previous_layer"]) == set(range(1, 12))
    assert set(tables.transitions["current_layer"]) == set(range(2, 13))
    assert (tables.transitions["n"] == 2).all()
    assert set(tables.conservation["layer"]) == set(range(1, 13))
    assert set(tables.cohort["n"]) == {4}
    assert set(tables.cohort["n_real"]) == {2}
    assert set(tables.cohort["n_synthetic"]) == {2}


def test_tables_attach_explicit_model_and_language_identity(eng_tables):
    frames = (
        eng_tables.performance,
        eng_tables.emergence,
        eng_tables.band_relevance,
        eng_tables.class_relevance,
        eng_tables.transitions,
        eng_tables.conservation,
        eng_tables.stdft_examples,
    )
    for frame in frames:
        assert set(frame["model"]) == {"hubert_base"}
        assert set(frame["source"]) == {"eng"}
        assert set(frame["target"]) == {"eng"}
        assert set(frame["language"]) == {"eng"}
    assert eng_tables.models == ("hubert_base",)
    assert eng_tables.languages == ("eng",)


def test_band_edges_come_from_recorded_config_when_available(eng_tables):
    provenance = eng_tables.band_edge_provenance
    assert list(provenance["language"]) == ["eng"]
    assert provenance.iloc[0]["origin"] == "config"
    edges = eng_tables.band_edges
    assert len(edges) == 8
    assert edges["low_hz"].iloc[0] == pytest.approx(20.0)
    assert edges["high_hz"].iloc[-1] == pytest.approx(7900.0)
    assert (
        edges["low_hz"].iloc[1:].to_numpy()
        == edges["high_hz"].iloc[:-1].to_numpy()
    ).all()


def test_band_edges_fall_back_to_default_mel_contract_and_record_it(tmp_path):
    root = write_scientific_result_root(tmp_path, with_config=False)
    tables = build_report_tables(
        [load_report_source(validate_result_directory(root))]
    )
    provenance = tables.band_edge_provenance.iloc[0]
    assert provenance["origin"] == "fallback_default_mel_contract"
    assert provenance["reason"] == "no_config_recorded"
    assert (provenance["n_bands"], provenance["f_min"], provenance["f_max"]) == (
        8,
        20.0,
        7900.0,
    )
    assert tables.band_relevance["band_low_hz"].notna().all()


def test_band_edges_fall_back_when_config_hash_diverges(tmp_path):
    root = write_scientific_result_root(tmp_path)
    plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    config_path = Path(plan["inputs"]["eng"]["config_path"])
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
    )
    tables = build_report_tables(
        [load_report_source(validate_result_directory(root))]
    )
    provenance = tables.band_edge_provenance.iloc[0]
    assert provenance["origin"] == "fallback_default_mel_contract"
    assert provenance["reason"] == "config_sha256_mismatch"


def test_loader_rejects_relevance_vectors_that_disagree_with_band_edges(tmp_path):
    root = write_scientific_result_root(tmp_path, config_n_bands=4)
    source = validate_result_directory(root)
    with pytest.raises(ValueError, match="band"):
        load_report_source(source)


def test_loader_rejects_cell_metrics_that_contradict_aggregates(tmp_path):
    root = write_scientific_result_root(tmp_path)
    _rewrite_cell_metrics(
        root,
        "hubert_base",
        5,
        lambda payload: payload["threshold_free"].update(roc_auc=0.01),
    )
    source = validate_result_directory(root)
    with pytest.raises(ValueError, match="roc_auc"):
        load_report_source(source)


def test_loader_rejects_cell_metrics_missing_required_fixed_threshold_field(
    tmp_path,
):
    root = write_scientific_result_root(tmp_path)
    _rewrite_cell_metrics(
        root,
        "hubert_base",
        3,
        lambda payload: payload["fixed_threshold"].pop("mcc"),
    )
    source = validate_result_directory(root)
    with pytest.raises(ValueError, match="mcc"):
        load_report_source(source)


# --- I1: non-aggregated metrics need a validated cell marker and consistency ---


def test_loader_rejects_metrics_json_changed_after_the_cell_marker(tmp_path):
    root = write_scientific_result_root(tmp_path)
    metrics_path = (
        LayerwiseSuitePaths(root).cell("hubert_base", 4, "eng", "eng")
        / "metrics.json"
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["fixed_threshold"]["tpr"] = 0.123
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="marker"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_cell_without_state_marker(tmp_path):
    root = write_scientific_result_root(tmp_path)
    marker = root / ".state" / "hubert_base__cell__06__eng__eng.json"
    marker.unlink()
    with pytest.raises(ValueError, match="marker"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_marker_whose_metrics_artifact_is_another_file(tmp_path):
    root = write_scientific_result_root(tmp_path)
    _refresh_cell_marker(
        root,
        "hubert_base",
        7,
        "eng",
        "eng",
        metrics_points_to="predictions.parquet",
    )
    with pytest.raises(ValueError, match="metrics artifact"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_unaggregated_metric_that_contradicts_predictions(tmp_path):
    root = write_scientific_result_root(tmp_path)
    _rewrite_cell_metrics(
        root,
        "hubert_base",
        8,
        lambda payload: payload["fixed_threshold"].update(tpr=0.999),
    )
    with pytest.raises(ValueError, match="tpr.*predictions"):
        load_report_source(validate_result_directory(root))


# --- Markers survive a moved/renamed bundle without losing integrity ---

_OLD_ROOT_NAME = "layerwise-eng-full"
_VM_PREFIX = f"/workspace/results/{_OLD_ROOT_NAME}"


def _cell_marker_file(root: Path, layer: int) -> Path:
    return root / ".state" / f"hubert_base__cell__{layer:02d}__eng__eng.json"


def _edit_cell_marker(root: Path, layer: int, mutate) -> None:
    path = _cell_marker_file(root, layer)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record_vm_paths(root: Path) -> None:
    for layer in LAYER_ORDER:
        _refresh_cell_marker(
            root, "hubert_base", layer, "eng", "eng", recorded_prefix=_VM_PREFIX
        )


def test_loader_reads_bundle_published_under_old_root_then_renamed(tmp_path):
    published = write_scientific_result_root(tmp_path, root_name=_OLD_ROOT_NAME)
    recorded = json.loads(
        _cell_marker_file(published, 1).read_text(encoding="utf-8")
    )["artifacts"]
    assert all(_OLD_ROOT_NAME in item["path"] for item in recorded)
    renamed = published.with_name("hubert_base__eng__layerwise_xai__full")
    published.rename(renamed)
    assert not published.exists()
    assert all(
        _OLD_ROOT_NAME in item["path"]
        for item in json.loads(
            _cell_marker_file(renamed, 1).read_text(encoding="utf-8")
        )["artifacts"]
    )

    loaded = load_report_source(validate_result_directory(renamed))

    assert set(loaded.performance["layer"]) == set(LAYER_ORDER)


def test_loader_reads_bundle_whose_markers_record_a_vm_posix_root(tmp_path):
    root = write_scientific_result_root(tmp_path)
    _record_vm_paths(root)
    recorded = json.loads(_cell_marker_file(root, 1).read_text(encoding="utf-8"))
    assert recorded["artifacts"][0]["path"].startswith("/workspace/results/")

    loaded = load_report_source(validate_result_directory(root))

    assert set(loaded.performance["layer"]) == set(LAYER_ORDER)


def test_loader_reads_marker_with_windows_separators_in_recorded_paths(tmp_path):
    root = write_scientific_result_root(tmp_path)

    def windows_paths(payload):
        for item in payload["artifacts"]:
            relative = (
                LayerwiseSuitePaths(root)
                .cell("hubert_base", 6, "eng", "eng")
                .relative_to(root)
                / Path(item["path"]).name
            )
            item["path"] = "C:\\old\\" + str(relative).replace("/", "\\")

    _edit_cell_marker(root, 6, windows_paths)

    load_report_source(validate_result_directory(root))


@pytest.mark.parametrize(
    "bad_path",
    [
        f"{_VM_PREFIX}/hubert_base/cells/layer_03/eng_to_eng/metrics.json",
        f"{_VM_PREFIX}/other_profile/cells/layer_05/eng_to_eng/metrics.json",
        f"{_VM_PREFIX}/hubert_base/cells/layer_05/eng_to_eng/other/metrics.json",
        f"{_VM_PREFIX}/hubert_base/cells/layer_05/eng_to_eng/predictions.parquet",
        "hubert_base/cells/layer_05/eng_to_eng/metrics.json",
        f"{_VM_PREFIX}/../hubert_base/cells/layer_05/eng_to_eng/metrics.json",
    ],
    ids=[
        "other_layer",
        "other_profile",
        "extra_directory",
        "other_file",
        "relative_path",
        "parent_traversal",
    ],
)
def test_loader_rejects_old_path_with_divergent_suffix_even_with_valid_hash(
    tmp_path, bad_path
):
    root = write_scientific_result_root(tmp_path)
    _record_vm_paths(root)

    def redirect(payload):
        for item in payload["artifacts"]:
            if item["name"] == "metrics":
                item["path"] = bad_path

    _edit_cell_marker(root, 5, redirect)
    marker = json.loads(_cell_marker_file(root, 5).read_text(encoding="utf-8"))
    current = (
        LayerwiseSuitePaths(root).cell("hubert_base", 5, "eng", "eng")
        / "metrics.json"
    )
    metrics_item = next(i for i in marker["artifacts"] if i["name"] == "metrics")
    assert metrics_item["sha256"] == _sha256_file(current)

    with pytest.raises(ValueError, match="metrics artifact path .*canonical"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_moved_bundle_when_the_current_file_hash_differs(tmp_path):
    root = write_scientific_result_root(tmp_path)
    _record_vm_paths(root)
    scores = (
        LayerwiseSuitePaths(root).cell("hubert_base", 2, "eng", "eng") / "scores.npy"
    )
    np.save(scores, np.zeros(3, dtype=np.float32))

    with pytest.raises(ValueError, match="scores artifact sha256"):
        load_report_source(validate_result_directory(root))


@pytest.mark.parametrize(
    ("label", "mutate", "message"),
    [
        ("schema_version", lambda p: p.update(schema_version=2), "identity or contract"),
        ("contract", lambda p: p.update(contract_version="v1"), "identity or contract"),
        ("stage", lambda p: p.update(stage_id="hubert_base:cell:01:eng:eng"), "identity or contract"),
        ("profile", lambda p: p.update(profile="wavlm_base"), "identity or contract"),
        ("fingerprint_text", lambda p: p.update(fingerprint="fixture"), "fingerprint"),
        ("fingerprint_missing", lambda p: p.pop("fingerprint"), "schema"),
        ("profile_missing", lambda p: p.pop("profile"), "schema"),
        ("extra_key", lambda p: p.update(extra=1), "schema"),
        ("artifact_dropped", lambda p: p["artifacts"].pop(), "artifact names"),
        (
            "artifact_duplicated",
            lambda p: p["artifacts"].append(dict(p["artifacts"][0])),
            "artifact names",
        ),
        (
            "artifact_unknown",
            lambda p: p["artifacts"][0].update(name="extra"),
            "artifact names",
        ),
        ("artifact_extra_field", lambda p: p["artifacts"][0].update(size=1), "malformed"),
    ],
)
def test_loader_rejects_cell_marker_with_invalid_schema_or_identity(
    tmp_path, label, mutate, message
):
    root = write_scientific_result_root(tmp_path)
    _record_vm_paths(root)
    _edit_cell_marker(root, 4, mutate)

    with pytest.raises(ValueError, match=f"cell marker.*{message}"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_aggregate_n_that_contradicts_predictions(tmp_path):
    root = write_scientific_result_root(tmp_path, aggregate_n_offset=1)
    with pytest.raises(ValueError, match="n .*predictions"):
        load_report_source(validate_result_directory(root))


# --- I3: the XAI cohort is fixed per target across layers ---


def test_loader_rejects_xai_cohort_sample_ids_that_change_between_layers(tmp_path):
    changed = (("real-a", 0), ("real-z", 0), ("spoof-a", 1), ("spoof-b", 1))
    root = write_scientific_result_root(tmp_path, cohort_overrides={5: changed})
    with pytest.raises(ValueError, match="XAI cohort.*layer"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_xai_cohort_class_counts_that_change_between_layers(tmp_path):
    changed = (("real-a", 0), ("real-b", 0), ("spoof-a", 1), ("spoof-b", 0))
    root = write_scientific_result_root(tmp_path, cohort_overrides={9: changed})
    with pytest.raises(ValueError, match=r"XAI cohort.*\(2, 2\) vs \(3, 1\)"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_label_swapped_between_layers_even_with_same_counts(tmp_path):
    changed = (("real-a", 0), ("real-b", 1), ("spoof-a", 0), ("spoof-b", 1))
    root = write_scientific_result_root(tmp_path, cohort_overrides={2: changed})
    with pytest.raises(ValueError, match="XAI cohort"):
        load_report_source(validate_result_directory(root))


# --- I4: by-class summaries are recomputed from sample_relevance ---


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("prediction_mean", "grouping 'prediction'"),
        ("y_true_n", "grouping 'y_true'"),
        ("missing_prediction_group", "grouping 'prediction'"),
        ("y_true_ci", "grouping 'y_true'"),
        ("all_ci", "grouping 'all'"),
        ("signed_group_mean", "grouping 'y_true'"),
    ],
)
def test_loader_rejects_group_summaries_that_disagree_with_sample_relevance(
    tmp_path, kind, message
):
    root = write_scientific_result_root(tmp_path, byclass_corruption=(3, kind))
    with pytest.raises(ValueError, match=message):
        load_report_source(validate_result_directory(root))


# --- I2: conservation scopes and limits ---


def test_loader_rejects_unrecognised_bias_zeroed_validation_kind(tmp_path):
    root = write_scientific_result_root(
        tmp_path, conservation_overrides={"validation_kind": "something_else"}
    )
    with pytest.raises(ValueError, match="validation_kind"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_validation_sample_outside_the_xai_cohort(tmp_path):
    root = write_scientific_result_root(
        tmp_path, conservation_overrides={"validation_sample_id": "ghost"}
    )
    with pytest.raises(ValueError, match="validation_sample_id"):
        load_report_source(validate_result_directory(root))


def test_loader_rejects_score_error_summary_that_contradicts_samples(tmp_path):
    root = write_scientific_result_root(
        tmp_path, conservation_overrides={"max_score_recompute_absolute_error": 0.5}
    )
    with pytest.raises(ValueError, match="max_score_recompute_absolute_error"):
        load_report_source(validate_result_directory(root))


def test_conservation_table_records_validation_scope_and_score_rule(eng_tables):
    table = eng_tables.conservation
    assert set(table["validation_kind"]) == {_VALIDATION_KIND}
    assert set(table["validation_sample_id"]) == {"real-a"}
    assert set(table["n"]) == {4}
    assert set(table["n_stdft"]) == {4}
    assert set(table["score_recompute_rtol"]) == {_SCORE_RTOL}
    assert set(table["score_recompute_atol"]) == {_SCORE_ATOL}
    assert (table["max_score_recompute_ratio"] < 1.0).all()


def test_conservation_series_separate_scopes_and_do_not_share_one_tolerance(
    eng_tables,
):
    series = build_conservation_series(eng_tables)
    panels = list(dict.fromkeys(series["panel"]))
    assert panels == [
        "bias_zeroed_single_sample",
        "dft_xai_cohort",
        "stdft_subset",
        "score_recompute_rtol_atol",
    ]
    by_panel = {panel: series[series["panel"] == panel] for panel in panels}
    for panel, frame in by_panel.items():
        assert list(frame["layer"]) == list(range(1, 13)), panel
    bias = by_panel["bias_zeroed_single_sample"].iloc[0]
    assert "amostra única" in bias["scope"]
    assert _VALIDATION_KIND in bias["scope"]
    assert "real-a" in bias["scope"]
    assert "n=4" in by_panel["dft_xai_cohort"].iloc[0]["scope"]
    assert "coorte XAI" in by_panel["dft_xai_cohort"].iloc[0]["scope"]
    assert "subconjunto STDFT" in by_panel["stdft_subset"].iloc[0]["scope"]
    assert "n=4" in by_panel["stdft_subset"].iloc[0]["scope"]
    score = by_panel["score_recompute_rtol_atol"].iloc[0]
    assert "rtol" in score["scope"] and "atol" in score["scope"]
    assert set(by_panel["score_recompute_rtol_atol"]["limit"]) == {1.0}
    assert "rtol" in score["limit_label"]
    for panel in ("bias_zeroed_single_sample", "dft_xai_cohort", "stdft_subset"):
        assert set(by_panel[panel]["limit"]) == {1e-3}
        assert "tolerância" in by_panel[panel].iloc[0]["limit_label"]
    layer_12_score = by_panel["score_recompute_rtol_atol"].iloc[-1]["value"]
    assert layer_12_score == pytest.approx(
        _score_error(12, 3) / (_SCORE_ATOL + _SCORE_RTOL * 0.75), rel=1e-9
    )


def test_conservation_record_names_every_scope_and_validation_sample(
    eng_tables, tmp_path
):
    record = {r.name: r for r in render_report_figures(eng_tables, tmp_path)}[
        "conservation_diagnostics"
    ]
    text = record.caption
    for needle in (
        "amostra única",
        _VALIDATION_KIND,
        "real-a",
        "coorte XAI",
        "subconjunto STDFT",
        "rtol",
        "atol",
    ):
        assert needle in text, needle
    assert "tolerância comum" not in text.lower()
    assert "razão" in record.units


# --- M1 / M2 / M7: scientific labels ---


def test_group_names_distinguish_true_class_from_predicted_class(eng_tables):
    frame = eng_tables.class_relevance
    true_names = set(frame.loc[frame["grouping"] == "y_true", "class_name"])
    predicted_names = set(frame.loc[frame["grouping"] == "prediction", "class_name"])
    assert true_names == {"real", "synthetic"}
    assert predicted_names == {"predicted_real", "predicted_synthetic"}


def test_captions_state_in_band_mass_and_synthetic_positive_class(
    eng_tables, tmp_path
):
    records = {r.name: r for r in render_report_figures(eng_tables, tmp_path)}
    for name in ("dft_relevance_heatmap", "class_relevance_by_layer"):
        text = f"{records[name].caption} {records[name].units}".lower()
        assert "em banda" in text, name
        assert "20" in text and "7900" in text, name
    fixed = records["fixed_threshold_by_layer"]
    text = f"{fixed.caption} {fixed.metric} {fixed.units}".lower()
    assert "sintético" in text and "positiva" in text
    assert "reais" in text


def test_loader_rejects_final_trace_without_eleven_transitions(tmp_path):
    root = write_scientific_result_root(tmp_path, transitions_per_sample=10)
    source = validate_result_directory(root)
    with pytest.raises(ValueError, match="11 transitions"):
        load_report_source(source)


def test_loader_attaches_stdft_examples_sorted_by_layer_class_then_sample(
    eng_tables,
):
    examples = eng_tables.stdft_examples
    assert len(examples) == 12 * 4
    order = list(
        zip(examples["layer"], examples["y_true"], examples["sample_id"])
    )
    assert order == sorted(order)
    first_layer = examples[examples["layer"] == 1]
    assert list(first_layer["sample_id"]) == [
        "real-a",
        "real-b",
        "spoof-a",
        "spoof-b",
    ]
    assert list(first_layer["class_name"]) == [
        "real",
        "real",
        "synthetic",
        "synthetic",
    ]
    row = eng_tables.stdft_examples.iloc[0]
    key = (row["model"], row["source"], row["target"], row["layer"], row["sample_id"])
    payload = eng_tables.stdft_payloads[key]
    assert payload["relevance"].shape == (5, 9)
    assert not payload["relevance"].flags.writeable


def test_single_model_and_language_declare_only_the_seven_figure_families(
    eng_tables,
):
    assert eng_tables.planned_figures == _EIGHT_FAMILIES
    assert not set(_COMPARISON_FAMILIES) & set(eng_tables.planned_figures)
    assert set(eng_tables.omitted_comparisons) == set(_COMPARISON_FAMILIES)
    assert eng_tables.omitted_comparisons["encoder_agreement"] == "single_model"
    for name in (
        "language_shift",
        "diagonal_vs_offdiagonal",
        "spectral_divergence",
    ):
        assert eng_tables.omitted_comparisons[name] == "single_language"


def test_render_creates_pdf_and_png_for_exactly_the_eight_families(
    eng_tables, tmp_path
):
    figures_dir = tmp_path / "figures"
    records = render_report_figures(eng_tables, figures_dir)

    expected = {
        f"{name}.{suffix}"
        for name in _EIGHT_FAMILIES
        for suffix in ("pdf", "png")
    }
    assert {path.name for path in figures_dir.iterdir()} == expected
    for path in figures_dir.iterdir():
        assert path.stat().st_size > 0
    assert tuple(record.name for record in records) == _EIGHT_FAMILIES
    for record in records:
        assert set(record.files) == {f"{record.name}.pdf", f"{record.name}.png"}
        for field in ("title", "caption", "metric", "units", "transformation"):
            assert getattr(record, field).strip()
        assert "hubert_base" in record.caption
        assert "eng" in record.caption
        assert record.models == ("hubert_base",)
        assert record.languages == ("eng",)


def test_render_pdf_has_vector_signature_and_png_is_decodable(
    eng_tables, tmp_path
):
    render_report_figures(eng_tables, tmp_path)
    assert (tmp_path / "performance_by_layer.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "stdft_examples.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_render_is_byte_deterministic(eng_tables, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    render_report_figures(eng_tables, first)
    render_report_figures(eng_tables, second)
    for name in _EIGHT_FAMILIES:
        for suffix in ("pdf", "png"):
            assert (first / f"{name}.{suffix}").read_bytes() == (
                second / f"{name}.{suffix}"
            ).read_bytes(), f"{name}.{suffix} differs between renders"


def test_table_row_order_is_deterministic_regardless_of_source_row_order(
    tmp_path, eng_tables
):
    root = write_scientific_result_root(tmp_path, shuffle=True)
    shuffled = build_report_tables(
        [load_report_source(validate_result_directory(root))]
    )
    for name in (
        "performance",
        "emergence",
        "band_relevance",
        "class_relevance",
        "transitions",
        "conservation",
        "cohort",
        "stdft_examples",
        "band_edges",
    ):
        pd.testing.assert_frame_equal(
            getattr(eng_tables, name), getattr(shuffled, name), obj=name
        )
    assert eng_tables.performance["layer"].is_monotonic_increasing


def test_loading_tables_and_rendering_leave_source_bytes_unchanged(tmp_path):
    root = write_scientific_result_root(tmp_path)
    before = _hash_tree(root)
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    render_report_figures(tables, tmp_path / "out")
    assert _hash_tree(root) == before


def test_loaded_source_preserves_task1_identity_and_immutability(eng_loaded):
    source = eng_loaded.source
    assert isinstance(source, ReportSource)
    assert source.identity == ExperimentIdentity(
        profile="hubert_base",
        languages=("eng",),
        protocol="layerwise_xai",
        scope="full",
    )
    with pytest.raises(TypeError):
        source.status["status"] = "failed"
    with pytest.raises(Exception):
        eng_loaded.source = source


def test_build_report_tables_rejects_empty_and_duplicate_sources(eng_loaded):
    with pytest.raises(ValueError, match="at least one"):
        build_report_tables([])
    with pytest.raises(ValueError, match="duplicate"):
        build_report_tables([eng_loaded, eng_loaded])


def test_multiple_languages_render_every_family_without_comparison_figures(
    tmp_path,
):
    root = write_scientific_result_root(tmp_path, languages=("eng", "por"))
    tables = build_report_tables(
        [load_report_source(validate_result_directory(root))]
    )
    assert tables.languages == ("eng", "por")
    assert set(tables.performance["language"]) == {"eng", "por"}
    assert len(tables.performance.groupby(["source", "target"])) == 4
    assert tables.omitted_comparisons["encoder_agreement"] == "single_model"
    for name in (
        "language_shift",
        "diagonal_vs_offdiagonal",
        "spectral_divergence",
    ):
        assert tables.omitted_comparisons[name] == "not_in_first_edition"
    records = render_report_figures(tables, tmp_path / "figs")
    assert tuple(record.name for record in records) == _EIGHT_FAMILIES
    assert len(list((tmp_path / "figs").iterdir())) == 16


def test_multiple_models_do_not_declare_encoder_agreement_figure(tmp_path):
    first = write_scientific_result_root(tmp_path, profile="hubert_base")
    second = write_scientific_result_root(tmp_path, profile="wavlm_base")
    tables = build_report_tables(
        [
            load_report_source(validate_result_directory(second)),
            load_report_source(validate_result_directory(first)),
        ]
    )
    assert tables.models == ("hubert_base", "wavlm_base")
    assert tables.omitted_comparisons["encoder_agreement"] == "not_in_first_edition"
    assert tables.planned_figures == _EIGHT_FAMILIES
    assert set(tables.performance["model"]) == {"hubert_base", "wavlm_base"}


# ---------------------------------------------------------------------------
# Task 3: LaTeX renderer, manifest, bundle, CLI and official docs
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_SECTION_TITLES = (
    "Resumo executivo",
    "Escopo e inventário dos experimentos",
    "Dados e protocolo de avaliação",
    "Desempenho ao longo das camadas",
    "Emergência estimada da decisão do detector",
    "Relevância em frequência ao longo das camadas",
    "Reorganização da decisão entre camadas",
    "Exemplos tempo-frequência STDFT selecionados",
    "Conservação e qualidade numérica",
    "Limitações atuais e próximos espaços de comparação",
    "Conclusão",
)
_BUNDLE_TABLES = (
    "performance_by_layer",
    "emergence_layers",
    "conservation_by_layer",
)
_REQUIRED_PACKAGES = (
    r"\usepackage[brazilian]{babel}",
    r"\usepackage{booktabs}",
    r"\usepackage{graphicx}",
    r"\usepackage{float}",
    r"\usepackage{subcaption}",
    r"\usepackage{xcolor}",
    r"\usepackage{hyperref}",
)


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _bundle_files(bundle: Path) -> list[str]:
    return sorted(
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    )


def _sections(tex: str) -> list[str]:
    return re.findall(r"\\section\{([^}]*)\}", tex)


@pytest.fixture(scope="module")
def eng_figures(eng_tables, tmp_path_factory):
    return render_report_figures(eng_tables, tmp_path_factory.mktemp("figures"))


@pytest.fixture(scope="module")
def eng_bundle(eng_root, tmp_path_factory):
    output = tmp_path_factory.mktemp("bundle") / "layerwise_xai_report"
    generate_report_bundle([eng_root], output)
    return output


@pytest.fixture(scope="module")
def eng_tex(eng_bundle):
    return _read_text(eng_bundle / "report.tex")


@pytest.fixture(scope="module")
def two_model_roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("two_models")
    return (
        write_scientific_result_root(base, profile="hubert_base"),
        write_scientific_result_root(base, profile="wavlm_base"),
    )


# --- LaTeX escaping ---


def test_escape_latex_escapes_the_plan_example():
    assert escape_latex("hubert_base & eng 100%") == r"hubert\_base \& eng 100\%"


def test_escape_latex_escapes_every_special_character_in_a_single_pass():
    assert escape_latex(r"\ $ # { } ~ ^") == (
        r"\textbackslash{} \$ \# \{ \} \textasciitilde{} \textasciicircum{}"
    )
    assert escape_latex(r"\_") == r"\textbackslash{}\_"
    assert escape_latex("plain text, ção") == "plain text, ção"


def test_escape_latex_maps_scientific_symbols_that_pdflatex_cannot_typeset():
    assert escape_latex("a \u2192 b \u2264 c \u2212 d \u0394") == (
        r"a $\rightarrow$ b $\leq$ c $-$ d $\Delta$"
    )


# --- manifest ---


def test_manifest_records_relative_sorted_hash_complete_provenance(
    eng_root, eng_loaded, eng_bundle
):
    manifest = build_report_manifest([eng_loaded], eng_bundle)

    assert set(manifest) == {"schema_version", "sources", "generated_files"}
    assert manifest["schema_version"] == 1
    (entry,) = manifest["sources"]
    source = eng_loaded.source
    assert entry["result_directory"] == eng_root.name
    assert entry["config_hash"] == "scientific-config"
    assert entry["profile"] == "hubert_base"
    assert entry["languages"] == ["eng"]
    assert entry["layers"] == list(range(1, 13))
    assert entry["aggregate_generation"] == source.aggregate_generation_id
    assert list(entry["layer_xai_generations"]) == sorted(
        entry["layer_xai_generations"]
    )
    assert len(entry["layer_xai_generations"]) == 12
    assert entry["layer_xai_generations"][
        "hubert_base/layer_01/eng->eng"
    ] == source.layer_xai_generations[("hubert_base", 1, "eng", "eng")].name
    assert entry["final_trace_generations"] == {
        "hubert_base/eng->eng": source.final_trace_generations[
            ("hubert_base", "eng", "eng")
        ].name
    }
    assert entry["inputs"]["eng"]["role_counts"] == _ROLE_COUNTS

    consumed = entry["consumed_artifacts"]
    paths = [item["path"] for item in consumed]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert {"run_status.json", "execution_plan.json", "aggregates/active.json"} <= set(
        paths
    )
    assert any(path.endswith("/metrics.json") for path in paths)
    assert any(path.startswith(".state/") for path in paths)
    for item in consumed:
        assert set(item) == {"path", "sha256"}
        assert not item["path"].startswith("/") and ":" not in item["path"]
        assert "\\" not in item["path"] and ".." not in item["path"].split("/")
        assert item["sha256"] == _sha256_file(eng_root / item["path"])


def test_manifest_generated_files_describe_the_bundle_without_itself(eng_bundle):
    manifest = json.loads(_read_text(eng_bundle / "report_manifest.json"))
    listed = manifest["generated_files"]
    paths = [item["path"] for item in listed]

    assert paths == sorted(paths)
    assert "report_manifest.json" not in paths
    assert set(paths) == set(_bundle_files(eng_bundle)) - {"report_manifest.json"}
    assert {
        "report.tex",
        "build_local.ps1",
        "figures/performance_by_layer.pdf",
        "figures/performance_by_layer.png",
        "tables/performance_by_layer.csv",
        "tables/performance_by_layer.tex",
    } <= set(paths)
    for item in listed:
        assert set(item) == {"path", "sha256"}
        assert item["sha256"] == _sha256_file(eng_bundle / item["path"])


def test_manifest_json_is_canonical_utf8_without_machine_paths(
    eng_root, eng_bundle
):
    raw = (eng_bundle / "report_manifest.json").read_bytes()
    text = raw.decode("utf-8")
    assert b"\r" not in raw and text.endswith("}\n")
    assert text == (
        json.dumps(
            json.loads(text),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    assert str(eng_root) not in text
    assert str(eng_root.parent) not in text.replace("\\\\", "\\")


# --- bundle ---


def test_bundle_contains_the_portable_layout(eng_bundle):
    files = set(_bundle_files(eng_bundle))
    assert {"report.tex", "report_manifest.json", "build_local.ps1"} <= files
    for name in _EIGHT_FAMILIES:
        assert f"figures/{name}.pdf" in files
        assert f"figures/{name}.png" in files
    for name in _BUNDLE_TABLES:
        assert f"tables/{name}.tex" in files
    assert "tables/performance_by_layer.csv" in files
    assert all(
        path.split("/")[0] in {"figures", "tables"} or "/" not in path
        for path in files
    )


def test_bundle_performance_table_matches_the_loaded_values(
    eng_bundle, eng_tables
):
    frame = pd.read_csv(eng_bundle / "tables" / "performance_by_layer.csv")
    expected = eng_tables.performance
    assert list(frame["layer"]) == list(expected["layer"])
    assert list(frame["model"]) == list(expected["model"])
    np.testing.assert_allclose(frame["roc_auc"], expected["roc_auc"], atol=1e-12)
    tex = _read_text(eng_bundle / "tables" / "performance_by_layer.tex")
    for command in (r"\toprule", r"\midrule", r"\bottomrule"):
        assert command in tex
    assert r"hubert\_base" in tex and "hubert_base" not in tex.replace(
        r"hubert\_base", ""
    )
    best = expected.loc[expected["roc_auc"].idxmax()]
    assert f"{best['roc_auc']:.3f}".replace(".", ",") in tex


def test_bundle_persists_transfer_reduction_and_xai_association_tables(
    eng_bundle,
):
    transfer = pd.read_csv(eng_bundle / "tables" / "transfer_selected_layers.csv")
    assert list(transfer.columns) == [
        "model",
        "source",
        "target",
        "selected_layer",
        "roc_auc",
        "mcc",
    ]
    association = pd.read_csv(
        eng_bundle / "tables" / "xai_performance_association.csv"
    )
    assert {
        "model",
        "source",
        "target",
        "n_layers",
        "concentration_metric",
        "association_method",
        "rho",
        "status",
    } == set(association.columns)
    assert set(association["n_layers"]) == {12}
    assert "figures/transfer_performance_heatmaps.pdf" in _bundle_files(eng_bundle)
    assert "figures/transfer_performance_heatmaps.png" in _bundle_files(eng_bundle)


def test_transfer_heatmap_caption_states_shared_layer_reduction(eng_figures):
    record = next(
        item for item in eng_figures if item.name == "transfer_performance_heatmaps"
    )
    assert "ROC-AUC máxima" in record.transformation
    assert "mesma camada selecionada" in record.transformation


def test_bundle_is_byte_deterministic_across_regeneration_and_locations(
    eng_root, eng_bundle, tmp_path
):
    elsewhere = tmp_path / "another_name"
    generate_report_bundle([eng_root], elsewhere)
    first = {p: _sha256_file(eng_bundle / p) for p in _bundle_files(eng_bundle)}
    second = {p: _sha256_file(elsewhere / p) for p in _bundle_files(elsewhere)}
    assert first == second


def test_bundle_generation_leaves_every_source_byte_unchanged(eng_root, tmp_path):
    before = _hash_tree(eng_root)
    generate_report_bundle([eng_root], tmp_path / "report")
    assert _hash_tree(eng_root) == before


def test_bundle_regeneration_in_place_replaces_the_previous_bundle(
    eng_root, tmp_path
):
    output = tmp_path / "report"
    generate_report_bundle([eng_root], output)
    first = {p: _sha256_file(output / p) for p in _bundle_files(output)}
    generate_report_bundle([eng_root], output)
    assert {p: _sha256_file(output / p) for p in _bundle_files(output)} == first


def test_bundle_refuses_unrelated_or_overlapping_output_directories(
    eng_root, tmp_path
):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="output"):
        generate_report_bundle([eng_root], foreign)
    assert (foreign / "notes.txt").read_text(encoding="utf-8") == "keep me"

    before = _hash_tree(eng_root)
    with pytest.raises(ValueError, match="output"):
        generate_report_bundle([eng_root], eng_root / "report")
    with pytest.raises(ValueError, match="output"):
        generate_report_bundle([eng_root], eng_root.parent)
    assert _hash_tree(eng_root) == before
    assert not (eng_root / "report").exists()


def test_bundle_rejects_duplicate_identities_before_writing(eng_root, tmp_path):
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="duplicate"):
        generate_report_bundle([eng_root, eng_root], output)
    assert not output.exists()


def test_bundle_rejects_an_invalid_source_before_writing(tmp_path):
    root = write_scientific_result_root(tmp_path)
    (root / "run_status.json").write_text(
        json.dumps({"schema_version": 1, "status": "failed"}), encoding="utf-8"
    )
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="completed non-dry-run"):
        generate_report_bundle([root], output)
    assert not output.exists()


def test_multi_source_bundle_is_independent_of_source_order(
    two_model_roots, tmp_path
):
    first, second = two_model_roots
    forward = tmp_path / "forward"
    backward = tmp_path / "backward"
    generate_report_bundle([first, second], forward)
    generate_report_bundle([second, first], backward)

    assert {p: _sha256_file(forward / p) for p in _bundle_files(forward)} == {
        p: _sha256_file(backward / p) for p in _bundle_files(backward)
    }
    manifest = json.loads(_read_text(forward / "report_manifest.json"))
    assert [item["profile"] for item in manifest["sources"]] == [
        "hubert_base",
        "wavlm_base",
    ]
    tex = _read_text(forward / "report.tex")
    assert "Estudo de caso" not in tex
    assert "HuBERT Base" in tex and "WavLM Base" in tex
    assert "WavLM Base+" not in tex
    assert "não estão disponíveis" in tex
    assert _sections(tex) == list(_SECTION_TITLES)


# --- scientific text ---


def test_tex_is_a_portuguese_a4_document_with_the_planned_packages(eng_tex):
    assert eng_tex.startswith("\\documentclass")
    assert "a4paper" in eng_tex.splitlines()[0]
    for package in _REQUIRED_PACKAGES:
        assert package in eng_tex
    assert "\r" not in eng_tex
    assert eng_tex.rstrip().endswith(r"\end{document}")


def test_tex_labels_the_report_as_a_single_model_single_language_case_study(
    eng_tex,
):
    assert "Estudo de caso: HuBERT Base em inglês" in eng_tex
    assert (
        "afirmações comparativas entre modelos ou entre idiomas não estão "
        "disponíveis"
    ) in eng_tex
    assert _sections(eng_tex) == list(_SECTION_TITLES)


def test_tex_distinguishes_total_selected_audios_from_the_test_subset(eng_tex):
    total = sum(_ROLE_COUNTS.values())
    assert f"{total} áudios selecionados no total" in eng_tex
    assert f"treino {_ROLE_COUNTS['train']}" in eng_tex
    assert f"calibração {_ROLE_COUNTS['calibration']}" in eng_tex
    assert f"subconjunto de teste ({_ROLE_COUNTS['test']} áudios)" in eng_tex
    assert "coorte XAI fixa" in eng_tex and "4 áudios (2 reais e 2 sintéticos)" in eng_tex


def test_tex_omits_role_count_claims_when_the_plan_records_none(tmp_path):
    root = write_scientific_result_root(tmp_path, role_counts=None)
    output = tmp_path / "report"
    generate_report_bundle([root], output)
    tex = _read_text(output / "report.tex")
    assert "áudios selecionados no total" not in tex
    assert "subconjunto de teste (" not in tex
    assert "não registradas" in tex


def test_tex_includes_every_generated_figure_and_table_that_exists(
    eng_bundle, eng_tex
):
    figures = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", eng_tex)
    assert sorted(Path(item).stem for item in figures) == sorted(_EIGHT_FAMILIES)
    for item in figures:
        assert item.startswith("figures/") and (eng_bundle / item).is_file()
    inputs = re.findall(r"\\input\{([^}]*)\}", eng_tex)
    assert {Path(item).stem for item in inputs} >= set(_BUNDLE_TABLES)
    for item in inputs:
        assert (eng_bundle / item).is_file(), item
    for name in _EIGHT_FAMILIES:
        assert rf"\label{{fig:{name}}}" in eng_tex


def test_tex_scientific_statements_follow_the_loaded_values(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    performance = eng_tables.performance.copy()
    performance.loc[performance["layer"] == 7, "roc_auc"] = 0.9991
    performance.loc[performance["layer"] != 7, "roc_auc"] = 0.6
    modified = dataclasses.replace(eng_tables, performance=performance)
    path = tmp_path / "report.tex"

    write_report_tex(path, modified, eng_figures, [eng_loaded])

    tex = _read_text(path)
    assert "a maior ROC-AUC (0,999) ocorre na camada 7" in tex
    baseline = tmp_path / "baseline.tex"
    write_report_tex(baseline, eng_tables, eng_figures, [eng_loaded])
    assert "a maior ROC-AUC (0,999) ocorre na camada 7" not in _read_text(baseline)


def test_transfer_reduction_selects_layer_by_roc_auc_and_reuses_its_mcc(
    eng_tables,
):
    performance = eng_tables.performance.copy()
    performance.loc[:, "roc_auc"] = 0.60
    performance.loc[:, "mcc"] = 0.10
    performance.loc[performance["layer"] == 4, ["roc_auc", "mcc"]] = [0.91, 0.22]
    performance.loc[performance["layer"] == 9, ["roc_auc", "mcc"]] = [0.88, 0.97]

    selected = report_module.build_transfer_selection(performance)

    assert list(selected.columns) == [
        "model",
        "source",
        "target",
        "selected_layer",
        "roc_auc",
        "mcc",
    ]
    row = selected.iloc[0]
    assert int(row["selected_layer"]) == 4
    assert row["roc_auc"] == pytest.approx(0.91)
    assert row["mcc"] == pytest.approx(0.22)


def test_xai_performance_association_uses_twelve_layers_and_marks_gaps_unavailable(
    eng_tables,
):
    relevance = eng_tables.band_relevance.copy()
    absolute = relevance["measure"] == "absolute_normalized"
    relevance.loc[absolute, "mean"] = relevance.loc[absolute, "layer"] / 12
    available = report_module.build_xai_performance_associations(
        eng_tables.performance, relevance
    )
    row = available.iloc[0]
    assert int(row["n_layers"]) == 12
    assert row["concentration_metric"] == "maximum_mean_band_mass"
    assert row["association_method"] == "spearman_rank_correlation"
    assert row["status"] == "available"
    assert np.isfinite(row["rho"])

    incomplete = report_module.build_xai_performance_associations(
        eng_tables.performance[eng_tables.performance["layer"] != 12], relevance
    )
    row = incomplete.iloc[0]
    assert int(row["n_layers"]) == 11
    assert row["status"] == "unavailable_requires_12_layers"
    assert pd.isna(row["rho"])


def test_report_opens_with_summary_closes_with_conclusion_and_separates_transfer(
    eng_tex,
):
    sections = _sections(eng_tex)
    assert sections[0] == "Resumo executivo"
    assert sections[-1] == "Conclusão"
    assert "Associação descritiva entre XAI e desempenho" in eng_tex
    assert "n=12" in eng_tex
    assert "sem interpretação causal" in eng_tex
    assert r"\subsection{Ranking sem limiar e calibração do limiar}" in eng_tex
    assert "ROC-AUC" in eng_tex and "MCC, TPR e FPR" in eng_tex


def test_consolidated_scope_does_not_claim_comparisons_are_unavailable(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    languages = ("eng", "por", "zho")

    def expand(frame):
        copies = []
        for source in languages:
            for target in languages:
                copy = frame.copy()
                copy["source"] = source
                copy["target"] = target
                if "language" in copy:
                    copy["language"] = target
                copies.append(copy)
        return pd.concat(copies, ignore_index=True)

    tables = dataclasses.replace(
        eng_tables,
        languages=languages,
        performance=expand(eng_tables.performance),
        band_relevance=expand(eng_tables.band_relevance),
        transfer_selection=report_module.build_transfer_selection(
            expand(eng_tables.performance)
        ),
        xai_performance_association=report_module.build_xai_performance_associations(
            expand(eng_tables.performance), expand(eng_tables.band_relevance)
        ),
    )
    path = tmp_path / "report.tex"
    figures = [
        item
        for item in eng_figures
        if item.name
        in {
            "performance_by_layer",
            "fixed_threshold_by_layer",
            "transfer_performance_heatmaps",
        }
    ]
    write_report_tex(path, tables, figures, [eng_loaded])
    tex = _read_text(path)

    assert "afirmações comparativas entre modelos ou entre idiomas não estão disponíveis" not in tex
    assert "transferência sob mudança de corpus/idioma" in tex
    assert "causado pelo idioma" not in tex
    assert "não identificam um efeito causal do idioma" in tex


def test_tex_omits_a_section_when_its_figure_is_unavailable(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    figures = [record for record in eng_figures if record.name != "stdft_examples"]
    path = tmp_path / "report.tex"

    write_report_tex(path, eng_tables, figures, [eng_loaded])

    tex = _read_text(path)
    assert "stdft_examples" not in tex
    assert _sections(tex) == [
        title for title in _SECTION_TITLES if "STDFT" not in title
    ]


def test_tex_escapes_text_that_comes_from_the_result_artifacts(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    conservation = eng_tables.conservation.copy()
    conservation["validation_sample_id"] = "real_a&50%"
    hostile = dataclasses.replace(eng_tables, conservation=conservation)
    path = tmp_path / "report.tex"

    write_report_tex(path, hostile, eng_figures, [eng_loaded])

    tex = _read_text(path).replace(r"\allowbreak{}", "")
    assert r"real\_a\&50\%" in tex
    assert "real_a&50%" not in tex
    assert r"\texttt{hubert\_base}" in tex
    assert "\\texttt{hubert_base}" not in tex


def test_tex_states_each_conservation_scope_and_limit(eng_tex):
    eng_tex = eng_tex.replace(r"\allowbreak{}", "")
    for needle in (
        "uma única amostra",
        "bias_zeroed_model_rule_check".replace("_", r"\_"),
        "coorte XAI",
        "subconjunto STDFT",
        "rtol",
        "atol",
    ):
        assert needle in eng_tex, needle
    assert "tolerância comum" not in eng_tex


def test_tex_lists_the_unavailable_comparison_slots(eng_tex):
    assert "acordo entre encoders" in eng_tex
    assert "apenas um modelo" in eng_tex
    assert "apenas um idioma" in eng_tex


def test_write_report_tables_emits_csv_and_tex_for_every_table(
    eng_tables, tmp_path
):
    written = write_report_tables(eng_tables, tmp_path / "tables")
    names = {Path(item).name for item in written}
    assert "performance_by_layer.csv" in names
    assert "performance_by_layer.tex" in names
    for name in _BUNDLE_TABLES:
        assert f"{name}.tex" in names
    assert names == {path.name for path in (tmp_path / "tables").iterdir()}
    assert all(
        (tmp_path / "tables" / Path(item).name).stat().st_size > 0
        for item in written
    )


# --- local build script ---


def test_build_local_script_compiles_twice_and_fails_without_a_pdf(eng_bundle):
    script = _read_text(eng_bundle / "build_local.ps1")
    expected_lines = (
        '$ErrorActionPreference = "Stop"',
        "pdflatex -interaction=nonstopmode -halt-on-error report.tex",
        'if (-not (Test-Path "report.pdf")) {',
        'throw "report.pdf was not produced"',
    )
    cursor = 0
    for line in expected_lines:
        cursor = script.index(line, cursor) + len(line)
    assert script.count(
        "pdflatex -interaction=nonstopmode -halt-on-error report.tex"
    ) == 2
    assert "$LASTEXITCODE" in script
    assert "\r" not in script


@pytest.mark.skipif(
    shutil.which("powershell") is None and shutil.which("pwsh") is None,
    reason="PowerShell is not available",
)
def test_build_local_script_is_valid_powershell(eng_bundle):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    command = (
        "$errors = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{eng_bundle / 'build_local.ps1'}', [ref]$null, [ref]$errors); "
        "if ($errors.Count -gt 0) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- CLI ---


def _cli_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SCRIPTS)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def test_cli_generates_a_bundle_from_repeated_result_options(
    two_model_roots, tmp_path, capsys
):
    first, second = two_model_roots
    output = tmp_path / "cli_report"

    code = main(
        ["--result", str(first), "--result", str(second), "--output", str(output)]
    )

    assert code == 0
    assert (output / "report.tex").is_file()
    manifest = json.loads(_read_text(output / "report_manifest.json"))
    assert len(manifest["sources"]) == 2
    assert str(output) in capsys.readouterr().out


def test_cli_rejects_duplicate_identities_without_writing(
    eng_root, tmp_path, capsys
):
    output = tmp_path / "cli_report"

    code = main(["--result", str(eng_root), "--result", str(eng_root), "--output", str(output)])

    assert code != 0
    assert "duplicate" in capsys.readouterr().err
    assert not output.exists()


def test_cli_never_scans_directories_and_requires_explicit_arguments(
    eng_root, tmp_path
):
    with pytest.raises(SystemExit) as no_result:
        main(["--output", str(tmp_path / "out")])
    assert no_result.value.code == 2
    with pytest.raises(SystemExit) as no_output:
        main(["--result", str(eng_root)])
    assert no_output.value.code == 2
    assert not (tmp_path / "out").exists()


def test_cli_reports_validation_failures_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "hubert_base__eng__layerwise_xai__full"
    missing.mkdir()
    output = tmp_path / "out"

    code = main(["--result", str(missing), "--output", str(output)])

    assert code == 1
    assert capsys.readouterr().err.startswith("error:")
    assert not output.exists()


def test_cli_module_entry_point_runs_without_a_gpu_or_encoder_import(
    eng_root, tmp_path
):
    output = tmp_path / "subprocess_report"
    script = (
        "import sys\n"
        "from brspeech_xai.layerwise_report import main\n"
        f"code = main(['--result', {str(eng_root)!r}, '--output', {str(output)!r}])\n"
        "heavy = sorted(n for n in ('torch', 'transformers') if n in sys.modules)\n"
        "print('RESULT', code, heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_cli_env(),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "RESULT 0 []" in result.stdout
    assert (output / "report.tex").is_file()

    help_result = subprocess.run(
        [sys.executable, "-m", "brspeech_xai.layerwise_report", "--help"],
        capture_output=True,
        text=True,
        env=_cli_env(),
        cwd=tmp_path,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--result" in help_result.stdout and "--output" in help_result.stdout


# --- official documentation ---


@pytest.mark.parametrize("relative", ["README.md", "scripts/README.md"])
def test_official_docs_describe_naming_workflow_and_read_only_guarantee(relative):
    text = _read_text(_REPO / relative)
    assert "<model>__<languages>__layerwise_xai__<scope>" in text
    assert "brspeech_xai.layerwise_report" in text
    assert "build_local.ps1" in text
    assert "--result" in text
    assert "read-only" in text.lower() or "somente leitura" in text.lower()
    assert "GPU" in text
    assert "layerwise-suite-eng-por-pilot" not in text
    assert "layerwise-suite-pilot" not in text


@pytest.mark.parametrize("relative", ["README.md", "scripts/README.md"])
def test_official_docs_list_all_seven_supported_language_combinations(relative):
    raw = (_REPO / relative).read_bytes()
    text = raw.decode("utf-8")
    assert raw.endswith(b"\n")
    section = text.split("layerwise_xai__<scope>", 1)[1]
    for segment, _ in _ALL_LANGUAGE_SUBSETS:
        assert f"`{segment}`" in section, segment
    assert "`eng-zho`" in section


@pytest.mark.parametrize(
    ("relative", "sentence"),
    [
        (
            "README.md",
            "is the\nEnglish-only case study that is already complete; it is separate "
            "from the future\ntrilingual executions above",
        ),
        (
            "scripts/README.md",
            "é o estudo\nde caso somente em inglês, já concluído, e é separado das "
            "futuras execuções\ntrilíngues acima",
        ),
    ],
)
def test_official_docs_separate_the_hubert_eng_case_study_from_trilingual_runs(
    relative, sentence
):
    text = _read_text(_REPO / relative)
    assert "hubert_base__eng__layerwise_xai__full" in text
    assert sentence in text


_CANONICAL_FULL_OUTPUTS = (
    "hubert_base__eng-por-zho__layerwise_xai__full",
    "wavlm_base__eng-por-zho__layerwise_xai__full",
    "wav2vec2_base__eng-por-zho__layerwise_xai__full",
)


@pytest.mark.parametrize("relative", ["README.md", "scripts/README.md"])
def test_official_docs_run_each_profile_in_its_own_canonical_output(relative):
    text = _read_text(_REPO / relative)

    assert "layerwise-suite" not in text
    assert not re.search(
        r"--profiles\s+hubert_base\s+wavlm_base|--profiles\s+\S+\s+\S+\s+wav2vec2_base",
        text,
    )
    for name in _CANONICAL_FULL_OUTPUTS:
        assert f"--output /mnt/results/{name}" in text, name
    full_runs = re.findall(
        r"--profiles (\w+) \\\n\s+--output /mnt/results/(\S+) \\\n\s+--xai-per-class 25\n",
        text,
    )
    assert full_runs == [
        ("hubert_base", _CANONICAL_FULL_OUTPUTS[0]),
        ("wavlm_base", _CANONICAL_FULL_OUTPUTS[1]),
        ("wav2vec2_base", _CANONICAL_FULL_OUTPUTS[2]),
    ]
    for output in re.findall(r"--output (/mnt/results/\S+)", text):
        directory = output.rsplit("/", 1)[1]
        if directory != "layerwise_xai_report":
            assert re.fullmatch(
                r"(hubert_base|wavlm_base|wav2vec2_base)__"
                r"(eng|por|zho|eng-por|eng-zho|por-zho|eng-por-zho)__layerwise_xai__"
                r"(pilot|full)",
                directory,
            ), output


@pytest.mark.parametrize("relative", ["README.md", "scripts/README.md"])
def test_official_docs_name_the_pretrained_only_wav2vec2_checkpoint(relative):
    text = _read_text(_REPO / relative)
    assert "facebook/wav2vec2-base" in text
    assert any(
        statement in text
        for statement in (
            "does not use the ASR-fine-tuned",
            "não faz parte deste protocolo",
        )
    )


# ---------------------------------------------------------------------------
# Task 3 review fixes (I1-I4, M2-M5)
# ---------------------------------------------------------------------------


def _pdflatex_safe(text: str) -> bool:
    return all(
        " " <= char <= "~"
        or char == "\n"
        or ("\u00c0" <= char <= "\u00ff" and char.isalpha())
        for char in text
    )


def _edit_manifest(output: Path, mutate) -> None:
    path = output / "report_manifest.json"
    payload = json.loads(_read_text(path))
    mutate(payload)
    path.write_bytes(json.dumps(payload).encode("utf-8"))


def _copied_bundle(eng_bundle: Path, tmp_path: Path) -> Path:
    output = tmp_path / "bundle"
    shutil.copytree(eng_bundle, output)
    return output


# --- I1: previous-manifest paths must never leave the output directory ---


@pytest.mark.parametrize(
    "kind",
    [
        "parent",
        "nested_parent",
        "posix_absolute",
        "drive_forward",
        "drive_backslash",
        "unc_forward",
        "unc_backslash",
        "backslash_parent",
        "empty",
        "dot",
        "non_string",
    ],
)
def test_regeneration_rejects_previous_manifest_paths_that_escape_the_output(
    kind, eng_root, eng_bundle, tmp_path
):
    output = _copied_bundle(eng_bundle, tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    drive_forward = victim.as_posix() if victim.drive else "C:/nonexistent/victim.txt"
    drive_backslash = str(victim) if victim.drive else "C:\\nonexistent\\victim.txt"
    bad = {
        "parent": "../victim.txt",
        "nested_parent": "figures/../../victim.txt",
        "posix_absolute": "/victim.txt",
        "drive_forward": drive_forward,
        "drive_backslash": drive_backslash,
        "unc_forward": "//server/share/victim.txt",
        "unc_backslash": "\\\\server\\share\\victim.txt",
        "backslash_parent": "figures\\..\\..\\victim.txt",
        "empty": "",
        "dot": ".",
        "non_string": 5,
    }[kind]
    _edit_manifest(
        output,
        lambda payload: payload["generated_files"].append(
            {"path": bad, "sha256": "0" * 64}
        ),
    )
    before = _hash_tree(output)

    with pytest.raises(ValueError, match="previous report manifest"):
        generate_report_bundle([eng_root], output)

    assert victim.read_text(encoding="utf-8") == "keep me"
    assert _hash_tree(output) == before


def test_regeneration_rejects_a_listed_path_that_resolves_outside_via_a_link(
    eng_root, eng_bundle, tmp_path
):
    output = _copied_bundle(eng_bundle, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("keep me", encoding="utf-8")
    try:
        os.symlink(outside, output / "linked", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    _edit_manifest(
        output,
        lambda payload: payload["generated_files"].append(
            {"path": "linked/victim.txt", "sha256": "0" * 64}
        ),
    )
    with pytest.raises(ValueError, match="previous report manifest"):
        generate_report_bundle([eng_root], output)
    assert (outside / "victim.txt").read_text(encoding="utf-8") == "keep me"


# --- I2: generated_files holds only what the current generation produced ---


def test_manifest_excludes_compilation_leftovers_and_foreign_files(
    eng_root, tmp_path
):
    output = tmp_path / "report"
    generate_report_bundle([eng_root], output)
    produced = [
        item["path"]
        for item in json.loads(_read_text(output / "report_manifest.json"))[
            "generated_files"
        ]
    ]
    leftovers = (
        "report.aux",
        "report.log",
        "report.out",
        "report.toc",
        "report.pdf",
        "report.synctex.gz",
        "figures/leftover.png",
        "tables/notes.csv",
        "notes.txt",
    )
    for name in leftovers:
        (output / name).write_bytes(b"temporary compilation output")

    generate_report_bundle([eng_root], output)

    manifest = json.loads(_read_text(output / "report_manifest.json"))
    listed = [item["path"] for item in manifest["generated_files"]]
    assert listed == produced
    assert not set(leftovers) & set(listed)


def test_manifest_default_scan_ignores_compilation_leftovers(
    eng_loaded, eng_bundle, tmp_path
):
    output = _copied_bundle(eng_bundle, tmp_path)
    expected = [
        item["path"]
        for item in json.loads(_read_text(output / "report_manifest.json"))[
            "generated_files"
        ]
    ]
    for name in ("report.aux", "report.log", "report.out", "report.pdf"):
        (output / name).write_bytes(b"temporary")

    manifest = build_report_manifest([eng_loaded], output)

    assert [item["path"] for item in manifest["generated_files"]] == expected


def test_manifest_accepts_only_existing_relative_generated_files(
    eng_loaded, eng_bundle
):
    with pytest.raises(ValueError, match="generated file"):
        build_report_manifest(
            [eng_loaded], eng_bundle, generated_files=["../outside.txt"]
        )
    with pytest.raises(ValueError, match="generated file"):
        build_report_manifest(
            [eng_loaded], eng_bundle, generated_files=["missing.tex"]
        )


# --- I3: no empty itemize environments ---


def _empty_itemize(tex: str) -> bool:
    return re.search(r"\\begin\{itemize\}\s*\\end\{itemize\}", tex) is not None


def test_generated_tex_has_no_empty_itemize(eng_tex):
    assert not _empty_itemize(eng_tex)


def test_protocol_omits_the_list_when_there_is_nothing_to_list(
    eng_tables, eng_figures, tmp_path
):
    without_cohort = dataclasses.replace(
        eng_tables, cohort=eng_tables.cohort.iloc[0:0]
    )
    path = tmp_path / "report.tex"

    write_report_tex(path, without_cohort, eng_figures, [])

    tex = _read_text(path)
    assert not _empty_itemize(tex)
    assert "Contagens de áudios declaradas" not in tex
    assert r"\midrule" + "\n" + r"\bottomrule" not in tex


def test_limitations_omit_the_comparison_slots_when_none_are_pending(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    complete = dataclasses.replace(eng_tables, omitted_comparisons={})
    path = tmp_path / "report.tex"

    write_report_tex(path, complete, eng_figures, [eng_loaded])

    tex = _read_text(path)
    assert not _empty_itemize(tex)
    assert "Espaços de comparação reservados" not in tex
    assert _sections(tex) == list(_SECTION_TITLES)


# --- I4: no unsupported character may reach pdflatex ---


def test_escape_latex_maps_common_typographic_characters_safely():
    assert escape_latex("a \u2013 b \u2014 c") == "a -- b --- c"
    assert escape_latex("\u00b15 \u00d7 3") == r"$\pm$5 $\times$ 3"
    assert escape_latex("\u201cx\u201d \u2018y\u2019") == "``x'' `y'"
    assert escape_latex("a\u00a0b\u2026") == r"a~b\ldots{}"
    assert escape_latex("ação é ü") == "ação é ü"


def test_escape_latex_replaces_unsupported_characters_with_ascii_codepoints():
    assert escape_latex("\u4e2d\u6587") == "[U+4E2D][U+6587]"
    assert escape_latex("a\U0001f4a5") == "a[U+1F4A5]"
    assert escape_latex("\x00x\x7f") == "[U+0000]x[U+007F]"
    assert escape_latex("a\tb\r\nc") == "a b  c"


def test_escape_latex_output_is_always_pdflatex_safe():
    codepoints = list(range(0x3000)) + [0x4E2D, 0x1F4A5, 0xD800, 0x10FFFF]
    for codepoint in codepoints:
        escaped = escape_latex(chr(codepoint))
        assert _pdflatex_safe(escaped), hex(codepoint)


def test_tex_neutralises_unsupported_characters_from_the_artifacts(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    conservation = eng_tables.conservation.copy()
    conservation["validation_sample_id"] = "\u771f_a\U0001f4a5"
    hostile = dataclasses.replace(eng_tables, conservation=conservation)
    path = tmp_path / "report.tex"

    write_report_tex(path, hostile, eng_figures, [eng_loaded])

    tex = _read_text(path).replace(r"\allowbreak{}", "")
    assert _pdflatex_safe(tex)
    assert r"[U+771F]\_a[U+1F4A5]" in tex


def test_generated_bundle_text_is_pdflatex_safe(eng_bundle):
    assert _pdflatex_safe(_read_text(eng_bundle / "report.tex"))
    for name in _BUNDLE_TABLES:
        assert _pdflatex_safe(_read_text(eng_bundle / "tables" / f"{name}.tex"))


def test_write_report_tex_fails_closed_on_unsafe_static_text(
    eng_tables, eng_loaded, eng_figures, tmp_path, monkeypatch
):
    titles = list(_SECTION_TITLES)
    titles[1] = "Dados \u4e2d"
    monkeypatch.setattr(report_module, "_SECTION_TITLES_PT", tuple(titles))
    path = tmp_path / "report.tex"

    with pytest.raises(ValueError, match="pdflatex"):
        write_report_tex(path, eng_tables, eng_figures, [eng_loaded])

    assert not path.exists()


# --- M2: the stale PDF is removed fail-closed (behavioural, Windows) ---

_WINDOWS_POWERSHELL = (
    os.name == "nt"
    and (shutil.which("powershell") is not None or shutil.which("pwsh") is not None)
)


def _run_build_script(
    tmp_path: Path,
    eng_bundle: Path,
    pdflatex_body: str,
    *,
    stale_pdf: bool = False,
    lock_pdf: bool = False,
) -> tuple[subprocess.CompletedProcess, Path]:
    work = tmp_path / "work"
    work.mkdir()
    script = work / "build_local.ps1"
    script.write_bytes((eng_bundle / "build_local.ps1").read_bytes())
    (work / "pdflatex.cmd").write_text(
        "@echo off\r\n"
        "echo %*>> pdflatex_calls.txt\r\n"
        f"{pdflatex_body}\r\n",
        encoding="ascii",
    )
    if stale_pdf or lock_pdf:
        (work / "report.pdf").write_bytes(b"STALE")
    lock = (
        f"$fs = [System.IO.File]::Open('{work / 'report.pdf'}','Open','ReadWrite','None'); "
        if lock_pdf
        else ""
    )
    command = (
        f"{lock}try {{ & '{script}'; exit 0 }} "
        "catch { Write-Output $_.Exception.Message; exit 3 }"
    )
    shell = shutil.which("powershell") or shutil.which("pwsh")
    env = dict(os.environ)
    env["PATH"] = str(work) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    return result, work


@pytest.mark.skipif(not _WINDOWS_POWERSHELL, reason="needs Windows PowerShell")
def test_build_local_script_succeeds_with_two_passes_and_a_fresh_pdf(
    tmp_path, eng_bundle
):
    result, work = _run_build_script(
        tmp_path, eng_bundle, "echo PDF> report.pdf", stale_pdf=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (work / "pdflatex_calls.txt").read_text().count("report.tex") == 2
    assert (work / "report.pdf").read_bytes().startswith(b"PDF")


@pytest.mark.skipif(not _WINDOWS_POWERSHELL, reason="needs Windows PowerShell")
def test_build_local_script_fails_when_pdflatex_fails(tmp_path, eng_bundle):
    result, work = _run_build_script(tmp_path, eng_bundle, "exit /b 1")
    assert result.returncode == 3
    assert "first pass" in result.stdout
    assert (work / "pdflatex_calls.txt").read_text().count("report.tex") == 1


@pytest.mark.skipif(not _WINDOWS_POWERSHELL, reason="needs Windows PowerShell")
def test_build_local_script_never_accepts_a_stale_pdf(tmp_path, eng_bundle):
    result, _ = _run_build_script(tmp_path, eng_bundle, "exit /b 0", stale_pdf=True)
    assert result.returncode == 3
    assert "report.pdf was not produced" in result.stdout


@pytest.mark.skipif(not _WINDOWS_POWERSHELL, reason="needs Windows PowerShell")
def test_build_local_script_fails_closed_when_the_old_pdf_cannot_be_removed(
    tmp_path, eng_bundle
):
    result, work = _run_build_script(
        tmp_path, eng_bundle, "echo PDF> report.pdf", lock_pdf=True
    )
    assert result.returncode == 3
    assert not (work / "pdflatex_calls.txt").exists()


def test_build_local_script_does_not_silence_the_stale_pdf_removal(eng_bundle):
    script = _read_text(eng_bundle / "build_local.ps1")
    assert "SilentlyContinue" not in script
    assert "-ErrorAction Stop" in script


# --- M3: the model appears once per caption ---


def _captions(tex: str) -> list[str]:
    return re.findall(r"\\caption\{(.*)\}\n\\label", tex)


def test_captions_state_the_model_once(eng_tex, two_model_roots, tmp_path):
    captions = [
        caption
        for caption in _captions(eng_tex)
        if not caption.startswith("Inventário")
    ]
    assert len(captions) >= 6
    for caption in captions:
        assert caption.count(r"\texttt{hubert\_base}") == 1, caption

    first, second = two_model_roots
    output = tmp_path / "two"
    generate_report_bundle([first, second], output)
    tex = _read_text(output / "report.tex")
    for caption in _captions(tex):
        for model in (r"\texttt{hubert\_base}", r"\texttt{wavlm\_base}"):
            assert caption.count(model) <= 1, caption
    performance = next(c for c in _captions(tex) if "ROC-AUC (sem limiar" in c)
    assert r"\texttt{hubert\_base}" in performance
    assert r"\texttt{wavlm\_base}" in performance


# --- M4: Portuguese for emergence values that were not identified ---


def test_tex_uses_idiomatic_portuguese_for_unidentified_emergence(
    eng_tables, eng_loaded, eng_figures, tmp_path
):
    emergence = eng_tables.emergence.copy()
    emergence["onset"] = pd.array([pd.NA], dtype="Int64")
    emergence["consolidation"] = pd.array([pd.NA], dtype="Int64")
    missing = dataclasses.replace(eng_tables, emergence=emergence)
    path = tmp_path / "report.tex"

    write_report_tex(path, missing, eng_figures, [eng_loaded])

    tex = _read_text(path)
    assert "o início da emergência não foi identificado" in tex
    assert "a consolidação não foi identificada" in tex
    assert "é não identific" not in tex
    written = write_report_tables(missing, tmp_path / "tables")
    assert "emergence_layers.tex" in written
    fragment = _read_text(tmp_path / "tables" / "emergence_layers.tex")
    assert "n/d" in fragment


def test_tex_reports_identified_emergence_layers(eng_tex):
    assert "o início estimado da emergência é a camada 2" in eng_tex
    assert "a consolidação é a camada 8" in eng_tex


# --- M5: consistent decimal and scientific formatting ---


def test_numbers_use_a_decimal_comma_and_one_scientific_style(eng_tex, eng_bundle):
    body = eng_tex.split(r"\begin{document}", 1)[1]
    assert re.search(r"\d\.\d", body) is None
    for name in _BUNDLE_TABLES:
        fragment = _read_text(eng_bundle / "tables" / f"{name}.tex")
        assert re.search(r"\d\.\d", fragment) is None, name
    for match in re.finditer(r"\d[eE][+-]?\d", body):
        start = max(0, match.start() - 6)
        assert re.search(
            r"\d,\d{2}e[+-]\d{2}", body[start : match.end() + 3]
        ), body[start : match.end() + 3]
    assert "rtol=5,00e-03" in body and "atol=2,00e-06" in body


# ===========================================================================
# Final integrated review: PT figures, adaptive ticks, band fail-closed, atomic
# ===========================================================================

_ENGLISH_VISIBLE_TEXT = re.compile(
    r"\b(layer|band|bands|centre|center|time|frequency|mean|relevance|difference|"
    r"synthetic|accuracy|chance|maximum|over|cells|single|cohort|subset|tolerance|"
    r"residual|transition|adjacent|similarity|change|unitless|fraction|flagged|"
    r"detected|previous|current|recompute|check|ratio|signed|scale|model|sample|"
    r"samples|units|metric|encoder layer|limit|bound)\b",
    re.IGNORECASE,
)


def _capture_figures(tables, directory):
    """Render the figures while recording every visible text and heatmap tick."""
    captured: dict[str, dict] = {}
    original = report_module._save_report_figure

    def capture(fig, name, figures_dir):
        fig.canvas.draw()
        renderer = fig._get_renderer()
        import matplotlib.text as mtext

        texts = [
            item.get_text()
            for item in fig.findobj(mtext.Text)
            if item.get_visible() and item.get_text().strip()
        ]
        heatmaps = []
        for ax in fig.axes:
            if not ax.images:
                continue
            labels = [item for item in ax.get_xticklabels() if item.get_text()]
            heatmaps.append(
                {
                    "ticks": [float(value) for value in ax.get_xticks()],
                    "labels": [item.get_text() for item in labels],
                    "boxes": [
                        (
                            item.get_window_extent(renderer).x0,
                            item.get_window_extent(renderer).x1,
                        )
                        for item in labels
                    ],
                    "columns": ax.images[0].get_array().shape[1],
                }
            )
        captured[name] = {"texts": texts, "heatmaps": heatmaps}
        return original(fig, name, figures_dir)

    patch = pytest.MonkeyPatch()
    patch.setattr(report_module, "_save_report_figure", capture)
    try:
        records = render_report_figures(tables, directory)
    finally:
        patch.undo()
    return records, captured


@pytest.fixture(scope="module")
def eng_captured(eng_tables, tmp_path_factory):
    return _capture_figures(eng_tables, tmp_path_factory.mktemp("captured_eng"))


@pytest.fixture(scope="module")
def bands24_root(tmp_path_factory):
    return write_scientific_result_root(
        tmp_path_factory.mktemp("bands24"), n_bands=24
    )


@pytest.fixture(scope="module")
def bands24_tables(bands24_root):
    return build_report_tables(
        [load_report_source(validate_result_directory(bands24_root))]
    )


@pytest.fixture(scope="module")
def bands24_captured(bands24_tables, tmp_path_factory):
    return _capture_figures(
        bands24_tables, tmp_path_factory.mktemp("captured_24")
    )


# --- (1) every visible figure text is Portuguese ---


def test_figures_contain_no_english_visible_text(eng_captured):
    records, captured = eng_captured
    assert set(captured) == set(_EIGHT_FAMILIES)
    for name, entry in captured.items():
        assert entry["texts"], name
        for text in entry["texts"]:
            assert _ENGLISH_VISIBLE_TEXT.search(text) is None, (name, text)
    for record in records:
        for field in ("title", "caption", "metric", "units", "transformation"):
            value = getattr(record, field)
            assert _ENGLISH_VISIBLE_TEXT.search(value) is None, (record.name, field, value)


@pytest.mark.parametrize(
    ("figure", "expected"),
    [
        (
            "performance_by_layer",
            ["Camada do encoder", "ROC-AUC (adimensional)", "EER diagnóstico (%)", "acaso (0,5)"],
        ),
        (
            "fixed_threshold_by_layer",
            ["Acurácia", "MCC", "TPR: sintéticos detectados", "FPR: reais marcados como sintéticos"],
        ),
        (
            "dft_relevance_heatmap",
            ["Centro da banda (Hz, espaçamento mel)", "Camada do encoder", "Relevância DFT absoluta normalizada média"],
        ),
        (
            "class_relevance_by_layer",
            ["real", "sintético", "sintético − real", "Centro da banda (Hz)", "Diferença"],
        ),
        (
            "decision_reorganization_by_layer",
            ["Similaridade cosseno", "Variação L1 normalizada", "Transição entre camadas adjacentes"],
        ),
        (
            "stdft_examples",
            ["Tempo (s)", "Frequência (Hz)", "camada 1", "Relevância STDFT com sinal"],
        ),
        (
            "conservation_diagnostics",
            ["máximo entre células", "Camada do encoder", "tolerância de conservação"],
        ),
    ],
)
def test_figures_use_clear_portuguese_labels_with_units(
    eng_captured, figure, expected
):
    _, captured = eng_captured
    joined = "\n".join(captured[figure]["texts"])
    for needle in expected:
        assert needle in joined, (figure, needle)


def test_stdft_titles_name_the_class_in_portuguese(eng_captured):
    _, captured = eng_captured
    titles = [t for t in captured["stdft_examples"]["texts"] if t.startswith("camada")]
    assert any("· sintético" in t for t in titles)
    assert any("· real" in t for t in titles)
    assert not any("synthetic" in t for t in titles)


def test_stdft_uses_one_symmetric_log_scale_for_all_panels(
    eng_tables, tmp_path, monkeypatch
):
    from dataclasses import replace

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    payloads = {}
    for key, payload in eng_tables.stdft_payloads.items():
        relevance = np.geomspace(
            1e-5, 0.1, payload["relevance"].size, dtype=np.float64
        )
        relevance[::2] *= -1.0
        relevance = relevance.reshape(payload["relevance"].shape)
        relevance.flat[0] = 46.0
        payloads[key] = {**payload, "relevance": relevance}
    skewed_tables = replace(eng_tables, stdft_payloads=payloads)

    captured = {}

    def capture(fig, name, figures_dir):
        fig.canvas.draw()
        meshes = [axis.collections[0] for axis in fig.axes[:6]]
        captured["norms"] = [mesh.norm for mesh in meshes]
        captured["colorbar_ticks"] = [
            label.get_text() for label in fig.axes[-1].get_yticklabels()
        ]
        captured["title_box"] = fig._suptitle.get_window_extent(fig._get_renderer())
        captured["figure_box"] = fig.bbox
        plt.close(fig)
        return (f"figures/{name}.png",)

    monkeypatch.setattr(report_module, "_save_report_figure", capture)
    record = report_module._figure_stdft(plt, skewed_tables, tmp_path)

    assert len(captured["norms"]) == 6
    assert all(isinstance(norm, SymLogNorm) for norm in captured["norms"])
    assert len({id(norm) for norm in captured["norms"]}) == 1
    assert captured["norms"][0].linthresh > 0
    assert len(captured["colorbar_ticks"]) <= 7
    assert captured["title_box"].y1 <= captured["figure_box"].y1
    assert "escala simétrica não linear comum" in record.units


def test_conservation_series_labels_are_portuguese(eng_tables):
    series = build_conservation_series(eng_tables)
    for column in ("scope", "limit_label", "value_label"):
        for text in set(series[column]):
            assert _ENGLISH_VISIBLE_TEXT.search(text) is None, (column, text)


# --- (2) band ticks adapt to 24 real bands ---


def test_fixture_with_24_bands_is_really_24_bands(bands24_tables):
    assert bands24_tables.band_edge_provenance.iloc[0]["n_bands"] == 24
    assert len(bands24_tables.band_edges) == 24
    assert bands24_tables.band_relevance["band"].nunique() == 24


def _assert_ticks_do_not_collide(heatmaps):
    assert heatmaps
    for heat in heatmaps:
        boxes = sorted(heat["boxes"])
        for (_, right), (left, _) in zip(boxes, boxes[1:]):
            assert right <= left + 0.5, heat["labels"]


@pytest.mark.parametrize("figure", ["dft_relevance_heatmap", "class_relevance_by_layer"])
def test_heatmap_ticks_keep_first_and_last_band_and_do_not_collide_at_24_bands(
    bands24_captured, bands24_tables, figure
):
    _, captured = bands24_captured
    edges = bands24_tables.band_edges
    first = f"{edges['center_hz'].iloc[0]:.0f}"
    last = f"{edges['center_hz'].iloc[-1]:.0f}"
    heatmaps = captured[figure]["heatmaps"]
    assert len(heatmaps) == (1 if figure == "dft_relevance_heatmap" else 3)
    for heat in heatmaps:
        assert heat["columns"] == 24
        assert heat["labels"][0] == first and heat["labels"][-1] == last
        assert heat["ticks"][0] == 0.0 and heat["ticks"][-1] == 23.0
        assert 2 <= len(heat["labels"]) < 24
        assert heat["ticks"] == sorted(set(heat["ticks"]))
    _assert_ticks_do_not_collide(heatmaps)


def test_heatmap_ticks_with_8_bands_do_not_collide_and_keep_the_extremes(
    eng_captured, eng_tables
):
    _, captured = eng_captured
    edges = eng_tables.band_edges
    for figure in ("dft_relevance_heatmap", "class_relevance_by_layer"):
        heatmaps = captured[figure]["heatmaps"]
        for heat in heatmaps:
            assert heat["labels"][0] == f"{edges['center_hz'].iloc[0]:.0f}"
            assert heat["labels"][-1] == f"{edges['center_hz'].iloc[-1]:.0f}"
        _assert_ticks_do_not_collide(heatmaps)


@pytest.mark.parametrize("n_bands", [1, 2, 3, 8, 24, 48])
@pytest.mark.parametrize("width", [0.8, 1.5, 4.5])
def test_band_tick_indices_always_keep_both_ends_without_duplicates(n_bands, width):
    indices = report_module._band_tick_indices(n_bands, width)
    assert indices[0] == 0 and indices[-1] == n_bands - 1
    assert indices == sorted(set(indices))
    assert all(0 <= value < n_bands for value in indices)


def test_bundle_with_24_bands_compiles_a_portuguese_tex(bands24_root, tmp_path):
    output = tmp_path / "report24"
    generate_report_bundle([bands24_root], output)
    tex = _read_text(output / "report.tex")
    assert "24 bandas mel" in tex


def test_trilingual_three_model_figures_facet_models_without_collapsing_axes(
    eng_tables, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt
    import warnings

    models = ("hubert_base", "wav2vec2_base", "wavlm_base")
    languages = ("eng", "por", "zho")

    def expand(frame):
        copies = []
        for model in models:
            for source in languages:
                for target in languages:
                    copy = frame.copy()
                    copy["model"] = model
                    copy["source"] = source
                    copy["target"] = target
                    if "language" in copy:
                        copy["language"] = target
                    copies.append(copy)
        return pd.concat(copies, ignore_index=True)

    tables = dataclasses.replace(
        eng_tables,
        models=models,
        languages=languages,
        performance=expand(eng_tables.performance),
        band_relevance=expand(eng_tables.band_relevance),
        class_relevance=expand(eng_tables.class_relevance),
        transitions=expand(eng_tables.transitions),
    )
    captured = {}

    def capture(fig, name, figures_dir):
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            fig.canvas.draw()
        renderer = fig._get_renderer()
        assert not any(
            "collapsed to zero" in str(item.message) for item in observed
        )
        captured[name] = {
            "axes": [
                axis
                for axis in fig.axes
                if axis.get_visible()
                and (axis.lines or axis.images)
            ],
            "figure_legend_labels": [
                text.get_text()
                for legend in fig.legends
                for text in legend.get_texts()
            ],
            "axis_xlabels": [
                axis.get_xlabel()
                for axis in fig.axes
                if axis.get_visible()
            ],
            "figure_xlabel": (
                fig._supxlabel.get_text()
                if fig._supxlabel is not None
                else ""
            ),
            "figure_xlabel_box": (
                fig._supxlabel.get_window_extent(renderer)
                if fig._supxlabel is not None
                else None
            ),
            "legend_boxes": [
                legend.get_window_extent(renderer) for legend in fig.legends
            ],
            "xtick_boxes": [
                label.get_window_extent(renderer)
                for axis in fig.axes
                if axis.get_visible()
                for label in axis.get_xticklabels()
                if label.get_visible() and label.get_text()
            ],
        }
        plt.close(fig)
        return (f"{name}.pdf", f"{name}.png")

    monkeypatch.setattr(report_module, "_save_report_figure", capture)
    report_module._figure_performance(plt, tables, tmp_path)
    report_module._figure_fixed_threshold(plt, tables, tmp_path)
    report_module._figure_heatmap(plt, tables, tmp_path)
    report_module._figure_class_relevance(plt, tables, tmp_path)
    report_module._figure_reorganization(plt, tables, tmp_path)

    assert len(captured["performance_by_layer"]["axes"]) == 6
    assert len(captured["fixed_threshold_by_layer"]["axes"]) == 12
    assert len(captured["decision_reorganization_by_layer"]["axes"]) == 6
    for name in (
        "performance_by_layer",
        "fixed_threshold_by_layer",
        "decision_reorganization_by_layer",
    ):
        assert len(captured[name]["figure_legend_labels"]) == 9
        widths = [axis.get_position().width for axis in captured[name]["axes"]]
        heights = [axis.get_position().height for axis in captured[name]["axes"]]
        assert min(widths) >= 0.14, (name, widths)
        assert min(heights) >= 0.12, (name, heights)
        assert not any(captured[name]["axis_xlabels"])
        assert captured[name]["figure_xlabel"]
        xlabel_box = captured[name]["figure_xlabel_box"]
        assert all(
            xlabel_box.y0 >= legend_box.y1
            for legend_box in captured[name]["legend_boxes"]
        ), name
        assert xlabel_box.y1 <= min(
            box.y0 for box in captured[name]["xtick_boxes"]
        ), name

    assert len(captured["dft_relevance_heatmap"]["axes"]) == 9
    assert len(captured["class_relevance_by_layer"]["axes"]) == 9
    for name in ("dft_relevance_heatmap", "class_relevance_by_layer"):
        assert not any(captured[name]["axis_xlabels"])
        assert captured[name]["figure_xlabel"]


# --- (3) band configuration fails closed and early ---


def _boom_if_late(monkeypatch):
    def late(*args, **kwargs):
        raise AssertionError("failure was detected late")

    monkeypatch.setattr(report_module, "_load_performance", late)


def test_missing_config_with_mismatching_persisted_bands_fails_early(
    tmp_path, monkeypatch
):
    root = write_scientific_result_root(tmp_path, n_bands=4, with_config=False)
    source = validate_result_directory(root)
    _boom_if_late(monkeypatch)
    with pytest.raises(ValueError) as caught:
        load_report_source(source)
    message = str(caught.value)
    assert "no_config_recorded" in message
    assert "fallback" in message and "n_bands=8" in message
    assert "persisted n_bands=4" in message
    assert "config_path" in message
    assert "band indices must be exactly" not in message


def test_hash_mismatch_with_mismatching_persisted_bands_cites_config_and_hashes(
    tmp_path, monkeypatch
):
    root = write_scientific_result_root(tmp_path, n_bands=4)
    plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    entry = plan["inputs"]["eng"]
    config_path = Path(entry["config_path"])
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
    )
    actual = hashlib.sha256(config_path.read_bytes()).hexdigest()
    source = validate_result_directory(root)
    _boom_if_late(monkeypatch)
    with pytest.raises(ValueError) as caught:
        load_report_source(source)
    message = str(caught.value)
    assert f"config_path={entry['config_path']}" in message
    assert "config_sha256_mismatch" in message
    assert entry["config_sha256"] in message and actual in message
    assert "fallback" in message and "n_bands=8" in message
    assert "persisted n_bands=4" in message


def test_missing_config_file_with_mismatching_persisted_bands_cites_the_path(
    tmp_path, monkeypatch
):
    root = write_scientific_result_root(tmp_path, n_bands=4)
    plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    config_path = Path(plan["inputs"]["eng"]["config_path"])
    config_path.unlink()
    source = validate_result_directory(root)
    _boom_if_late(monkeypatch)
    with pytest.raises(ValueError) as caught:
        load_report_source(source)
    message = str(caught.value)
    assert "config_file_missing" in message
    assert str(config_path) in message
    assert "persisted n_bands=4" in message


def test_verified_config_that_disagrees_with_persisted_bands_fails_early(
    tmp_path, monkeypatch
):
    root = write_scientific_result_root(tmp_path, config_n_bands=4)
    source = validate_result_directory(root)
    _boom_if_late(monkeypatch)
    with pytest.raises(ValueError) as caught:
        load_report_source(source)
    message = str(caught.value)
    assert "config_sha256_verified" in message
    assert "n_bands=4" in message and "persisted n_bands=8" in message


def test_fallback_that_matches_the_persisted_count_keeps_auditable_provenance(
    tmp_path,
):
    root = write_scientific_result_root(tmp_path, n_bands=8, with_config=False)
    tables = build_report_tables(
        [load_report_source(validate_result_directory(root))]
    )
    provenance = tables.band_edge_provenance.iloc[0]
    assert provenance["origin"] == "fallback_default_mel_contract"
    assert provenance["reason"] == "no_config_recorded"
    assert provenance["n_bands"] == 8


# --- (4) generate_report_bundle is atomic ---


def _siblings(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_failed_generation_leaves_the_previous_bundle_and_siblings_untouched(
    eng_root, tmp_path, monkeypatch
):
    parent = tmp_path / "work"
    parent.mkdir()
    output = parent / "report"
    generate_report_bundle([eng_root], output)
    (output / "report.pdf").write_bytes(b"previous compile")
    before = _hash_tree(output)
    listing = _siblings(parent)

    def failing(path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("injected failure")

    monkeypatch.setattr(report_module, "write_report_tex", failing)
    with pytest.raises(RuntimeError, match="injected"):
        generate_report_bundle([eng_root], output)

    assert _hash_tree(output) == before
    assert _siblings(parent) == listing


def test_failed_generation_on_a_new_output_leaves_nothing_behind(
    eng_root, tmp_path, monkeypatch
):
    parent = tmp_path / "work"
    parent.mkdir()

    def failing(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(report_module, "build_report_manifest", failing)
    with pytest.raises(RuntimeError, match="injected"):
        generate_report_bundle([eng_root], parent / "report")
    assert _siblings(parent) == []


def test_generation_writes_into_a_sibling_staging_directory(
    eng_root, tmp_path, monkeypatch
):
    parent = tmp_path / "work"
    parent.mkdir()
    output = parent / "report"
    seen = []
    original = report_module.write_report_tex

    def spy(path, *args, **kwargs):
        seen.append(Path(path).parent)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(report_module, "write_report_tex", spy)
    generate_report_bundle([eng_root], output)
    (staging,) = seen
    assert staging != output and staging.parent == parent
    assert not staging.exists()
    assert _siblings(parent) == ["report"]


def test_failure_while_swapping_restores_the_previous_bundle(
    eng_root, tmp_path, monkeypatch
):
    parent = tmp_path / "work"
    parent.mkdir()
    output = parent / "report"
    generate_report_bundle([eng_root], output)
    (output / "report.pdf").write_bytes(b"previous compile")
    before = _hash_tree(output)
    listing = _siblings(parent)
    real_replace = os.replace

    def flaky(src, dst, *args, **kwargs):
        if Path(dst) == output and ".new-" in Path(src).name:
            raise OSError("injected swap failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(report_module.os, "replace", flaky)
    with pytest.raises(OSError, match="injected swap failure"):
        generate_report_bundle([eng_root], output)
    monkeypatch.undo()

    assert _hash_tree(output) == before
    assert _siblings(parent) == listing


def test_retry_after_a_failure_succeeds_without_manual_cleanup(
    eng_root, tmp_path, monkeypatch
):
    parent = tmp_path / "work"
    parent.mkdir()
    output = parent / "report"
    calls = {"count": 0}
    original = report_module.write_report_tables

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            original(*args, **kwargs)
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(report_module, "write_report_tables", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        generate_report_bundle([eng_root], output)
    assert not output.exists() and _siblings(parent) == []

    generate_report_bundle([eng_root], output)
    reference = tmp_path / "reference"
    monkeypatch.undo()
    generate_report_bundle([eng_root], reference)
    assert _hash_tree(output) == _hash_tree(reference)
    assert _siblings(parent) == ["report"]


def test_successful_regeneration_replaces_the_whole_previous_bundle(
    eng_root, eng_bundle, tmp_path
):
    parent = tmp_path / "work"
    parent.mkdir()
    output = parent / "report"
    shutil.copytree(eng_bundle, output)
    (output / "report.aux").write_bytes(b"stale")
    (output / "figures" / "performance_by_layer.png").write_bytes(b"corrupted")

    generate_report_bundle([eng_root], output)

    assert _hash_tree(output) == _hash_tree(eng_bundle)
    assert _siblings(parent) == ["report"]


# --- minor: n_bands in the protocol text is escaped ---


def test_protocol_escapes_a_range_of_band_counts(tmp_path):
    base = tmp_path / "mixed"
    base.mkdir()
    first = write_scientific_result_root(base, profile="hubert_base", n_bands=8)
    second = write_scientific_result_root(base, profile="wavlm_base", n_bands=24)
    output = tmp_path / "report"
    generate_report_bundle([first, second], output)
    tex = _read_text(output / "report.tex")
    assert "8--24 bandas mel" in tex
    assert "\u2013" not in tex
    assert _pdflatex_safe(tex)


# --- turn 6: readable sample IDs, tables that fit the page, babel, titles ---

_LONG_COHORT = (
    ("eng-test-bonafide-asvspoof2024-d_0002785525-963a18614df5", 0),
    ("eng-test-bonafide-asvspoof2024-d_0002785526-77b0c3a1e9d2", 0),
    ("eng-test-spoof-asvspoof2024-d_0003785525-5c1e0f77ab34", 1),
    ("eng-test-spoof-asvspoof2024-d_0003785526-0d9e4b21c6f8", 1),
)
_LONG_IDS = tuple(sample_id for sample_id, _ in _LONG_COHORT)
_MAX_DISPLAYED_ID = 24
_MAX_TITLE_LINE = 28


def test_abbreviate_sample_id_leaves_short_ids_unchanged():
    assert abbreviate_sample_id("real-a") == "real-a"
    exact = "x" * _MAX_DISPLAYED_ID
    assert abbreviate_sample_id(exact) == exact


def test_abbreviate_sample_id_keeps_prefix_and_suffix_within_the_limit():
    long_id = _LONG_IDS[0]
    short = abbreviate_sample_id(long_id)
    assert len(short) == _MAX_DISPLAYED_ID
    assert short.startswith(long_id[:12]) and short.endswith(long_id[-11:])
    assert "\u2026" in short
    assert abbreviate_sample_id(long_id) == short
    assert abbreviate_sample_id(long_id, 10) == long_id[:5] + "\u2026" + long_id[-4:]
    with pytest.raises(ValueError):
        abbreviate_sample_id(long_id, 4)


def test_display_sample_ids_stay_distinct_even_when_ends_collide():
    first = "prefix-AAAAAAAAAAAA-first-middle-suffix-ZZZZZZZZZZZ"
    second = "prefix-AAAAAAAAAAAA-other-middle-suffix-ZZZZZZZZZZZ"
    mapping = report_module._display_sample_ids([first, second, first])
    assert set(mapping) == {first, second}
    assert mapping[first] != mapping[second]
    assert report_module._display_sample_ids([]) == {}


@pytest.fixture(scope="module")
def long_id_root(tmp_path_factory):
    overrides = {layer: _LONG_COHORT for layer in range(1, 13)}
    return write_scientific_result_root(
        tmp_path_factory.mktemp("long_ids"), cohort_overrides=overrides
    )


@pytest.fixture(scope="module")
def long_id_tables(long_id_root):
    return build_report_tables(
        [load_report_source(validate_result_directory(long_id_root))]
    )


@pytest.fixture(scope="module")
def long_id_captured(long_id_tables, tmp_path_factory):
    return _capture_figures(
        long_id_tables, tmp_path_factory.mktemp("captured_long")
    )


@pytest.fixture(scope="module")
def long_id_bundle(long_id_root, tmp_path_factory):
    output = tmp_path_factory.mktemp("bundle_long") / "layerwise_xai_report"
    generate_report_bundle([long_id_root], output)
    return output


def test_stdft_panel_titles_use_a_short_id_that_fits_the_panel(long_id_captured):
    _, captured = long_id_captured
    panel_titles = [
        text for text in captured["stdft_examples"]["texts"] if "camada" in text and "\n" in text
    ]
    assert len(panel_titles) == 6
    for title in panel_titles:
        for line in title.split("\n"):
            assert len(line) <= _MAX_TITLE_LINE, title
        shown_id = title.split("\n")[1]
        assert "\u2026" in shown_id
        assert not any(shown_id == full for full in _LONG_IDS)
        assert any(
            shown_id.startswith(full[:12]) and shown_id.endswith(full[-11:])
            for full in _LONG_IDS
        ), title
    titles_by_id = {title.split("\n")[1] for title in panel_titles}
    assert len(titles_by_id) == 2


def test_no_figure_text_carries_a_full_long_sample_id(long_id_captured):
    _, captured = long_id_captured
    for name, entry in captured.items():
        for text in entry["texts"]:
            for full in _LONG_IDS:
                assert full not in text, (name, text)


def test_conservation_series_scope_uses_the_abbreviated_validation_id(
    long_id_tables,
):
    series = build_conservation_series(long_id_tables)
    scopes = "\n".join(map(str, series["scope"]))
    assert _LONG_IDS[0] not in scopes
    assert abbreviate_sample_id(_LONG_IDS[0]) in scopes


def test_full_sample_ids_remain_in_the_provenance_tables(
    long_id_tables, long_id_bundle
):
    assert set(long_id_tables.conservation["validation_sample_id"]) == {
        _LONG_IDS[0]
    }
    assert set(long_id_tables.stdft_examples["sample_id"]) <= set(_LONG_IDS)
    conservation = pd.read_csv(long_id_bundle / "tables" / "conservation_by_layer.csv")
    assert set(conservation["validation_sample_id"]) == {_LONG_IDS[0]}
    stdft = pd.read_csv(long_id_bundle / "tables" / "stdft_examples.csv")
    assert set(stdft["sample_id"]) == set(_LONG_IDS)
    manifest_text = _read_text(long_id_bundle / "report_manifest.json")
    assert "\u2026" not in manifest_text


def test_tex_narrative_shows_only_the_short_id_and_points_to_the_csv(
    long_id_bundle,
):
    tex = _read_text(long_id_bundle / "report.tex").replace(r"\allowbreak{}", "")
    for full in _LONG_IDS:
        assert full not in tex
        assert full.replace("_", r"\_") not in tex
    short = abbreviate_sample_id(_LONG_IDS[0]).replace("\u2026", r"\ldots{}")
    assert short.replace("_", r"\_") in tex
    assert "identificadores completos" in tex
    assert r"tables/conservation\_by\_layer.csv" in tex
    assert r"tables/stdft\_examples.csv" in tex
    assert _pdflatex_safe(tex)


def test_short_ids_need_no_abbreviation_note(eng_tex):
    assert "identificadores completos" not in eng_tex
    assert "\u2026" not in eng_tex


def _table_block(tex: str, label: str) -> str:
    match = re.search(
        r"\\begin\{table\}.*?\\label\{" + re.escape(label) + r"[^}]*\}.*?\\end\{table\}",
        tex,
        re.DOTALL,
    )
    assert match, label
    return match.group(0)


@pytest.mark.parametrize("label", ["tab:inventario", "tab:performance-"])
def test_wide_tables_are_constrained_to_the_text_width(
    eng_tex, eng_bundle, label
):
    source = (
        eng_tex
        if label == "tab:inventario"
        else _read_text(eng_bundle / "tables" / "performance_by_layer.tex")
    )
    block = _table_block(source, label)
    assert r"\small" in block
    assert re.search(r"\\setlength\{\\tabcolsep\}\{[0-9.]+pt\}", block)
    assert r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%" in block
    assert block.index(r"\resizebox") < block.index(r"\begin{tabular}")
    assert r"\end{tabular}}" in block


def test_inventory_header_is_compact(eng_tex):
    block = _table_block(eng_tex, "tab:inventario")
    header = next(line for line in block.splitlines() if "Camadas" in line)
    cells = [cell.strip() for cell in header.rstrip("\\ ").split("&")]
    assert "Hash da config." in cells
    assert max(len(cell) for cell in cells) <= 18
    assert "12 primeiros" not in header
    assert "12 primeiros caracteres" in block


def test_section_title_is_short_and_babel_is_brazilian(eng_tex):
    assert "Reorganização da decisão entre camadas" in _sections(eng_tex)
    assert "consecutivas" not in "".join(_sections(eng_tex))
    assert max(len(title) for title in _sections(eng_tex)) <= 56
    assert r"\usepackage[brazilian]{babel}" in eng_tex
    assert "[brazil]" not in eng_tex


_PDFLATEX = shutil.which("pdflatex")


@pytest.mark.skipif(_PDFLATEX is None, reason="pdflatex is not installed")
def test_long_id_bundle_compiles_without_significant_overfull_boxes(
    long_id_bundle, tmp_path
):
    workdir = tmp_path / "compile"
    shutil.copytree(long_id_bundle, workdir)
    completed = subprocess.run(
        [_PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=600,
    )
    log = (workdir / "report.log").read_bytes().decode("utf-8", "replace")
    assert completed.returncode == 0, log[-2000:]
    assert (workdir / "report.pdf").is_file()
    overfull = [
        float(value)
        for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide", log)
    ]
    assert [value for value in overfull if value >= 1.0] == [], overfull


def test_report_omits_probe_stability_input_when_rows_miss_performance_cells(
    tmp_path,
):
    root = write_scientific_result_root(tmp_path / "orphan_stability_case")
    paths = LayerwiseSuitePaths(root)
    profile = "hubert_base"
    paths.probe_stability_dir(profile).mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "profile": profile,
                "layer": 99,
                "source": "eng",
                "target": "por",
                "n_seeds": 3,
                "roc_auc_mean": 0.5,
                "roc_auc_std": 0.01,
                "roc_auc_min": 0.49,
                "roc_auc_max": 0.51,
                "mcc_mean": 0.1,
                "mcc_std": 0.01,
                "mcc_min": 0.09,
                "mcc_max": 0.11,
            }
        ]
    ).to_csv(paths.probe_stability_summary(profile), index=False)
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    assert not tables.probe_stability.empty
    report_path = tmp_path / "report_orphan.tex"
    write_report_tex(report_path, tables, [], [loaded])
    text = report_path.read_text(encoding="utf-8")
    assert "probe_stability_by_layer.tex" not in text


def test_report_probe_stability_section_states_non_xai_scope(tmp_path):
    root = write_scientific_result_root(tmp_path / "probe_stability_case")
    paths = LayerwiseSuitePaths(root)
    profile = "hubert_base"
    stability_dir = paths.probe_stability_dir(profile)
    stability_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "profile": profile,
                "layer": 12,
                "source": "eng",
                "target": "eng",
                "n_seeds": 3,
                "roc_auc_mean": 0.91,
                "roc_auc_std": 0.01,
                "roc_auc_min": 0.9,
                "roc_auc_max": 0.92,
                "mcc_mean": 0.55,
                "mcc_std": 0.02,
                "mcc_min": 0.53,
                "mcc_max": 0.57,
            }
        ]
    ).to_csv(paths.probe_stability_summary(profile), index=False)
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    assert not tables.probe_stability.empty
    report_path = tmp_path / "report.tex"
    write_report_tex(report_path, tables, [], [loaded])
    text = report_path.read_text(encoding="utf-8")
    assert "reamostragem estratificada do treino do probe" in text.lower()
    assert "amostra finita de treino" in text
    assert "AttnLRP/DFT-LRP" in text
    assert "ponta-a-ponta" in text


def _write_layer_faithfulness_generation(
    paths: LayerwiseSuitePaths,
    profile: str,
    language: str,
) -> None:
    destination = paths.layer_faithfulness_cell(profile, 12, language, language)

    def writer(directory: Path) -> None:
        pd.DataFrame(
            [
                {
                    "level": "aggregate",
                    "index": "",
                    "k": 1,
                    "true_class": "spoof",
                    "comparison": "top_minus_random",
                    "lhs": 0.5,
                    "rhs": 0.3,
                    "difference": float("nan"),
                    "mean_difference": 0.2,
                    "ci_low": 0.05,
                    "ci_high": 0.35,
                    "n_pairs": 8,
                    "wilcoxon_p": 0.04,
                    "status": "ok",
                }
            ]
        ).to_csv(directory / "paired_comparisons.csv", index=False)
        (directory / "faithfulness_run_manifest.json").write_text(
            json.dumps({"fingerprint": "test", "layer": 12}, indent=2),
            encoding="utf-8",
        )

    publish_generation(destination, writer, role="layer_faithfulness")


def test_report_omits_layer_faithfulness_without_artifacts(tmp_path):
    root = write_scientific_result_root(tmp_path / "no_fidelity")
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    assert tables.layer_faithfulness.empty
    report_path = tmp_path / "report.tex"
    write_report_tex(report_path, tables, [], [loaded])
    text = report_path.read_text(encoding="utf-8")
    assert "Fidelidade por intervenção" not in text
    assert "layer_faithfulness_summary.tex" not in text


def test_report_layer_faithfulness_section_is_cautious_in_portuguese(tmp_path):
    root = write_scientific_result_root(tmp_path / "with_fidelity")
    paths = LayerwiseSuitePaths(root)
    _write_layer_faithfulness_generation(paths, "hubert_base", "eng")
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    assert not tables.layer_faithfulness.empty
    report_path = tmp_path / "report.tex"
    write_report_tex(report_path, tables, [], [loaded])
    text = report_path.read_text(encoding="utf-8")
    assert "Fidelidade por intervenção (camada 12, diagonal)" in text
    assert "layer_faithfulness_summary.tex" in text
    assert "energia RMS igualada" in text
    assert "fontes de variação introduzidas pelo procedimento" in text
    assert "causal" in text.lower()
    assert "camada~12" in text
    assert "fora da diagonal" in text


def test_faithfulness_aggregate_counts_arithmetic():
    frame = pd.DataFrame(
        [
            {
                "model": "hubert_base",
                "comparison": "top_minus_random",
                "mean_difference": 0.2,
                "ci_low": 0.05,
                "ci_high": 0.35,
                "status": "ok",
            },
            {
                "model": "hubert_base",
                "comparison": "top_minus_random",
                "mean_difference": -0.1,
                "ci_low": -0.2,
                "ci_high": -0.01,
                "status": "ok",
            },
            {
                "model": "hubert_base",
                "comparison": "top_minus_bottom",
                "mean_difference": 0.01,
                "ci_low": -0.01,
                "ci_high": 0.02,
                "status": "ok",
            },
        ]
    )
    counts = faithfulness_aggregate_counts(frame)
    random_row = counts.loc[
        counts["comparison"] == "top_minus_random"
    ].iloc[0]
    assert int(random_row["total"]) == 2
    assert int(random_row["n_mean_positive"]) == 1
    assert int(random_row["n_ci_above_zero"]) == 1
    assert int(random_row["n_ci_below_zero"]) == 1
    bottom_row = counts.loc[
        counts["comparison"] == "top_minus_bottom"
    ].iloc[0]
    assert int(bottom_row["n_ci_above_zero"]) == 0
    assert int(bottom_row["n_ci_below_zero"]) == 0


def test_probe_stability_aggregate_by_model_median_and_max():
    frame = pd.DataFrame(
        [
            {
                "model": "wav2vec2_base",
                "roc_auc_std": 0.01,
                "mcc_std": 0.02,
            },
            {
                "model": "wav2vec2_base",
                "roc_auc_std": 0.05,
                "mcc_std": 0.04,
            },
        ]
    )
    summary = probe_stability_aggregate_by_model(frame)
    row = summary.iloc[0]
    assert float(row["roc_auc_std_median"]) == pytest.approx(0.03)
    assert float(row["roc_auc_std_max"]) == pytest.approx(0.05)
    assert float(row["mcc_std_median"]) == pytest.approx(0.03)
    assert float(row["mcc_std_max"]) == pytest.approx(0.04)


def test_probe_stability_pattern_selects_each_metric_extreme_independently():
    summary = pd.DataFrame(
        [
            {
                "model": "hubert_base",
                "roc_auc_std_median": 0.03,
                "roc_auc_std_max": 0.05,
                "mcc_std_median": 0.01,
                "mcc_std_max": 0.07,
            },
            {
                "model": "wav2vec2_base",
                "roc_auc_std_median": 0.01,
                "roc_auc_std_max": 0.09,
                "mcc_std_median": 0.03,
                "mcc_std_max": 0.08,
            },
            {
                "model": "wavlm_base",
                "roc_auc_std_median": 0.02,
                "roc_auc_std_max": 0.06,
                "mcc_std_median": 0.02,
                "mcc_std_max": 0.12,
            },
        ]
    )

    text = report_module._probe_stability_pattern_paragraph(summary)

    assert "ROC-AUC aparece em Wav2Vec2 Base (0,010)" in text
    assert "para o MCC, em HuBERT Base (0,010)" in text
    assert "ROC-AUC aparece em Wav2Vec2 Base (0,090)" in text
    assert "para o MCC, em WavLM Base (0,120)" in text


def test_faithfulness_tex_is_single_compact_table(tmp_path):
    root = write_scientific_result_root(tmp_path / "compact_fidelity_tex")
    paths = LayerwiseSuitePaths(root)
    _write_layer_faithfulness_generation(paths, "hubert_base", "eng")
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    tex = report_module._layer_faithfulness_tex(tables)
    assert tex.count(r"\begin{table}") == 1
    assert "Resumo descritivo da fidelidade" in tex
    assert r"\begin{tabular}{rrlrrrrl}" not in tex


def test_report_fidelity_and_stability_use_compact_tables_and_cautious_prose(
    tmp_path,
):
    root = write_scientific_result_root(tmp_path / "compact_sections")
    paths = LayerwiseSuitePaths(root)
    _write_layer_faithfulness_generation(paths, "hubert_base", "eng")
    paths.probe_stability_dir("hubert_base").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "layer": 12,
                "source": "eng",
                "target": "eng",
                "n_seeds": 3,
                "roc_auc_mean": 0.91,
                "roc_auc_std": 0.01,
                "roc_auc_min": 0.9,
                "roc_auc_max": 0.92,
                "mcc_mean": 0.55,
                "mcc_std": 0.02,
                "mcc_min": 0.53,
                "mcc_max": 0.57,
            }
        ]
    ).to_csv(paths.probe_stability_summary("hubert_base"), index=False)
    loaded = load_report_source(validate_result_directory(root))
    tables = build_report_tables([loaded])
    report_path = tmp_path / "report_compact.tex"
    write_report_tex(report_path, tables, [], [loaded])
    text = report_path.read_text(encoding="utf-8")
    assert "comparações múltiplas" in text
    assert "Leitura cautelosa" in text
    assert "três reamostragens" in text.lower() or "três bootstraps" in text
    assert "Resumo executivo" in text
    assert "concentra o maior número" in text or "fidelidade por intervenção" in text
    assert "coorte" not in text.split("Fidelidade por intervenção")[1].split(
        "Variabilidade"
    )[0]
    tables_dir = tmp_path / "tables_out"
    write_report_tables(tables, tables_dir)
    stability_tex = (tables_dir / "probe_stability_by_layer.tex").read_text(
        encoding="utf-8"
    )
    assert stability_tex.count(r"\begin{table}") == 1
