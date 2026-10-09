from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import brspeech_xai.adaptation as adaptation
import brspeech_xai.encoder_suite as encoder_suite
import brspeech_xai.encoder_suite_runtime as encoder_suite_runtime
import brspeech_xai.suite_aggregation as suite_aggregation
from brspeech_xai.config import LayerwiseXaiConfig
from brspeech_xai.encoder_suite import LANGUAGE_ORDER, SuiteFactories, run_encoder_suite
from brspeech_xai.encoder_suite_runtime import ProductionStageAdapter
from brspeech_xai.layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
)


TEST_LAYERS = (1, 2, 3)
TEST_PROFILES = ("hubert_base", "wavlm_base", "wav2vec2_base")


def _write_smoke_input(root: Path, language: str) -> Path:
    manifest = root / f"{language}.csv"
    rows = []
    for role in ("train", "calibration", "test"):
        for label in (0, 1):
            for index in range(2):
                rows.append(
                    {
                        "sample_id": f"{language}-{role}-{label}-{index}",
                        "language": language,
                        "role": role,
                        "label": label,
                        "processed_path": f"{language}-{role}-{label}-{index}.wav",
                    }
                )
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = root / f"{language}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "run_name": f"task11-{language}",
                "data": {
                    "dataset_kind": "local_manifest",
                    "manifest_path": str(manifest),
                    "train_split": "train",
                    "calibration_split": "calibration",
                    "eval_split": "test",
                    "n_train_per_class": 2,
                    "n_calibration_per_class": 2,
                    "n_test_per_class": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


class _FakeLayerwiseEncoder:
    def __init__(self, tracker: dict[str, int], profile: str) -> None:
        self.tracker = tracker
        self.profile = profile
        self.device = "cpu"

    def eval(self):
        return self

    def requires_grad_(self, value: bool):
        assert value is False
        return self

    def extract_all_layer_embeddings(self, audios, srs):
        self.tracker["extractions"] += 1
        assert len(audios) == len(srs) == 4
        profile_scale = 1.0 + 0.1 * TEST_PROFILES.index(self.profile)
        result = np.empty((4, 13, 3), dtype=np.float32)
        for row, audio in enumerate(audios):
            base = float(np.asarray(audio)[0]) * profile_scale
            for layer in range(13):
                result[row, layer] = (
                    base + 0.05 * layer,
                    2.0 * base - 0.03 * layer,
                    -base + 0.02 * layer,
                )
        return result


class _FactorialSmokeAdapter(ProductionStageAdapter):
    def __init__(self, tracker: dict[str, int]) -> None:
        super().__init__()
        self.tracker = tracker
        self.permute_test_labels = False

    def _load_role(self, request, language: str, role: str) -> tuple:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        if role == "test" and self.permute_test_labels:
            labels = labels[::-1].copy()
        offsets = np.asarray([-0.25, 0.25, -0.25, 0.25], dtype=np.float32)
        values = np.where(labels == 0, -3.0, 3.0).astype(np.float32) + offsets
        audios = [np.asarray([value], dtype=np.float32) for value in values]
        sample_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        catalog = pd.DataFrame(
            {
                "sample_id": [
                    f"{language}-{role}-{label}-{index}"
                    for label in (0, 1)
                    for index in range(2)
                ],
                "label": labels,
                "processed_path": [
                    f"{language}-{role}-{label}-{index}.wav"
                    for label in (0, 1)
                    for index in range(2)
                ],
            }
        )
        if role != "test":
            assert np.array_equal(labels, sample_labels)
        cfg = yaml.safe_load(
            request.inputs[language].config_path.read_text(encoding="utf-8")
        )
        from brspeech_xai.config import load_config

        resolved = load_config(request.inputs[language].config_path)
        assert cfg["data"]["n_calibration_per_class"] == 2
        return audios, [16000] * 4, labels, catalog, resolved

    def _run_cell(self, request, encoder):
        self.tracker["cells"] += 1
        return super()._run_cell(request, encoder)

    def _run_xai(self, request, _encoder):
        self.tracker["xai_cells"] += 1
        paths = self._paths(request)
        profile = str(request.profile)
        source = str(request.source)
        target = str(request.target)
        layer = int(request.layer)
        cohort = self._suite_cohort(request, target)
        predictions = pd.read_parquet(paths.cell(profile, layer, source, target) / "predictions.parquet")
        selected = predictions.set_index("sample_id").loc[cohort["sample_id"]].reset_index()
        profile_offset = TEST_PROFILES.index(profile)
        rows = []
        for index, row in enumerate(selected.itertuples(index=False)):
            signed = np.roll(
                np.arange(1.0, 9.0, dtype=np.float64),
                index + layer,
            ) + profile_offset * 0.1
            absolute = np.abs(signed)
            rows.append(
                {
                    "profile": profile,
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "sample_id": row.sample_id,
                    "y_true": int(row.y_true),
                    "processed_path": str(row.processed_path),
                    "score": float(row.score),
                    "prediction": int(row.prediction),
                    "quadrant": "TP" if row.y_true == row.prediction == 1 else (
                        "TN" if row.y_true == row.prediction == 0 else (
                            "FP" if row.prediction == 1 else "FN"
                        )
                    ),
                    "logit": float(row.score),
                    "equivalence_error": 0.0,
                    "bias_inclusive_attribution_gap": 0.0,
                    "dft_residual": 0.0,
                    "band_signed": signed.tolist(),
                    "band_abs_normalized": (absolute / absolute.sum()).tolist(),
                }
            )
        samples = pd.DataFrame(rows)
        self.tracker["xai_backwards"] += len(samples)
        identity = pd.DataFrame(
            [{"profile": profile, "layer": layer, "source": source, "target": target}]
        )

        def write(directory: Path) -> None:
            samples.to_parquet(directory / "sample_relevance.parquet", index=False)
            identity.to_csv(directory / "dft_band_relevance.csv", index=False)
            identity.to_csv(directory / "dft_band_relevance_byclass.csv", index=False)
            (directory / "attnlrp_conservation.json").write_text(
                json.dumps(
                    {
                        "profile": profile,
                        "layer": layer,
                        "source": source,
                        "target": target,
                        "n": len(samples),
                        "validation_kind": "deterministic_fake_boundary",
                        "bias_zeroed_validation_residual": 0.0,
                        "max_bias_inclusive_attribution_gap": 0.0,
                        "max_dft_residual": 0.0,
                        "tolerance": request.suite_config.conservation_tolerance,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        destination = paths.layer_xai(profile, layer, source, target)
        publish_generation(destination, write, role="layer_xai")
        return {"active_pointer": destination / "active.json"}

    def _run_trace(self, request, _encoder):
        paths = self._paths(request)
        profile = str(request.profile)
        source = str(request.source)
        target = str(request.target)
        cohort = self._suite_cohort(request, target)
        self.tracker["trace_backwards"] += len(cohort)
        layer_rows = []
        transition_rows = []
        for row in cohort.itertuples(index=False):
            for layer in range(1, 13):
                layer_rows.append(
                    {
                        "profile": profile,
                        "source": source,
                        "target": target,
                        "sample_id": row.sample_id,
                        "layer": layer,
                        "signed_sum": float(layer),
                        "absolute_mass": float(layer + 1),
                        "temporal_entropy": 0.5,
                    }
                )
            for layer in range(1, 12):
                transition_rows.append(
                    {
                        "profile": profile,
                        "source": source,
                        "target": target,
                        "sample_id": row.sample_id,
                        "previous_layer": layer,
                        "current_layer": layer + 1,
                        "similarity": 0.9,
                        "normalized_l1_change": 0.1,
                        "absolute_mass": float(layer + 1),
                        "temporal_entropy": 0.5,
                    }
                )

        def write(directory: Path) -> None:
            pd.DataFrame(layer_rows).to_parquet(
                directory / "layer_relevance.parquet", index=False
            )
            pd.DataFrame(transition_rows).to_csv(
                directory / "layer_transition_metrics.csv", index=False
            )

        destination = paths.final_trace_cell(profile, source, target)
        publish_generation(destination, write, role="final_trace")
        return {"active_pointer": destination / "active.json"}


def _threshold_bytes(root: Path) -> dict[tuple[str, int, str], bytes]:
    paths = LayerwiseSuitePaths(root)
    return {
        (profile, layer, source): paths.thresholds(profile, layer, source).read_bytes()
        for profile in TEST_PROFILES
        for layer in TEST_LAYERS
        for source in LANGUAGE_ORDER
    }


def test_task11_factorial_smoke_exercises_real_dag_contracts_and_aggregator(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(encoder_suite, "LAYER_ORDER", TEST_LAYERS)
    monkeypatch.setattr(encoder_suite_runtime, "LAYER_ORDER", TEST_LAYERS)
    monkeypatch.setattr(suite_aggregation, "LAYERS", TEST_LAYERS)
    tracker = {
        "extractions": 0,
        "fits": 0,
        "cells": 0,
        "xai_cells": 0,
        "xai_backwards": 0,
        "trace_backwards": 0,
    }
    adapter = _FactorialSmokeAdapter(tracker)
    real_fit = adaptation.fit_head

    def counted_fit(*args, **kwargs):
        tracker["fits"] += 1
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(adaptation, "fit_head", counted_fit)
    factories = SuiteFactories(
        encoder_factory=lambda profile, _spec, _device: _FakeLayerwiseEncoder(
            tracker, profile
        ),
        runner_factory=lambda: adapter,
    )
    configs = {
        language: _write_smoke_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    output = tmp_path / "suite"
    kwargs = {
        "eng_config": configs["eng"],
        "por_config": configs["por"],
        "zho_config": configs["zho"],
        "output": output,
        "suite_config": LayerwiseXaiConfig(
            profiles=TEST_PROFILES,
            xai_per_class=2,
            stdft_examples_per_class=1,
            bootstrap_samples=20,
            classical_audit=False,
        ),
        "seed": 42,
        "device": "cpu",
        "factories": factories,
    }

    first = run_encoder_suite(**kwargs)
    assert first["executed_stages"] == 277
    assert tracker == {
        "extractions": 27,
        "fits": 27,
        "cells": 81,
        "xai_cells": 81,
        "xai_backwards": 324,
        "trace_backwards": 108,
    }
    paths = LayerwiseSuitePaths(output)
    assert len(
        list(output.glob("*/cells/layer_*/*_to_*/metrics.json"))
    ) == 81
    threshold_snapshot = _threshold_bytes(output)
    for target in LANGUAGE_ORDER:
        expected_ids = pd.read_parquet(paths.suite_cohort(target))["sample_id"].tolist()
        for profile in TEST_PROFILES:
            for layer in TEST_LAYERS:
                for source in LANGUAGE_ORDER:
                    generation = resolve_active_generation(
                        paths.layer_xai(profile, layer, source, target),
                        expected_role="layer_xai",
                    )
                    observed = pd.read_parquet(
                        generation / "sample_relevance.parquet"
                    )["sample_id"].tolist()
                    assert observed == expected_ids
    aggregate_generation = resolve_active_generation(
        output / "aggregates", expected_role="suite_aggregates"
    )
    aggregate_manifest = json.loads(
        (aggregate_generation / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    assert aggregate_manifest["classical_audit"] == {
        "enabled": False,
        "runs": [],
        "scope": "three diagonals at layer 12 only",
    }
    table_records = aggregate_manifest["tables"]
    assert tuple(record["name"] for record in table_records) == (
        suite_aggregation.TABLE_NAMES
    )
    for record in table_records:
        table = pd.read_csv(aggregate_generation / record["name"])
        contract = suite_aggregation._table_scientific_metadata(record["name"])
        assert record["columns"] == table.columns.tolist()
        assert record["rows"] == len(table) > 0
        assert record["uncertainty"]["statistic"] == contract["statistic"]
        assert record["uncertainty"]["unit"] == contract["unit"]
    assert len(
        pd.read_csv(aggregate_generation / "layerwise_performance.csv")
    ) == 81

    before_resume = dict(tracker)
    second = run_encoder_suite(**kwargs)
    assert second["executed_stages"] == 0
    assert second["skipped_stages"] == 277
    assert tracker == before_resume

    adapter.permute_test_labels = True
    for profile in TEST_PROFILES:
        for layer in TEST_LAYERS:
            for source in LANGUAGE_ORDER:
                for target in LANGUAGE_ORDER:
                    (paths.cell(profile, layer, source, target) / "scores.npy").write_bytes(
                        b"corrupt"
                    )
    run_encoder_suite(**kwargs)
    assert tracker["fits"] == 27
    assert tracker["extractions"] == 27
    assert tracker["xai_backwards"] == 324
    assert tracker["trace_backwards"] == 108
    assert _threshold_bytes(output) == threshold_snapshot

    damaged = paths.layer_xai("hubert_base", 2, "eng", "por")
    generation = resolve_active_generation(damaged, expected_role="layer_xai")
    (generation / "sample_relevance.parquet").write_bytes(b"corrupt")
    before_corruption_recovery = dict(tracker)
    run_encoder_suite(**kwargs)
    assert tracker["xai_cells"] == before_corruption_recovery["xai_cells"] + 1
    assert tracker["xai_backwards"] == before_corruption_recovery["xai_backwards"] + 4
    assert tracker["extractions"] == before_corruption_recovery["extractions"]
    assert tracker["fits"] == before_corruption_recovery["fits"]
    assert tracker["trace_backwards"] == before_corruption_recovery["trace_backwards"]
