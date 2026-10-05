from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
import yaml
import numpy as np
import pandas as pd

from brspeech_xai.config import LayerwiseXaiConfig
import brspeech_xai.encoder_suite as encoder_suite
from brspeech_xai.encoder_suite import (
    ARTIFACT_CONTRACTS,
    LANGUAGE_ORDER,
    ROLE_ORDER,
    ProductionSuiteRunner,
    SuiteFactories,
    build_parser,
    build_production_factories,
    run_encoder_suite,
)
from brspeech_xai.encoder_suite_runtime import ProductionStageAdapter
from brspeech_xai.layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
)


def _write_language_input(
    root: Path,
    language: str,
    revision: str = "v1",
    config_revision: str = "v1",
) -> Path:
    manifest = root / f"{language}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "language", "role", "label", "processed_path"),
        )
        writer.writeheader()
        for role in ROLE_ORDER:
            for label in (0, 1):
                writer.writerow(
                    {
                        "sample_id": f"{language}-{revision}-{role}-{label}",
                        "language": language,
                        "role": role,
                        "label": label,
                        "processed_path": f"{language}-{role}-{label}.wav",
                    }
                )
    config = root / f"{language}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "run_name": f"{language}-{config_revision}",
                "data": {
                    "dataset_kind": "local_manifest",
                    "manifest_path": str(manifest),
                    "train_split": "train",
                    "calibration_split": "calibration",
                    "eval_split": "test",
                    "n_train_per_class": 1,
                    "n_calibration_per_class": 1,
                    "n_test_per_class": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


class _Encoder:
    def __init__(self, tracker: dict[str, int]) -> None:
        self.tracker = tracker
        self.closed = False
        tracker["alive"] += 1
        tracker["max_alive"] = max(tracker["max_alive"], tracker["alive"])

    def eval(self):
        return self

    def requires_grad_(self, value: bool):
        assert value is False
        self.tracker["frozen"] += 1
        return self


def test_freeze_encoder_supports_production_wrapper() -> None:
    tracker = {"alive": 0, "max_alive": 0, "frozen": 0}
    model = _Encoder(tracker)
    wrapper = SimpleNamespace(_model=model)

    encoder_suite._freeze_encoder(wrapper)

    assert tracker["frozen"] == 1


class _Runner:
    def __init__(self, tracker: dict[str, object], fail_kind: str | None = None) -> None:
        self.tracker = tracker
        self.fail_kind = fail_kind

    def run_stage(self, request, encoder):
        self.tracker["stages"].append(request.stage_id)
        if request.kind == self.fail_kind:
            raise RuntimeError(f"failed {request.kind}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for name in ARTIFACT_CONTRACTS[request.kind]:
            artifact = request.output_dir / f"{name}.artifact"
            artifact.write_text(
                json.dumps(
                    {"stage": request.stage_id, "fingerprint": request.fingerprint}
                ),
                encoding="utf-8",
            )
            artifacts[name] = artifact
        return artifacts

    def cleanup_encoder(self, encoder):
        if encoder is not None and not encoder.closed:
            encoder.closed = True
            self.tracker["alive"] -= 1
            self.tracker["cleanups"] += 1


def _factories(fail_kind: str | None = None):
    tracker: dict[str, object] = {
        "encoder_calls": 0,
        "runner_calls": 0,
        "alive": 0,
        "max_alive": 0,
        "frozen": 0,
        "cleanups": 0,
        "stages": [],
    }

    def encoder_factory(profile, spec, device):
        assert profile == spec.profile_id
        assert device == "cpu"
        tracker["encoder_calls"] += 1
        return _Encoder(tracker)

    def runner_factory():
        tracker["runner_calls"] += 1
        return _Runner(tracker, fail_kind=fail_kind)

    return SuiteFactories(
        encoder_factory=encoder_factory,
        runner_factory=runner_factory,
    ), tracker


def _run(
    tmp_path: Path,
    *,
    profiles=("hubert_base",),
    languages=LANGUAGE_ORDER,
    por_revision="v1",
    dry_run=False,
    force=False,
    factories=None,
):
    inputs = {
        language: _write_language_input(
            tmp_path,
            language,
            por_revision if language == "por" else "v1",
        )
        for language in languages
    }
    return run_encoder_suite(
        eng_config=inputs.get("eng"),
        por_config=inputs.get("por"),
        zho_config=inputs.get("zho"),
        output=tmp_path / "suite",
        suite_config=LayerwiseXaiConfig(
            profiles=profiles,
            xai_per_class=2,
            stdft_examples_per_class=1,
            bootstrap_samples=5,
        ),
        seed=42,
        device="cpu",
        dry_run=dry_run,
        force=force,
        factories=factories,
    )


def test_layerwise_config_validates_all_fields():
    config = LayerwiseXaiConfig()
    assert config.profiles == (
        "hubert_base",
        "wavlm_base_plus",
        "wav2vec2_base",
    )
    for kwargs in (
        {"profiles": ("hubert_base", "hubert_base")},
        {"profiles": ("unknown",)},
        {"xai_per_class": 0},
        {"stdft_examples_per_class": 26},
        {"bootstrap_samples": True},
        {"conservation_tolerance": float("nan")},
        {"classical_audit": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(config, **kwargs)


def test_selected_languages_requires_at_least_one_config():
    with pytest.raises(ValueError, match="at least one language config"):
        encoder_suite._selected_languages({"eng": None, "por": None, "zho": None})


def test_public_api_requires_at_least_one_config(tmp_path):
    with pytest.raises(ValueError, match="at least one language config"):
        run_encoder_suite(output=tmp_path / "suite", dry_run=True)
    assert not (tmp_path / "suite" / "execution_plan.json").exists()


def test_selected_languages_rejects_unknown_language(tmp_path):
    with pytest.raises(ValueError, match="unsupported language config"):
        encoder_suite._selected_languages({"fra": tmp_path / "fra.yaml"})


def test_selected_languages_accepts_one_config(tmp_path):
    eng = tmp_path / "eng.yaml"
    assert encoder_suite._selected_languages(
        {"eng": eng, "por": None, "zho": None}
    ) == ("eng",)


def test_selected_languages_preserves_canonical_order(tmp_path):
    por = tmp_path / "por.yaml"
    zho = tmp_path / "zho.yaml"
    assert encoder_suite._selected_languages(
        {"zho": zho, "por": por, "eng": None}
    ) == ("por", "zho")


def test_selected_languages_preserves_trilingual_behavior(tmp_path):
    configs = {
        language: tmp_path / f"{language}.yaml" for language in reversed(LANGUAGE_ORDER)
    }
    assert encoder_suite._selected_languages(configs) == LANGUAGE_ORDER


def test_selected_languages_annotation_accepts_string_paths():
    annotation = get_type_hints(encoder_suite._selected_languages)["config_paths"]
    value_type = get_args(annotation)[1]
    assert set(get_args(value_type)) == {Path, str, type(None)}


def test_parser_makes_language_configs_optional_without_heavy_imports():
    args = build_parser().parse_args(
        [
            "--output",
            "suite",
            "--profiles",
            "hubert_base",
            "--xai-per-class",
            "8",
            "--dry-run",
        ]
    )
    assert args.eng_config is None
    assert args.por_config is None
    assert args.zho_config is None
    assert args.xai_per_class == 8
    assert args.dry_run is True


def test_dry_run_writes_complete_plan_without_calling_factories(tmp_path):
    factories, tracker = _factories()
    result = _run(
        tmp_path,
        profiles=("hubert_base", "wavlm_base_plus", "wav2vec2_base"),
        dry_run=True,
        factories=factories,
    )
    plan = json.loads((tmp_path / "suite" / "execution_plan.json").read_text())
    assert result["status"] == "dry-run"
    assert plan["counts"] == {
        "profiles": 3,
        "layers_per_profile": 12,
        "cells_per_layer": 9,
        "probes": 108,
        "cells": 324,
    }
    assert plan["estimated_work"]["explanation_backprops"] == 1296
    assert plan["estimated_work"]["bias_zeroed_certificate_backprops"] == 324
    assert plan["estimated_work"]["bias_inclusive_gaps"] == 1296
    assert plan["estimated_work"]["bias_inclusive_gap_extra_backprops"] == 0
    assert plan["estimated_work"]["stdft_examples"] == 648
    assert plan["estimated_work"]["final_trace_backprops"] == 108
    assert plan["embedding_bytes"]["value"] is None
    assert plan["embedding_bytes"]["reason"]
    assert set(plan["inputs"]) == set(LANGUAGE_ORDER)
    assert plan["schema_version"] == 1
    assert plan["languages"] == {
        language: {
            "role": "source_and_target",
            "roles": list(ROLE_ORDER),
        }
        for language in LANGUAGE_ORDER
    }
    assert all(Path(item["config_path"]).is_absolute() for item in plan["inputs"].values())
    assert all(len(item["manifest_sha256"]) == 64 for item in plan["inputs"].values())
    assert tracker["encoder_calls"] == tracker["runner_calls"] == 0


def _assert_partial_dry_run_counts(
    tmp_path, *, languages, cells_per_layer, probes, cells
):
    result = _run(tmp_path, languages=languages, dry_run=True)
    plan = json.loads((tmp_path / "suite" / "execution_plan.json").read_text())

    assert result["status"] == "dry-run"
    assert plan["schema_version"] == 1
    assert plan["languages"] == {
        language: {
            "role": "source_and_target",
            "roles": list(ROLE_ORDER),
        }
        for language in languages
    }
    assert plan["counts"]["cells_per_layer"] == cells_per_layer
    assert plan["counts"]["probes"] == probes
    assert plan["counts"]["cells"] == cells


def test_single_language_dry_run_builds_only_one_by_one_matrix(tmp_path):
    _assert_partial_dry_run_counts(
        tmp_path,
        languages=("por",),
        cells_per_layer=1,
        probes=12,
        cells=12,
    )


def test_two_language_dry_run_builds_two_by_two_matrix(tmp_path):
    _assert_partial_dry_run_counts(
        tmp_path,
        languages=("eng", "por"),
        cells_per_layer=4,
        probes=24,
        cells=48,
    )


@pytest.mark.parametrize("languages", [("por",), ("eng", "por")])
def test_single_language_and_two_language_graphs_exclude_unselected_languages(
    tmp_path, languages
):
    configs = {
        language: _write_language_input(tmp_path, language)
        for language in languages
    }
    inputs = encoder_suite._validate_inputs(configs, languages)
    suite_config = LayerwiseXaiConfig(profiles=("hubert_base",))
    cohorts, core, profile_aggregates, suite_aggregates = (
        encoder_suite._build_stage_graph(
            output=tmp_path / "suite",
            suite_config=suite_config,
            inputs=inputs,
            selected_languages=languages,
            seed=42,
            device="cpu",
            config_hash=encoder_suite._suite_config_hash(
                suite_config, seed=42, device="cpu"
            ),
        )
    )
    requests = (
        *cohorts,
        *core["hubert_base"],
        *profile_aggregates,
        *suite_aggregates,
    )
    selected = set(languages)

    for request in requests:
        assert request.language is None or request.language in selected
        assert request.source is None or request.source in selected
        assert request.target is None or request.target in selected
        assert set(request.relevant_manifest_hashes) <= selected
    for request in (*profile_aggregates, *suite_aggregates):
        assert set(request.relevant_manifest_hashes) == selected


def test_dry_run_rejects_cross_language_waveform_mismatch_before_plan(tmp_path):
    defaults = {"sample_rate": 16000, "num_samples": 64600}
    divergent = {"sample_rate": 22050, "num_samples": 32000}

    for field, value in divergent.items():
        root = tmp_path / field
        root.mkdir()
        configs = {
            language: _write_language_input(root, language)
            for language in LANGUAGE_ORDER
        }
        raw = yaml.safe_load(configs["por"].read_text(encoding="utf-8"))
        raw["audio"] = defaults | {field: value}
        configs["por"].write_text(yaml.safe_dump(raw), encoding="utf-8")
        output = root / "suite"

        with pytest.raises(
            ValueError,
            match=rf"waveform configuration mismatch.*audio\.{field}.*por={value}",
        ):
            run_encoder_suite(
                eng_config=configs["eng"],
                por_config=configs["por"],
                zho_config=configs["zho"],
                output=output,
                suite_config=LayerwiseXaiConfig(profiles=("hubert_base",)),
                dry_run=True,
            )
        assert not (output / "execution_plan.json").exists()

    equal_root = tmp_path / "equal"
    equal_root.mkdir()
    equal_configs = {
        language: _write_language_input(equal_root, language)
        for language in LANGUAGE_ORDER
    }
    for config in equal_configs.values():
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        raw["audio"] = defaults
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_encoder_suite(
        eng_config=equal_configs["eng"],
        por_config=equal_configs["por"],
        zho_config=equal_configs["zho"],
        output=equal_root / "suite",
        suite_config=LayerwiseXaiConfig(profiles=("hubert_base",)),
        dry_run=True,
    )
    assert result["status"] == "dry-run"
    assert (equal_root / "suite" / "execution_plan.json").is_file()


@pytest.mark.parametrize(
    ("field", "eng_value", "por_value"),
    [
        ("sample_rate", 16000, 22050),
        ("num_samples", 64600, 32000),
    ],
)
def test_two_language_waveform_mismatch_names_only_selected_languages(
    tmp_path, field, eng_value, por_value
):
    configs = {
        language: _write_language_input(tmp_path, language)
        for language in ("eng", "por")
    }
    for language, value in (("eng", eng_value), ("por", por_value)):
        raw = yaml.safe_load(configs[language].read_text(encoding="utf-8"))
        raw["audio"] = {field: value}
        configs[language].write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        run_encoder_suite(
            eng_config=configs["eng"],
            por_config=configs["por"],
            output=tmp_path / "suite",
            suite_config=LayerwiseXaiConfig(profiles=("hubert_base",)),
            dry_run=True,
        )

    message = str(raised.value)
    assert f"waveform configuration mismatch across eng, por" in message
    assert f"audio.{field} (eng={eng_value}, por={por_value})" in message
    assert "zho" not in message.lower()
    assert not (tmp_path / "suite" / "execution_plan.json").exists()


def test_por_only_run_builds_factory_with_por_num_samples(tmp_path, monkeypatch):
    por_config = _write_language_input(tmp_path, "por")
    raw = yaml.safe_load(por_config.read_text(encoding="utf-8"))
    raw["audio"] = {"sample_rate": 16000, "num_samples": 32123}
    por_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    factories, _tracker = _factories()
    observed = []

    def fake_build_production_factories(*, num_samples):
        observed.append(num_samples)
        return factories

    monkeypatch.setattr(
        encoder_suite,
        "build_production_factories",
        fake_build_production_factories,
    )

    run_encoder_suite(
        por_config=por_config,
        output=tmp_path / "suite",
        suite_config=LayerwiseXaiConfig(
            profiles=("hubert_base",),
            xai_per_class=1,
            stdft_examples_per_class=1,
            bootstrap_samples=5,
        ),
        device="cpu",
    )

    assert observed == [32123]


def test_trilingual_suite_aggregate_fingerprint_is_stable():
    inputs = {
        language: encoder_suite.ValidatedInput(
            language=language,
            config_path=Path(f"{language}.yaml"),
            config_sha256=character * 64,
            manifest_path=Path(f"{language}.csv"),
            manifest_sha256=str(index) * 64,
            role_counts={role: 2 for role in ROLE_ORDER},
            audio_sample_rate=16000,
            audio_num_samples=64600,
        )
        for index, (language, character) in enumerate(
            zip(LANGUAGE_ORDER, ("a", "b", "c")), start=1
        )
    }
    suite_config = LayerwiseXaiConfig(
        profiles=("hubert_base",),
        xai_per_class=2,
        stdft_examples_per_class=1,
        bootstrap_samples=5,
    )
    _cohorts, _core, _profile_aggregates, suite_aggregates = (
        encoder_suite._build_stage_graph(
            output=Path("suite"),
            suite_config=suite_config,
            inputs=inputs,
            selected_languages=LANGUAGE_ORDER,
            seed=42,
            device="cpu",
            config_hash=encoder_suite._suite_config_hash(
                suite_config, seed=42, device="cpu"
            ),
        )
    )

    assert suite_aggregates[0].fingerprint == (
        "b6043074fda399f5087d0eb8456f925db81f60ea84fdd82982d6344b4ee90dbe"
    )


def test_resume_force_and_single_encoder_lifetime(tmp_path):
    factories, tracker = _factories()
    _run(tmp_path, factories=factories)
    first_stages = tuple(tracker["stages"])
    assert sum(":embedding:" in stage for stage in first_stages) == 9
    assert sum(":probe:" in stage for stage in first_stages) == 36
    assert tracker["max_alive"] == 1
    assert tracker["alive"] == 0
    assert tracker["frozen"] == 1

    tracker["stages"].clear()
    _run(tmp_path, factories=factories)
    assert tracker["stages"] == []
    assert tracker["encoder_calls"] == 1

    _run(tmp_path, factories=factories, force=True)
    assert sum(":embedding:" in stage for stage in tracker["stages"]) == 9
    assert tracker["encoder_calls"] == 2
    assert tracker["max_alive"] == 1


def test_partial_resume_after_language_set_transition_invalidates_only_aggregates(
    tmp_path,
):
    factories, tracker = _factories()
    _run(tmp_path, factories=factories)
    tracker["stages"].clear()

    _run(tmp_path, languages=("eng", "por"), factories=factories)
    transition_stages = tuple(tracker["stages"])

    assert transition_stages == (
        "hubert_base:profile_aggregate",
        "suite:aggregate",
    )
    assert not any("zho" in stage for stage in transition_stages)

    tracker["stages"].clear()
    _run(tmp_path, languages=("eng", "por"), factories=factories)

    assert tracker["stages"] == []


def test_por_manifest_change_invalidates_only_por_dependency_cone(tmp_path):
    factories, tracker = _factories()
    _run(tmp_path, factories=factories)
    tracker["stages"].clear()

    _run(tmp_path, factories=factories, por_revision="v2")
    stages = tuple(tracker["stages"])
    embeddings = [stage for stage in stages if ":embedding:" in stage]
    probes = [stage for stage in stages if ":probe:" in stage]
    cells = [stage for stage in stages if ":cell:" in stage]
    assert len(embeddings) == 3
    assert all(":por:" in stage for stage in embeddings)
    assert len(probes) == 12
    assert all(stage.endswith(":por") for stage in probes)
    assert len(cells) == 12 * 5
    assert all(":por:" in stage or stage.endswith(":por") for stage in cells)
    assert not any(":embedding:eng:" in stage or ":embedding:zho:" in stage for stage in stages)


def test_missing_artifact_and_corrupt_marker_never_skip(tmp_path):
    factories, tracker = _factories()
    _run(tmp_path, factories=factories)
    tracker["stages"].clear()
    marker = next((tmp_path / "suite" / ".state").rglob("*embedding*train*.json"))
    payload = json.loads(marker.read_text())
    Path(payload["artifacts"][0]["path"]).unlink()
    marker.write_text("{broken", encoding="utf-8")

    _run(tmp_path, factories=factories)
    assert any(":embedding:" in stage and stage.endswith(":train") for stage in tracker["stages"])


def test_failure_writes_status_and_always_cleans_encoder(tmp_path):
    factories, tracker = _factories(fail_kind="probe")
    with pytest.raises(RuntimeError, match="failed probe"):
        _run(tmp_path, factories=factories)
    status = json.loads((tmp_path / "suite" / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"].startswith("hubert_base:probe:")
    assert status["profile"] == "hubert_base"
    assert tracker["alive"] == 0
    assert tracker["cleanups"] == 1
    failed_marker = (
        tmp_path / "suite" / ".state" / f"{status['stage'].replace(':', '__')}.json"
    )
    assert not failed_marker.exists()


def test_plan_is_written_before_any_execution_failure(tmp_path):
    factories, _ = _factories(fail_kind="cohort")
    with pytest.raises(RuntimeError, match="failed cohort"):
        _run(tmp_path, factories=factories)
    assert (tmp_path / "suite" / "execution_plan.json").is_file()


def test_config_content_change_invalidates_its_dependency_cone(tmp_path):
    factories, tracker = _factories()
    inputs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    kwargs = dict(
        eng_config=inputs["eng"],
        por_config=inputs["por"],
        zho_config=inputs["zho"],
        output=tmp_path / "suite",
        suite_config=LayerwiseXaiConfig(
            profiles=("hubert_base",),
            xai_per_class=2,
            stdft_examples_per_class=1,
            bootstrap_samples=5,
        ),
        seed=42,
        device="cpu",
        factories=factories,
    )
    run_encoder_suite(**kwargs)
    tracker["stages"].clear()
    _write_language_input(tmp_path, "por", config_revision="v2")
    run_encoder_suite(**kwargs)
    stages = tuple(tracker["stages"])
    assert sum(":embedding:por:" in stage for stage in stages) == 3
    assert not any(":embedding:eng:" in stage or ":embedding:zho:" in stage for stage in stages)
    assert sum(":probe:" in stage for stage in stages) == 12
    assert all(stage.endswith(":por") for stage in stages if ":probe:" in stage)


def test_status_replaces_old_complete_before_validation_failure(tmp_path):
    factories, _ = _factories()
    _run(tmp_path, factories=factories)
    broken = tmp_path / "eng.yaml"
    broken.write_text("data: []", encoding="utf-8")
    with pytest.raises(ValueError, match="scientific calibration"):
        run_encoder_suite(
            eng_config=broken,
            por_config=tmp_path / "por.yaml",
            zho_config=tmp_path / "zho.yaml",
            output=tmp_path / "suite",
            suite_config=LayerwiseXaiConfig(
                profiles=("hubert_base",),
                xai_per_class=2,
                stdft_examples_per_class=1,
                bootstrap_samples=5,
            ),
            device="cpu",
            factories=factories,
        )
    status = json.loads((tmp_path / "suite" / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "validate_inputs"


def test_status_covers_runner_factory_failure(tmp_path):
    class BrokenFactories:
        def encoder_factory(self, profile, spec, device):
            raise AssertionError("not reached")

        def runner_factory(self):
            raise RuntimeError("runner factory failed")

    factories = SuiteFactories(
        encoder_factory=BrokenFactories().encoder_factory,
        runner_factory=BrokenFactories().runner_factory,
    )
    with pytest.raises(RuntimeError, match="runner factory failed"):
        _run(tmp_path, factories=factories)
    status = json.loads((tmp_path / "suite" / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "runner_factory"


def test_cohort_is_suite_level_and_created_once_per_target(tmp_path):
    factories, tracker = _factories()
    _run(
        tmp_path,
        profiles=("hubert_base", "wavlm_base_plus"),
        factories=factories,
    )
    cohorts = [stage for stage in tracker["stages"] if ":cohort:" in stage]
    assert cohorts == [
        "suite:cohort:eng",
        "suite:cohort:por",
        "suite:cohort:zho",
    ]


def test_default_estimates_distinguish_backprops_and_diagnostics(tmp_path):
    inputs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    run_encoder_suite(
        eng_config=inputs["eng"],
        por_config=inputs["por"],
        zho_config=inputs["zho"],
        output=tmp_path / "suite",
        suite_config=LayerwiseXaiConfig(),
        dry_run=True,
    )
    work = json.loads((tmp_path / "suite" / "execution_plan.json").read_text())[
        "estimated_work"
    ]
    assert work["explanation_backprops"] == 16_200
    assert work["bias_zeroed_certificate_backprops"] == 324
    assert work["bias_inclusive_gaps"] == 16_200
    assert work["bias_inclusive_gap_extra_backprops"] == 0


def test_marker_rejects_artifacts_that_do_not_match_stage_contract(tmp_path):
    factories, _ = _factories()

    class InvalidRunner(_Runner):
        def run_stage(self, request, encoder):
            artifact = request.output_dir / "artifact.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}", encoding="utf-8")
            return {"unexpected": artifact}

    invalid = SuiteFactories(
        encoder_factory=factories.encoder_factory,
        runner_factory=lambda: InvalidRunner(
            {
                "stages": [],
                "alive": 0,
                "max_alive": 0,
                "frozen": 0,
                "cleanups": 0,
            }
        ),
    )
    with pytest.raises(ValueError, match="artifact contract"):
        _run(tmp_path, factories=invalid)
    assert not list((tmp_path / "suite" / ".state").glob("*.json"))


def test_main_non_dry_builds_production_adapters_at_heavy_boundaries(
    tmp_path, monkeypatch
):
    inputs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    tracker = {
        "encoder_calls": 0,
        "runner_calls": 0,
        "alive": 0,
        "max_alive": 0,
        "frozen": 0,
        "cleanups": 0,
        "stages": [],
    }

    monkeypatch.setattr(
        encoder_suite,
        "_production_encoder_factory",
        lambda profile, spec, device, *, num_samples: _Encoder(tracker),
    )
    monkeypatch.setattr(
        ProductionSuiteRunner,
        "run_stage",
        lambda self, request, encoder: _Runner(tracker).run_stage(request, encoder),
    )
    monkeypatch.setattr(
        ProductionSuiteRunner,
        "cleanup_encoder",
        lambda self, encoder: _Runner(tracker).cleanup_encoder(encoder),
    )
    monkeypatch.setattr(
        ProductionSuiteRunner,
        "validate_stage",
        lambda self, request, artifacts: None,
    )
    code = encoder_suite.main(
        [
            "--eng-config",
            str(inputs["eng"]),
            "--por-config",
            str(inputs["por"]),
            "--zho-config",
            str(inputs["zho"]),
            "--output",
            str(tmp_path / "suite"),
            "--profiles",
            "hubert_base",
            "--device",
            "cpu",
            "--xai-per-class",
            "2",
            "--stdft-examples-per-class",
            "1",
            "--bootstrap-samples",
            "5",
        ]
    )
    assert code == 0
    assert tracker["stages"]
    assert tracker["alive"] == 0
    assert isinstance(build_production_factories(), SuiteFactories)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration_split", ""),
        ("n_calibration_per_class", 0),
    ],
)
def test_preflight_rejects_implicit_or_empty_calibration(tmp_path, field, value):
    configs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    raw = yaml.safe_load(configs["por"].read_text(encoding="utf-8"))
    raw["data"][field] = value
    configs["por"].write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="scientific calibration.*explicit",
    ):
        run_encoder_suite(
            eng_config=configs["eng"],
            por_config=configs["por"],
            zho_config=configs["zho"],
            output=tmp_path / "suite",
            suite_config=LayerwiseXaiConfig(profiles=("hubert_base",)),
            dry_run=True,
        )
    assert not (tmp_path / "suite" / "execution_plan.json").exists()


def test_production_adapter_reaches_explicit_calibration_extraction(
    tmp_path, monkeypatch
):
    configs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    validated = encoder_suite._validate_inputs(configs, LANGUAGE_ORDER)
    suite_config = LayerwiseXaiConfig(
        profiles=("hubert_base",),
        xai_per_class=1,
        stdft_examples_per_class=1,
        bootstrap_samples=5,
    )
    _cohorts, core, _profile_aggregates, _suite_aggregates = (
        encoder_suite._build_stage_graph(
            output=tmp_path / "suite",
            suite_config=suite_config,
            inputs=validated,
            selected_languages=LANGUAGE_ORDER,
            seed=42,
            device="cpu",
            config_hash=encoder_suite._suite_config_hash(
                suite_config, seed=42, device="cpu"
            ),
        )
    )
    request = next(
        item
        for item in core["hubert_base"]
        if item.kind == "embedding"
        and item.language == "eng"
        and item.role == "calibration"
    )
    calls = []

    def fake_build_balanced_split(dataset_id, split, quota, **kwargs):
        calls.append((split, quota, kwargs["manifest_path"]))
        return (
            [np.zeros(16, dtype=np.float32), np.ones(16, dtype=np.float32)],
            [16000, 16000],
            [0, 1],
            [
                {
                    "sample_id": f"eng-calibration-{label}",
                    "processed_path": str(tmp_path / f"{label}.wav"),
                }
                for label in (0, 1)
            ],
        )

    monkeypatch.setattr(
        "brspeech_xai.data.build_balanced_split",
        fake_build_balanced_split,
    )

    class FakeLayerwiseEncoder:
        def extract_all_layer_embeddings(self, audios, srs):
            assert len(audios) == len(srs) == 2
            return np.zeros((2, 13, 4), dtype=np.float32)

    artifacts = ProductionStageAdapter().run_stage(
        request, FakeLayerwiseEncoder()
    )
    assert calls == [
        ("calibration", 1, str(validated["eng"].manifest_path))
    ]
    assert set(artifacts) == {"embeddings", "metadata"}


def test_por_change_invalidates_only_five_trace_pairs(tmp_path):
    factories, tracker = _factories()
    _run(tmp_path, factories=factories)
    tracker["stages"].clear()
    _run(tmp_path, factories=factories, por_revision="v2")
    traces = [stage for stage in tracker["stages"] if ":trace:" in stage]
    assert len(traces) == 5
    assert all(":por:" in stage or stage.endswith(":por") for stage in traces)
    assert not any(
        source in stage and stage.endswith(target)
        for source, target in (
            (":eng:", "eng"),
            (":eng:", "zho"),
            (":zho:", "eng"),
            (":zho:", "zho"),
        )
        for stage in traces
    )


def test_generation_manifest_hashes_artifacts_and_rejects_tampering(tmp_path):
    destination = tmp_path / "generation"

    def writer(directory):
        (directory / "payload.json").write_text(
            json.dumps({"schema_version": 1, "value": "complete"}),
            encoding="utf-8",
        )

    publish_generation(destination, writer, role="layer_xai")
    generation = resolve_active_generation(destination, expected_role="layer_xai")
    manifest = json.loads((generation / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["role"] == "layer_xai"
    assert manifest["artifacts"] == [
        {
            "path": "payload.json",
            "sha256": encoder_suite._sha256_file(generation / "payload.json"),
            "size_bytes": (generation / "payload.json").stat().st_size,
        }
    ]

    (generation / "payload.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="valid active generation"):
        resolve_active_generation(destination, expected_role="layer_xai")


def test_trace_adapter_uses_canonical_rate_and_target_config_fingerprint(
    tmp_path, monkeypatch
):
    configs = {
        language: _write_language_input(
            tmp_path,
            language,
            config_revision=f"{language}-cfg",
        )
        for language in LANGUAGE_ORDER
    }
    validated = encoder_suite._validate_inputs(configs, LANGUAGE_ORDER)
    suite_config = LayerwiseXaiConfig(
        profiles=("hubert_base",),
        xai_per_class=1,
        stdft_examples_per_class=1,
        bootstrap_samples=5,
    )
    _cohorts, core, _profile_aggregates, _suite_aggregates = (
        encoder_suite._build_stage_graph(
            output=tmp_path / "suite",
            suite_config=suite_config,
            inputs=validated,
            selected_languages=LANGUAGE_ORDER,
            seed=42,
            device="cpu",
            config_hash=encoder_suite._suite_config_hash(
                suite_config, seed=42, device="cpu"
            ),
        )
    )
    request = next(
        item
        for item in core["hubert_base"]
        if item.kind == "trace"
        and item.source == "eng"
        and item.target == "por"
    )
    cohort_dir = (
        request.output_root
        / ".stage-artifacts"
        / "suite__cohort__por"
    )
    cohort_dir.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame(
        {
            "sample_id": ["por-0", "por-1"],
            "label": [0, 1],
            "processed_path": ["zero.wav", "one.wav"],
        }
    ).to_parquet(cohort_dir / "cohort.parquet", index=False)
    observed = {}

    def fake_trace_cell(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(
        "brspeech_xai.final_trace.run_final_decision_trace_cell",
        fake_trace_cell,
    )
    adapter = ProductionStageAdapter()
    monkeypatch.setattr(
        adapter,
        "_load_role",
        lambda request, language, role: (
            [],
            [],
            np.asarray([]),
            pd.DataFrame(),
            SimpleNamespace(
                audio=SimpleNamespace(
                    sample_rate={"eng": 16000, "por": 22050, "zho": 24000}[language]
                )
            ),
        ),
    )
    fake_encoder = SimpleNamespace(
        _model=object(),
        _processor=object(),
        device="cpu",
    )
    adapter._run_trace(request, fake_encoder)

    assert observed["source"] == "eng"
    assert observed["target"] == "por"
    assert observed["sample_rate"] == 16000
    assert request.relevant_config_hashes["por"] == validated["por"].config_sha256


def _relevance_samples(profile: str, order=("s0", "s1", "s2", "s3")):
    values = {
        "s0": {
            "y_true": 0,
            "prediction": 0,
            "band_signed": [-2.0, 1.0, 3.0],
            "band_abs_normalized": [1 / 3, 1 / 6, 1 / 2],
        },
        "s1": {
            "y_true": 0,
            "prediction": 0,
            "band_signed": [1.0, 4.0, 2.0],
            "band_abs_normalized": [1 / 7, 4 / 7, 2 / 7],
        },
        "s2": {
            "y_true": 1,
            "prediction": 1,
            "band_signed": [3.0, -1.0, 2.0],
            "band_abs_normalized": [1 / 2, 1 / 6, 1 / 3],
        },
        "s3": {
            "y_true": 1,
            "prediction": 1,
            "band_signed": [2.0, 5.0, 1.0],
            "band_abs_normalized": [1 / 4, 5 / 8, 1 / 8],
        },
    }
    return pd.DataFrame(
        [
            {
                "profile": profile,
                "source": "eng",
                "target": "eng",
                "layer": 1,
                "sample_id": sample_id,
                **values[sample_id],
            }
            for sample_id in order
        ]
    )


def test_relevance_agreement_joins_by_complete_identity_not_position():
    from brspeech_xai.suite_aggregation import encoder_relevance_agreement

    samples = pd.concat(
        [
            _relevance_samples("hubert_base", ("s0", "s1", "s2", "s3")),
            _relevance_samples("wavlm_base_plus", ("s3", "s1", "s0", "s2")),
        ],
        ignore_index=True,
    )
    result = encoder_relevance_agreement(
        samples,
        band_edges_by_target={"eng": np.asarray([0.0, 100.0, 200.0, 300.0])},
    )

    assert len(result) == 4
    assert set(result["conditioning"]) == {
        "true_label",
        "prediction_relation",
    }
    assert set(
        result.loc[result["conditioning"] == "true_label", "class_value"]
    ) == {0, 1}
    assert set(
        result.loc[
            result["conditioning"] == "prediction_relation", "class_value"
        ]
    ) == {"both_spoof", "both_bonafide"}
    assert set(result["n"]) == {2}
    assert np.allclose(result["spearman_signed_mean"], 1.0)
    assert np.allclose(result["cosine_signed_mean"], 1.0)
    assert np.allclose(result["cosine_absolute_normalized_mean"], 1.0)
    assert np.allclose(result["jensen_shannon_absolute_normalized_mean"], 0.0)
    assert set(result["ci_method"]) == {"normal_95_sem"}
    assert set(result["ci_confidence"]) == {0.95}
    assert set(result["statistic"]) == {"mean_of_per_sample_encoder_agreement"}
    assert set(result["unit"]) == {"metric_specific"}
    assert set(result["spearman_signed_unit"]) == {"unitless"}
    assert set(result["jensen_shannon_absolute_normalized_unit"]) == {"bits"}


@pytest.mark.parametrize("failure", ["duplicate", "missing", "edges", "zero_mass", "nan"])
def test_relevance_agreement_fails_closed_on_invalid_pairing(failure):
    from brspeech_xai.suite_aggregation import encoder_relevance_agreement

    left = _relevance_samples("hubert_base")
    right = _relevance_samples("wavlm_base_plus", ("s3", "s1", "s0", "s2"))
    edges = {
        "eng": {
            "hubert_base": np.asarray([0.0, 100.0, 200.0, 300.0]),
            "wavlm_base_plus": np.asarray([0.0, 100.0, 200.0, 300.0]),
        }
    }
    if failure == "duplicate":
        right = pd.concat([right, right.iloc[[0]]], ignore_index=True)
    elif failure == "missing":
        right = right.iloc[:-1].copy()
    elif failure == "edges":
        edges["eng"]["wavlm_base_plus"] = np.asarray(
            [0.0, 50.0, 200.0, 300.0]
        )
    elif failure == "zero_mass":
        right.at[0, "band_abs_normalized"] = [0.0, 0.0, 0.0]
    elif failure == "nan":
        right.at[0, "band_signed"] = [1.0, np.nan, 2.0]

    with pytest.raises(ValueError):
        encoder_relevance_agreement(
            pd.concat([left, right], ignore_index=True),
            band_edges_by_target=edges,
        )


def test_final_reorganization_requires_eleven_transitions_and_joins_labels_by_id():
    from brspeech_xai.suite_aggregation import final_decision_reorganization

    transitions = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "source": "eng",
                "target": "por",
                "sample_id": sample_id,
                "previous_layer": layer,
                "current_layer": layer + 1,
                "similarity": 0.9,
                "normalized_l1_change": 0.1,
                "absolute_mass": float(layer),
                "temporal_entropy": 0.5,
            }
            for sample_id in ("s1", "s0")
            for layer in range(1, 12)
        ]
    )
    labels = pd.DataFrame(
        {
            "profile": ["hubert_base", "hubert_base"],
            "source": ["eng", "eng"],
            "target": ["por", "por"],
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "prediction": [1, 1],
        }
    )
    result = final_decision_reorganization(transitions, labels)

    assert len(result) == 44
    assert set(result["conditioning"]) == {"all", "y_true", "prediction"}
    assert set(result["n"]) == {1, 2}
    assert set(result["ci_method"]) == {"normal_95_sem"}
    assert set(result["ci_confidence"]) == {0.95}
    assert set(result["statistic"]) == {"transition_metric_mean"}
    assert set(result["unit"]) == {"metric_specific"}
    assert set(result["temporal_entropy_unit"]) == {"nats"}
    with pytest.raises(ValueError, match="11 transitions"):
        final_decision_reorganization(transitions.iloc[:-1], labels)


@pytest.mark.parametrize("failure", ["wrong_context", "duplicate", "missing", "extra"])
def test_final_reorganization_rejects_nonexact_label_identity(failure):
    from brspeech_xai.suite_aggregation import final_decision_reorganization

    transitions = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "source": "eng",
                "target": "por",
                "sample_id": sample_id,
                "previous_layer": layer,
                "current_layer": layer + 1,
                "similarity": 0.9,
                "normalized_l1_change": 0.1,
                "absolute_mass": float(layer),
                "temporal_entropy": 0.5,
            }
            for sample_id in ("s0", "s1")
            for layer in range(1, 12)
        ]
    )
    labels = pd.DataFrame(
        {
            "profile": ["hubert_base", "hubert_base"],
            "source": ["eng", "eng"],
            "target": ["por", "por"],
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "prediction": [0, 1],
        }
    )
    if failure == "wrong_context":
        labels.loc[labels["sample_id"] == "s0", "target"] = "eng"
    elif failure == "duplicate":
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    elif failure == "missing":
        labels = labels.iloc[[0]].copy()
    else:
        labels = pd.concat(
            [
                labels,
                pd.DataFrame(
                    {
                        "profile": ["hubert_base"],
                        "source": ["eng"],
                        "target": ["por"],
                        "sample_id": ["extra"],
                        "y_true": [0],
                        "prediction": [0],
                    }
                ),
            ],
            ignore_index=True,
        )
    with pytest.raises(ValueError):
        final_decision_reorganization(transitions, labels)


@pytest.mark.parametrize("failure", ["empty", "duplicate", "missing", "extra", "n"])
def test_predictions_must_exactly_match_xai_cohort(failure):
    from brspeech_xai.suite_aggregation import _validate_predictions_against_xai

    predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "score": [0.2, 0.8],
            "prediction": [0, 1],
        }
    )
    xai = pd.DataFrame(
        {
            "profile": ["hubert_base", "hubert_base"],
            "source": ["eng", "eng"],
            "target": ["por", "por"],
            "layer": [1, 1],
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "prediction": [0, 1],
        }
    )
    if failure == "empty":
        predictions.loc[0, "sample_id"] = ""
    elif failure == "duplicate":
        predictions.loc[1, "sample_id"] = "s0"
    elif failure == "missing":
        predictions = predictions.iloc[[0]].copy()
    elif failure == "extra":
        predictions.loc[1, "sample_id"] = "extra"
    expected_n = 3 if failure == "n" else 2
    with pytest.raises(ValueError):
        _validate_predictions_against_xai(
            predictions,
            xai,
            profile="hubert_base",
            layer=1,
            source="eng",
            target="por",
            expected_n=expected_n,
        )


def test_agreement_rejects_single_sample_class_instead_of_inflating_bins():
    from brspeech_xai.suite_aggregation import encoder_relevance_agreement

    samples = pd.concat(
        [
            _relevance_samples("hubert_base").query("sample_id != 's3'"),
            _relevance_samples("wavlm_base_plus").query("sample_id != 's3'"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="at least 2 samples"):
        encoder_relevance_agreement(
            samples,
            band_edges_by_target={
                "eng": np.asarray([0.0, 100.0, 200.0, 300.0])
            },
        )


def test_uncertainty_metadata_distinguishes_normal_and_bootstrap_tables():
    from brspeech_xai.suite_aggregation import _table_uncertainty_metadata

    assert _table_uncertainty_metadata(
        "encoder_relevance_agreement.csv"
    ) == {
        "method": "normal_95_sem",
        "confidence": 0.95,
        "bootstrap": False,
    }
    assert _table_uncertainty_metadata("emergence_layers.csv") == {
        "method": "stratified_bootstrap_auc_percentile",
        "confidence_column": "bootstrap_confidence",
        "bootstrap": True,
    }


def test_authoritative_suite_cohort_rejects_consistent_cell_omission(tmp_path):
    from brspeech_xai.layerwise_paths import LayerwiseSuitePaths
    from brspeech_xai.suite_aggregation import (
        _load_authoritative_cohorts,
        _validate_cell_against_authoritative_cohort,
    )

    paths = LayerwiseSuitePaths(tmp_path)
    cohort_path = paths.suite_cohort("por")
    cohort_path.parent.mkdir(parents=True)
    cohort = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "label": [0, 1, 1],
            "processed_path": ["zero.wav", "one.wav", "two.wav"],
        }
    )
    cohort.to_parquet(cohort_path, index=False)
    metadata_path = paths.suite_cohort_metadata("por")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "por",
                "sample_ids": ["s0", "s1", "s2"],
                "per_class": 1,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )
    marker_dir = tmp_path / ".state"
    marker_dir.mkdir()
    marker_dir.joinpath("suite__cohort__por.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_version": "encoder-suite-v2",
                "stage_id": "suite:cohort:por",
                "profile": None,
                "fingerprint": "cohort-fingerprint",
                "artifacts": [
                    {
                        "name": "cohort",
                        "path": str(cohort_path.resolve()),
                        "sha256": encoder_suite._sha256_file(cohort_path),
                    },
                    {
                        "name": "metadata",
                        "path": str(metadata_path.resolve()),
                        "sha256": encoder_suite._sha256_file(metadata_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    authority = _load_authoritative_cohorts(
        tmp_path, inputs={"por": object()}
    )["por"]
    predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "processed_path": ["zero.wav", "one.wav"],
            "score": [0.2, 0.8],
            "prediction": [0, 1],
        }
    )
    xai = pd.DataFrame(
        {
            "profile": ["hubert_base", "hubert_base"],
            "source": ["eng", "eng"],
            "target": ["por", "por"],
            "layer": [1, 1],
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "processed_path": ["zero.wav", "one.wav"],
            "prediction": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="authoritative suite cohort"):
        _validate_cell_against_authoritative_cohort(
            predictions,
            xai,
            authority,
            profile="hubert_base",
            layer=1,
            source="eng",
            target="por",
        )


def test_manifest_table_contract_declares_and_validates_statistic_unit(tmp_path):
    from brspeech_xai.suite_aggregation import _table_manifest_record

    frame = pd.DataFrame(
        {
            "jensen_shannon": [0.1],
            "cosine_similarity": [0.9],
            "cosine_distance": [0.1],
            "statistic": ["mean_distribution_language_shift"],
            "unit": ["metric_specific"],
        }
    )
    path = tmp_path / "spectral_divergence_by_layer.csv"
    frame.to_csv(path, index=False)
    record = _table_manifest_record(
        "spectral_divergence_by_layer.csv", frame, path
    )
    assert record["uncertainty"] == {
        "method": "none",
        "confidence": None,
        "bootstrap": False,
        "statistic": "mean_distribution_language_shift",
        "unit": "metric_specific",
    }

    inconsistent = frame.assign(statistic="wrong")
    with pytest.raises(ValueError, match="scientific contract"):
        _table_manifest_record(
            "spectral_divergence_by_layer.csv", inconsistent, path
        )


def test_emergence_publication_matches_manifest_scientific_contract(tmp_path):
    from brspeech_xai.suite_aggregation import (
        _annotate_emergence_scientific_contract,
        _table_manifest_record,
    )

    published = _annotate_emergence_scientific_contract(
        pd.DataFrame(
            {
                "profile": ["hubert_base"],
                "source": ["eng"],
                "target": ["por"],
                "bootstrap_confidence": [0.95],
            }
        )
    )
    path = tmp_path / "emergence_layers.csv"
    published.to_csv(path, index=False)

    record = _table_manifest_record("emergence_layers.csv", published, path)

    assert set(published["statistic"]) == {
        "discriminative_onset_and_consolidation_layer"
    }
    assert set(published["unit"]) == {"layer_index_and_roc_auc"}
    assert record["uncertainty"]["statistic"] == published["statistic"].iat[0]
    assert record["uncertainty"]["unit"] == published["unit"].iat[0]


def _partial_aggregation_frames(languages):
    performance = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "layer": layer,
                "source": source,
                "target": target,
                "n": 2,
                "auc": 0.75,
                "accuracy": 0.5,
                "diagonal": source == target,
                "external_validation": source != target,
                "corpus_shift": source != target,
            }
            for layer in range(1, 13)
            for source in languages
            for target in languages
        ]
    )
    emergence = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "source": source,
                "target": target,
                "bootstrap_confidence": 0.95,
            }
            for source in languages
            for target in languages
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "source": source,
                "target": target,
                "layer": layer,
                "sample_id": f"{target}-{label}",
                "y_true": label,
                "prediction": label,
                "band_signed": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "band_abs_normalized": [0.125] * 8,
            }
            for layer in range(1, 13)
            for source in languages
            for target in languages
            for label in (0, 1)
        ]
    )
    reorganization = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "source": source,
                "target": target,
                "previous_layer": 1,
                "current_layer": 2,
                "conditioning": "all",
                "class_value": "all",
                "ci_method": "normal_95_sem",
                "ci_confidence": 0.95,
                "statistic": "transition_metric_mean",
                "unit": "metric_specific",
            }
            for source in languages
            for target in languages
        ]
    )
    return performance, emergence, samples, reorganization


def _write_stage_marker(root, stage_id, artifacts):
    marker = root / ".state" / f"{stage_id.replace(':', '__')}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_version": "encoder-suite-v2",
                "stage_id": stage_id,
                "fingerprint": f"fingerprint-{stage_id}",
                "artifacts": [
                    {
                        "name": name,
                        "path": str(path.resolve()),
                        "sha256": encoder_suite._sha256_file(path),
                    }
                    for name, path in artifacts.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    return marker


def _write_aggregation_matrix(root, languages):
    paths = LayerwiseSuitePaths(root)
    profile = "hubert_base"
    cohorts = {}
    for target in languages:
        cohort = pd.DataFrame(
            {
                "sample_id": [f"{target}-0", f"{target}-1"],
                "label": [0, 1],
                "processed_path": [f"{target}-0.wav", f"{target}-1.wav"],
            }
        )
        cohort_path = paths.suite_cohort(target)
        cohort_path.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_parquet(cohort_path, index=False)
        metadata_path = paths.suite_cohort_metadata(target)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target": target,
                    "sample_ids": cohort["sample_id"].tolist(),
                }
            ),
            encoding="utf-8",
        )
        _write_stage_marker(
            root,
            f"suite:cohort:{target}",
            {"cohort": cohort_path, "metadata": metadata_path},
        )
        cohorts[target] = cohort

    cell_markers = {}
    for layer in range(1, 13):
        for source in languages:
            for target in languages:
                cohort = cohorts[target]
                cell_dir = paths.cell(profile, layer, source, target)
                cell_dir.mkdir(parents=True, exist_ok=True)
                scores_path = cell_dir / "scores.npy"
                np.save(scores_path, np.asarray([0.1, 0.9], dtype=np.float64))
                predictions_path = cell_dir / "predictions.parquet"
                predictions = cohort.rename(columns={"label": "y_true"}).assign(
                    score=[0.1, 0.9],
                    prediction=[0, 1],
                )
                predictions.to_parquet(predictions_path, index=False)
                metrics_path = cell_dir / "metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        {
                            "threshold_free": {"roc_auc": 1.0},
                            "fixed_threshold": {"accuracy": 1.0},
                        }
                    ),
                    encoding="utf-8",
                )
                stage_id = f"{profile}:cell:{layer:02d}:{source}:{target}"
                cell_markers[(profile, layer, source, target)] = _write_stage_marker(
                    root,
                    stage_id,
                    {
                        "scores": scores_path,
                        "predictions": predictions_path,
                        "metrics": metrics_path,
                    },
                )

                samples = predictions.assign(
                    profile=profile,
                    layer=layer,
                    source=source,
                    target=target,
                    band_signed=[[1.0, 2.0]] * len(predictions),
                    band_abs_normalized=[[0.5, 0.5]] * len(predictions),
                )
                xai_destination = paths.layer_xai(profile, layer, source, target)

                def write_xai(directory, frame=samples):
                    frame.to_parquet(
                        directory / "sample_relevance.parquet", index=False
                    )

                publish_generation(xai_destination, write_xai, role="layer_xai")
                _write_stage_marker(
                    root,
                    f"{profile}:xai:{layer:02d}:{source}:{target}",
                    {"active_pointer": xai_destination / "active.json"},
                )

    for source in languages:
        for target in languages:
            emergence_path = (
                root
                / ".stage-artifacts"
                / f"{profile}__emergence__{source}__{target}"
                / "emergence.csv"
            )
            emergence_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "profile": profile,
                        "source": source,
                        "target": target,
                        "bootstrap_confidence": 0.95,
                    }
                ]
            ).to_csv(emergence_path, index=False)
            _write_stage_marker(
                root,
                f"{profile}:emergence:{source}:{target}",
                {"summary": emergence_path},
            )

            transitions = pd.DataFrame(
                [
                    {
                        "profile": profile,
                        "source": source,
                        "target": target,
                        "sample_id": sample_id,
                        "previous_layer": layer,
                        "current_layer": layer + 1,
                        "similarity": 0.9,
                        "normalized_l1_change": 0.1,
                        "absolute_mass": 1.0,
                        "temporal_entropy": 0.5,
                    }
                    for sample_id in cohorts[target]["sample_id"]
                    for layer in range(1, 12)
                ]
            )
            trace_destination = paths.final_trace_cell(profile, source, target)

            def write_trace(directory, frame=transitions):
                frame.to_csv(
                    directory / "layer_transition_metrics.csv", index=False
                )

            publish_generation(trace_destination, write_trace, role="final_trace")
            _write_stage_marker(
                root,
                f"{profile}:trace:{source}:{target}",
                {"active_pointer": trace_destination / "active.json"},
            )
    return cell_markers


def test_languages_from_inputs_rejects_unknown_keys():
    from brspeech_xai.suite_aggregation import _languages_from_inputs

    with pytest.raises(ValueError, match="unsupported language"):
        _languages_from_inputs({"eng": object(), "fra": object()})


@pytest.mark.parametrize("languages", [("por",), ("eng", "por")])
def test_read_inputs_enumerates_real_selected_cell_matrix(tmp_path, languages):
    from brspeech_xai.suite_aggregation import _read_inputs

    _write_aggregation_matrix(tmp_path, languages)
    performance, emergence, samples, reorganization = _read_inputs(
        tmp_path,
        ("hubert_base",),
        {language: object() for language in languages},
    )
    expected_cells = {
        ("hubert_base", layer, source, target)
        for layer in range(1, 13)
        for source in languages
        for target in languages
    }

    assert set(
        performance[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    ) == expected_cells
    assert set(
        samples[["profile", "layer", "source", "target"]].itertuples(
            index=False, name=None
        )
    ) == expected_cells
    assert len(performance) == 12 * len(languages) ** 2
    assert len(emergence) == len(languages) ** 2
    assert set(reorganization["source"]) == set(languages)
    assert set(reorganization["target"]) == set(languages)


@pytest.mark.parametrize(
    ("languages", "expected_cells"),
    [(("por",), 12), (("eng", "por"), 48)],
)
def test_aggregate_suite_uses_only_declared_languages(
    tmp_path, monkeypatch, languages, expected_cells
):
    import brspeech_xai.suite_aggregation as aggregation

    configs = {
        language: _write_language_input(tmp_path, language)
        for language in languages
    }
    inputs = encoder_suite._validate_inputs(configs, languages)
    observed = {}

    def fake_read_inputs(root, profiles, declared_inputs):
        observed["languages"] = tuple(declared_inputs)
        return _partial_aggregation_frames(languages)

    audit_calls = []

    def audit_spy(**kwargs):
        audit_calls.append((kwargs["source"], kwargs["target"], kwargs["layer"]))
        output = kwargs["output_dir"] / "audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return {"audit": output}

    monkeypatch.setattr(aggregation, "_read_inputs", fake_read_inputs)
    active = aggregation.aggregate_suite(
        root=tmp_path / "suite",
        profiles=("hubert_base",),
        config_hash="config",
        upstream_fingerprints=("upstream",),
        inputs=inputs,
        classical_audit=True,
        classical_audit_adapter=audit_spy,
        plotter=lambda _tables, _directory: [],
    )
    generation = resolve_active_generation(
        active.parent, expected_role="suite_aggregates"
    )
    manifest = json.loads(
        (generation / "aggregate_manifest.json").read_text(encoding="utf-8")
    )

    assert observed["languages"] == languages
    assert len(pd.read_csv(generation / "layerwise_performance.csv")) == expected_cells
    assert audit_calls == [(language, language, 12) for language in languages]
    spectral = pd.read_csv(generation / "spectral_divergence_by_layer.csv")
    spectral_record = next(
        record
        for record in manifest["tables"]
        if record["name"] == "spectral_divergence_by_layer.csv"
    )
    if len(languages) == 1:
        assert spectral.empty
        assert (
            spectral_record["status"]
            == "not_applicable_less_than_two_languages"
        )
    else:
        assert not spectral.empty
        assert "status" not in spectral_record


def test_read_inputs_fails_closed_when_selected_language_cell_is_missing(
    tmp_path,
):
    from brspeech_xai.suite_aggregation import _read_inputs

    languages = ("por", "zho")
    cell_markers = _write_aggregation_matrix(tmp_path, languages)
    missing_cell = ("hubert_base", 7, "por", "zho")
    cell_markers[missing_cell].unlink()
    assert len([path for path in cell_markers.values() if path.exists()]) == 47

    with pytest.raises(ValueError, match="invalid upstream marker"):
        _read_inputs(
            tmp_path,
            ("hubert_base",),
            {language: object() for language in languages},
        )


def test_classical_audit_off_calls_zero_and_on_calls_three_diagonal_layer12(tmp_path):
    from brspeech_xai.suite_aggregation import run_classical_audit

    inputs = {language: object() for language in LANGUAGE_ORDER}
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_dir"] / "audit.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return {"audit": output}

    assert run_classical_audit(
        enabled=False,
        output_dir=tmp_path,
        adapter=spy,
        inputs=inputs,
    ) == []
    assert calls == []

    records = run_classical_audit(
        enabled=True,
        output_dir=tmp_path,
        adapter=spy,
        inputs=inputs,
    )
    assert len(records) == len(calls) == 3
    assert {(call["source"], call["target"], call["layer"]) for call in calls} == {
        ("eng", "eng", 12),
        ("por", "por", 12),
        ("zho", "zho", 12),
    }


def test_production_suite_aggregate_stage_calls_real_aggregator(tmp_path, monkeypatch):
    suite_config = LayerwiseXaiConfig(
        profiles=("hubert_base",),
        xai_per_class=1,
        stdft_examples_per_class=1,
        bootstrap_samples=5,
    )
    configs = {
        language: _write_language_input(tmp_path, language)
        for language in LANGUAGE_ORDER
    }
    inputs = encoder_suite._validate_inputs(configs, LANGUAGE_ORDER)
    _cohorts, _core, _profile, suite = encoder_suite._build_stage_graph(
        output=tmp_path / "suite",
        suite_config=suite_config,
        inputs=inputs,
        selected_languages=LANGUAGE_ORDER,
        seed=42,
        device="cpu",
        config_hash=encoder_suite._suite_config_hash(
            suite_config, seed=42, device="cpu"
        ),
    )
    observed = []
    aggregate_index = tmp_path / "suite" / "aggregates" / "active.json"
    aggregate_index.parent.mkdir(parents=True)
    aggregate_index.write_text("{}", encoding="utf-8")

    def fake_aggregate(**kwargs):
        observed.append(kwargs)
        return aggregate_index

    monkeypatch.setattr(
        "brspeech_xai.suite_aggregation.aggregate_suite",
        fake_aggregate,
    )
    artifacts = ProductionStageAdapter().run_stage(suite[0], None)
    assert artifacts == {"index": aggregate_index}
    assert observed[0]["profiles"] == ("hubert_base",)
    assert observed[0]["classical_audit"] is False


def test_language_shift_has_complete_deterministic_identities_and_claim_flags():
    from brspeech_xai.suite_aggregation import language_shift_by_layer

    performance = pd.DataFrame(
        [
            {
                "profile": "hubert_base",
                "layer": 1,
                "source": source,
                "target": target,
                "auc": auc,
            }
            for source, target, auc in (
                ("por", "eng", 0.7),
                ("eng", "eng", 0.8),
                ("por", "por", 0.9),
            )
        ]
    )
    result = language_shift_by_layer(performance)
    assert list(result[["source", "target"]].itertuples(index=False, name=None)) == [
        ("eng", "eng"),
        ("por", "eng"),
        ("por", "por"),
    ]
    shifted = result.loc[
        (result["source"] == "por") & (result["target"] == "eng")
    ].iloc[0]
    assert shifted["delta_auc"] == pytest.approx(-0.1)
    assert bool(shifted["diagonal"]) is False
    assert bool(shifted["external_validation"]) is True
    assert bool(shifted["corpus_shift"]) is True


def test_upstream_marker_corruption_fails_closed(tmp_path):
    from brspeech_xai.suite_aggregation import _validated_marker

    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    marker_dir = tmp_path / ".state"
    marker_dir.mkdir()
    marker = marker_dir / "p__emergence__eng__eng.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract_version": "encoder-suite-v2",
                "stage_id": "p:emergence:eng:eng",
                "profile": "p",
                "fingerprint": "f",
                "artifacts": [
                    {
                        "name": "summary",
                        "path": str(artifact.resolve()),
                        "sha256": encoder_suite._sha256_file(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _validated_marker(tmp_path, "p:emergence:eng:eng", {"summary"})
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid upstream marker"):
        _validated_marker(tmp_path, "p:emergence:eng:eng", {"summary"})


def test_aggregate_filters_full_test_predictions_to_authoritative_xai_cohort():
    from brspeech_xai.suite_aggregation import (
        _validate_cell_against_authoritative_cohort,
    )

    authority = pd.DataFrame(
        {
            "sample_id": ["s0", "s2"],
            "label": [0, 1],
            "processed_path": ["zero.wav", "two.wav"],
        }
    )
    predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "y_true": [0, 0, 1, 1],
            "processed_path": ["zero.wav", "one.wav", "two.wav", "three.wav"],
            "score": [0.1, 0.2, 0.8, 0.9],
            "prediction": [0, 0, 1, 1],
        }
    )
    xai = pd.DataFrame(
        {
            "profile": ["hubert_base", "hubert_base"],
            "source": ["eng", "eng"],
            "target": ["por", "por"],
            "layer": [1, 1],
            "sample_id": ["s2", "s0"],
            "y_true": [1, 0],
            "processed_path": ["two.wav", "zero.wav"],
            "prediction": [1, 0],
        }
    )

    filtered = _validate_cell_against_authoritative_cohort(
        predictions,
        xai,
        authority,
        profile="hubert_base",
        layer=1,
        source="eng",
        target="por",
    )

    assert filtered["sample_id"].tolist() == ["s0", "s2"]
    assert filtered["prediction"].tolist() == [0, 1]


def test_cross_encoder_agreement_allows_prediction_disagreement():
    from brspeech_xai.suite_aggregation import encoder_relevance_agreement

    left = _relevance_samples("hubert_base")
    right = _relevance_samples("wavlm_base_plus")
    right.loc[right["sample_id"].isin(["s1", "s2"]), "prediction"] = [1, 0]

    result = encoder_relevance_agreement(
        pd.concat([left, right], ignore_index=True),
        band_edges_by_target={
            "eng": np.asarray([0.0, 100.0, 200.0, 300.0])
        },
    )

    assert set(result["conditioning"]) == {"true_label", "prediction_relation"}
    relation = result.loc[result["conditioning"] == "prediction_relation"]
    assert set(relation["prediction_relation"]) == {
        "both_bonafide",
        "both_spoof",
        "disagree",
    }
    assert {"pred_a", "pred_b"}.issubset(result.columns)


def test_single_profile_agreement_is_valid_not_applicable_table_and_plot(tmp_path):
    from brspeech_xai.suite_aggregation import (
        _plot_tables,
        _table_manifest_record,
        encoder_relevance_agreement,
    )

    table = encoder_relevance_agreement(
        _relevance_samples("hubert_base"),
        band_edges_by_target={
            "eng": np.asarray([0.0, 100.0, 200.0, 300.0])
        },
    )
    path = tmp_path / "encoder_relevance_agreement.csv"
    table.to_csv(path, index=False)
    record = _table_manifest_record(path.name, table, path)
    assert table.empty
    assert record["status"] == "not_applicable_less_than_two_profiles"

    minimal = {
        "layerwise_performance.csv": pd.DataFrame(
            [{"profile": "hubert_base", "source": "eng", "target": "eng", "layer": 1, "auc": 1.0}]
        ),
        "language_shift_by_layer.csv": pd.DataFrame(
            [{"layer": 1, "auc": 1.0, "diagonal": True}]
        ),
        "encoder_relevance_agreement.csv": table,
        "spectral_divergence_by_layer.csv": pd.DataFrame(
            [{"layer": 1, "jensen_shannon": 0.1}]
        ),
        "final_decision_reorganization.csv": pd.DataFrame(
            [{"current_layer": 2, "normalized_l1_change_mean": 0.1}]
        ),
    }
    statuses = _plot_tables(minimal, tmp_path)
    agreement = next(
        item for item in statuses if item["name"] == "relevance_agreement_by_layer"
    )
    assert agreement == {
        "name": "relevance_agreement_by_layer",
        "status": "skipped",
        "reason": "not_applicable_less_than_two_profiles",
    }


def test_spectral_divergence_compares_language_targets_to_source_diagonal():
    from brspeech_xai.suite_aggregation import spectral_divergence_by_layer

    rows = []
    distributions = {
        "eng": ([0.8, 0.2], [0.6, 0.4]),
        "por": ([0.2, 0.8], [0.4, 0.6]),
        "zho": ([0.5, 0.5], [0.5, 0.5]),
    }
    for target, values in distributions.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "profile": "hubert_base",
                    "source": "eng",
                    "target": target,
                    "layer": 1,
                    "sample_id": f"{target}-{index}",
                    "y_true": 0,
                    "prediction": index,
                    "band_signed": value,
                    "band_abs_normalized": value,
                }
            )

    result = spectral_divergence_by_layer(
        pd.DataFrame(rows),
        band_edges_by_target={
            target: np.asarray([0.0, 100.0, 200.0])
            for target in ("eng", "por", "zho")
        },
    )

    assert set(result["reference_target"]) == {"eng"}
    assert set(result["comparison_target"]) == {"por", "zho"}
    assert set(result["true_label"]) == {0}
    assert result["corpus_shift"].all()
    assert (result["jensen_shannon"] >= 0).all()
    assert np.allclose(
        result["cosine_distance"],
        1.0 - result["cosine_similarity"],
    )


def test_production_encoder_factory_passes_configured_num_samples(monkeypatch):
    observed = {}

    class FakeHF:
        n_transformer_layers = 12

        def __init__(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr("brspeech_xai.encoders.hf_ssl.HFSSLEmbedder", FakeHF)
    spec = SimpleNamespace(checkpoint="fake/checkpoint", n_transformer_layers=12)

    encoder_suite._production_encoder_factory(
        "hubert_base", spec, "cpu", num_samples=12345
    )

    assert observed["num_samples"] == 12345


def test_runtime_shared_audio_loader_uses_embedder_canonical_preprocessing():
    from brspeech_xai.encoder_suite_runtime import build_preprocessed_audio_loader

    calls = []

    class FakeEmbedder:
        def preprocess_waveform(self, audio, sample_rate):
            calls.append((np.asarray(audio).copy(), sample_rate))
            return np.asarray(audio, dtype=np.float32)[:5]

    raw = np.arange(12, dtype=np.float32)
    loader = build_preprocessed_audio_loader(
        FakeEmbedder(),
        read_audio=lambda path: (raw, 22050),
    )

    xai_input, xai_rate = loader(Path("long.wav"))
    trace_input, trace_rate = loader(Path("long.wav"))
    np.testing.assert_array_equal(xai_input, trace_input)
    np.testing.assert_array_equal(xai_input, raw[:5])
    assert xai_rate == trace_rate == 16000
    assert len(calls) == 2
    embedding_logit = float(xai_input.sum() / 10.0)
    trace_logit = float(trace_input.sum() / 10.0)
    embedding_score = 1.0 / (1.0 + np.exp(-embedding_logit))
    trace_score = 1.0 / (1.0 + np.exp(-trace_logit))
    assert trace_logit == embedding_logit
    assert trace_score == embedding_score
