"""Controlled DFT-LRP intervention faithfulness for the layer-12 diagonal XAI cohort."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import faithfulness_bands as fb  # noqa: E402

from brspeech_xai.dft_lrp import rfft_frequencies  # noqa: E402
from brspeech_xai.layerwise_faithfulness import (  # noqa: E402
    DEFAULT_FAITHFULNESS_PER_CLASS,
    FINAL_LAYER,
    active_xai_generation_id,
    build_faithfulness_fingerprint,
    build_ssl_detector,
    load_authoritative_xai_cohort,
    probe_artifact_identity,
    release_profile_encoder,
    run_faithfulness_cell,
    run_layerwise_faithfulness,
    select_faithfulness_cohort,
    should_skip_completed_run,
    validate_faithfulness_params,
)
from brspeech_xai.layerwise_paths import LayerwiseSuitePaths, publish_generation  # noqa: E402


def _write_xai_cohort(
    paths: LayerwiseSuitePaths,
    profile: str,
    language: str,
    *,
    n_per_class: int = 4,
) -> None:
    rows = []
    for label in (0, 1):
        for index in range(n_per_class):
            sample_id = f"{language}-{'real' if label == 0 else 'fake'}-{index:02d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "y_true": label,
                    "processed_path": f"/data/{sample_id}.wav",
                    "prediction": label,
                    "score": 1.0 if label else -1.0,
                    "band_signed": [0.1] * 8,
                    "band_abs_normalized": [0.125] * 8,
                }
            )
    frame = pd.DataFrame(rows)
    destination = paths.layer_xai(profile, FINAL_LAYER, language, language)

    def writer(directory: Path) -> None:
        frame.to_parquet(directory / "sample_relevance.parquet", index=False)

    publish_generation(destination, writer, role="layer_xai")


def _minimal_suite(tmp_path: Path, *, languages: tuple[str, ...] = ("eng", "por")) -> Path:
    root = tmp_path / "hubert_base__eng-por__layerwise_xai__full"
    root.mkdir()
    (root / "execution_plan.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "inputs": {
                    language: {
                        "config_path": str(tmp_path / f"{language}.yaml"),
                        "manifest_path": str(tmp_path / f"{language}.csv"),
                    }
                    for language in languages
                },
            }
        ),
        encoding="utf-8",
    )
    for language in languages:
        (tmp_path / f"{language}.yaml").write_text(
            "bands:\n  n_bands: 8\n  f_min: 20.0\n  f_max: 7900.0\n",
            encoding="utf-8",
        )
        (tmp_path / f"{language}.csv").write_text("sample_id,path\n", encoding="utf-8")
    paths = LayerwiseSuitePaths(root)
    for language in languages:
        _write_xai_cohort(paths, "hubert_base", language)
    return root


def _write_probe_and_calibration(
    paths: LayerwiseSuitePaths,
    profile: str,
    language: str,
    layer: int,
    *,
    hidden: int = 8,
    n_calibration: int = 16,
) -> None:
    labels = np.array([0] * (n_calibration // 2) + [1] * (n_calibration // 2))
    vectors = np.random.default_rng(0).standard_normal((n_calibration, hidden))
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=200),
    ).fit(vectors, labels)
    probe_path = paths.probe(profile, layer, language)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, probe_path)
    paths.thresholds(profile, layer, language).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_language": language,
                "layer": layer,
                "detectors": {"ad": {"threshold": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    embeddings = np.zeros((n_calibration, 13, hidden), dtype=np.float32)
    for index, vector in enumerate(vectors):
        embeddings[index, layer, :] = vector.astype(np.float32)
    calibration_path = paths.embeddings(profile, language, "calibration")
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(calibration_path, embeddings, allow_pickle=False)


def test_select_faithfulness_cohort_is_balanced_and_deterministic():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s-{label}-{index}" for label in (0, 1) for index in range(10)],
            "y_true": [0] * 10 + [1] * 10,
            "processed_path": [f"/p/{sid}" for sid in [f"s-{label}-{index}" for label in (0, 1) for index in range(10)]],
        }
    )
    first = select_faithfulness_cohort(frame, per_class=3, seed=7)
    second = select_faithfulness_cohort(frame, per_class=3, seed=7)
    third = select_faithfulness_cohort(frame, per_class=3, seed=8)
    assert len(first) == 6
    assert set(first["y_true"]) == {0, 1}
    assert tuple(first["sample_id"]) == tuple(second["sample_id"])
    assert tuple(first["sample_id"]) != tuple(third["sample_id"])


def test_load_authoritative_xai_cohort_fail_closed(tmp_path):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    cohort = load_authoritative_xai_cohort(
        paths, "hubert_base", FINAL_LAYER, "eng", "eng"
    )
    assert set(cohort.columns) >= {"sample_id", "y_true", "processed_path"}
    bad = paths.layer_xai("hubert_base", FINAL_LAYER, "eng", "eng")
    generation = bad / "generations" / json.loads((bad / "active.json").read_text())["generation"]
    (generation / "sample_relevance.parquet").unlink()
    with pytest.raises(ValueError, match="cohort|sample_relevance|no valid active generation"):
        load_authoritative_xai_cohort(paths, "hubert_base", FINAL_LAYER, "eng", "eng")


def test_run_layerwise_faithfulness_selects_diagonal_layer12_and_languages(tmp_path):
    root = _minimal_suite(tmp_path, languages=("eng", "por"))
    calls: list[tuple[str, str, str, int]] = []

    def fake_cell(**kwargs):
        calls.append(
            (kwargs["profile"], kwargs["source"], kwargs["target"], kwargs["layer"])
        )
        return {"status": "published", "n_clips": 2}

    result = run_layerwise_faithfulness(
        root,
        profiles=["hubert_base"],
        languages=["por"],
        per_class=2,
        ks=[1],
        n_random=1,
        seed=11,
        skip_if_complete=False,
        run_cell_fn=fake_cell,
    )
    assert calls == [("hubert_base", "por", "por", FINAL_LAYER)]
    assert result[0]["languages"] == ["por"]
    assert result[0]["layer"] == FINAL_LAYER


def test_run_layerwise_faithfulness_records_provenance_and_conditions(tmp_path, monkeypatch):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)
    edges = np.array([0.0, 4000.0, 8000.0])
    stdft_win = 512
    freqs = rfft_frequencies(stdft_win, 16000)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(stdft_win)
    r_time = rng.standard_normal(stdft_win)
    r_tf = np.ones((1, freqs.size))

    def fake_relevance(_model, _processor, _wav, _device):
        return x.copy(), r_time.copy(), 1.0

    scored: list[int] = []

    def fake_score(_model, waves, _device, batch_size=16):
        scored.append(len(waves))
        return np.arange(len(waves), dtype=float)

    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.relevance_for_clip",
        fake_relevance,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.score_ssl_input_waves",
        fake_score,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.load_frequency_edges",
        lambda _root, _language: edges,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.build_ssl_detector",
        lambda **kwargs: (object(), object(), kwargs, {}),
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.stdft_lrp",
        lambda *_args, **_kwargs: (None, freqs, r_tf, None),
    )
    monkeypatch.setattr(
        "brspeech_xai.encoder_suite._production_encoder_factory",
        lambda profile, spec, device, **kwargs: type(
            "Embedder",
            (),
            {"_model": object(), "_processor": object(), "device": device},
        )(),
    )

    run_layerwise_faithfulness(
        root,
        profiles=["hubert_base"],
        languages=["eng"],
        per_class=2,
        ks=[1],
        n_random=2,
        seed=5,
        device="cpu",
        skip_if_complete=False,
        build_detector_fn=lambda **kwargs: (object(), object(), kwargs, {}),
        relevance_fn=fake_relevance,
        stdft_fn=lambda *_args, **_kwargs: (None, freqs, r_tf, None),
        audio_loader=lambda path: (np.zeros(stdft_win, dtype=np.float32), 16000),
        stdft_win=stdft_win,
    )
    paths = LayerwiseSuitePaths(root)
    destination = paths.layer_faithfulness_cell("hubert_base", FINAL_LAYER, "eng", "eng")
    generation = destination / "generations" / json.loads((destination / "active.json").read_text())["generation"]
    manifest = json.loads((generation / "faithfulness_run_manifest.json").read_text())
    assert manifest["layer"] == FINAL_LAYER
    assert manifest["source"] == manifest["target"] == "eng"
    assert manifest["per_class"] == 2
    assert manifest["rms_match"] is True
    assert manifest["sample_ids"]
    assert "intervention_filter_confound" in manifest["limitations"]
    clip_rows = pd.read_csv(generation / "clip_interventions.csv")
    assert set(clip_rows["condition"]) >= {
        "keep_dftlrp",
        "delete_dftlrp",
        "keep_dftlrp_bottom",
        "delete_dftlrp_bottom",
        "keep_random",
        "delete_random",
    }
    assert (clip_rows["sample_id"].nunique() == 4)
    assert scored and scored[0] == 4 + 2 * 2


def test_should_skip_completed_run_when_fingerprint_matches(tmp_path):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)
    destination = paths.layer_faithfulness_cell("hubert_base", FINAL_LAYER, "eng", "eng")
    cohort = load_authoritative_xai_cohort(
        paths, "hubert_base", FINAL_LAYER, "eng", "eng"
    )
    per_class = 2
    selected = select_faithfulness_cohort(cohort, per_class=per_class, seed=9)
    fingerprint = build_faithfulness_fingerprint(
        checkpoint="facebook/hubert-base-ls960",
        layer=FINAL_LAYER,
        source="eng",
        target="eng",
        ks=(1, 2),
        n_random=3,
        seed=9,
        per_class=per_class,
        rms_match=True,
        stdft_win=512,
        stdft_hop=128,
        eps=1e-9,
        sample_ids=selected["sample_id"].tolist(),
        xai_generation_id=active_xai_generation_id(
            paths, "hubert_base", FINAL_LAYER, "eng", "eng"
        ),
        frequency_band_config={
            "config_path": str(tmp_path / "eng.yaml"),
            "n_bands": 8,
            "f_min": 20.0,
            "f_max": 7900.0,
        },
        probe_artifacts=probe_artifact_identity(
            paths, "hubert_base", FINAL_LAYER, "eng"
        ),
    )

    def writer(directory: Path) -> None:
        (directory / "faithfulness_run_manifest.json").write_text(
            json.dumps({"fingerprint": fingerprint, "layer": FINAL_LAYER}),
            encoding="utf-8",
        )

    publish_generation(destination, writer, role="layer_faithfulness")
    assert should_skip_completed_run(destination, fingerprint)
    assert not should_skip_completed_run(destination, fingerprint + "x")


def test_default_per_class_is_eight():
    assert DEFAULT_FAITHFULNESS_PER_CLASS == 8


def test_validate_faithfulness_params_rejects_overlapping_top_bottom_k():
    cohort = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "y_true": [0, 0, 1, 1],
            "processed_path": ["/a", "/b", "/c", "/d"],
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_faithfulness_params(
            cohort=cohort,
            per_class=2,
            ks=[5],
            n_random=1,
            batch_size=8,
            eps=1e-9,
            stdft_win=512,
            stdft_hop=128,
            n_bands=8,
        )


def test_cell_builds_encoder_and_audio_loader_once(tmp_path, monkeypatch):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    cohort = load_authoritative_xai_cohort(
        paths, "hubert_base", FINAL_LAYER, "eng", "eng"
    )
    stdft_win = 512
    freqs = rfft_frequencies(stdft_win, 16000)
    r_tf = np.ones((1, freqs.size))
    x = np.zeros(stdft_win, dtype=np.float64)
    factory_calls = {"encoder": 0, "loader": 0}

    def counting_factory(profile, spec, device, **kwargs):
        factory_calls["encoder"] += 1
        return type(
            "Embedder",
            (),
            {"_model": object(), "_processor": object(), "device": device},
        )()

    def counting_loader(embedder):
        factory_calls["loader"] += 1
        return lambda path: (np.zeros(stdft_win, dtype=np.float32), 16000)

    monkeypatch.setattr(
        "brspeech_xai.encoder_suite._production_encoder_factory",
        counting_factory,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.build_cell_audio_loader",
        counting_loader,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.load_frequency_band_config",
        lambda _root, _lang: (
            {"config_path": "cfg", "n_bands": 8, "f_min": 20.0, "f_max": 7900.0},
            np.array([0.0, 4000.0, 8000.0]),
        ),
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.build_ssl_detector",
        lambda **kwargs: (object(), object(), kwargs, {}),
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.relevance_for_clip",
        lambda *_args, **_kwargs: (x, x, 1.0),
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.process_ssl_intervention_clip",
        lambda *_args, **_kwargs: [("delete_dftlrp", 1, 0.5)],
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.stdft_lrp",
        lambda *_args, **_kwargs: (None, freqs, r_tf, None),
    )
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)

    run_faithfulness_cell(
        suite_root=root,
        paths=paths,
        profile="hubert_base",
        layer=FINAL_LAYER,
        source="eng",
        target="eng",
        cohort=cohort,
        ks=[1],
        n_random=1,
        seed=3,
        per_class=2,
        rms_match=True,
        stdft_win=stdft_win,
        stdft_hop=128,
        eps=1e-9,
        device="cpu",
        batch_size=4,
        skip_if_complete=False,
    )
    assert factory_calls == {"encoder": 1, "loader": 1}


def test_fingerprint_changes_when_sample_ids_change(tmp_path):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)
    cohort = load_authoritative_xai_cohort(
        paths, "hubert_base", FINAL_LAYER, "eng", "eng"
    )
    common = dict(
        checkpoint="facebook/hubert-base-ls960",
        layer=FINAL_LAYER,
        source="eng",
        target="eng",
        ks=(1,),
        n_random=1,
        seed=1,
        per_class=2,
        rms_match=True,
        stdft_win=512,
        stdft_hop=128,
        eps=1e-9,
        xai_generation_id=active_xai_generation_id(
            paths, "hubert_base", FINAL_LAYER, "eng", "eng"
        ),
        frequency_band_config={
            "config_path": str(tmp_path / "eng.yaml"),
            "n_bands": 8,
            "f_min": 20.0,
            "f_max": 7900.0,
        },
        probe_artifacts=probe_artifact_identity(
            paths, "hubert_base", FINAL_LAYER, "eng"
        ),
    )
    first_ids = select_faithfulness_cohort(cohort, per_class=2, seed=1)["sample_id"].tolist()
    second_ids = select_faithfulness_cohort(cohort, per_class=2, seed=2)["sample_id"].tolist()
    assert build_faithfulness_fingerprint(**common, sample_ids=first_ids) != build_faithfulness_fingerprint(
        **common, sample_ids=second_ids
    )


def test_fingerprint_changes_when_probe_joblib_changes(tmp_path):
    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)
    cohort = load_authoritative_xai_cohort(
        paths, "hubert_base", FINAL_LAYER, "eng", "eng"
    )
    base = dict(
        checkpoint="facebook/hubert-base-ls960",
        layer=FINAL_LAYER,
        source="eng",
        target="eng",
        ks=(1,),
        n_random=1,
        seed=1,
        per_class=2,
        rms_match=True,
        stdft_win=512,
        stdft_hop=128,
        eps=1e-9,
        sample_ids=select_faithfulness_cohort(cohort, per_class=2, seed=1)[
            "sample_id"
        ].tolist(),
        xai_generation_id=active_xai_generation_id(
            paths, "hubert_base", FINAL_LAYER, "eng", "eng"
        ),
        frequency_band_config={
            "config_path": str(tmp_path / "eng.yaml"),
            "n_bands": 8,
            "f_min": 20.0,
            "f_max": 7900.0,
        },
    )
    first = build_faithfulness_fingerprint(
        **base,
        probe_artifacts=probe_artifact_identity(
            paths, "hubert_base", FINAL_LAYER, "eng"
        ),
    )
    paths.probe("hubert_base", FINAL_LAYER, "eng").write_bytes(b"refit-probe")
    second = build_faithfulness_fingerprint(
        **base,
        probe_artifacts=probe_artifact_identity(
            paths, "hubert_base", FINAL_LAYER, "eng"
        ),
    )
    assert first != second


def test_build_ssl_detector_runs_calibration_equivalence(tmp_path, monkeypatch):
    import torch

    root = _minimal_suite(tmp_path, languages=("eng",))
    paths = LayerwiseSuitePaths(root)
    _write_probe_and_calibration(paths, "hubert_base", "eng", FINAL_LAYER)
    torch_encoder = torch.nn.Linear(1, 1, bias=False)
    encoder = type(
        "Embedder",
        (),
        {"_model": torch_encoder, "_processor": object()},
    )()
    monkeypatch.setattr(
        "brspeech_xai.attnlrp.ensure_ssl_encoder_attnlrp",
        lambda *args, **kwargs: {},
    )
    model, _processor, _spec, diagnostics = build_ssl_detector(
        profile="hubert_base",
        layer=FINAL_LAYER,
        source="eng",
        paths=paths,
        encoder=encoder,
        device="cpu",
        seed=5,
    )
    from brspeech_xai.lrp_detector import SSLDetectorAD

    assert isinstance(model, SSLDetectorAD)
    assert diagnostics["calibration_equivalence_n"] > 0
    assert diagnostics["max_score_equivalence_error"] >= 0.0
    assert "d_ad_joblib_sha256" in diagnostics["probe_artifacts"]


def test_profile_encoder_reused_across_languages_and_released(tmp_path, monkeypatch):
    root = _minimal_suite(tmp_path, languages=("eng", "por"))
    factory_calls: list[int] = []
    released: list[bool] = []

    def counting_factory(profile, spec, device, **kwargs):
        factory_calls.append(1)
        return type(
            "Embedder",
            (),
            {"_model": object(), "_processor": object(), "device": device},
        )()

    def track_release(encoder, *, device):
        released.append(encoder is not None)

    seen_encoder_ids: list[int] = []

    def track_cell(**kwargs):
        seen_encoder_ids.append(id(kwargs.get("encoder")))
        return {"status": "tracked", "n_clips": 0}

    monkeypatch.setattr(
        "brspeech_xai.encoder_suite._production_encoder_factory",
        counting_factory,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.release_profile_encoder",
        track_release,
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.run_faithfulness_cell",
        track_cell,
    )

    run_layerwise_faithfulness(
        root,
        profiles=["hubert_base"],
        languages=["eng", "por"],
    )
    assert factory_calls == [1]
    assert len(seen_encoder_ids) == 2
    assert seen_encoder_ids[0] == seen_encoder_ids[1]
    assert released == [True]


def test_profile_encoder_released_when_a_cell_raises(tmp_path, monkeypatch):
    root = _minimal_suite(tmp_path, languages=("eng", "por"))
    released: list[bool] = []
    attempts = {"count": 0}

    monkeypatch.setattr(
        "brspeech_xai.encoder_suite._production_encoder_factory",
        lambda *args, **kwargs: type(
            "Embedder",
            (),
            {"_model": object(), "_processor": object()},
        )(),
    )
    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.release_profile_encoder",
        lambda encoder, device: released.append(encoder is not None),
    )

    def fail_on_second(**kwargs):
        attempts["count"] += 1
        if attempts["count"] > 1:
            raise RuntimeError("cell failed")
        return {"status": "ok"}

    monkeypatch.setattr(
        "brspeech_xai.layerwise_faithfulness.run_faithfulness_cell",
        fail_on_second,
    )
    with pytest.raises(RuntimeError, match="cell failed"):
        run_layerwise_faithfulness(
            root,
            profiles=["hubert_base"],
            languages=["eng", "por"],
        )
    assert released == [True]
