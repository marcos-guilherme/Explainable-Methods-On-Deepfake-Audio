from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from brspeech_xai.config import (
    AdaptConfig,
    AssociationConfig,
    BandsConfig,
    DataConfig,
    OcclusionConfig,
    RunConfig,
)
from brspeech_xai.matrix import (
    LANGUAGES,
    MatrixPaths,
    build_parser,
    prepare_matrix_inputs,
    run_matrix_from_embeddings,
)
from brspeech_xai.matrix_xai import run_xai_scopes


def _configs(manifest_dir=None) -> dict[str, RunConfig]:
    return {
        language: RunConfig(
            seed=7,
            data=DataConfig(
                dataset_kind="local_manifest",
                manifest_path=str(manifest_dir / f"{language}.csv")
                if manifest_dir is not None
                else f"{language}.csv",
                calibration_split="calibration",
                n_train_per_class=6,
                n_calibration_per_class=4,
                n_test_per_class=5,
            ),
        )
        for language in LANGUAGES
    }


def _write_precomputed(
    root,
    *,
    flip_test_labels_for: str | None = None,
    with_audio: bool = False,
) -> MatrixPaths:
    paths = MatrixPaths(root)
    offsets = {"eng": 0.0, "por": 0.3, "zho": -0.2}
    sizes = {"train": 12, "calibration": 8, "test": 10}
    for language in LANGUAGES:
        for role, n in sizes.items():
            y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
            if role == "test" and language == flip_test_labels_for:
                y = 1 - y
            rng = np.random.default_rng(100 + LANGUAGES.index(language) * 10 + list(sizes).index(role))
            emb = rng.normal(size=(n, 5))
            emb[:, 0] += y * 2.5 + offsets[language]
            paths.embedding(language, role).parent.mkdir(parents=True, exist_ok=True)
            np.save(paths.embedding(language, role), emb.astype(np.float32))
            catalog = pd.DataFrame(
                {
                    "sample_id": [f"{language}-{role}-{i}" for i in range(n)],
                    "label": y,
                    "language": language,
                    "role": role,
                }
            )
            if with_audio and role == "test":
                audio_dir = paths.root / "fixture_audio" / language
                audio_dir.mkdir(parents=True, exist_ok=True)
                processed_paths = []
                timeline = np.arange(3200) / 16000
                for i, label in enumerate(y):
                    wav_path = audio_dir / f"{i}.wav"
                    frequency = 220 + 500 * int(label) + 13 * i
                    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline)
                    sf.write(wav_path, waveform, 16000, subtype="PCM_16")
                    processed_paths.append(str(wav_path))
                catalog["processed_path"] = processed_paths
            paths.catalog(language, role).parent.mkdir(parents=True, exist_ok=True)
            catalog.to_parquet(paths.catalog(language, role), index=False)
    return paths


def test_matrix_fits_three_models_and_writes_nine_cells(tmp_path, monkeypatch):
    paths = _write_precomputed(tmp_path)
    calls = []

    from brspeech_xai import matrix as matrix_module

    real_fit = matrix_module.fit_head

    def _counted_fit(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(matrix_module, "fit_head", _counted_fit)
    performance = run_matrix_from_embeddings(_configs(), paths)

    assert len(calls) == 3
    assert len(performance) == 9
    assert set(zip(performance["source"], performance["target"])) == {
        (source, target) for source in LANGUAGES for target in LANGUAGES
    }
    for source in LANGUAGES:
        assert paths.model(source).is_file()
        assert paths.thresholds(source).is_file()
        for target in LANGUAGES:
            assert paths.cell_scores(source, target).is_file()
            assert paths.cell_metrics(source, target).is_file()
            assert paths.cell_predictions(source, target).is_file()


def test_target_labels_cannot_change_source_model_or_threshold(tmp_path):
    cfgs = _configs()
    paths_a = _write_precomputed(tmp_path / "a")
    run_matrix_from_embeddings(cfgs, paths_a)
    head_a = joblib.load(paths_a.model("eng"))
    threshold_a = json.loads(paths_a.thresholds("eng").read_text())["detectors"]["ad"]["threshold"]

    paths_b = _write_precomputed(tmp_path / "b", flip_test_labels_for="zho")
    run_matrix_from_embeddings(cfgs, paths_b)
    head_b = joblib.load(paths_b.model("eng"))
    threshold_b = json.loads(paths_b.thresholds("eng").read_text())["detectors"]["ad"]["threshold"]

    np.testing.assert_allclose(
        head_a.named_steps["logisticregression"].coef_,
        head_b.named_steps["logisticregression"].coef_,
    )
    assert threshold_a == threshold_b


def test_each_cell_uses_its_source_threshold(tmp_path):
    paths = _write_precomputed(tmp_path)
    performance = run_matrix_from_embeddings(_configs(), paths)

    for row in performance.itertuples(index=False):
        source_threshold = json.loads(
            paths.thresholds(row.source).read_text()
        )["detectors"]["ad"]["threshold"]
        metrics = json.loads(paths.cell_metrics(row.source, row.target).read_text())
        predictions = pd.read_parquet(paths.cell_predictions(row.source, row.target))
        assert metrics["source"] == row.source
        assert metrics["target"] == row.target
        expected_scope = "diagonal" if row.source == row.target else "cross_corpus_shift"
        assert metrics["condition"] == expected_scope
        assert row.scope == expected_scope
        assert metrics["fixed_threshold"]["threshold"] == source_threshold
        np.testing.assert_array_equal(
            predictions["prediction"].to_numpy(),
            (predictions["p_spoof"].to_numpy() >= source_threshold).astype(int),
        )


def test_prepare_inputs_embeds_each_language_role_once_and_reuses_cache(
    tmp_path, monkeypatch
):
    load_calls = []

    def _fake_load(_dataset_id, role, _quota, **kwargs):
        language = Path(kwargs["manifest_path"]).stem
        load_calls.append((language, role))
        audios = [
            np.full(160, 0.1, dtype=np.float32),
            np.full(160, 0.2, dtype=np.float32),
            np.full(160, 0.8, dtype=np.float32),
            np.full(160, 0.9, dtype=np.float32),
        ]
        labels = [0, 0, 1, 1]
        provenance = [
            {
                "sample_id": f"{language}-{role}-{i}",
                "language": language,
                "role": role,
                "label": label,
            }
            for i, label in enumerate(labels)
        ]
        return audios, [16000] * 4, labels, provenance

    class _Embedder:
        def __init__(self):
            self.calls = 0

        def extract_embeddings(self, audios, _srs):
            self.calls += 1
            return np.array([[float(np.mean(audio)), i] for i, audio in enumerate(audios)])

    monkeypatch.setattr("brspeech_xai.data.build_balanced_split", _fake_load)
    embedder = _Embedder()
    for language in LANGUAGES:
        (tmp_path / f"{language}.csv").write_text(f"manifest-{language}")
    configs = _configs(tmp_path)
    paths = MatrixPaths(tmp_path / "output")
    prepare_matrix_inputs(configs, paths, embedder=embedder)
    prepare_matrix_inputs(configs, paths, embedder=embedder)

    assert len(load_calls) == 9
    assert embedder.calls == 9
    assert set(load_calls) == {
        (language, role)
        for language in LANGUAGES
        for role in ("train", "calibration", "test")
    }

    (tmp_path / "por.csv").write_text("manifest-por-updated")
    prepare_matrix_inputs(configs, paths, embedder=embedder)
    assert len(load_calls) == 12
    assert embedder.calls == 12


def test_matrix_rejects_incomparable_configs(tmp_path):
    paths = _write_precomputed(tmp_path)
    configs = _configs()
    configs["por"].adapt = AdaptConfig(head="mlp")
    with np.testing.assert_raises_regex(ValueError, "head"):
        run_matrix_from_embeddings(configs, paths)


def test_xai_scopes_run_full_diagonal_and_reduced_off_diagonal(tmp_path, monkeypatch):
    paths = _write_precomputed(tmp_path, with_audio=True)
    configs = _configs()
    run_matrix_from_embeddings(configs, paths)
    calls = []

    def _record(name):
        def _stage(ctx):
            calls.append((name, ctx.paths.root.relative_to(paths.root).as_posix()))

        return _stage

    for name in ("features", "association", "occlusion", "confirmatory", "report"):
        monkeypatch.setattr(f"brspeech_xai.matrix_xai.stage_{name}", _record(name))

    run_xai_scopes(configs, paths, embedder=object(), reduced_per_quadrant=2)

    assert sum(name == "features" for name, _ in calls) == 9
    assert sum(name == "association" for name, _ in calls) == 9
    assert sum(name == "occlusion" for name, _ in calls) == 9
    assert sum(name == "confirmatory" for name, _ in calls) == 3
    assert sum(name == "report" for name, _ in calls) == 3
    for language in LANGUAGES:
        scope = json.loads((paths.root / "xai" / language / "scope.json").read_text())
        assert scope["scope"] == "full_diagonal_xai"
        assert scope["confirmatory_h3"] is True
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            scope = json.loads(
                (paths.root / "shift" / f"{source}_to_{target}" / "scope.json").read_text()
            )
            assert scope["scope"] == "reduced_cross_corpus_shift"
            assert scope["confirmatory_h3"] is False
            assert scope["model_and_threshold_from"] == source
            assert scope["evaluation_data_from"] == target


def test_matrix_cli_exposes_optional_xai_scopes():
    args = build_parser().parse_args(
        [
            "--eng-config",
            "eng.yaml",
            "--por-config",
            "por.yaml",
            "--zho-config",
            "zho.yaml",
            "--output",
            "out",
            "--with-xai",
            "--reduced-per-quadrant",
            "3",
            "--xai-plots",
        ]
    )
    assert args.with_xai is True
    assert args.reduced_per_quadrant == 3
    assert args.xai_plots is True


def test_real_xai_scope_smoke_on_nine_cells(tmp_path):
    paths = _write_precomputed(tmp_path, with_audio=True)
    configs = _configs()
    for cfg in configs.values():
        cfg.bands = BandsConfig(n_bands=3, f_min=20.0, f_max=7000.0)
        cfg.occlusion = OcclusionConfig(per_quadrant=1, n_boot=10, convergence_k=1)
        cfg.association = AssociationConfig(top_n=2)
    run_matrix_from_embeddings(configs, paths)

    class _Embedder:
        # A matriz não publica D_zs; o XAI deve usar apenas detectores presentes
        # mesmo quando o encoder subjacente oferece zero-shot.
        has_zero_shot = True
        name = "fixture"

        def extract_embeddings(self, audios, _srs):
            rows = []
            for audio in audios:
                audio = np.asarray(audio)
                spectrum = np.abs(np.fft.rfft(audio))
                rows.append(
                    [
                        float(np.mean(audio)),
                        float(np.std(audio)),
                        float(np.max(np.abs(audio))),
                        float(np.mean(spectrum[: max(1, len(spectrum) // 4)])),
                        float(np.mean(spectrum[max(1, len(spectrum) // 4) :])),
                    ]
                )
            return np.asarray(rows, dtype=np.float32)

    run_xai_scopes(
        configs,
        paths,
        embedder=_Embedder(),
        reduced_per_quadrant=1,
    )

    for language in LANGUAGES:
        diagonal = paths.root / "xai" / language
        assert (diagonal / "confirmatory_tests.csv").is_file()
        assert (diagonal / "performance_table.csv").is_file()
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source != target:
                shift = paths.root / "shift" / f"{source}_to_{target}"
                assert (shift / "spearman_table.csv").is_file()
                assert (shift / "occlusion_table.csv").is_file()
                assert not (shift / "confirmatory_tests.csv").exists()
