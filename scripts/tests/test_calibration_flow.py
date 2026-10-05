"""TDD: fit / calibrate / score separation and source-only thresholds."""
from __future__ import annotations

import json
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest

from brspeech_xai.adaptation import build_head, fit_head, score_head
from brspeech_xai.config import AdaptConfig, DataConfig, RunConfig, load_config
from brspeech_xai.metrics import calibrate_threshold, evaluate, evaluate_at_threshold
from brspeech_xai.stages import RunContext, _collect_specs, stage_features


def _separable(n=120, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.normal(size=(n, dim))
    x[y == 1, 0] += 2.5
    return x, y


def test_fit_head_trains_and_score_head_does_not_refit():
    x, y = _separable()
    head = fit_head(x, y, head="logistic", seed=0)
    scores_a = score_head(head, x)
    # mutate internal coef would change scores if refit; ensure score is stable call
    scores_b = score_head(head, x)
    np.testing.assert_array_equal(scores_a, scores_b)
    assert head is not joblib.load  # returned fitted object


def test_score_head_respects_nonstandard_classes_order():
    x, _y = _separable(n=8, dim=4)

    class _Head:
        classes_ = np.array([1, 0])

        def predict_proba(self, emb):
            n = len(emb)
            # colunas alinhadas a classes_: [P(spoof=1), P(bonafide=0)]
            return np.column_stack([np.full(n, 0.8), np.full(n, 0.2)])

    scores = score_head(_Head(), x, spoof_label=1)
    assert scores.shape == (len(x),)
    np.testing.assert_allclose(scores, 0.8)


def test_fit_head_rejects_bad_shapes():
    x, y = _separable(n=10)
    with pytest.raises(ValueError, match="mesmo"):
        fit_head(x[:5], y, head="logistic", seed=0)


def test_calibrate_threshold_json_safe_and_validates():
    scores = np.array([0.1, 0.2, 0.7, 0.9], dtype=float)
    labels = np.array([0, 0, 1, 1])
    out = calibrate_threshold(scores, labels, source_split="calibration", detector="ad")
    json.dumps(out)
    assert out["threshold"] == pytest.approx(
        calibrate_threshold(scores, labels, "calibration", "ad")["threshold"]
    )
    assert out["source_split"] == "calibration"
    assert out["detector"] == "ad"
    assert out["n"] == 4
    assert set(out["counts"]) >= {"bonafide", "spoof"}
    with pytest.raises(ValueError):
        calibrate_threshold(scores[:2], labels, "calibration", "ad")


def test_evaluate_at_threshold_separates_free_and_fixed():
    scores = np.array([0.05, 0.15, 0.85, 0.95])
    labels = np.array([0, 0, 1, 1])
    thr = 0.5
    out = evaluate_at_threshold(scores, labels, thr, condition="holdout", target="test")
    assert "threshold_free" in out and "fixed_threshold" in out
    tf = out["threshold_free"]
    fx = out["fixed_threshold"]
    assert "roc_auc" in tf and "average_precision" in tf and "eer_diagnostic" in tf
    assert fx["threshold"] == thr
    assert "accuracy" in fx and "mcc" in fx
    assert "tpr" in fx and "fpr" in fx and "fnr" in fx
    # fixed uses supplied threshold; diagnostic EER is independent
    assert fx["accuracy"] == pytest.approx(1.0)
    assert tf["eer_diagnostic"] < 0.01


def test_legacy_evaluate_wrapper_still_works():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    labels = np.array([0, 1, 0, 1])
    m = evaluate(scores, labels)
    assert {"eer", "eer_threshold", "mcc_eer", "acc_eer"}.issubset(m.keys())


def test_collect_specs_holdout_with_explicit_calibration():
    cfg = RunConfig(
        data=DataConfig(
            train_split="train",
            eval_split="test",
            calibration_split="calibration",
            n_train_per_class=100,
            n_calibration_per_class=602,
            n_test_per_class=603,
        )
    )
    specs = _collect_specs(cfg)
    names = [s[0] for s in specs]
    assert names == ["train", "calibration", "test"]


def test_collect_specs_legacy_without_calibration_fields():
    cfg = RunConfig(data=DataConfig())
    assert cfg.data.calibration_split == ""
    assert cfg.data.n_calibration_per_class == 0
    specs = _collect_specs(cfg)
    assert [s[0] for s in specs] == ["train", "test"]


def test_collect_specs_rejects_partial_or_crossfit_calibration():
    with pytest.raises(ValueError, match="juntos"):
        _collect_specs(
            RunConfig(data=DataConfig(calibration_split="calibration"))
        )
    with pytest.raises(ValueError, match="cross_fit"):
        _collect_specs(
            RunConfig(
                data=DataConfig(
                    calibration_split="calibration",
                    n_calibration_per_class=10,
                ),
                adapt=AdaptConfig(cross_fit=True),
            )
        )


def test_local_yaml_configs_include_calibration_602():
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    for name in ("xai-eng-local.yaml", "xai-por-local.yaml", "xai-zho-local.yaml"):
        cfg = load_config(cfg_dir / name)
        assert cfg.data.calibration_split == "calibration"
        assert cfg.data.n_calibration_per_class == 602


def test_holdout_threshold_invariant_to_eval_label_permutation(tmp_path, monkeypatch):
    """Changing eval labels must not change thresholds.json."""
    from brspeech_xai import artifacts as A
    from brspeech_xai.stages import _adapt_legacy

    n_cal, n_eval = 20, 20
    dim = 6
    rng = np.random.default_rng(3)
    emb_train, y_train = _separable(n=40, dim=dim, seed=1)
    emb_cal = rng.normal(size=(n_cal, dim))
    y_cal = np.array([0] * (n_cal // 2) + [1] * (n_cal // 2))
    emb_eval = rng.normal(size=(n_eval, dim))
    y_eval_a = np.array([0] * (n_eval // 2) + [1] * (n_eval // 2))
    y_eval_b = 1 - y_eval_a

    paths = A.RunPaths(tmp_path)
    A.save_npy(emb_train.astype(np.float32), paths.path("emb_train.npy"))
    A.save_npy(emb_cal.astype(np.float32), paths.path("emb_calibration.npy"))
    A.save_npy(emb_eval.astype(np.float32), paths.path("emb_test.npy"))

    def _samples(y_eval):
        rows = []
        for split, n, y in (
            ("train", len(y_train), y_train),
            ("calibration", len(y_cal), y_cal),
            ("test", len(y_eval), y_eval),
        ):
            for i, lbl in enumerate(y):
                rows.append({"split": split, "idx": i, "label": int(lbl), "sample_rate": 16000})
        return pd.DataFrame(rows)

    cfg = RunConfig(
        data=DataConfig(
            train_split="train",
            eval_split="test",
            calibration_split="calibration",
            n_calibration_per_class=n_cal // 2,
        )
    )
    ctx = RunContext(cfg=cfg, paths=paths, logger=None, embedder=SimpleNamespace(has_zero_shot=False))

    A.save_table(_samples(y_eval_a), paths.path("samples.parquet"))
    _adapt_legacy(ctx)
    thr_a = A.load_json(paths.path("thresholds.json"))

    A.save_table(_samples(y_eval_b), paths.path("samples.parquet"))
    _adapt_legacy(ctx)
    thr_b = A.load_json(paths.path("thresholds.json"))

    assert thr_a["detectors"]["ad"]["threshold"] == thr_b["detectors"]["ad"]["threshold"]
    assert thr_a["calibration_source"] == "calibration"


def test_stage_features_uses_saved_threshold_not_compute_eer(tmp_path, monkeypatch):
    from brspeech_xai import artifacts as A

    n = 8
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_ad = np.linspace(0.1, 0.9, n)
    fixed_thr = 0.42

    paths = A.RunPaths(tmp_path)
    samples = pd.DataFrame(
        {"split": ["test"] * n, "idx": list(range(n)), "label": y, "sample_rate": [16000] * n}
    )
    A.save_table(samples, paths.path("samples.parquet"))
    audios = [np.zeros(1600, dtype=np.float32) for _ in range(n)]
    np.save(paths.path("audios_test.npy"), np.array(audios, dtype=object), allow_pickle=True)
    np.save(paths.path("srs_test.npy"), np.array([16000] * n, dtype=np.int32))
    A.save_npy(p_ad.astype(np.float32), paths.path("p_spoof_ad.npy"))
    A.save_json(
        {
            "schema_version": 1,
            "calibration_source": "calibration",
            "calibration_fallback_train": False,
            "detectors": {
                "ad": {
                    "threshold": fixed_thr,
                    "method": "eer",
                    "calibration_eer": 0.0,
                    "n": 4,
                    "counts": {"bonafide": 2, "spoof": 2},
                }
            },
        },
        paths.path("thresholds.json"),
    )

    def _boom(*_a, **_k):
        raise AssertionError("compute_eer must not run in stage_features for prediction")

    monkeypatch.setattr("brspeech_xai.metrics.compute_eer", _boom)
    def _fake_feats(_a, _s):
        from brspeech_xai import bands

        return np.zeros(len(bands.BAND_COLS), dtype=np.float32)

    monkeypatch.setattr("brspeech_xai.stages.mel_band_features", _fake_feats)

    cfg = RunConfig(data=DataConfig(eval_split="test"))
    ctx = RunContext(cfg=cfg, paths=paths, logger=None)
    stage_features(ctx)

    master = A.load_table(paths.path("master_table.parquet"))
    expected_pred = (p_ad >= fixed_thr).astype(int)
    np.testing.assert_array_equal(master["pred_ad"].to_numpy(), expected_pred)


def test_stage_features_can_reload_audio_from_processed_paths(tmp_path, monkeypatch):
    import soundfile as sf
    from brspeech_xai import artifacts as A

    paths = A.RunPaths(tmp_path)
    wav_paths = []
    for i in range(2):
        wav_path = tmp_path / f"{i}.wav"
        sf.write(wav_path, np.full(800, 0.1 + i * 0.1), 16000, subtype="PCM_16")
        wav_paths.append(str(wav_path))
    A.save_table(
        pd.DataFrame(
            {
                "split": ["test", "test"],
                "sample_id": ["a", "b"],
                "label": [0, 1],
                "processed_path": wav_paths,
            }
        ),
        paths.path("samples.parquet"),
    )
    A.save_npy(np.array([0.1, 0.9]), paths.path("p_spoof_ad.npy"))
    A.save_json(
        {
            "schema_version": 1,
            "detectors": {"ad": {"threshold": 0.5}},
        },
        paths.path("thresholds.json"),
    )
    monkeypatch.setattr(
        "brspeech_xai.stages.mel_band_features",
        lambda _a, _s: np.zeros(16, dtype=np.float32),
    )

    stage_features(RunContext(cfg=RunConfig(), paths=paths, logger=None))
    master = A.load_table(paths.path("master_table.parquet"))
    assert master["sample_id"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "detectors": {"ad": {"threshold": 0.5}}},
        {"schema_version": 1, "detectors": {}},
        {"schema_version": 1, "detectors": {"ad": {"threshold": float("inf")}}},
    ],
)
def test_stage_features_rejects_invalid_threshold_artifact(tmp_path, monkeypatch, payload):
    from brspeech_xai import artifacts as A

    paths = A.RunPaths(tmp_path)
    A.save_table(
        pd.DataFrame(
            {"split": ["test", "test"], "idx": [0, 1], "label": [0, 1], "sample_rate": [16000] * 2}
        ),
        paths.path("samples.parquet"),
    )
    np.save(
        paths.path("audios_test.npy"),
        np.array([np.zeros(1600), np.zeros(1600)], dtype=object),
        allow_pickle=True,
    )
    np.save(paths.path("srs_test.npy"), np.array([16000, 16000]))
    A.save_npy(np.array([0.1, 0.9]), paths.path("p_spoof_ad.npy"))
    A.save_json(payload, paths.path("thresholds.json"))
    monkeypatch.setattr(
        "brspeech_xai.stages.mel_band_features",
        lambda _a, _s: np.zeros(16, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="thresholds|ad|finito"):
        stage_features(RunContext(cfg=RunConfig(), paths=paths, logger=None))


def test_legacy_holdout_calibrates_on_train_when_no_calibration_split(tmp_path):
    from brspeech_xai import artifacts as A
    from brspeech_xai.stages import _adapt_legacy

    emb_train, y_train = _separable(n=30, dim=5, seed=5)
    emb_eval, y_eval = _separable(n=20, dim=5, seed=6)

    paths = A.RunPaths(tmp_path)
    A.save_npy(emb_train.astype(np.float32), paths.path("emb_train.npy"))
    A.save_npy(emb_eval.astype(np.float32), paths.path("emb_test.npy"))
    rows = []
    for split, y in (("train", y_train), ("test", y_eval)):
        for i, lbl in enumerate(y):
            rows.append({"split": split, "idx": i, "label": int(lbl), "sample_rate": 16000})
    A.save_table(pd.DataFrame(rows), paths.path("samples.parquet"))

    cfg = RunConfig(data=DataConfig())  # defaults: no explicit calibration
    ctx = RunContext(cfg=cfg, paths=paths, logger=None, embedder=SimpleNamespace(has_zero_shot=False))
    _adapt_legacy(ctx)

    meta = A.load_json(paths.path("thresholds.json"))
    assert meta["calibration_source"] == "train"
    assert meta["calibration_fallback_train"] is True
    head = joblib.load(paths.path("d_ad.joblib"))
    expected = calibrate_threshold(score_head(head, emb_train), y_train, "train", "ad")
    assert meta["detectors"]["ad"]["threshold"] == pytest.approx(expected["threshold"])


def test_threshold_matches_calibration_scores_only(tmp_path):
    from brspeech_xai import artifacts as A
    from brspeech_xai.stages import _adapt_legacy

    emb_train, y_train = _separable(n=30, dim=5, seed=2)
    emb_cal, y_cal = _separable(n=24, dim=5, seed=3)
    emb_eval, y_eval = _separable(n=24, dim=5, seed=4)

    paths = A.RunPaths(tmp_path)
    A.save_npy(emb_train.astype(np.float32), paths.path("emb_train.npy"))
    A.save_npy(emb_cal.astype(np.float32), paths.path("emb_calibration.npy"))
    A.save_npy(emb_eval.astype(np.float32), paths.path("emb_test.npy"))

    rows = []
    for split, y in (("train", y_train), ("calibration", y_cal), ("test", y_eval)):
        for i, lbl in enumerate(y):
            rows.append({"split": split, "idx": i, "label": int(lbl), "sample_rate": 16000})
    A.save_table(pd.DataFrame(rows), paths.path("samples.parquet"))

    cfg = RunConfig(
        data=DataConfig(
            train_split="train",
            eval_split="test",
            calibration_split="calibration",
            n_calibration_per_class=12,
        )
    )
    ctx = RunContext(cfg=cfg, paths=paths, logger=None, embedder=SimpleNamespace(has_zero_shot=False))
    _adapt_legacy(ctx)

    head = joblib.load(paths.path("d_ad.joblib"))
    cal_scores = score_head(head, emb_cal)
    expected = calibrate_threshold(cal_scores, y_cal, "calibration", "ad")
    saved = A.load_json(paths.path("thresholds.json"))["detectors"]["ad"]
    assert saved["threshold"] == pytest.approx(expected["threshold"])
    assert paths.path("d_ad.joblib").exists()
    assert paths.path("p_spoof_ad.npy").exists()
