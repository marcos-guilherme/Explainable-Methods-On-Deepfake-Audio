"""TDD contract for layer-wise source-to-target probes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest

from brspeech_xai.layerwise_paths import LayerwiseSuitePaths
from brspeech_xai.layerwise_probes import run_layerwise_probe_matrix


LANGUAGES = ("eng", "por", "zho")
ROLES = ("train", "calibration", "test")


def _inputs(*, zho_test_labels: tuple[int, ...] = (0, 1, 0, 1)):
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    catalogs: dict[str, dict[str, pd.DataFrame]] = {}
    for language_index, language in enumerate(LANGUAGES):
        embeddings[language] = {}
        catalogs[language] = {}
        for role_index, role in enumerate(ROLES):
            values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
            embeddings[language][role] = (
                values + language_index * 0.1 + role_index * 0.01
            ).astype(np.float32)
            labels = np.array([0, 1, 0, 1], dtype=np.int64)
            if language == "zho" and role == "test":
                labels = np.asarray(zho_test_labels, dtype=np.int64)
            catalogs[language][role] = pd.DataFrame(
                {
                    "sample_id": [
                        f"{language}-{role}-{index}" for index in range(4)
                    ],
                    "label": labels,
                }
            )
    return embeddings, catalogs


def _injected_functions():
    fit = Mock(
        side_effect=lambda x, y, *, head, seed: {
            "mean": float(np.mean(x)),
            "labels": tuple(int(value) for value in y),
            "head": head,
            "seed": seed,
        }
    )

    def score(head, x):
        return np.clip(np.asarray(x)[:, 0] / 30.0 + head["mean"] / 100.0, 0, 1)

    score_mock = Mock(side_effect=score)
    calibrate = Mock(
        side_effect=lambda scores, labels, source_split, detector: {
            "detector": detector,
            "source_split": source_split,
            "method": "eer",
            "threshold": float(np.median(scores)),
            "calibration_eer": 0.25,
            "n": len(labels),
            "counts": {
                "bonafide": int((np.asarray(labels) == 0).sum()),
                "spoof": int((np.asarray(labels) == 1).sum()),
            },
        }
    )
    return fit, score_mock, calibrate


def _run(tmp_path: Path, *, zho_test_labels=(0, 1, 0, 1)):
    embeddings, catalogs = _inputs(zho_test_labels=zho_test_labels)
    fit, score, calibrate = _injected_functions()
    performance = run_layerwise_probe_matrix(
        profile_id="hubert_base",
        embeddings=embeddings,
        catalogs=catalogs,
        paths=LayerwiseSuitePaths(tmp_path),
        layers=(1, 2),
        seed=17,
        fit_head_fn=fit,
        score_head_fn=score,
        calibrate_threshold_fn=calibrate,
    )
    return performance, fit, score, calibrate, embeddings, catalogs


def _duplicate_test_ids(_embeddings, catalogs):
    catalogs["por"]["test"].loc[:, "sample_id"] = ["duplicate"] * 4


def _remove_train_class(_embeddings, catalogs):
    catalogs["eng"]["train"].loc[:, "label"] = [0, 0, 0, 0]


def test_two_layers_fit_six_probes_and_write_eighteen_cells(tmp_path):
    performance, fit, score, calibrate, embeddings, catalogs = _run(tmp_path)

    assert fit.call_count == 2 * 3
    assert calibrate.call_count == 2 * 3
    assert score.call_count == 2 * 3 * (1 + 3)
    assert len(performance) == 2 * 3 * 3
    assert set(performance.columns) >= {
        "profile",
        "layer",
        "source",
        "target",
        "n",
        "threshold",
        "auc",
        "accuracy",
        "corpus_shift",
    }

    paths = LayerwiseSuitePaths(tmp_path)
    for layer in (1, 2):
        for source in LANGUAGES:
            assert paths.probe("hubert_base", layer, source).is_file()
            assert paths.thresholds("hubert_base", layer, source).is_file()
            for target in LANGUAGES:
                cell = paths.cell("hubert_base", layer, source, target)
                assert (cell / "scores.npy").is_file()
                assert (cell / "predictions.parquet").is_file()
                assert (cell / "metrics.json").is_file()
                predictions = pd.read_parquet(cell / "predictions.parquet")
                assert predictions["sample_id"].tolist() == catalogs[target]["test"][
                    "sample_id"
                ].tolist()
                np.testing.assert_array_equal(
                    predictions["y_true"], catalogs[target]["test"]["label"]
                )
                np.testing.assert_allclose(
                    predictions["score"], np.load(cell / "scores.npy")
                )
                threshold = performance.loc[
                    (performance["layer"] == layer)
                    & (performance["source"] == source)
                    & (performance["target"] == target),
                    "threshold",
                ].iloc[0]
                np.testing.assert_array_equal(
                    predictions["prediction"],
                    (predictions["score"] >= threshold).astype(int),
                )

    for call_index, call in enumerate(fit.call_args_list):
        layer = (1, 2)[call_index // 3]
        source = LANGUAGES[call_index % 3]
        np.testing.assert_array_equal(
            call.args[0], embeddings[source]["train"][:, layer, :]
        )
        np.testing.assert_array_equal(
            call.args[1], catalogs[source]["train"]["label"].to_numpy()
        )


def test_test_label_change_cannot_change_heads_or_thresholds(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, *_ = _run(first_root, zho_test_labels=(0, 1, 0, 1))
    second, *_ = _run(second_root, zho_test_labels=(0, 0, 1, 1))

    first_paths = LayerwiseSuitePaths(first_root)
    second_paths = LayerwiseSuitePaths(second_root)
    for layer in (1, 2):
        for source in LANGUAGES:
            assert (
                first_paths.probe("hubert_base", layer, source).read_bytes()
                == second_paths.probe("hubert_base", layer, source).read_bytes()
            )
            assert (
                first_paths.thresholds("hubert_base", layer, source).read_bytes()
                == second_paths.thresholds("hubert_base", layer, source).read_bytes()
            )
            assert joblib.load(first_paths.probe("hubert_base", layer, source)) == (
                joblib.load(second_paths.probe("hubert_base", layer, source))
            )

    unchanged = ["profile", "layer", "source", "target", "n", "threshold"]
    pd.testing.assert_frame_equal(first[unchanged], second[unchanged])
    assert not first.loc[first["target"] == "zho", "accuracy"].equals(
        second.loc[second["target"] == "zho", "accuracy"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda emb, cat: emb["eng"].pop("train"), "missing|train"),
        (
            lambda emb, cat: emb["eng"].__setitem__(
                "train", emb["eng"]["train"].astype(np.float64)
            ),
            "float32",
        ),
        (
            lambda emb, cat: emb["eng"]["train"].__setitem__((0, 0, 0), np.nan),
            "finite",
        ),
        (_duplicate_test_ids, "unique"),
        (
            lambda emb, cat: cat["por"].__setitem__(
                "test", cat["por"]["test"].iloc[:-1].copy()
            ),
            "length|rows",
        ),
        (_remove_train_class, "classes"),
    ],
)
def test_invalid_inputs_fail_before_fit(tmp_path, mutation, message):
    embeddings, catalogs = _inputs()
    mutation(embeddings, catalogs)
    fit, score, calibrate = _injected_functions()

    with pytest.raises((ValueError, KeyError), match=message):
        run_layerwise_probe_matrix(
            profile_id="hubert_base",
            embeddings=embeddings,
            catalogs=catalogs,
            paths=LayerwiseSuitePaths(tmp_path),
            layers=(1, 2),
            fit_head_fn=fit,
            score_head_fn=score,
            calibrate_threshold_fn=calibrate,
        )

    fit.assert_not_called()


def test_rejects_unknown_language_role_and_layer(tmp_path):
    embeddings, catalogs = _inputs()
    fit, score, calibrate = _injected_functions()
    embeddings["fra"] = embeddings.pop("zho")
    catalogs["fra"] = catalogs.pop("zho")

    with pytest.raises(ValueError, match="language"):
        run_layerwise_probe_matrix(
            "hubert_base",
            embeddings,
            catalogs,
            LayerwiseSuitePaths(tmp_path),
            layers=(1, 2),
            fit_head_fn=fit,
            score_head_fn=score,
            calibrate_threshold_fn=calibrate,
        )

    embeddings, catalogs = _inputs()
    embeddings["eng"]["validation"] = embeddings["eng"].pop("calibration")
    catalogs["eng"]["validation"] = catalogs["eng"].pop("calibration")
    with pytest.raises(ValueError, match="role"):
        run_layerwise_probe_matrix(
            "hubert_base",
            embeddings,
            catalogs,
            LayerwiseSuitePaths(tmp_path),
            layers=(1, 2),
            fit_head_fn=fit,
            score_head_fn=score,
            calibrate_threshold_fn=calibrate,
        )

    embeddings, catalogs = _inputs()
    with pytest.raises(ValueError, match="layer"):
        run_layerwise_probe_matrix(
            "hubert_base",
            embeddings,
            catalogs,
            LayerwiseSuitePaths(tmp_path),
            layers=(3,),
            fit_head_fn=fit,
            score_head_fn=score,
            calibrate_threshold_fn=calibrate,
        )


def test_rejects_layer_13_before_fit_or_calibration(tmp_path):
    embeddings, catalogs = _inputs()
    for language in LANGUAGES:
        for role in ROLES:
            embeddings[language][role] = np.repeat(
                embeddings[language][role][:, :1, :],
                repeats=14,
                axis=1,
            )
    fit, score, calibrate = _injected_functions()

    with pytest.raises(ValueError, match="layer.*1.*12"):
        run_layerwise_probe_matrix(
            "hubert_base",
            embeddings,
            catalogs,
            LayerwiseSuitePaths(tmp_path),
            layers=(13,),
            fit_head_fn=fit,
            score_head_fn=score,
            calibrate_threshold_fn=calibrate,
        )

    fit.assert_not_called()
    calibrate.assert_not_called()
