"""TDD contract for multi-seed probe performance stability (not XAI stability)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from brspeech_xai.layerwise_paths import (
    CACHE_SCHEMA_VERSION,
    LayerwiseSuitePaths,
    sample_id_catalog_hash,
)
from brspeech_xai.layerwise_probe_stability import (
    DEFAULT_PROBE_STABILITY_SEEDS,
    STRATIFIED_TRAIN_BOOTSTRAP_METHOD,
    aggregate_probe_stability,
    load_materialized_embeddings,
    run_probe_stability,
    stratified_train_bootstrap_indices,
    validate_bootstrap_seeds,
    verify_catalog_embedding_alignment,
)


LANGUAGES = ("eng", "por", "zho")
ROLES = ("train", "calibration", "test")


def _write_execution_plan(root: Path, *, seed: int = 42) -> None:
    (root / "execution_plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": seed,
                "inputs": {
                    language: {
                        "config_path": str(root / f"{language}.yaml"),
                        "manifest_path": str(root / f"{language}.csv"),
                    }
                    for language in LANGUAGES
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_language_fixtures(root: Path) -> None:
    for language_index, language in enumerate(LANGUAGES):
        manifest = root / f"{language}.csv"
        with manifest.open("w", encoding="utf-8") as handle:
            handle.write("sample_id,language,role,label,processed_path\n")
            for role_index, role in enumerate(ROLES):
                for label in (0, 1):
                    for index in range(2):
                        sid = f"{language}-{role}-{label}-{index}"
                        handle.write(
                            f"{sid},{language},{role},{label},{sid}.wav\n"
                        )
        config = root / f"{language}.yaml"
        config.write_text(
            f"seed: 42\n"
            f"data:\n"
            f"  dataset_kind: local_manifest\n"
            f"  manifest_path: {manifest.name}\n"
            f"  train_split: train\n"
            f"  calibration_split: calibration\n"
            f"  eval_split: test\n"
            f"  n_train_per_class: 2\n"
            f"  n_calibration_per_class: 2\n"
            f"  n_test_per_class: 2\n",
            encoding="utf-8",
        )


def _materialize_embeddings(root: Path, profile: str = "hubert_base") -> None:
    paths = LayerwiseSuitePaths(root)
    for language_index, language in enumerate(LANGUAGES):
        for role_index, role in enumerate(ROLES):
            core = np.arange(104, dtype=np.float32).reshape(4, 13, 2)
            values = (core + language_index * 0.1 + role_index * 0.01).astype(
                np.float32
            )
            array_path = paths.embeddings(profile, language, role)
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_path, values, allow_pickle=False)
            meta_path = paths.embedding_metadata(profile, language, role)
            sample_ids = [
                f"{language}-{role}-{label}-{index}"
                for label in (0, 1)
                for index in range(2)
            ]
            meta_path.write_text(
                json.dumps(
                    {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "profile": profile,
                        "language": language,
                        "role": role,
                        "shape": list(values.shape),
                        "dtype": "float32",
                        "sample_ids": sample_ids,
                        "sample_ids_sha256": sample_id_catalog_hash(sample_ids),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _canonical_probe_bytes(root: Path, profile: str) -> dict[tuple[int, str], bytes]:
    paths = LayerwiseSuitePaths(root)
    out: dict[tuple[int, str], bytes] = {}
    for layer in (1, 2):
        for source in LANGUAGES:
            probe = paths.probe(profile, layer, source)
            probe.parent.mkdir(parents=True, exist_ok=True)
            payload = f"canonical-{layer}-{source}".encode()
            probe.write_bytes(payload)
            out[(layer, source)] = payload
    return out


def test_aggregate_single_seed_std_is_zero_not_nan():
    per_seed = pd.DataFrame(
        [
            {
                "seed": 42,
                "profile": "hubert_base",
                "layer": 1,
                "source": "eng",
                "target": "eng",
                "roc_auc": 0.8,
                "mcc": 0.4,
            }
        ]
    )
    summary = aggregate_probe_stability(per_seed)
    row = summary.iloc[0]
    assert int(row["n_seeds"]) == 1
    assert row["roc_auc_std"] == 0.0
    assert row["mcc_std"] == 0.0
    assert json.dumps(row.to_dict(), allow_nan=False)


def test_aggregate_reports_mean_std_and_min_max_for_roc_auc_and_mcc():
    per_seed = pd.DataFrame(
        [
            {
                "seed": 42,
                "profile": "hubert_base",
                "layer": 1,
                "source": "eng",
                "target": "eng",
                "roc_auc": 0.9,
                "mcc": 0.5,
            },
            {
                "seed": 43,
                "profile": "hubert_base",
                "layer": 1,
                "source": "eng",
                "target": "eng",
                "roc_auc": 0.7,
                "mcc": 0.3,
            },
            {
                "seed": 44,
                "profile": "hubert_base",
                "layer": 1,
                "source": "eng",
                "target": "eng",
                "roc_auc": 0.8,
                "mcc": 0.4,
            },
        ]
    )
    summary = aggregate_probe_stability(per_seed)
    row = summary.iloc[0]
    assert row["roc_auc_mean"] == pytest.approx(0.8)
    assert row["roc_auc_std"] == pytest.approx(np.std([0.9, 0.7, 0.8], ddof=1))
    assert row["roc_auc_min"] == pytest.approx(0.7)
    assert row["roc_auc_max"] == pytest.approx(0.9)
    assert row["mcc_mean"] == pytest.approx(0.4)
    assert row["mcc_min"] == pytest.approx(0.3)
    assert row["mcc_max"] == pytest.approx(0.5)
    assert int(row["n_seeds"]) == 3


def _catalogs_for_materialized() -> dict[str, dict[str, pd.DataFrame]]:
    catalogs: dict[str, dict[str, pd.DataFrame]] = {}
    for language in LANGUAGES:
        catalogs[language] = {}
        for role in ROLES:
            catalogs[language][role] = pd.DataFrame(
                {
                    "sample_id": [
                        f"{language}-{role}-{label}-{index}"
                        for label in (0, 1)
                        for index in range(2)
                    ],
                    "label": [0, 0, 1, 1],
                }
            )
    return catalogs


def test_run_probe_stability_leaves_canonical_probes_and_writes_seed_artifacts(
    tmp_path,
):
    root = tmp_path / "suite"
    root.mkdir()
    _write_execution_plan(root)
    profile = "hubert_base"
    _materialize_embeddings(root, profile)
    canonical = _canonical_probe_bytes(root, profile)

    fit_calls: list[tuple[int, float]] = []

    def fit(x, y, *, head, seed):
        fit_calls.append((seed, float(np.sum(x))))
        return {"sum_x": float(np.sum(x)), "head": head}

    def score(head, x):
        base = np.asarray(x)[:, 0] / 30.0
        return np.clip(base + head["sum_x"] * 1e-5, 0.0, 1.0)

    result = run_probe_stability(
        root,
        profiles=(profile,),
        layers=(1, 2),
        seeds=(42, 43),
        catalogs=_catalogs_for_materialized(),
        fit_head_fn=fit,
        score_head_fn=score,
    )

    paths = LayerwiseSuitePaths(root)
    manifest = json.loads(
        paths.probe_stability_manifest(profile).read_text(encoding="utf-8")
    )
    assert manifest["resampling_method"] == STRATIFIED_TRAIN_BOOTSTRAP_METHOD
    assert "amostra finita de treino" in manifest["scope"].lower()
    for seed in (42, 43):
        bootstrap_path = paths.probe_stability_train_bootstrap(profile, seed, "eng")
        assert bootstrap_path.is_file()
        payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        assert payload["resampling_method"] == STRATIFIED_TRAIN_BOOTSTRAP_METHOD
        assert payload["bootstrap_seed"] == seed
        assert payload["bootstrap_class_counts"] == {"0": 2, "1": 2}
        assert len(payload["train_indices"]) == 4
        assert len(payload["train_indices_sha256"]) == 64
    eng_42 = json.loads(
        paths.probe_stability_train_bootstrap(profile, 42, "eng").read_text(
            encoding="utf-8"
        )
    )
    eng_43 = json.loads(
        paths.probe_stability_train_bootstrap(profile, 43, "eng").read_text(
            encoding="utf-8"
        )
    )
    assert eng_42["train_indices"] != eng_43["train_indices"]
    assert len(fit_calls) == 2 * 3 * 2  # layers × sources × seeds
    for key, payload in canonical.items():
        assert paths.probe(profile, key[0], key[1]).read_bytes() == payload
    assert not paths.cell(profile, 1, "eng", "eng").exists()
    assert paths.probe_stability_summary(profile).is_file()
    assert result["profiles"] == [profile]
    assert result["seeds"] == [42, 43]


def test_probe_stability_bootstrap_rerun_is_bitwise_deterministic(tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    _write_execution_plan(root)
    profile = "hubert_base"
    _materialize_embeddings(root, profile)

    def fit(x, y, *, head, seed):
        return {"sum_x": float(np.sum(x))}

    kwargs = dict(
        profiles=(profile,),
        layers=(1,),
        seeds=(42,),
        catalogs=_catalogs_for_materialized(),
        fit_head_fn=fit,
        score_head_fn=lambda head, x: np.full(len(x), 0.5),
    )
    run_probe_stability(root, **kwargs)
    paths = LayerwiseSuitePaths(root)
    first = paths.probe_stability_train_bootstrap(profile, 42, "eng").read_bytes()
    first_perf = paths.probe_stability_performance(profile, 42).read_bytes()
    run_probe_stability(root, **kwargs)
    second = paths.probe_stability_train_bootstrap(profile, 42, "eng").read_bytes()
    second_perf = paths.probe_stability_performance(profile, 42).read_bytes()
    assert first == second
    assert first_perf == second_perf


def test_default_seeds_are_42_43_44():
    assert DEFAULT_PROBE_STABILITY_SEEDS == (42, 43, 44)


def test_validate_bootstrap_seeds_rejects_empty_duplicate_and_non_int():
    with pytest.raises(ValueError, match="non-empty"):
        validate_bootstrap_seeds(())
    with pytest.raises(ValueError, match="duplicate"):
        validate_bootstrap_seeds([42, 42])
    with pytest.raises(ValueError, match="integer"):
        validate_bootstrap_seeds([42, True])  # type: ignore[list-item]


def test_stratified_bootstrap_preserves_class_counts_and_is_deterministic():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    first = stratified_train_bootstrap_indices(
        labels, bootstrap_seed=42, source="eng"
    )
    second = stratified_train_bootstrap_indices(
        labels, bootstrap_seed=42, source="eng"
    )
    np.testing.assert_array_equal(first, second)
    boot_labels = labels[first]
    assert int((boot_labels == 0).sum()) == 2
    assert int((boot_labels == 1).sum()) == 2


def test_stratified_bootstrap_differs_across_seeds_42_43_44():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    indices = {
        seed: stratified_train_bootstrap_indices(
            labels, bootstrap_seed=seed, source="eng"
        )
        for seed in (42, 43, 44)
    }
    pairs = [(42, 43), (42, 44), (43, 44)]
    assert all(
        not np.array_equal(indices[a], indices[b]) for a, b in pairs
    ), "expected distinct bootstrap draws for seeds 42, 43, 44"


def test_verify_alignment_rejects_reordered_catalog_ids(tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    profile = "hubert_base"
    _materialize_embeddings(root, profile)
    catalogs = _catalogs_for_materialized()
    catalogs["eng"]["train"] = catalogs["eng"]["train"].iloc[[2, 3, 0, 1]].reset_index(
        drop=True
    )
    paths = LayerwiseSuitePaths(root)
    embeddings = load_materialized_embeddings(paths, profile)
    with pytest.raises(ValueError, match="hubert_base/eng/train"):
        verify_catalog_embedding_alignment(
            paths, profile, catalogs, embeddings, languages=LANGUAGES
        )


def test_run_probe_stability_with_alignment_check_passes_for_consistent_metadata(
    tmp_path,
):
    root = tmp_path / "suite"
    root.mkdir()
    _write_execution_plan(root)
    profile = "hubert_base"
    _materialize_embeddings(root, profile)
    run_probe_stability(
        root,
        profiles=(profile,),
        layers=(1,),
        seeds=(42,),
        catalogs=_catalogs_for_materialized(),
        verify_embedding_alignment=True,
        fit_head_fn=lambda x, y, *, head, seed: {"sum": float(np.sum(x))},
        score_head_fn=lambda head, x: np.full(len(x), 0.5),
    )


def test_load_materialized_embeddings_requires_existing_arrays(tmp_path):
    root = tmp_path / "suite"
    root.mkdir()
    profile = "wavlm_base"
    paths = LayerwiseSuitePaths(root)
    paths.embeddings(profile, "eng", "train").parent.mkdir(parents=True)
    np.save(
        paths.embeddings(profile, "eng", "train"),
        np.zeros((2, 3, 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="missing embedding"):
        load_materialized_embeddings(paths, profile, languages=("eng",))
