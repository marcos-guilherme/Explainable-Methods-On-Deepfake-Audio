"""TDD contract for fixed-cohort layer-wise AttnLRP -> DFT/STDFT."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brspeech_xai.layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
    sample_id_catalog_hash,
)
from brspeech_xai.layerwise_xai import (
    resolve_active_xai_generation,
    run_layerwise_xai,
    select_fixed_cohort,
)


def _catalog(tmp_path, *, order=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (0, 1):
        for index in range(3):
            sample_id = f"sample-{label}-{index}"
            path = tmp_path / f"{sample_id}.wav"
            path.write_bytes(b"fake")
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "processed_path": str(path),
                    "score": index / 10,
                    "prediction": index % 2,
                    "quadrant": "ignored",
                }
            )
    frame = pd.DataFrame(rows)
    return frame if order is None else frame.iloc[order].reset_index(drop=True)


def test_fixed_cohort_depends_only_on_ids_labels_and_seed(tmp_path):
    catalog = _catalog(tmp_path)
    first = select_fixed_cohort(catalog, per_class=2, seed=42)
    changed = catalog.sample(frac=1, random_state=7).assign(
        score=np.arange(len(catalog)) * 99,
        prediction=1,
        quadrant="TP",
    )
    second = select_fixed_cohort(changed, per_class=2, seed=42)

    assert tuple(first["sample_id"]) == tuple(second["sample_id"])
    assert first.groupby("label").size().to_dict() == {0: 2, 1: 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="processed_path"), "processed_path"),
        (lambda frame: frame.assign(sample_id=["x"] * len(frame)), "unique"),
        (lambda frame: frame.assign(label=2), "binary"),
        (lambda frame: frame.loc[frame["label"] == 0], "both classes"),
    ],
)
def test_fixed_cohort_rejects_invalid_catalogs(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        select_fixed_cohort(mutation(_catalog(tmp_path)), per_class=1, seed=1)

    with pytest.raises(ValueError, match="per_class"):
        select_fixed_cohort(_catalog(tmp_path), per_class=0, seed=1)
    with pytest.raises(ValueError, match="enough"):
        select_fixed_cohort(_catalog(tmp_path), per_class=4, seed=1)


def _write_cells(root, profiles, layers, sources, target, catalog):
    paths = LayerwiseSuitePaths(root)
    probability = 1.0 / (1.0 + np.exp(-0.4))
    for profile in profiles:
        for layer in layers:
            for source in sources:
                probe = paths.probe(profile, layer, source)
                probe.parent.mkdir(parents=True, exist_ok=True)
                probe.write_bytes(b"persisted-head")
                paths.thresholds(profile, layer, source).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "source_language": source,
                            "layer": layer,
                            "detectors": {"ad": {"threshold": 0.5}},
                        }
                    ),
                    encoding="utf-8",
                )
                cell = paths.cell(profile, layer, source, target)
                cell.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "sample_id": catalog["sample_id"],
                        "y_true": catalog["label"],
                        "score": probability,
                        "prediction": 1,
                    }
                ).to_parquet(cell / "predictions.parquet", index=False)
                np.save(cell / "scores.npy", np.full(len(catalog), probability))
    return paths


def _embedding_artifact(profile, target, values, sample_ids):
    ids = list(sample_ids)
    return {
        "values": values,
        "metadata": {
            "schema_version": 1,
            "profile": profile,
            "language": target,
            "role": "test",
            "shape": list(values.shape),
            "dtype": "float32",
            "sample_ids": ids,
            "sample_ids_sha256": sample_id_catalog_hash(ids),
        },
    }


class _FixtureDetector(torch.nn.Module):
    def __init__(self, encoder, layer, w, b):
        super().__init__()
        self.encoder = encoder
        self.layer = layer
        self.register_buffer("w", torch.as_tensor(w, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32))


def _run_fixture(
    tmp_path,
    *,
    mutate=None,
    cell_overrides=None,
    patch_counts=None,
    encoder_factory=None,
    use_default_patch=False,
):
    profiles = ("hubert_base", "wavlm_base_plus")
    layers = (1, 2)
    sources = ("eng", "por")
    target = "zho"
    catalog = _catalog(tmp_path)
    paths = _write_cells(tmp_path / "suite", profiles, layers, sources, target, catalog)
    if mutate is not None:
        mutate(paths, profiles, layers, sources, target, catalog)

    relevance = Mock(
        side_effect=lambda model, processor, wav, device: (
            np.ones(4),
            np.full(4, 0.1),
            0.4,
        )
    )
    stdft = Mock(
        side_effect=lambda x, r, sample_rate: (
            np.array([0.0]),
            np.array([0.0, 1.0]),
            np.array([[0.1, 0.3]]),
            np.ones((1, 2)),
        )
    )
    temporal_certificate = Mock(return_value=0.0)
    patch_encoder = Mock(
        return_value=patch_counts
        or {"layer_norm": 1, "group_norm": 1, "gelu": 1, "attention": 1}
    )
    cohort_calls = []

    def dft(x, r):
        return np.array([-0.4, 0.3, 0.5])

    def aggregate(r_freq, freqs, edges):
        values = np.asarray(r_freq)
        return np.array([values[0] + values[1], values[2]])

    dependencies = {
        "audio_loader": lambda path: (np.ones(4), 16000),
        "head_loader": lambda path: object(),
        "port_head_fn": lambda head: (
            np.ones(2, dtype=np.float32),
            np.float32(0.0),
        ),
        "verify_equivalence_fn": lambda head, emb, w, b: 0.0,
        "score_head_fn": lambda head, emb: np.full(
            len(emb), 1.0 / (1.0 + np.exp(-0.4))
        ),
        "detector_factory": _FixtureDetector,
        "relevance_fn": relevance,
        "temporal_certificate_fn": temporal_certificate,
        "dft_fn": dft,
        "aggregate_fn": aggregate,
        "stdft_fn": stdft,
    }
    dependencies.update(cell_overrides or {})
    run_kwargs = {
        "profiles": profiles,
        "catalogs": {target: catalog},
        "embeddings": {
            profile: {
                target: _embedding_artifact(
                    profile,
                    target,
                    np.ones((len(catalog), 3, 2), dtype=np.float32),
                    catalog["sample_id"].tolist(),
                )
            }
            for profile in profiles
        },
        "paths": paths,
        "frequency_edges": np.array([0.0, 1.0, 2.0]),
        "layers": layers,
        "sources": sources,
        "targets": (target,),
        "per_class": 2,
        "stdft_per_class": 1,
        "seed": 19,
        "encoder_spec_fn": lambda profile: SimpleNamespace(
            capabilities=frozenset({"attnlrp_cp", "dft_lrp"}),
            attention_rule="cp_lrp",
        ),
        "encoder_factory": encoder_factory
        or (lambda profile: (_PatchableToyEncoder(), object())),
        "cohort_observer": (
            lambda profile, layer, source, target, ids: cohort_calls.append(tuple(ids))
        ),
        **dependencies,
    }
    if not use_default_patch:
        run_kwargs["patch_encoder_fn"] = patch_encoder
    summary = run_layerwise_xai(
        **run_kwargs,
    )
    return (
        summary,
        paths,
        relevance,
        temporal_certificate,
        stdft,
        cohort_calls,
        catalog,
        patch_encoder,
    )


def test_run_reuses_cohort_counts_cells_backwards_and_stdft_budget(tmp_path):
    (
        summary,
        paths,
        relevance,
        temporal_certificate,
        stdft,
        cohort_calls,
        _,
        patch_encoder,
    ) = _run_fixture(tmp_path)

    assert len(summary) == 2 * 2 * 2
    assert relevance.call_count == 2 * 2 * 2 * (2 * 2)
    assert temporal_certificate.call_count == 2 * 2 * 2
    assert stdft.call_count == 2 * 2 * 2 * (1 * 2)
    assert patch_encoder.call_count == 2
    assert len(set(cohort_calls)) == 1

    for row in summary.itertuples(index=False):
        destination = resolve_active_xai_generation(
            paths.layer_xai(row.profile, row.layer, row.source, row.target)
        )
        assert (destination / "sample_relevance.parquet").is_file()
        assert (destination / "dft_band_relevance.csv").is_file()
        assert (destination / "dft_band_relevance_byclass.csv").is_file()
        assert (destination / "attnlrp_conservation.json").is_file()


def test_task7_persists_authoritative_cohort_identity(tmp_path):
    summary, paths, *_, catalog, _ = _run_fixture(tmp_path)
    row = summary.iloc[0]
    generation = resolve_active_generation(
        paths.layer_xai(row.profile, int(row.layer), row.source, row.target)
    )
    persisted = pd.read_parquet(generation / "sample_relevance.parquet")
    selected = catalog.set_index("sample_id").loc[persisted["sample_id"]]

    assert {"sample_id", "y_true", "processed_path"}.issubset(persisted.columns)
    assert persisted["y_true"].tolist() == selected["label"].tolist()
    assert persisted["processed_path"].tolist() == selected["processed_path"].map(str).tolist()


def test_task7_consumes_public_generation_publisher(tmp_path, monkeypatch):
    import brspeech_xai.layerwise_xai as module

    observed = Mock(side_effect=publish_generation)
    monkeypatch.setattr(module, "publish_generation", observed)

    summary, *_ = _run_fixture(tmp_path)

    assert observed.call_count == len(summary)


def test_signed_and_absolute_normalized_aggregates_are_separate(tmp_path):
    summary, paths, *_ = _run_fixture(tmp_path)
    row = summary.iloc[0]
    destination = resolve_active_xai_generation(
        paths.layer_xai(
            row["profile"], int(row["layer"]), row["source"], row["target"]
        )
    )
    samples = pd.read_parquet(destination / "sample_relevance.parquet")
    np.testing.assert_allclose(np.stack(samples["band_signed"]), [[-0.1, 0.5]] * 4)
    np.testing.assert_allclose(
        np.stack(samples["band_abs_normalized"]), [[7 / 12, 5 / 12]] * 4
    )
    aggregate = pd.read_csv(destination / "dft_band_relevance.csv")
    assert set(aggregate["measure"]) == {"signed", "absolute_normalized"}
    assert set(aggregate.loc[aggregate["measure"] == "signed", "mean"]) == {-0.1, 0.5}
    np.testing.assert_allclose(
        aggregate.loc[
            aggregate["measure"] == "absolute_normalized", "mean"
        ].to_numpy(),
        [7 / 12, 5 / 12],
    )


def test_cell_id_divergence_fails_before_any_backward(tmp_path):
    def mutate(paths, profiles, layers, sources, target, catalog):
        cell = paths.cell(profiles[-1], layers[-1], sources[-1], target)
        predictions = pd.read_parquet(cell / "predictions.parquet")
        predictions.loc[0, "sample_id"] = "wrong-id"
        predictions.to_parquet(cell / "predictions.parquet", index=False)

    with pytest.raises(ValueError, match="cohort|IDs|sample_id"):
        _run_fixture(tmp_path, mutate=mutate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda paths, profiles, layers, sources, target, catalog: paths.probe(
                profiles[0], layers[0], sources[0]
            ).unlink(),
            "probe",
        ),
        (
            lambda paths, profiles, layers, sources, target, catalog: (
                catalog.__setitem__(
                    "processed_path",
                    ["/definitely/missing.wav"] + catalog["processed_path"].tolist()[1:],
                )
            ),
            "processed_path",
        ),
    ],
)
def test_missing_artifacts_fail_closed_before_backward(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        _run_fixture(tmp_path, mutate=mutation)


def test_missing_capability_fails_before_encoder_or_backward(tmp_path):
    catalog = _catalog(tmp_path)
    paths = _write_cells(
        tmp_path / "suite", ("hubert_base",), (1,), ("eng",), "zho", catalog
    )
    encoder_factory = Mock()
    relevance = Mock()

    with pytest.raises(ValueError, match="capability"):
        run_layerwise_xai(
            profiles=("hubert_base",),
            catalogs={"zho": catalog},
            embeddings={
                "hubert_base": {
                    "zho": np.ones((len(catalog), 2, 2), dtype=np.float32)
                }
            },
            paths=paths,
            frequency_edges=np.array([0.0, 1.0]),
            layers=(1,),
            sources=("eng",),
            targets=("zho",),
            per_class=1,
            stdft_per_class=1,
            encoder_spec_fn=lambda profile: SimpleNamespace(
                capabilities=frozenset(),
                attention_rule="cp_lrp",
            ),
            encoder_factory=encoder_factory,
            relevance_fn=relevance,
        )

    encoder_factory.assert_not_called()
    relevance.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"verify_equivalence_fn": lambda *args: (_ for _ in ()).throw(
                ValueError("Torch score divergence")
            )},
            "divergence",
        ),
        (
            {
                "relevance_fn": lambda *args: (
                    np.ones(4),
                    np.array([0.1, np.nan, 0.1, 0.1]),
                    0.4,
                )
            },
            "non-finite|AttnLRP",
        ),
        (
            {"temporal_certificate_fn": lambda *args: 0.5},
            "bias-zeroed model/rule conservation",
        ),
        (
            {"dft_fn": lambda x, r: np.array([0.0, 0.0, 0.0])},
            "DFT conservation",
        ),
    ],
)
def test_numerical_failures_are_closed(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _run_fixture(tmp_path, cell_overrides=overrides)


def test_small_recomputed_score_drift_is_audited(tmp_path):
    persisted_score = float(1.0 / (1.0 + np.exp(-0.4)))
    recomputed_score = persisted_score + 1.1e-6
    recomputed_logit = float(
        np.log(recomputed_score / (1.0 - recomputed_score))
    )

    summary, paths, *_ = _run_fixture(
        tmp_path,
        cell_overrides={
            "relevance_fn": lambda model, processor, wav, device: (
                np.ones(4),
                np.full(4, 0.1),
                recomputed_logit,
            )
        },
    )

    row = summary.iloc[0]
    generation = resolve_active_xai_generation(
        paths.layer_xai(
            row["profile"], int(row["layer"]), row["source"], row["target"]
        )
    )
    samples = pd.read_parquet(generation / "sample_relevance.parquet")
    validation = json.loads(
        (generation / "attnlrp_conservation.json").read_text(encoding="utf-8")
    )
    assert samples["recomputed_score"].iloc[0] == pytest.approx(recomputed_score)
    assert samples["score_recompute_absolute_error"].iloc[0] == pytest.approx(
        1.1e-6
    )
    assert validation["max_score_recompute_absolute_error"] == pytest.approx(
        1.1e-6
    )


def test_recomputed_score_drift_remains_bounded(tmp_path):
    with pytest.raises(ValueError, match="logit/score divergence"):
        _run_fixture(
            tmp_path,
            cell_overrides={
                "relevance_fn": lambda model, processor, wav, device: (
                    np.ones(4),
                    np.full(4, 0.1),
                    0.5,
                )
            },
        )


def test_recomputed_score_cannot_change_fixed_threshold_prediction(tmp_path):
    with pytest.raises(ValueError, match="recomputed prediction/threshold divergence"):
        _run_fixture(
            tmp_path,
            cell_overrides={
                "relevance_fn": lambda model, processor, wav, device: (
                    np.ones(4),
                    np.full(4, 0.1),
                    -0.001,
                )
            },
        )


def test_stdft_conservation_fails_closed_on_tampered_relevance_map(tmp_path):
    def tampered_stdft(x_time, r_time, sample_rate):
        return (
            np.array([0.0]),
            np.array([0.0, 1.0]),
            np.array([[0.1, 0.4]]),
            np.ones((1, 2)),
        )

    with pytest.raises(ValueError, match="STDFT conservation"):
        _run_fixture(tmp_path, cell_overrides={"stdft_fn": tampered_stdft})


def test_invalid_layer_and_frequency_edges_fail_before_encoder(tmp_path):
    catalog = _catalog(tmp_path)
    paths = LayerwiseSuitePaths(tmp_path / "suite")
    encoder_factory = Mock()
    common = {
        "profiles": ("hubert_base",),
        "catalogs": {"zho": catalog},
        "embeddings": {
            "hubert_base": {
                "zho": np.ones((len(catalog), 14, 2), dtype=np.float32)
            }
        },
        "paths": paths,
        "sources": ("eng",),
        "targets": ("zho",),
        "per_class": 1,
        "stdft_per_class": 0,
        "encoder_factory": encoder_factory,
    }
    with pytest.raises(ValueError, match="layer"):
        run_layerwise_xai(
            **common, layers=(13,), frequency_edges=np.array([0.0, 1.0])
        )
    with pytest.raises(ValueError, match="frequency_edges"):
        run_layerwise_xai(
            **common, layers=(1,), frequency_edges=np.array([1.0, 0.0])
        )
    encoder_factory.assert_not_called()


def test_patch_uses_registry_rule_once_and_rejects_incomplete_coverage(tmp_path):
    *_, patch_encoder = _run_fixture(tmp_path / "ok")
    assert patch_encoder.call_count == 2
    assert {call.kwargs["attention"] for call in patch_encoder.call_args_list} == {"cp"}

    shared_encoder = _PatchableToyEncoder()
    *_, shared_patch = _run_fixture(
        tmp_path / "shared",
        encoder_factory=lambda profile: (shared_encoder, object()),
    )
    assert shared_patch.call_count == 1
    *_, repeated_patch = _run_fixture(
        tmp_path / "shared-again",
        encoder_factory=lambda profile: (shared_encoder, object()),
    )
    repeated_patch.assert_not_called()

    blocked_relevance = Mock()
    with pytest.raises(ValueError, match="patch.*coverage|incomplete"):
        _run_fixture(
            tmp_path / "bad",
            patch_counts={
                "layer_norm": 1,
                "group_norm": 1,
                "gelu": 1,
                "attention": 0,
            },
            cell_overrides={"relevance_fn": blocked_relevance},
        )
    blocked_relevance.assert_not_called()


def test_certificate_is_cell_level_and_published_gap_is_bias_inclusive(tmp_path):
    summary, paths, relevance, certificate, *_ = _run_fixture(tmp_path)
    assert certificate.call_count == len(summary)
    assert relevance.call_count == int(summary["n"].sum())
    row = summary.iloc[0]
    generation = resolve_active_xai_generation(
        paths.layer_xai(
            row["profile"], int(row["layer"]), row["source"], row["target"]
        )
    )
    samples = pd.read_parquet(generation / "sample_relevance.parquet")
    assert "bias_inclusive_attribution_gap" in samples
    assert "temporal_residual" not in samples
    validation = json.loads(
        (generation / "attnlrp_conservation.json").read_text(encoding="utf-8")
    )
    assert validation["validation_kind"] == "bias_zeroed_model_rule_check"
    assert validation["bias_zeroed_validation_residual"] == 0.0
    assert "max_bias_inclusive_attribution_gap" in validation


def test_preflight_aligns_embeddings_by_metadata_id_and_checks_cell_scores(tmp_path):
    relevance = Mock()
    with pytest.raises(ValueError, match="cell score.*head|preflight"):
        _run_fixture(
            tmp_path,
            cell_overrides={
                "score_head_fn": lambda head, emb: np.zeros(len(emb)),
                "relevance_fn": relevance,
            },
        )
    relevance.assert_not_called()


def test_embedding_cache_metadata_hash_mismatch_fails_before_backward(tmp_path):
    relevance = Mock()
    catalog = _catalog(tmp_path)
    paths = _write_cells(
        tmp_path / "suite", ("hubert_base",), (1,), ("eng",), "zho", catalog
    )
    values = np.ones((len(catalog), 2, 2), dtype=np.float32)
    artifact = _embedding_artifact(
        "hubert_base", "zho", values, catalog["sample_id"].tolist()
    )
    artifact["metadata"]["sample_ids_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="cache metadata"):
        run_layerwise_xai(
            profiles=("hubert_base",),
            catalogs={"zho": catalog},
            embeddings={"hubert_base": {"zho": artifact}},
            paths=paths,
            frequency_edges=np.array([0.0, 1.0]),
            layers=(1,),
            sources=("eng",),
            targets=("zho",),
            per_class=1,
            stdft_per_class=0,
            encoder_factory=Mock(),
            encoder_spec_fn=lambda profile: SimpleNamespace(
                capabilities=frozenset({"attnlrp_cp", "dft_lrp"}),
                attention_rule="cp_lrp",
            ),
            relevance_fn=relevance,
        )
    relevance.assert_not_called()


def test_stdft_filename_is_hashed_and_keeps_internal_sample_id(tmp_path):
    summary, paths, *_ = _run_fixture(tmp_path)
    row = summary.iloc[0]
    generation = resolve_active_xai_generation(
        paths.layer_xai(
            row["profile"], int(row["layer"]), row["source"], row["target"]
        )
    )
    examples = list((generation / "stdft_examples").glob("*.npz"))
    assert len(examples) == 2
    for example in examples:
        with np.load(example, allow_pickle=False) as payload:
            sample_id = str(payload["sample_id"].item())
            assert payload["relevance_time_sum"].item() == pytest.approx(0.4)
            assert payload["relevance_tf_sum"].item() == pytest.approx(0.4)
            assert payload["conservation_absolute_error"].item() == pytest.approx(0.0)
            assert payload["conservation_relative_error"].item() == pytest.approx(0.0)
            assert payload["conservation_tolerance"].item() == pytest.approx(1e-3)
        assert example.stem == hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        assert sample_id not in example.name


def test_generation_pointer_keeps_previous_output_visible_on_publish_failure(
    tmp_path, monkeypatch
):
    summary, paths, *_ = _run_fixture(tmp_path)
    row = summary.iloc[0]
    destination = paths.layer_xai(
        row["profile"], int(row["layer"]), row["source"], row["target"]
    )
    previous = resolve_active_xai_generation(destination)
    import brspeech_xai.layerwise_paths as module

    original_replace = module.os.replace

    def fail_pointer(source, target):
        if str(target).endswith("active.json"):
            raise OSError("synthetic pointer failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="pointer"):
        _run_fixture(tmp_path)
    assert resolve_active_xai_generation(destination) == previous
    assert (previous / "sample_relevance.parquet").is_file()


def test_published_generation_survives_pointer_reread_failure(tmp_path, monkeypatch):
    import brspeech_xai.layerwise_xai as module

    destination = tmp_path / "xai-cell"
    original_read_text = Path.read_text

    def fail_active_pointer_read(path, *args, **kwargs):
        if path == destination / "active.json":
            raise OSError("synthetic reread failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_active_pointer_read)
    module.publish_generation(
        destination,
        lambda generation: (generation / "artifact.txt").write_text(
            "complete", encoding="utf-8"
        ),
    )

    generations = [
        path
        for path in (destination / "generations").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(generations) == 1
    assert (generations[0] / "artifact.txt").is_file()


def test_concurrent_pointer_advance_does_not_delete_published_generation(
    tmp_path, monkeypatch
):
    import brspeech_xai.layerwise_paths as module

    destination = tmp_path / "xai-cell"
    original_replace = module.os.replace
    published_tokens = []

    def advance_after_publish(source, target):
        result = original_replace(source, target)
        if Path(target) == destination / "active.json":
            published = json.loads(
                (destination / "active.json").read_text(encoding="utf-8")
            )["generation"]
            published_tokens.append(published)
            concurrent = "f" * 32
            concurrent_dir = destination / "generations" / concurrent
            concurrent_dir.mkdir()
            artifact = concurrent_dir / "artifact.txt"
            artifact.write_text("other", encoding="utf-8")
            (concurrent_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "generation": concurrent,
                        "role": "generic",
                        "artifacts": [
                            {
                                "path": "artifact.txt",
                                "sha256": module._file_sha256(artifact),
                                "size_bytes": artifact.stat().st_size,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (destination / "active.json").write_text(
                json.dumps({"schema_version": 1, "generation": concurrent}),
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(module.os, "replace", advance_after_publish)
    module.publish_generation(
        destination,
        lambda generation: (generation / "artifact.txt").write_text(
            "ours", encoding="utf-8"
        ),
    )

    assert len(published_tokens) == 1
    assert (destination / "generations" / published_tokens[0]).is_dir()
    assert resolve_active_xai_generation(destination).name == "f" * 32


def test_resolver_rejects_generation_with_missing_manifest_artifact(tmp_path):
    summary, paths, *_ = _run_fixture(tmp_path)
    row = summary.iloc[0]
    destination = paths.layer_xai(
        row["profile"], int(row["layer"]), row["source"], row["target"]
    )
    generation = resolve_active_xai_generation(destination)
    (generation / "sample_relevance.parquet").unlink()

    with pytest.raises(ValueError, match="artifact|valid active"):
        resolve_active_xai_generation(destination)


class _DeviceSpyDetector(_FixtureDetector):
    def __init__(self, encoder, layer, w, b):
        super().__init__(encoder, layer, w, b)
        self.to_devices = []
        self.eval_calls = 0

    def to(self, device):
        self.to_devices.append(str(device))
        return super().to(device)

    def eval(self):
        self.eval_calls += 1
        return super().eval()


def test_detector_is_moved_to_device_and_eval_before_any_backward(tmp_path):
    created = []

    def factory(encoder, layer, w, b):
        detector = _DeviceSpyDetector(encoder, layer, w, b)
        created.append(detector)
        return detector

    def certificate(model, processor, wavs, device, b, tol):
        assert model.to_devices == ["cpu"]
        assert model.eval_calls == 1
        assert model.training is False
        assert model.w.device.type == "cpu"
        assert model.b.device.type == "cpu"
        return 0.0

    _run_fixture(
        tmp_path,
        cell_overrides={
            "detector_factory": factory,
            "temporal_certificate_fn": certificate,
        },
    )
    assert created


class GELUActivation(torch.nn.Module):
    def forward(self, values):
        return torch.nn.functional.gelu(values)


class HubertAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None, None


class _PatchableToyEncoder(torch.nn.Module):
    def __init__(self, *, attention=True):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(2)
        self.group_norm = torch.nn.GroupNorm(1, 2)
        self.activation = GELUActivation()
        if attention:
            self.attention = HubertAttention()


def test_orchestration_uses_real_default_attnlrp_patch_and_fails_closed(tmp_path):
    encoders = []

    def factory(profile):
        encoder = _PatchableToyEncoder()
        encoders.append(encoder)
        return encoder, object()

    _run_fixture(
        tmp_path / "complete",
        encoder_factory=factory,
        use_default_patch=True,
    )
    assert len(encoders) == 2
    for encoder in encoders:
        marker = encoder._brspeech_attnlrp_patch
        assert marker["schema_version"] == 1
        assert marker["managed_by"] == "ensure_ssl_encoder_attnlrp"
        assert marker["attention_rule"] == "cp_lrp"
        assert marker["attention"] == "cp"
        assert marker["capability"] == "attnlrp_cp"
        assert marker["counts"] == {
            "layer_norm": 1,
            "group_norm": 1,
            "gelu": 1,
            "attention": 1,
        }

    relevance = Mock()
    with pytest.raises(ValueError, match="patch coverage.*incomplete"):
        _run_fixture(
            tmp_path / "incomplete",
            encoder_factory=lambda profile: (
                _PatchableToyEncoder(attention=False),
                object(),
            ),
            use_default_patch=True,
            cell_overrides={"relevance_fn": relevance},
        )
    relevance.assert_not_called()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda paths, profiles, layers, sources, target, catalog: (
                paths.thresholds(profiles[0], layers[0], sources[0]).write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "source_language": "zho",
                            "layer": layers[0],
                            "detectors": {"ad": {"threshold": 0.5}},
                        }
                    ),
                    encoding="utf-8",
                )
            ),
            "threshold.*source",
        ),
        (
            lambda paths, profiles, layers, sources, target, catalog: (
                pd.DataFrame(
                    {
                        **pd.read_parquet(
                            paths.cell(
                                profiles[0], layers[0], sources[0], target
                            )
                            / "predictions.parquet"
                        ).to_dict("list"),
                        "y_true": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    }
                ).to_parquet(
                    paths.cell(profiles[0], layers[0], sources[0], target)
                    / "predictions.parquet",
                    index=False,
                )
            ),
            "y_true.*integer",
        ),
    ],
)
def test_threshold_identity_and_y_true_schema_are_strict(
    tmp_path, mutation, message
):
    with pytest.raises(ValueError, match=message):
        _run_fixture(tmp_path, mutate=mutation)


class _ToyProcessor:
    def __call__(self, wavs, sampling_rate, return_tensors):
        assert sampling_rate == 16000 and return_tensors == "pt"
        return {"input_values": torch.tensor(np.stack(wavs), dtype=torch.float32)}


class _ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(2)
        self.group_norm = torch.nn.GroupNorm(1, 2)
        self.activation = GELUActivation()
        self.attention = HubertAttention()

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        hidden = torch.stack((input_values, 2.0 * input_values), dim=-1)
        return SimpleNamespace(hidden_states=(hidden, hidden))


def test_real_head_detector_and_dft_integrate_with_id_aligned_preflight(tmp_path):
    catalog = _catalog(tmp_path).iloc[:2].copy()
    catalog = pd.concat(
        [_catalog(tmp_path).iloc[:2], _catalog(tmp_path).iloc[3:5]],
        ignore_index=True,
    )
    wavs = {
        sample_id: np.full(8, -1.0 if label == 0 else 1.0, dtype=np.float32)
        for sample_id, label in zip(catalog["sample_id"], catalog["label"])
    }
    values = np.asarray(
        [
            np.stack(([wav.mean(), 2 * wav.mean()],) * 2)
            for wav in wavs.values()
        ],
        dtype=np.float32,
    )
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=1, max_iter=200),
    ).fit(values[:, 1, :], catalog["label"].to_numpy())
    from brspeech_xai.adaptation import score_head

    scores = score_head(head, values[:, 1, :])
    paths = LayerwiseSuitePaths(tmp_path / "suite")
    probe = paths.probe("hubert_base", 1, "eng")
    probe.parent.mkdir(parents=True)
    joblib.dump(head, probe)
    paths.thresholds("hubert_base", 1, "eng").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_language": "eng",
                "layer": 1,
                "detectors": {"ad": {"threshold": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    cell = paths.cell("hubert_base", 1, "eng", "zho")
    cell.mkdir(parents=True)
    pd.DataFrame(
        {
            "sample_id": catalog["sample_id"],
            "y_true": catalog["label"],
            "score": scores,
            "prediction": (scores >= 0.5).astype(np.int64),
        }
    ).to_parquet(cell / "predictions.parquet", index=False)
    np.save(cell / "scores.npy", scores)
    order = np.array([2, 0, 3, 1])
    patch = Mock(
        return_value={"layer_norm": 1, "group_norm": 1, "gelu": 1, "attention": 1}
    )

    result = run_layerwise_xai(
        profiles=("hubert_base",),
        catalogs={"zho": catalog},
        embeddings={
            "hubert_base": {
                "zho": _embedding_artifact(
                    "hubert_base",
                    "zho",
                    values[order],
                    catalog["sample_id"].to_numpy()[order].tolist(),
                )
            }
        },
        paths=paths,
        frequency_edges=np.array([0.0, 2000.0, 8000.0]),
        encoder_factory=lambda profile: (_ToyEncoder(), _ToyProcessor()),
        layers=(1,),
        sources=("eng",),
        targets=("zho",),
        per_class=1,
        stdft_per_class=0,
        encoder_spec_fn=lambda profile: SimpleNamespace(
            capabilities=frozenset({"attnlrp_cp", "dft_lrp"}),
            attention_rule="cp_lrp",
        ),
        patch_encoder_fn=patch,
        audio_loader=lambda path: (
            wavs[Path(path).stem],
            16000,
        ),
        conservation_tolerance=1e-5,
    )

    assert len(result) == 1
    assert patch.call_count == 1
    generation = resolve_active_xai_generation(
        paths.layer_xai("hubert_base", 1, "eng", "zho")
    )
    samples = pd.read_parquet(generation / "sample_relevance.parquet")
    assert np.isfinite(samples["dft_residual"]).all()
