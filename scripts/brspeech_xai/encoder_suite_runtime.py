"""Production adapters wiring encoder-suite stages to Tasks 2-8."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .encoder_suite import LANGUAGE_ORDER, LAYER_ORDER, ROLE_ORDER, StageRequest
from .layerwise_paths import LayerwiseSuitePaths


def build_preprocessed_audio_loader(
    embedder: object,
    *,
    read_audio: Callable[[Path], tuple[np.ndarray, int]],
) -> Callable[[Path], tuple[np.ndarray, int]]:
    """Build the shared waveform boundary for XAI and final tracing."""

    def load(path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = read_audio(path)
        processed = embedder.preprocess_waveform(audio, sample_rate)
        return np.asarray(processed, dtype=np.float32), 16000

    return load


class ProductionStageAdapter:
    """Execute granular orchestration stages through the existing task APIs."""

    def __init__(self, *, classical_audit_adapter=None) -> None:
        self._role_data: dict[tuple[str, str, str], tuple] = {}
        self._classical_audit_adapter = classical_audit_adapter

    def _paths(self, request: StageRequest) -> LayerwiseSuitePaths:
        return LayerwiseSuitePaths(request.output_root)

    def _load_role(self, request: StageRequest, language: str, role: str) -> tuple:
        key = (request.inputs[language].config_sha256, language, role)
        if key in self._role_data:
            return self._role_data[key]
        from .config import load_config
        from .data import build_balanced_split

        cfg = load_config(request.inputs[language].config_path)
        cfg.data.manifest_path = str(request.inputs[language].manifest_path)
        split = {
            "train": cfg.data.train_split,
            "calibration": cfg.data.calibration_split,
            "test": cfg.data.eval_split,
        }[role]
        quota = {
            "train": cfg.data.n_train_per_class,
            "calibration": cfg.data.n_calibration_per_class,
            "test": cfg.data.n_test_per_class,
        }[role]
        if not split:
            raise ValueError(f"{language}/{role} requires an explicit source split")
        audios, srs, labels, provenance = build_balanced_split(
            cfg.data.dataset_id,
            split,
            quota,
            loader=cfg.data.loader,
            seed=request.seed,
            dataset_kind=cfg.data.dataset_kind,
            manifest_path=cfg.data.manifest_path,
        )
        catalog = pd.DataFrame(provenance)
        catalog["label"] = np.asarray(labels, dtype=np.int64)
        required = {"sample_id", "label", "processed_path"}
        if not required.issubset(catalog.columns):
            raise ValueError(
                f"{language}/{role} catalog lacks {sorted(required - set(catalog.columns))}"
            )
        result = (audios, srs, np.asarray(labels, dtype=np.int64), catalog, cfg)
        self._role_data[key] = result
        return result

    def run_stage(
        self, request: StageRequest, encoder: object | None
    ) -> Mapping[str, Path]:
        handler = getattr(self, f"_run_{request.kind}", None)
        if not callable(handler):
            raise ValueError(f"no production adapter for stage kind {request.kind!r}")
        return handler(request, encoder)

    def _run_cohort(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        from .layerwise_xai import select_fixed_cohort

        _audios, _srs, _labels, catalog, _cfg = self._load_role(
            request, str(request.target), "test"
        )
        cohort = select_fixed_cohort(
            catalog,
            per_class=request.suite_config.xai_per_class,
            seed=request.seed,
        )
        paths = self._paths(request)
        cohort_path = paths.suite_cohort(str(request.target))
        metadata_path = paths.suite_cohort_metadata(str(request.target))
        cohort_path.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_parquet(cohort_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "target": request.target,
                    "sample_ids": cohort["sample_id"].tolist(),
                    "per_class": request.suite_config.xai_per_class,
                    "seed": request.seed,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"cohort": cohort_path, "metadata": metadata_path}

    def _run_embedding(
        self, request: StageRequest, encoder: object | None
    ) -> Mapping[str, Path]:
        from .layerwise_paths import (
            build_embedding_cache_metadata,
            embedding_cache_key,
            write_embedding_cache,
        )

        if encoder is None or not callable(
            getattr(encoder, "extract_all_layer_embeddings", None)
        ):
            raise ValueError("embedding stage requires a layer-wise encoder")
        language, role = str(request.language), str(request.role)
        audios, srs, _labels, catalog, cfg = self._load_role(
            request, language, role
        )
        array = encoder.extract_all_layer_embeddings(audios, srs)
        paths = self._paths(request)
        array_path = paths.embeddings(str(request.profile), language, role)
        metadata_path = paths.embedding_metadata(str(request.profile), language, role)
        cache_key = embedding_cache_key(
            profile_id=str(request.profile),
            checkpoint=str(request.checkpoint),
            manifest_sha256=request.inputs[language].manifest_sha256,
            language=language,
            role=role,
            seed=request.seed,
            audio_config=cfg.audio,
        )
        metadata = build_embedding_cache_metadata(
            cache_key=cache_key,
            profile_id=str(request.profile),
            checkpoint=str(request.checkpoint),
            manifest_sha256=request.inputs[language].manifest_sha256,
            language=language,
            role=role,
            array=array,
            sample_ids=catalog["sample_id"].tolist(),
        )
        metadata["config_sha256"] = request.inputs[language].config_sha256
        # Task 3's writer requires its exact schema; publish config identity beside it.
        task3_metadata = dict(metadata)
        task3_metadata.pop("config_sha256")
        write_embedding_cache(array_path, metadata_path, array, task3_metadata)
        return {"embeddings": array_path, "metadata": metadata_path}

    def _embedding_and_catalog(
        self, request: StageRequest, language: str, role: str
    ) -> tuple[np.ndarray, pd.DataFrame]:
        paths = self._paths(request)
        array = np.load(
            paths.embeddings(str(request.profile), language, role),
            allow_pickle=False,
        )
        return array, self._load_role(request, language, role)[3]

    def _run_probe(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        from .adaptation import fit_head, score_head
        from .layerwise_probes import _persist_probe
        from .metrics import calibrate_threshold

        source, layer = str(request.source), int(request.layer)
        train, train_catalog = self._embedding_and_catalog(request, source, "train")
        calibration, calibration_catalog = self._embedding_and_catalog(
            request, source, "calibration"
        )
        head = fit_head(
            train[:, layer, :],
            train_catalog["label"].to_numpy(),
            head="logistic",
            seed=request.seed,
        )
        scores = score_head(head, calibration[:, layer, :])
        threshold = calibrate_threshold(
            scores,
            calibration_catalog["label"].to_numpy(),
            f"{source}/calibration/layer_{layer:02d}",
            "ad",
        )
        paths = self._paths(request)
        _persist_probe(paths, str(request.profile), layer, source, head, threshold)
        return {
            "head": paths.probe(str(request.profile), layer, source),
            "thresholds": paths.thresholds(str(request.profile), layer, source),
        }

    def _run_cell(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        import joblib

        from .adaptation import score_head
        from .layerwise_probes import _default_metrics, _publish_directory

        paths = self._paths(request)
        profile, source, target, layer = (
            str(request.profile),
            str(request.source),
            str(request.target),
            int(request.layer),
        )
        head = joblib.load(paths.probe(profile, layer, source))
        threshold_payload = json.loads(
            paths.thresholds(profile, layer, source).read_text(encoding="utf-8")
        )
        threshold = float(threshold_payload["detectors"]["ad"]["threshold"])
        values, catalog = self._embedding_and_catalog(request, target, "test")
        scores = np.asarray(score_head(head, values[:, layer, :]), dtype=np.float64)
        labels = catalog["label"].to_numpy(dtype=np.int64)
        predictions = (scores >= threshold).astype(np.int64)
        condition = "diagonal" if source == target else "cross_corpus_shift"
        metrics = _default_metrics(
            scores,
            labels,
            threshold,
            condition=condition,
            source=source,
            target=target,
        )
        destination = paths.cell(profile, layer, source, target)

        def write(directory: Path) -> None:
            np.save(directory / "scores.npy", scores, allow_pickle=False)
            pd.DataFrame(
                {
                    "sample_id": catalog["sample_id"].tolist(),
                    "y_true": labels,
                    "processed_path": catalog["processed_path"].map(str).tolist(),
                    "score": scores,
                    "prediction": predictions,
                }
            ).to_parquet(directory / "predictions.parquet", index=False)
            (directory / "metrics.json").write_text(
                json.dumps(metrics, sort_keys=True), encoding="utf-8"
            )

        _publish_directory(destination, write)
        return {
            "scores": destination / "scores.npy",
            "predictions": destination / "predictions.parquet",
            "metrics": destination / "metrics.json",
        }

    def _run_emergence(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        from .emergence import summarize_emergence

        paths = self._paths(request)
        rows = []
        profile, source, target = (
            str(request.profile),
            str(request.source),
            str(request.target),
        )
        for layer in LAYER_ORDER:
            directory = paths.cell(profile, layer, source, target)
            predictions = pd.read_parquet(directory / "predictions.parquet")
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "profile": profile,
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "auc": metrics["threshold_free"]["roc_auc"],
                    "corpus_shift": source != target,
                    "scores": predictions["score"].to_numpy(),
                    "labels": predictions["y_true"].to_numpy(),
                }
            )
        summary = summarize_emergence(
            pd.DataFrame(rows),
            layer_indices=LAYER_ORDER,
            n_bootstrap=request.suite_config.bootstrap_samples,
            seed=request.seed,
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        path = request.output_dir / "emergence.csv"
        summary.to_csv(path, index=False)
        return {"summary": path}

    def _suite_cohort(self, request: StageRequest, target: str) -> pd.DataFrame:
        return pd.read_parquet(self._paths(request).suite_cohort(target))

    def _run_xai(
        self, request: StageRequest, encoder: object | None
    ) -> Mapping[str, Path]:
        import joblib
        import soundfile as sf

        from .bands import mel_band_edges
        from .attnlrp import ensure_ssl_encoder_attnlrp, patch_ssl_encoder_for_attnlrp
        from .layerwise_xai import (
            _prepare_cell,
            _read_prediction_cell,
            run_layer_xai_cell,
        )

        if encoder is None or not hasattr(encoder, "_model"):
            raise ValueError("XAI stage requires the production HF SSL encoder")
        profile, source, target, layer = (
            str(request.profile),
            str(request.source),
            str(request.target),
            int(request.layer),
        )
        paths = self._paths(request)
        from .xai_registry import get_encoder_spec

        spec = get_encoder_spec(profile)
        ensure_ssl_encoder_attnlrp(
            encoder._model,
            attention_rule=spec.attention_rule,
            capabilities=spec.capabilities,
            patch_fn=patch_ssl_encoder_for_attnlrp,
        )
        cohort = self._suite_cohort(request, target)
        _values, target_catalog = self._embedding_and_catalog(request, target, "test")
        values = np.load(paths.embeddings(profile, target, "test"), allow_pickle=False)
        predictions = _read_prediction_cell(
            paths,
            profile,
            layer,
            source,
            target,
            target_catalog,
            tuple(cohort["sample_id"]),
        )
        prepared = _prepare_cell(
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            cohort=cohort,
            target_catalog=target_catalog,
            target_embeddings=values,
            predictions=predictions,
            paths=paths,
            head_loader=joblib.load,
            threshold_loader=__import__(
                "brspeech_xai.layerwise_xai", fromlist=["_load_threshold"]
            )._load_threshold,
            port_head_fn=__import__(
                "brspeech_xai.lrp_detector", fromlist=["port_logistic_head"]
            ).port_logistic_head,
            verify_equivalence_fn=__import__(
                "brspeech_xai.lrp_detector", fromlist=["verify_score_equivalence"]
            ).verify_score_equivalence,
            score_head_fn=__import__(
                "brspeech_xai.adaptation", fromlist=["score_head"]
            ).score_head,
        )
        cfg = self._load_role(request, target, "test")[4]
        edges = mel_band_edges(cfg.bands.n_bands, cfg.bands.f_min, cfg.bands.f_max)
        stdft_ids = []
        for label in (0, 1):
            stdft_ids.extend(
                cohort.loc[cohort["label"] == label, "sample_id"]
                .iloc[: request.suite_config.stdft_examples_per_class]
                .tolist()
            )
        load_audio = build_preprocessed_audio_loader(
            encoder,
            read_audio=lambda path: sf.read(
                path, dtype="float32", always_2d=False
            ),
        )
        run_layer_xai_cell(
            profile=profile,
            layer=layer,
            source=source,
            target=target,
            cohort=cohort,
            target_catalog=target_catalog,
            target_embeddings=values,
            predictions=predictions,
            paths=paths,
            encoder=encoder._model,
            processor=encoder._processor,
            prepared=prepared,
            frequency_edges=edges,
            stdft_sample_ids=frozenset(stdft_ids),
            sample_rate=16000,
            conservation_tolerance=request.suite_config.conservation_tolerance,
            device=encoder.device,
            audio_loader=load_audio,
        )
        return {"active_pointer": paths.layer_xai(profile, layer, source, target) / "active.json"}

    def _run_trace(
        self, request: StageRequest, encoder: object | None
    ) -> Mapping[str, Path]:
        import soundfile as sf

        from .final_trace import run_final_decision_trace_cell

        if encoder is None or not hasattr(encoder, "_model"):
            raise ValueError("trace stage requires the production HF SSL encoder")
        profile, source, target = (
            str(request.profile),
            str(request.source),
            str(request.target),
        )
        paths = self._paths(request)
        load_audio = build_preprocessed_audio_loader(
            encoder,
            read_audio=lambda path: sf.read(
                path, dtype="float32", always_2d=False
            ),
        )
        run_final_decision_trace_cell(
            profile=profile,
            source=source,
            target=target,
            cohort=self._suite_cohort(request, target).rename(
                columns={"label": "y_true"}
            ),
            paths=paths,
            encoder=encoder._model,
            processor=encoder._processor,
            audio_loader=load_audio,
            expected_layers=12,
            sample_rate=16000,
            device=encoder.device,
        )
        active = paths.final_trace_cell(profile, source, target) / "active.json"
        return {"active_pointer": active}

    def _run_profile_aggregate(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        return self._write_index(request)

    def _run_suite_aggregate(
        self, request: StageRequest, _encoder: object | None
    ) -> Mapping[str, Path]:
        from .encoder_suite import _suite_config_hash
        from .suite_aggregation import aggregate_suite

        adapter = self._classical_audit_adapter
        if adapter is None and request.suite_config.classical_audit:
            adapter = lambda **kwargs: self._run_classical_audit_cell(
                request=request, **kwargs
            )
        active = aggregate_suite(
            root=request.output_root,
            profiles=request.suite_config.profiles,
            config_hash=_suite_config_hash(
                request.suite_config,
                seed=request.seed,
                device=request.device,
            ),
            upstream_fingerprints=request.upstream_fingerprints,
            inputs=request.inputs,
            classical_audit=request.suite_config.classical_audit,
            classical_audit_adapter=adapter,
        )
        return {"index": active}

    def _run_classical_audit_cell(
        self,
        *,
        request: StageRequest,
        source: str,
        target: str,
        layer: int,
        output_dir: Path,
    ) -> Mapping[str, Path]:
        """Run the existing H1/H2/H3 stages for one layer-12 diagonal."""
        if source != target or layer != 12:
            raise ValueError("classical audit is restricted to diagonal layer 12")
        import joblib

        from . import artifacts as A
        from .encoder_suite import _production_encoder_factory
        from .stages import (
            RunContext,
            stage_association,
            stage_confirmatory,
            stage_features,
            stage_occlusion,
        )
        from .xai_registry import get_encoder_spec

        _audios, _srs, _labels, catalog, cfg = self._load_role(
            request, target, "test"
        )
        records = []
        for profile in request.suite_config.profiles:
            profile_dir = output_dir / profile
            paths = A.RunPaths(profile_dir)
            samples = catalog.rename(columns={"label": "label"}).copy()
            samples["split"] = "test"
            A.save_table(samples, paths.path("samples.parquet"))
            cell = self._paths(request).cell(profile, layer, source, target)
            predictions = pd.read_parquet(cell / "predictions.parquet")
            by_id = predictions.set_index("sample_id")
            expected_ids = catalog["sample_id"].tolist()
            if set(by_id.index) != set(expected_ids):
                raise ValueError("classical audit predictions do not match target IDs")
            ordered = by_id.loc[expected_ids]
            A.save_npy(
                ordered["score"].to_numpy(dtype=np.float32),
                paths.path("p_spoof_ad.npy"),
            )
            threshold = self._paths(request).thresholds(profile, layer, source)
            head = self._paths(request).probe(profile, layer, source)
            A.save_json(
                json.loads(threshold.read_text(encoding="utf-8")),
                paths.path("thresholds.json"),
            )
            joblib.dump(joblib.load(head), paths.path("d_ad.joblib"))
            encoder = _production_encoder_factory(
                profile, get_encoder_spec(profile), request.device
            )
            try:
                ctx = RunContext(cfg=cfg, paths=paths, logger=None, embedder=encoder)
                ctx.plots = False
                stage_features(ctx)
                stage_association(ctx)
                stage_occlusion(ctx)
                stage_confirmatory(ctx)
            finally:
                self.cleanup_encoder(encoder)
            records.append(
                {
                    "profile": profile,
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "h1": "spearman_table.csv",
                    "h2": "occlusion_table.csv",
                    "h3": "confirmatory_tests.csv",
                }
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "classical_audit_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": source,
                    "target": target,
                    "layer": layer,
                    "records": records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"manifest": manifest}

    def _write_index(self, request: StageRequest) -> Mapping[str, Path]:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        path = request.output_dir / "orchestration-index.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": request.kind,
                    "stage_id": request.stage_id,
                    "upstream_fingerprints": list(request.upstream_fingerprints),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {"index": path}

    def validate_stage(
        self, request: StageRequest, artifacts: Mapping[str, Path]
    ) -> None:
        if request.kind == "cohort":
            frame = pd.read_parquet(artifacts["cohort"])
            metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
            ids = frame["sample_id"].tolist() if "sample_id" in frame else []
            labels = frame["label"].to_numpy() if "label" in frame else np.asarray([])
            if (
                not {"sample_id", "label", "processed_path"}.issubset(frame.columns)
                or not ids
                or len(ids) != len(set(ids))
                or set(labels.tolist()) != {0, 1}
                or any(
                    int((labels == label).sum())
                    != request.suite_config.xai_per_class
                    for label in (0, 1)
                )
                or metadata.get("schema_version") != 1
                or metadata.get("target") != request.target
                or metadata.get("sample_ids") != ids
            ):
                raise ValueError("invalid Task 7 suite cohort artifact")
        elif request.kind == "embedding":
            from .layerwise_paths import (
                embedding_cache_key,
                load_valid_embedding_cache,
            )

            language, role = str(request.language), str(request.role)
            _a, _s, _l, catalog, cfg = self._load_role(request, language, role)
            expected_key = embedding_cache_key(
                profile_id=str(request.profile),
                checkpoint=str(request.checkpoint),
                manifest_sha256=request.inputs[language].manifest_sha256,
                language=language,
                role=role,
                seed=request.seed,
                audio_config=cfg.audio,
            )
            valid = load_valid_embedding_cache(
                artifacts["embeddings"],
                artifacts["metadata"],
                expected_cache_key=expected_key,
                expected_profile_id=str(request.profile),
                expected_checkpoint=str(request.checkpoint),
                expected_manifest_sha256=request.inputs[language].manifest_sha256,
                expected_language=language,
                expected_role=role,
                expected_sample_ids=catalog["sample_id"].tolist(),
            )
            if valid is None:
                raise ValueError("invalid Task 3 embedding cache artifact")
        elif request.kind == "probe":
            import joblib

            joblib.load(artifacts["head"])
            payload = json.loads(artifacts["thresholds"].read_text(encoding="utf-8"))
            threshold = payload.get("detectors", {}).get("ad", {}).get("threshold")
            if (
                payload.get("schema_version") != 1
                or payload.get("source_language") != request.source
                or payload.get("layer") != request.layer
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not np.isfinite(float(threshold))
            ):
                raise ValueError("invalid Task 4 probe artifact")
        elif request.kind == "cell":
            scores = np.load(artifacts["scores"], allow_pickle=False)
            predictions = pd.read_parquet(artifacts["predictions"])
            metrics = json.loads(artifacts["metrics"].read_text(encoding="utf-8"))
            if (
                scores.ndim != 1
                or not np.isfinite(scores).all()
                or not {
                    "sample_id",
                    "y_true",
                    "processed_path",
                    "score",
                    "prediction",
                }.issubset(predictions.columns)
                or len(scores) != len(predictions)
                or not np.allclose(scores, predictions["score"].to_numpy())
                or not isinstance(metrics, dict)
            ):
                raise ValueError("invalid Task 4 cell artifact")
        elif request.kind == "emergence":
            summary = pd.read_csv(artifacts["summary"])
            if not {
                "profile",
                "source",
                "target",
                "onset",
                "consolidation",
            }.issubset(summary.columns):
                raise ValueError("invalid Task 5 emergence artifact")
        elif request.kind in {"xai", "trace"}:
            from .layerwise_paths import resolve_active_generation

            generation = resolve_active_generation(
                artifacts["active_pointer"].parent,
                expected_role="layer_xai" if request.kind == "xai" else "final_trace",
            )
            required = (
                {
                    "sample_relevance.parquet",
                    "dft_band_relevance.csv",
                    "dft_band_relevance_byclass.csv",
                    "attnlrp_conservation.json",
                }
                if request.kind == "xai"
                else {"layer_relevance.parquet", "layer_transition_metrics.csv"}
            )
            if not all((generation / name).is_file() for name in required):
                raise ValueError(f"invalid Task {'7' if request.kind == 'xai' else '8'} generation")
            if request.kind == "xai":
                samples = pd.read_parquet(generation / "sample_relevance.parquet")
                bands = pd.read_csv(generation / "dft_band_relevance.csv")
                conservation = json.loads(
                    (generation / "attnlrp_conservation.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    not {
                        "profile",
                        "layer",
                        "source",
                        "target",
                        "sample_id",
                        "y_true",
                    }.issubset(samples.columns)
                    or not {"profile", "layer", "source", "target"}.issubset(
                        bands.columns
                    )
                    or any(
                        conservation.get(name) != value
                        for name, value in (
                            ("profile", request.profile),
                            ("layer", request.layer),
                            ("source", request.source),
                            ("target", request.target),
                        )
                    )
                ):
                    raise ValueError("invalid Task 7 generation schema or identity")
            else:
                layers = pd.read_parquet(generation / "layer_relevance.parquet")
                transitions = pd.read_csv(
                    generation / "layer_transition_metrics.csv"
                )
                identity = {
                    "profile": request.profile,
                    "source": request.source,
                    "target": request.target,
                }
                if (
                    not {
                        "profile",
                        "source",
                        "target",
                        "sample_id",
                        "layer",
                        "signed_sum",
                        "absolute_mass",
                    }.issubset(layers.columns)
                    or not {
                        "profile",
                        "source",
                        "target",
                        "sample_id",
                        "previous_layer",
                        "current_layer",
                    }.issubset(transitions.columns)
                    or any(
                        not (frame[name] == value).all()
                        for frame in (layers, transitions)
                        for name, value in identity.items()
                    )
                ):
                    raise ValueError("invalid Task 8 generation schema or identity")
        elif request.kind == "suite_aggregate":
            from .layerwise_paths import resolve_active_generation

            generation = resolve_active_generation(
                artifacts["index"].parent,
                expected_role="suite_aggregates",
            )
            manifest = json.loads(
                (generation / "aggregate_manifest.json").read_text(encoding="utf-8")
            )
            required_tables = {
                "layerwise_performance.csv",
                "emergence_layers.csv",
                "language_shift_by_layer.csv",
                "encoder_relevance_agreement.csv",
                "spectral_divergence_by_layer.csv",
                "final_decision_reorganization.csv",
            }
            if (
                manifest.get("schema_version") != 1
                or manifest.get("profiles") != list(request.suite_config.profiles)
                or {item.get("name") for item in manifest.get("tables", [])}
                != required_tables
                or not all((generation / name).is_file() for name in required_tables)
            ):
                raise ValueError("invalid Task 10 aggregate generation")
        else:
            payload = json.loads(artifacts["index"].read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") != 1
                or payload.get("kind") != request.kind
                or payload.get("stage_id") != request.stage_id
            ):
                raise ValueError("invalid orchestration index artifact")

    def cleanup_encoder(self, encoder: object | None) -> None:
        if encoder is None:
            return
        model = getattr(encoder, "_model", None)
        if model is not None and callable(getattr(model, "to", None)):
            model.to("cpu")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
