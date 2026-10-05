"""Orquestração da matriz trilíngue fonte→alvo com calibração source-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from . import artifacts as A
from .adaptation import fit_head, score_head
from .config import RunConfig, load_config
from .metrics import calibrate_threshold, evaluate_at_threshold

LANGUAGES = ("eng", "por", "zho")
ROLES = ("train", "calibration", "test")


@dataclass
class MatrixPaths:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for directory in ("data", "embeddings", "models", "cells", "matrix"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def catalog(self, language: str, role: str) -> Path:
        return self.root / "data" / language / role / "catalog.parquet"

    def cache_metadata(self, language: str, role: str) -> Path:
        return self.root / "data" / language / role / "embedding_cache.json"

    def embedding(self, language: str, role: str) -> Path:
        return self.root / "embeddings" / language / f"{role}.npy"

    def model_dir(self, source: str) -> Path:
        return self.root / "models" / source

    def model(self, source: str) -> Path:
        return self.model_dir(source) / "d_ad.joblib"

    def thresholds(self, source: str) -> Path:
        return self.model_dir(source) / "thresholds.json"

    def model_metadata(self, source: str) -> Path:
        return self.model_dir(source) / "metadata.json"

    def cell_dir(self, source: str, target: str) -> Path:
        return self.root / "cells" / f"{source}_to_{target}"

    def cell_scores(self, source: str, target: str) -> Path:
        return self.cell_dir(source, target) / "p_spoof_ad.npy"

    def cell_metrics(self, source: str, target: str) -> Path:
        return self.cell_dir(source, target) / "metrics.json"

    def cell_predictions(self, source: str, target: str) -> Path:
        return self.cell_dir(source, target) / "predictions.parquet"

    def performance(self) -> Path:
        return self.root / "matrix" / "performance.csv"

    def manifest(self) -> Path:
        return self.root / "matrix" / "run_manifest.json"


def _validate_configs(configs: Mapping[str, RunConfig]) -> None:
    if set(configs) != set(LANGUAGES):
        raise ValueError(f"configs deve conter exatamente {list(LANGUAGES)}")
    reference = configs[LANGUAGES[0]]
    for language in LANGUAGES:
        cfg = configs[language]
        if cfg.data.dataset_kind != "local_manifest":
            raise ValueError(f"{language}: matriz requer dataset_kind='local_manifest'")
        if not cfg.data.calibration_split or cfg.data.n_calibration_per_class <= 0:
            raise ValueError(f"{language}: calibração explícita é obrigatória")
        if cfg.adapt.cross_fit:
            raise ValueError(f"{language}: matriz não aceita adapt.cross_fit")
        if cfg.data.train_split != "train" or cfg.data.calibration_split != "calibration":
            raise ValueError(f"{language}: matriz requer roles canônicos train/calibration")
        if cfg.data.eval_split != "test":
            raise ValueError(f"{language}: matriz requer role canônico test")
        if asdict(cfg.model) != asdict(reference.model):
            raise ValueError("todos os idiomas devem usar a mesma configuração de encoder")
        if asdict(cfg.audio) != asdict(reference.audio):
            raise ValueError("todos os idiomas devem usar a mesma configuração de áudio")
        if asdict(cfg.bands) != asdict(reference.bands):
            raise ValueError("todos os idiomas devem usar a mesma grade de bandas")
        if asdict(cfg.occlusion) != asdict(reference.occlusion):
            raise ValueError("todos os idiomas devem usar a mesma configuração de oclusão")
        if asdict(cfg.association) != asdict(reference.association):
            raise ValueError("todos os idiomas devem usar a mesma configuração de associação")
        if cfg.seed != reference.seed:
            raise ValueError("todos os idiomas devem usar a mesma seed")
        if cfg.adapt.head != reference.adapt.head:
            raise ValueError("todos os idiomas devem usar o mesmo head")
        quotas = (
            cfg.data.n_train_per_class,
            cfg.data.n_calibration_per_class,
            cfg.data.n_test_per_class,
        )
        reference_quotas = (
            reference.data.n_train_per_class,
            reference.data.n_calibration_per_class,
            reference.data.n_test_per_class,
        )
        if quotas != reference_quotas:
            raise ValueError("todos os idiomas devem usar as mesmas quotas por role")


def _load_role(paths: MatrixPaths, language: str, role: str) -> tuple[np.ndarray, pd.DataFrame]:
    embedding_path = paths.embedding(language, role)
    catalog_path = paths.catalog(language, role)
    if not embedding_path.is_file() or not catalog_path.is_file():
        raise ValueError(f"artefatos ausentes para {language}/{role}")
    emb = A.load_npy(embedding_path)
    catalog = A.load_table(catalog_path)
    required = {"sample_id", "label"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"catálogo {language}/{role} sem colunas {sorted(required)}")
    if emb.ndim != 2 or len(emb) != len(catalog):
        raise ValueError(
            f"embedding/catálogo incompatíveis em {language}/{role}: "
            f"{emb.shape} vs {len(catalog)}"
        )
    labels = catalog["label"].to_numpy()
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{language}/{role} deve conter ambas as classes binárias")
    if not np.all(np.isfinite(emb)):
        raise ValueError(f"embeddings não finitos em {language}/{role}")
    return emb, catalog


def _threshold_payload(source: str, calibration: dict) -> dict:
    return {
        "schema_version": 1,
        "source_language": source,
        "calibration_source": f"{source}/calibration",
        "calibration_fallback_train": False,
        "detectors": {"ad": calibration},
    }


def run_matrix_from_embeddings(
    configs: Mapping[str, RunConfig],
    paths: MatrixPaths | str | Path,
    *,
    logger=None,
) -> pd.DataFrame:
    """Treina três heads, calibra na fonte e pontua as nove células sem retrain."""
    _validate_configs(configs)
    paths = paths if isinstance(paths, MatrixPaths) else MatrixPaths(paths)
    heads: dict[str, object] = {}
    thresholds: dict[str, float] = {}

    for source in LANGUAGES:
        cfg = configs[source]
        emb_train, train_catalog = _load_role(paths, source, "train")
        emb_cal, cal_catalog = _load_role(paths, source, "calibration")
        head = fit_head(
            emb_train,
            train_catalog["label"].to_numpy(),
            head=cfg.adapt.head,
            seed=cfg.seed,
        )
        calibration = calibrate_threshold(
            score_head(head, emb_cal),
            cal_catalog["label"].to_numpy(),
            f"{source}/calibration",
            "ad",
        )
        model_dir = paths.model_dir(source)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(head, paths.model(source))
        A.save_json(_threshold_payload(source, calibration), paths.thresholds(source))
        A.save_json(
            {
                "schema_version": 1,
                "source_language": source,
                "head": cfg.adapt.head,
                "seed": cfg.seed,
                "n_train": int(len(train_catalog)),
                "n_calibration": int(len(cal_catalog)),
                "config_hash": cfg.config_hash(),
            },
            paths.model_metadata(source),
        )
        heads[source] = head
        thresholds[source] = float(calibration["threshold"])

    rows: list[dict] = []
    target_data = {
        target: _load_role(paths, target, "test")
        for target in LANGUAGES
    }
    for source in LANGUAGES:
        for target in LANGUAGES:
            emb_target, catalog = target_data[target]
            labels = catalog["label"].to_numpy()
            scores = score_head(heads[source], emb_target)
            threshold = thresholds[source]
            metrics = evaluate_at_threshold(
                scores,
                labels,
                threshold,
                condition="diagonal" if source == target else "cross_corpus_shift",
                source=source,
                target=target,
            )
            cell_dir = paths.cell_dir(source, target)
            cell_dir.mkdir(parents=True, exist_ok=True)
            A.save_npy(scores.astype(np.float32), paths.cell_scores(source, target))
            A.save_json(metrics, paths.cell_metrics(source, target))
            predictions = catalog.copy()
            predictions["p_spoof"] = scores
            predictions["prediction"] = (scores >= threshold).astype(int)
            predictions["source_language"] = source
            predictions["target_language"] = target
            predictions["threshold_source"] = source
            A.save_table(predictions, paths.cell_predictions(source, target))

            threshold_free = metrics["threshold_free"]
            fixed = metrics["fixed_threshold"]
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "scope": "diagonal" if source == target else "cross_corpus_shift",
                    "roc_auc": threshold_free["roc_auc"],
                    "average_precision": threshold_free["average_precision"],
                    "eer_diagnostic": threshold_free["eer_diagnostic"],
                    "threshold_fixed": fixed["threshold"],
                    "accuracy": fixed["accuracy"],
                    "mcc": fixed["mcc"],
                    "tpr": fixed["tpr"],
                    "fpr": fixed["fpr"],
                    "fnr": fixed["fnr"],
                    "n": int(len(labels)),
                }
            )
            if logger:
                logger.info(f"célula {source}->{target}: n={len(labels)}")

    performance = pd.DataFrame(rows)
    A.save_table(performance, paths.performance())
    A.save_json(
        {
            "schema_version": 1,
            "languages": list(LANGUAGES),
            "cells": len(rows),
            "models": len(heads),
            "calibration_policy": "source_language_only",
            "off_diagonal_interpretation": "external_validation_under_corpus_shift",
        },
        paths.manifest(),
    )
    return performance


def _role_quota(cfg: RunConfig, role: str) -> int:
    return {
        "train": cfg.data.n_train_per_class,
        "calibration": cfg.data.n_calibration_per_class,
        "test": cfg.data.n_test_per_class,
    }[role]


def _cache_key(language: str, role: str, cfg: RunConfig) -> str:
    manifest = Path(cfg.data.manifest_path).resolve()
    if not manifest.is_file():
        raise ValueError(f"manifesto ausente para cache: {manifest}")
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = {
        "language": language,
        "role": role,
        "config_hash": cfg.config_hash(),
        "manifest_path": str(manifest),
        "manifest_sha256": digest.hexdigest(),
        "quota": _role_quota(cfg, role),
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_matrix_inputs(
    configs: Mapping[str, RunConfig],
    paths: MatrixPaths | str | Path,
    *,
    embedder=None,
    force: bool = False,
    logger=None,
) -> MatrixPaths:
    """Carrega cada idioma/papel uma vez e mantém embeddings reutilizáveis em disco."""
    from .data import build_balanced_split
    from .encoders import build_encoder

    _validate_configs(configs)
    paths = paths if isinstance(paths, MatrixPaths) else MatrixPaths(paths)
    if embedder is None:
        reference = configs[LANGUAGES[0]]
        embedder = build_encoder(reference.model, reference.device)

    for language in LANGUAGES:
        cfg = configs[language]
        for role in ROLES:
            embedding_path = paths.embedding(language, role)
            catalog_path = paths.catalog(language, role)
            metadata_path = paths.cache_metadata(language, role)
            key = _cache_key(language, role, cfg)
            if not force and embedding_path.is_file() and catalog_path.is_file() and metadata_path.is_file():
                try:
                    if A.load_json(metadata_path).get("cache_key") == key:
                        _load_role(paths, language, role)
                        continue
                except (OSError, ValueError):
                    pass

            audios, srs, labels, provenance = build_balanced_split(
                cfg.data.dataset_id,
                role,
                _role_quota(cfg, role),
                loader=cfg.data.loader,
                seed=cfg.seed,
                dataset_kind=cfg.data.dataset_kind,
                manifest_path=cfg.data.manifest_path,
            )
            if not provenance or not (
                len(audios) == len(srs) == len(labels) == len(provenance)
            ):
                raise ValueError(f"loader retornou artefatos incompatíveis para {language}/{role}")
            if (
                len({row.get("language") for row in provenance}) != 1
                or provenance[0].get("language") != language
            ):
                raise ValueError(f"manifesto de {language} retornou idioma incompatível")
            emb = np.asarray(
                embedder.extract_embeddings(audios, [int(sr) for sr in srs]),
                dtype=np.float32,
            )
            if emb.ndim != 2 or len(emb) != len(labels):
                raise ValueError(f"encoder retornou shape inválido para {language}/{role}: {emb.shape}")
            if not np.all(np.isfinite(emb)):
                raise ValueError(f"encoder retornou embeddings não finitos para {language}/{role}")
            catalog = pd.DataFrame(provenance)
            catalog["label"] = np.asarray(labels, dtype=int)
            catalog["sample_rate"] = np.asarray(srs, dtype=int)
            catalog["matrix_language"] = language
            catalog["matrix_role"] = role
            embedding_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            A.save_npy(emb, embedding_path)
            A.save_table(catalog, catalog_path)
            A.save_json(
                {
                    "schema_version": 1,
                    "cache_key": key,
                    "language": language,
                    "role": role,
                    "n": int(len(catalog)),
                    "embedding_dim": int(emb.shape[1]),
                },
                metadata_path,
            )
            if logger:
                logger.info(f"embedding {language}/{role}: {emb.shape}")
    return paths


def run_matrix(
    configs: Mapping[str, RunConfig],
    output_dir: str | Path,
    *,
    embedder=None,
    force: bool = False,
    logger=None,
) -> pd.DataFrame:
    paths = prepare_matrix_inputs(
        configs,
        output_dir,
        embedder=embedder,
        force=force,
        logger=logger,
    )
    return run_matrix_from_embeddings(configs, paths, logger=logger)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brspeech_xai.matrix")
    for language in LANGUAGES:
        parser.add_argument(f"--{language}-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--with-xai",
        action="store_true",
        help="executa XAI completo nas diagonais e diagnóstico reduzido off-diagonal",
    )
    parser.add_argument("--reduced-per-quadrant", type=int, default=25)
    parser.add_argument("--xai-plots", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configs = {
        language: load_config(getattr(args, f"{language}_config"))
        for language in LANGUAGES
    }
    import torch

    from .encoders import build_encoder
    from .seed import set_seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cfg in configs.values():
        if cfg.device == "auto":
            cfg.device = device
    set_seed(configs[LANGUAGES[0]].seed)
    reference = configs[LANGUAGES[0]]
    embedder = build_encoder(reference.model, reference.device)
    paths = MatrixPaths(args.output)
    run_matrix(configs, paths.root, embedder=embedder, force=args.force)
    if args.with_xai:
        from .matrix_xai import run_xai_scopes

        run_xai_scopes(
            configs,
            paths,
            embedder=embedder,
            reduced_per_quadrant=args.reduced_per_quadrant,
            plots=args.xai_plots,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
