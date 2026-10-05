"""Standalone CLI for canonical XAI dataset preparation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .pipelines.xai_dataset import (
    build_xai_dataset,
    default_xai_dataset_deps,
    require_canonical_sample_rate,
)
from .pipelines.xai_materialization import ArchiveSpecKey, standard_archive_spec
from .pipelines.xai_selection import (
    SelectionCandidate,
    adapt_add_metadata,
    adapt_aishell3_metadata,
    adapt_coraa_metadata,
    adapt_english_processed,
    adapt_mlaad_metadata,
)
from .profiles.xai import XAI_PROFILE_BY_LANGUAGE
from .storage.xai_layout import XaiLayout


@dataclass(frozen=True)
class XaiDatasetCliConfig:
    language: str
    output_root: Path
    seed: int
    sample_rate: int
    english_manifest: Path | None
    pristine_manifest: Path | None
    generated_manifest: Path | None
    coraa_train_rar_part1: Path | None
    coraa_dev_zip: Path | None
    coraa_test_zip: Path | None
    aishell_archive: Path | None
    source_provenance: Path | None
    unrar_executable: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = _config_from_arguments(arguments)
        _validate_config(config)
        candidates = _load_candidates(config)
        archive_specs = _archive_specs_for_config(config)
        input_paths = _input_paths_for_config(config)
        profile = XAI_PROFILE_BY_LANGUAGE[config.language]
        build_xai_dataset(
            candidates,
            profile=profile,
            seed=config.seed,
            layout=XaiLayout(config.output_root),
            archive_specs=archive_specs,
            sample_rate=config.sample_rate,
            input_paths=input_paths,
            deps=default_xai_dataset_deps(),
            unrar_executable=config.unrar_executable,
            source_provenance_path=config.source_provenance,
        )
    except (OSError, ValueError) as error:
        message = " ".join(str(error).splitlines())
        print(f"error: {message}", file=sys.stdout)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xai-dataset",
        description=(
            "Select, materialize, audit and publish one canonical XAI dataset "
            "without modifying source archives."
        ),
    )
    parser.add_argument(
        "language",
        nargs="?",
        choices=sorted(XAI_PROFILE_BY_LANGUAGE),
        help="Language code (eng, por, zho).",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        choices=sorted(XAI_PROFILE_BY_LANGUAGE),
        help="Language code alternative to the positional subcommand.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for manifests, reports and processed cache.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Deterministic seed.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Target processed sample rate (must be 16000).",
    )
    parser.add_argument(
        "--english-manifest",
        type=Path,
        help="Processed English manifest with both classes.",
    )
    parser.add_argument(
        "--pristine-manifest",
        type=Path,
        help="Native pristine metadata CSV (CORAA or AISHELL-3).",
    )
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        help="Native generated metadata CSV (MLAAD or ADD).",
    )
    parser.add_argument(
        "--coraa-train-rar-part1",
        type=Path,
        help="CORAA train.part1.rar path (Portuguese).",
    )
    parser.add_argument(
        "--coraa-dev-zip",
        type=Path,
        help="CORAA dev archive path (Portuguese).",
    )
    parser.add_argument(
        "--coraa-test-zip",
        type=Path,
        help="CORAA test archive path (Portuguese).",
    )
    parser.add_argument(
        "--aishell-archive",
        type=Path,
        help="AISHELL-3 archive path for WAV extraction (Mandarin).",
    )
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="Optional upstream metadata provenance JSON (e.g. mandarin_metadata_provenance.json).",
    )
    parser.add_argument(
        "--unrar-executable",
        default="UnRAR",
        help="UnRAR executable used for CORAA train extraction.",
    )
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> XaiDatasetCliConfig:
    language = arguments.language_flag or arguments.language
    if language is None:
        raise ValueError("language is required")
    return XaiDatasetCliConfig(
        language=language,
        output_root=Path(arguments.output_root),
        seed=arguments.seed,
        sample_rate=arguments.sample_rate,
        english_manifest=arguments.english_manifest,
        pristine_manifest=arguments.pristine_manifest,
        generated_manifest=arguments.generated_manifest,
        coraa_train_rar_part1=arguments.coraa_train_rar_part1,
        coraa_dev_zip=arguments.coraa_dev_zip,
        coraa_test_zip=arguments.coraa_test_zip,
        aishell_archive=arguments.aishell_archive,
        source_provenance=arguments.source_provenance,
        unrar_executable=arguments.unrar_executable,
    )


def _validate_config(config: XaiDatasetCliConfig) -> None:
    if isinstance(config.seed, bool) or config.seed < 0:
        raise ValueError(f"Invalid seed: {config.seed}")
    require_canonical_sample_rate(config.sample_rate)

    if config.source_provenance is not None:
        _require_existing_file(config.source_provenance, label="source provenance")

    if config.language == "eng":
        if config.english_manifest is None:
            raise ValueError("English preparation requires --english-manifest")
        _require_existing_file(config.english_manifest, label="english manifest")
        return

    if config.language == "por":
        if config.pristine_manifest is None:
            raise ValueError("Portuguese preparation requires --pristine-manifest")
        if config.generated_manifest is None:
            raise ValueError("Portuguese preparation requires --generated-manifest")
        for label, path in (
            ("coraa train rar part1", config.coraa_train_rar_part1),
            ("coraa dev zip", config.coraa_dev_zip),
            ("coraa test zip", config.coraa_test_zip),
        ):
            if path is None:
                raise ValueError("Portuguese preparation requires --coraa-* paths")
            _require_existing_file(path, label=label)
        _validate_coraa_train_volumes(config.coraa_train_rar_part1)
        _require_existing_file(config.pristine_manifest, label="pristine manifest")
        _require_existing_file(config.generated_manifest, label="generated manifest")
        return

    if config.language == "zho":
        if config.pristine_manifest is None:
            raise ValueError("Mandarin preparation requires --pristine-manifest")
        if config.generated_manifest is None:
            raise ValueError("Mandarin preparation requires --generated-manifest")
        if config.aishell_archive is None:
            raise ValueError("Mandarin preparation requires --aishell-archive")
        _require_existing_file(config.pristine_manifest, label="pristine manifest")
        _require_existing_file(config.generated_manifest, label="generated manifest")
        _require_existing_file(config.aishell_archive, label="aishell archive")
        return

    raise ValueError(f"Unsupported language: {config.language}")


def _validate_coraa_train_volumes(part1: Path | None) -> None:
    if part1 is None:
        raise ValueError("Missing CORAA train.part1.rar path")
    spec = standard_archive_spec(
        corpus="CORAA",
        native_split="train",
        archive_path=part1,
    )
    missing = [path for path in spec.volume_paths if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"Missing CORAA train RAR volume(s): {joined}")


def _require_existing_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _load_candidates(config: XaiDatasetCliConfig) -> tuple[SelectionCandidate, ...]:
    if config.language == "eng":
        assert config.english_manifest is not None
        frame = _read_csv(config.english_manifest)
        return adapt_english_processed(
            frame,
            metadata_source=str(config.english_manifest),
        )

    if config.language == "por":
        assert config.pristine_manifest is not None
        assert config.generated_manifest is not None
        pristine = adapt_coraa_metadata(
            _read_csv(config.pristine_manifest),
            metadata_source=str(config.pristine_manifest),
        )
        generated = adapt_mlaad_metadata(
            _read_csv(config.generated_manifest),
            metadata_source=str(config.generated_manifest),
        )
        return pristine + generated

    assert config.pristine_manifest is not None
    assert config.generated_manifest is not None
    pristine = adapt_aishell3_metadata(
        _read_csv(config.pristine_manifest),
        metadata_source=str(config.pristine_manifest),
    )
    generated = adapt_add_metadata(
        _read_csv(config.generated_manifest),
        metadata_source=str(config.generated_manifest),
    )
    return pristine + generated


def _archive_specs_for_config(
    config: XaiDatasetCliConfig,
) -> dict[ArchiveSpecKey, ArchiveSpec]:
    if config.language == "por":
        assert config.coraa_train_rar_part1 is not None
        assert config.coraa_dev_zip is not None
        assert config.coraa_test_zip is not None
        return {
            ("CORAA", "train"): standard_archive_spec(
                corpus="CORAA",
                native_split="train",
                archive_path=config.coraa_train_rar_part1,
            ),
            ("CORAA", "dev"): standard_archive_spec(
                corpus="CORAA",
                native_split="dev",
                archive_path=config.coraa_dev_zip,
            ),
            ("CORAA", "test"): standard_archive_spec(
                corpus="CORAA",
                native_split="test",
                archive_path=config.coraa_test_zip,
            ),
        }
    if config.language == "zho":
        assert config.aishell_archive is not None
        return {
            ("AISHELL-3", "train"): standard_archive_spec(
                corpus="AISHELL-3",
                native_split="train",
                archive_path=config.aishell_archive,
            ),
            ("AISHELL-3", "test"): standard_archive_spec(
                corpus="AISHELL-3",
                native_split="test",
                archive_path=config.aishell_archive,
            ),
        }
    return {}


def _input_paths_for_config(config: XaiDatasetCliConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if config.english_manifest is not None:
        paths["english_manifest"] = config.english_manifest
    if config.pristine_manifest is not None:
        paths["pristine_manifest"] = config.pristine_manifest
    if config.generated_manifest is not None:
        paths["generated_manifest"] = config.generated_manifest
    if config.aishell_archive is not None:
        paths["aishell_archive"] = config.aishell_archive
    if config.coraa_train_rar_part1 is not None:
        paths["coraa_train_rar_part1"] = config.coraa_train_rar_part1
    if config.coraa_dev_zip is not None:
        paths["coraa_dev_zip"] = config.coraa_dev_zip
    if config.coraa_test_zip is not None:
        paths["coraa_test_zip"] = config.coraa_test_zip
    return paths


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
