"""Tests for the standalone XAI dataset CLI argument validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jmds_prepare import xai_dataset as cli
from jmds_prepare.pipelines.xai_materialization import CORAA_TRAIN_RAR_VOLUME_NAMES
from jmds_prepare.sources.aishell3 import AISHELL3_COLUMNS


def test_cli_eng_requires_english_manifest(tmp_path: Path):
    exit_code = cli.main(
        [
            "eng",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
        ]
    )
    assert exit_code == 1


def test_cli_por_requires_coraa_archives(tmp_path: Path):
    pristine = tmp_path / "coraa.csv"
    generated = tmp_path / "mlaad.csv"
    pristine.write_text("file_path,split\n", encoding="utf-8")
    generated.write_text("utt_id,split\n", encoding="utf-8")
    exit_code = cli.main(
        [
            "por",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--pristine-manifest",
            str(pristine),
            "--generated-manifest",
            str(generated),
        ]
    )
    assert exit_code == 1


def test_cli_zho_requires_pristine_and_aishell_archive(tmp_path: Path):
    generated = tmp_path / "add.csv"
    generated.write_text("utt_id,split\n", encoding="utf-8")
    exit_code = cli.main(
        [
            "zho",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--generated-manifest",
            str(generated),
        ]
    )
    assert exit_code == 1


def test_load_candidates_zho_reads_pristine_manifest_csv(tmp_path: Path):
    pristine = tmp_path / "aishell3_metadata.csv"
    generated = tmp_path / "add.csv"
    pristine.write_text(
        "utt_id,split,archive_member_path,speaker_id,gender,age,accent,"
        "transcription,pinyin,prosody_available,metadata_source_file,"
        "metadata_source_row\n"
        "utt-1,train,train/wav/spk1/utt-1.wav,spk1,F,20,standard,"
        "ni,ni hao,yes,archive.tgz,10\n",
        encoding="utf-8",
    )
    generated.write_text(
        "spk_id,utt_id,gender,codec,attack_id,label,native,language,dataset,split,"
        "metadata_source_file,metadata_source_row,audio_path\n"
        "unk,T_0000123456,unk,-,unk,generated,yes,zho,ADD,train,"
        "jmds.csv,2,/jmds/train/T_0000123456.wav\n",
        encoding="utf-8",
    )
    config = cli.XaiDatasetCliConfig(
        language="zho",
        output_root=tmp_path / "out",
        seed=7,
        sample_rate=16_000,
        english_manifest=None,
        pristine_manifest=pristine,
        generated_manifest=generated,
        coraa_train_rar_part1=None,
        coraa_dev_zip=None,
        coraa_test_zip=None,
        aishell_archive=tmp_path / "aishell.tgz",
        source_provenance=None,
        unrar_executable="UnRAR",
    )
    candidates = cli._load_candidates(config)
    assert len(candidates) >= 1
    aishell = [item for item in candidates if item.corpus == "AISHELL-3"]
    assert aishell[0].original_ref == "train/wav/spk1/utt-1.wav"
    assert list(pd.read_csv(pristine, dtype=str, keep_default_na=False).columns) == list(
        AISHELL3_COLUMNS
    )


def test_cli_por_missing_coraa_train_sibling_volume_fails_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_build(*_args, **_kwargs):
        raise AssertionError("build_xai_dataset must not run")

    monkeypatch.setattr(cli, "build_xai_dataset", fail_build)
    pristine = tmp_path / "coraa.csv"
    generated = tmp_path / "mlaad.csv"
    pristine.write_text("file_path,split\n", encoding="utf-8")
    generated.write_text("utt_id,split\n", encoding="utf-8")
    part1 = tmp_path / "coraa" / CORAA_TRAIN_RAR_VOLUME_NAMES[0]
    part1.parent.mkdir(parents=True)
    part1.write_bytes(b"part1")
    dev_zip = tmp_path / "coraa" / "dev.zip"
    test_zip = tmp_path / "coraa" / "test.zip"
    dev_zip.write_bytes(b"dev")
    test_zip.write_bytes(b"test")
    exit_code = cli.main(
        [
            "por",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--pristine-manifest",
            str(pristine),
            "--generated-manifest",
            str(generated),
            "--coraa-train-rar-part1",
            str(part1),
            "--coraa-dev-zip",
            str(dev_zip),
            "--coraa-test-zip",
            str(test_zip),
        ]
    )
    assert exit_code == 1


def test_cli_rejects_non_canonical_sample_rate(tmp_path: Path):
    exit_code = cli.main(
        [
            "eng",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--sample-rate",
            "8000",
            "--english-manifest",
            str(tmp_path / "missing.csv"),
        ]
    )
    assert exit_code == 1


def test_cli_unknown_language_rejected(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--language",
                "fra",
                "--output-root",
                str(tmp_path / "out"),
                "--seed",
                "7",
            ]
        )
    assert exc.value.code == 2


def test_cli_missing_manifest_fails_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("materialization must not run")

    monkeypatch.setattr(cli, "build_xai_dataset", fail_materialize)
    exit_code = cli.main(
        [
            "eng",
            "--output-root",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--english-manifest",
            str(tmp_path / "missing.csv"),
        ]
    )
    assert exit_code == 1
