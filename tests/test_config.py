from pathlib import Path

import pytest

from jmds_prepare.config import PreparationConfig


PROTOCOL_FILES = (
    "open_v2_train.cm.csv",
    "open_v2_dev.cm.csv",
    "open_v2_eval.cm.csv",
)


def make_valid_jmds_root(root: Path) -> Path:
    protocols = root / "cm_protocols"
    protocols.mkdir(parents=True)
    for filename in PROTOCOL_FILES:
        (protocols / filename).touch()
    (root / "dataset" / "English_ASVspoof2024_Generated").mkdir(parents=True)
    return root


def test_loads_yaml_configuration(tmp_path):
    jmds_root = make_valid_jmds_root(tmp_path / "jmds")
    data_root = tmp_path / "prepared"
    config_path = tmp_path / "english.yaml"
    config_path.write_text(
        "\n".join(
            (
                f"jmds_root: {jmds_root.as_posix()}",
                f"data_root: {data_root.as_posix()}",
                "zenodo_record_id: 123456",
                "supported_splits:",
                "  - train",
                "  - dev",
                "sample_rate: 22050",
            )
        ),
        encoding="utf-8",
    )

    config = PreparationConfig.load(config_path)

    assert config == PreparationConfig(
        jmds_root=jmds_root,
        data_root=data_root,
        zenodo_record_id=123456,
        supported_splits=("train", "dev"),
        sample_rate=22_050,
    )
    config.validate()


def test_rejects_missing_jmds_root(tmp_path):
    config = PreparationConfig(
        jmds_root=tmp_path / "missing",
        data_root=tmp_path / "prepared",
    )

    with pytest.raises(FileNotFoundError, match="JMDS root"):
        config.validate()


@pytest.mark.parametrize("missing_protocol", PROTOCOL_FILES)
def test_requires_all_jmds_protocols(tmp_path, missing_protocol):
    jmds_root = make_valid_jmds_root(tmp_path / "jmds")
    (jmds_root / "cm_protocols" / missing_protocol).unlink()
    config = PreparationConfig(
        jmds_root=jmds_root,
        data_root=tmp_path / "prepared",
    )

    with pytest.raises(FileNotFoundError, match=missing_protocol):
        config.validate()


def test_requires_generated_english_directory(tmp_path):
    jmds_root = make_valid_jmds_root(tmp_path / "jmds")
    (jmds_root / "dataset" / "English_ASVspoof2024_Generated").rmdir()
    config = PreparationConfig(
        jmds_root=jmds_root,
        data_root=tmp_path / "prepared",
    )

    with pytest.raises(FileNotFoundError, match="English_ASVspoof2024_Generated"):
        config.validate()


def test_rejects_eval_split(tmp_path):
    config = PreparationConfig(
        jmds_root=tmp_path,
        data_root=tmp_path / "prepared",
        supported_splits=("train", "eval"),
    )

    with pytest.raises(ValueError, match="eval"):
        config.validate()


def test_rejects_non_positive_sample_rate(tmp_path):
    config = PreparationConfig(
        jmds_root=tmp_path,
        data_root=tmp_path / "prepared",
        sample_rate=0,
    )

    with pytest.raises(ValueError, match="sample_rate"):
        config.validate()


def test_validation_does_not_create_data_root(tmp_path):
    jmds_root = make_valid_jmds_root(tmp_path / "jmds")
    data_root = tmp_path / "prepared"
    config = PreparationConfig(jmds_root=jmds_root, data_root=data_root)

    config.validate()

    assert not data_root.exists()
