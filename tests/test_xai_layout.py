"""Tests for the immutable XAI output layout and path safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from jmds_prepare.core.xai_sample import SUPPORTED_LANGUAGES, SUPPORTED_ROLES
from jmds_prepare.storage.xai_layout import XaiLayout


def test_layout_exposes_data_manifests_reports_and_staging(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")

    assert layout.data_dir == tmp_path / "xai_out" / "data"
    assert layout.manifests_dir == tmp_path / "xai_out" / "manifests"
    assert layout.reports_dir == tmp_path / "xai_out" / "reports"
    assert layout.staging_dir == tmp_path / "xai_out" / "staging"


@pytest.mark.parametrize(
    ("language", "role", "label", "sample_id", "label_slug"),
    [
        ("eng", "train", 0, "eng-train-bonafide-c1", "bonafide"),
        ("por", "calibration", 1, "por-cal-spoof-c2", "spoof"),
        ("zho", "test", 0, "zho-test-bonafide-c3", "bonafide"),
    ],
)
def test_sample_path_follows_canonical_hierarchy(
    tmp_path: Path,
    language: str,
    role: str,
    label: int,
    sample_id: str,
    label_slug: str,
):
    layout = XaiLayout(tmp_path / "xai_out")
    path = layout.sample_path(language, role, label, sample_id)

    assert path == (
        tmp_path
        / "xai_out"
        / "data"
        / language
        / role
        / label_slug
        / f"{sample_id}.wav"
    )
    assert path.resolve().is_relative_to(layout.data_dir.resolve())


def test_sample_path_rejects_unsupported_language_role_or_label(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")

    with pytest.raises(ValueError, match="Unsupported language"):
        layout.sample_path("fra", "train", 0, "sample-a")
    with pytest.raises(ValueError, match="Unsupported role"):
        layout.sample_path("eng", "validation", 0, "sample-a")
    with pytest.raises(ValueError, match="Unsupported label"):
        layout.sample_path("eng", "train", 2, "sample-a")


@pytest.mark.parametrize(
    "sample_id",
    [
        "",
        "../escape",
        "has/slash",
        "has\\backslash",
        "C:drive",
        "CON",
        "a" * 201,
    ],
)
def test_sample_path_rejects_unsafe_sample_ids(tmp_path: Path, sample_id: str):
    layout = XaiLayout(tmp_path / "xai_out")
    with pytest.raises(ValueError, match="sample_id"):
        layout.sample_path("eng", "train", 0, sample_id)


def test_validate_sample_id_accepts_filesystem_safe_ids():
    for sample_id in ("eng-train-bonafide-00001", "por-cal-spoof-aishell3-abc123"):
        XaiLayout.validate_sample_id(sample_id)


def test_supported_vocabularies_match_xai_contract():
    assert set(SUPPORTED_LANGUAGES) == {"eng", "por", "zho"}
    assert set(SUPPORTED_ROLES) == {"train", "calibration", "test"}


@pytest.mark.parametrize("language", ["eng", "por", "zho"])
def test_language_artifact_paths_are_confined(tmp_path: Path, language: str):
    layout = XaiLayout(tmp_path / "xai_out")
    paths = (
        layout.samples_csv(language),
        layout.selection_report_json(language),
        layout.audio_audit_json(language),
        layout.provenance_json(language),
    )
    assert layout.samples_csv(language) == (
        tmp_path / "xai_out" / "manifests" / language / "xai_samples.csv"
    )
    assert layout.selection_report_json(language) == (
        tmp_path / "xai_out" / "reports" / language / "selection_report.json"
    )
    for path in paths:
        assert path.resolve().is_relative_to(layout.output_root.resolve())


def test_language_artifact_paths_reject_unsupported_language(tmp_path: Path):
    layout = XaiLayout(tmp_path / "xai_out")
    with pytest.raises(ValueError, match="Unsupported language"):
        layout.samples_csv("fra")
