"""Tests for shared two-source metadata configuration validation helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from jmds_prepare.metadata_config_common import (
    reject_nested_output,
    validate_jmds_protocols,
)
from jmds_prepare.profiles.portuguese import PORTUGUESE_PROFILE


def test_reject_nested_output_detects_child_with_label(tmp_path):
    outer = tmp_path / "jmds"
    inner = outer / "nested_output"
    outer.mkdir()
    inner.mkdir()
    with pytest.raises(ValueError) as error:
        reject_nested_output(
            output_root=inner.resolve(),
            forbidden_roots=[(outer.resolve(), "jmds_root")],
        )
    assert str(error.value) == "output_root must not be nested inside jmds_root"


def test_validate_jmds_protocols_requires_all_splits(tmp_path, monkeypatch):
    profile = replace(
        PORTUGUESE_PROFILE,
        protocol_splits=("train",),
    )
    jmds_root = tmp_path / "jmds"
    (jmds_root / "cm_protocols").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="open_v2_train"):
        validate_jmds_protocols(jmds_root, profile)
