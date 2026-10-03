"""Opt-in integration tests against a local AISHELL-3 archive.

Set ``AISHELL3_ARCHIVE`` to the path of ``data_aishell3.tgz`` and run::

    pytest -m integration tests/test_aishell3_integration.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jmds_prepare.profiles.mandarin import MANDARIN_PROFILE
from jmds_prepare.sources.aishell3 import read_aishell3_metadata

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_aishell_archive() -> Path:
    raw = os.environ.get("AISHELL3_ARCHIVE")
    if not raw:
        pytest.skip("AISHELL3_ARCHIVE is not set")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"AISHELL3_ARCHIVE not found: {path}")
    return path


def test_smoke_read_real_archive(real_aishell_archive: Path) -> None:
    frame, reconciliation = read_aishell3_metadata(
        real_aishell_archive, profile=MANDARIN_PROFILE
    )

    assert len(frame) == 88035
    assert int((frame["split"] == "train").sum()) == 63262
    assert int((frame["split"] == "test").sum()) == 24773
    assert frame["utt_id"].is_unique
    assert set(frame["split"]) == {"train", "test"}

    train = reconciliation.by_split["train"]
    test = reconciliation.by_split["test"]
    assert train["matched_count"] == 63262
    assert test["matched_count"] == 24773
    assert train["content_without_wav"] == []
    assert train["wav_without_content"] == []
    assert test["content_without_wav"] == []
    assert test["wav_without_content"] == []

    assert reconciliation.observed_test_content_line_count == 24773
    assert reconciliation.official_test_sample_count == 23262
