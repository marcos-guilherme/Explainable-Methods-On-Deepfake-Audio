from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from brspeech_xai.shortcut_audit import audit_manifest


def _tone(path: Path, *, seconds: float, amplitude: float) -> None:
    sample_rate = 16_000
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * 220 * time)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_audit_manifest_quantifies_class_corpus_and_low_level_shortcuts(tmp_path):
    rows = []
    for label, corpus, seconds, amplitude in (
        (0, "CORAA", 1.0, 0.1),
        (1, "MLAAD", 3.0, 0.8),
    ):
        for index in range(4):
            path = tmp_path / f"{label}-{index}.wav"
            _tone(path, seconds=seconds, amplitude=amplitude)
            rows.append(
                {
                    "sample_id": f"sample-{label}-{index}",
                    "language": "por",
                    "role": "test",
                    "label": label,
                    "corpus": corpus,
                    "processed_path": str(path),
                }
            )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = audit_manifest(manifest, role="test")

    assert set(result["feature"]) == {
        "duration_s",
        "peak",
        "rms",
        "first_250ms_rms",
        "leading_silence_s",
    }
    assert set(result["bonafide_corpora"]) == {"CORAA"}
    assert set(result["spoof_corpora"]) == {"MLAAD"}
    assert set(result["corpus_label_cramers_v"]) == {1.0}
    duration = result[result["feature"] == "duration_s"].iloc[0]
    assert duration["bonafide_median"] == 1.0
    assert duration["spoof_median"] == 3.0
    assert duration["roc_auc"] == 1.0
    assert duration["file_formats"] == "WAV/PCM_16/mono/16000Hz"
