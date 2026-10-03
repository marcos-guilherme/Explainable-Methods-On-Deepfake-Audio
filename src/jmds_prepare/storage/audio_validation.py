"""Decode-level validation of stored audio files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def is_valid_flac(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with sf.SoundFile(path, mode="r") as audio:
            expected_frames = len(audio)
            if (
                audio.format != "FLAC"
                or expected_frames <= 0
                or audio.channels <= 0
                or audio.samplerate <= 0
            ):
                return False
            total_frames = 0
            while True:
                block = audio.read(
                    frames=65_536,
                    dtype="float64",
                    always_2d=True,
                )
                if block.shape[0] == 0:
                    break
                if (
                    block.shape[1] != audio.channels
                    or not np.isfinite(block).all()
                ):
                    return False
                total_frames += block.shape[0]
    except (OSError, RuntimeError, ValueError):
        return False
    return total_frames > 0 and total_frames == expected_frames
