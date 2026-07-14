import io

import numpy as np
import soundfile as sf

from brspeech_xai.data import decode_audio


def test_decode_audio_from_bytes():
    sig = np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, 16000, format="WAV")
    wav, sr = decode_audio({"bytes": buf.getvalue(), "path": None})
    assert sr == 16000
    assert wav.shape[0] == 8000
