import numpy as np

from brspeech_xai.occlusion import mel_band_edges, bandstop


def test_mel_band_edges_monotonic():
    edges = mel_band_edges(n_bands=8, f_min=20.0, f_max=7900.0)
    assert len(edges) == 9              # n_bands + 1
    assert np.all(np.diff(edges) > 0)   # crescente
    assert edges[0] >= 20.0 - 1e-6 and edges[-1] <= 7900.0 + 1e-6


def test_bandstop_reduces_band_energy():
    sr = 16000
    t = np.arange(sr) / sr
    # tom em 1 kHz deve ser atenuado por um band-stop 500-2000 Hz
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = bandstop(x, low=500.0, high=2000.0, sr=sr)
    assert np.var(y) < np.var(x)
