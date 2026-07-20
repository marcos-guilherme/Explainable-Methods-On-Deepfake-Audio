import numpy as np

from brspeech_xai.occlusion import bootstrap_ci, mel_band_edges, bandstop, occlusion_drop


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


def test_occlusion_drop_returns_per_clip_matrix():
    edges = mel_band_edges(n_bands=3, f_min=20.0, f_max=7900.0)
    sr = 16000
    audios = [np.random.default_rng(k).standard_normal(sr).astype(np.float32) for k in range(4)]
    srs = [sr] * len(audios)
    # p_spoof "cai" proporcional à energia residual -> depende da banda ocluída.
    p_fn = lambda a, s: float(np.mean(a ** 2))
    drops = occlusion_drop(p_fn, audios, srs, edges)
    assert drops.shape == (len(audios), len(edges) - 1)   # (n_clips, n_bands)
    # baseline - occluído: remover energia reduz o "score" -> quedas positivas.
    assert np.all(drops >= -1e-6)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=[1.0, -2.0], scale=0.5, size=(200, 2))
    lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
    mean = values.mean(axis=0)
    assert np.all(lo <= mean) and np.all(mean <= hi)
    assert lo[0] > 0 and hi[1] < 0        # ICs separados do zero, sinais opostos
