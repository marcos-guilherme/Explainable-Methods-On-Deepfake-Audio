import numpy as np

from brspeech_xai import bands
from brspeech_xai.features import band_group, band_label, mel_band_features


def teardown_function(_):
    bands.configure_bands(8)  # restaura a grade de referência após cada teste


def test_mel_band_features_shape_and_finite():
    bands.configure_bands(8)                                     # grade de referência
    sig = np.sin(np.linspace(0, 200, 32000)).astype(np.float32)  # 2s @ 16k
    feats = mel_band_features(sig, orig_sr=16000)
    assert feats.shape == (len(bands.BAND_COLS),)   # 2 * N_BANDS = 16
    assert np.isfinite(feats).all()


def test_mel_band_features_follows_configured_grid():
    bands.configure_bands(16)                                    # muda a resolução
    sig = np.sin(np.linspace(0, 200, 32000)).astype(np.float32)
    feats = mel_band_features(sig, orig_sr=16000)
    assert feats.shape == (32,)   # 2 * 16
    bands.configure_bands(8)                                     # restaura p/ outros testes


def test_band_label_and_group():
    assert "kHz" in band_label("band1_mean")
    assert band_label("band1_mean").endswith("\u03bc")   # μ
    assert band_label("band1_std").endswith("\u03c3")    # σ
    assert band_group("band1_mean") == "low"             # banda grave (<500 Hz)
    assert band_group("band8_std") == "high"             # banda aguda (>2000 Hz)
