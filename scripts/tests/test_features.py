import numpy as np

from brspeech_xai.features import mfcc_features


def test_mfcc_returns_26_features():
    sig = np.sin(np.linspace(0, 200, 32000)).astype(np.float32)  # 2s @ 16k
    feats = mfcc_features(sig, orig_sr=16000)
    assert feats.shape == (26,)
    assert np.isfinite(feats).all()
