import numpy as np

from brspeech_xai.features import mfcc_features, mfcc_group, mfcc_label


def test_mfcc_returns_26_features():
    sig = np.sin(np.linspace(0, 200, 32000)).astype(np.float32)  # 2s @ 16k
    feats = mfcc_features(sig, orig_sr=16000)
    assert feats.shape == (26,)
    assert np.isfinite(feats).all()


def test_mfcc_label_offby_one_and_groups():
    # mfcc1 == c0 (energia); mfcc2 == c1 (envelope); mfcc13 == c12 (detail)
    assert mfcc_label("mfcc1_mean").startswith("c0 energy")
    assert mfcc_label("mfcc1_mean").endswith("\u03bc")   # μ
    assert mfcc_label("mfcc1_std").endswith("\u03c3")    # σ
    assert mfcc_group("mfcc1_mean") == "energy"
    assert mfcc_group("mfcc3_mean") == "envelope"
    assert mfcc_group("mfcc13_std") == "detail"
    assert "c12" in mfcc_label("mfcc13_std")
