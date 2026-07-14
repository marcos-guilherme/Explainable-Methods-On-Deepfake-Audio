import numpy as np
import torch

from brspeech_xai.preprocessing import to_mono, fix_length, preprocess


def test_to_mono_averages_channels():
    stereo = torch.stack([torch.zeros(100), torch.ones(100)])  # (2, 100)
    mono = to_mono(stereo)
    assert mono.shape[-1] == 100
    assert torch.allclose(mono.float().mean(), torch.tensor(0.5), atol=1e-6)


def test_fix_length_pads_and_truncates():
    short = torch.ones(10)
    assert fix_length(short, num_samples=64600).shape[-1] == 64600
    long = torch.ones(70000)
    assert fix_length(long, num_samples=64600).shape[-1] == 64600


def test_preprocess_pipeline_shape():
    sig = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)  # 0.5s @ 16k
    out = preprocess(sig, orig_sr=16000, num_samples=64600)
    assert out.shape[-1] == 64600
