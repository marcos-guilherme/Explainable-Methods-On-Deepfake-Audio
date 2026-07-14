"""Features MFCC: 13 coeficientes colapsados por média/desvio -> 26 features.

Extraídas do MESMO sinal pré-processado que entra no detector (mono/16k/janela/layer_norm),
garantindo que a descrição acústica corresponde ao que o modelo ouve.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio

from .preprocessing import preprocess

_MFCC = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=13,
    melkwargs={"n_fft": 400, "win_length": 400, "hop_length": 160, "n_mels": 40},
)  # win 25 ms, hop 10 ms @ 16 kHz

MFCC_COLS = [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]


def mfcc_features(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    """13 MFCCs -> colapso temporal (média + desvio) -> 26 features.

    Returns:
        Vetor (26,): [mean_1..13, std_1..13].
    """
    wav = preprocess(audio_array, orig_sr)                 # (T,) normalizado
    coeffs = _MFCC(wav.unsqueeze(0)).squeeze(0)            # (13, n_frames)
    mean = coeffs.mean(dim=1)
    std = coeffs.std(dim=1)
    return torch.cat([mean, std]).numpy().astype(np.float32)
