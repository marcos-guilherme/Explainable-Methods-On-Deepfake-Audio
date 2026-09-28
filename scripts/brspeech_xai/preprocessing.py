"""Pré-processamento de áudio (mono, resample 16k, fix length, normalização).

Portado verbatim de ``notebooks/deepfake_brspeech_explainability.ipynb``.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Reduz um waveform para mono (média dos canais) e achata para 1D."""
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    return waveform.reshape(-1)


def resample_to_16k(waveform: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """Reamostra o waveform para ``target_sr`` (16 kHz por padrão)."""
    if orig_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)


def fix_length(waveform: torch.Tensor, num_samples: int = 64600) -> torch.Tensor:
    """Ajusta o waveform para ``num_samples`` (trunca ou repete/preenche)."""
    length = waveform.shape[0]
    if length >= num_samples:
        return waveform[:num_samples]
    n_repeats = num_samples // length + 1
    return waveform.repeat(n_repeats)[:num_samples]


def normalize_waveform(waveform: torch.Tensor, mode: str = "layernorm") -> torch.Tensor:
    """Normaliza o waveform via ``layernorm`` ou ``peak``."""
    if mode == "layernorm":
        return F.layer_norm(waveform, waveform.shape)
    if mode == "peak":
        peak = waveform.abs().max()
        return waveform / peak if peak > 0 else waveform
    raise ValueError(f"mode invalido: {mode!r}. Use 'layernorm' ou 'peak'.")


def preprocess(
    audio_array: np.ndarray,
    orig_sr: int,
    num_samples: int = 64600,
    norm_mode: str = "layernorm",
) -> torch.Tensor:
    """Pipeline completo: tensor -> mono -> resample 16k -> fix length -> normalização."""
    waveform = torch.as_tensor(audio_array, dtype=torch.float32)
    waveform = to_mono(waveform)
    waveform = resample_to_16k(waveform, orig_sr)
    waveform = fix_length(waveform, num_samples)
    waveform = normalize_waveform(waveform, mode=norm_mode)
    return waveform
