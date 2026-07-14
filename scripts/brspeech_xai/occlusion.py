"""Oclusão espectral causal: bordas mel, filtro band-stop e queda de P(spoof) por banda."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .preprocessing import resample_to_16k, to_mono


def mel_band_edges(n_bands: int = 8, f_min: float = 20.0, f_max: float = 7900.0) -> np.ndarray:
    """n_bands+1 bordas de frequência (Hz) igualmente espaçadas em escala mel."""
    to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    return to_hz(np.linspace(to_mel(f_min), to_mel(f_max), n_bands + 1))


def bandstop(wav: np.ndarray, low: float, high: float, sr: int = 16000) -> np.ndarray:
    """Remove a banda [low, high] Hz via Butterworth band-stop (zero-phase)."""
    sos = butter(4, [low, high], btype="bandstop", fs=sr, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def occlusion_drop(p_spoof_fn, audios, srs, band_edges) -> np.ndarray:
    """Queda média de P(spoof) ao ocluir cada banda (baseline = áudio íntegro).

    p_spoof_fn: função (audio, sr) -> P(spoof) do detector alvo.
    Returns: array (n_bands,) de queda média (baseline - ocluído).
    """
    base = np.array([p_spoof_fn(a, sr) for a, sr in zip(audios, srs)])
    drops = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        occ = np.array([p_spoof_fn(bandstop(a.astype(np.float32), lo, hi, sr), sr)
                        for a, sr in zip(audios, srs)])
        drops.append(float(np.mean(base - occ)))
    return np.array(drops)


def stratified_idx(master, quad_col: str, per_quad: int = 150, seed: int = 42) -> np.ndarray:
    """Índices estratificados por quadrante (TP/TN/FP/FN), até `per_quad` por quadrante."""
    rng = np.random.default_rng(seed)
    idx = []
    for q in ["TP", "TN", "FP", "FN"]:
        pool = master.index[master[quad_col] == q].to_numpy()
        idx.extend(pool if len(pool) <= per_quad else rng.choice(pool, per_quad, replace=False))
    return np.array(idx)


def to_16k_mono(a: np.ndarray, sr: int) -> np.ndarray:
    """Reamostra para 16 kHz mono (numpy). A oclusão filtra sempre a 16 kHz, garantindo que
    as bordas de banda (até 7900 Hz) fiquem abaixo da Nyquist (8000 Hz) para qualquer sr nativo."""
    wav = to_mono(torch.as_tensor(a, dtype=torch.float32))
    return resample_to_16k(wav, sr).numpy()
