"""Features acústicas: energia log-mel por banda, colapsada por média/desvio.

Extraídas do MESMO sinal pré-processado que entra no detector (mono/16k/janela/
layer_norm), garantindo que a descrição acústica corresponde ao que o modelo ouve.

Usa a MESMA partição de frequência da oclusão (`bands.BAND_EDGES`), de modo que a
Espinha 1 (associação) e a Espinha 2 (oclusão causal) vivem no mesmo eixo de Hz.
"""
from __future__ import annotations

import numpy as np

# Grade de banda: fonte única em bands.py. Importamos o MÓDULO (não os globais) para
# enxergar reconfigurações de resolução feitas por configure_bands em tempo de execução.
from . import bands
from .bands import band_group, band_index, band_label, mel_band_edges  # noqa: F401

# Framing do espectrograma (mesmo da antiga MFCC): win 25 ms, hop 10 ms @ 16 kHz.
_N_FFT = 400
_HOP = 160
_SR = 16000
_EPS = 1e-10


_MASK_CACHE: dict[bytes, np.ndarray] = {}


def _band_bin_masks() -> np.ndarray:
    """Máscara booleana (n_bands, n_freq_bins) dos bins de FFT em cada banda.

    Cache com chave nas bordas atuais: invalida sozinho quando configure_bands muda a grade.
    """
    edges = bands.BAND_EDGES
    key = edges.tobytes()
    masks = _MASK_CACHE.get(key)
    if masks is None:
        freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / _SR)  # (n_freq_bins,)
        lo, hi = edges[:-1], edges[1:]
        masks = np.stack([(freqs >= lo[b]) & (freqs < hi[b]) for b in range(bands.N_BANDS)])
        _MASK_CACHE[key] = masks
    return masks


def mel_band_features(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    """Energia log-mel por banda -> colapso temporal (média + desvio) -> 2*N_BANDS.

    Integra a potência da STFT nos bins de cada banda (bordas idênticas às da
    oclusão), toma o log e resume por média e desvio ao longo dos frames.

    Returns:
        Vetor (2*N_BANDS,): [mean_band1..N, std_band1..N].
    """
    import torch

    from .preprocessing import preprocess
    wav = preprocess(audio_array, orig_sr)                 # (T,) normalizado
    window = torch.hann_window(_N_FFT)
    spec = torch.stft(wav, n_fft=_N_FFT, hop_length=_HOP, win_length=_N_FFT,
                      window=window, return_complex=True)  # (n_freq, n_frames)
    power = spec.abs().pow(2).numpy()                      # (n_freq, n_frames)
    masks = _band_bin_masks()                              # (n_bands, n_freq)
    band_energy = masks.astype(np.float32) @ power         # (n_bands, n_frames)
    log_energy = np.log(band_energy + _EPS)
    mean = log_energy.mean(axis=1)
    std = log_energy.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)
