"""Sanidade do filtro por bandas da validação de fidelidade (só a matemática, sem GPU/modelo).

Garante a invertibilidade exata da ida-e-volta na STFT: manter todas as bandas reproduz o sinal
e remover todas zera. É o que sustenta a leitura das curvas (keep/delete) como efeito da máscara,
não do artefato da transformada.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

from faithfulness_bands import (band_mask_bins, band_of_bin,  # noqa: E402
                                reconstruct_with_band_mask)
from brspeech_xai import dft_lrp  # noqa: E402
from brspeech_xai.bands import mel_band_edges  # noqa: E402

WIN, HOP, SR = 512, 128, 16000


def _signal(n=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t)
    return x + 0.05 * rng.standard_normal(n)


def test_keep_all_bands_reproduces_signal():
    x = _signal()
    edges = mel_band_edges(24, 20.0, 7900.0)
    freqs = dft_lrp.rfft_frequencies(WIN, SR)
    mask = band_mask_bins(freqs, edges, keep_idx=range(len(edges) - 1))
    assert np.allclose(mask, 1.0)                      # todas as bandas cobrem todos os bins
    y = reconstruct_with_band_mask(x, mask, WIN, HOP)
    assert np.max(np.abs(y - x)) < 1e-9


def test_delete_all_bands_zeros_signal():
    x = _signal()
    edges = mel_band_edges(24, 20.0, 7900.0)
    freqs = dft_lrp.rfft_frequencies(WIN, SR)
    mask = band_mask_bins(freqs, edges, keep_idx=[])
    y = reconstruct_with_band_mask(x, mask, WIN, HOP)
    assert np.max(np.abs(y)) < 1e-9


def test_band_of_bin_covers_full_spectrum():
    edges = mel_band_edges(24, 20.0, 7900.0)
    freqs = dft_lrp.rfft_frequencies(WIN, SR)
    bob = band_of_bin(freqs, edges)
    assert bob.min() == 0 and bob.max() == len(edges) - 2   # DC -> banda 0; Nyquist -> última
    assert bob.shape == freqs.shape


def test_keep_disjoint_bands_are_additive():
    """keep(A) + keep(complemento de A) deve reconstruir o sinal (partição das bandas)."""
    x = _signal(seed=3)
    edges = mel_band_edges(24, 20.0, 7900.0)
    freqs = dft_lrp.rfft_frequencies(WIN, SR)
    a = list(range(0, 12))
    b = list(range(12, 24))
    ya = reconstruct_with_band_mask(x, band_mask_bins(freqs, edges, a), WIN, HOP)
    yb = reconstruct_with_band_mask(x, band_mask_bins(freqs, edges, b), WIN, HOP)
    assert np.max(np.abs((ya + yb) - x)) < 1e-9
