"""Oclusão espectral causal: bordas mel, filtro band-stop e queda de P(spoof) por banda."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .bands import mel_band_edges  # noqa: F401 (re-exportado; fonte única em bands.py)
from .logging_utils import progress
from .preprocessing import resample_to_16k, to_mono


def bandstop(wav: np.ndarray, low: float, high: float, sr: int = 16000) -> np.ndarray:
    """Remove a banda [low, high] Hz via Butterworth band-stop (zero-phase)."""
    sos = butter(4, [low, high], btype="bandstop", fs=sr, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def occlusion_drop(p_spoof_fn, audios, srs, band_edges, desc="oclusão") -> np.ndarray:
    """Queda de P(spoof) por clipe ao ocluir cada banda (baseline = áudio íntegro).

    Sinal (mesma convenção do notebook): valor positivo = ocluir a banda derruba
    P(spoof) (a banda SUSTENTA spoof); negativo = ocluir sobe P(spoof) (a banda
    empurra para bonafide).

    p_spoof_fn: função (audio, sr) -> P(spoof) do detector alvo.
    desc: rótulo da barra de progresso (varre uma banda por passo).
    Returns: matriz (n_clips, n_bands) de quedas por clipe (baseline - ocluído).
        A média por banda é `drops.mean(axis=0)`; a incerteza vem de `bootstrap_ci`.
    """
    base = np.array([p_spoof_fn(a, sr) for a, sr in zip(audios, srs)])
    cols = []
    bands = list(zip(band_edges[:-1], band_edges[1:]))
    for lo, hi in progress(bands, desc=desc, unit="band"):
        occ = np.array([p_spoof_fn(bandstop(a.astype(np.float32), lo, hi, sr), sr)
                        for a, sr in zip(audios, srs)])
        cols.append(base - occ)
    return np.column_stack(cols) if cols else np.empty((len(base), 0))


def grouped_occlusion_drop(p_spoof_fn, audios, srs, band_edges, band_indices,
                           desc="oclusão de grupo") -> np.ndarray:
    """Queda de P(spoof) por clipe ao ocluir um GRUPO de bandas de uma só vez.

    Diferente de `occlusion_drop` (uma banda por vez), aqui removemos todas as bandas
    de `band_indices` simultaneamente (band-stops encadeados). Serve ao teste de
    convergência: ocluir o grupo de bandas mais associadas (H1) versus o menos
    associadas, medindo o efeito causal conjunto por clipe.

    Args:
        band_indices: índices 0-based das bandas a remover (referem-se a `band_edges`).

    Returns:
        Vetor (n_clips,) de quedas `base - ocluído` por clipe. Grupo vazio => zeros.
    """
    band_indices = list(band_indices)
    base = np.array([p_spoof_fn(a, sr) for a, sr in zip(audios, srs)])
    if not band_indices:
        return np.zeros(len(base), dtype=np.float64)
    occ = []
    for a, sr in progress(list(zip(audios, srs)), desc=desc, unit="clip"):
        wav = a.astype(np.float32)
        for bi in band_indices:
            wav = bandstop(wav, band_edges[bi], band_edges[bi + 1], sr)
        occ.append(p_spoof_fn(wav, sr))
    return base - np.array(occ)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """IC percentil por bootstrap da MÉDIA, por coluna de uma matriz (n_obs, n_cols).

    Reamostra as observações (linhas) com reposição `n_boot` vezes e devolve os
    percentis (alpha/2, 1-alpha/2) das médias por coluna.

    Returns: (ci_low, ci_high), cada um array (n_cols,).
    """
    values = np.atleast_2d(values)
    n_obs, n_cols = values.shape
    if n_obs < 2:
        m = values.mean(axis=0) if n_obs else np.zeros(n_cols)
        return m.copy(), m.copy()
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_boot, n_cols), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        boot_means[b] = values[idx].mean(axis=0)
    lo = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


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
