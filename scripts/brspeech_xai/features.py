"""Features MFCC: 13 coeficientes colapsados por média/desvio -> 26 features.

Extraídas do MESMO sinal pré-processado que entra no detector (mono/16k/janela/layer_norm),
garantindo que a descrição acústica corresponde ao que o modelo ouve.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

MFCC_COLS = [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]


@lru_cache(maxsize=1)
def _mfcc_transform():
    """Instancia o transform MFCC sob demanda (evita importar torch no import do módulo,
    permitindo usar os utilitários de rótulo sem a dependência pesada)."""
    import torchaudio
    return torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=13,
        melkwargs={"n_fft": 400, "win_length": 400, "hop_length": 160, "n_mels": 40},
    )  # win 25 ms, hop 10 ms @ 16 kHz

# torchaudio retorna c0..c12; nossas colunas mfcc1..mfcc13 mapeiam para c0..c12
# (off-by-one: mfcc1 == c0 == energia log). Grupos por ordem cepstral:
#   c0 -> energy | c1..c4 -> envelope (tilt) | c5..c12 -> detail (fino)
_MFCC_RE = re.compile(r"^mfcc(\d+)_(mean|std)$")


def _cep_index(col: str) -> tuple[int, str]:
    """Extrai (índice cepstral c, estatística) de uma coluna 'mfcc{i}_{mean,std}'."""
    m = _MFCC_RE.match(col)
    if not m:
        raise ValueError(f"coluna MFCC inválida: {col!r}")
    return int(m.group(1)) - 1, m.group(2)  # mfcc1 -> c0


def mfcc_group(col: str) -> str:
    """Grupo acústico da feature: 'energy', 'envelope' ou 'detail'."""
    c, _ = _cep_index(col)
    if c == 0:
        return "energy"
    return "envelope" if c <= 4 else "detail"


def mfcc_label(col: str) -> str:
    """Rótulo descritivo p/ figuras, ex.: 'c0 energy·σ', 'c3 envelope·μ'."""
    c, stat = _cep_index(col)
    stat_sym = "\u03bc" if stat == "mean" else "\u03c3"  # μ / σ
    return f"c{c} {mfcc_group(col)}\u00b7{stat_sym}"


def mfcc_features(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    """13 MFCCs -> colapso temporal (média + desvio) -> 26 features.

    Returns:
        Vetor (26,): [mean_1..13, std_1..13].
    """
    import torch

    from .preprocessing import preprocess
    wav = preprocess(audio_array, orig_sr)                 # (T,) normalizado
    coeffs = _mfcc_transform()(wav.unsqueeze(0)).squeeze(0)  # (13, n_frames)
    mean = coeffs.mean(dim=1)
    std = coeffs.std(dim=1)
    return torch.cat([mean, std]).numpy().astype(np.float32)
