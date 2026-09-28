"""Grade de bandas mel e rótulos por banda (sem dependências pesadas: só numpy).

Fonte única da verdade da partição de frequência usada tanto pela Espinha 1
(associação por energia de banda) quanto pela Espinha 2 (oclusão espectral),
garantindo que as duas vivem no MESMO eixo de frequência (comparação 1:1).
"""
from __future__ import annotations

import re

import numpy as np

# Grade canônica (idêntica ao default de OcclusionConfig).
N_BANDS = 8
F_MIN = 20.0
F_MAX = 7900.0


def mel_band_edges(n_bands: int = N_BANDS, f_min: float = F_MIN,
                   f_max: float = F_MAX) -> np.ndarray:
    """n_bands+1 bordas de frequência (Hz) igualmente espaçadas em escala mel."""
    to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    return to_hz(np.linspace(to_mel(f_min), to_mel(f_max), n_bands + 1))


def _band_cols(n_bands: int) -> list[str]:
    """Colunas de energia log-mel por banda, colapsada por média (μ) e desvio (σ)."""
    return ([f"band{k}_mean" for k in range(1, n_bands + 1)]
            + [f"band{k}_std" for k in range(1, n_bands + 1)])


BAND_EDGES = mel_band_edges()
BAND_COLS = _band_cols(N_BANDS)

_BAND_RE = re.compile(r"^band(\d+)_(mean|std)$")


def configure_bands(n_bands: int = N_BANDS, f_min: float = F_MIN,
                    f_max: float = F_MAX) -> None:
    """Reconfigura a grade compartilhada (H1 e H2) no módulo, em tempo de execução.

    Recomputa os globais `N_BANDS`, `BAND_EDGES` e `BAND_COLS` para que features, oclusão,
    rótulos e estatística passem a usar a mesma resolução. Deve ser chamada uma vez no
    início do pipeline, antes dos estágios. Consumidores devem ler estes globais via o
    módulo `bands` (não copiá-los no import) para enxergar a mudança.
    """
    global N_BANDS, BAND_EDGES, BAND_COLS
    N_BANDS = int(n_bands)
    BAND_EDGES = mel_band_edges(N_BANDS, f_min, f_max)
    BAND_COLS = _band_cols(N_BANDS)


def band_index(col: str) -> tuple[int, str]:
    """Extrai (índice 0-based da banda, estatística) de 'band{k}_{mean,std}'."""
    m = _BAND_RE.match(col)
    if not m:
        raise ValueError(f"coluna de banda inválida: {col!r}")
    return int(m.group(1)) - 1, m.group(2)  # band1 -> índice 0


def band_group(col: str) -> str:
    """Grupo de frequência da banda pela sua borda central: 'low'/'mid'/'high'."""
    k, _ = band_index(col)
    center = 0.5 * (BAND_EDGES[k] + BAND_EDGES[k + 1])
    if center < 500.0:
        return "low"
    return "mid" if center < 2000.0 else "high"


def band_label(col: str) -> str:
    """Rótulo descritivo p/ figuras, ex.: '0.02–0.28 kHz·μ', '1.79–2.69 kHz·σ'."""
    k, stat = band_index(col)
    lo, hi = BAND_EDGES[k], BAND_EDGES[k + 1]
    sym = "\u03bc" if stat == "mean" else "\u03c3"  # μ / σ
    return f"{lo / 1000:.2f}\u2013{hi / 1000:.2f} kHz\u00b7{sym}"
