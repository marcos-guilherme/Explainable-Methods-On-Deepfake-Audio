"""DFT-LRP: leva a relevância do tempo para a frequência (camada de inspeção virtual).

Implementa o passo final do método de Vielhaben et al. (2024), "Explainable AI for time
series via Virtual Inspection Layers": dada a relevância por instante do sinal (R_n, vinda
de um método de atribuição como LRP/AttnLRP no domínio do tempo), propaga essa relevância
por uma camada linear inversa (a DFT inversa) para obter a relevância por frequência (R_k).

Ideia: a DFT é linear e invertível, então propagar relevância por ela é a regra LRP-0/ε de
uma camada linear. Aqui derivamos a forma fechada equivalente (Eq. 10-11 do paper) e a
calculamos com FFT, em O(N log N), evitando o somatório O(N^2) da fórmula explícita.

Convenção de sinal preservada: R_k herda o sinal de R_n. Positivo = a frequência empurra a
decisão no sentido do alvo explicado (no nosso uso, o logit de "spoof"): mesma leitura da
oclusão (H2). Este módulo NÃO carrega modelo nem torch; só recebe (x_n, R_n) já calculados.
"""
from __future__ import annotations

import numpy as np


def rfft_frequencies(n_samples: int, sample_rate: int) -> np.ndarray:
    """Frequências (Hz) dos bins do espectro real (rfft) de um sinal de ``n_samples``."""
    return np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)


def time_to_freq_relevance(x_time: np.ndarray, r_time: np.ndarray,
                           eps: float = 1e-9) -> np.ndarray:
    """Relevância por frequência (R_k) a partir da relevância por instante (R_n).

    Propaga R_n pela DFT inversa (camada linear) via regra LRP-0: para um sinal real, a
    relevância do bin k combina as contribuições de Re(y_k) e Im(y_k) do espectro. Devolve
    a relevância "física" já dobrada nos bins internos (1..N/2-1), de modo que a soma sobre
    os bins do rfft conserva a relevância total: ``sum_k R_k == sum_n R_n``.

    Derivação (DFT ortonormal, 1/sqrt(N), como no paper):
        g_n   = R_n / x_n              (0 onde |x_n| <= eps; convenção 0/0 = 0 do paper)
        base_k = (1/N) * Re( rfft(x)_k * conj(rfft(g)_k) )
        R_k    = f_k * base_k,   f_k = 1 em k=0 e k=N/2 (par); 2 nos demais bins.

    Args:
        x_time: sinal no tempo, shape (..., N). É o MESMO sinal que entrou no modelo.
        r_time: relevância por instante, shape (..., N) (ex.: x * grad do backward LRP).
        eps: estabilização de R_n/x_n perto do zero (silêncio/padding do clipe).

    Returns:
        R_freq: relevância por bin de frequência, shape (..., N//2 + 1). Mesmo sinal de R_n.
    """
    x = np.asarray(x_time, dtype=np.float64)
    r = np.asarray(r_time, dtype=np.float64)
    if x.shape != r.shape:
        raise ValueError(f"x_time {x.shape} e r_time {r.shape} devem ter o mesmo shape")
    n = x.shape[-1]
    # g_n = R_n/x_n; onde o sinal é ~0 (padding/silêncio) a contribuição é nula (0/0 := 0).
    g = np.divide(r, x, out=np.zeros_like(r), where=np.abs(x) > eps)
    y = np.fft.rfft(x, axis=-1)
    gk = np.fft.rfft(g, axis=-1)
    base = (y * np.conj(gk)).real / n
    fold = np.full(base.shape[-1], 2.0)
    fold[0] = 1.0                       # DC não tem par conjugado
    if n % 2 == 0:
        fold[-1] = 1.0                 # Nyquist (N par) também não tem par
    return base * fold


def aggregate_to_bands(r_freq: np.ndarray, freqs: np.ndarray,
                       edges: np.ndarray) -> np.ndarray:
    """Soma a relevância por frequência dentro de cada banda ``[edges[i], edges[i+1])``.

    Usa as MESMAS bordas de ``bands.mel_band_edges`` de H1/H2, para o perfil de relevância
    ficar diretamente comparável (banda a banda) com associação e oclusão. Bins fora de
    ``[edges[0], edges[-1]]`` (ex.: DC, ou acima de f_max) ficam de fora, como na oclusão.

    Args:
        r_freq: relevância por bin, shape (..., F).
        freqs: frequência (Hz) de cada bin, shape (F,) (ver ``rfft_frequencies``).
        edges: n_bands+1 bordas de frequência (Hz).

    Returns:
        Relevância por banda, shape (..., n_bands).
    """
    r_freq = np.asarray(r_freq, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    n_bands = len(edges) - 1
    out = np.zeros(r_freq.shape[:-1] + (n_bands,), dtype=np.float64)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        # Última banda inclui a borda superior; as demais são semiabertas [lo, hi).
        mask = (freqs >= lo) & (freqs <= hi if i == n_bands - 1 else freqs < hi)
        out[..., i] = r_freq[..., mask].sum(axis=-1)
    return out


def stdft_lrp(x_time: np.ndarray, r_time: np.ndarray, sample_rate: int,
              win_size: int = 512, hop: int = 256, eps: float = 1e-9
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """STDFT-LRP: leva a relevância do tempo para o plano tempo-frequência (Vielhaben et al.).

    É a versão janelada da camada de inspeção da DFT: em cada janela (Hann, salto ``hop``)
    aplicamos a mesma forma fechada de ``time_to_freq_relevance``. A relevância de cada amostra
    R_n é REPARTIDA entre as janelas que a cobrem, na proporção do peso da janela (WOLA), com
    ``W_n = sum_m w^{(m)}_n``. Assim ``sum_{m,k} R_{m,k} == sum_n R_n`` (nas amostras cobertas),
    então o mapa conserva a relevância total, igual à DFT global.

    Args:
        x_time: sinal no tempo (N,). O MESMO que entrou no modelo.
        r_time: relevância por instante (N,) (ex.: x*grad do backward LRP).
        sample_rate: taxa de amostragem (Hz).
        win_size: tamanho da janela (amostras). Resolução em frequência = sample_rate/win_size.
        hop: salto entre janelas (amostras). ``win_size//2`` dá 50% de sobreposição.
        eps: estabilização de R_n/x_n perto do zero.

    Returns:
        (times, freqs, R_tf, S_tf):
          times  (M,)         centro de cada janela, em segundos;
          freqs  (F,)         frequência de cada bin (Hz), F = win_size//2 + 1;
          R_tf   (M, F)       relevância assinada por (tempo, frequência), mesma convenção de R_n;
          S_tf   (M, F)       magnitude do STDFT janelado (para desenhar o espectrograma de fundo).
    """
    x = np.asarray(x_time, dtype=np.float64).ravel()
    r = np.asarray(r_time, dtype=np.float64).ravel()
    n = x.shape[-1]
    if n < win_size:
        raise ValueError(f"sinal ({n}) menor que a janela ({win_size})")
    # Acolchoa em win//2 para TODA amostra original cair no centro de alguma janela (a Hann zera
    # nas pontas). As amostras acolchoadas têm sinal e relevância nulos, então não afetam a soma.
    pad = win_size // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    rp = np.concatenate([np.zeros(pad), r, np.zeros(pad)])
    npad = xp.shape[-1]
    w = np.hanning(win_size)
    starts = np.arange(0, npad - win_size + 1, hop)
    # WOLA: soma dos pesos de janela por amostra, para repartir R_n sem criar nem destruir.
    wsum = np.zeros(npad, dtype=np.float64)
    for s in starts:
        wsum[s:s + win_size] += w
    freqs = rfft_frequencies(win_size, sample_rate)
    r_tf = np.empty((len(starts), len(freqs)), dtype=np.float64)
    s_tf = np.empty((len(starts), len(freqs)), dtype=np.float64)
    for m, s in enumerate(starts):
        sl = slice(s, s + win_size)
        xw = xp[sl] * w                               # sinal janelado (STDFT)
        alpha = w / np.where(wsum[sl] > 0, wsum[sl], 1.0)   # fração da janela m na amostra
        rf = rp[sl] * alpha                           # relevância repartida para esta janela
        r_tf[m] = time_to_freq_relevance(xw, rf, eps=eps)
        s_tf[m] = np.abs(np.fft.rfft(xw))
    times = (starts + win_size / 2.0 - pad) / sample_rate   # tempo no relógio do sinal original
    return times, freqs, r_tf, s_tf


def conservation_error(x_time: np.ndarray, r_time: np.ndarray,
                       r_freq: np.ndarray) -> np.ndarray:
    """Erro absoluto de conservação da camada DFT: ``|sum_k R_k - sum_n R_n|``.

    Deve ser ~0 (a menos de eps e ponto flutuante) para QUALQUER método de atribuição,
    pois a DFT é linear. É o teste de sanidade do passo espectral, independente do LRP.
    """
    return np.abs(np.asarray(r_freq).sum(axis=-1) - np.asarray(r_time).sum(axis=-1))
