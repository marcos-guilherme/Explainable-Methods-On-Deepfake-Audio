"""Testes da matemática do DFT-LRP (numpy puro, sem carregar modelo nem torch)."""
import numpy as np
import pytest

from brspeech_xai.dft_lrp import (aggregate_to_bands, conservation_error,
                                  rfft_frequencies, stdft_lrp, time_to_freq_relevance)


def _inverse_dft_relevance_reference(x: np.ndarray, r: np.ndarray,
                                     eps: float = 1e-12) -> np.ndarray:
    """Referência independente: constrói o operador da DFT inversa (a partir das DOFs do
    rfft) e aplica a regra LRP-0 genérica (R_j = v_j * sum_n A[n,j] * R_n/x_n).

    Não reutiliza a forma fechada de ``time_to_freq_relevance``; parte do operador linear
    explícito x = A v, o que valida a derivação de forma cega."""
    n = len(x)
    g = np.where(np.abs(x) > eps, r / x, 0.0)
    idx = np.arange(n)
    y = np.fft.rfft(x, norm="ortho")            # DOFs reais/imaginárias por bin
    f = n // 2 + 1
    r_k = np.zeros(f)
    for k in range(f):
        ck = 1.0 if (k == 0 or (n % 2 == 0 and k == f - 1)) else 2.0
        cos = (ck / np.sqrt(n)) * np.cos(2 * np.pi * k * idx / n)   # coluna Re(y_k)
        sin = -(ck / np.sqrt(n)) * np.sin(2 * np.pi * k * idx / n)  # coluna Im(y_k)
        r_re = y.real[k] * np.sum(cos * g)      # LRP-0: v_j * sum_n A[n,j] g_n
        r_im = y.imag[k] * np.sum(sin * g)
        r_k[k] = r_re + r_im
    return r_k


def _reconstructs_signal(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Confere que o operador da referência de fato reconstrói x (x = A v)."""
    n = len(x)
    idx = np.arange(n)
    y = np.fft.rfft(x, norm="ortho")
    f = n // 2 + 1
    rec = np.zeros(n)
    for k in range(f):
        ck = 1.0 if (k == 0 or (n % 2 == 0 and k == f - 1)) else 2.0
        rec += (ck / np.sqrt(n)) * (y.real[k] * np.cos(2 * np.pi * k * idx / n)
                                    - y.imag[k] * np.sin(2 * np.pi * k * idx / n))
    return rec


@pytest.mark.parametrize("n", [8, 9, 16, 33])
def test_matches_independent_lrp0_reference(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    r = rng.standard_normal(n)
    # sanidade do operador de referência: reconstrói o sinal.
    assert np.allclose(_reconstructs_signal(x), x, atol=1e-9)
    got = time_to_freq_relevance(x, r)
    ref = _inverse_dft_relevance_reference(x, r)
    assert got.shape == (n // 2 + 1,)
    assert np.allclose(got, ref, atol=1e-9)


@pytest.mark.parametrize("n", [8, 9, 64, 65])
def test_total_relevance_is_conserved(n):
    """Propriedade central do paper: sum_k R_k == sum_n R_n (a DFT é linear)."""
    rng = np.random.default_rng(1000 + n)
    x = rng.standard_normal(n)
    r = rng.standard_normal(n)
    r_freq = time_to_freq_relevance(x, r)
    assert conservation_error(x, r, r_freq) < 1e-8


def test_batch_matches_loop():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 32))
    r = rng.standard_normal((4, 32))
    batched = time_to_freq_relevance(x, r)
    assert batched.shape == (4, 17)
    for i in range(4):
        assert np.allclose(batched[i], time_to_freq_relevance(x[i], r[i]), atol=1e-10)


def test_zero_signal_gives_zero_relevance():
    """Onde o sinal é ~0 (silêncio/padding) a razão R_n/x_n é tratada como 0 (0/0 := 0)."""
    x = np.zeros(16)
    r = np.ones(16)
    r_freq = time_to_freq_relevance(x, r)
    assert np.allclose(r_freq, 0.0)


def test_relevance_localizes_on_the_signal_frequency():
    """Sinal senoidal puro: a relevância deve concentrar no bin da sua frequência."""
    n, sr, f0 = 256, 256, 16          # 16 Hz cai exatamente no bin k=16
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * f0 * t)
    # Gradiente proporcional ao sinal (R_n = x_n^2 => g_n = x_n): imita um modelo que
    # responde à energia da frequência, dando R_k proporcional a |X_k|^2 (localizado em f0).
    r = x * x
    r_freq = time_to_freq_relevance(x, r)
    freqs = rfft_frequencies(n, sr)
    assert freqs[np.argmax(np.abs(r_freq))] == pytest.approx(f0, abs=1.0)


def test_aggregate_to_bands_partitions_relevance():
    freqs = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    r_freq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    edges = np.array([100.0, 250.0, 400.0])          # 2 bandas: [100,250), [250,400]
    bands = aggregate_to_bands(r_freq, freqs, edges)
    assert bands.shape == (2,)
    assert bands[0] == pytest.approx(2.0 + 3.0)      # 100 e 200 Hz
    assert bands[1] == pytest.approx(4.0 + 5.0)      # 300 e 400 Hz (borda superior inclusa)


def test_aggregate_to_bands_batch_shape():
    freqs = rfft_frequencies(128, 16000)
    r_freq = np.random.default_rng(0).standard_normal((5, freqs.size))
    edges = np.linspace(20.0, 7900.0, 9)
    bands = aggregate_to_bands(r_freq, freqs, edges)
    assert bands.shape == (5, 8)


def test_stdft_lrp_conserves_over_covered_samples():
    """STDFT-LRP: sum_{m,k} R_{m,k} == sum_n R_n nas amostras cobertas por alguma janela.

    Com N = win + k*hop e hop=win/2, todas as amostras ficam cobertas, então a conservação da
    DFT global vale também no plano tempo-frequência (a repartição WOLA não cria nem destrói)."""
    rng = np.random.default_rng(42)
    win, hop = 64, 32
    n = win + 10 * hop                                # cobre todas as amostras
    x = rng.standard_normal(n)
    r = rng.standard_normal(n)
    _, _, r_tf, _ = stdft_lrp(x, r, sample_rate=16000, win_size=win, hop=hop)
    assert abs(r_tf.sum() - r.sum()) < 1e-7


def test_stdft_lrp_shapes():
    win, hop, sr = 128, 64, 16000
    n = win + 5 * hop
    x = np.random.default_rng(0).standard_normal(n)
    r = np.random.default_rng(1).standard_normal(n)
    times, freqs, r_tf, s_tf = stdft_lrp(x, r, sr, win_size=win, hop=hop)
    # Com acolchoamento de win//2 nas duas pontas, o nº de janelas é (n + win - win)//hop + 1.
    n_frames = (n + 2 * (win // 2) - win) // hop + 1
    assert freqs.shape == (win // 2 + 1,)
    assert times.shape == (n_frames,)
    assert r_tf.shape == (n_frames, win // 2 + 1)
    assert s_tf.shape == r_tf.shape
