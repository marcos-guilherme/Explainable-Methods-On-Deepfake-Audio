"""Testes da grade de bandas reconfigurável (fonte única para H1 e H2)."""
import numpy as np

from brspeech_xai import bands


def teardown_function(_):
    bands.configure_bands(8)  # restaura a grade de referência após cada teste


def test_configure_bands_updates_globals():
    bands.configure_bands(16)
    assert bands.N_BANDS == 16
    assert len(bands.BAND_EDGES) == 17            # n_bands + 1
    assert len(bands.BAND_COLS) == 32             # 2 * n_bands (μ e σ)
    assert bands.BAND_COLS[:2] == ["band1_mean", "band2_mean"]


def test_edges_monotonic_and_span():
    bands.configure_bands(24, f_min=20.0, f_max=7900.0)
    edges = bands.BAND_EDGES
    assert len(edges) == 25
    assert np.all(np.diff(edges) > 0)             # monotônicas crescentes
    assert abs(edges[0] - 20.0) < 1e-6 and abs(edges[-1] - 7900.0) < 1e-6


def test_labels_follow_grid():
    bands.configure_bands(16)
    # a última banda existe e o rótulo reflete a nova borda superior
    assert bands.band_index("band16_std") == (15, "std")
    assert "kHz" in bands.band_label("band16_mean")


def test_features_and_occlusion_share_edges():
    """H1 (features) e H2 (oclusão) devem usar as MESMAS bordas na grade configurada.

    A oclusão recomputa as bordas via mel_band_edges(cfg.bands.*); aqui verificamos que,
    para o mesmo N, elas coincidem com a grade global usada pelas features.
    """
    bands.configure_bands(12)
    occ_edges = bands.mel_band_edges(bands.N_BANDS, 20.0, 7900.0)
    assert np.allclose(occ_edges, bands.BAND_EDGES)
