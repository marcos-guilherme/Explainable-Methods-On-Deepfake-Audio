"""Sanidade do filtro por bandas da validação de fidelidade (só a matemática, sem GPU/modelo).

Garante a invertibilidade exata da ida-e-volta na STFT: manter todas as bandas reproduz o sinal
e remover todas zera. É o que sustenta a leitura das curvas (keep/delete) como efeito da máscara,
não do artefato da transformada.
"""
import sys
import csv
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

import faithfulness_bands as fb  # noqa: E402
from faithfulness_bands import (CONDITION_ORDER, band_mask_bins, band_of_bin,  # noqa: E402
                                paired_comparison_rows, process_clip,
                                reconstruct_with_band_mask, rms_match_waveform,
                                select_band_extremes)
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


def test_select_band_extremes_returns_top_and_bottom_without_overlap():
    ranking = np.array([4, 1, 3, 0, 2])
    top, bottom = select_band_extremes(ranking, 2)
    assert top.tolist() == [4, 1]
    assert bottom.tolist() == [0, 2]


def test_rms_match_waveform_matches_original_without_clipping():
    original = np.array([2.0, -2.0])
    perturbed = np.array([0.25, -0.25])
    matched = rms_match_waveform(original, perturbed)
    assert np.sqrt(np.mean(matched ** 2)) == pytest.approx(2.0)
    assert np.max(np.abs(matched)) == pytest.approx(2.0)


def test_rms_match_waveform_has_explicit_zero_and_finite_guards():
    assert np.array_equal(rms_match_waveform(np.zeros(3), np.ones(3)), np.zeros(3))
    assert np.array_equal(rms_match_waveform(np.ones(3), np.zeros(3)), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        rms_match_waveform(np.array([np.nan]), np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        rms_match_waveform(np.ones(1), np.array([np.inf]))


def test_process_clip_emits_deterministic_condition_order_and_rms_matches(monkeypatch):
    edges = np.array([0.0, 1000.0, 2000.0, 4000.0])
    freqs = dft_lrp.rfft_frequencies(8, 8000)
    r_tf = np.array([[3.0, 2.0, 1.0, 0.0, 0.0]])
    x = np.array([0.5, -0.25, 0.75, -0.5, 0.25, 0.1, -0.2, 0.3])
    seen = []

    def fake_score(_model, waves, _device):
        seen.extend(np.asarray(w) for w in waves)
        return np.arange(len(waves), dtype=float)

    monkeypatch.setattr(fb, "score_waves", fake_score)
    out = process_clip(None, r_tf, x, 1.0, freqs, edges, 8, 4, [1], 2,
                       np.random.default_rng(7), "cpu", rms_match=True)
    assert [condition for condition, _k, _p in out] == list(CONDITION_ORDER)
    target_rms = np.sqrt(np.mean(x ** 2))
    for wave in seen:
        wave_rms = np.sqrt(np.mean(wave ** 2))
        if wave_rms > 0:
            assert wave_rms == pytest.approx(target_rms)


def _comparison_input():
    rows = []
    p_orig = {}
    # Comprehensiveness values top/random/bottom are respectively:
    # clip 0: .6/.4/.2; clip 1: .5/.3/.1.
    for idx, values in enumerate(((0.3, 0.5, 0.7), (0.4, 0.6, 0.8))):
        p_orig[idx] = 0.9
        for condition, p_pred in zip(
                ("delete_dftlrp", "delete_random", "delete_dftlrp_bottom"), values):
            rows.append({"index": idx, "true_class": "spoof", "condition": condition,
                         "k": 1, "p_pred": p_pred})
    return rows, p_orig


def test_paired_comparison_rows_include_clip_gaps_and_aggregate_statistics():
    rows, p_orig = _comparison_input()
    out = paired_comparison_rows(rows, [1], ["spoof"], p_orig, seed=11, n_boot=200)
    clip = [r for r in out if r["level"] == "clip"]
    aggregate = [r for r in out if r["level"] == "aggregate"]
    assert [(r["index"], r["comparison"], r["difference"]) for r in clip] == [
        (0, "top_minus_random", pytest.approx(0.2)),
        (0, "top_minus_bottom", pytest.approx(0.4)),
        (0, "random_minus_bottom", pytest.approx(0.2)),
        (1, "top_minus_random", pytest.approx(0.2)),
        (1, "top_minus_bottom", pytest.approx(0.4)),
        (1, "random_minus_bottom", pytest.approx(0.2)),
    ]
    assert [r["comparison"] for r in aggregate] == [
        "top_minus_random", "top_minus_bottom", "random_minus_bottom"]
    assert all(r["n_pairs"] == 2 for r in aggregate)
    assert all(r["status"] == "ok" for r in aggregate)
    assert all(np.isfinite(r["wilcoxon_p"]) for r in aggregate)


def test_paired_comparison_rows_report_insufficient_and_degenerate_statuses():
    one_clip = [
        {"index": 0, "true_class": "spoof", "condition": cond, "k": 1, "p_pred": 0.5}
        for cond in ("delete_dftlrp", "delete_random", "delete_dftlrp_bottom")
    ]
    insufficient = paired_comparison_rows(one_clip, [1], ["spoof"], {0: 0.9})
    aggregate = [r for r in insufficient if r["level"] == "aggregate"]
    assert all(r["status"] == "insufficient_pairs" for r in aggregate)
    assert all(np.isnan(r["wilcoxon_p"]) for r in aggregate)

    rows = one_clip + [
        {"index": 1, "true_class": "spoof", "condition": cond, "k": 1, "p_pred": 0.5}
        for cond in ("delete_dftlrp", "delete_random", "delete_dftlrp_bottom")
    ]
    degenerate = paired_comparison_rows(rows, [1], ["spoof"], {0: 0.9, 1: 0.9})
    aggregate = [r for r in degenerate if r["level"] == "aggregate"]
    assert all(r["status"] == "degenerate_all_zero" for r in aggregate)
    assert all(np.isnan(r["wilcoxon_p"]) for r in aggregate)


def test_aggregate_curves_and_aopc_include_bottom_in_deterministic_order(tmp_path):
    rows = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        rows.append({"index": 0, "true_class": "spoof", "condition": condition,
                     "k": 1, "p_pred": 0.1 * (condition_index + 1)})

    agg = fb._aggregate(rows, [1], ["spoof"])
    curves_path = tmp_path / "curves.csv"
    fb._write_curves_csv(curves_path, agg, [1], ["spoof"])
    with curves_path.open(newline="") as f:
        curves = list(csv.DictReader(f))
    assert [row["condition"] for row in curves] == list(CONDITION_ORDER)

    aopc_path = tmp_path / "aopc.csv"
    fb._write_aopc_csv(aopc_path, rows, [1], ["spoof"], {0: 0.9})
    with aopc_path.open(newline="") as f:
        aopc = list(csv.DictReader(f))
    assert [row["condition"] for row in aopc] == [
        "keep_dftlrp", "keep_dftlrp_bottom", "keep_random",
        "delete_dftlrp", "delete_dftlrp_bottom", "delete_random",
    ]
