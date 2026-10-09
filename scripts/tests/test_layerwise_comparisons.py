"""TDD contract for the deterministic cross-model/-language comparison builders.

These builders are pure functions over the identity-labelled report tables
(``cell_predictions``, ``sample_relevance`` and ``band_edges``). They never read
from disk, load no models, and must be byte-for-byte deterministic regardless of
the order of their input rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score

from brspeech_xai.layerwise_report import (
    BOOTSTRAP_CI_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    build_diagonal_vs_offdiagonal,
    build_encoder_agreement,
    build_language_shift,
    build_spectral_divergence,
)

_JS_BASE = 2.0


# ---------------------------------------------------------------------------
# Fixture helpers (tiny, hand-built DataFrames)
# ---------------------------------------------------------------------------
def _prob(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return (arr / arr.sum()).tolist()


def _pred_row(
    model: str,
    source: str,
    target: str,
    layer: int,
    sample_id: str,
    y_true: int,
    score: float,
    threshold: float = 0.5,
) -> dict:
    return {
        "model": model,
        "source": source,
        "target": target,
        "language": target,
        "layer": layer,
        "sample_id": sample_id,
        "y_true": y_true,
        "score": score,
        "prediction": int(score >= threshold),
        "threshold": threshold,
    }


def _rel_row(
    model: str,
    source: str,
    target: str,
    layer: int,
    sample_id: str,
    y_true: int,
    band_abs_normalized: list[float],
    band_signed: list[float] | None = None,
    score: float = 0.5,
) -> dict:
    return {
        "model": model,
        "source": source,
        "target": target,
        "language": target,
        "layer": layer,
        "sample_id": sample_id,
        "y_true": y_true,
        "prediction": int(score >= 0.5),
        "score": score,
        "band_signed": band_signed or [0.0] * len(band_abs_normalized),
        "band_abs_normalized": list(band_abs_normalized),
    }


def _shuffle(frame: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


_ENCODER_COLUMNS = {
    "model_a",
    "model_b",
    "source",
    "target",
    "layer",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
}
_LANGUAGE_SHIFT_COLUMNS = {
    "model",
    "source",
    "target_a",
    "target_b",
    "layer",
    "group",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n_a",
    "n_b",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
}
_DIAGONAL_COLUMNS = {
    "model",
    "source",
    "target",
    "layer",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
}
_SPECTRAL_COLUMNS = {
    "model",
    "language_a",
    "language_b",
    "layer",
    "group",
    "metric",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "n_a",
    "n_b",
    "status",
    "n_resamples",
    "seed",
    "ci_level",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _two_model_predictions() -> pd.DataFrame:
    cohort = [("s1", 0, 0.2, 0.3), ("s2", 0, 0.4, 0.35), ("s3", 1, 0.6, 0.55), ("s4", 1, 0.8, 0.9)]
    rows = []
    for sid, y, a, b in cohort:
        rows.append(_pred_row("hubert", "eng", "eng", 1, sid, y, a))
        rows.append(_pred_row("wavlm", "eng", "eng", 1, sid, y, b))
    return pd.DataFrame(rows)


def _two_model_relevance() -> pd.DataFrame:
    vectors = {
        ("hubert", "s1"): _prob([0.6, 0.3, 0.1]),
        ("hubert", "s2"): _prob([0.2, 0.5, 0.3]),
        ("hubert", "s3"): _prob([0.1, 0.2, 0.7]),
        ("hubert", "s4"): _prob([0.3, 0.4, 0.3]),
        ("wavlm", "s1"): _prob([0.5, 0.4, 0.1]),
        ("wavlm", "s2"): _prob([0.25, 0.45, 0.3]),
        ("wavlm", "s3"): _prob([0.15, 0.25, 0.6]),
        ("wavlm", "s4"): _prob([0.2, 0.5, 0.3]),
    }
    labels = {"s1": 0, "s2": 0, "s3": 1, "s4": 1}
    rows = []
    for (model, sid), vec in vectors.items():
        rows.append(_rel_row(model, "eng", "eng", 1, sid, labels[sid], vec))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------
def test_bootstrap_contract_constants_are_fixed():
    assert BOOTSTRAP_RESAMPLES == 2000
    assert BOOTSTRAP_SEED == 42
    assert BOOTSTRAP_CI_LEVEL == 0.95


# ---------------------------------------------------------------------------
# 1) encoder_agreement
# ---------------------------------------------------------------------------
def test_encoder_agreement_exact_full_prediction_formulas():
    predictions = _two_model_predictions()
    relevance = _two_model_relevance()
    table = build_encoder_agreement(predictions, relevance)
    assert set(table.columns) == _ENCODER_COLUMNS
    assert (table["model_a"] == "hubert").all()
    assert (table["model_b"] == "wavlm").all()
    assert (table["n_resamples"] == BOOTSTRAP_RESAMPLES).all()
    assert (table["seed"] == BOOTSTRAP_SEED).all()
    assert (table["ci_level"] == BOOTSTRAP_CI_LEVEL).all()

    row = table.set_index("metric")

    a = predictions[predictions["model"] == "hubert"].sort_values("sample_id")
    b = predictions[predictions["model"] == "wavlm"].sort_values("sample_id")
    expected_spearman = spearmanr(a["score"], b["score"]).correlation
    expected_agree = float(np.mean(a["prediction"].to_numpy() == b["prediction"].to_numpy()))
    expected_kappa = cohen_kappa_score(a["prediction"], b["prediction"])

    assert row.loc["score_spearman", "estimate"] == pytest.approx(expected_spearman)
    assert row.loc["prediction_agreement", "estimate"] == pytest.approx(expected_agree)
    assert row.loc["cohen_kappa", "estimate"] == pytest.approx(expected_kappa)
    assert int(row.loc["score_spearman", "n"]) == 4


def test_encoder_agreement_exact_relevance_formulas():
    predictions = _two_model_predictions()
    relevance = _two_model_relevance()
    table = build_encoder_agreement(predictions, relevance).set_index("metric")

    a = relevance[relevance["model"] == "hubert"].sort_values("sample_id")
    b = relevance[relevance["model"] == "wavlm"].sort_values("sample_id")
    cosines, js = [], []
    for va, vb in zip(a["band_abs_normalized"], b["band_abs_normalized"]):
        va = np.asarray(va)
        vb = np.asarray(vb)
        cosines.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
        js.append(float(jensenshannon(va, vb, base=_JS_BASE)))

    assert table.loc["relevance_cosine_similarity", "estimate"] == pytest.approx(np.mean(cosines))
    assert table.loc["relevance_jensen_shannon_distance", "estimate"] == pytest.approx(np.mean(js))
    assert int(table.loc["relevance_cosine_similarity", "n"]) == 4
    assert table.loc["relevance_jensen_shannon_distance", "unit"].lower().startswith("bit")


def test_encoder_agreement_requires_identical_cohorts():
    predictions = _two_model_predictions()
    broken = predictions.copy()
    broken.loc[(broken["model"] == "wavlm") & (broken["sample_id"] == "s4"), "sample_id"] = "sX"
    with pytest.raises(ValueError, match="identical|cohort|sample"):
        build_encoder_agreement(broken, _two_model_relevance())


def test_encoder_agreement_requires_identical_labels():
    predictions = _two_model_predictions()
    broken = predictions.copy()
    broken.loc[(broken["model"] == "wavlm") & (broken["sample_id"] == "s4"), "y_true"] = 0
    with pytest.raises(ValueError, match="label|y_true"):
        build_encoder_agreement(broken, _two_model_relevance())


def test_encoder_agreement_marks_constant_score_series_explicitly():
    predictions = _two_model_predictions()
    predictions.loc[predictions["model"] == "wavlm", "score"] = 0.7  # constant series
    table = build_encoder_agreement(predictions, _two_model_relevance()).set_index("metric")
    assert table.loc["score_spearman", "status"] == "constant_series"
    assert not np.isfinite(table.loc["score_spearman", "estimate"])
    # Other metrics must remain finite.
    assert np.isfinite(table.loc["prediction_agreement", "estimate"])


def test_encoder_agreement_is_empty_for_single_model():
    predictions = _two_model_predictions()
    single = predictions[predictions["model"] == "hubert"]
    relevance = _two_model_relevance()
    single_rel = relevance[relevance["model"] == "hubert"]
    table = build_encoder_agreement(single, single_rel)
    assert table.empty
    assert set(table.columns) == _ENCODER_COLUMNS


def test_encoder_agreement_ci_contains_estimate_without_clamping():
    table = build_encoder_agreement(_two_model_predictions(), _two_model_relevance())
    finite = table[np.isfinite(table["estimate"])]
    for _, row in finite.iterrows():
        assert row["ci_low"] <= row["estimate"] + 1e-9
        assert row["estimate"] - 1e-9 <= row["ci_high"]
        assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])


def test_encoder_agreement_is_deterministic_under_input_shuffle():
    predictions = _two_model_predictions()
    relevance = _two_model_relevance()
    base = build_encoder_agreement(predictions, relevance)
    shuffled = build_encoder_agreement(_shuffle(predictions, 1), _shuffle(relevance, 2))
    pd.testing.assert_frame_equal(base, shuffled)


# ---------------------------------------------------------------------------
# 2) language_shift
# ---------------------------------------------------------------------------
def _single_model_cross_language_relevance() -> pd.DataFrame:
    rows = []
    data = {
        ("eng", "por"): {
            "p1": (0, [0.7, 0.2, 0.1]),
            "p2": (0, [0.6, 0.3, 0.1]),
            "p3": (1, [0.1, 0.3, 0.6]),
            "p4": (1, [0.2, 0.3, 0.5]),
        },
        ("eng", "zho"): {
            "z1": (0, [0.2, 0.5, 0.3]),
            "z2": (0, [0.3, 0.4, 0.3]),
            "z3": (1, [0.4, 0.4, 0.2]),
            "z4": (1, [0.5, 0.3, 0.2]),
            "z5": (1, [0.45, 0.35, 0.2]),
            "z6": (0, [0.25, 0.45, 0.3]),
        },
    }
    for (src, tgt), samples in data.items():
        for sid, (y, vec) in samples.items():
            rows.append(_rel_row("hubert", src, tgt, 1, sid, y, _prob(vec)))
    return pd.DataFrame(rows)


def test_language_shift_exact_point_estimate_and_independent_counts():
    relevance = _single_model_cross_language_relevance()
    table = build_language_shift(relevance)
    assert set(table.columns) == _LANGUAGE_SHIFT_COLUMNS
    assert set(table["group"]) == {"all", "real", "synthetic"}
    assert (table["target_a"] < table["target_b"]).all()  # unordered -> sorted
    assert (table["metric"] == "jensen_shannon_distance").all()

    all_row = table[(table["group"] == "all")].iloc[0]
    a = relevance[(relevance["target"] == all_row["target_a"])]
    b = relevance[(relevance["target"] == all_row["target_b"])]
    mean_a = np.mean(np.stack(a["band_abs_normalized"].map(np.asarray)), axis=0)
    mean_b = np.mean(np.stack(b["band_abs_normalized"].map(np.asarray)), axis=0)
    expected = float(jensenshannon(mean_a, mean_b, base=_JS_BASE))
    assert all_row["estimate"] == pytest.approx(expected)
    # Independent stratified resampling tolerates unequal cohort sizes.
    assert int(all_row["n_a"]) == 4
    assert int(all_row["n_b"]) == 6


def test_language_shift_class_groups_use_only_their_class():
    relevance = _single_model_cross_language_relevance()
    table = build_language_shift(relevance)
    real = table[table["group"] == "real"].iloc[0]
    assert int(real["n_a"]) == 2  # por has 2 real
    assert int(real["n_b"]) == 3  # zho has 3 real
    synth = table[table["group"] == "synthetic"].iloc[0]
    assert int(synth["n_a"]) == 2
    assert int(synth["n_b"]) == 3


def test_language_shift_is_empty_for_single_language_target():
    relevance = _single_model_cross_language_relevance()
    one_target = relevance[relevance["target"] == "por"]
    table = build_language_shift(one_target)
    assert table.empty
    assert set(table.columns) == _LANGUAGE_SHIFT_COLUMNS


def test_language_shift_is_deterministic_under_input_shuffle():
    relevance = _single_model_cross_language_relevance()
    base = build_language_shift(relevance)
    shuffled = build_language_shift(_shuffle(relevance, 3))
    pd.testing.assert_frame_equal(base, shuffled)


# ---------------------------------------------------------------------------
# 3) diagonal_vs_offdiagonal
# ---------------------------------------------------------------------------
def _diagonal_offdiagonal_predictions() -> pd.DataFrame:
    diag = [("a", 0, 0.1), ("b", 0, 0.2), ("c", 1, 0.8), ("d", 1, 0.9)]
    off = [("a", 0, 0.6), ("b", 0, 0.2), ("c", 1, 0.4), ("d", 1, 0.9)]
    rows = []
    for sid, y, s in diag:
        rows.append(_pred_row("hubert", "eng", "eng", 1, sid, y, s))
    for sid, y, s in off:
        rows.append(_pred_row("hubert", "por", "eng", 1, sid, y, s))
    return pd.DataFrame(rows)


def test_diagonal_vs_offdiagonal_exact_delta():
    predictions = _diagonal_offdiagonal_predictions()
    table = build_diagonal_vs_offdiagonal(predictions)
    assert set(table.columns) == _DIAGONAL_COLUMNS
    assert set(table["metric"]) == {"delta_roc_auc", "delta_mcc"}

    diag = predictions[(predictions["source"] == "eng")].sort_values("sample_id")
    off = predictions[(predictions["source"] == "por")].sort_values("sample_id")
    y = diag["y_true"].to_numpy()
    expected_auc = roc_auc_score(y, off["score"]) - roc_auc_score(y, diag["score"])
    expected_mcc = matthews_corrcoef(y, off["prediction"]) - matthews_corrcoef(
        y, diag["prediction"]
    )

    row = table.set_index("metric")
    assert row.loc["delta_roc_auc", "estimate"] == pytest.approx(expected_auc)
    assert row.loc["delta_mcc", "estimate"] == pytest.approx(expected_mcc)
    assert int(row.loc["delta_roc_auc", "n"]) == 4
    assert (table["source"] == "por").all()
    assert (table["target"] == "eng").all()


def test_diagonal_vs_offdiagonal_fails_when_ids_differ():
    predictions = _diagonal_offdiagonal_predictions()
    predictions.loc[
        (predictions["source"] == "por") & (predictions["sample_id"] == "d"), "sample_id"
    ] = "z"
    with pytest.raises(ValueError, match="ID|identical|cohort|sample"):
        build_diagonal_vs_offdiagonal(predictions)


def test_diagonal_vs_offdiagonal_is_empty_without_offdiagonal_cells():
    predictions = _diagonal_offdiagonal_predictions()
    diag_only = predictions[predictions["source"] == "eng"]
    table = build_diagonal_vs_offdiagonal(diag_only)
    assert table.empty
    assert set(table.columns) == _DIAGONAL_COLUMNS


def test_diagonal_vs_offdiagonal_is_deterministic_under_input_shuffle():
    predictions = _diagonal_offdiagonal_predictions()
    base = build_diagonal_vs_offdiagonal(predictions)
    shuffled = build_diagonal_vs_offdiagonal(_shuffle(predictions, 5))
    pd.testing.assert_frame_equal(base, shuffled)


# ---------------------------------------------------------------------------
# 4) spectral_divergence
# ---------------------------------------------------------------------------
def _diagonal_relevance_two_languages() -> pd.DataFrame:
    rows = []
    data = {
        "eng": {
            "e1": (0, [0.7, 0.2, 0.1]),
            "e2": (0, [0.6, 0.3, 0.1]),
            "e3": (1, [0.1, 0.3, 0.6]),
            "e4": (1, [0.2, 0.2, 0.6]),
        },
        "por": {
            "p1": (0, [0.3, 0.4, 0.3]),
            "p2": (0, [0.25, 0.45, 0.3]),
            "p3": (1, [0.5, 0.3, 0.2]),
            "p4": (1, [0.45, 0.35, 0.2]),
        },
    }
    for lang, samples in data.items():
        for sid, (y, vec) in samples.items():
            rows.append(_rel_row("hubert", lang, lang, 1, sid, y, _prob(vec)))
    return pd.DataFrame(rows)


def _band_edges_two_languages() -> pd.DataFrame:
    rows = []
    centers = {"eng": [100.0, 200.0, 300.0], "por": [150.0, 250.0, 350.0]}
    for lang, cs in centers.items():
        for band, center in enumerate(cs, start=1):
            rows.append(
                {
                    "model": "hubert",
                    "language": lang,
                    "band": band,
                    "low_hz": center - 50.0,
                    "high_hz": center + 50.0,
                    "center_hz": center,
                    "label": f"{center:.0f} Hz",
                }
            )
    return pd.DataFrame(rows)


def test_spectral_divergence_exact_wasserstein_in_hz():
    relevance = _diagonal_relevance_two_languages()
    band_edges = _band_edges_two_languages()
    table = build_spectral_divergence(relevance, band_edges)
    assert set(table.columns) == _SPECTRAL_COLUMNS
    assert set(table["group"]) == {"all", "real", "synthetic"}
    assert (table["unit"] == "Hz").all()
    assert (table["metric"] == "wasserstein_distance").all()
    assert (table["language_a"] < table["language_b"]).all()

    all_row = table[table["group"] == "all"].iloc[0]
    a = relevance[relevance["target"] == "eng"]
    b = relevance[relevance["target"] == "por"]
    wa = np.mean(np.stack(a["band_abs_normalized"].map(np.asarray)), axis=0)
    wb = np.mean(np.stack(b["band_abs_normalized"].map(np.asarray)), axis=0)
    expected = wasserstein_distance(
        [100.0, 200.0, 300.0], [150.0, 250.0, 350.0], wa, wb
    )
    assert all_row["estimate"] == pytest.approx(expected)
    assert int(all_row["n_a"]) == 4
    assert int(all_row["n_b"]) == 4


def test_spectral_divergence_is_empty_for_single_language():
    relevance = _diagonal_relevance_two_languages()
    one = relevance[relevance["target"] == "eng"]
    table = build_spectral_divergence(one, _band_edges_two_languages())
    assert table.empty
    assert set(table.columns) == _SPECTRAL_COLUMNS


def test_spectral_divergence_ignores_offdiagonal_cells():
    relevance = _diagonal_relevance_two_languages()
    extra = _rel_row("hubert", "eng", "por", 1, "x1", 0, _prob([0.9, 0.05, 0.05]))
    relevance = pd.concat([relevance, pd.DataFrame([extra])], ignore_index=True)
    table = build_spectral_divergence(relevance, _band_edges_two_languages())
    # The offdiagonal eng->por sample must not change n_a/n_b of the diagonal rows.
    all_row = table[table["group"] == "all"].iloc[0]
    assert int(all_row["n_a"]) == 4
    assert int(all_row["n_b"]) == 4


def test_spectral_divergence_is_deterministic_under_input_shuffle():
    relevance = _diagonal_relevance_two_languages()
    band_edges = _band_edges_two_languages()
    base = build_spectral_divergence(relevance, band_edges)
    shuffled = build_spectral_divergence(_shuffle(relevance, 9), _shuffle(band_edges, 11))
    pd.testing.assert_frame_equal(base, shuffled)
