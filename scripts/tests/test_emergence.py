"""TDD contract for bootstrap confidence intervals and emergence indices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brspeech_xai.emergence import (
    bootstrap_auc_ci,
    consolidation_layer,
    discriminative_onset,
    summarize_emergence,
)


def test_onset_requires_two_consecutive_layers_above_chance():
    lower = np.array([0.48, 0.51, 0.49, 0.54, 0.56])

    assert discriminative_onset(lower, consecutive=2) == 4


def test_onset_uses_explicit_layer_indices_and_strict_comparison():
    lower = np.array([0.50, 0.53, 0.54])

    assert (
        discriminative_onset(
            lower,
            consecutive=2,
            layer_indices=(2, 4, 8),
        )
        == 4
    )


def test_onset_returns_none_without_a_complete_run():
    assert discriminative_onset([0.51, 0.49, 0.52], consecutive=2) is None


def test_consolidation_is_first_sustained_95_percent_layer():
    auc = np.array([0.55, 0.70, 0.88, 0.90, 0.91])

    # target = 0.5 + 0.95 * (0.91 - 0.5) = 0.8895
    assert consolidation_layer(auc, fraction=0.95) == 4


def test_consolidation_does_not_assume_monotonicity():
    auc = np.array([0.90, 0.84, 0.91])

    assert consolidation_layer(auc, fraction=0.95) == 3


def test_consolidation_returns_none_when_final_auc_is_not_above_chance():
    assert consolidation_layer([0.70, 0.50]) is None
    assert consolidation_layer([0.70, 0.49]) is None


def test_onset_rejects_implicit_domain_beyond_layer_12():
    with pytest.raises(ValueError, match="1 through 12"):
        discriminative_onset(np.full(13, 0.6))


def test_consolidation_rejects_implicit_domain_beyond_layer_12():
    with pytest.raises(ValueError, match="1 through 12"):
        consolidation_layer(np.full(13, 0.6))


def test_layer_indices_reject_float_values():
    with pytest.raises(ValueError, match="integers"):
        discriminative_onset([0.6, 0.7], layer_indices=(1.0, 2.0))
    with pytest.raises(ValueError, match="integers"):
        consolidation_layer([0.6, 0.7], layer_indices=(1.0, 2.0))


@pytest.mark.parametrize(
    ("function", "values", "kwargs", "message"),
    [
        (discriminative_onset, [0.6, np.nan], {}, "finite"),
        (discriminative_onset, [0.6], {"consecutive": 0}, "consecutive"),
        (
            discriminative_onset,
            [0.6, 0.7],
            {"layer_indices": (1, 1)},
            "strictly increasing",
        ),
        (
            discriminative_onset,
            [0.6, 0.7],
            {"layer_indices": (1,)},
            "length",
        ),
        (
            discriminative_onset,
            [0.6, 0.7],
            {"layer_indices": (0, 1)},
            "1 through 12",
        ),
        (
            consolidation_layer,
            [0.6, 0.7],
            {"layer_indices": (1, 13)},
            "1 through 12",
        ),
        (consolidation_layer, [], {}, "non-empty"),
        (consolidation_layer, [0.6, np.inf], {}, "finite"),
        (consolidation_layer, [0.6], {"fraction": 0}, "fraction"),
        (consolidation_layer, [0.6], {"fraction": 1.1}, "fraction"),
    ],
)
def test_index_functions_reject_invalid_inputs(function, values, kwargs, message):
    with pytest.raises(ValueError, match=message):
        function(values, **kwargs)


def test_bootstrap_auc_is_deterministic_for_perfect_imbalanced_data():
    scores = np.array([0.05, 0.10, 0.15, 0.20, 0.8, 0.9])
    labels = np.array([0, 0, 0, 0, 1, 1])

    first = bootstrap_auc_ci(scores, labels, n_bootstrap=40, seed=7)
    second = bootstrap_auc_ci(scores, labels, n_bootstrap=40, seed=7)

    assert first == second
    assert first.mean == pytest.approx(1.0)
    assert first.lower == pytest.approx(1.0)
    assert first.upper == pytest.approx(1.0)
    assert first.n_requested == 40
    assert first.n_effective == 40
    assert first.confidence == pytest.approx(0.95)


def test_bootstrap_auc_handles_inverted_data():
    result = bootstrap_auc_ci(
        scores=[0.9, 0.8, 0.2, 0.1],
        labels=[0, 0, 1, 1],
        n_bootstrap=25,
        seed=3,
    )

    assert result.mean == pytest.approx(0.0)
    assert result.lower == pytest.approx(0.0)
    assert result.upper == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("scores", "labels", "kwargs", "message"),
    [
        ([0.1, 0.2], [0, 0], {}, "both classes"),
        ([0.1, np.nan], [0, 1], {}, "finite"),
        ([0.1, np.inf], [0, 1], {}, "finite"),
        ([0.1], [0, 1], {}, "same length"),
        ([[0.1], [0.2]], [0, 1], {}, "one-dimensional"),
        ([0.1, 0.2], [0, 2], {}, "binary"),
        ([0.1, 0.2], [0, 1], {"n_bootstrap": 0}, "n_bootstrap"),
        ([0.1, 0.2], [0, 1], {"confidence": 1.0}, "confidence"),
    ],
)
def test_bootstrap_auc_rejects_invalid_inputs(scores, labels, kwargs, message):
    options = {"n_bootstrap": 10, "seed": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        bootstrap_auc_ci(scores, labels, **options)


def _precomputed_rows() -> pd.DataFrame:
    rows = []
    aucs = [0.55, 0.70, 0.88, 0.90, 0.91]
    lowers = [0.48, 0.51, 0.49, 0.54, 0.56]
    for source, target in (("eng", "eng"), ("eng", "por")):
        for layer, auc, lower in zip((1, 2, 3, 4, 5), aucs, lowers):
            rows.append(
                {
                    "profile": "hubert_base",
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "auc": auc,
                    "auc_ci_lower": lower,
                    "auc_ci_upper": min(1.0, auc + 0.05),
                    "corpus_shift": source != target,
                }
            )
    return pd.DataFrame(rows)


def test_summary_groups_cells_sorts_layers_and_uses_precomputed_intervals():
    performance = _precomputed_rows().sample(frac=1, random_state=9)

    summary = summarize_emergence(
        performance,
        layer_indices=(1, 2, 3, 4, 5),
        n_bootstrap=200,
        confidence=0.95,
    )

    assert summary[["source", "target"]].to_records(index=False).tolist() == [
        ("eng", "eng"),
        ("eng", "por"),
    ]
    assert summary["onset"].tolist() == [4, 4]
    assert summary["consolidation"].tolist() == [4, 4]
    assert summary["final_auc"].tolist() == pytest.approx([0.91, 0.91])
    assert summary["corpus_shift"].tolist() == [False, True]
    assert summary["bootstrap_confidence"].tolist() == [0.95, 0.95]
    assert summary["bootstrap_n"].tolist() == [200, 200]


def test_summary_derives_group_order_independent_bootstrap_seeds_from_identity():
    labels = np.array([0, 0, 0, 1, 1, 1])
    rows = []
    for source, target in (("eng", "eng"), ("por", "zho")):
        for layer in (1, 2):
            rows.append(
                {
                    "profile": "wavlm_base_plus",
                    "layer": layer,
                    "source": source,
                    "target": target,
                    "auc": 1.0,
                    "scores": np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
                    "labels": labels,
                    "corpus_shift": source != target,
                }
            )
    performance = pd.DataFrame(rows)

    first = summarize_emergence(
        performance,
        layer_indices=(1, 2),
        n_bootstrap=30,
        seed=19,
    )
    second = summarize_emergence(
        performance.iloc[::-1].reset_index(drop=True),
        layer_indices=(1, 2),
        n_bootstrap=30,
        seed=19,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["onset"].tolist() == [1, 1]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.drop(
                frame[
                    (frame["source"] == "eng")
                    & (frame["target"] == "eng")
                    & (frame["layer"] == 3)
                ].index
            ),
            "exactly one row",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "exactly one row",
        ),
        (
            lambda frame: frame.assign(
                corpus_shift=lambda value: np.where(
                    (value["source"] == "eng")
                    & (value["target"] == "eng")
                    & (value["layer"] == 1),
                    True,
                    value["corpus_shift"],
                )
            ),
            "corpus_shift",
        ),
        (
            lambda frame: frame.assign(
                auc_ci_lower=lambda value: np.where(
                    value["layer"] == 1,
                    0.9,
                    value["auc_ci_lower"],
                ),
                auc_ci_upper=lambda value: np.where(
                    value["layer"] == 1,
                    0.8,
                    value["auc_ci_upper"],
                ),
            ),
            "confidence interval",
        ),
    ],
)
def test_summary_rejects_incomplete_duplicate_or_inconsistent_cells(mutation, message):
    with pytest.raises(ValueError, match=message):
        summarize_emergence(
            mutation(_precomputed_rows()),
            layer_indices=(1, 2, 3, 4, 5),
            n_bootstrap=20,
        )


def test_summary_rejects_float_values_in_performance_layer_column():
    performance = _precomputed_rows()
    performance["layer"] = performance["layer"].astype(float)

    with pytest.raises(ValueError, match="layer.*integers"):
        summarize_emergence(
            performance,
            layer_indices=(1, 2, 3, 4, 5),
            n_bootstrap=20,
        )


def _summary_with_final_auc_at_chance() -> pd.Series:
    performance = _precomputed_rows()
    mask = (performance["source"] == "eng") & (performance["target"] == "eng")
    performance.loc[mask, "auc"] = [0.70, 0.72, 0.68, 0.65, 0.50]

    summary = summarize_emergence(
        performance,
        layer_indices=(1, 2, 3, 4, 5),
        n_bootstrap=20,
    )
    return summary.loc[summary["target"] == "eng"].iloc[0]


def test_summary_preserves_onset_when_final_auc_is_chance():
    row = _summary_with_final_auc_at_chance()

    assert row["onset"] == 4


def test_summary_keeps_consolidation_undefined_when_final_auc_is_chance():
    row = _summary_with_final_auc_at_chance()

    assert pd.isna(row["consolidation"])
