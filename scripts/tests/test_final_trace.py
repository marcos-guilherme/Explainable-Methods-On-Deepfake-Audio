from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import brspeech_xai.final_trace as final_trace_module
from brspeech_xai.attnlrp import ensure_ssl_encoder_attnlrp
from brspeech_xai.final_trace import (
    FinalDecisionTrace,
    layer_transition_metrics,
    run_final_decision_trace,
    run_final_decision_trace_cell,
    trace_final_decision,
)
from brspeech_xai.layerwise_paths import (
    LayerwiseSuitePaths,
    publish_generation,
    resolve_active_generation,
)
from brspeech_xai.lrp_detector import SSLDetectorAD, apply_head_to_hidden


class GELUActivation(torch.nn.Module):
    def forward(self, values):
        return torch.nn.functional.gelu(values)


class HubertAttention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None, None


class _ToyEncoder(torch.nn.Module):
    def __init__(self, layers=3, *, variable_lengths=False):
        super().__init__()
        self.layers = layers
        self.variable_lengths = variable_lengths
        self.forward_calls = 0
        self.backward_calls = 0
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.layer_norm = torch.nn.LayerNorm(2)
        self.group_norm = torch.nn.GroupNorm(1, 2)
        self.activation = GELUActivation()
        self.attention = HubertAttention()

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        valid = attention_mask.sum(dim=-1)
        converted = torch.div(
            valid * length,
            attention_mask.shape[1],
            rounding_mode="floor",
        ).clamp_min(1)
        positions = torch.arange(length, device=attention_mask.device)
        return positions[None, :] < converted[:, None]

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        self.forward_calls += 1
        base = torch.stack(
            (input_values * self.scale, input_values.square() * self.scale + 0.5),
            dim=-1,
        )
        hidden_states = [base]
        hidden = base
        for layer in range(1, self.layers + 1):
            hidden = hidden * (1.0 + layer / 10.0) + 0.05
            if self.variable_lengths and hidden.shape[1] > 2:
                hidden = hidden[:, :-1, :]
            hidden_states.append(hidden)
        hidden_states[1].register_hook(self._count_backward)
        return SimpleNamespace(hidden_states=tuple(hidden_states))

    def _count_backward(self, gradient):
        self.backward_calls += 1
        return gradient


def _model(*, layers=3, variable_lengths=False, weight=(0.7, -0.2)):
    encoder = _ToyEncoder(layers, variable_lengths=variable_lengths)
    ensure_ssl_encoder_attnlrp(
        encoder,
        attention_rule="cp_lrp",
        capabilities={"attnlrp_cp"},
        patch_fn=lambda model, attention: {
            "layer_norm": 1,
            "group_norm": 1,
            "gelu": 1,
            "attention": 1,
        },
    )
    return SSLDetectorAD(
        encoder,
        layer=layers,
        w=np.asarray(weight, dtype=np.float32),
        b=0.1,
    )


def _trace(model=None, **kwargs):
    model = model or _model()
    return trace_final_decision(
        model,
        torch.tensor([[0.5, -0.25, 0.75, 1.0, -0.5]], dtype=torch.float32),
        profile="hubert_base",
        source="eng",
        target="por",
        sample_id="sample-1",
        expected_layers=3,
        **kwargs,
    )


def test_trace_uses_one_forward_and_backward_and_captures_all_layers():
    model = _model()

    trace = _trace(model)

    assert isinstance(trace, FinalDecisionTrace)
    assert trace.layer_indices == (1, 2, 3)
    assert trace.signed_sum.shape == (3,)
    assert trace.absolute_mass.shape == (3,)
    assert trace.temporal_entropy.shape == (3,)
    assert trace.adjacent_similarity.shape == (2,)
    assert trace.adjacent_l1_change.shape == (2,)
    assert model.encoder.forward_calls == 1
    assert model.encoder.backward_calls == 1
    assert np.isfinite(trace.logit)
    assert all(np.isfinite(value).all() for value in trace.temporal_mass)


def test_trace_freezes_and_evaluates_encoder_without_losing_hidden_gradients():
    model = _model()
    model.encoder.train()
    model.encoder.requires_grad_(True)

    trace = _trace(model)

    assert np.isfinite(trace.signed_sum).all()
    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.encoder.backward_calls == 1


def test_trace_uses_shared_hidden_head_helper(monkeypatch):
    model = _model()
    calls = []

    def observed(current_model, hidden, *, attention_mask=None):
        calls.append(hidden)
        return apply_head_to_hidden(
            current_model,
            hidden,
            attention_mask=attention_mask,
        )

    monkeypatch.setattr(final_trace_module, "apply_head_to_hidden", observed)

    _trace(model)

    assert len(calls) == 1


def test_trace_masks_padding_before_every_aggregate():
    values = torch.tensor([[0.5, -0.25, 0.75, 99.0, -88.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0, 0]])

    padded = trace_final_decision(
        _model(),
        values,
        attention_mask=mask,
        expected_layers=3,
    )
    trimmed = trace_final_decision(
        _model(),
        values[:, :3],
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        expected_layers=3,
    )

    np.testing.assert_allclose(padded.signed_sum, trimmed.signed_sum, atol=1e-7)
    np.testing.assert_allclose(padded.absolute_mass, trimmed.absolute_mass, atol=1e-7)
    np.testing.assert_allclose(
        padded.temporal_entropy, trimmed.temporal_entropy, atol=1e-7
    )
    assert padded.valid_lengths == (3, 3, 3)
    assert all(vector.shape == (3,) for vector in padded.temporal_mass)


def test_trace_interpolates_adjacent_profiles_to_minimum_positive_length():
    trace = _trace(
        _model(variable_lengths=True),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
    )

    assert trace.valid_lengths == (3, 2, 1)
    assert trace.transition_lengths == (2, 1)
    assert trace.interpolated == (True, True)
    assert np.isfinite(trace.adjacent_similarity).all()
    assert np.isfinite(trace.adjacent_l1_change).all()


def test_zero_mass_has_explicit_finite_transition_semantics():
    trace = _trace(_model(weight=(0.0, 0.0)))

    np.testing.assert_array_equal(trace.absolute_mass, np.zeros(3))
    np.testing.assert_array_equal(trace.temporal_entropy, np.zeros(3))
    np.testing.assert_array_equal(trace.adjacent_similarity, np.ones(2))
    np.testing.assert_array_equal(trace.adjacent_l1_change, np.zeros(2))
    assert trace.zero_mass == (True, True, True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model, values, mask: delattr(model.encoder, "_brspeech_attnlrp_patch"), "patch"),
        (
            lambda model, values, mask: setattr(model, "layer", 2),
            "final",
        ),
        (
            lambda model, values, mask: setattr(model.encoder, "layers", 2),
            "hidden states",
        ),
        (
            lambda model, values, mask: values.fill_(float("nan")),
            "finit",
        ),
        (
            lambda model, values, mask: mask.fill_(2),
            "attention_mask",
        ),
        (
            lambda model, values, mask: mask.zero_(),
            "frames",
        ),
    ],
)
def test_trace_fails_closed_for_invalid_contracts(mutation, message):
    model = _model()
    values = torch.ones(1, 5)
    mask = torch.ones(1, 5, dtype=torch.long)
    mutation(model, values, mask)

    with pytest.raises(ValueError, match=message):
        trace_final_decision(
            model,
            values,
            attention_mask=mask,
            expected_layers=3,
        )


def test_trace_rejects_batch_and_incompatible_head_dimension():
    with pytest.raises(ValueError, match="batch"):
        trace_final_decision(_model(), torch.ones(2, 5), expected_layers=3)
    with pytest.raises(ValueError, match="hidden dim"):
        trace_final_decision(
            _model(weight=(1.0, 2.0, 3.0)),
            torch.ones(1, 5),
            expected_layers=3,
        )


def test_layer_transition_metrics_publishes_identity_and_transition_values():
    trace = _trace()

    frame = layer_transition_metrics([trace])

    assert frame.shape[0] == 2
    assert frame[["profile", "source", "target", "sample_id"]].drop_duplicates().to_dict(
        "records"
    ) == [
        {
            "profile": "hubert_base",
            "source": "eng",
            "target": "por",
            "sample_id": "sample-1",
        }
    ]
    assert frame["previous_layer"].tolist() == [1, 2]
    assert frame["current_layer"].tolist() == [2, 3]
    assert np.isfinite(
        frame[
            [
                "similarity",
                "normalized_l1_change",
                "absolute_mass",
                "temporal_entropy",
                "signed_sum",
                "logit",
            ]
        ].to_numpy()
    ).all()


def test_layer_transition_metrics_rejects_missing_or_duplicate_identity():
    trace = _trace()
    missing = FinalDecisionTrace(
        **{**trace.__dict__, "sample_id": ""}
    )
    with pytest.raises(ValueError, match="identity"):
        layer_transition_metrics([missing])
    with pytest.raises(ValueError, match="duplicate"):
        layer_transition_metrics([trace, trace])


def _persist_task7_cohort(paths, profile, source, target, cohort):
    destination = paths.layer_xai(profile, 3, source, target)
    persisted = cohort.copy()
    persisted["processed_path"] = persisted["processed_path"].map(str)
    publish_generation(
        destination,
        lambda directory: persisted.to_parquet(
            directory / "sample_relevance.parquet", index=False
        ),
    )


class _Processor:
    def __call__(self, wavs, sampling_rate, return_tensors):
        return {
            "input_values": torch.tensor(np.stack(wavs), dtype=torch.float32),
            "attention_mask": torch.ones(1, len(wavs[0]), dtype=torch.long),
        }


def test_orchestration_reuses_task7_ids_final_head_and_publishes_generation(
    tmp_path, monkeypatch
):
    paths = LayerwiseSuitePaths(tmp_path / "suite")
    cohort = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "processed_path": [tmp_path / "s0.wav", tmp_path / "s1.wav"],
        }
    )
    for path in cohort["processed_path"]:
        path.write_bytes(b"toy")
    _persist_task7_cohort(
        paths,
        "hubert_base",
        "eng",
        "por",
        cohort.iloc[::-1].reset_index(drop=True),
    )
    _persist_task7_cohort(paths, "hubert_base", "por", "por", cohort)
    head_loader = Mock(return_value=object())
    port_head = Mock(return_value=(np.array([0.7, -0.2], dtype=np.float32), 0.1))
    encoder = _ToyEncoder()
    publisher = Mock(side_effect=publish_generation)
    monkeypatch.setattr(final_trace_module, "publish_generation", publisher)

    result = run_final_decision_trace(
        profiles=("hubert_base",),
        sources=("eng", "por"),
        targets=("por",),
        paths=paths,
        encoder_factory=lambda profile: (encoder, _Processor()),
        audio_loader=lambda path: (np.linspace(-1, 1, 5, dtype=np.float32), 16000),
        expected_layers=3,
        head_loader=head_loader,
        port_head_fn=port_head,
        patch_encoder_fn=lambda encoder, attention: {
            "attention": 1,
            "layer_norm": 1,
            "group_norm": 1,
            "gelu": 1,
        },
        encoder_spec_fn=lambda profile: SimpleNamespace(
            capabilities={"attnlrp_cp"},
            attention_rule="cp_lrp",
        ),
    )

    assert result.shape[0] == 8
    assert publisher.call_count == 1
    assert head_loader.call_args_list == [
        ((paths.probe("hubert_base", 3, source),), {})
        for source in ("eng", "por")
    ]
    published = paths.final_trace("hubert_base")
    generation = resolve_active_generation(published)
    assert (generation / "layer_relevance.parquet").is_file()
    assert (generation / "layer_transition_metrics.csv").is_file()
    assert tuple(pd.read_csv(generation / "layer_transition_metrics.csv")["sample_id"].unique()) == (
        "s0",
        "s1",
    )


def test_trace_cell_publishes_independent_source_target_generation(tmp_path):
    paths = LayerwiseSuitePaths(tmp_path / "suite")
    cohort = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "y_true": [0, 1],
            "processed_path": [tmp_path / "s0.wav", tmp_path / "s1.wav"],
        }
    )
    for path in cohort["processed_path"]:
        path.write_bytes(b"toy")
    encoder = _ToyEncoder()
    kwargs = dict(
        profile="hubert_base",
        target="por",
        cohort=cohort,
        paths=paths,
        encoder=encoder,
        processor=_Processor(),
        audio_loader=lambda path: (
            np.linspace(-1, 1, 5, dtype=np.float32),
            22050,
        ),
        sample_rate=22050,
        expected_layers=3,
        head_loader=Mock(return_value=object()),
        port_head_fn=Mock(
            return_value=(np.array([0.7, -0.2], dtype=np.float32), 0.1)
        ),
        patch_encoder_fn=lambda encoder, attention: {
            "attention": 1,
            "layer_norm": 1,
            "group_norm": 1,
            "gelu": 1,
        },
        encoder_spec_fn=lambda profile: SimpleNamespace(
            capabilities={"attnlrp_cp"},
            attention_rule="cp_lrp",
        ),
    )

    run_final_decision_trace_cell(source="eng", **kwargs)
    run_final_decision_trace_cell(source="zho", **kwargs)

    eng = resolve_active_generation(
        paths.final_trace_cell("hubert_base", "eng", "por"),
        expected_role="final_trace",
    )
    zho = resolve_active_generation(
        paths.final_trace_cell("hubert_base", "zho", "por"),
        expected_role="final_trace",
    )
    assert eng != zho
    assert set(pd.read_csv(eng / "layer_transition_metrics.csv")["source"]) == {"eng"}
    assert set(pd.read_csv(zho / "layer_transition_metrics.csv")["source"]) == {"zho"}


@pytest.mark.parametrize("field", ["y_true", "processed_path"])
def test_orchestration_rejects_task7_identity_divergence_before_forward(
    tmp_path, field
):
    paths = LayerwiseSuitePaths(tmp_path / "suite")
    first = pd.DataFrame(
        {
            "sample_id": ["same"],
            "y_true": [0],
            "processed_path": [str(tmp_path / "sample.wav")],
        }
    )
    second = first.copy()
    second.loc[0, field] = 1 if field == "y_true" else str(tmp_path / "other.wav")
    Path(first.loc[0, "processed_path"]).write_bytes(b"toy")
    _persist_task7_cohort(paths, "hubert_base", "eng", "por", first)
    _persist_task7_cohort(paths, "hubert_base", "por", "por", second)
    encoder = _ToyEncoder()

    with pytest.raises(ValueError, match=field):
        run_final_decision_trace(
            profiles=("hubert_base",),
            sources=("eng", "por"),
            targets=("por",),
            paths=paths,
            encoder_factory=lambda profile: (encoder, _Processor()),
            audio_loader=Mock(),
            expected_layers=3,
            head_loader=Mock(),
            port_head_fn=Mock(),
            patch_encoder_fn=Mock(),
            encoder_spec_fn=lambda profile: SimpleNamespace(
                capabilities={"attnlrp_cp"},
                attention_rule="cp_lrp",
            ),
        )

    assert encoder.forward_calls == 0
