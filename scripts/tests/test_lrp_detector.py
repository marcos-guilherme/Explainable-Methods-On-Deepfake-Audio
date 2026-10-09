from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import brspeech_xai.lrp_detector as lrp_detector
from brspeech_xai.adaptation import score_head
from brspeech_xai.lrp_detector import (
    SSLDetectorAD,
    apply_head_to_hidden,
    conservation_certificate,
    port_logistic_head,
    relevance_for_clip,
    verify_score_equivalence,
)


def _fit_logistic(labels=(0, 1)):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(40, 3))
    y = np.asarray([labels[i % 2] for i in range(len(x))])
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=3, max_iter=500),
    ).fit(x, y)
    return head, x


def test_port_logistic_head_reproduces_oriented_binary_scores_without_mutation():
    head, x = _fit_logistic()
    before = deepcopy(head)

    w, b = port_logistic_head(head, spoof_label=1)
    logits = x.astype(np.float32) @ w + b

    assert w.dtype == np.float32
    assert np.asarray(b).dtype == np.float32
    assert np.all(np.isfinite(w)) and np.isfinite(b)
    np.testing.assert_allclose(logits, head.decision_function(x), rtol=1e-5, atol=1e-6)
    for current, original in zip(head.named_steps.values(), before.named_steps.values()):
        for name in ("mean_", "scale_", "coef_", "intercept_", "classes_"):
            if hasattr(original, name):
                np.testing.assert_array_equal(getattr(current, name), getattr(original, name))


def test_port_logistic_head_preserves_requested_float64_precision():
    head, x = _fit_logistic()

    w, b = port_logistic_head(head, spoof_label=1, dtype=np.float64)

    assert w.dtype == np.float64
    assert np.asarray(b).dtype == np.float64
    np.testing.assert_allclose(
        x @ w + b,
        head.decision_function(x),
        rtol=1e-12,
        atol=1e-12,
    )
    assert verify_score_equivalence(head, x, w, b) <= 1e-12


def test_port_logistic_head_negates_logit_when_spoof_is_classes_zero():
    head, x = _fit_logistic(labels=(-1, 1))

    w, b = port_logistic_head(head, spoof_label=-1)
    logits = x.astype(np.float32) @ w + b

    np.testing.assert_allclose(logits, -head.decision_function(x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-logits)),
        score_head(head, x, spoof_label=-1),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("with_mean", "with_std"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_port_logistic_head_honors_standard_scaler_flags(with_mean, with_std):
    rng = np.random.default_rng(31)
    x = rng.normal(loc=4.0, scale=2.5, size=(40, 3))
    y = np.arange(len(x)) % 2
    head = make_pipeline(
        StandardScaler(with_mean=with_mean, with_std=with_std),
        LogisticRegression(random_state=7, max_iter=500),
    ).fit(x, y)

    w, b = port_logistic_head(head)

    np.testing.assert_allclose(
        x.astype(np.float32) @ w + b,
        head.decision_function(x),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "head, match",
    [
        (make_pipeline(StandardScaler(), MLPClassifier()), "LogisticRegression"),
        (make_pipeline(LogisticRegression()), "StandardScaler"),
        (make_pipeline(StandardScaler(), LogisticRegression()), "ajustad"),
    ],
)
def test_port_logistic_head_rejects_wrong_or_unfitted_pipeline(head, match):
    with pytest.raises(ValueError, match=match):
        port_logistic_head(head)


def test_port_logistic_head_rejects_multiclass_and_missing_spoof_label():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(30, 3))
    multiclass = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        x, np.arange(30) % 3
    )
    with pytest.raises(ValueError, match="bin"):
        port_logistic_head(multiclass, spoof_label=1)

    binary, _ = _fit_logistic()
    with pytest.raises(ValueError, match="spoof_label"):
        port_logistic_head(binary, spoof_label=7)


@pytest.mark.parametrize("attribute", ["mean_", "scale_", "coef_", "intercept_"])
def test_port_logistic_head_rejects_nonfinite_state(attribute):
    head, _ = _fit_logistic()
    owner = head[0] if attribute in {"mean_", "scale_"} else head[-1]
    value = getattr(owner, attribute).copy()
    value.flat[0] = np.nan
    setattr(owner, attribute, value)

    with pytest.raises(ValueError, match="finit"):
        port_logistic_head(head)


def test_port_logistic_head_rejects_incompatible_shapes():
    head, _ = _fit_logistic()
    head[0].scale_ = head[0].scale_[:-1]
    with pytest.raises(ValueError, match="shape|dimens"):
        port_logistic_head(head)


def test_verify_score_equivalence_uses_torch_probability_and_reports_divergence():
    head, x = _fit_logistic()
    w, b = port_logistic_head(head)

    max_error = verify_score_equivalence(head, x, w, b)
    assert max_error <= 1e-6

    with pytest.raises(ValueError, match=r"erro máximo=.*rtol=.*atol="):
        verify_score_equivalence(head, x, w + np.float32(0.25), b)


class _TinyEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=2):
        super().__init__()
        self.projection = torch.nn.Linear(1, hidden_dim)
        self.forward_calls = []

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        self.forward_calls.append((attention_mask, output_hidden_states))
        hidden = self.projection(input_values.unsqueeze(-1))
        return SimpleNamespace(hidden_states=(hidden * 0.5, hidden))


def test_ssl_detector_validates_contract_freezes_encoder_and_masks_pooling():
    encoder = _TinyEncoder(hidden_dim=2)
    model = SSLDetectorAD(encoder, layer=1, w=np.array([1.5, -0.5]), b=0.25)
    x = torch.tensor([[1.0, 3.0, 100.0], [2.0, 4.0, 6.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    logits = model(x, attention_mask=mask)
    logits.sum().backward()

    assert logits.shape == (2,)
    assert encoder.forward_calls[-1][1] is True
    assert encoder.forward_calls[-1][0] is mask
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert dict(model.named_parameters()) == {
        f"encoder.{name}": parameter for name, parameter in encoder.named_parameters()
    }
    assert set(dict(model.named_buffers())) == {"w", "b"}

    with torch.no_grad():
        hidden = encoder.projection(x.unsqueeze(-1))
        expected_pooled = torch.stack((hidden[0, :2].mean(0), hidden[1].mean(0)))
        expected = expected_pooled @ model.w + model.b
    torch.testing.assert_close(logits, expected)


def test_apply_head_to_hidden_is_shared_with_detector_forward(monkeypatch):
    encoder = _TinyEncoder(hidden_dim=2)
    model = SSLDetectorAD(encoder, layer=1, w=np.array([1.5, -0.5]), b=0.25)
    values = torch.tensor([[1.0, 3.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]])
    hidden = encoder.projection(values.unsqueeze(-1))
    expected = apply_head_to_hidden(model, hidden, attention_mask=mask)
    calls = []
    original = lrp_detector.apply_head_to_hidden

    def observed(current_model, current_hidden, *, attention_mask=None):
        calls.append(current_hidden)
        return original(
            current_model,
            current_hidden,
            attention_mask=attention_mask,
        )

    monkeypatch.setattr(lrp_detector, "apply_head_to_hidden", observed)

    torch.testing.assert_close(model(values, attention_mask=mask), expected)
    assert calls and calls[0].shape == hidden.shape


@pytest.mark.parametrize("layer", [-1, 0, 13])
def test_ssl_detector_public_api_rejects_nonprincipal_layers(layer):
    encoder = _TinyEncoder()
    encoder.config = SimpleNamespace(num_hidden_layers=24)
    with pytest.raises(ValueError, match="layer"):
        SSLDetectorAD(encoder, layer, np.ones(2), 0.0)


class _DeepEncoder(torch.nn.Module):
    def __init__(self, n_hidden_layers):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=n_hidden_layers)

    def forward(self, input_values, output_hidden_states=False):
        base = input_values.unsqueeze(-1)
        return SimpleNamespace(
            hidden_states=tuple(base * float(index + 1) for index in range(25))
        )


def test_legacy_ssl_detector_adapter_resolves_actual_last_hidden_state():
    encoder = _DeepEncoder(n_hidden_layers=24)

    model = lrp_detector._legacy_ssl_detector_ad(
        encoder, layer=-1, w=np.array([1.0]), b=0.0
    )

    assert model.layer == 24
    torch.testing.assert_close(model(torch.ones(1, 2)), torch.tensor([25.0]))


def test_ssl_detector_rejects_invalid_head_and_hidden_dimension():
    with pytest.raises(ValueError, match="1-D"):
        SSLDetectorAD(_TinyEncoder(), 1, np.ones((1, 2)), 0.0)
    with pytest.raises(ValueError, match="finit"):
        SSLDetectorAD(_TinyEncoder(), 1, np.array([1.0, np.inf]), 0.0)
    with pytest.raises(ValueError, match="bias"):
        SSLDetectorAD(_TinyEncoder(), 1, np.ones(2), np.nan)

    model = SSLDetectorAD(_TinyEncoder(hidden_dim=3), 1, np.ones(2), 0.0)
    with pytest.raises(ValueError, match="hidden"):
        model(torch.ones(1, 4))


class _Processor:
    def __init__(self, values, mask=None):
        self.values = values
        self.mask = mask

    def __call__(self, wavs, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        result = {"input_values": torch.tensor(self.values, dtype=torch.float32)}
        if self.mask is not None:
            result["attention_mask"] = torch.tensor(self.mask)
        return result


def test_relevance_for_clip_returns_effective_input_relevance_and_clears_grads():
    model = SSLDetectorAD(_TinyEncoder(hidden_dim=2), 1, np.array([0.7, -0.2]), 0.1)
    for parameter in model.encoder.parameters():
        parameter.grad = torch.ones_like(parameter)
    processor = _Processor([[0.25, -0.5, 0.75]], [[1, 1, 1]])

    x_time, r_time, logit = relevance_for_clip(
        model, processor, np.array([2.0, -3.0]), "cpu"
    )

    assert x_time.shape == r_time.shape == (3,)
    assert np.all(np.isfinite(x_time)) and np.all(np.isfinite(r_time))
    assert np.isfinite(logit)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.encoder.forward_calls[-1][0] is not None


def test_relevance_for_clip_matches_double_precision_detector_dtype():
    model = SSLDetectorAD(
        _TinyEncoder(hidden_dim=2),
        1,
        np.array([0.7, -0.2]),
        0.1,
    ).double()

    x_time, r_time, logit = relevance_for_clip(
        model,
        _Processor([[0.25, -0.5, 0.75]]),
        np.array([0.25, -0.5, 0.75]),
        "cpu",
    )

    assert x_time.dtype == np.float64
    assert r_time.dtype == np.float64
    assert np.isfinite(logit)


@pytest.mark.parametrize(
    "wav",
    [np.array([]), np.array([[1.0]]), np.array([np.nan])],
)
def test_relevance_for_clip_rejects_invalid_audio(wav):
    model = SSLDetectorAD(_TinyEncoder(), 1, np.ones(2), 0.0)
    with pytest.raises(ValueError, match="áudio"):
        relevance_for_clip(model, _Processor([[1.0]]), wav, "cpu")


class _HomogeneousEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.75))

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        hidden = input_values.unsqueeze(-1) * self.weight + self.bias
        return SimpleNamespace(hidden_states=(hidden, hidden))


def test_conservation_certificate_preserves_formula_and_restores_bias():
    encoder = _HomogeneousEncoder()
    model = SSLDetectorAD(encoder, 1, np.array([1.25]), 0.4)
    original_bias = encoder.bias.detach().clone()
    wavs = [np.array([0.2, -0.1, 0.7]), np.array([1.0, 0.5])]

    diagnostics = conservation_certificate(
        model, _Processor([[0.2, -0.1, 0.7]]), wavs[:1], "cpu", b=0.4
    )

    assert diagnostics.relative_error < 1e-6
    assert diagnostics.accepted
    torch.testing.assert_close(encoder.bias, original_bias)


@pytest.mark.parametrize(
    ("relevance_sum", "accepted"),
    [(0.0989, True), (0.09, False)],
)
def test_conservation_certificate_uses_mixed_absolute_relative_bound(
    monkeypatch, relevance_sum, accepted
):
    encoder = _HomogeneousEncoder()
    model = SSLDetectorAD(encoder, 1, np.array([1.25]), 0.4)
    monkeypatch.setattr(
        lrp_detector,
        "relevance_for_clip",
        lambda *args: (
            np.ones(1, dtype=np.float32),
            np.asarray([relevance_sum], dtype=np.float64),
            0.5,
        ),
    )

    diagnostics = conservation_certificate(
        model,
        _Processor([[1.0]]),
        [np.array([1.0])],
        "cpu",
        b=0.4,
        tol=1e-3,
        atol=5e-3,
    )

    assert diagnostics.evidence == pytest.approx(0.1)
    assert diagnostics.relevance_sum == pytest.approx(relevance_sum)
    assert diagnostics.absolute_error == pytest.approx(abs(relevance_sum - 0.1))
    assert diagnostics.relative_error == pytest.approx(
        abs(relevance_sum - 0.1) / (0.1 + 1e-9)
    )
    assert diagnostics.bound == pytest.approx(5e-3 + 1e-3 * 0.1)
    assert diagnostics.accepted is accepted


def test_default_conservation_bound_accepts_observed_small_logit_error(
    monkeypatch,
):
    encoder = _HomogeneousEncoder()
    model = SSLDetectorAD(encoder, 1, np.array([1.25]), 0.4)
    monkeypatch.setattr(
        lrp_detector,
        "relevance_for_clip",
        lambda *args: (
            np.ones(1, dtype=np.float32),
            np.asarray([0.22783050751587645], dtype=np.float64),
            0.636154,
        ),
    )

    diagnostics = conservation_certificate(
        model,
        _Processor([[1.0]]),
        [np.array([1.0])],
        "cpu",
        b=0.4,
    )

    assert diagnostics.absolute_error == pytest.approx(0.008323492484123562)
    assert diagnostics.absolute_tolerance == pytest.approx(0.015)
    assert diagnostics.accepted


def test_conservation_certificate_restores_bias_after_exception():
    encoder = _HomogeneousEncoder()
    model = SSLDetectorAD(encoder, 1, np.array([1.25]), 0.4)
    original_bias = encoder.bias.detach().clone()

    with pytest.raises(ValueError):
        conservation_certificate(
            model,
            _Processor([[np.nan]]),
            [np.array([1.0])],
            "cpu",
            b=0.4,
        )
    torch.testing.assert_close(encoder.bias, original_bias)


class _TwoBiasEncoder(_HomogeneousEncoder):
    def __init__(self):
        super().__init__()
        self.extra_bias = torch.nn.Parameter(torch.tensor(-0.5))


def test_conservation_certificate_restores_touched_bias_when_zeroing_raises(monkeypatch):
    encoder = _TwoBiasEncoder()
    model = SSLDetectorAD(encoder, 1, np.array([1.25]), 0.4)
    original_bias = encoder.bias.detach().clone()
    original_extra_bias = encoder.extra_bias.detach().clone()
    failing_pointer = encoder.extra_bias.data_ptr()
    original_zero = torch.Tensor.zero_

    def fail_on_second_bias(tensor):
        if tensor.data_ptr() == failing_pointer:
            raise RuntimeError("synthetic zero failure")
        return original_zero(tensor)

    monkeypatch.setattr(torch.Tensor, "zero_", fail_on_second_bias)
    with pytest.raises(RuntimeError, match="synthetic zero failure"):
        conservation_certificate(
            model,
            _Processor([[1.0]]),
            [np.array([1.0])],
            "cpu",
            b=0.4,
        )

    torch.testing.assert_close(encoder.bias, original_bias)
    torch.testing.assert_close(encoder.extra_bias, original_extra_bias)


@pytest.mark.parametrize(
    "wavs, tol, atol",
    [
        ([], 1e-3, 5e-3),
        ([np.array([1.0])], 0.0, 5e-3),
        ([np.array([1.0])], np.inf, 5e-3),
        ([np.array([1.0])], 1e-3, -1.0),
        ([np.array([1.0])], 1e-3, np.inf),
    ],
)
def test_conservation_certificate_validates_inputs(wavs, tol, atol):
    model = SSLDetectorAD(_HomogeneousEncoder(), 1, np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        conservation_certificate(
            model, _Processor([[1.0]]), wavs, "cpu", 0.0, tol, atol
        )
