"""Detector adaptado diferenciável e relevância temporal para HF SSL.

A regressão logística ajustada pelo scikit-learn é transplantada exatamente para
buffers PyTorch. O encoder permanece congelado, mas o grafo em relação à waveform
continua ativo para Input×Gradient/AttnLRP.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
import torch

from .adaptation import score_head
from .data import SPOOF_LABEL


def _pipeline_parts(head):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(head, Pipeline) or len(head.steps) != 2:
        raise ValueError(
            "head deve ser uma pipeline StandardScaler + LogisticRegression"
        )
    scaler, classifier = head[0], head[-1]
    if not isinstance(scaler, StandardScaler):
        raise ValueError("head deve começar com StandardScaler")
    if not isinstance(classifier, LogisticRegression):
        raise ValueError("head deve terminar com LogisticRegression")
    try:
        check_is_fitted(scaler)
        check_is_fitted(classifier, attributes=("coef_", "intercept_", "classes_"))
    except Exception as exc:
        raise ValueError("head deve estar ajustada antes do transplante") from exc
    return scaler, classifier


def port_logistic_head(
    head,
    spoof_label: object = SPOOF_LABEL,
) -> tuple[np.ndarray, np.float32]:
    """Transplanta ``StandardScaler + LogisticRegression`` sem retreino.

    O ``decision_function`` binário do sklearn é orientado para ``classes_[1]``.
    Quando spoof é ``classes_[0]``, o logit transplantado é negado para continuar
    representando evidência em direção a spoof.
    """
    scaler, classifier = _pipeline_parts(head)
    classes = np.asarray(classifier.classes_)
    if classes.ndim != 1 or classes.size != 2:
        raise ValueError("head deve ser uma classificação binária ajustada")
    matches = np.flatnonzero(classes == spoof_label)
    if matches.size != 1:
        raise ValueError(
            f"spoof_label={spoof_label!r} ausente ou ambíguo em classes_={classes.tolist()}"
        )

    coef = np.asarray(classifier.coef_, dtype=np.float64)
    intercept = np.asarray(classifier.intercept_, dtype=np.float64)
    if coef.ndim != 2 or coef.shape[0] != 1 or intercept.shape != (1,):
        raise ValueError(
            "coef_ e intercept_ têm shapes incompatíveis com uma head logística binária"
        )
    n_features = coef.shape[1]
    mean = (
        np.asarray(scaler.mean_, dtype=np.float64)
        if scaler.with_mean
        else np.zeros(n_features, dtype=np.float64)
    )
    scale = (
        np.asarray(scaler.scale_, dtype=np.float64)
        if scaler.with_std
        else np.ones(n_features, dtype=np.float64)
    )
    if mean.shape != (n_features,) or scale.shape != (n_features,):
        raise ValueError("mean_ e scale_ têm shapes incompatíveis com coef_")
    arrays = (mean, scale, coef, intercept)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("estado da head deve conter somente valores finitos")
    if np.any(scale <= 0.0):
        raise ValueError("scale_ deve conter somente escalas positivas e finitas")

    w = coef[0] / scale
    b = intercept[0] - np.sum(coef[0] * mean / scale)
    if int(matches[0]) == 0:
        w = -w
        b = -b
    w32 = np.asarray(w, dtype=np.float32)
    b32 = np.float32(b)
    if not np.all(np.isfinite(w32)) or not np.isfinite(b32):
        raise ValueError("transplante da head produziu pesos ou bias não finitos")
    return w32, b32


def _validate_linear_head(w: np.ndarray, b: float) -> tuple[np.ndarray, np.float32]:
    try:
        weight = np.asarray(w, dtype=np.float32)
        bias_array = np.asarray(b, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("w e bias devem ser numéricos") from exc
    if weight.ndim != 1 or weight.size == 0:
        raise ValueError("w deve ser um vetor 1-D não vazio")
    if bias_array.ndim != 0:
        raise ValueError("bias deve ser um escalar finito")
    bias = np.float32(bias_array)
    if not np.all(np.isfinite(weight)):
        raise ValueError("w deve conter somente valores finitos")
    if not np.isfinite(bias):
        raise ValueError("bias deve ser finito")
    return weight, bias


def _resolve_hidden_state_layer(encoder: torch.nn.Module, layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, Integral):
        raise ValueError("layer deve ser um inteiro")
    requested = int(layer)
    configured = getattr(getattr(encoder, "config", None), "num_hidden_layers", None)
    if configured is not None:
        if (
            isinstance(configured, bool)
            or not isinstance(configured, Integral)
            or int(configured) < 1
        ):
            raise ValueError("encoder.config.num_hidden_layers deve ser um inteiro positivo")
        last_layer = int(configured)
    else:
        last_layer = 12
    resolved = last_layer if requested == -1 else requested
    if not 1 <= resolved <= last_layer:
        raise ValueError(
            f"layer deve estar entre 1 e {last_layer}, ou ser -1 para a última camada"
        )
    return resolved


class SSLDetectorAD(torch.nn.Module):
    """Encoder SSL congelado, pooling temporal e logit de spoof transplantado."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        layer: int,
        w: np.ndarray,
        b: float,
    ) -> None:
        super().__init__()
        if isinstance(layer, bool) or not isinstance(layer, Integral) or not 1 <= int(layer) <= 12:
            raise ValueError("layer principal deve ser um inteiro entre 1 e 12")
        self._initialize(encoder, int(layer), w, b)

    def _initialize(
        self,
        encoder: torch.nn.Module,
        layer: int,
        w: np.ndarray,
        b: float,
    ) -> None:
        if not isinstance(encoder, torch.nn.Module):
            raise ValueError("encoder deve ser um torch.nn.Module")
        weight, bias = _validate_linear_head(w, b)
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.layer = int(layer)
        self.register_buffer("w", torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer("b", torch.as_tensor(bias, dtype=torch.float32))

    def _hidden_frame_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != hidden.shape[0]:
            raise ValueError("attention_mask deve ter shape (batch, input_samples)")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask deve conter somente 0 e 1")
        if attention_mask.shape[1] == hidden.shape[1]:
            frame_mask = attention_mask
        else:
            converter = getattr(self.encoder, "_get_feature_vector_attention_mask", None)
            if callable(converter):
                frame_mask = converter(hidden.shape[1], attention_mask)
            else:
                length_converter = getattr(
                    self.encoder, "_get_feat_extract_output_lengths", None
                )
                if not callable(length_converter):
                    raise ValueError(
                        "encoder não converte attention_mask para os frames ocultos"
                    )
                lengths = torch.as_tensor(
                    length_converter(attention_mask.sum(-1)),
                    device=hidden.device,
                    dtype=torch.long,
                )
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                frame_mask = positions[None, :] < lengths[:, None]
        frame_mask = torch.as_tensor(frame_mask, device=hidden.device)
        if frame_mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask convertida tem shape incompatível com hidden state")
        counts = frame_mask.sum(dim=1)
        if torch.any(counts <= 0):
            raise ValueError("attention_mask não pode mascarar todos os frames")
        return frame_mask.to(dtype=torch.bool)

    def _hidden_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_mask = self._hidden_frame_mask(hidden, attention_mask)
        counts = frame_mask.sum(dim=1)
        mask = frame_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(input_values, torch.Tensor) or input_values.ndim != 2:
            raise ValueError("input_values deve ter shape (batch, input_samples)")
        if input_values.shape[0] == 0 or input_values.shape[1] == 0:
            raise ValueError("input_values não pode ser vazio")
        if not torch.all(torch.isfinite(input_values)):
            raise ValueError("input_values deve conter somente valores finitos")
        kwargs = {"output_hidden_states": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        output = self.encoder(input_values, **kwargs)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or len(hidden_states) <= self.layer:
            actual = 0 if hidden_states is None else len(hidden_states)
            raise ValueError(
                f"encoder não retornou hidden state da layer {self.layer}; recebeu {actual}"
            )
        hidden = hidden_states[self.layer]
        if hidden.ndim != 3 or hidden.shape[0] != input_values.shape[0]:
            raise ValueError("hidden state deve ter shape (batch, frames, hidden)")
        if hidden.shape[-1] != self.w.numel():
            raise ValueError(
                f"hidden dim={hidden.shape[-1]} incompatível com w={self.w.numel()}"
            )
        return apply_head_to_hidden(self, hidden, attention_mask=attention_mask)


def apply_head_to_hidden(
    model: SSLDetectorAD,
    hidden: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aplica o pooling do detector e sua head a um hidden state já calculado."""
    if not isinstance(model, SSLDetectorAD):
        raise ValueError("model deve ser um SSLDetectorAD")
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 3
        or hidden.shape[0] == 0
        or hidden.shape[1] == 0
        or hidden.shape[2] != model.w.numel()
    ):
        raise ValueError("hidden dim/shape incompatível com a head")
    if not torch.all(torch.isfinite(hidden)):
        raise ValueError("hidden state contém valores não finitos")
    pooled = (
        hidden.mean(dim=1)
        if attention_mask is None
        else model._hidden_mask(hidden, attention_mask)
    )
    logits = pooled @ model.w + model.b
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("logit contém valores não finitos")
    return logits


def _legacy_ssl_detector_ad(
    encoder: torch.nn.Module,
    layer: int,
    w: np.ndarray,
    b: float,
) -> SSLDetectorAD:
    """Adaptador privado para configs dev históricos com ``layer=-1``."""
    resolved = _resolve_hidden_state_layer(encoder, layer)
    detector = SSLDetectorAD.__new__(SSLDetectorAD)
    torch.nn.Module.__init__(detector)
    detector._initialize(encoder, resolved, w, b)
    return detector


def relevance_for_clip(
    model: SSLDetectorAD,
    processor,
    wav16k: np.ndarray,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Retorna entrada efetiva, relevância Input×Gradient e logit de spoof."""
    wav = np.asarray(wav16k)
    if wav.ndim != 1 or wav.size == 0 or not np.all(np.isfinite(wav)):
        raise ValueError("áudio deve ser um vetor 1-D não vazio e finito")
    inputs = processor([wav], sampling_rate=16000, return_tensors="pt")
    if "input_values" not in inputs:
        raise ValueError("processor não retornou input_values")
    values = torch.as_tensor(inputs["input_values"])
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
        raise ValueError("processor retornou input_values com shape inválido")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("processor retornou input_values não finitos")
    input_values = values.to(device).detach().clone().requires_grad_(True)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = torch.as_tensor(attention_mask).to(device)

    model.zero_grad(set_to_none=True)
    logits = model(input_values, attention_mask=attention_mask)
    if logits.shape != (1,):
        raise ValueError(f"modelo deve retornar um logit por amostra; recebeu {tuple(logits.shape)}")
    logit = logits[0]
    if not torch.isfinite(logit):
        raise ValueError("modelo retornou logit não finito")
    logit.backward()
    gradient = input_values.grad
    if gradient is None or gradient.shape != input_values.shape:
        raise ValueError("gradiente da entrada ausente ou com shape inválido")
    relevance = input_values * gradient
    if not torch.all(torch.isfinite(gradient)) or not torch.all(torch.isfinite(relevance)):
        raise ValueError("gradiente ou relevância da entrada contém valores não finitos")
    x_time = input_values.detach().cpu().numpy()[0]
    r_time = relevance.detach().cpu().numpy()[0]
    return x_time, r_time, float(logit.detach().cpu())


def verify_score_equivalence(
    head,
    embeddings: np.ndarray,
    w: np.ndarray | None = None,
    b: float | None = None,
    *,
    spoof_label: object = SPOOF_LABEL,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> float:
    """Valida ``sigmoid(logit Torch) == score_head`` e retorna o erro máximo."""
    emb = np.asarray(embeddings)
    if emb.ndim != 2 or emb.shape[0] == 0 or not np.all(np.isfinite(emb)):
        raise ValueError("embeddings deve ser uma matriz 2-D não vazia e finita")
    if not np.isfinite(rtol) or rtol < 0 or not np.isfinite(atol) or atol < 0:
        raise ValueError("rtol e atol devem ser finitos e não negativos")
    if (w is None) != (b is None):
        raise ValueError("w e b devem ser fornecidos juntos")
    if w is None:
        weight, bias = port_logistic_head(head, spoof_label=spoof_label)
    else:
        weight, bias = _validate_linear_head(w, b)
    if emb.shape[1] != weight.size:
        raise ValueError("dimensão dos embeddings incompatível com w")

    with torch.no_grad():
        tensor = torch.as_tensor(emb, dtype=torch.float32)
        probabilities = torch.sigmoid(tensor @ torch.as_tensor(weight) + float(bias))
        actual = probabilities.cpu().numpy().astype(np.float64)
    expected = score_head(head, emb, spoof_label=spoof_label)
    error = np.abs(actual - expected)
    max_error = float(np.max(error))
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(
            "probabilidade Torch diverge de score_head: "
            f"erro máximo={max_error:.6g}, rtol={rtol:g}, atol={atol:g}"
        )
    return max_error


def conservation_certificate(
    model: SSLDetectorAD,
    processor,
    wavs: Sequence[np.ndarray],
    device: str | torch.device,
    b: float,
    tol: float = 1e-3,
) -> float:
    """Mede conservação com biases do encoder temporariamente zerados.

    A fórmula é a mesma do script original:
    ``abs(sum(R_input) - (logit - b_head)) / (abs(logit - b_head) + 1e-9)``.
    """
    clips = list(wavs)
    if not clips:
        raise ValueError("wavs não pode ser vazio")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol deve ser finita e positiva")
    if not np.isfinite(b):
        raise ValueError("b_head deve ser finito")

    saved: dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, parameter in model.encoder.named_parameters():
                if name.endswith("bias"):
                    saved[name] = parameter.detach().clone()
                    parameter.zero_()
        residuals = []
        for wav in clips:
            _, r_time, logit = relevance_for_clip(model, processor, wav, device)
            target = logit - float(b)
            residual = abs(float(r_time.sum()) - target) / (abs(target) + 1e-9)
            if not np.isfinite(residual):
                raise ValueError("certificado produziu resíduo não finito")
            residuals.append(residual)
        return float(np.max(residuals))
    finally:
        with torch.no_grad():
            parameters = dict(model.encoder.named_parameters())
            for name, value in saved.items():
                parameters[name].copy_(value)
