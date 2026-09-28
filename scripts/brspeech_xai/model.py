"""Modelo detector de deepfake (front-end XLS-R via fairseq Wav2Vec2).

Portado verbatim de ``notebooks/deepfake_brspeech_explainability.ipynb``.

CRÍTICO: ``install_dataclass_mutable_default_shim()`` e ``reset_modules(...)``
DEVEM rodar no import do módulo, ANTES do ``from fairseq.models.wav2vec import ...``,
caso contrário o fairseq falha ao importar no Python 3.12.
"""
import dataclasses
import sys

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .preprocessing import preprocess


def install_dataclass_mutable_default_shim() -> bool:
    if getattr(dataclasses, "_mutable_default_shim_installed", False):
        return False
    original_get_field = dataclasses._get_field

    def _get_field_with_shim(cls, a_name, a_type, *args, **kwargs):
        raw = getattr(cls, a_name, dataclasses.MISSING)
        candidate = raw.default if isinstance(raw, dataclasses.Field) else raw
        if (
            candidate is not dataclasses.MISSING
            and not isinstance(candidate, type)
            and dataclasses.is_dataclass(candidate)
            and getattr(type(candidate), "__hash__", None) is None
        ):
            type(candidate).__hash__ = object.__hash__
        return original_get_field(cls, a_name, a_type, *args, **kwargs)

    dataclasses._get_field = _get_field_with_shim
    dataclasses._mutable_default_shim_installed = True
    return True


def reset_modules(prefixes: tuple[str, ...]) -> None:
    for name in [m for m in sys.modules if m.startswith(prefixes)]:
        del sys.modules[name]


install_dataclass_mutable_default_shim()
reset_modules(("fairseq", "hydra"))

from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model


class SSLModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0,
        )
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data: torch.Tensor) -> torch.Tensor:
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        return self.model(input_data, mask=False, features_only=True)["x"]


class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self) -> None:
        super().__init__()
        self.ssl_orig_output_dim = 1024
        self.num_classes = 2
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(self.ssl_orig_output_dim, self.num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        emb = self.m_ssl.extract_feat(wav)
        emb = emb.transpose(1, 2)
        pooled = self.adap_pool1d(emb).squeeze(-1)
        return self.proj_fc(pooled)


class AntiDeepfakeDetector:
    def __init__(self, model: DeepfakeDetector, spoof_index: int = 0, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.spoof_index = spoof_index
        self.device = device

    @property
    def transformer_layers(self) -> torch.nn.ModuleList:
        return self.model.m_ssl.model.encoder.layers

    def logits(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)

    @torch.inference_mode()
    def spoof_prob_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.model(waveform.to(self.device)), dim=1)
        return probs[:, self.spoof_index]

    def spoof_prob(self, audio_array: np.ndarray, orig_sr: int) -> float:
        wav = preprocess(audio_array, orig_sr).unsqueeze(0)
        return float(self.spoof_prob_tensor(wav).item())


def build_detector(
    checkpoint: str,
    spoof_index: int = 0,
    device: str = "cpu",
) -> AntiDeepfakeDetector:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepfakeDetector.from_pretrained(checkpoint)
    return AntiDeepfakeDetector(model, spoof_index=spoof_index, device=device)
