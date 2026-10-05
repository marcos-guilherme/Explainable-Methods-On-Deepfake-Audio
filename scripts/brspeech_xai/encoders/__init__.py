"""Encoders de áudio plugáveis: contrato + registry.

Um encoder transforma áudios brutos em embeddings (um vetor por clipe), sobre os quais
a head leve (D_ad) é treinada. Alguns encoders também expõem P(spoof) zero-shot (D_zs);
isso é sinalizado por ``has_zero_shot``.

Organização: o acoplamento a um framework específico (fairseq, transformers, ...) vive só
no módulo daquele encoder. Os estágios do pipeline falam apenas com esta interface.
Registrar um encoder novo = uma entrada em ``_BUILDERS`` (com import preguiçoso lá dentro).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AudioEmbedder(Protocol):
    """Contrato mínimo de um encoder de áudio.

    Atributos:
        name: identificador do encoder.
        has_zero_shot: se expõe P(spoof) zero-shot (D_zs).
    Obrigatório:
        extract_embeddings(audios, srs, batch_size) -> np.ndarray (N, D) float32.
    Opcional (só quando ``has_zero_shot`` é True):
        spoof_prob_batch(audios, srs, batch_size) -> np.ndarray (N,) com P(spoof).
        spoof_prob(audio, sr) -> float com P(spoof) de um clipe.
    """
    name: str
    has_zero_shot: bool

    def extract_embeddings(self, audios: list[np.ndarray], srs: list[int],
                           batch_size: int = 8) -> np.ndarray: ...


@runtime_checkable
class LayerwiseAudioEmbedder(AudioEmbedder, Protocol):
    """Encoder capaz de extrair todos os hidden states em um único forward."""

    n_transformer_layers: int

    def extract_all_layer_embeddings(
        self,
        audios: list[np.ndarray],
        srs: list[int],
        batch_size: int = 8,
    ) -> np.ndarray: ...


def _build_xlsr_fairseq(model_cfg, device: str) -> AudioEmbedder:
    from .xlsr_fairseq import XlsrFairseqEmbedder
    return XlsrFairseqEmbedder(checkpoint=model_cfg.checkpoint,
                               spoof_index=model_cfg.spoof_index, device=device)


def _build_hf_ssl(model_cfg, device: str) -> AudioEmbedder:
    from .hf_ssl import HFSSLEmbedder
    return HFSSLEmbedder(checkpoint=model_cfg.checkpoint, layer=model_cfg.layer,
                         pooling=model_cfg.pooling, device=device)


# nome -> construtor. O import pesado (torch/fairseq/transformers) fica dentro de cada
# construtor, para não carregar nada quando o encoder não é usado.
_BUILDERS = {
    "xlsr_fairseq": _build_xlsr_fairseq,
    "hf_ssl": _build_hf_ssl,
}


def available_encoders() -> list[str]:
    """Nomes de encoders registrados."""
    return sorted(_BUILDERS)


def build_encoder(model_cfg, device: str = "cpu") -> AudioEmbedder:
    """Constrói o encoder escolhido em ``model_cfg.encoder`` (default: ``xlsr_fairseq``)."""
    name = getattr(model_cfg, "encoder", "xlsr_fairseq")
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"encoder desconhecido: {name!r}. Disponíveis: {available_encoders()}"
        ) from None
    return builder(model_cfg, device)
