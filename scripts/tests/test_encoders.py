"""Testes do registry de encoders (sem torch/fairseq: só a lógica de despacho)."""
from types import SimpleNamespace

import numpy as np
import pytest

from brspeech_xai import encoders
from brspeech_xai.encoders import AudioEmbedder, available_encoders, build_encoder


class _FakeEmbedder:
    """Encoder mínimo que satisfaz o contrato, sem dependências pesadas."""
    name = "fake"
    has_zero_shot = False

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def extract_embeddings(self, audios, srs, batch_size: int = 8) -> np.ndarray:
        return np.zeros((len(audios), 3), dtype=np.float32)


def test_default_registry_lists_xlsr():
    assert "xlsr_fairseq" in available_encoders()


def test_unknown_encoder_raises():
    cfg = SimpleNamespace(encoder="nao_existe", checkpoint="x", spoof_index=0)
    with pytest.raises(KeyError):
        build_encoder(cfg, device="cpu")


def test_build_dispatches_and_passes_device(monkeypatch):
    monkeypatch.setitem(encoders._BUILDERS, "fake", lambda cfg, device: _FakeEmbedder(device))
    enc = build_encoder(SimpleNamespace(encoder="fake"), device="cuda")
    assert enc.device == "cuda"
    assert enc.extract_embeddings([np.zeros(4)], [16000]).shape == (1, 3)


def test_default_encoder_when_field_absent(monkeypatch):
    # Sem o atributo 'encoder', cai no default xlsr_fairseq; troca-se o builder por um fake
    # para não carregar torch/fairseq neste teste.
    monkeypatch.setitem(encoders._BUILDERS, "xlsr_fairseq",
                        lambda cfg, device: _FakeEmbedder(device))
    enc = build_encoder(SimpleNamespace(), device="cpu")
    assert isinstance(enc, _FakeEmbedder)


def test_fake_satisfies_protocol():
    assert isinstance(_FakeEmbedder(), AudioEmbedder)
