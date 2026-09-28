"""Encoder SSL genérico do HuggingFace ``transformers`` (Wav2Vec2 / WavLM / HuBERT).

Extrator de features apenas (``has_zero_shot=False``): gera um embedding por clipe pela
média dos frames de uma camada oculta configurável. O acoplamento ao ``transformers``
fica contido aqui; os estágios do pipeline só falam com a interface ``AudioEmbedder``.

Ex.: ``model.encoder=hf_ssl`` + ``model.checkpoint=facebook/hubert-base-ls960``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..logging_utils import progress
from ..preprocessing import fix_length, resample_to_16k, to_mono


class HFSSLEmbedder:
    """Embedding pooled (média dos frames) de um modelo SSL do HuggingFace.

    O encoder fica congelado; só a head (D_ad) treina sobre os embeddings.
    """

    has_zero_shot = False

    def __init__(self, checkpoint: str, layer: int = -1, pooling: str = "mean",
                 device: str = "cpu", num_samples: int = 64600,
                 attn_implementation: str | None = None) -> None:
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
        if pooling != "mean":
            raise ValueError(f"pooling não suportado: {pooling!r} (use 'mean')")
        self.name = f"hf_ssl:{checkpoint}"
        self.checkpoint = checkpoint
        self.layer = int(layer)
        self.pooling = pooling
        self.device = device
        # attn_implementation: None mantém o default do transformers (sdpa, mais rápido) para
        # a extração de embeddings do pipeline. O rollout de atenção precisa de 'eager', pois
        # sdpa devolve None em output_attentions.
        self.attn_implementation = attn_implementation
        # Mesmo comprimento fixo do XLS-R (~4.04s a 16 kHz): limita a memória do feature
        # extractor convolucional (clipes longos estouravam a GPU) e mantém a comparação
        # entre encoders justa (mesma duração de entrada por clipe).
        self.num_samples = int(num_samples)
        # Backbones SSL crus (hubert/wav2vec2/wavlm base) não trazem tokenizer, então o
        # AutoProcessor tenta montar um Wav2Vec2Processor e estoura (ValueError/OSError ou
        # TypeError por vocab_file=None). Só precisamos do feature extractor para extrair
        # embeddings, então em qualquer falha caímos nele.
        try:
            self._processor = AutoProcessor.from_pretrained(checkpoint)
        except Exception:  # noqa: BLE001
            self._processor = AutoFeatureExtractor.from_pretrained(checkpoint)
        model_kwargs = {} if attn_implementation is None else {
            "attn_implementation": attn_implementation}
        self._model = AutoModel.from_pretrained(checkpoint, **model_kwargs).to(device).eval()
        self.dim = int(getattr(self._model.config, "hidden_size", 0))

    def _to_16k_mono(self, audio: np.ndarray, sr: int) -> np.ndarray:
        wav = to_mono(torch.as_tensor(audio, dtype=torch.float32))
        wav = resample_to_16k(wav, sr)
        return fix_length(wav, self.num_samples).numpy()

    def _masked_mean(self, hidden: torch.Tensor,
                     attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Média sobre os frames válidos (ignora padding), por clipe."""
        if attention_mask is None:
            return hidden.mean(dim=1)
        try:  # comprimento de frames por clipe a partir do mask de entrada (samples)
            feat_lens = self._model._get_feat_extract_output_lengths(
                attention_mask.sum(-1)).to(torch.long)
        except Exception:
            return hidden.mean(dim=1)
        t = hidden.shape[1]
        idx = torch.arange(t, device=hidden.device)[None, :]
        mask = (idx < feat_lens[:, None]).unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    @torch.inference_mode()
    def extract_embeddings(self, audios: list[np.ndarray], srs: list[int],
                           batch_size: int = 8) -> np.ndarray:
        """Embedding por clipe: média dos frames da camada ``self.layer`` (H-dim)."""
        embs: list[np.ndarray] = []
        # Batch único (ex.: scoring de 1 clipe na oclusão) não mostra barra, senão o log
        # vira uma enxurrada de "1/1".
        single_batch = len(audios) <= batch_size
        for i in progress(range(0, len(audios), batch_size), unit="batch",
                          desc=f"embeddings {self.checkpoint.split('/')[-1]}",
                          disable=single_batch):
            wavs = [self._to_16k_mono(a, sr) for a, sr in zip(audios[i:i + batch_size],
                                                              srs[i:i + batch_size])]
            inputs = self._processor(wavs, sampling_rate=16000, return_tensors="pt",
                                     padding=True, return_attention_mask=True)
            input_values = inputs["input_values"].to(self.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            out = self._model(input_values, attention_mask=attention_mask,
                              output_hidden_states=True)
            hidden = out.hidden_states[self.layer]            # (B, T', H)
            pooled = self._masked_mean(hidden, attention_mask)
            embs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)
