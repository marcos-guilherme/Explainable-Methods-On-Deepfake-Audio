"""Encoder SSL genérico do HuggingFace ``transformers`` (Wav2Vec2 / WavLM / HuBERT).

Extrator de features apenas (``has_zero_shot=False``): gera um embedding por clipe pela
média dos frames de uma camada oculta configurável. O acoplamento ao ``transformers``
fica contido aqui; os estágios do pipeline só falam com a interface ``AudioEmbedder``.

Ex.: ``model.encoder=hf_ssl`` + ``model.checkpoint=facebook/hubert-base-ls960``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..layerwise import validate_layer_embeddings
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
        self.n_transformer_layers = int(
            getattr(self._model.config, "num_hidden_layers", 0)
        )

    def preprocess_waveform(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Canonical waveform used by embeddings, XAI and final trace."""
        wav = to_mono(torch.as_tensor(audio, dtype=torch.float32))
        wav = resample_to_16k(wav, sr)
        wav = fix_length(wav, self.num_samples)
        return wav.numpy()

    def _to_16k_mono(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Backward-compatible alias for the canonical preprocessing routine."""
        return self.preprocess_waveform(audio, sr)

    def _masked_mean(self, hidden: torch.Tensor,
                     attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Média sobre os frames válidos (ignora padding), por clipe."""
        if attention_mask is None:
            return hidden.mean(dim=1)
        if attention_mask.ndim != 2 or attention_mask.shape[0] != hidden.shape[0]:
            raise ValueError(
                "attention mask must have shape (batch, input_samples)"
            )
        length_converter = getattr(
            self._model, "_get_feat_extract_output_lengths", None
        )
        if not callable(length_converter):
            raise RuntimeError(
                "model cannot derive feature lengths from the attention mask"
            )
        try:
            feat_lens = length_converter(attention_mask.sum(-1))
        except Exception as exc:
            raise RuntimeError(
                "failed to derive feature lengths from the attention mask"
            ) from exc
        feat_lens = torch.as_tensor(
            feat_lens, device=hidden.device, dtype=torch.long
        )
        t = hidden.shape[1]
        if feat_lens.shape != (hidden.shape[0],):
            raise ValueError(
                "model returned invalid feature lengths for masked pooling"
            )
        if torch.any(feat_lens <= 0) or torch.any(feat_lens > t):
            raise ValueError(
                "model returned feature lengths outside the hidden-state range"
            )
        idx = torch.arange(t, device=hidden.device)[None, :]
        mask = (idx < feat_lens[:, None]).unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    @torch.inference_mode()
    def extract_all_layer_embeddings(
        self,
        audios: list[np.ndarray],
        srs: list[int],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Extrai e agrupa todos os hidden states em ``[N, L + 1, H]``."""
        if len(audios) != len(srs):
            raise ValueError("audios and srs must have the same length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not audios:
            return np.empty(
                (0, self.n_transformer_layers + 1, self.dim),
                dtype=np.float32,
            )

        batches: list[np.ndarray] = []
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
            if attention_mask is None:
                sample_lengths = torch.tensor(
                    [len(wav) for wav in wavs],
                    device=input_values.device,
                    dtype=torch.long,
                )
                sample_positions = torch.arange(
                    input_values.shape[1], device=input_values.device
                )
                attention_mask = (
                    sample_positions[None, :] < sample_lengths[:, None]
                ).to(torch.long)
            else:
                attention_mask = attention_mask.to(self.device)
            out = self._model(input_values, attention_mask=attention_mask,
                              output_hidden_states=True)
            hidden_states = out.hidden_states
            expected_states = self.n_transformer_layers + 1
            if hidden_states is None or len(hidden_states) != expected_states:
                actual = 0 if hidden_states is None else len(hidden_states)
                raise ValueError(
                    f"invalid number of hidden layers: expected "
                    f"{expected_states}, got {actual}"
                )
            pooled_layers = [
                self._masked_mean(hidden, attention_mask)
                for hidden in hidden_states
            ]
            batch_layers = torch.stack(pooled_layers, dim=1)
            batch_array = batch_layers.cpu().numpy().astype(np.float32)
            validate_layer_embeddings(
                batch_array,
                expected_samples=len(wavs),
                expected_layers=self.n_transformer_layers,
            )
            if batch_array.shape[2] != self.dim:
                raise ValueError(
                    f"inconsistent embedding dimension: expected {self.dim}, "
                    f"got {batch_array.shape[2]}"
                )
            batches.append(batch_array)

        embeddings = np.concatenate(batches, axis=0)
        validate_layer_embeddings(
            embeddings,
            expected_samples=len(audios),
            expected_layers=self.n_transformer_layers,
        )
        return embeddings

    def extract_embeddings(self, audios: list[np.ndarray], srs: list[int],
                           batch_size: int = 8) -> np.ndarray:
        """Embedding por clipe da camada ``self.layer``, com semântica Python."""
        all_layers = self.extract_all_layer_embeddings(audios, srs, batch_size)
        layer_count = self.n_transformer_layers + 1
        resolved_layer = self.layer if self.layer >= 0 else layer_count + self.layer
        if resolved_layer < 0 or resolved_layer >= layer_count:
            raise IndexError(
                f"layer index {self.layer} out of range for {layer_count} hidden states"
            )
        return all_layers[:, resolved_layer, :]
