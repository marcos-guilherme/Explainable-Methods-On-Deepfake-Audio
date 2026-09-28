"""Encoder XLS-R (front-end wav2vec2 via fairseq), detector completo (embeddings + zero-shot).

Este é o único módulo que conhece a estrutura interna do detector fairseq
(``det.model.m_ssl``). Todo o acoplamento ao XLS-R fica contido aqui; os estágios do
pipeline só falam com a interface ``AudioEmbedder``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..logging_utils import progress
from ..preprocessing import preprocess


class XlsrFairseqEmbedder:
    """Embeddings pooled (1024-D) do XLS-R + P(spoof) zero-shot da head anti-deepfake."""

    name = "xlsr_fairseq"
    has_zero_shot = True
    dim = 1024

    def __init__(self, checkpoint: str, spoof_index: int = 0, device: str = "cpu") -> None:
        from ..model import build_detector
        self._det = build_detector(checkpoint, spoof_index, device)
        self.device = self._det.device

    @torch.inference_mode()
    def extract_embeddings(self, audios: list[np.ndarray], srs: list[int],
                           batch_size: int = 8) -> np.ndarray:
        """Embedding por áudio: média sobre os frames do XLS-R (== AdaptiveAvgPool1d)."""
        embs: list[np.ndarray] = []
        # Batch único (ex.: scoring de 1 clipe na oclusão) não mostra barra, senão o log
        # vira uma enxurrada de "1/1".
        single_batch = len(audios) <= batch_size
        for i in progress(range(0, len(audios), batch_size), desc="embeddings XLS-R",
                          unit="batch", disable=single_batch):
            wavs = [preprocess(a, sr) for a, sr in zip(audios[i:i + batch_size],
                                                       srs[i:i + batch_size])]
            batch = torch.stack(wavs).to(self.device)            # (B, T)
            frames = self._det.model.m_ssl.extract_feat(batch)   # (B, T', 1024)
            pooled = frames.mean(dim=1)                           # (B, 1024)
            embs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)

    def spoof_prob_batch(self, audios: list[np.ndarray], srs: list[int],
                         batch_size: int = 8, norm_mode: str = "layernorm",
                         desc: str = "inferência zero-shot") -> np.ndarray:
        """P(spoof) zero-shot para uma lista de áudios brutos (em lotes, sem gradiente)."""
        scores: list[float] = []
        for start in progress(range(0, len(audios), batch_size), desc=desc, unit="batch"):
            batch_audios = audios[start:start + batch_size]
            batch_srs = srs[start:start + batch_size]
            batch = torch.stack(
                [preprocess(a, sr, norm_mode=norm_mode) for a, sr in zip(batch_audios, batch_srs)]
            ).to(self.device)
            scores.extend(self._det.spoof_prob_tensor(batch).cpu().tolist())
        return np.asarray(scores, dtype=float)

    def spoof_prob(self, audio: np.ndarray, sr: int) -> float:
        """P(spoof) zero-shot de um único áudio bruto (usado pela oclusão de D_zs)."""
        return self._det.spoof_prob(audio, sr)
