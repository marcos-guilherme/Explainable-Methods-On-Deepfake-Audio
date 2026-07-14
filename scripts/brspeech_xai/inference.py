"""Inferência: P(spoof) do detector zero-shot, embeddings XLS-R e fábrica do D_ad."""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .preprocessing import preprocess


def score_audios(det, audios: list[np.ndarray], sample_rates: list[int],
                 batch_size: int = 8, norm_mode: str = "layernorm",
                 desc: str = "Inferencia") -> np.ndarray:
    """P(spoof) do detector zero-shot para uma lista de áudios brutos (em lotes, sem gradiente)."""
    scores: list[float] = []
    for start in tqdm(range(0, len(audios), batch_size), desc=desc):
        batch_audios = audios[start : start + batch_size]
        batch_srs = sample_rates[start : start + batch_size]
        batch = torch.stack(
            [preprocess(a, sr, norm_mode=norm_mode) for a, sr in zip(batch_audios, batch_srs)]
        ).to(det.device)
        scores.extend(det.spoof_prob_tensor(batch).cpu().tolist())
    return np.asarray(scores, dtype=float)


@torch.inference_mode()
def extract_embeddings(det, audios: list[np.ndarray], srs: list[int],
                       batch_size: int = 8) -> np.ndarray:
    """Embedding pooled (1024-D) do XLS-R por áudio (média sobre os frames == AdaptiveAvgPool1d)."""
    embs: list[np.ndarray] = []
    for i in range(0, len(audios), batch_size):
        wavs = [preprocess(a, sr) for a, sr in zip(audios[i:i+batch_size], srs[i:i+batch_size])]
        batch = torch.stack(wavs).to(det.device)                 # (B, T)
        frames = det.model.m_ssl.extract_feat(batch)             # (B, T', 1024)
        pooled = frames.mean(dim=1)                              # (B, 1024)
        embs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


def make_p_spoof_ad(logreg, detector):
    """Fábrica do detector adaptado. Encapsula o modelo logístico treinado + o detector.

    Returns:
        `(p_spoof_ad, p_spoof_ad_from_audio)`:
          - `p_spoof_ad(emb) -> np.ndarray`: P(spoof) para uma matriz de embeddings (N, 1024).
          - `p_spoof_ad_from_audio(audio, sr) -> float`: P(spoof) de um único áudio bruto
            (usado pela oclusão de D_ad). Classe positiva = spoof (índice 1).
    """
    def p_spoof_ad(emb: np.ndarray) -> np.ndarray:
        return logreg.predict_proba(emb)[:, 1]

    def p_spoof_ad_from_audio(audio: np.ndarray, sr: int) -> float:
        emb = extract_embeddings(detector, [audio], [sr])
        return float(p_spoof_ad(emb)[0])

    return p_spoof_ad, p_spoof_ad_from_audio
