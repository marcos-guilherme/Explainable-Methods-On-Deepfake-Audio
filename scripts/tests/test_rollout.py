"""Testes da matemática do Attention Roll-out (sem carregar modelo HF)."""
import numpy as np
import torch

from dev.attention_rollout import compute_rollout, frame_times, temporal_relevance


def _softmax_attn(n_layers: int, n_heads: int, seq_len: int, seed: int = 0):
    """Atenções sintéticas row-stochastic (como as reais: softmax sobre a última dim)."""
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.softmax(torch.rand(1, n_heads, seq_len, seq_len, generator=g), dim=-1)
                 for _ in range(n_layers))


def test_frame_times_endpoints_and_length():
    t = frame_times(5, num_samples=16000, sample_rate=16000)  # 1.0 s
    assert len(t) == 5
    assert t[0] == 0.0
    assert np.isclose(t[-1], 1.0)


def test_compute_rollout_identity_attention_gives_identity():
    seq_len = 6
    eye_layers = tuple(torch.eye(seq_len).expand(1, 2, seq_len, seq_len).clone()
                       for _ in range(3))
    joint = compute_rollout(eye_layers)
    assert torch.allclose(joint, torch.eye(seq_len), atol=1e-6)


def test_compute_rollout_rows_sum_to_one():
    joint = compute_rollout(_softmax_attn(4, 3, 8))
    row_sums = joint.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_compute_rollout_is_deterministic():
    atts = _softmax_attn(4, 3, 8, seed=123)
    a = compute_rollout(atts)
    b = compute_rollout(atts)
    assert torch.allclose(a, b)


def test_temporal_relevance_shape_matches_seq_len():
    seq_len = 10
    rel = temporal_relevance(compute_rollout(_softmax_attn(2, 2, seq_len)))
    assert rel.shape == (seq_len,)
