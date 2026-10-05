"""Validação compartilhada de embeddings layer-wise."""
from __future__ import annotations

import numpy as np


def validate_layer_embeddings(
    array: np.ndarray,
    expected_samples: int,
    expected_layers: int,
) -> None:
    """Valida embeddings ``[amostras, blocos + entrada, dimensão]``."""
    expected_prefix = (expected_samples, expected_layers + 1)
    if not isinstance(array, np.ndarray) or array.ndim != 3:
        raise ValueError(
            "layer embeddings must be a 3D numpy array with shape "
            f"{expected_prefix + ('H',)}"
        )
    if array.shape[:2] != expected_prefix:
        raise ValueError(
            f"invalid layer embedding shape {array.shape}; expected "
            f"({expected_samples}, {expected_layers + 1}, H)"
        )
    if not np.isfinite(array).all():
        raise ValueError("layer embeddings must contain only finite values")
