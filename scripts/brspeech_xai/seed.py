"""Determinismo: fixa as seeds de random/numpy/torch."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fixa as seeds de `random`, `numpy` e (se disponível) `torch`."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
