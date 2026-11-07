"""Window construction helpers."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

def windows(x: ArrayLike, w: int = 256, s: int = 1) -> NDArray:
    """Return a read-only sliding-window view sampled at stride s."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if w < 1 or s < 1:
        raise ValueError("w and s must be positive")
    if len(arr) < w:
        raise ValueError("series length must be at least the window size")
    return np.lib.stride_tricks.sliding_window_view(arr, w)[::s]
