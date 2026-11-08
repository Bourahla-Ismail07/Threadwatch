"""Python wrappers for the sequential C kernels (D-04).

Every wrapper returns (result, stats) where stats is a TwStats (windows, ms measured inside the C
code, threads, status). Return codes are mapped to exceptions: -1 ValueError, -2 MemoryError,
-3 RuntimeError. Inputs are converted to contiguous float32 arrays; the references stay alive in
the wrapper until the C call returns.
"""
from __future__ import annotations

import ctypes as C

import numpy as np

from .lib import TwStats, load

_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = load()
    return _LIB


def _check(rc):
    if rc == 0:
        return
    msg = {-1: "invalid argument", -2: "allocation failed", -3: "internal error"}.get(rc, f"error {rc}")
    raise {-1: ValueError, -2: MemoryError, -3: RuntimeError}.get(rc, RuntimeError)(f"threadwatch: {msg} (code {rc})")


def _series(x):
    a = np.ascontiguousarray(x, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("x must be one-dimensional")
    return a


def _window_args(a, w, s, threads):
    w, s, threads = int(w), int(s), int(threads)
    if w < 2 or s < 1 or threads < 1 or len(a) < w:
        raise ValueError("need w >= 2, s >= 1, threads >= 1 and len(x) >= w")
    return w, s, threads, (len(a) - w) // s + 1


def _progress_ptr(progress):
    if progress is None:
        return None
    if (not isinstance(progress, np.ndarray) or progress.dtype != np.int64
            or not progress.flags.c_contiguous or progress.size < 1):
        raise ValueError("progress must be a contiguous int64 numpy array")
    return progress.ctypes.data_as(C.POINTER(C.c_int64))


def _run(name, x, w, s, threads, progress):
    a = _series(x)
    w, s, threads, nw = _window_args(a, w, s, threads)
    out, st = np.empty(nw, np.float32), TwStats()
    _check(getattr(lib(), name)(a, len(a), w, s, out, threads, _progress_ptr(progress), C.byref(st)))
    return out, st


def window_features(x, w=256, s=1, threads=1, progress=None):
    a = _series(x)
    w, s, threads, nw = _window_args(a, w, s, threads)
    out, st = np.empty((nw, 4), np.float32), TwStats()
    _check(lib().tw_window_features(a, len(a), w, s, out.reshape(-1), threads,
                                    _progress_ptr(progress), C.byref(st)))
    return out, st


def zscore(x, w=256, s=1, threads=1, progress=None):
    return _run("tw_zscore", x, w, s, threads, progress)


def mad(x, w=256, s=1, threads=1, progress=None):
    return _run("tw_mad", x, w, s, threads, progress)


def kmeans_score(features, centroids, scale, threads=1, progress=None):
    F = np.ascontiguousarray(features, dtype=np.float32)
    Cn = np.ascontiguousarray(centroids, dtype=np.float32)
    sc = np.ascontiguousarray(scale, dtype=np.float32)
    if (F.ndim != 2 or Cn.ndim != 2 or F.shape[1] != Cn.shape[1] or sc.shape != (F.shape[1],)
            or len(Cn) < 1 or int(threads) < 1):
        raise ValueError("incompatible features, centroids or scale, or threads < 1")
    out, st = np.empty(len(F), np.float32), TwStats()
    _check(lib().tw_kmeans_score(F, len(F), F.shape[1], Cn, sc, len(Cn), out, int(threads),
                                 _progress_ptr(progress), C.byref(st)))
    return out, st
