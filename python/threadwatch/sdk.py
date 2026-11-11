"""Python wrappers for the C kernels (D-04 sequential, D-05 OpenMP).

Every wrapper returns (result, stats) where stats is a TwStats (windows, ms measured inside the C
parallel region, threads actually used, status). `threads` is capped at the logical processor count
(see max_threads()) and the real team size is in stats.threads. The result never depends on `threads`.
Progress: pass progress_array(threads); counter t is progress[t * 8]; progress_total() sums them. Return codes are mapped to exceptions: -1 ValueError, -2 MemoryError,
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


def max_threads():
    """Logical processors the OpenMP runtime will use by default (the cap for `threads`)."""
    return int(lib().tw_max_threads())


def progress_array(threads):
    """Zeroed counter array for a call with `threads` threads: 8 int64 (64 bytes) per thread."""
    return np.zeros(int(threads) * 8, dtype=np.int64)


def progress_total(progress):
    """Windows processed so far, summed over the per-thread counters."""
    return int(progress[::8].sum())


def _progress_ptr(progress, threads):
    if progress is None:
        return None
    if (not isinstance(progress, np.ndarray) or progress.dtype != np.int64
            or not progress.flags.c_contiguous or progress.size < int(threads) * 8):
        raise ValueError("progress must be a contiguous int64 numpy array of at least threads * 8 elements "
                         "(use progress_array(threads))")
    return progress.ctypes.data_as(C.POINTER(C.c_int64))


def _out_buffer(out, n):
    """Result buffer: a fresh array, or the caller's (float32, contiguous, writeable, exactly n entries).
    With a caller buffer the scores appear in it while the C call runs (live view, D-07)."""
    if out is None:
        return np.empty(n, np.float32)
    if (not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.ndim != 1 or out.size != n
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise ValueError(f"out must be a writeable contiguous float32 array with {n} entries")
    return out


def _run(name, x, w, s, threads, progress, *extra, out=None):
    a = _series(x)
    w, s, threads, nw = _window_args(a, w, s, threads)
    out, st = _out_buffer(out, nw), TwStats()
    _check(getattr(lib(), name)(a, len(a), w, s, out, threads, _progress_ptr(progress, threads),
                                C.byref(st), *extra))
    return out, st


def window_features(x, w=256, s=1, threads=1, progress=None):
    a = _series(x)
    w, s, threads, nw = _window_args(a, w, s, threads)
    out, st = np.empty((nw, 4), np.float32), TwStats()
    _check(lib().tw_window_features(a, len(a), w, s, out.reshape(-1), threads,
                                    _progress_ptr(progress, threads), C.byref(st)))
    return out, st


def zscore(x, w=256, s=1, threads=1, progress=None, out=None):
    return _run("tw_zscore", x, w, s, threads, progress, out=out)


def mad(x, w=256, s=1, threads=1, progress=None, schedule="static", out=None):
    """schedule: "static" (equal blocks) or "dynamic" (chunks of 4096 windows); same output either way."""
    if schedule not in ("static", "dynamic"):
        raise ValueError('schedule must be "static" or "dynamic"')
    return _run("tw_mad_ex", x, w, s, threads, progress, 0 if schedule == "static" else 1, out=out)


def kmeans_score(features, centroids, scale, threads=1, progress=None, out=None):
    F = np.ascontiguousarray(features, dtype=np.float32)
    Cn = np.ascontiguousarray(centroids, dtype=np.float32)
    sc = np.ascontiguousarray(scale, dtype=np.float32)
    if (F.ndim != 2 or Cn.ndim != 2 or F.shape[1] != Cn.shape[1] or sc.shape != (F.shape[1],)
            or len(Cn) < 1 or int(threads) < 1):
        raise ValueError("incompatible features, centroids or scale, or threads < 1")
    out, st = _out_buffer(out, len(F)), TwStats()
    _check(lib().tw_kmeans_score(F, len(F), F.shape[1], Cn, sc, len(Cn), out, int(threads),
                                 _progress_ptr(progress, threads), C.byref(st)))
    return out, st
