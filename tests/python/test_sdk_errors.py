"""Bad arguments: C returns -1, Python raises ValueError. Flat windows must stay fast."""
import ctypes as C
import time

import numpy as np
import pytest

from threadwatch import sdk
from threadwatch.lib import TwStats, default_path, load

X = np.random.default_rng(1).normal(size=500).astype(np.float32)


@pytest.mark.parametrize("fn", [sdk.zscore, sdk.mad, sdk.window_features])
def test_python_rejects_bad_arguments(fn):
    for kwargs in ({"w": 1}, {"w": 0}, {"s": 0}, {"threads": 0}, {"w": 1000}):
        with pytest.raises(ValueError):
            fn(X, **kwargs)
    with pytest.raises(ValueError):
        fn(np.zeros((5, 5), np.float32))


def test_python_rejects_bad_kmeans_shapes_and_progress():
    F = np.zeros((4, 4), np.float32)
    with pytest.raises(ValueError):
        sdk.kmeans_score(F, np.zeros((2, 3)), np.ones(4))
    with pytest.raises(ValueError):
        sdk.kmeans_score(F, np.zeros((2, 4)), np.ones(3))
    with pytest.raises(ValueError):
        sdk.zscore(X, 16, progress=np.zeros(4, np.int32))


@pytest.mark.parametrize("name", ["tw_zscore", "tw_mad", "tw_window_features"])
def test_c_returns_minus_one_on_bad_arguments(name):
    raw = C.CDLL(str(default_path()))                       # separate handle: plain pointer types so None works
    fn = getattr(raw, name)
    fn.argtypes = [C.c_void_p, C.c_int64, C.c_int32, C.c_int32, C.c_void_p, C.c_int32, C.c_void_p, C.c_void_p]
    fn.restype = C.c_int
    out = np.empty(4 * len(X), np.float32)                    # window_features needs 4 floats per window
    px, po = X.ctypes.data, out.ctypes.data
    n = len(X)
    assert fn(None, n, 16, 1, po, 1, None, None) == -1      # null input
    assert fn(px, n, 16, 1, None, 1, None, None) == -1      # null output
    assert fn(px, n, 1, 1, po, 1, None, None) == -1         # w < 2
    assert fn(px, n, 16, 0, po, 1, None, None) == -1        # s < 1
    assert fn(px, 10, 16, 1, po, 1, None, None) == -1       # n < w
    assert fn(px, n, 16, 1, po, 0, None, None) == -1        # threads < 1
    assert fn(px, n, 16, 1, po, 1, None, None) == 0         # sanity: valid call works


def test_c_kmeans_returns_minus_one_on_bad_arguments():
    raw = C.CDLL(str(default_path()))
    fn = raw.tw_kmeans_score
    fn.argtypes = [C.c_void_p, C.c_int64, C.c_int32, C.c_void_p, C.c_void_p, C.c_int32,
                   C.c_void_p, C.c_int32, C.c_void_p, C.c_void_p]
    fn.restype = C.c_int
    F, Cn, sc, out = np.zeros((3, 4), np.float32), np.zeros((2, 4), np.float32), np.ones(4, np.float32), np.zeros(3, np.float32)
    a = (F.ctypes.data, 3, 4, Cn.ctypes.data, sc.ctypes.data, 2, out.ctypes.data, 1, None, None)
    assert fn(*a) == 0
    for pos, bad in ((0, None), (3, None), (4, None), (6, None), (2, 0), (5, 0), (7, 0)):
        args = list(a)
        args[pos] = bad
        assert fn(*args) == -1


def test_progress_array_must_hold_threads_times_eight():
    with pytest.raises(ValueError):
        sdk.zscore(X, 16, threads=4, progress=np.zeros(8, np.int64))        # needs 32 elements
    out, st = sdk.zscore(X, 16, threads=4, progress=sdk.progress_array(4))
    assert st.windows == len(out)


def test_non_contiguous_input_is_accepted():
    strided = np.random.default_rng(2).normal(size=1000).astype(np.float32)[::2]
    assert not strided.flags.c_contiguous
    out, st = sdk.zscore(strided, 32)
    assert len(out) == st.windows == len(strided) - 31


def test_threads_are_capped_at_logical_processors_and_reported():
    out, st = sdk.zscore(X, 16, threads=10_000)
    assert st.threads == sdk.max_threads() and len(out) == len(X) - 15


def test_flat_windows_do_not_slow_mad_down():
    """Quickselect pitfall: constant / repeated values must not turn it quadratic."""
    n = 20_000
    random = np.random.default_rng(0).normal(size=n).astype(np.float32)
    flat = np.zeros(n, np.float32)
    two = (np.arange(n) % 2).astype(np.float32)

    def best(x):
        times = []
        for _ in range(3):
            t = time.perf_counter()
            sdk.mad(x, 256)
            times.append(time.perf_counter() - t)
        return min(times)

    base = best(random)
    assert best(flat) < 4 * base + 0.05
    assert best(two) < 4 * base + 0.05


def test_out_buffer_is_filled_in_place_and_validated():
    x = np.random.default_rng(3).normal(size=2000).astype(np.float32)
    ref, _ = sdk.zscore(x, 32, 1)
    out = np.empty(len(ref), np.float32)
    got, _ = sdk.zscore(x, 32, 1, out=out)
    assert got is out and np.array_equal(out, ref)
    for bad in (np.empty(len(ref) - 1, np.float32), np.empty(len(ref), np.float64), np.empty(2 * len(ref), np.float32)[::2]):
        with pytest.raises(ValueError):
            sdk.zscore(x, 32, 1, out=bad)
