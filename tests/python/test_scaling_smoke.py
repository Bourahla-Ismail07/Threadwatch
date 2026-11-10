"""Smoke test only: more threads must not make a clearly parallel kernel slower. Real scaling: experiments/omp_quick_scaling.py."""
import time

import numpy as np
import pytest

from threadwatch import sdk


def best(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


@pytest.mark.skipif(sdk.max_threads() < 2, reason="needs at least 2 logical processors")
def test_two_threads_are_not_slower_than_one_on_mad():
    x = np.random.default_rng(3).normal(size=200_000).astype(np.float32)
    t1 = best(lambda: sdk.mad(x, 256, 1, threads=1))
    t2 = best(lambda: sdk.mad(x, 256, 1, threads=2))
    assert t2 < t1 * 0.9, f"2 threads ({t2:.3f}s) not faster than 1 thread ({t1:.3f}s)"
