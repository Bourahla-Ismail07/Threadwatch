"""Output must not depend on the thread count or the schedule: np.array_equal, 3 repeated runs."""
import numpy as np
import pytest

from threadwatch import sdk
from threadwatch.data import list_series, load_series
from threadwatch.data.loader import release
from threadwatch.detectors.ml import fit_models

DATA = pytest.mark.skipif(not list_series(prefix="syn_"), reason="run: python -m threadwatch.data all")
THREADS = [2, 4, 8, 10_000]                                  # 10_000 = "the maximum" (capped at the logical processors)


def _signal():
    x, _, _ = load_series("syn_1M")
    a = np.array(x[:300_000])                                 # real data; first 300k samples keeps pytest fast
    release()
    return a


@DATA
@pytest.mark.parametrize("kernel", ["features", "zscore", "mad", "mad_dynamic", "kmeans"])
def test_output_identical_for_every_thread_count(kernel):
    x = _signal()
    F = sdk.window_features(x, 256, 1)[0]
    idx = np.sort(np.random.default_rng(0).choice(len(F), 50_000, replace=False))
    cen, sc, _ = fit_models(F[idx], k=8, seed=0)
    call = {
        "features": lambda t: sdk.window_features(x, 256, 1, threads=t),
        "zscore": lambda t: sdk.zscore(x, 256, 1, threads=t),
        "mad": lambda t: sdk.mad(x, 256, 1, threads=t),
        "mad_dynamic": lambda t: sdk.mad(x, 256, 1, threads=t, schedule="dynamic"),
        "kmeans": lambda t: sdk.kmeans_score(F, cen, sc, threads=t),
    }[kernel]
    baseline, st1 = call(1)
    assert st1.threads == 1
    for t in THREADS:
        for _ in range(3):
            out, st = call(t)
            assert np.array_equal(baseline, out), f"{kernel}: threads={t} differs from 1 thread"
            assert st.threads == min(t, sdk.max_threads())


def test_stride_and_odd_sizes_stay_identical():
    x = np.random.default_rng(5).normal(size=10_007).astype(np.float32)
    for fn in (sdk.zscore, sdk.mad, sdk.window_features):
        base = fn(x, 33, 7, threads=1)[0]
        for t in (2, 3, 5, 10_000):
            assert np.array_equal(base, fn(x, 33, 7, threads=t)[0])
