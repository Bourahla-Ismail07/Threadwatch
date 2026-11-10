import threading
import time

import numpy as np
import pytest

from threadwatch import sdk


@pytest.mark.parametrize("fn", [sdk.zscore, sdk.mad, sdk.window_features])
@pytest.mark.parametrize("threads", [1, 2, 4])
def test_counters_sum_to_window_count(fn, threads):
    x = np.random.default_rng(1).normal(size=30_000).astype(np.float32)
    prog = sdk.progress_array(threads)
    out, st = fn(x, 64, 1, threads=threads, progress=prog)
    n = len(x) - 63
    assert sdk.progress_total(prog) == n == st.windows
    assert (prog.reshape(-1, 8)[:, 1:] == 0).all()           # only the first slot of each 64-byte block is written


def test_counters_increase_monotonically_during_a_long_call():
    x = np.random.default_rng(2).normal(size=400_000).astype(np.float32)
    threads = min(4, sdk.max_threads())
    prog = sdk.progress_array(threads)
    result = {}
    worker = threading.Thread(target=lambda: result.update(r=sdk.mad(x, 256, 1, threads=threads, progress=prog)))
    seen, t0 = [], time.perf_counter()
    worker.start()                                           # the C call releases the GIL, so we can poll meanwhile
    while worker.is_alive():
        seen.append(sdk.progress_total(prog))
        time.sleep(0.005)
    worker.join()
    duration = time.perf_counter() - t0
    assert all(b >= a for a, b in zip(seen, seen[1:])), "progress went backwards"
    assert sdk.progress_total(prog) == len(x) - 255
    if duration > 0.3:                                       # long enough to observe it in flight
        assert any(0 < v < len(x) - 255 for v in seen), "never saw a partial count"
