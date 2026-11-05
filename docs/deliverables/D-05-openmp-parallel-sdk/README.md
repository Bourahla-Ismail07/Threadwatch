# D-05 - OpenMP parallel SDK

| | |
|---|---|
| **Teacher's item** | Programme parallèle |
| **Depends on** | D-04 |
| **Unblocks** | D-06, D-07, D-08 |

## What this part is responsible for

Turning the sequential kernels into OpenMP kernels that use `threads` threads, keep the output identical to the 1-thread output, and publish per-thread progress counters for the dashboard. This is the core HPC contribution of the project.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| OpenMP (`-fopenmp`, `libgomp`) | work sharing over windows, `num_threads`, scheduling |
| C11 | the kernels, one writer per output slot |
| `ctypes` + NumPy | call with a thread count, read counters |
| psutil | physical and logical core counts for the scaling plan |
| pytest | determinism and equivalence tests |

## Tasks

| Task | What |
|---|---|
| D-05.1 | Parallelise `tw_window_features` and `tw_zscore` |
| D-05.2 | Parallelise `tw_mad` (per-thread buffers) |
| D-05.3 | Parallelise `tw_kmeans_score` |
| D-05.4 | Per-thread progress counters (64-byte stride) |
| D-05.5 | Determinism and equivalence tests; first scaling check (1, 2, 4, 8 threads) |
| D-05.6 | Experiment: `schedule(static)` against `schedule(dynamic, 4096)` on the MAD kernel |
| D-05.7 | Optional: `tw_zscore_sliding` (O(1) update per window) |

## Pseudo code

```c
int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s, float* score,
              int32_t threads, int64_t* progress, tw_stats* st) {
    if (!x || !score || w < 2 || s < 1 || n < w || threads < 1) return TW_E_ARG;
    int64_t nw = (n - w) / s + 1;
    double t0 = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int64_t done = 0;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < nw; i++) {
            /* identical body to D-04: one window, one thread, fixed order */
            score[i] = zscore_one_window(x + i * s, w);
            if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
        }
        if (progress) progress[(int64_t)tid * 8] = done;
    }

    if (st) { st->windows = nw; st->ms = (omp_get_wtime() - t0) * 1000.0; st->threads = threads; st->status = TW_OK; }
    return TW_OK;
}

/* MAD: scratch buffers are allocated inside the parallel region, one per thread */
#pragma omp parallel num_threads(threads)
{
    float* buf = malloc(2 * (size_t)w * sizeof(float));
    int failed = (buf == NULL);
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < nw; i++) { if (!failed) score[i] = mad_one_window(x + i * s, w, buf); }
    free(buf);
    if (failed) { /* record error with an atomic flag, return TW_E_NOMEM after the region */ }
}
```

Rules that keep it correct:

- **One writer per output slot:** `score[i]` is written only by the thread that owns window `i`. No locks, no atomics on data.
- **No reductions across threads:** each score is computed entirely by one thread in a fixed order. This is why the output does not depend on the thread count.
- **Counters:** `progress[tid * 8]`, a 64-byte stride, written only by thread `tid`. Counters are plain stores read by another process or thread; they are informational and need no ordering.
- **`threads` is a parameter:** `num_threads(threads)` per call, never `omp_set_num_threads` (global state).
- **`schedule(static)`:** equal contiguous blocks. For MAD the cost per window varies with the data, so compare `schedule(dynamic, 4096)` in D-05.6 and report the difference.

## Optional: sliding-sum z-score (D-05.7)

Each thread processes a contiguous block. It computes `sum` and `sum_sq` for its first window directly, then updates in O(1): `sum += x_in - x_out`, `sum_sq += x_in^2 - x_out^2`. Windows inside a block depend on the previous one, so the parallel unit is the block, not the window. Initial sums differ per block, so results are within float tolerance of the direct kernel but **not bit-identical across thread counts**. Report it separately as an algorithmic optimisation with a different scaling profile (little work per byte read, so it saturates memory bandwidth earlier). Accumulate in `double` and subtract a per-block offset to avoid cancellation.

## Files to create

- `sdk/src/{features.c, zscore.c, mad.c, kmeans.c}` (modified), optional `sdk/src/zscore_sliding.c`
- `python/threadwatch/sdk.py` (adds `threads=` and `progress=` use)
- `tests/python/{test_determinism.py, test_scaling_smoke.py}`
- `experiments/omp_quick_scaling.py`, `docs/omp-notes.md` (first scaling table, schedule comparison)

## Tests and acceptance

- **Determinism:** for each kernel on `syn_1M`, output for threads = 2, 4, 8 and the maximum equals the 1-thread output exactly (`np.array_equal`), over 3 repeated runs.
- **Equivalence:** still within 1e-5 of the NumPy reference.
- **Scaling check:** speedup measured at 2, 4 and 8 threads on `syn_5M` for all kernels, median of 5 runs, table committed.
- **Progress counters:** after a call, the sum of counters equals `nw`; during a long call, counters increase monotonically (read from a second Python thread).
- **Thread count honoured:** `stats.threads` equals the requested count (up to the logical core count).

## Pitfalls

- **False sharing:** counters packed in one array element each (8-byte stride) make threads fight over a cache line and slow the kernel. Keep the 64-byte stride.
- **Oversubscription:** more threads than logical cores gives worse results; cap `threads` and report it.
- **Hyper-threading:** speedup usually flattens at the physical core count; plot against both counts and say which machine you used.
- **`omp_get_wtime` versus Python timers:** the DLL's `ms` measures the parallel region only; the Python timer around the call also includes argument handling. Record both and use the Python timer for end-to-end.
- **Python's own threads:** NumPy and scikit-learn can start their own threads; fix library thread counts (`threadpoolctl`, `n_jobs`) so they do not distort the C timings.
- **No `-ffast-math` and no `reduction(+:...)` on float scores**; both would make output depend on the thread count.

## Changelog line

`D-05: OpenMP kernels with per-thread counters, output identical for 1..max threads, first scaling table`
