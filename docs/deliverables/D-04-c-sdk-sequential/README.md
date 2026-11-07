# D-04 - C SDK, sequential (T1)

| | |
|---|---|
| **Teacher's item** | Résultats séquentiels |
| **Depends on** | D-03 |
| **Unblocks** | D-05 |

## What this part is responsible for

The four C kernels, the public header and the Python loader, all running with **1 thread**. The result is the official reference time `T1` and the language gain `G_lang = T_py / T1`. No parallel code is written here: the single-thread version must be correct and measured first.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| C11 | the kernels |
| GCC (`-O3`, no `-ffast-math`) | builds `threadwatch.dll` |
| `ctypes` | typed Python signatures, arrays passed without copying |
| NumPy | holds the arrays and the reference results |
| pytest | equivalence tests against D-03 |

## Tasks

| Task | What |
|---|---|
| D-04.1 | Header `threadwatch.h` with the full ABI, `tw_stats`, error codes |
| D-04.2 | `tw_window_features` and `tw_zscore` |
| D-04.3 | `tw_mad` (quickselect) |
| D-04.4 | `tw_kmeans_score` |
| D-04.5 | Python wrapper with typed signatures and error mapping |
| D-04.6 | Equivalence tests, then T1 and G_lang table |

## Header (full ABI)

```c
typedef struct { int64_t windows; double ms; int32_t threads; int32_t status; } tw_stats;

#define TW_OK        0
#define TW_E_ARG    -1
#define TW_E_NOMEM  -2
#define TW_E_INTERNAL -3

/* progress: optional, int64[threads * 8]; counter of thread t at index t * 8; NULL = no progress */
TW_API int tw_window_features(const float* x, int64_t n, int32_t w, int32_t s,
                              float* features, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s,
                     float* score, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_mad(const float* x, int64_t n, int32_t w, int32_t s,
                  float* score, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_kmeans_score(const float* features, int64_t nw, int32_t d,
                           const float* centroids, const float* scale, int32_t k,
                           float* score, int32_t threads, int64_t* progress, tw_stats* stats);
```

In this deliverable `threads` must be 1 (other values are accepted and ignored until D-05). Keep the parameter now so the ABI does not change later.

## Pseudo code

```c
/* one window, double accumulation */
static inline void window_stats(const float* p, int w, double* mean, double* sd, float* mn, float* mx) {
    double sum = 0.0; float lo = p[0], hi = p[0];
    for (int k = 0; k < w; k++) { sum += p[k]; if (p[k] < lo) lo = p[k]; if (p[k] > hi) hi = p[k]; }
    double mu = sum / w, ss = 0.0;
    for (int k = 0; k < w; k++) { double d = p[k] - mu; ss += d * d; }
    *mean = mu; *sd = sqrt(ss / (w - 1)); *mn = lo; *mx = hi;
}

int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s, float* score,
              int32_t threads, int64_t* progress, tw_stats* st) {
    if (!x || !score || w < 2 || s < 1 || n < w || threads < 1) return TW_E_ARG;
    int64_t nw = (n - w) / s + 1;
    double t0 = now_ms();                          /* omp_get_wtime() * 1000 */
    for (int64_t i = 0; i < nw; i++) {
        const float* p = x + i * s;
        double mu, sd; float mn, mx;
        window_stats(p, w, &mu, &sd, &mn, &mx);
        score[i] = sd > 1e-12 ? (float)(fabs(p[w - 1] - mu) / sd) : 0.0f;
        if (progress && ((i + 1) & 4095) == 0) progress[0] = i + 1;
    }
    if (progress) progress[0] = nw;
    if (st) { st->windows = nw; st->ms = now_ms() - t0; st->threads = 1; st->status = TW_OK; }
    return TW_OK;
}

/* MAD: copy the window to a buffer, quickselect the median, then the median of |x - m| */
static float median_inplace(float* a, int n);       /* quickselect; mean of two middles if n is even */

int tw_mad(...) {
    float* buf = malloc(2 * w * sizeof(float));     /* buf and dev */
    if (!buf) return TW_E_NOMEM;
    for (int64_t i = 0; i < nw; i++) {
        const float* p = x + i * s;
        memcpy(buf, p, w * sizeof(float));
        float m = median_inplace(buf, w);
        for (int k = 0; k < w; k++) buf[w + k] = fabsf(p[k] - m);
        float mad = median_inplace(buf + w, w);
        score[i] = fabsf(p[w - 1] - m) / (1.4826f * mad + 1e-12f);
    }
    free(buf);
    ...
}

int tw_kmeans_score(const float* F, int64_t nw, int32_t d, const float* C, const float* scale, int32_t k, ...) {
    for (int64_t i = 0; i < nw; i++) {
        float best = INFINITY;
        for (int c = 0; c < k; c++) {
            float acc = 0.f;
            for (int j = 0; j < d; j++) { float t = (F[i*d + j] - C[c*d + j]) * scale[j]; acc += t * t; }
            if (acc < best) best = acc;
        }
        score[i] = sqrtf(best);
    }
}
```

Use `size_t` / `int64_t` for every index that can exceed 2^31 (`i * s`, `i * d`); `int` indexing overflows above about 2 billion elements.

## Python wrapper

```python
class TwStats(C.Structure):
    _fields_ = [("windows", C.c_int64), ("ms", C.c_double), ("threads", C.c_int32), ("status", C.c_int32)]

F32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")

lib.tw_zscore.argtypes = [F32, C.c_int64, C.c_int32, C.c_int32, F32, C.c_int32,
                          C.POINTER(C.c_int64), C.POINTER(TwStats)]
lib.tw_zscore.restype = C.c_int

def zscore(x, w=256, s=1, threads=1, progress=None):
    nw = (len(x) - w) // s + 1
    out = np.empty(nw, np.float32); st = TwStats()
    pp = progress.ctypes.data_as(C.POINTER(C.c_int64)) if progress is not None else None
    rc = lib.tw_zscore(x, len(x), w, s, out, threads, pp, C.byref(st))
    if rc != 0: raise TwError(rc)
    return out, st
```

`TwError` maps `-1` to `ValueError`, `-2` to `MemoryError`, `-3` to `RuntimeError`.

## Files to create

- `sdk/include/threadwatch.h`, `sdk/src/{common.c, features.c, zscore.c, mad.c, kmeans.c, select.c}`
- `python/threadwatch/lib.py` (extended), `python/threadwatch/sdk.py` (wrappers)
- `tests/python/{test_sdk_equivalence.py, test_sdk_errors.py}`
- `docs/t1-results.md` (T_py, T1 and G_lang table)

## Tests and acceptance

- For each kernel, C output is within relative error 1e-5 of the D-03 NumPy output on `syn_1M` and on two NAB series (use `np.allclose(rtol=1e-5, atol=1e-6)`).
- Bad arguments (`w < 2`, `s < 1`, `n < w`, null pointers, `threads < 1`) return `-1` and raise `ValueError` in Python.
- Flags from the C scores and from the NumPy scores give the same F1 (difference below 1e-6).
- `T1` recorded for `syn_1M`, `syn_2M`, `syn_5M` per kernel; `G_lang` computed for 1M and 2M.

## Pitfalls

- Float summation order: a different loop order changes the last bits. Keep the same order in the loops you will later parallelise (one window per thread).
- Never use `-ffast-math`: it lets the compiler reorder sums and breaks determinism and the 1e-5 comparison.
- Pass contiguous arrays only; a sliced NumPy view with a stride makes `ndpointer` raise. Convert with `np.ascontiguousarray`.
- Keep an object reference to arrays passed to the DLL until the call returns.
- Quickselect worst cases: pick a median-of-three pivot, and test on constant and sorted windows (flat metrics are common in monitoring data).

## Changelog line

`D-04: C SDK kernels (threads=1) with ctypes wrapper, equivalence < 1e-5 vs NumPy, T1 and G_lang recorded`
