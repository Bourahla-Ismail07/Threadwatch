#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include "internal.h"

/* one window; buf holds 2*w doubles (private to the calling thread) */
static float mad_one_window(const float* p, int32_t w, double* buf) {
    for (int32_t k = 0; k < w; k++) buf[k] = (double)p[k];
    double med = tw_median(buf, w);
    for (int32_t k = 0; k < w; k++) buf[w + k] = fabs((double)p[k] - med);
    double mad = tw_median(buf + w, w);
    return (float)(fabs((double)p[w - 1] - med) / (1.4826 * mad + 1e-12));
}

int tw_mad_ex(const float* x, int64_t n, int32_t w, int32_t s,
              float* score, int32_t threads, int64_t* progress, tw_stats* stats, int32_t schedule) {
    int rc = tw_validate(x, n, w, s, score, threads);
    if (rc) return rc;
    if (schedule != TW_SCHED_STATIC && schedule != TW_SCHED_DYNAMIC) return TW_E_ARG;
    int64_t nw = (n - w) / s + 1;
    int32_t nthreads = tw_effective_threads(threads), team = nthreads;
    int failed = 0;
    double t0 = tw_now_ms();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t done = 0;
        if (tid == 0) team = omp_get_num_threads();
        double* buf = (double*)malloc((size_t)2 * (size_t)w * sizeof(double));   /* one scratch buffer per thread */
        if (!buf) {
            #pragma omp atomic write
            failed = 1;
        }
        if (schedule == TW_SCHED_STATIC) {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < nw; i++) {
                if (buf) score[i] = mad_one_window(x + i * s, w, buf);
                if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
            }
        } else {
            #pragma omp for schedule(dynamic, 4096)
            for (int64_t i = 0; i < nw; i++) {
                if (buf) score[i] = mad_one_window(x + i * s, w, buf);
                if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
            }
        }
        if (progress) progress[(int64_t)tid * 8] = done;
        free(buf);
    }
    if (failed) return TW_E_NOMEM;
    tw_finish(stats, nw, t0, team);
    return TW_OK;
}

int tw_mad(const float* x, int64_t n, int32_t w, int32_t s,
           float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    return tw_mad_ex(x, n, w, s, score, threads, progress, stats, TW_SCHED_STATIC);
}
