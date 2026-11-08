#include <math.h>
#include <omp.h>
#include "internal.h"

/* ---- D-01 ---- */
int tw_version(void) { return 100; }          /* 1.0.0 */
int tw_max_threads(void) { return omp_get_max_threads(); }
int tw_hello_threads(int32_t threads, int32_t* seen) {
    if (!seen || threads < 1) return -1;
    int count = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp atomic
        count++;
    }
    *seen = count;
    return 0;
}

/* ---- D-04 helpers (shared by the kernels) ---- */
double tw_now_ms(void) { return omp_get_wtime() * 1000.0; }

int tw_validate(const float* x, int64_t n, int32_t w, int32_t s, const void* out, int32_t threads) {
    if (!x || !out || w < 2 || s < 1 || n < w || threads < 1) return TW_E_ARG;
    return TW_OK;
}

void tw_finish(tw_stats* st, int64_t windows, double start_ms) {
    if (st) {
        st->windows = windows;
        st->ms = tw_now_ms() - start_ms;
        st->threads = 1;
        st->status = TW_OK;
    }
}

/* one window, double accumulation: mean, sample std (ddof = 1), min, max */
void tw_window_stats(const float* p, int32_t w, double* mean, double* sd, float* mn, float* mx) {
    double sum = 0.0;
    float lo = p[0], hi = p[0];
    for (int32_t k = 0; k < w; k++) {
        sum += (double)p[k];
        if (p[k] < lo) lo = p[k];
        if (p[k] > hi) hi = p[k];
    }
    double mu = sum / w, ss = 0.0;
    for (int32_t k = 0; k < w; k++) {
        double d = (double)p[k] - mu;
        ss += d * d;
    }
    *mean = mu;
    *sd = sqrt(ss / (w - 1));
    *mn = lo;
    *mx = hi;
}
