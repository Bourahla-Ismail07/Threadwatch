#include <omp.h>
#include "internal.h"

int tw_window_features(const float* x, int64_t n, int32_t w, int32_t s,
                       float* features, int32_t threads, int64_t* progress, tw_stats* stats) {
    int rc = tw_validate(x, n, w, s, features, threads);
    if (rc) return rc;
    int64_t nw = (n - w) / s + 1;
    int32_t nthreads = tw_effective_threads(threads), team = nthreads;
    double t0 = tw_now_ms();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t done = 0;
        if (tid == 0) team = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < nw; i++) {                 /* one window, one thread, fixed order */
            double mu, sd;
            float mn, mx;
            tw_window_stats(x + i * s, w, &mu, &sd, &mn, &mx);
            float* f = features + i * 4;                   /* one writer per output slot */
            f[0] = (float)mu;
            f[1] = (float)sd;
            f[2] = mn;
            f[3] = mx;
            if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
        }
        if (progress) progress[(int64_t)tid * 8] = done;
    }
    tw_finish(stats, nw, t0, team);
    return TW_OK;
}
