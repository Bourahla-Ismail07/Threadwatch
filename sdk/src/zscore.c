#include <math.h>
#include <omp.h>
#include "internal.h"

int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s,
              float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    int rc = tw_validate(x, n, w, s, score, threads);
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
        for (int64_t i = 0; i < nw; i++) {
            const float* p = x + i * s;
            double mu, sd;
            float mn, mx;
            tw_window_stats(p, w, &mu, &sd, &mn, &mx);
            score[i] = sd > 1e-12 ? (float)(fabs((double)p[w - 1] - mu) / sd) : 0.0f;
            if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
        }
        if (progress) progress[(int64_t)tid * 8] = done;
    }
    tw_finish(stats, nw, t0, team);
    return TW_OK;
}
