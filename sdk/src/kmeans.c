#include <math.h>
#include <omp.h>
#include "internal.h"

int tw_kmeans_score(const float* features, int64_t nw, int32_t d,
                    const float* centroids, const float* scale, int32_t k,
                    float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    if (!features || !centroids || !scale || !score || nw < 0 || d < 1 || k < 1 || threads < 1)
        return TW_E_ARG;
    int32_t nthreads = tw_effective_threads(threads), team = nthreads;
    double t0 = tw_now_ms();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t done = 0;
        if (tid == 0) team = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < nw; i++) {
            const float* f = features + i * (int64_t)d;
            float best = INFINITY;
            for (int32_t c = 0; c < k; c++) {
                const float* cen = centroids + (int64_t)c * d;
                float acc = 0.0f;
                for (int32_t j = 0; j < d; j++) {
                    float t = (f[j] - cen[j]) * scale[j];
                    acc += t * t;
                }
                if (acc < best) best = acc;
            }
            score[i] = sqrtf(best);
            if (progress && ((++done) & 4095) == 0) progress[(int64_t)tid * 8] = done;
        }
        if (progress) progress[(int64_t)tid * 8] = done;
    }
    tw_finish(stats, nw, t0, team);
    return TW_OK;
}
