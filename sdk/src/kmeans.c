#include <math.h>
#include "internal.h"

int tw_kmeans_score(const float* features, int64_t nw, int32_t d,
                    const float* centroids, const float* scale, int32_t k,
                    float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    if (!features || !centroids || !scale || !score || nw < 0 || d < 1 || k < 1 || threads < 1)
        return TW_E_ARG;
    double t0 = tw_now_ms();
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
        if (progress && ((i + 1) & 4095) == 0) progress[0] = i + 1;
    }
    if (progress) progress[0] = nw;
    tw_finish(stats, nw, t0);
    return TW_OK;
}
