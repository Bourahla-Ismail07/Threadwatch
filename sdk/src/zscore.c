#include <math.h>
#include "internal.h"

int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s,
              float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    int rc = tw_validate(x, n, w, s, score, threads);
    if (rc) return rc;
    int64_t nw = (n - w) / s + 1;
    double t0 = tw_now_ms();
    for (int64_t i = 0; i < nw; i++) {
        const float* p = x + i * s;
        double mu, sd;
        float mn, mx;
        tw_window_stats(p, w, &mu, &sd, &mn, &mx);
        score[i] = sd > 1e-12 ? (float)(fabs((double)p[w - 1] - mu) / sd) : 0.0f;
        if (progress && ((i + 1) & 4095) == 0) progress[0] = i + 1;
    }
    if (progress) progress[0] = nw;
    tw_finish(stats, nw, t0);
    return TW_OK;
}
