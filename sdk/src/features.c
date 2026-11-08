#include "internal.h"

int tw_window_features(const float* x, int64_t n, int32_t w, int32_t s,
                       float* features, int32_t threads, int64_t* progress, tw_stats* stats) {
    int rc = tw_validate(x, n, w, s, features, threads);
    if (rc) return rc;
    int64_t nw = (n - w) / s + 1;
    double t0 = tw_now_ms();
    for (int64_t i = 0; i < nw; i++) {
        double mu, sd;
        float mn, mx;
        tw_window_stats(x + i * s, w, &mu, &sd, &mn, &mx);
        float* f = features + i * 4;
        f[0] = (float)mu;
        f[1] = (float)sd;
        f[2] = mn;
        f[3] = mx;
        if (progress && ((i + 1) & 4095) == 0) progress[0] = i + 1;
    }
    if (progress) progress[0] = nw;
    tw_finish(stats, nw, t0);
    return TW_OK;
}
