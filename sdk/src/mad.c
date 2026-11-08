#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "internal.h"

int tw_mad(const float* x, int64_t n, int32_t w, int32_t s,
           float* score, int32_t threads, int64_t* progress, tw_stats* stats) {
    int rc = tw_validate(x, n, w, s, score, threads);
    if (rc) return rc;
    int64_t nw = (n - w) / s + 1;
    double* buf = (double*)malloc((size_t)2 * (size_t)w * sizeof(double));   /* window copy + deviations */
    if (!buf) return TW_E_NOMEM;
    double t0 = tw_now_ms();
    for (int64_t i = 0; i < nw; i++) {
        const float* p = x + i * s;
        for (int32_t k = 0; k < w; k++) buf[k] = (double)p[k];
        double med = tw_median(buf, w);
        for (int32_t k = 0; k < w; k++) buf[w + k] = fabs((double)p[k] - med);
        double mad = tw_median(buf + w, w);
        score[i] = (float)(fabs((double)p[w - 1] - med) / (1.4826 * mad + 1e-12));
        if (progress && ((i + 1) & 4095) == 0) progress[0] = i + 1;
    }
    free(buf);
    if (progress) progress[0] = nw;
    tw_finish(stats, nw, t0);
    return TW_OK;
}
