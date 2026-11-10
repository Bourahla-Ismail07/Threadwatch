#ifndef TW_INTERNAL_H
#define TW_INTERNAL_H
#include "threadwatch.h"

double tw_now_ms(void);                                   /* omp_get_wtime() * 1000 */
int    tw_validate(const float* x, int64_t n, int32_t w, int32_t s, const void* out, int32_t threads);
void   tw_finish(tw_stats* st, int64_t windows, double start_ms, int32_t threads);
int32_t tw_effective_threads(int32_t requested);          /* min(requested, logical processors) */
void   tw_window_stats(const float* p, int32_t w, double* mean, double* sd, float* mn, float* mx);
double tw_median(double* a, int n);                       /* quickselect, reorders a */
#endif
