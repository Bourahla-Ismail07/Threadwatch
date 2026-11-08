#ifndef THREADWATCH_H
#define THREADWATCH_H
#include <stdint.h>

#ifdef _WIN32
  #define TW_API __declspec(dllexport)
#else
  #define TW_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- D-01: hello checks ---- */
TW_API int tw_version(void);
TW_API int tw_max_threads(void);
TW_API int tw_hello_threads(int32_t threads, int32_t* seen);   /* counts threads that actually ran */

/* ---- D-04: detection kernels ---- */
typedef struct { int64_t windows; double ms; int32_t threads; int32_t status; } tw_stats;

#define TW_OK          0
#define TW_E_ARG      -1
#define TW_E_NOMEM    -2
#define TW_E_INTERNAL -3

/* progress: optional int64[threads * 8]; the counter of thread t is at index t * 8; NULL = no progress.
   threads must be >= 1; values above 1 are accepted and ignored until D-05 (the ABI stays the same). */
TW_API int tw_window_features(const float* x, int64_t n, int32_t w, int32_t s,
                              float* features, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_zscore(const float* x, int64_t n, int32_t w, int32_t s,
                     float* score, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_mad(const float* x, int64_t n, int32_t w, int32_t s,
                  float* score, int32_t threads, int64_t* progress, tw_stats* stats);
TW_API int tw_kmeans_score(const float* features, int64_t nw, int32_t d,
                           const float* centroids, const float* scale, int32_t k,
                           float* score, int32_t threads, int64_t* progress, tw_stats* stats);

#ifdef __cplusplus
}
#endif
#endif
