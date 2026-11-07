#ifndef THREADWATCH_H
#define THREADWATCH_H
#include <stdint.h>
#ifdef _WIN32
  #define TW_API __declspec(dllexport)
#else
  #define TW_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
TW_API int tw_version(void);
TW_API int tw_max_threads(void);
TW_API int tw_hello_threads(int32_t threads, int32_t* seen);  /* counts threads that actually ran */
#ifdef __cplusplus
}
#endif
#endif
