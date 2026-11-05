#include <omp.h>
#include "threadwatch.h"

int tw_version(void) { return 100; }          /* 1.0.0 */
int tw_max_threads(void) { return omp_get_max_threads(); }
int tw_hello_threads(int32_t threads, int32_t* seen) {
    if (!seen || threads < 1) return -1;
    int count = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp atomic
        count++;
    }
    *seen = count;
    return 0;
}
