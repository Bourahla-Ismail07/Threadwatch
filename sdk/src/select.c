#include "internal.h"

static void swap_d(double* a, double* b) { double t = *a; *a = *b; *b = t; }

static double median_of_three(double x, double y, double z) {
    if (x < y) return (y < z) ? y : ((x < z) ? z : x);
    return (x < z) ? x : ((y < z) ? z : y);
}

/* k-th smallest (0-based) of a[0..n-1]. Three-way partition (< pivot, == pivot, > pivot):
   flat or repeated-value windows, which are common in monitoring data, stay linear instead
   of degrading to quadratic time. Pivot = median of first, middle and last. */
static double select_kth(double* a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = median_of_three(a[lo], a[lo + (hi - lo) / 2], a[hi]);
        int lt = lo, i = lo, gt = hi;
        while (i <= gt) {
            if (a[i] < pivot)      swap_d(&a[lt++], &a[i++]);
            else if (a[i] > pivot) swap_d(&a[i], &a[gt--]);
            else                   i++;
        }
        if (k < lt)       hi = lt - 1;
        else if (k > gt)  lo = gt + 1;
        else              return pivot;
    }
    return a[lo];
}

/* median; for even n, the mean of the two middle values */
double tw_median(double* a, int n) {
    if (n & 1) return select_kth(a, n, n / 2);
    double lo = select_kth(a, n, n / 2 - 1);
    double hi = select_kth(a, n, n / 2);
    return (lo + hi) * 0.5;
}
