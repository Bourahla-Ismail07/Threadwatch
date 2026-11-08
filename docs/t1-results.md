# T1 results (D-04, C SDK with 1 thread)

Machine: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel, logical cores 12, physical 8, Python 3.13.2, numpy 2.5.4, scikit-learn 1.9.1.

Window w=256, stride s=1. `T1_ms` is the median of 5 wall-clock runs after one warm-up, measured around the Python call (includes the output allocation); the individual runs are in `runs_ms`. The `kmeans` kernel is the full pipeline (C features + C K-Means score), the same work as `T_py` in [`first-results.md`](first-results.md). `T_py_ms` is read from the D-03 run on this machine; `G_lang = T_py / T1`. Compiled with `-O3`, no `-ffast-math`; one thread. `spread_pct` = (max - min) / median of the runs; rows with `noisy = YES` (spread above 10%) were disturbed by something else on the machine and must be rerun before they are used as T1.

| dataset | kernel | T_py_ms | T1_ms | G_lang | min_ms | spread_pct | noisy |
|---|---|---|---|---|---|---|---|
| syn_1M | features |  | 503.3 |  | 436.0 | 112.0 | YES |
| syn_1M | zscore | 2,140.2 | 434.2 | 4.9 | 431.6 | 1.4 |  |
| syn_1M | mad | 9,153.7 | 6,260.6 | 1.5 | 6,219.4 | 0.8 |  |
| syn_1M | kmeans | 2,840.0 | 453.1 | 6.3 | 448.6 | 8.7 |  |
| syn_2M | features |  | 867.2 |  | 865.9 | 1.2 |  |
| syn_2M | zscore | 3,748.2 | 868.3 | 4.3 | 864.9 | 2.3 |  |
| syn_2M | mad | 15,211.6 | 13,352.0 | 1.1 | 12,269.4 | 15.3 | YES |
| syn_2M | kmeans | 4,910.5 | 930.5 | 5.3 | 900.0 | 5.4 |  |
| syn_5M | features |  | 2,168.3 |  | 2,166.8 | 0.8 |  |
| syn_5M | zscore |  | 2,165.0 |  | 2,163.8 | 0.2 |  |
| syn_5M | mad |  | 31,177.2 |  | 31,158.5 | 1.2 |  |
| syn_5M | kmeans |  | 2,235.9 |  | 2,235.3 | 2.0 |  |

Equivalence with the NumPy reference on the full `syn_1M` (max relative error, tolerance 1e-5; 0 means identical float32 outputs): zscore 0.0e+00, mad 0.0e+00, kmeans 0.0e+00.

Note: syn_1M features (min 436 ms) and syn_2M mad (min 12,269 ms) had one slow run; their minimum is consistent with the linear scaling of the other sizes.
