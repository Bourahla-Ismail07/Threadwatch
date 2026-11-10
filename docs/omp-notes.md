# OpenMP notes (D-05)

Machine: 12th Gen Intel(R) Core(TM) i5-12450HX; logical cores 12, physical 8; OpenMP cap (sdk.max_threads) 12; Python 3.13.2, numpy 2.5.4, scikit-learn 1.9.1.

Dataset `syn_5M`, w=256, s=1. Median of 5 runs after one warm-up; T1 is measured in the same session. `wall_ms` is the Python timer around the call (end-to-end); `c_ms` is the parallel region measured inside the DLL. `speedup = T1 / Tp` (wall, medians), `speedup_min` uses the best run of each, `efficiency = speedup / used`. `used` is the team size the DLL reports (requests above the logical core count are capped). `identical` is `np.array_equal` of the output against the 1-thread output on every run. Rows with `noisy = YES` (spread above 10%) were disturbed and should be rerun.

| kernel | threads | used | wall_ms | min_ms | spread_pct | noisy | c_ms | speedup | speedup_min | efficiency | identical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| features | 1 | 1 | 2,243.2 | 2,230.0 | 1.3 |  | 2,240.0 | 1.00 | 1.00 | 1.00 | True |
| features | 2 | 2 | 1,081.3 | 1,080.7 | 0.4 |  | 1,077.0 | 2.07 | 2.06 | 1.04 | True |
| features | 4 | 4 | 545.6 | 542.2 | 20.6 | YES | 543.0 | 4.11 | 4.11 | 1.03 | True |
| features | 8 | 8 | 350.5 | 346.7 | 2.6 |  | 349.0 | 6.40 | 6.43 | 0.80 | True |
| features | 12 | 12 | 261.9 | 258.3 | 1.7 |  | 258.0 | 8.57 | 8.63 | 0.71 | True |
| zscore | 1 | 1 | 2,218.9 | 2,195.4 | 4.5 |  | 2,219.0 | 1.00 | 1.00 | 1.00 | True |
| zscore | 2 | 2 | 1,079.6 | 1,077.2 | 1.5 |  | 1,079.0 | 2.06 | 2.04 | 1.03 | True |
| zscore | 4 | 4 | 568.1 | 545.8 | 8.1 |  | 568.0 | 3.91 | 4.02 | 0.98 | True |
| zscore | 8 | 8 | 366.8 | 351.0 | 7.1 |  | 366.0 | 6.05 | 6.25 | 0.76 | True |
| zscore | 12 | 12 | 262.1 | 259.9 | 3.0 |  | 262.0 | 8.47 | 8.45 | 0.71 | True |
| mad | 1 | 1 | 32,487.1 | 31,874.5 | 3.8 |  | 32,486.0 | 1.00 | 1.00 | 1.00 | True |
| mad | 2 | 2 | 15,391.3 | 14,929.8 | 26.0 | YES | 15,391.0 | 2.11 | 2.13 | 1.06 | True |
| mad | 4 | 4 | 7,732.2 | 7,707.1 | 0.7 |  | 7,731.0 | 4.20 | 4.14 | 1.05 | True |
| mad | 8 | 8 | 4,645.2 | 4,572.0 | 2.6 |  | 4,644.0 | 6.99 | 6.97 | 0.87 | True |
| mad | 12 | 12 | 3,467.6 | 3,464.2 | 1.8 |  | 3,466.0 | 9.37 | 9.20 | 0.78 | True |
| kmeans | 1 | 1 | 78.3 | 75.4 | 4.8 |  | 77.0 | 1.00 | 1.00 | 1.00 | True |
| kmeans | 2 | 2 | 39.3 | 38.8 | 4.7 |  | 38.0 | 1.99 | 1.95 | 1.00 | True |
| kmeans | 4 | 4 | 22.5 | 20.0 | 53.6 | YES | 22.0 | 3.48 | 3.78 | 0.87 | True |
| kmeans | 8 | 8 | 19.7 | 18.1 | 15.3 | YES | 18.0 | 3.97 | 4.16 | 0.50 | True |
| kmeans | 12 | 12 | 14.1 | 14.0 | 28.8 | YES | 13.0 | 5.55 | 5.39 | 0.46 | True |

## Schedule comparison on the MAD kernel (D-05.6)

`static` = equal contiguous blocks; `dynamic` = chunks of 4096 windows. `dynamic_vs_static` below 1 means dynamic is faster.

| threads | schedule | wall_ms | min_ms | spread_pct | identical | dynamic_vs_static |
|---|---|---|---|---|---|---|
| 2 | static | 15,413.8 | 15,393.1 | 0.4 | True |  |
| 2 | dynamic | 15,454.5 | 13,911.2 | 10.3 | True | 1.0 |
| 4 | static | 7,725.8 | 7,714.1 | 0.6 | True |  |
| 4 | dynamic | 7,738.6 | 7,711.4 | 1.2 | True | 1.0 |
| 8 | static | 4,616.1 | 4,579.6 | 3.2 | True |  |
| 8 | dynamic | 4,626.0 | 4,539.2 | 12.2 | True | 1.0 |
| 12 | static | 3,468.6 | 3,040.1 | 13.5 | True |  |
| 12 | dynamic | 3,492.9 | 3,479.3 | 1.5 | True | 1.0 |

## How to read this on a hybrid CPU

Intel 12th gen and newer mix performance cores (P, with hyper-threading) and efficiency cores (E). Logical CPUs 0-7 are the P-core threads and 8-11 the E-cores on a 4P+4E part, so the speedup curve is expected to bend after 4 threads (SMT siblings share a core) and again after 8 (E-cores are slower). Compare the curve with the physical (8) and logical (12) counts above.
