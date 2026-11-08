# First results (D-03, Python sequential baseline)

Machine: Intel64 Family 6 Model 151 Stepping 2, GenuineIntel, logical cores 12, physical 8, Python 3.13.2, numpy 2.5.4, scikit-learn 1.9.1.

Window w=256, stride s=1, threshold = 1 - rho quantile of the scores of a fixed 100,000-window fit sample (rho = 0.001, seed 0). Models are fitted once, outside the timed region. `elapsed_ms` is the median of the runs listed in `runs_ms` (see the CSV). For `kmeans` and `isolation_forest` the time includes feature extraction. Everything runs on one thread (`n_jobs=1`). `elapsed_ms` of `syn_1M` and `syn_2M` is the language baseline T_py for D-04.

| dataset | detector | precision | recall | f1 | event_recall | elapsed_ms |
|---|---|---|---|---|---|---|
| syn_1M | zscore | 0.0179 | 0.0161 | 0.0169 | 0.6111 | 2,140.2 |
| syn_1M | mad | 0.0291 | 0.0237 | 0.0261 | 0.6111 | 9,153.7 |
| syn_1M | kmeans | 0.3990 | 0.3896 | 0.3942 | 0.2778 | 2,840.0 |
| syn_1M | isolation_forest | 0.0000 | 0.0000 | 0.0000 | 0.0556 | 5,734.4 |
| syn_2M | zscore | 0.0240 | 0.0233 | 0.0237 | 0.6667 | 3,748.2 |
| syn_2M | mad | 0.0432 | 0.0438 | 0.0435 | 0.5185 | 15,211.6 |
| syn_2M | kmeans | 0.6617 | 0.5240 | 0.5849 | 0.2963 | 4,910.5 |
| syn_2M | isolation_forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 9,582.5 |
| nab_ec2_cpu_utilization_24ae8d | zscore | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6.6 |
| nab_ec2_cpu_utilization_24ae8d | mad | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 26.7 |
| nab_ec2_cpu_utilization_24ae8d | kmeans | 1.0000 | 0.0149 | 0.0294 | 1.0000 | 8.4 |
| nab_ec2_cpu_utilization_24ae8d | isolation_forest | 1.0000 | 0.0174 | 0.0342 | 1.0000 | 22.4 |
| nab_ec2_cpu_utilization_5f5533 | zscore | 0.7500 | 0.0075 | 0.0148 | 1.0000 | 6.8 |
| nab_ec2_cpu_utilization_5f5533 | mad | 0.2500 | 0.0025 | 0.0049 | 0.5000 | 31.9 |
| nab_ec2_cpu_utilization_5f5533 | kmeans | 1.0000 | 0.0100 | 0.0197 | 0.5000 | 8.5 |
| nab_ec2_cpu_utilization_5f5533 | isolation_forest | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 23.1 |

Labels are point-wise; a window is placed at the index of its last sample. Series with no labelled anomaly have undefined F1 (reported 0).