# State of the art: parallel anomaly detection in server-metric time series

> **DRAFT: before sending, read each source below, check every `[verify]` tag against the paper, fix what differs, then delete this box.** Claims without a tag are either definitions or taken from a page that was actually read while drafting (NAB paper, MERL page). A source you have not read must be removed from the list: the project rule is to cite only what you read.

## 1. Sources

| # | Reference | Role in this project |
|---|---|---|
| [1] | J. Blázquez-García, A. Conde, U. Mori, J. A. Lozano. *A review on outlier/anomaly detection in time series data.* ACM Computing Surveys 54(3), 2021 (arXiv:2002.04236). | Families of detectors, vocabulary |
| [2] | F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* IEEE ICDM 2008, pp. 413-422. | Detector 4 |
| [3] | A. Lavin, S. Ahmad. *Evaluating real-time anomaly detection algorithms: the Numenta Anomaly Benchmark.* IEEE ICMLA 2015 (arXiv:1510.03336). | Real dataset (NAB), evaluation practice |
| [4] | S. Su, Y. Niu, Y. Li, W. Chen, D. Pei. *Robust anomaly detection for multivariate time series through stochastic recurrent neural network (OmniAnomaly).* ACM KDD 2019. | Real dataset (SMD) |
| [5] | A. H. Karp, H. P. Flatt. *Measuring parallel processor performance.* Communications of the ACM 33(5), 1990, pp. 539-543. (Amdahl's law: G. M. Amdahl, AFIPS 1967.) | Speedup, efficiency, serial fraction |
| [6] | J. Zhang, D. Nikovski, T. Nakamura. *GPU-APUMPEDI: a parallel algorithm for computing approximate pan matrix profiles of time series.* Int. Conf. on Time Series and Forecasting, 2023 (MERL TR2023-091). | Recent accelerated detection work |

Parallel-programming background (OpenMP, MPI): add one textbook or survey chapter you actually used, for example a chapter of an introductory parallel-programming book, with its exact reference `[verify]`.

## 2. Synthesis

**Detector families.** Time-series anomaly detection is usually organised by the kind of model that scores a point or window: statistical rules, distance or clustering scores, isolation-based methods, forecasting and deep models `[1; verify the taxonomy and its names]`. The four detectors of this project cover the first three families on purpose: rolling z-score and rolling median/MAD (statistical), K-Means distance (clustering) and Isolation Forest (isolation). Median and MAD are used next to the mean and standard deviation because a few extreme points move the mean and the standard deviation but barely move the median and the MAD.

**Isolation Forest.** The method isolates a point with random axis-parallel splits on sub-samples; points that need fewer splits get a higher anomaly score `[2; verify score definition, default sub-sample size and number of trees]`. Fitting needs only a small sample, and scoring a window is independent of every other window, which is what makes the scoring step parallel.

**NAB.** The corpus holds 58 files of 1,000 to 22,000 instances each, 365,551 points in total (§II.A of [3]). Labels are anomaly windows whose size is 10% of the file length divided by the number of anomalies (§II.B); the NAB score rewards the earliest detection inside a window, penalises false positives and misses, and is normalised between a null and a perfect detector (§II.B-C). The paper requires detectors to run in real time with no look-ahead and without manual tuning (§I, §II.B), yet reports that each algorithm's threshold was tuned over the full dataset (§III). Consequence for Threadwatch: we convert windows to per-sample labels and report point-wise precision, recall, F1 and event recall instead of the NAB score, we fix one threshold from a fit sample reused by every execution mode, and our batch runs do not claim the no-look-ahead streaming property (the live replay mode is the streaming part).

**SMD.** SMD (OmniAnomaly) is a multivariate server-machine dataset: 28 machines with 38 metrics each, with a train and a labelled test part `[4; verify duration, sampling interval, split and how labels were produced]`. We use the test part, one metric per machine, chosen by a written rule (see `docs/data.md`).

**Measuring parallel performance.** Speedup is S(p) = T1/Tp and efficiency is E(p) = S(p)/p. Amdahl's law bounds the speedup by the serial fraction. The Karp-Flatt metric, e = (1/S - 1/p) / (1 - 1/p), estimates that fraction from measured speedups; a constant e points to a fixed serial part and a growing e to growing overhead `[5; verify the formula and the interpretation in the paper]`. The same sources motivate keeping the language effect (Python to C) out of the speedup: T1 is the single-thread C time.

**Accelerated detection.** Recent work on speeding up time-series anomaly discovery concentrates on GPUs and on specific algorithms: [6] approximates pan matrix profiles on GPUs and validates on the UCR Anomaly Archive; its abstract reports no speedup figures and does not name hardware or baseline, so check the full paper `[6; verify what speedups and baselines it reports]`.

## 3. Positioning

To our reading of the sources above, published work compares detector accuracy on shared benchmarks ([3], [4]) or accelerates one specific algorithm, often on GPUs ([6]). Threadwatch adds a controlled benchmark of the same kernels under five execution modes (sequential Python, sequential C, OpenMP, MPI, hybrid) on one Windows machine, where the language effect is separated from the parallel effect, outputs are checked to be identical across thread and process counts, and speedup, efficiency, serial fraction and F1 are reported together for several series sizes. `[Adjust this sentence to what you found when reading.]`

## 4. References

See table in section 1.
