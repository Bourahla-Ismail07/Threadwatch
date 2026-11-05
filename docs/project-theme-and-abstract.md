# Threadwatch — Parallel Anomaly Detection for Server Metrics

> **One-liner:** Detect anomalies in very long server-metric series with a C SDK (DLL), then measure how OpenMP, MPI and hybrid parallelism cut the detection time, whether the detections stay identical, and when parallelism stops paying off.

## 1. Why this theme

| Course / skill area | Where it lives in Threadwatch |
|---|---|
| **HPC** | OpenMP threads, MPI processes, hybrid runs, speedup, efficiency, scalability, Amdahl and Karp-Flatt analysis |
| **AI / ML** | Isolation Forest and K-Means distance detectors, fitted with scikit-learn and scored at scale |
| **Data analysis** | Real server-metric datasets, a labelled synthetic generator, F1 and event recall |
| **Systems engineering** | A C library with a stable ABI (DLL) called from Python, tested for determinism |
| **Web** | A small live dashboard (HTML, CSS, JavaScript) that shows each approach running |
| **Market fit** | Monitoring and AIOps: every software company needs fast anomaly detection on metrics (CPU, latency, request rate) |

Course link: *« Accélération des algorithmes d'intelligence artificielle par le calcul parallèle et distribué »*. The project follows **Project 4** (time-series anomaly detection) using computer-system metrics, a data source Project 4 explicitly allows. It also covers the OpenMP, MPI and hybrid comparison that Project 1 asks for.

## 2. Research question

> For which series sizes and thread or process counts does parallel anomaly detection pay off, and does it keep the detection quality identical?

### Hypotheses (all falsifiable — results are reported either way)

- **H1 (OpenMP):** kernels with O(w) work per window (direct z-score, rolling median/MAD) keep parallel efficiency above 70% up to the physical core count; the optional O(1)-per-window sliding-sum variant saturates earlier, because each window does little work per byte read.
- **H2 (equivalence):** parallel outputs are identical to the 1-thread outputs, so F1 does not change with the number of threads or ranks.
- **H3 (size):** below a threshold series size, parallel overhead makes a run slower than 1 thread; the crossover size is measured.
- **H4 (MPI):** on a single machine MPI does not beat OpenMP, because messages cost time; hybrid helps only when ranks × threads is tuned.

## 3. What is new (stated honestly)

Not a new algorithm. The contribution is the **controlled comparison**:

1. One C kernel set called by five execution modes (Python, C-1, OpenMP, MPI, hybrid), so any difference comes from how the work is run.
2. The **language effect** (Python to C) reported separately from the **parallel effect** (C-1 to C-p), so speedup is not inflated.
3. Library-level parallelism (scikit-learn `n_jobs`) compared with our own OpenMP code on the same data.
4. A **live view** (per-thread lanes, per-core load, throughput) that makes load imbalance and saturation visible.

Before writing the report, do a short literature check (D-02) to position the work against existing studies of parallel anomaly detection and adjust the claims to what you find.

## 4. Abstract (draft — fill the `[ ]` after D-08)

Monitoring systems must detect anomalies in server metrics fast enough to keep up with the data they produce. We present Threadwatch, a C software development kit (a Windows DLL) that computes rolling window features and four anomaly detectors (rolling z-score, rolling median/MAD, K-Means distance and Isolation Forest) and exposes them to Python through a flat ABI. We run the same kernels sequentially in Python, sequentially in C, with OpenMP, with MPI and with a hybrid MPI+OpenMP scheme, on real server-metric series (NAB, SMD) and on labelled synthetic series from `[N_min]` to `[N_max]` samples. On `[machine]`, OpenMP reached a speedup of `[S]` with `[p]` threads (efficiency `[E]`) on the MAD kernel and `[S2]` on the z-score kernel; moving from Python to C gave a further `[G]` gain. Parallel outputs were `[identical / within e]` to sequential ones, so F1 stayed at `[F1]`. Parallelism became beneficial above `[n*]` samples. MPI on one node `[did / did not]` outperform OpenMP; the hybrid scheme reached `[S3]` with `[q x p]`. A live dashboard shows per-thread progress and per-core load during execution. We discuss limits: one machine, a few datasets, and statistical and tree-based detectors only.

## 5. Scope

**In scope:** univariate series (one metric per run, several metrics as an extension); a single Windows machine for OpenMP, MPI and hybrid runs; public datasets plus a labelled synthetic generator; offline batch detection plus a live replay in the dashboard.

**Out of scope:** production monitoring claims, GPU acceleration, multi-machine clusters, Spark, deep-learning detectors, energy measurement.

## 6. Data sources (verify sizes, licences and citations when downloading)

- **Numenta Anomaly Benchmark (NAB):** about 58 labelled series, including AWS CloudWatch server metrics. Used for detection quality on real data.
- **Server Machine Dataset (SMD):** 28 machines, 38 metrics each, 1-minute sampling, labelled anomalies. Used for longer realistic series, one metric at a time.
- **Synthetic generator (D-02):** trend + daily and weekly cycles + AR(1) noise, with labelled spikes, level shifts and variance bursts. Used for scaling tests and exact ground truth.

## 7. Success metrics

| Area | Metric | Target |
|---|---|---|
| Correctness | C output vs NumPy reference | relative error below 1e-5 |
| Determinism | parallel output vs 1-thread output | identical for every thread count |
| Quality | precision, recall, F1, event recall | reported per detector; unchanged by parallelism |
| Speed | speedup and efficiency, 1 to max threads | reported for every size, with Karp-Flatt serial fraction |
| Language effect | Python time ÷ C-1 time | reported separately from parallel speedup |
| MPI | gathered output vs C-1 output | identical, including boundary windows |
| Dashboard | slowdown from live monitoring | below 2% against a headless run |
| Delivery | every D-nn merged, tested, changelogged, sent | 100% |
