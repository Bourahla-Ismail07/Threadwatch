# Running the benchmark campaign (D-08)

```
pip install pyyaml pandas psutil scikit-learn threadpoolctl joblib
python experiments/machine_info.py            # check the pre-flight warnings first
python experiments/run_matrix.py --dry-run    # cells, runs, cost bound, missing data files
python experiments/run_matrix.py              # the campaign; Ctrl-C or a power cut is fine, run it again to resume
python experiments/summarize.py               # experiments/out/summary_*.csv, coverage.csv, drift.csv, docs/perf.md
python experiments/run_matrix.py --rerun-noisy   # then summarize again
```

Before starting: plug in the laptop, High performance power plan, close other applications, keep the dashboard off.
Try a small slice first: `--datasets syn_1M --detectors zscore --modes C-1,OMP-4,MPI-2`.

## How it works

* **Blocks.** One block = one (dataset, detector, window, stride). The modes of a block are interleaved round by round:
  a warm-up round, then the timed rounds, with the order rotated each round. Blocks run one after another with a pause.
* **Rows.** One row per run in `results.csv`, appended and fsynced at once. The warm-up is kept (`status = warmup`,
  `repeat = 0`) and excluded from statistics. Timed runs are `repeat 1..5`, `status = ok`.
* **Failures are rows.** `timeout`, `error`, `crash`. A failed cell is closed (its remaining repeats are not attempted, and a
  restart does not retry it); the summary lists it under *Coverage* with the reason. MPI cells are killed as a process tree.
  In-process cells have a watchdog and the harness runs under a supervisor that restarts it and records crashes.
* **Equivalence.** Every run is compared with the C-1 output of its block (`identical`). For MPI runs the comparison is done
  by the rank driver (`--reference`); only a differing result is shipped back to compute F1 from the MPI scores themselves.
* **Quality.** `f1, precision, recall, event_recall` in every row, with one threshold per (dataset, detector, window): the
  (1 - rho) quantile of the scores of a fixed 100,000-window sample (seed 0), stored in `thresholds.json` with the models.
* **Timing.** `total_s` is end to end including mapping the input, the same interval as the MPI driver's barrier-to-barrier
  time. `load_s`, `features_s`, `score_s`, `comm_s` are phases; `dll_ms` is the DLL's own time (empty for PY-SEQ).
  For K-Means in MPI/HYB the feature and score steps are fused inside the rank kernel, so `features_s` is empty (noted in the row).
* **Mode names in the CSV.** `PY-SEQ, C-1, OMP, MPI, HYB`, and **`MPI-scatter`** for the scatter input mode (the spec's column
  list has no input-mode column, and scatter must not be averaged with file mode). D-09 should group by `mode, q_ranks, p_threads`.
* **Reruns.** `--rerun-noisy` finds cells above the CV limit, marks their rows `superseded` (the old file is saved in
  `experiments/out/backups/`), and runs them again as batch 1 (`run_id` ends in `@1`). Nothing is deleted.
* **Derived numbers** (median, IQR, CV, speedup, efficiency, throughput) exist only in the summaries, never in the raw file.

## Not applicable by design (listed in the matrix, not "missing")

PY-SEQ beyond the cost limits (MAD above 1M, others above 2M); Isolation Forest above 2M and any MPI/HYB Isolation Forest
(scikit-learn parallelism is library-level); scatter input except on the two listed sizes; OMP/MPI/HYB shapes above the
logical core count.

## Assumptions to check against your code

* SDK wrapper names used by `CBackend` (`runner.py`): `sdk.zscore/mad(x, w=, s=, threads=)`, `sdk.window_features(...)`,
  `sdk.kmeans_score(F, centroids, scale, threads=)`, each returning `(array, stats)` with `stats.ms`.
* D-03 modules: `detectors.{zscore,mad,features,ml}` (`zscore_py`, `mad_py`, `features_py`, `fit_models`, `kmeans_score_py`)
  and `metrics.evaluate(flags, labels, w, s)` returning `f1, precision, recall, event_recall`.
* Isolation Forest score = `-score_samples` (higher = more anomalous), used for both the threshold and the runs, so it is
  self-consistent. The fit sample is drawn here (uniform, seed 0); if D-03 draws its sample differently, F1 may differ slightly
  from `first-results.md` while staying identical across modes.
* The cost bound and timeouts use the per-million-sample rates in `matrix.yaml` (from `t1-results.md` and `first-results.md`;
  the Isolation Forest C-1 rate is an estimate: replace it with your measured value).
