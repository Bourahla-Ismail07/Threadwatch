# D-08 - Benchmark campaign

| | |
|---|---|
| **Teacher's item** | Tableaux de résultats |
| **Depends on** | D-05, D-06 |
| **Unblocks** | D-09 |

## What this part is responsible for

Running every configuration under the same protocol, automatically, and recording one row per run in a raw results file. No interpretation happens here: the output is the data, the machine description, and summary tables.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| Python harness | builds the matrix, runs each cell, records timings |
| `ctypes` + DLL | C-1 and OMP-p cells (in-process) |
| `mpiexec` + driver (D-06) | MPI and hybrid cells (one process launch per cell) |
| scikit-learn | Isolation Forest scoring cells (`n_jobs` as the thread parameter) |
| psutil, `platform` | machine description |
| pandas | results table and summary tables |

## Tasks

| Task | What |
|---|---|
| D-08.1 | Machine record (`machine.json`) |
| D-08.2 | Matrix definition and runner (resumable) |
| D-08.3 | Timing phases and result row |
| D-08.4 | Quality columns (F1 etc.) in every row, and the equivalence check against C-1 |
| D-08.5 | Summary tables (median, spread, speedup, efficiency, throughput) |

## Experiment matrix

| Factor | Levels |
|---|---|
| Series size | 1M, 2M, 5M, 10M, 20M samples (50M optional) |
| Detector | z-score, MAD, K-Means (C kernels); Isolation Forest (features in C, scoring in scikit-learn) |
| Mode | PY-SEQ, C-1, OMP (2, 4, 8, … up to logical cores), MPI (2, 4), HYB (2×2, 2×4) |
| Window | w = 256, s = 1 (second stride tested on one size) |
| Input mode (MPI) | file; scatter on two sizes |

Limits for cost: PY-SEQ up to 2M, Isolation Forest up to 2M by default, MAD in Python up to 1M. Larger Python and Isolation Forest sizes are optional extras.

## Protocol

1. Idle machine, power plan on high performance, laptop on mains power, other applications closed.
2. Same input file for every mode (`np.memmap`, pages warmed once before timing).
3. For each cell: 1 warm-up run (discarded), then **at least 5 timed runs**, interleaving modes rather than running all runs of one mode back to back.
4. Record the median, minimum, maximum and the interquartile range.
5. Timers: `perf_counter` around the whole call (end to end) and the DLL's own `ms`.
6. Phases: `load_s`, `features_s`, `score_s`, `comm_s` (MPI), `total_s`.
7. Library threads fixed: scikit-learn `n_jobs` per cell, `threadpoolctl` limits for NumPy.
8. After every cell, compare flags with the C-1 flags; store `identical` (bool) and `f1`.
9. Never delete a failed or odd run: store it with `status` and a note.

## Result row (columns of `results.csv`)

`run_id, timestamp, dataset, n, w, s, detector, mode, q_ranks, p_threads, repeat, load_s, features_s, score_s, comm_s, total_s, dll_ms, throughput_sps, f1, precision, recall, event_recall, identical, status, note`

## Pseudo code

```python
def run_matrix(matrix, out_csv):
    done = set(read_done_ids(out_csv))                       # resumable
    for cell in interleave(matrix.cells(), repeats=5 + 1):   # +1 warm-up
        if cell.id in done: continue
        row = run_cell(cell)                                 # in-process or mpiexec subprocess
        append_row(out_csv, row)

def run_cell(cell):
    if cell.mode in ("MPI", "HYB"):
        proc = subprocess.run(["mpiexec", "-n", str(cell.q), "python", "-m", "threadwatch.mpi.run", *cell.args],
                              capture_output=True, text=True, timeout=cell.timeout)
        return parse_row(proc.stdout, cell)
    t = perf_counter(); score, st = sdk.compute(cell.detector, cell.x, cell.w, cell.s, cell.p); total = perf_counter() - t
    flags = score > cell.threshold
    return make_row(cell, total, st, evaluate(flags, cell.labels), identical=np.array_equal(score, cell.ref_score))
```

Derived columns are computed at analysis time (D-09), never stored in the raw file: speedup `T1 / Tp` (median based), efficiency, `G_lang`, Karp-Flatt `e`.

## Files to create

- `experiments/{run_matrix.py, matrix.yaml, machine_info.py, summarize.py}`
- `python/threadwatch/bench/{matrix.py, runner.py, rows.py}`
- `experiments/out/{results.csv, machine.json}`
- `experiments/out/summary_*.csv` (one table per detector: size × mode, median seconds)
- `docs/perf.md` (the summary tables, formatted)

## Tests and acceptance

- Every cell of the matrix has at least 5 timed runs; any missing cell is listed with the reason.
- `identical == True` for every OMP, MPI and HYB row of the deterministic kernels (z-score, MAD, K-Means).
- The raw CSV and `machine.json` are committed; the whole campaign restarts from one command and resumes after an interruption.
- The summary tables are generated from the CSV by a script.
- Coefficient of variation of the timed runs is reported per cell; cells above 10% are flagged for a rerun.

## Pitfalls

- **Thermal throttling:** long campaigns heat a laptop and change timings. Interleave modes, take breaks between sizes, and look for a downward drift of repeated C-1 runs; rerun if it appears.
- **Background activity:** antivirus scans, updates and browsers disturb timings. Close them; keep the dashboard off.
- **Cold file cache:** the first read of a memmap is slower. The warm-up run handles it; note it in the protocol.
- **Timeouts:** give each cell a timeout and record `status = timeout` instead of hanging.
- **Mixed units:** store seconds in the CSV and convert only for display.
- **Cherry-picking:** report every run; if a cell is rerun, keep both and mark the first as superseded.

## Changelog line

`D-08: benchmark campaign (matrix, protocol, raw results.csv, machine.json, summary tables)`
