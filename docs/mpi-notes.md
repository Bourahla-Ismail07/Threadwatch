# MPI and hybrid notes (D-06)

## What runs

`mpiexec -n q python -m threadwatch.mpi.run ...` runs the same DLL kernels as C-1 and OMP-p on `q` processes.

| Label | Launch | Meaning |
|---|---|---|
| MPI-q | `-n q --threads 1` | q ranks, 1 thread each |
| HYB-qxp | `-n q --threads p` | q ranks, p threads each; `q x p` must stay within the logical cores (the driver refuses otherwise unless `--allow-oversubscribe`, and then flags the row) |

Input modes: `file` (every rank memory-maps the same file; only the scores travel) and `scatter` (rank 0 sends each slice with `Scatterv`; the input transfer is communication time).

## Partition and overlap rule

nw = (n - w) / s + 1 windows are split over q ranks in blocks that differ by at most one window (the remainder goes to the lowest ranks). Rank r owns windows `[a_r, b_r)` and reads samples `[a_r*s, (b_r-1)*s + w)`. Neighbouring slices share `w - s` samples (255 for w = 256, s = 1), so every window is complete on one rank and its score is computed from the same values as in C-1. If q > nw the last ranks own no window and skip the kernel.

## Timing phases

All measured inside the program, between two barriers, so `mpiexec` and Python start-up are excluded. Each reported value is the **maximum over ranks**.

| Column | Meaning |
|---|---|
| `load_s` (`in_s`) | input phase: slice mapping (file) or `Scatterv` (scatter) |
| `score_s` (`compute_s`) | the DLL call; slowest rank |
| `gather_s` | `Gatherv` of the scores to rank 0 |
| `comm_s` | gather, plus the input phase in scatter mode (0 input in file mode) |
| `total_s` | barrier to barrier |

In file mode the slice is a lazy memory map: the operating system reads the pages during the first pass of the kernel, so that cost sits inside `compute_s`, exactly as it does for C-1 and OMP-p on a memory-mapped input. Warm the file once before timing (D-08 protocol).

## Communication versus computation table

Generate it on the target machine (idle, high performance power plan):

```
python experiments/mpi_quick_table.py --dataset syn_5M --detector mad > mpi_table.md
```

and paste the output here. **No numbers are committed until they come from that run.**

| run | input | in_s | compute_s | gather_s | comm_s | total_s | comm share |
|---|---|---|---|---|---|---|---|
| _to be filled_ | | | | | | | |

## How to test

```
pytest tests/python/test_partition.py
mpiexec -n 1 python -m pytest tests/mpi -q -p no:cacheprovider
mpiexec -n 2 python -m pytest tests/mpi -q -p no:cacheprovider
mpiexec -n 4 python -m pytest tests/mpi -q -p no:cacheprovider
```

Each rank runs pytest; results are agreed across ranks so a failure on one rank fails all. The first test checks that `COMM_WORLD` has the size the launcher announced, which catches an `mpi4py` built against a different MPI than `mpiexec`. Output is compared with `np.array_equal` (exact) against the whole-series, 1-thread C-1 result.

## Honest expectations

On one machine all ranks share the same memory bandwidth and cores, so MPI is not expected to beat OpenMP (H4). Scatter mode additionally holds the full series on rank 0 plus every slice. Report every case where MPI or hybrid is slower.
