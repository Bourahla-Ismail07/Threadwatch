# D-06 - MPI and hybrid driver

| | |
|---|---|
| **Teacher's item** | Programme parallèle (completes D-05) |
| **Depends on** | D-05 |
| **Unblocks** | D-08 |

## What this part is responsible for

Running the same DLL across several processes: MPI-q (q ranks, 1 thread each) and HYB-q×p (q ranks, p threads each). The driver splits the series with the overlap rule, gathers the scores, and separates computation time from communication time, so the final comparison between OpenMP, MPI and hybrid is fair.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| MS-MPI (`mpiexec`) | starts and connects the processes |
| `mpi4py` | scatter, gather, barrier, reduce from Python |
| `ctypes` + DLL | every rank calls the same kernels with its own `threads` |
| NumPy (`np.memmap`) | file mode: each rank maps the same input file |
| pytest | equivalence tests (run through `mpiexec`) |

## Tasks

| Task | What |
|---|---|
| D-06.1 | Window partition and overlap computation (pure function, unit tested) |
| D-06.2 | File mode driver: each rank maps the file and processes its block |
| D-06.3 | Scatter mode driver: rank 0 reads and sends slices with `Scatterv` |
| D-06.4 | Gather scores on rank 0; timers with barriers; per-phase timing |
| D-06.5 | Hybrid: `threads=p` per rank; check `q × p` against the core count |
| D-06.6 | Equivalence test against C-1 for q = 2 and 4, including boundary windows |

## Pseudo code

```python
def partition(n, w, s, q):
    """windows [a_r, b_r) per rank, balanced; samples read [lo_r, hi_r)"""
    nw = (n - w) // s + 1
    base, rem = divmod(nw, q)
    out, a = [], 0
    for r in range(q):
        b = a + base + (1 if r < rem else 0)
        lo, hi = a * s, (b - 1) * s + w          # hi exclusive; overlap with the neighbour is w - s
        out.append((a, b, lo, hi)); a = b
    return out

def run(mode, path, w, s, threads, detector):
    comm = MPI.COMM_WORLD; r, q = comm.Get_rank(), comm.Get_size()
    parts = partition(n, w, s, q)
    a, b, lo, hi = parts[r]

    comm.Barrier(); t_all = perf_counter()

    t = perf_counter()
    if mode == "file":
        x = np.memmap(path, np.float32, "r")[lo:hi]                  # no data travels
    else:                                                            # scatter
        counts = [h - l for (_, _, l, h) in parts]
        displs = [l for (_, _, l, _) in parts]                       # overlapping displacements: legal on the send side
        x = np.empty(hi - lo, np.float32)
        comm.Scatterv([full, counts, displs, MPI.FLOAT] if r == 0 else None, x, root=0)
    t_in = perf_counter() - t

    t = perf_counter()
    score, st = sdk.compute(detector, np.ascontiguousarray(x), w, s, threads)   # same DLL call as OMP-p
    t_cmp = perf_counter() - t

    t = perf_counter()
    counts_w = [b_ - a_ for (a_, b_, _, _) in parts]
    full_score = np.empty(sum(counts_w), np.float32) if r == 0 else None
    comm.Gatherv(score, [full_score, counts_w, None, MPI.FLOAT] if r == 0 else None, root=0)
    t_out = perf_counter() - t

    comm.Barrier(); t_total = perf_counter() - t_all
    worst = comm.reduce(t_cmp, op=MPI.MAX, root=0)                   # slowest rank sets the pace
    return {"in_s": t_in, "compute_s": worst, "gather_s": t_out, "total_s": t_total}, full_score
```

Launch: `mpiexec -n 4 python -m threadwatch.mpi.run --mode file --dataset syn_5M --detector mad --threads 1` for MPI-4, and `--threads 2` with `-n 2` for HYB-2×2.

Report, for every MPI cell: input time, compute time (slowest rank), gather time, total.

## Files to create

- `python/threadwatch/mpi/{partition.py, driver.py, run.py}`
- `tests/python/test_partition.py` (plain pytest), `tests/mpi/test_mpi_equivalence.py` (launched with `mpiexec`)
- `docs/mpi-notes.md` (communication versus computation table)

## Tests and acceptance

- `partition` covers every window exactly once, with no gaps, for several `(n, w, s, q)` including `q > nw`.
- Gathered MPI output equals the C-1 output exactly for q = 2 and 4, on file mode and scatter mode, including the windows at chunk boundaries.
- Hybrid runs with `q × p` up to the logical core count give the same output.
- Communication and computation times are reported separately.
- With one rank (`-n 1`), the driver reproduces the C-1 time within 5%.

## Pitfalls

- **Overlap off by one:** a window at a chunk edge needs samples from the neighbour. The test on boundary windows is the guard.
- **Slowest rank:** report the maximum compute time over ranks, not rank 0's time.
- **Oversubscription:** `q × p` above the logical cores slows everything; MPI processes and OpenMP threads share the same cores on one machine.
- **Memory:** scatter mode holds the full series on rank 0 plus every slice; file mode does not.
- **Process start cost:** `mpiexec` and Python start-up take hundreds of milliseconds; time inside the program after a barrier, not around the launch.
- **Gatherv on Windows with large arrays:** counts above 2^31 elements overflow `int`; keep sizes below that or split the gather.
- **Honest expectation:** on one machine all ranks share the same memory bandwidth, so MPI is not expected to beat OpenMP (H4).

## Changelog line

`D-06: MPI file and scatter drivers with overlap rule, hybrid runs, output identical to C-1`
