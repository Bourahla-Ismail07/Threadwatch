"""D-06.2 to D-06.5 - MPI / hybrid driver.

Every rank calls the same DLL kernels as C-1 and OMP-p, on its own slice, with
its own `threads`. Rank 0 gathers the scores. Computation, input and gather
times are measured separately, between barriers.

MPI-q    : q ranks, threads = 1
HYB-qxp  : q ranks, threads = p
"""
from __future__ import annotations

import os
from time import perf_counter

import numpy as np

from .partition import partition

MODES = ("file", "scatter")
DETECTORS = ("zscore", "mad", "kmeans")
INT_MAX = 2**31 - 1          # MPI counts are C ints (see pitfall: Gatherv on Windows)


# --------------------------------------------------------------------------
# the one place that touches the SDK wrapper (python/threadwatch/sdk.py, D-04/D-05)
# --------------------------------------------------------------------------
def sdk_compute(detector, x, w, s, threads, model=None):
    """Same DLL call as C-1 / OMP-p. Returns (score float32, dll_ms).

    zscore, mad : sdk.zscore / sdk.mad(x, w, s, threads)  -> (scores, stats)
    kmeans      : C features, then C K-Means score; `model` = (centroids, scale)
    """
    from threadwatch import sdk                       # imported late: partition.py stays MPI/SDK free

    if detector == "zscore":
        score, st = sdk.zscore(x, w=w, s=s, threads=threads)
        return score, float(st.ms)
    if detector == "mad":
        score, st = sdk.mad(x, w=w, s=s, threads=threads)
        return score, float(st.ms)
    if detector == "kmeans":
        if model is None:
            raise ValueError("kmeans needs model=(centroids, scale)")
        centroids, scale = model
        feats, st1 = sdk.window_features(x, w=w, s=s, threads=threads)
        score, st2 = sdk.kmeans_score(feats, centroids, scale, threads=threads)
        return score, float(st1.ms) + float(st2.ms)
    raise ValueError(f"unknown detector {detector!r}")


# --------------------------------------------------------------------------
def logical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def check_threads(q: int, p: int, logical: int | None = None, allow_oversubscribe: bool = False) -> str | None:
    """D-06.5: q x p must stay within the logical cores (ranks and threads share them).

    Returns a warning text if oversubscribed and allowed, None otherwise.
    """
    if q < 1 or p < 1:
        raise ValueError("q and p must be >= 1")
    logical = logical or logical_cores()
    if q * p <= logical:
        return None
    msg = f"oversubscription: q x p = {q}x{p} = {q * p} > {logical} logical cores"
    if not allow_oversubscribe:
        raise ValueError(msg + " (pass allow_oversubscribe=True / --allow-oversubscribe to force)")
    return msg


def file_length(path: str) -> int:
    size = os.path.getsize(path)
    if size % 4:
        raise ValueError(f"{path}: size {size} is not a multiple of 4 (float32)")
    return size // 4


def run(mode, path, w, s, threads, detector, comm=None, model=None, compute=None):
    """Run one detector over the series in `path` on all ranks of `comm`.

    Returns (timings, scores). Both are None on ranks other than 0.

    timings (seconds): in_s, compute_s, gather_s, total_s, each the MAXIMUM over ranks
    (the slowest rank sets the pace), plus comm_s (= in_s for scatter, 0 for file, + gather_s),
    dll_ms (max over ranks), compute_by_rank (list) and parts (the partition used).
    """
    from mpi4py import MPI

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}")
    compute = compute or sdk_compute
    comm = comm or MPI.COMM_WORLD
    r, q = comm.Get_rank(), comm.Get_size()

    # n is agreed before timing starts (rank 0 knows the file; scatter does not need a shared file system)
    n = comm.bcast(file_length(path) if r == 0 else None, root=0)
    parts = partition(n, w, s, q)
    a, b, lo, hi = parts[r]
    counts_w = [p.b - p.a for p in parts]
    counts_x = [p.hi - p.lo for p in parts]
    if max(counts_x) > INT_MAX or sum(counts_w) > INT_MAX:
        raise ValueError("a slice or the gathered output exceeds 2^31 elements; use more ranks or split the gather")

    full = None
    if mode == "scatter" and r == 0:
        full = np.memmap(path, np.float32, "r")          # rank 0 reads the file (not timed: only the send is)
        # NB: scatter holds the full series on rank 0 plus every slice (see Pitfalls)

    comm.Barrier()
    t_all = perf_counter()

    # ---- input ---------------------------------------------------------
    t = perf_counter()
    if mode == "file":
        x = np.memmap(path, np.float32, "r")[lo:hi]      # no data travels; pages are read lazily
    else:
        displs = [p.lo for p in parts]                   # overlapping displacements: legal on the send side
        x = np.empty(hi - lo, np.float32)
        sendbuf = [full, counts_x, displs, MPI.FLOAT] if r == 0 else None
        comm.Scatterv(sendbuf, x, root=0)
    t_in = perf_counter() - t

    # ---- compute (same DLL call as OMP-p, own `threads`) ----------------
    t = perf_counter()
    if b > a:
        score, dll_ms = compute(detector, np.ascontiguousarray(x), w, s, threads, model)
        score = np.ascontiguousarray(score, dtype=np.float32)
        if len(score) != b - a:
            raise RuntimeError(f"rank {r}: kernel returned {len(score)} scores, expected {b - a}")
    else:                                                # q > nw: nothing to do on this rank
        score, dll_ms = np.empty(0, np.float32), 0.0
    t_cmp = perf_counter() - t

    # ---- gather on rank 0 ----------------------------------------------
    t = perf_counter()
    full_score = np.empty(sum(counts_w), np.float32) if r == 0 else None
    comm.Gatherv(score, [full_score, counts_w, None, MPI.FLOAT] if r == 0 else None, root=0)
    t_out = perf_counter() - t

    comm.Barrier()
    t_total = perf_counter() - t_all

    per_rank = comm.gather((t_in, t_cmp, t_out, t_total, dll_ms), root=0)
    if r != 0:
        return None, None
    in_s, cmp_s, out_s, tot_s, dll = (max(v) for v in zip(*per_rank))
    timings = {
        "in_s": in_s,
        "compute_s": cmp_s,                              # slowest rank, not rank 0
        "gather_s": out_s,
        "comm_s": (in_s if mode == "scatter" else 0.0) + out_s,
        "total_s": tot_s,
        "dll_ms": dll,
        "compute_by_rank": [v[1] for v in per_rank],
        "parts": [tuple(p) for p in parts],
    }
    return timings, full_score
