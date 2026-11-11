"""D-06.6 - gathered MPI output equals the C-1 output exactly.

Launch (every rank runs pytest; assertions are agreed across ranks so none hangs):

    mpiexec -n 2 python -m pytest tests/mpi -q -p no:cacheprovider
    mpiexec -n 4 python -m pytest tests/mpi -q -p no:cacheprovider

Covers file and scatter mode, several (w, s), threads per rank 1 and 2 (hybrid),
the windows on both sides of every chunk border, constant stretches (MAD = 0),
and more ranks than windows.
"""
import os
import tempfile

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from threadwatch.mpi.driver import run, sdk_compute  # noqa: E402
from threadwatch.mpi.partition import num_windows, partition  # noqa: E402

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


def test_world_matches_launcher():
    """Guard: if mpi4py is built against another MPI than mpiexec, every process is a 1-rank world and
    the equivalence tests below would pass without testing anything."""
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKS"):
        if var in os.environ:
            assert SIZE == int(os.environ[var]), f"{var}={os.environ[var]} but COMM_WORLD size is {SIZE}"
            return
    pytest.skip("launcher does not export the world size")


def make_series(n, seed=7):
    """noise + cycle + spikes + a flat stretch + a sorted ramp (quickselect worst cases)."""
    rng = np.random.default_rng(seed)
    x = 50 + 10 * np.sin(np.arange(n) / 37.0) + rng.normal(0, 1, n)
    x[n // 5] += 9
    if n >= 1000:                                        # long series only: the tiny q > nw case skips these
        x[n // 2: n // 2 + 120] = 42.0                   # constant: MAD = 0, std = 0
        x[3 * n // 4: 3 * n // 4 + 200] = np.linspace(0, 1, 200)   # sorted window content
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def series_file():
    path = None
    if RANK == 0:
        fd, path = tempfile.mkstemp(suffix=".f32")
        os.close(fd)
        make_series(6003).tofile(path)
    path = COMM.bcast(path, root=0)
    yield path
    COMM.Barrier()
    if RANK == 0:
        os.remove(path)


def agree(ok):
    """True only if every rank says True, so all ranks pass or fail together."""
    return COMM.allreduce(bool(ok), op=MPI.LAND)


def model():
    rng = np.random.default_rng(1)
    return rng.normal(50, 10, (8, 4)).astype(np.float32), np.full(4, 0.5, np.float32)


def c1_reference(path, detector, w, s):
    x = np.fromfile(path, np.float32)
    score, _ = sdk_compute(detector, x, w, s, 1, model())          # C-1: whole series, 1 thread, no MPI
    return score


@pytest.mark.parametrize("mode", ["file", "scatter"])
@pytest.mark.parametrize("detector", ["zscore", "mad", "kmeans"])
@pytest.mark.parametrize("w,s", [(32, 1), (32, 7), (64, 64), (16, 40)])
@pytest.mark.parametrize("threads", [1, 2])
def test_gathered_output_equals_c1(series_file, mode, detector, w, s, threads):
    timings, got = run(mode, series_file, w, s, threads, detector, comm=COMM, model=model())
    ok, msg = True, ""
    if RANK == 0:
        ref = c1_reference(series_file, detector, w, s)
        ok = np.array_equal(got, ref)                                # exact, not allclose
        if not ok:
            bad = np.flatnonzero(got != ref)
            msg = f"{len(bad)} windows differ, first at {bad[:5]}"
        # windows on both sides of every chunk border, checked explicitly
        for p in partition(len(np.fromfile(series_file, np.float32)), w, s, SIZE):
            for i in range(max(p.a - 2, 0), min(p.a + 2, len(ref))):
                if got[i] != ref[i]:
                    ok, msg = False, msg + f" | border window {i} differs"
        # timings are reported separately and are consistent
        ok = ok and all(timings[k] >= 0 for k in ("in_s", "compute_s", "gather_s", "total_s"))
        ok = ok and timings["total_s"] >= timings["compute_s"]
        ok = ok and (timings["comm_s"] >= timings["gather_s"])
    assert agree(ok), msg or "mismatch on another rank"


@pytest.mark.parametrize("mode", ["file", "scatter"])
def test_more_ranks_than_windows(tmp_path_factory, mode):
    """n = w + 1 gives 2 windows; ranks beyond the second get nothing and must not hang or corrupt."""
    w, s = 32, 1
    path = None
    if RANK == 0:
        d = tmp_path_factory.mktemp("tiny")
        path = str(d / "tiny.f32")
        make_series(w + 1).tofile(path)
    path = COMM.bcast(path, root=0)
    timings, got = run(mode, path, w, s, 1, "zscore", comm=COMM)
    ok = True
    if RANK == 0:
        ok = num_windows(w + 1, w, s) == 2 and np.array_equal(got, c1_reference(path, "zscore", w, s))
    assert agree(ok)


def test_one_rank_partition_is_identity(series_file):
    n = len(np.fromfile(series_file, np.float32)) if RANK == 0 else None
    n = COMM.bcast(n, root=0)
    (p,) = partition(n, 32, 1, 1)
    assert (p.a, p.b, p.lo, p.hi) == (0, num_windows(n, 32, 1), 0, n)


@pytest.mark.parametrize("mode", ["file", "scatter"])
@pytest.mark.parametrize("threads", [1, 2])
def test_shared_progress_and_scores(series_file, tmp_path_factory, mode, threads):
    """D-07 hooks: counters land in the right slots and add up to nw; the shared score buffer holds the result."""
    w, s = 32, 3
    n = COMM.bcast(len(np.fromfile(series_file, np.float32)) if RANK == 0 else None, root=0)
    nw = num_windows(n, w, s)
    base = COMM.bcast(str(tmp_path_factory.mktemp("shared")) if RANK == 0 else None, root=0)
    pf, sf = f"{base}/progress.bin", f"{base}/scores.f32"
    if RANK == 0:
        np.memmap(pf, np.int64, "w+", shape=(SIZE * threads * 8,)).flush()
        np.memmap(sf, np.float32, "w+", shape=(nw,)).flush()
    COMM.Barrier()
    progress, out = np.memmap(pf, np.int64, "r+"), np.memmap(sf, np.float32, "r+")
    _, got = run(mode, series_file, w, s, threads, "mad", comm=COMM, progress=progress, out=out)
    ok = True
    if RANK == 0:
        ref = c1_reference(series_file, "mad", w, s)
        counters = np.memmap(pf, np.int64, "r")[::8]
        ok = int(counters.sum()) == nw and np.array_equal(np.asarray(out), ref) and np.array_equal(got, ref)
        # each rank only writes slots of its own block (slot index r*threads + t)
        for r, p in enumerate(partition(n, w, s, SIZE)):
            block = counters[r * threads:(r + 1) * threads]
            ok = ok and int(block.sum()) == p.b - p.a
    assert agree(ok)
