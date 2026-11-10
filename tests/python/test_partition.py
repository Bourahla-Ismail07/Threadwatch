"""D-06.1 - partition covers every window exactly once, with the right overlap (plain pytest, no MPI)."""
import pytest

from threadwatch.mpi.partition import num_windows, overlap_samples, partition

CASES = [
    (n, w, s, q)
    for n, w, s in [(1000, 256, 1), (1000, 256, 7), (999, 32, 5), (300, 256, 1), (256, 256, 1),
                    (50, 10, 10), (50, 10, 25), (1_000_000, 256, 1), (5_000_003, 256, 3)]
    for q in (1, 2, 3, 4, 7, 8, 64)
]


@pytest.mark.parametrize("n,w,s,q", CASES)
def test_every_window_exactly_once(n, w, s, q):
    parts = partition(n, w, s, q)
    nw = num_windows(n, w, s)
    assert len(parts) == q
    assert parts[0].a == 0 and parts[-1].b == nw          # no gap at either end
    for p, nxt in zip(parts, parts[1:]):
        assert p.b == nxt.a                                # contiguous, no gap, no overlap of windows
    assert sum(p.b - p.a for p in parts) == nw
    sizes = [p.b - p.a for p in parts]
    assert max(sizes) - min(sizes) <= 1                    # balanced
    assert sizes == sorted(sizes, reverse=True)            # larger blocks first (remainder to low ranks)


@pytest.mark.parametrize("n,w,s,q", CASES)
def test_slices_hold_complete_windows(n, w, s, q):
    for p in partition(n, w, s, q):
        if p.b == p.a:
            assert p.lo == p.hi
            continue
        assert 0 <= p.lo < p.hi <= n                      # inside the series
        assert p.lo == p.a * s
        assert p.hi == (p.b - 1) * s + w                  # last window of the block ends exactly at hi
        assert p.n_samples == (p.b - p.a - 1) * s + w
        # window i of the block starts at i*s - lo inside the slice and ends inside it
        for i in (p.a, p.b - 1):
            assert i * s >= p.lo and i * s + w <= p.hi


@pytest.mark.parametrize("n,w,s,q", [c for c in CASES if c[2] < c[1]])
def test_neighbour_overlap_is_w_minus_s(n, w, s, q):
    parts = [p for p in partition(n, w, s, q) if p.b > p.a]
    for p, nxt in zip(parts, parts[1:]):
        assert p.hi - nxt.lo == w - s == overlap_samples(w, s)


@pytest.mark.parametrize("n,w,s,q", [(300, 256, 1, 64), (50, 10, 10, 8), (256, 256, 1, 4)])
def test_more_ranks_than_windows(n, w, s, q):
    nw = num_windows(n, w, s)
    assert q > nw
    parts = partition(n, w, s, q)
    assert [p.b - p.a for p in parts] == [1] * nw + [0] * (q - nw)
    for p in parts[nw:]:
        assert p.lo == p.hi <= n


def test_window_to_rank_map_matches_brute_force():
    n, w, s, q = 2003, 64, 5, 6
    owner = {}
    for r, p in enumerate(partition(n, w, s, q)):
        for i in range(p.a, p.b):
            assert i not in owner
            owner[i] = r
    assert sorted(owner) == list(range(num_windows(n, w, s)))


@pytest.mark.parametrize("args", [(100, 1, 1, 2), (100, 10, 0, 2), (5, 10, 1, 2), (100, 10, 1, 0)])
def test_bad_arguments(args):
    with pytest.raises(ValueError):
        partition(*args)
