"""D-06.1 - window partition and overlap rule (pure functions, no MPI import).

Rank r handles windows [a_r, b_r) and reads samples [lo_r, hi_r) with
    lo_r = a_r * s,   hi_r = (b_r - 1) * s + w      (hi exclusive).
Every window then lies entirely inside one rank's slice, so its score is
computed from the same w values as in the sequential run. Two neighbouring
slices share max(w - s, 0) samples.
"""
from __future__ import annotations

from typing import List, NamedTuple


class Part(NamedTuple):
    a: int    # first window (inclusive)
    b: int    # last window (exclusive)
    lo: int   # first sample read (inclusive)
    hi: int   # last sample read (exclusive)

    @property
    def n_windows(self) -> int:
        return self.b - self.a

    @property
    def n_samples(self) -> int:
        return self.hi - self.lo


def num_windows(n: int, w: int, s: int) -> int:
    """nw = floor((n - w) / s) + 1, as in the sequential kernels."""
    _check(n, w, s)
    return (n - w) // s + 1


def overlap_samples(w: int, s: int) -> int:
    """Samples shared by two neighbouring slices."""
    return max(w - s, 0)


def partition(n: int, w: int, s: int, q: int) -> List[Part]:
    """Balanced split of the nw windows over q ranks (sizes differ by at most 1).

    When q > nw the last q - nw ranks get no window: their Part is empty
    (a == b, lo == hi) and they must skip the kernel call.
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    nw = num_windows(n, w, s)
    base, rem = divmod(nw, q)
    out: List[Part] = []
    a = 0
    for r in range(q):
        b = a + base + (1 if r < rem else 0)
        if b == a:
            lo = hi = min(a * s, n)                         # empty slice
        else:
            lo, hi = a * s, (b - 1) * s + w
        out.append(Part(a, b, lo, hi))
        a = b
    return out


def _check(n: int, w: int, s: int) -> None:
    # same argument rules as the C kernels (TW_E_ARG)
    if w < 2 or s < 1 or n < w:
        raise ValueError(f"bad arguments: n={n}, w={w}, s={s} (need w >= 2, s >= 1, n >= w)")
