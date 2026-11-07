"""Labelled synthetic server-metric generator (trend + two cycles + AR(1) noise + anomalies).

Differences from the pseudo-code in the D-02 README (documented in docs/data.md):
  * events are drawn until the labelled samples reach rate * n, so the anomaly rate is met
    (the pseudo-code's "average length 50" gives about 1.7x the requested rate);
  * events never overlap and keep a gap, so labels are exact;
  * the series is produced in blocks of BLOCK samples with a continuing phase and AR state,
    and written straight to disk, so peak memory stays small.
The same (n, seed) always gives the same bytes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .loader import DATA_ROOT, _paths, write_meta

GENERATOR_VERSION = "1.0"
BLOCK = 5_000_000
SIZES = {"syn_1M": 1_000_000, "syn_2M": 2_000_000, "syn_5M": 5_000_000,
         "syn_10M": 10_000_000, "syn_20M": 20_000_000, "syn_50M": 50_000_000}
KINDS = ("spike", "level", "variance")
_P_KIND = (0.5, 0.3, 0.2)
_GAP = 200


def _draw_events(n, seed, rate):
    """Return [(start, length, kind, sign, amp, event_seed)] sorted by start, non-overlapping."""
    rng = np.random.default_rng([seed, 1])
    target = int(round(n * rate))
    events, used = [], 0
    guard = 0
    while used < target and guard < 100_000:
        guard += 1
        s = int(rng.integers(1000, n - 1000))
        kind = KINDS[int(rng.choice(3, p=_P_KIND))]
        if kind == "spike":
            length = 1
        else:
            length = int(rng.integers(30, 301))
        sign = int(rng.choice([-1, 1]))
        amp = float(rng.uniform(6, 12) if kind == "spike" else rng.uniform(4, 8))
        eseed = int(rng.integers(0, 2**31 - 1))
        if s + length > n - 1000:
            continue
        if any(s < e[0] + e[1] + _GAP and e[0] < s + length + _GAP for e in events):
            continue
        events.append((s, length, kind, sign, amp, eseed))
        used += length
    return sorted(events)


def _apply_events(x, y, lo, events, seed):
    hi = lo + len(x)
    for s, length, kind, sign, amp, eseed in events:
        a, b = max(s, lo), min(s + length, hi)
        if a >= b:
            continue
        if kind == "spike":
            x[a - lo:b - lo] += sign * amp
        elif kind == "level":
            x[a - lo:b - lo] += sign * amp
        else:  # variance burst: noise drawn for the whole event, sliced to this block
            full = np.random.default_rng(eseed).normal(0, 4, size=length)
            x[a - lo:b - lo] += full[a - s:b - s]
        y[a - lo:b - lo] = 1


def generate_to_disk(name, n, seed=7, period_day=1440, rate=0.001, phi=0.7, sigma=1.0,
                     root=DATA_ROOT, block=BLOCK):
    """Stream a series of n samples to <root>/<name>.{f32,labels.u8,json}. Returns the metadata."""
    import pathlib
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    fx, fy, _ = _paths(name, root)
    events = _draw_events(n, seed, rate)
    rng = np.random.default_rng([seed, 0])
    zi = np.zeros(1)
    n_anom = 0
    with open(fx, "wb") as fox, open(fy, "wb") as foy:
        for lo in range(0, n, block):
            m = min(block, n - lo)
            t = np.arange(lo, lo + m, dtype=np.float64)           # continuing phase
            base = 50 + 0.00001 * t
            base += 10 * np.sin(2 * np.pi * t / period_day)
            base += 4 * np.sin(2 * np.pi * t / (7 * period_day))
            noise, zi = lfilter([1.0], [1.0, -phi], rng.normal(0, sigma, m), zi=zi)
            x = base + noise
            y = np.zeros(m, dtype=np.uint8)
            _apply_events(x, y, lo, events, seed)
            x.astype("<f4").tofile(fox)
            y.tofile(foy)
            n_anom += int(y.sum())
    meta = {"source": "synthetic", "sampling": "1 min (period_day=%d)" % period_day,
            "seed": seed, "generator_version": GENERATOR_VERSION, "rate_requested": rate,
            "rate_actual": n_anom / n, "n_events": len(events), "phi": phi, "sigma": sigma}
    write_meta(name, n, meta, root)
    return meta


def generate(n, seed=7, **kw):
    """In-memory version for small n (tests, plots). Returns (float32 series, uint8 labels)."""
    import tempfile
    from .loader import load_series, release
    with tempfile.TemporaryDirectory() as d:
        generate_to_disk("tmp", n, seed=seed, root=d, **kw)
        x, y, _ = load_series("tmp", d)
        out = np.array(x), np.array(y)
        release(x, y)
    return out
