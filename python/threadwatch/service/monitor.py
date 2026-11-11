"""D-07.2 - monitor thread: reads the progress counters, per-core CPU and new flags every 100 ms.

Cost control (spec pitfalls): 10 Hz at most; only the NEW part of the score array is read (the slice each
thread has written since the last poll); at most 50 flags per event; no parsing of large arrays.
"""
from __future__ import annotations

import threading
from time import perf_counter
from typing import Callable, List, Optional, Tuple

import numpy as np

from threadwatch.mpi.partition import partition

MAX_FLAGS = 50
SCAN_CHUNK = 1 << 20


def slot_blocks(n: int, w: int, s: int, ranks: int, threads: int) -> List[Tuple[int, int]]:
    """(first window, number of windows) owned by every counter slot (rank r, thread t) = slot r*threads + t.

    Ranks split the windows as in D-06 (`partition`). Inside a rank, `schedule(static)` gives thread t the
    block that starts at t*q + min(t, r), with q = m // threads, r = m % threads and m = windows of that rank.
    """
    blocks: List[Tuple[int, int]] = []
    for p in partition(n, w, s, ranks):
        m = p.b - p.a
        q, r = divmod(m, threads)
        for t in range(threads):
            blocks.append((p.a + t * q + min(t, r), q + (1 if t < r else 0)))
    return blocks


class Monitor:
    def __init__(self, *, label: str, progress, scores, blocks, nw: int, s: int, w: int, threshold: float,
                 live: bool, emit: Callable[[dict], None], poll_s: float = 0.1, cpu: bool = True):
        self.label, self.progress, self.scores, self.blocks = label, progress, scores, blocks
        self.nw, self.s, self.w, self.threshold, self.live = nw, s, w, threshold, live
        self.emit, self.poll_s, self.cpu = emit, poll_s, cpu
        self.slots = len(blocks)
        self.seen = [0] * self.slots                 # windows already scanned per slot
        self.flags_total = 0
        self.t0 = perf_counter()
        self.last_total, self.last_t = 0, self.t0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.polls = 0

    # ---- lifecycle --------------------------------------------------------
    def start(self, t0: Optional[float] = None) -> None:
        if self.cpu:
            import psutil
            psutil.cpu_percent(percpu=True)          # first call is meaningless: prime the counters
        self.t0 = self.last_t = t0 if t0 is not None else perf_counter()
        self._thread = threading.Thread(target=self._loop, name=f"monitor-{self.label}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop, then emit one final event that scans whatever is left."""
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.poll(final=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll()
            except Exception:                          # a monitoring glitch must never touch the run
                pass

    # ---- one poll ---------------------------------------------------------
    def poll(self, final: bool = False) -> dict:
        counters = np.array(self.progress[::8][: self.slots])        # tiny copy: slots values
        now = perf_counter()
        total = int(counters.sum())
        dt = max(now - self.last_t, 1e-9)
        found_i: List[np.ndarray] = []
        found_s: List[np.ndarray] = []

        if self.live:
            for k, (start, ln) in enumerate(self.blocks):
                cur = ln if (final and self.live) else min(int(counters[k]), ln)
                prev = self.seen[k]
                if cur > prev:
                    seg = np.asarray(self.scores[start + prev: start + cur])
                    idx = np.flatnonzero(seg > self.threshold)
                    if idx.size:
                        found_i.append(idx + start + prev)
                        found_s.append(seg[idx])
                    self.seen[k] = cur
        if final and not self.live:                   # scores became visible only now: scan once, in chunks
            for a in range(0, self.nw, SCAN_CHUNK):
                seg = np.asarray(self.scores[a: a + SCAN_CHUNK])
                idx = np.flatnonzero(seg > self.threshold)
                if idx.size:
                    found_i.append(idx + a)
                    found_s.append(seg[idx])

        flags: list = []
        if found_i:
            ii, ss = np.concatenate(found_i), np.concatenate(found_s)
            self.flags_total += int(ii.size)
            if ii.size > MAX_FLAGS:                    # keep the strongest ones
                top = np.argpartition(ss, -MAX_FLAGS)[-MAX_FLAGS:]
                ii, ss = ii[top], ss[top]
            order = np.argsort(ii)
            # i is the SAMPLE index the window score belongs to: t_i = i*s + w - 1
            flags = [{"i": int(ii[j]) * self.s + self.w - 1, "score": round(float(ss[j]), 3)} for j in order]

        # the final poll covers only the last sliver of time: report the run average instead of a noisy instant
        rate = (total * self.s / max(now - self.t0, 1e-9)) if final else (total - self.last_total) * self.s / dt
        ev = {
            "type": "progress", "t": round(now - self.t0, 3), "mode": self.label,
            "done": total, "total": self.nw,
            "throughputSps": rate,
            "threads": counters.tolist(),
            "cpu": [int(round(c)) for c in self._cpu()],
            "flags": flags, "flagsTotal": self.flags_total,
            "scoresLive": self.live,
        }
        self.last_total, self.last_t = total, now
        self.polls += 1
        self.emit(ev)
        return ev

    def _cpu(self):
        if not self.cpu:
            return []
        import psutil
        return psutil.cpu_percent(percpu=True)
