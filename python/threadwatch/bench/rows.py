"""D-08.3 - the result row and the append-only results file.

The raw file is append-only and written one row at a time (flush + fsync), so an interruption loses at most the
run in flight. Nothing derived is stored here (speedup, efficiency, ... are computed at analysis time).
The only edit ever made to existing rows is `status` -> `superseded` when a cell is rerun (see rewrite_status).
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

COLUMNS = ["run_id", "timestamp", "dataset", "n", "w", "s", "detector", "mode", "q_ranks", "p_threads", "repeat",
           "load_s", "features_s", "score_s", "comm_s", "total_s", "dll_ms", "throughput_sps",
           "f1", "precision", "recall", "event_recall", "identical", "status", "note"]

OK, WARMUP, TIMEOUT, ERROR, CRASH, SUPERSEDED = "ok", "warmup", "timeout", "error", "crash", "superseded"
FAILURES = (TIMEOUT, ERROR, CRASH)          # a cell with one of these is closed: its remaining repeats are not attempted
_ID = re.compile(r"^(?P<cell>.+)#r(?P<rep>\d+)(?:@(?P<batch>\d+))?$")


def run_id(cell_key: str, repeat: int, batch: int = 0) -> str:
    return f"{cell_key}#r{repeat}" + (f"@{batch}" if batch else "")


def parse_run_id(rid: str) -> Tuple[str, int, int]:
    m = _ID.match(rid)
    if not m:
        raise ValueError(f"bad run_id {rid!r}")
    return m["cell"], int(m["rep"]), int(m["batch"] or 0)


def _fmt(v):
    """CSV text of a value. NumPy scalars are converted to Python ones first: in NumPy 2, repr(np.float64(1.0)) is the
    text 'np.float64(1.0)', which must never reach the file."""
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "" if v != v else repr(v)             # NaN -> empty
    return v


def make_row(cell, repeat: int, status: str, batch: int = 0, note: str = "", **fields) -> Dict[str, object]:
    """One row. `cell` is a bench.matrix.Cell; unknown field names are an error (no silent column drift)."""
    bad = set(fields) - set(COLUMNS)
    if bad:
        raise KeyError(f"unknown result columns: {sorted(bad)}")
    row: Dict[str, object] = {c: None for c in COLUMNS}
    row.update(run_id=run_id(cell.key, repeat, batch), timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               dataset=cell.dataset, n=cell.n, w=cell.w, s=cell.s, detector=cell.detector, mode=cell.mode,
               q_ranks=cell.q, p_threads=cell.p, repeat=repeat, status=status, note=note)
    row.update(fields)
    if row.get("total_s") and row.get("throughput_sps") is None and status in (OK, WARMUP):
        row["throughput_sps"] = cell.n / float(row["total_s"])
    return row


class ResultsFile:
    def __init__(self, path):
        self.path = Path(path)

    # ---- writing -----------------------------------------------------------
    def append(self, row: Dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="raise")
            if new:
                w.writeheader()
            w.writerow({k: _fmt(row.get(k)) for k in COLUMNS})
            f.flush()
            os.fsync(f.fileno())

    # ---- reading -----------------------------------------------------------
    def rows(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def done(self) -> Set[Tuple[str, int, int]]:
        """(cell_key, repeat, batch) of every recorded run, whatever its status: resuming never repeats a row."""
        return {parse_run_id(r["run_id"]) for r in self.rows()}

    def closed(self) -> Set[Tuple[str, int]]:
        """(cell_key, batch) of cells that ended in timeout / error / crash: not retried automatically."""
        out = set()
        for r in self.rows():
            if r["status"] in FAILURES:
                c, _, b = parse_run_id(r["run_id"])
                out.add((c, b))
        return out

    def next_batch(self, cell_key: str) -> int:
        batches = [parse_run_id(r["run_id"])[2] for r in self.rows() if parse_run_id(r["run_id"])[0] == cell_key]
        return (max(batches) + 1) if batches else 0

    # ---- the one permitted edit ----------------------------------------------
    def rewrite_status(self, run_ids: Iterable[str], status: str, note_suffix: str, backup_dir) -> int:
        """Set `status` (and append to `note`) of the given runs. The previous file is kept in backup_dir first."""
        ids = set(run_ids)
        if not ids or not self.path.exists():
            return 0
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, backup_dir / f"results.{time.strftime('%Y%m%d-%H%M%S')}.csv")
        rows = self.rows()
        n = 0
        for r in rows:
            if r["run_id"] in ids:
                r["status"] = status
                r["note"] = (r["note"] + "; " if r["note"] else "") + note_suffix
                n += 1
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)                          # atomic: the file is never half written
        return n
