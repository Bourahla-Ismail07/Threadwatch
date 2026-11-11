"""D-07.1 - run manager: one run at a time, progress memmap, worker thread, event log.

A run is independent of any browser: the browser only reads the event log, so closing the tab never stops
or slows a run (a reconnecting client resumes from `Last-Event-ID`).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import subprocess
import sys
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from threadwatch.mpi.driver import accepts_out, logical_cores, sdk_compute
from .monitor import Monitor, slot_blocks

DETECTORS = ("zscore", "mad", "kmeans")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


# ---------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Settings:
    data_root: Path = Path("data/series")
    runs_root: Path = Path("runs")
    results_csv: Path = Path("experiments/out/results.csv")
    ui_dir: Path = Path("apps/dashboard")
    w: int = 256
    s: int = 1
    rho: float = 0.001                 # expected anomaly rate: threshold = (1 - rho) quantile of sample scores
    thresh_windows: int = 100_000      # same sample size as the D-03 fit sample
    poll_s: float = 0.1                # 10 Hz, never faster
    max_runs: int = 20                 # stored finished runs (event logs); run directories are deleted on completion
    mpiexec: str = "mpiexec"
    strip_columns: int = 800


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    detector: Literal["zscore", "mad", "kmeans"] = "mad"
    mode: Literal["PY-SEQ", "C-1", "OMP", "MPI", "HYB"]
    threads: int = Field(1, ge=1, le=256)
    ranks: int = Field(1, ge=1, le=256)
    allowOversubscribe: bool = False
    inputMode: Literal["file", "scatter"] = "file"      # MPI/HYB only: how the ranks get their slice (D-06)

    @model_validator(mode="after")
    def _consistent(self):
        m, p, q = self.mode, self.threads, self.ranks
        bad = ((m in ("PY-SEQ", "C-1") and (p != 1 or q != 1)) or (m == "OMP" and q != 1)
               or (m == "MPI" and p != 1) or (m == "HYB" and q < 2 and p < 2))
        if bad:
            raise ValueError(f"mode {m} does not match ranks={q}, threads={p}")
        if self.inputMode == "scatter" and q == 1:
            raise ValueError("inputMode scatter needs ranks >= 2")
        return self

    @property
    def label(self) -> str:
        return {"PY-SEQ": "PY-SEQ", "C-1": "C-1", "OMP": f"OMP-{self.threads}", "MPI": f"MPI-{self.ranks}",
                "HYB": f"HYB-{self.ranks}\u00d7{self.threads}"}[self.mode]


def request_from_token(token: str, dataset: str, detector: str, allow_oversub: bool = False) -> RunRequest:
    """'C-1', 'OMP-4', 'MPI-4', 'HYB-2x4' (or '2×4') -> RunRequest."""
    t = token.strip().upper().replace("\u00d7", "X")
    if t in ("PY-SEQ", "C-1"):
        kw = dict(mode=t, threads=1, ranks=1)
    elif m := re.fullmatch(r"OMP-(\d+)", t):
        kw = dict(mode="OMP", threads=int(m[1]), ranks=1)
    elif m := re.fullmatch(r"MPI-(\d+)", t):
        kw = dict(mode="MPI", ranks=int(m[1]), threads=1)
    elif m := re.fullmatch(r"HYB-(\d+)X(\d+)", t):
        kw = dict(mode="HYB", ranks=int(m[1]), threads=int(m[2]))
    else:
        raise ApiError(400, "bad_mode", f"unknown mode {token!r} (use PY-SEQ, C-1, OMP-p, MPI-q, HYB-qxp)")
    return RunRequest(dataset=dataset, detector=detector, allowOversubscribe=allow_oversub, **kw)


# ---------------------------------------------------------------------------
class EventLog:
    """Append-only event list with blocking reads; any number of readers, late readers replay from the start."""

    def __init__(self):
        self._ev: List[dict] = []
        self._cv = threading.Condition()
        self._closed = False

    def append(self, ev: dict) -> None:
        with self._cv:
            self._ev.append(ev)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def read(self, start: int, timeout: float) -> Tuple[List[dict], bool]:
        with self._cv:
            if len(self._ev) <= start and not self._closed:
                self._cv.wait(timeout)
            return self._ev[start:], self._closed

    def __len__(self):
        return len(self._ev)


# ---------------------------------------------------------------------------
class SdkEngine:
    """C kernels through the SDK wrapper (C-1, OMP-p). One place to adapt if the wrapper changes."""
    name = "sdk"

    def live_scores(self, detector: str) -> bool:
        from threadwatch import sdk
        fn = sdk.kmeans_score if detector == "kmeans" else getattr(sdk, detector)
        return accepts_out(fn)

    def compute(self, detector, x, w, s, threads, progress=None, out=None, model=None):
        return sdk_compute(detector, x, w, s, threads, model, progress=progress, out=out)


class PyEngine:
    """PY-SEQ: the D-03 NumPy detectors, run in chunks so a progress counter can advance."""
    name = "py"
    CHUNK_WINDOWS = 200_000

    def live_scores(self, detector: str) -> bool:
        return True

    def compute(self, detector, x, w, s, threads, progress=None, out=None, model=None):
        if detector == "zscore":
            from threadwatch.detectors.zscore import zscore_py as fn
        elif detector == "mad":
            from threadwatch.detectors.mad import mad_py as fn
        else:
            raise ApiError(400, "unsupported", "PY-SEQ live view supports zscore and mad")
        nw = (len(x) - w) // s + 1
        out = np.empty(nw, np.float32) if out is None else out
        t0 = perf_counter()
        for a in range(0, nw, self.CHUNK_WINDOWS):
            b = min(a + self.CHUNK_WINDOWS, nw)
            out[a:b] = fn(np.asarray(x[a * s:(b - 1) * s + w]), w, s)
            if progress is not None:
                progress[0] = b
        return out, (perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
class DataStore:
    def __init__(self, settings: Settings):
        self.cfg = settings
        self._strip: Dict[str, dict] = {}
        self._thr: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def _path(self, name: str, suffix: str = ".f32") -> Path:
        if not NAME_RE.match(name):
            raise ApiError(400, "bad_dataset", f"bad dataset name {name!r}")
        return self.cfg.data_root / f"{name}{suffix}"

    def list(self) -> List[dict]:
        out = []
        if self.cfg.data_root.is_dir():
            for p in sorted(self.cfg.data_root.glob("*.f32")):
                out.append({"name": p.stem, "length": p.stat().st_size // 4,
                            "labelled": p.with_suffix(".labels.u8").exists(),
                            "kmeansModel": p.with_suffix(".kmeans.npz").exists()})
        return out

    def open(self, name: str):
        p = self._path(name)
        if not p.exists():
            raise ApiError(404, "no_dataset", f"dataset {name!r} not found")
        x = np.memmap(p, np.float32, "r")
        lp = self._path(name, ".labels.u8")
        y = np.memmap(lp, np.uint8, "r") if lp.exists() else None
        return x, y

    def model(self, name: str):
        p = self._path(name, ".kmeans.npz")
        if not p.exists():
            raise ApiError(400, "no_model", f"kmeans needs {p.name} (arrays: centroids, scale)")
        z = np.load(p)
        return np.ascontiguousarray(z["centroids"], np.float32), np.ascontiguousarray(z["scale"], np.float32)

    def strip(self, name: str, x) -> dict:
        """Min and max per column: a few hundred values stand for millions of samples."""
        with self._lock:
            if name not in self._strip:
                cols = min(self.cfg.strip_columns, len(x))
                idx = np.linspace(0, len(x), cols, endpoint=False).astype(np.int64)
                self._strip[name] = {"min": np.round(np.minimum.reduceat(x, idx), 4).tolist(),
                                     "max": np.round(np.maximum.reduceat(x, idx), 4).tolist()}
            return self._strip[name]

    def label_bins(self, y, cols: int) -> List[int]:
        if y is None:
            return []
        idx = np.linspace(0, len(y), cols, endpoint=False).astype(np.int64)
        return np.flatnonzero(np.maximum.reduceat(y, idx) > 0).tolist()

    def threshold(self, engine, name, x, detector, w, s, model) -> float:
        """(1 - rho) quantile of the scores of a sample of windows; computed once per (dataset, detector)."""
        key = (name, detector, w, s)
        with self._lock:
            if key not in self._thr:
                nw = (len(x) - w) // s + 1
                m = min(nw, self.cfg.thresh_windows)
                k = max(1, min(100, m))                          # k contiguous pieces spread over the series
                cw = max(1, m // k)
                parts = []
                for j in range(k):
                    a = 0 if k == 1 else j * (nw - cw) // (k - 1)
                    seg = np.asarray(x[a * s:(a + cw - 1) * s + w])
                    sc, _ = engine.compute(detector, seg, w, s, 1, model=model)
                    parts.append(np.asarray(sc))
                self._thr[key] = float(np.quantile(np.concatenate(parts), 1.0 - self.cfg.rho))
            return self._thr[key]


# ---------------------------------------------------------------------------
class Run:
    def __init__(self, mgr: "RunManager", req: RunRequest, sink: Optional[Callable[[dict], None]] = None,
                 keep_scores: bool = False, ref_scores=None, t1_ms: Optional[float] = None):
        self.mgr, self.req, self.sink = mgr, req, sink
        self.keep_scores, self.ref_scores, self.t1_ms = keep_scores, ref_scores, t1_ms
        self.id = uuid.uuid4().hex[:10]
        self.label = req.label
        self.log = EventLog()
        self.done = threading.Event()
        self.result: Optional[dict] = None
        self.error: Optional[dict] = None
        self.scores: Optional[np.ndarray] = None
        self.dir = mgr.cfg.runs_root / self.id
        self._on_finish: Optional[Callable[[], None]] = None

    def _finish(self, ev: dict) -> None:
        """Publish the terminal event. The manager lock is released FIRST: a client that reacts to `done`
        by starting the next run must not meet a 409."""
        cb, self._on_finish = self._on_finish, None
        if cb:
            cb()
        self.emit(ev)

    def emit(self, ev: dict) -> None:
        ev.setdefault("runId", self.id)
        self.log.append(ev)
        if self.sink:
            self.sink(ev)

    def start_async(self, on_finish: Optional[Callable[[], None]] = None) -> None:
        self._on_finish = on_finish

        def target():
            try:
                self.execute()
            finally:
                cb, self._on_finish = self._on_finish, None      # safety net if execute() died before _finish
                if cb:
                    cb()
        threading.Thread(target=target, name=f"run-{self.id}", daemon=True).start()

    # ------------------------------------------------------------------
    def execute(self) -> None:
        cfg, req = self.mgr.cfg, self.req
        monitor = None
        try:
            w, s = cfg.w, cfg.s
            x, y = self.mgr.store.open(req.dataset)
            n = len(x)
            if n < w:
                raise ApiError(400, "too_short", f"series has {n} samples, window is {w}")
            nw = (n - w) // s + 1
            slots = req.ranks * req.threads
            engine = self.mgr.py_engine if req.mode == "PY-SEQ" else self.mgr.engine
            model = self.mgr.store.model(req.dataset) if req.detector == "kmeans" else None
            thr = self.mgr.store.threshold(self.mgr.engine, req.dataset, x, req.detector, w, s, model)
            strip = self.mgr.store.strip(req.dataset, x)

            self.dir.mkdir(parents=True, exist_ok=True)
            progress = np.memmap(self.dir / "progress.bin", np.int64, "w+", shape=(slots * 8,))
            scores = np.memmap(self.dir / "scores.f32", np.float32, "w+", shape=(nw,))
            live = engine.live_scores(req.detector)
            if req.ranks > 1 and req.inputMode == "scatter":
                live = False                      # scatter: rank 0 fills the score buffer after the gather

            blocks = slot_blocks(n, w, s, req.ranks, req.threads) if req.mode != "PY-SEQ" else [(0, nw)]
            self.emit({"type": "start", "mode": self.label, "blocks": blocks, "kind": req.mode, "ranks": req.ranks,
                       "threads": req.threads, "inputMode": req.inputMode, "detector": req.detector, "dataset": req.dataset,
                       "n": n, "nw": nw, "w": w, "s": s, "threshold": thr, "slots": slots,
                       "strip": strip, "labelBins": self.mgr.store.label_bins(y, len(strip["min"])),
                       "scoresLive": live, "startedAt": datetime.now(timezone.utc).isoformat()})

            monitor = Monitor(label=self.label, progress=progress, scores=scores, blocks=blocks, nw=nw, s=s, w=w,
                              threshold=thr, live=live, emit=self.emit, poll_s=cfg.poll_s)
            monitor.start()
            t0 = perf_counter()
            if req.ranks == 1:
                _, dll_ms = engine.compute(req.detector, x, w, s, req.threads, progress=progress, out=scores, model=model)
                ms = (perf_counter() - t0) * 1000.0
                extra = {"dllMs": dll_ms}
            else:
                row = self._run_mpi(progress.filename, scores.filename)
                ms = float(row["total_s"]) * 1000.0
                extra = {"dllMs": row["dll_ms"], "commMs": float(row["comm_s"]) * 1000.0}
            scores.flush()
            monitor.stop()
            self.result = self._summarise(x, y, scores, n, nw, ms, monitor, extra)
            self._finish({"type": "done", "mode": self.label, "result": self.result})
        except ApiError as e:
            self._fail(monitor, e.code, e.message)
        except Exception as e:                      # keep the failure visible, never swallow it
            self._fail(monitor, "run_failed", f"{type(e).__name__}: {e}")
        finally:
            self.done.set()
            self.log.close()
            shutil.rmtree(self.dir, ignore_errors=True)      # no stale run directories

    def _fail(self, monitor, code, message):
        if monitor is not None:
            try:
                monitor._stop.set()
            except Exception:
                pass
        self.error = {"code": code, "message": message}
        self._finish({"type": "error", "mode": self.label, **self.error})

    def _run_mpi(self, progress_path: str, scores_path: str) -> dict:
        cfg, req = self.mgr.cfg, self.req
        cmd = [cfg.mpiexec, "-n", str(req.ranks), sys.executable, "-m", "threadwatch.mpi.run", "--mode", req.inputMode,
               "--dataset", req.dataset, "--root", str(cfg.data_root), "--detector", req.detector,
               "--threads", str(req.threads), "--w", str(cfg.w), "--s", str(cfg.s),
               "--progress", progress_path, "--scores-out", scores_path, "--allow-oversubscribe"]
        if req.detector == "kmeans":
            cmd += ["--model", str(self.mgr.cfg.data_root / f"{req.dataset}.kmeans.npz")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rows = [json.loads(l[7:]) for l in proc.stdout.splitlines() if l.startswith("TW_ROW")]
        if proc.returncode != 0 or not rows:
            raise RuntimeError(f"mpiexec failed (exit {proc.returncode}): {proc.stderr.strip()[-300:]}")
        return rows[0]

    def _summarise(self, x, y, scores, n, nw, ms, monitor, extra) -> dict:
        cfg, req = self.mgr.cfg, self.req
        thr = monitor.threshold
        flags = np.asarray(scores) > thr
        t1 = self.t1_ms or (ms if req.mode == "C-1" else self.mgr.t1_ms(req.dataset, req.detector, n))
        if req.mode == "C-1":
            self.mgr.remember_t1(req.dataset, req.detector, n, ms)
        speedup = (t1 / ms) if t1 else None
        res = {
            "mode": self.label, "kind": req.mode, "ranks": req.ranks, "threads": req.threads,
            "detector": req.detector, "dataset": req.dataset, "n": n, "nw": nw,
            "ms": ms, "throughputSps": n / (ms / 1000.0),
            "t1Ms": t1, "speedup": speedup,
            "efficiency": (speedup / (req.ranks * req.threads) * 100.0) if speedup else None,
            "flags": int(flags.sum()), "threshold": thr,
            "countersSum": int(monitor.last_total), "identical": None,
            "f1": None, "precision": None, "recall": None, "eventRecall": None, **extra,
        }
        if self.ref_scores is not None:
            res["identical"] = bool(np.array_equal(np.asarray(scores), self.ref_scores))
        if y is not None:
            try:
                from threadwatch.metrics import evaluate
                q = evaluate(flags, y, cfg.w, cfg.s)
                res.update(f1=q["f1"], precision=q["precision"], recall=q["recall"], eventRecall=q["event_recall"])
            except ImportError:
                pass
        if self.keep_scores:
            self.scores = np.array(scores)
        return res


# ---------------------------------------------------------------------------
class RunManager:
    def __init__(self, settings: Optional[Settings] = None, engine=None, py_engine=None):
        self.cfg = settings or Settings()
        self.engine = engine or SdkEngine()
        self.py_engine = py_engine or PyEngine()
        self.store = DataStore(self.cfg)
        self.runs: "OrderedDict[str, Run]" = OrderedDict()
        self.races: "OrderedDict[str, object]" = OrderedDict()
        self._busy = threading.Lock()
        self._t1: Dict[tuple, float] = {}
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)

    # -- one active run or race at a time: they would compete for the same cores
    def acquire(self) -> None:
        if not self._busy.acquire(blocking=False):
            raise ApiError(409, "busy", "a run is already in progress; modes never run at the same time")

    def release(self) -> None:
        self._busy.release()

    def validate(self, req: RunRequest) -> None:
        self.store.open(req.dataset)
        if req.detector == "kmeans":
            if req.mode == "PY-SEQ":
                raise ApiError(400, "unsupported", "PY-SEQ live view supports zscore and mad")
            self.store.model(req.dataset)
        cores = logical_cores()
        if req.ranks * req.threads > cores and not req.allowOversubscribe:
            raise ApiError(400, "oversubscribed", f"ranks x threads = {req.ranks * req.threads} exceeds {cores} logical cores")
        if req.ranks > 1 and not shutil.which(self.cfg.mpiexec):
            raise ApiError(400, "no_mpiexec", f"{self.cfg.mpiexec!r} not found; MPI modes need an MPI runtime")

    def create(self, req: RunRequest, **kw) -> Run:
        self.validate(req)
        run = Run(self, req, **kw)
        self.runs[run.id] = run
        self._evict()
        return run

    def start(self, req: RunRequest) -> Run:
        self.validate(req)
        self.acquire()
        try:
            run = self.create(req)
        except Exception:
            self.release()
            raise
        run.start_async(on_finish=self.release)
        return run

    def get(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise ApiError(404, "no_run", f"run {run_id!r} not found")
        return self.runs[run_id]

    def _evict(self) -> None:
        finished = [k for k, r in self.runs.items() if r.done.is_set()]
        while len(self.runs) > self.cfg.max_runs and finished:
            self.runs.pop(finished.pop(0), None)

    # -- T1 (C-1 time) for speedup: this session first, then the benchmark CSV
    def remember_t1(self, dataset, detector, n, ms):
        self._t1[(dataset, detector, n, self.cfg.w, self.cfg.s)] = ms

    def t1_ms(self, dataset, detector, n) -> Optional[float]:
        key = (dataset, detector, n, self.cfg.w, self.cfg.s)
        if key in self._t1:
            return self._t1[key]
        vals = []
        try:
            with open(self.cfg.results_csv, newline="") as f:
                for r in csv.DictReader(f):
                    if (r.get("mode") == "C-1" and r.get("dataset") == dataset and r.get("detector") == detector
                            and r.get("n") == str(n) and r.get("w") == str(self.cfg.w) and r.get("s") == str(self.cfg.s)
                            and r.get("status", "ok") == "ok"):
                        vals.append(float(r["total_s"]) * 1000.0)
        except (OSError, KeyError, ValueError):
            return None
        return statistics.median(vals) if vals else None
