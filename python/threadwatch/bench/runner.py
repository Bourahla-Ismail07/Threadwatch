"""D-08.2 to D-08.4 - run the matrix: one process-safe, resumable, interleaved campaign.

Design points (all from the protocol):
  * one row per run, appended and fsynced at once; an interrupted campaign resumes where it stopped;
  * the modes of a (dataset, detector) block are interleaved round by round (warm-up round first);
  * every run is checked against the C-1 output of its block (`identical`) and scored (`f1`, ...);
  * a run that exceeds its timeout becomes a `timeout` row, never a hang: MPI cells are killed as a process tree,
    in-process cells by a watchdog that records the row and exits with code 75 (the supervisor restarts the harness);
  * a crash (segfault, kill) is recorded by the supervisor as a `crash` row; failed rows are never deleted;
  * the only edit of existing rows is `ok -> superseded` when a cell is rerun, and the old file is kept as a backup.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .matrix import Block, Cell, blocks_of
from .rows import CRASH, ERROR, OK, SUPERSEDED, TIMEOUT, WARMUP, ResultsFile, make_row

EX_TIMEOUT = 75


class CellTimeout(Exception):
    pass


class CellError(Exception):
    pass


# =============================================================================== backends
class CBackend:
    """C kernels through the SDK wrapper (D-04/D-05). The only place that names the wrapper's functions."""

    def score(self, detector, x, w, s, p):
        from threadwatch import sdk
        out, st = (sdk.zscore if detector == "zscore" else sdk.mad)(x, w=w, s=s, threads=p)
        return out, float(st.ms)

    def features(self, x, w, s, p):
        from threadwatch import sdk
        out, st = sdk.window_features(x, w=w, s=s, threads=p)
        return out, float(st.ms)

    def kmeans(self, F, centroids, scale, p):
        from threadwatch import sdk
        out, st = sdk.kmeans_score(F, centroids, scale, threads=p)
        return out, float(st.ms)


class PyBackend:
    """The D-03 NumPy detectors (PY-SEQ). No DLL time: dll_ms stays empty."""

    def score(self, detector, x, w, s, p):
        if detector == "zscore":
            from threadwatch.detectors.zscore import zscore_py as fn
        else:
            from threadwatch.detectors.mad import mad_py as fn
        return fn(x, w, s), None

    def features(self, x, w, s, p):
        from threadwatch.detectors.features import features_py
        return features_py(x, w, s), None

    def kmeans(self, F, centroids, scale, p):
        from threadwatch.detectors.ml import kmeans_score_py
        return kmeans_score_py(F, centroids, scale), None


def iforest_score(iso, F, n_jobs: int):
    """Isolation Forest anomaly score (higher = more anomalous). `n_jobs` is the thread parameter of the cell."""
    iso.set_params(n_jobs=n_jobs)
    return -iso.score_samples(F)


def kill_tree(proc) -> None:
    """Kill a process and everything it started (mpiexec and its ranks). Works on Windows and POSIX."""
    import psutil
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


# =============================================================================== data, models, thresholds
@dataclass
class Prepared:
    thresholds: Dict[str, float]            # detector -> threshold
    centroids: np.ndarray
    scale: np.ndarray
    iso: object
    model_npz: Path


def sample_stack(x, w: int, size: int, seed: int) -> np.ndarray:
    """The fit sample: `size` windows drawn uniformly (fixed seed), stacked back to back, so that a kernel call with
    stride w gives exactly one score per sampled window."""
    nstart = len(x) - w + 1
    m = min(size, nstart)
    starts = np.sort(np.random.default_rng(seed).choice(nstart, size=m, replace=False))
    view = np.lib.stride_tricks.sliding_window_view(x, w)
    return np.ascontiguousarray(view[starts], dtype=np.float32).reshape(-1)


class Context:
    def __init__(self, cfg: dict, data_root, out_dir, c_backend=None, py_backend=None):
        self.cfg, self.data_root, self.out_dir = cfg, Path(data_root), Path(out_dir)
        self.c, self.py = c_backend or CBackend(), py_backend or PyBackend()
        self._prep: Dict[tuple, Prepared] = {}
        self._labels: Dict[str, object] = {}

    def path(self, dataset: str) -> Path:
        return self.data_root / f"{dataset}.f32"

    def series(self, dataset: str):
        return np.memmap(self.path(dataset), np.float32, "r")

    def n_of(self, dataset: str) -> int:
        return self.path(dataset).stat().st_size // 4

    def labels(self, dataset: str):
        if dataset not in self._labels:
            p = self.data_root / f"{dataset}.labels.u8"
            self._labels[dataset] = np.memmap(p, np.uint8, "r") if p.exists() else None
        return self._labels[dataset]

    def warm_file(self, dataset: str) -> None:
        """Read the whole file once so the first timed run does not pay for a cold file cache."""
        x = self.series(dataset)
        step = 1 << 22
        acc = 0.0
        for a in range(0, len(x), step):
            acc += float(x[a:a + step][::64].sum())

    def prepare(self, dataset: str, w: int) -> Prepared:
        """Fit the models and the thresholds once per (dataset, w) on the fixed sample, and keep them on disk.
        Every mode of every run uses these same thresholds."""
        key = (dataset, w)
        if key in self._prep:
            return self._prep[key]
        import joblib
        from threadwatch.detectors.ml import fit_models
        pr = self.cfg["protocol"]
        mdir = self.out_dir / "models"
        mdir.mkdir(parents=True, exist_ok=True)
        npz, ifp, thp = mdir / f"{dataset}.w{w}.kmeans.npz", mdir / f"{dataset}.w{w}.iforest.joblib", self.out_dir / "thresholds.json"
        allthr = json.loads(thp.read_text()) if thp.exists() else {}
        names = ("zscore", "mad", "kmeans", "iforest")
        if npz.exists() and ifp.exists() and all(f"{dataset}|{d}|{w}" in allthr for d in names):
            try:
                z = np.load(npz)
                prep = Prepared({d: allthr[f"{dataset}|{d}|{w}"]["threshold"] for d in names},
                                np.ascontiguousarray(z["centroids"], np.float32), np.ascontiguousarray(z["scale"], np.float32),
                                joblib.load(ifp), npz)
                self._prep[key] = prep
                return prep
            except Exception:
                pass                                           # unreadable model (e.g. other scikit-learn): refit below
        x = self.series(dataset)
        flat = sample_stack(x, w, int(pr["fit_sample"]), int(pr["seed"]))
        m = len(flat) // w
        F, _ = self.c.features(flat, w, w, 1)                 # 1 thread: deterministic
        centroids, scale, iso = fit_models(F, k=8, seed=int(pr["seed"]))
        centroids, scale = np.ascontiguousarray(centroids, np.float32), np.ascontiguousarray(scale, np.float32)
        scores = {"zscore": self.c.score("zscore", flat, w, w, 1)[0], "mad": self.c.score("mad", flat, w, w, 1)[0],
                  "kmeans": self.c.kmeans(F, centroids, scale, 1)[0], "iforest": iforest_score(iso, F, 1)}
        thr = {d: float(np.quantile(v, 1.0 - float(pr["rho"]))) for d, v in scores.items()}
        for d, t in thr.items():
            allthr[f"{dataset}|{d}|{w}"] = {"threshold": t, "sample_windows": m, "seed": int(pr["seed"]), "rho": float(pr["rho"])}
        thp.write_text(json.dumps(allthr, indent=1, sort_keys=True))
        np.savez(npz, centroids=centroids, scale=scale)
        joblib.dump(iso, ifp)
        prep = Prepared(thr, centroids, scale, iso, npz)
        self._prep[key] = prep
        return prep

    def quality(self, dataset: str, score, thr: float, w: int, s: int) -> dict:
        y = self.labels(dataset)
        if y is None:
            return {}
        from threadwatch.metrics import evaluate
        q = evaluate(np.asarray(score) > thr, y, w, s)
        return {"f1": q["f1"], "precision": q["precision"], "recall": q["recall"], "event_recall": q["event_recall"]}


# =============================================================================== one block
@dataclass
class BlockState:
    prep: Prepared
    ref: Optional[np.ndarray] = None
    quality: Optional[dict] = None
    ref_file: Optional[Path] = None
    mismatch_file: Optional[Path] = None


class Runner:
    def __init__(self, cfg: dict, ctx: Context, results: ResultsFile, out_dir, logical_cores: int,
                 log: Callable[[str], None] = print, abort: Optional[Callable[[int], None]] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg, self.ctx, self.results, self.out_dir = cfg, ctx, results, Path(out_dir)
        self.logical, self._log, self.sleep = logical_cores, log, sleep
        self.abort = abort or self._hard_exit
        self.tmp = self.out_dir / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._finished: set = set()
        self._timed_out: set = set()
        try:                                                      # NumPy/BLAS thread pools: fixed to 1 for every cell
            from threadpoolctl import threadpool_limits
            self._tp = threadpool_limits(limits=1)
        except ImportError:
            self._tp = None

    # -------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        self._log(line)
        with open(self.out_dir / "campaign.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _hard_exit(code: int) -> None:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(code)

    # -------------------------------------------------------------- campaign
    def run(self, cells: Sequence[Cell], batches: Optional[Dict[str, int]] = None, note: str = "") -> int:
        """Run every block; returns the number of runs executed. `batches` maps cell key -> batch (reruns)."""
        executed = 0
        blocks = blocks_of(cells)
        pause = float(self.cfg["protocol"].get("pause_between_blocks_s", 0))
        for i, block in enumerate(blocks):
            n = self.run_block(block, batches or {}, note)
            executed += n
            if n and i < len(blocks) - 1 and pause > 0:
                self.log(f"pause {pause:.0f}s (cool-down)")
                self.sleep(pause)
        return executed

    def run_block(self, block: Block, batches: Dict[str, int], note: str = "") -> int:
        ds, det, w, s = block.key
        pr = self.cfg["protocol"]
        done, closed = self.results.done(), self.results.closed()
        rounds = int(pr["warmup"]) + int(pr["repeats"])
        assert int(pr["warmup"]) == 1, "this harness numbers the single warm-up as repeat 0"
        todo = [(c, r) for c in block.cells for r in range(rounds)
                if (c.key, r, batches.get(c.key, 0)) not in done and (c.key, batches.get(c.key, 0)) not in closed]
        if not todo:
            return 0
        self.log(f"block {ds} / {det} (w={w}, s={s}): {len(todo)} runs to do in {len(block.cells)} cells")
        self._check_background()
        self.ctx.warm_file(ds)
        state = BlockState(self.ctx.prepare(ds, w))
        executed = 0
        try:
            for rnd in range(rounds):
                for cell in block.order(rnd):
                    b = batches.get(cell.key, 0)
                    if (cell.key, rnd, b) in done or (cell.key, b) in closed:
                        continue
                    ok = self.execute_one(cell, rnd, b, state, block, note)
                    executed += 1
                    if not ok:
                        closed.add((cell.key, b))
        finally:
            for f in (state.ref_file, state.mismatch_file):
                if f is not None:
                    f.unlink(missing_ok=True)
        return executed

    def _check_background(self) -> None:
        import psutil
        limit = float(self.cfg["protocol"].get("max_background_cpu_pct", 15))
        if limit >= 100:                                           # check disabled
            return
        for _ in range(12):
            busy = psutil.cpu_percent(interval=1.0)
            if busy <= limit:
                return
            self.log(f"background CPU {busy:.0f}% above {limit:.0f}%: waiting 5 s (close other applications)")
            self.sleep(5)
        self.log("background CPU still high: continuing, the runs of this block may be noisy")

    # -------------------------------------------------------------- one run
    def execute_one(self, cell: Cell, rnd: int, batch: int, state: BlockState, block: Block, note: str = "") -> bool:
        """Run one (cell, repeat); append exactly one row. Returns False if the cell failed (it is then closed)."""
        token = (cell.key, rnd, batch)
        status = WARMUP if rnd == 0 else OK
        base_note = ("warm-up (discarded). " if rnd == 0 else "") + (note or "")
        if batch:
            base_note += f"rerun batch {batch}. "
        in_proc = cell.mode in ("PY-SEQ", "C-1", "OMP")
        self._mark_inflight(cell, rnd, batch)
        timer = None
        if in_proc:
            timer = threading.Timer(cell.timeout_s, self._watchdog_fired, args=(cell, rnd, batch, base_note))
            timer.daemon = True
            timer.start()
        m, row, ok = None, None, True
        try:
            m = self._measure_mpi(cell, state) if not in_proc else self._measure_inprocess(cell, state)
        except CellTimeout as e:
            row, ok = make_row(cell, rnd, TIMEOUT, batch, (base_note + str(e)).strip()), False
        except Exception as e:
            tail = "".join(traceback.format_exception_only(type(e), e)).strip().replace("\n", " | ")[:400]
            row, ok = make_row(cell, rnd, ERROR, batch, (base_note + tail).strip()), False
            self.log(f"ERROR {cell.key} r{rnd}: {tail}")
        finally:
            if timer:                                            # the timed call is over: nothing below may trigger a timeout
                timer.cancel()
        if m is not None:
            try:
                row = self._finish_row(cell, rnd, batch, status, state, m, block, base_note)
            except Exception as e:
                tail = "".join(traceback.format_exception_only(type(e), e)).strip().replace("\n", " | ")[:400]
                row, ok = make_row(cell, rnd, ERROR, batch, (base_note + "post-processing: " + tail).strip()), False
                self.log(f"ERROR {cell.key} r{rnd} (post-processing): {tail}")
        with self._lock:
            if token in self._timed_out:                          # the watchdog already wrote the row
                return False
            self._finished.add(token)
            self.results.append(row)
        self._clear_inflight()
        if row["status"] in (OK, WARMUP):
            self.log(f"{cell.key} r{rnd}: {row['total_s']:.3f} s, identical={row['identical']}")
        else:
            self.log(f"{row['status'].upper()} {cell.key} r{rnd}: {row['note']}")
        return ok

    def _watchdog_fired(self, cell: Cell, rnd: int, batch: int, base_note: str) -> None:
        token = (cell.key, rnd, batch)
        with self._lock:
            if token in self._finished:
                return
            self._timed_out.add(token)
            self.results.append(make_row(cell, rnd, TIMEOUT, batch, (base_note + f"timeout after {cell.timeout_s:.0f} s (in-process call; harness restarted)").strip()))
        self._clear_inflight()
        self.abort(EX_TIMEOUT)

    # -------------------------------------------------------------- in-process measurement
    def _measure_inprocess(self, cell: Cell, state: BlockState) -> dict:
        ctx = self.ctx
        be = ctx.py if cell.mode == "PY-SEQ" else ctx.c
        w, s, p, det = cell.w, cell.s, cell.p, cell.detector
        feat_s = score_s = None
        dll = None if cell.mode == "PY-SEQ" else 0.0

        t0 = perf_counter()                                        # end to end: mapping the input is part of the call
        x = ctx.series(cell.dataset)
        load_s = perf_counter() - t0
        if det in ("zscore", "mad"):
            t = perf_counter()
            score, ms = be.score(det, x, w, s, p)
            score_s = perf_counter() - t
            dll = None if ms is None else dll + ms
        else:
            t = perf_counter()
            F, ms = be.features(x, w, s, p)
            feat_s = perf_counter() - t
            dll = None if ms is None else dll + ms
            t = perf_counter()
            if det == "kmeans":
                score, ms = be.kmeans(F, state.prep.centroids, state.prep.scale, p)
                dll = None if ms is None else dll + ms
            else:
                score = iforest_score(state.prep.iso, F, p)
            score_s = perf_counter() - t
        total = perf_counter() - t0
        return {"load_s": load_s, "features_s": feat_s, "score_s": score_s, "comm_s": None, "total_s": total,
                "dll_ms": dll, "score": score}

    # -------------------------------------------------------------- MPI measurement (one process launch per run)
    def _measure_mpi(self, cell: Cell, state: BlockState) -> dict:
        ref_file = self._ref_file(cell, state)
        state.mismatch_file = state.mismatch_file or (self.tmp / f"mismatch_{os.getpid()}.npy")
        state.mismatch_file.unlink(missing_ok=True)
        cmd = [self.cfg.get("mpiexec", "mpiexec"), "-n", str(cell.q), sys.executable, "-m", "threadwatch.mpi.run",
               "--mode", cell.input_mode, "--dataset", cell.dataset, "--root", str(self.ctx.data_root),
               "--detector", cell.detector, "--threads", str(cell.p), "--w", str(cell.w), "--s", str(cell.s),
               "--save-on-mismatch", str(state.mismatch_file)]
        if ref_file is not None:
            cmd += ["--reference", str(ref_file)]
        if cell.detector == "kmeans":
            cmd += ["--model", str(state.prep.model_npz)]
        if self.cfg["protocol"].get("allow_oversubscribe"):
            cmd += ["--allow-oversubscribe"]
        pkg_root = str(Path(__file__).resolve().parents[2])      # ranks must import the same threadwatch as this process
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH", "")]))}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            out, err = proc.communicate(timeout=cell.timeout_s)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            proc.communicate()
            raise CellTimeout(f"timeout after {cell.timeout_s:.0f} s (mpiexec killed with its ranks)")
        rows = [json.loads(l[7:]) for l in out.splitlines() if l.startswith("TW_ROW")]
        if proc.returncode != 0 or not rows:
            raise CellError(f"mpiexec exit {proc.returncode}: {err.strip()[-300:]}")
        r = rows[0]
        m = {"load_s": r["load_s"], "features_s": None, "score_s": r["score_s"], "comm_s": r["comm_s"],
             "total_s": r["total_s"], "dll_ms": r["dll_ms"], "identical": r.get("identical"), "score": None}
        if r.get("identical") is False and state.mismatch_file.exists():
            m["score"] = np.load(state.mismatch_file)
        return m

    def _ref_file(self, cell: Cell, state: BlockState) -> Optional[Path]:
        ref = self._ensure_ref(cell, state)
        if ref is None:
            return None
        if state.ref_file is None:
            state.ref_file = self.tmp / f"ref_{os.getpid()}.npy"
            np.save(state.ref_file, ref)
        return state.ref_file

    # -------------------------------------------------------------- reference (C-1) and quality
    def _ensure_ref(self, cell: Cell, state: BlockState) -> Optional[np.ndarray]:
        """C-1 output of the block. Normally the C-1 warm-up run provides it; after a resume it is recomputed here
        (an extra, unrecorded C-1 pass, outside any timing)."""
        if state.ref is not None:
            return state.ref
        c1 = Cell(cell.dataset, cell.n, cell.detector, "C-1", 1, 1, cell.w, cell.s, cell.timeout_s)
        if (c1.key, 0) in {(k, b) for k, b in self.results.closed()}:
            return None
        try:
            state.ref = self._measure_inprocess(c1, state)["score"]
        except Exception as e:
            self.log(f"cannot compute the C-1 reference for {cell.dataset}/{cell.detector}: {e}")
            return None
        thr = state.prep.thresholds[cell.detector]
        state.quality = self.ctx.quality(cell.dataset, state.ref, thr, cell.w, cell.s)
        return state.ref

    def _finish_row(self, cell: Cell, rnd: int, batch: int, status: str, state: BlockState, m: dict, block: Block, note: str):
        thr = state.prep.thresholds[cell.detector]
        score = m.pop("score")
        identical = m.pop("identical", None)
        if score is not None:                                       # in-process run, or an MPI run that differed from C-1
            if state.ref is None and cell.mode == "C-1":
                state.ref = np.array(score)                         # the first C-1 run of the block is the reference
                state.quality = self.ctx.quality(cell.dataset, state.ref, thr, cell.w, cell.s)
            ref = self._ensure_ref(cell, state)
            identical = None if ref is None else bool(np.array_equal(np.asarray(score), ref))
            q = (state.quality if identical else self.ctx.quality(cell.dataset, score, thr, cell.w, cell.s))
        else:                                                       # MPI run identical to C-1: same scores, same quality
            self._ensure_ref(cell, state)
            q = state.quality
        if identical is None and note.find("no C-1 reference") < 0:
            note += "no C-1 reference. "
        if cell.detector == "kmeans" and cell.mode in ("MPI", "HYB", "MPI-scatter"):
            note += "features_s not separable (fused in the rank kernel). "
        if cell.detector == "iforest":
            note += f"features in C, scoring in scikit-learn n_jobs={cell.p}. "
        if cell.mode == "MPI-scatter":
            note += "input=scatter. "
        if cell.q * cell.p > self.logical:
            note += "OVERSUBSCRIBED. "
        return make_row(cell, rnd, status, batch, note.strip(), identical=identical, **(q or {}), **m)

    # -------------------------------------------------------------- in-flight marker for the supervisor
    def _mark_inflight(self, cell: Cell, rnd: int, batch: int) -> None:
        (self.out_dir / "current.json").write_text(json.dumps(
            {"cell": cell.as_dict(), "repeat": rnd, "batch": batch, "started": time.time()}))

    def _clear_inflight(self) -> None:
        (self.out_dir / "current.json").unlink(missing_ok=True)

    # -------------------------------------------------------------- reruns of noisy cells
    def rerun(self, cells: Sequence[Cell], reasons: Dict[str, str]) -> int:
        """Mark the existing runs of the given cells as superseded (file backed up), then run the cells again as a
        new batch. Nothing is deleted: the first runs stay in the CSV with status `superseded`."""
        batches: Dict[str, int] = {}
        superseded = 0
        for c in cells:
            batches[c.key] = self.results.next_batch(c.key)
            ids = [r["run_id"] for r in self.results.rows()
                   if r["run_id"].split("#r")[0] == c.key and r["status"] in (OK, WARMUP)]
            superseded += self.results.rewrite_status(
                ids, SUPERSEDED, f"superseded by rerun batch {batches[c.key]} ({reasons.get(c.key, 'noisy')})",
                self.out_dir / "backups")
        if not superseded:
            return 0
        return self.run(cells, batches, note="")


# =============================================================================== supervisor
def supervise(child_argv: List[str], out_dir, results: ResultsFile, max_restarts: int = 200,
              log: Callable[[str], None] = print) -> int:
    """Run the harness as a child. A watchdog timeout (exit 75) already wrote its row: restart. Any other non-zero exit
    while a run was in flight (segfault, kill, out of memory) is recorded as a `crash` row, then restart."""
    from .matrix import Cell
    cur = Path(out_dir) / "current.json"
    restarts = 0
    while True:
        proc = subprocess.Popen(child_argv)
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            kill_tree(proc)
            log("interrupted: the campaign resumes from the CSV on the next start")
            return 130
        if rc == 0:
            return 0
        if rc == EX_TIMEOUT:
            log("a run timed out; restarting the harness")
        elif cur.exists():
            info = json.loads(cur.read_text())
            cell = Cell(**info["cell"])
            results.append(make_row(cell, info["repeat"], CRASH, info["batch"],
                                    f"process exited with code {rc & 0xFFFFFFFF:#x} during the run (crash or kill)"))
            cur.unlink()
            log(f"CRASH in {cell.key} r{info['repeat']} (exit {rc & 0xFFFFFFFF:#x}); restarting")
        else:
            log(f"harness exited with code {rc} outside any run; not restarting")
            return rc
        restarts += 1
        if restarts > max_restarts:
            log("too many restarts; stopping")
            return rc
