"""D-07 - API, SSE, monitor and race tests (FastAPI TestClient; no browser, no real DLL needed for most)."""
import json
import os
import shutil
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from threadwatch.service.app import create_app, sse_events
from threadwatch.service.monitor import MAX_FLAGS, Monitor, slot_blocks
from threadwatch.service.runs import EventLog, Settings

N, W, S = 20_000, 32, 1
NW = (N - W) // S + 1


# ---------------------------------------------------------------- fixtures
class FakeEngine:
    """Advances the D-05 counters block by block and writes scores in place, like a slow OpenMP kernel."""
    name = "fake"

    def __init__(self, delay=0.02, steps=25, live=True):
        self.delay, self.steps, self.live = delay, steps, live
        self.spans = []                              # (start, end) of every instrumented call
        self.lock = threading.Lock()

    def live_scores(self, detector):
        return self.live

    @staticmethod
    def scores(x, w, s):
        """|last sample of the window - 50|: depends on the data only, so any slicing gives the same scores."""
        nw = (len(x) - w) // s + 1
        return np.abs(np.asarray(x[w - 1: (nw - 1) * s + w: s], np.float64) - 50.0).astype(np.float32)

    def compute(self, detector, x, w, s, threads, progress=None, out=None, model=None):
        nw = (len(x) - w) // s + 1
        full = self.scores(x, w, s)
        if progress is None and out is None:         # threshold sample: instant
            return full, 0.0
        t0 = time.perf_counter()
        blocks = slot_blocks(len(x), w, s, 1, threads)
        for step in range(1, self.steps + 1):
            for t, (a, ln) in enumerate(blocks):
                done = ln * step // self.steps
                prev = ln * (step - 1) // self.steps
                if self.live:
                    out[a + prev:a + done] = full[a + prev:a + done]
                progress[t * 8] = done
            time.sleep(self.delay)
        if not self.live:
            out[:] = full
        with self.lock:
            self.spans.append((t0, time.perf_counter()))
        return out, (time.perf_counter() - t0) * 1000


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "series"
    root.mkdir()
    rng = np.random.default_rng(3)
    x = (50 + rng.normal(0, 1, N)).astype(np.float32)
    x[[3000, 9000, 10_020, 15_000]] += 25.0                        # four clear spikes (two inside the labelled event)
    x.tofile(root / "syn_t.f32")
    y = np.zeros(N, np.uint8)
    y[10_000:10_050] = 1
    y.tofile(root / "syn_t.labels.u8")
    return root


def make_client(data, tmp_path, engine=None, **kw):
    cfg = Settings(data_root=data, runs_root=tmp_path / "runs", results_csv=tmp_path / "results.csv",
                   ui_dir=tmp_path / "no-ui", w=W, s=S, thresh_windows=5_000, **kw)
    app = create_app(cfg, engine=engine or FakeEngine())
    return TestClient(app), app.state.mgr


def frames(resp):
    out, ev, data = [], None, None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:])
        elif line == "" and ev:
            out.append((ev, data)); ev = None
    return out


def run_and_collect(client, **body):
    r = client.post("/api/runs", json=body)
    assert r.status_code == 202, r.text
    with client.stream("GET", r.json()["stream"]) as s:
        return r.json()["runId"], frames(s)


# ---------------------------------------------------------------- slot blocks
@pytest.mark.parametrize("n,ranks,threads", [(20_000, 1, 4), (20_000, 2, 4), (20_000, 3, 2), (300, 1, 8), (300, 4, 4), (5_000, 1, 1)])
def test_slot_blocks_cover_every_window_once(n, ranks, threads):
    nw = (n - W) // S + 1
    blocks = slot_blocks(n, W, S, ranks, threads)
    assert len(blocks) == ranks * threads
    covered = np.zeros(nw, int)
    for a, ln in blocks:
        covered[a:a + ln] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_slot_blocks_match_the_static_schedule_formula():
    nw = (N - W) // S + 1
    q, r = divmod(nw, 4)
    assert slot_blocks(N, W, S, 1, 4) == [(t * q + min(t, r), q + (t < r)) for t in range(4)]


# ---------------------------------------------------------------- monitor
def test_monitor_reads_only_new_windows_and_caps_flags():
    nw, threads = 1000, 4
    blocks = slot_blocks(nw + W - 1, W, 1, 1, threads)
    scores = np.zeros(nw, np.float32)
    progress = np.zeros(threads * 8, np.int64)
    events = []
    mon = Monitor(label="OMP-4", progress=progress, scores=scores, blocks=blocks, nw=nw, s=1, w=W,
                  threshold=5.0, live=True, emit=events.append, cpu=False)
    a0, ln0 = blocks[0]
    scores[a0 + 3] = 7.0                                    # inside what thread 0 will have done
    scores[blocks[1][0] + 2] = 8.0                          # thread 1 has not got there yet
    progress[0] = 10
    mon.poll()
    assert [f["score"] for f in events[-1]["flags"]] == [7.0]
    assert events[-1]["flags"][0]["i"] == a0 + 3 + W - 1    # sample index of the window end
    assert events[-1]["threads"][:2] == [10, 0]
    progress[8] = 10
    mon.poll()
    assert [f["score"] for f in events[-1]["flags"]] == [8.0]          # new flag only, old one not repeated
    scores[a0 + 10: a0 + 10 + 120] = 9.0                              # a burst of 120 flags
    progress[0] = 10 + 120
    mon.poll()
    assert len(events[-1]["flags"]) == MAX_FLAGS and events[-1]["flagsTotal"] == 122


def test_monitor_not_live_scans_once_at_the_end():
    nw = 500
    scores = np.zeros(nw, np.float32)
    events = []
    mon = Monitor(label="C-1", progress=np.zeros(8, np.int64), scores=scores, blocks=[(0, nw)], nw=nw, s=1, w=W,
                  threshold=5.0, live=False, emit=events.append, cpu=False)
    mon.poll()
    assert events[-1]["flags"] == []
    scores[[3, 4, 400]] = 6.0
    mon.stop()                                              # never started: stop() must still do the final poll
    assert events[-1]["flagsTotal"] == 3 and len(events[-1]["flags"]) == 3


# ---------------------------------------------------------------- SSE plumbing
def test_sse_sends_keepalive_comments_and_ids():
    log = EventLog()
    gen = sse_events(log, keepalive_s=0.05)
    assert next(gen) == ": keepalive\n\n"
    log.append({"type": "progress", "x": 1})
    frame = next(gen)
    assert frame.startswith("id: 0\nevent: progress\ndata: ") and frame.endswith("\n\n")
    log.append({"type": "done"})
    assert "event: done" in next(gen)
    with pytest.raises(StopIteration):
        next(gen)


# ---------------------------------------------------------------- HTTP API
def test_health_datasets_and_error_shape(data, tmp_path):
    c, _ = make_client(data, tmp_path)
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["logicalCores"] >= 1
    ds = c.get("/api/datasets").json()
    assert ds == [{"name": "syn_t", "length": N, "labelled": True, "kmeansModel": False}]
    r = c.post("/api/runs", json={"dataset": "nope", "mode": "C-1"})
    assert r.status_code == 404 and set(r.json()) == {"code", "message"}
    r = c.post("/api/runs", json={"dataset": "../x", "mode": "C-1"})
    assert r.status_code == 400 and r.json()["code"] == "bad_dataset"
    r = c.post("/api/runs", json={"dataset": "syn_t", "mode": "OMP", "threads": 2, "ranks": 2})
    assert r.status_code == 422 and r.json()["code"] == "bad_request"
    r = c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1", "size": 5})        # unknown field is refused
    assert r.status_code == 422
    assert c.get("/api/runs/zzz/stream").status_code == 404
    r = c.post("/api/runs", json={"dataset": "syn_t", "mode": "OMP", "threads": 4096 * 0 + 256})
    assert r.status_code == 400 and r.json()["code"] == "oversubscribed"


def test_run_streams_start_progress_done_and_lanes_add_up(data, tmp_path):
    c, mgr = make_client(data, tmp_path)
    rid, ev = run_and_collect(c, dataset="syn_t", mode="OMP", threads=4, detector="zscore", allowOversubscribe=True)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "start" and kinds[-1] == "done" and kinds.count("done") == 1
    start, done = ev[0][1], ev[-1][1]["result"]
    prog = [d for k, d in ev if k == "progress"]
    assert len(prog) >= 3
    assert start["mode"] == "OMP-4" and start["nw"] == NW and start["slots"] == 4
    assert len(start["strip"]["min"]) == len(start["strip"]["max"]) <= 800 and start["labelBins"]
    for p in prog:
        assert len(p["threads"]) == 4 and sum(p["threads"]) == p["done"] <= p["total"] == NW
    dones = [p["done"] for p in prog]
    assert dones == sorted(dones)                                    # monotone
    assert prog[-1]["done"] == NW and sum(prog[-1]["threads"]) == NW  # lanes add up to the total at the end
    assert done["countersSum"] == NW and done["ms"] > 0 and done["throughputSps"] > 0
    # flags seen live add up to the flags in the result
    assert prog[-1]["flagsTotal"] == done["flags"] > 0
    assert {f["i"] for p in prog for f in p["flags"]} <= set(range(N))
    assert done["f1"] is not None
    assert not (tmp_path / "runs" / rid).exists()                    # run directory removed after completion


def test_flags_appear_before_the_run_ends_when_scores_are_live(data, tmp_path):
    c, _ = make_client(data, tmp_path, engine=FakeEngine(delay=0.02, steps=20, live=True))
    _, ev = run_and_collect(c, dataset="syn_t", mode="C-1", detector="mad")
    prog = [d for k, d in ev if k == "progress"]
    first = next(i for i, p in enumerate(prog) if p["flags"])
    assert first < len(prog) - 1


def test_not_live_scores_flag_at_the_end_only(data, tmp_path):
    c, _ = make_client(data, tmp_path, engine=FakeEngine(live=False))
    _, ev = run_and_collect(c, dataset="syn_t", mode="C-1")
    prog = [d for k, d in ev if k == "progress"]
    assert all(not p["flags"] for p in prog[:-1])
    assert prog[-1]["flagsTotal"] == ev[-1][1]["result"]["flags"] > 0          # counted once, not twice
    assert prog[-1]["scoresLive"] is False


def test_only_one_run_at_a_time_and_reconnect_replays(data, tmp_path):
    c, mgr = make_client(data, tmp_path, engine=FakeEngine(delay=0.02, steps=15))
    r1 = c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"})
    assert r1.status_code == 202
    r2 = c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"})
    assert r2.status_code == 409 and r2.json()["code"] == "busy"
    # a client that connects, reads a bit and disconnects does not stop the run
    with c.stream("GET", r1.json()["stream"]) as s:
        for i, line in enumerate(s.iter_lines()):
            if i > 6:
                break
    run = mgr.get(r1.json()["runId"])
    assert run.done.wait(15) and run.result and not run.error
    # a late client replays everything; Last-Event-ID resumes after the given event
    with c.stream("GET", r1.json()["stream"]) as s:
        full = frames(s)
    with c.stream("GET", r1.json()["stream"], headers={"Last-Event-ID": str(len(full) - 2)}) as s:
        tail = frames(s)
    assert [k for k, _ in tail] == [full[-1][0]] == ["done"]
    assert c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"}).status_code == 202     # free again


def test_speedup_uses_t1_from_this_session_then_results_csv(data, tmp_path):
    c, mgr = make_client(data, tmp_path)
    _, ev = run_and_collect(c, dataset="syn_t", mode="OMP", threads=2, allowOversubscribe=True)
    assert ev[-1][1]["result"]["speedup"] is None                    # no T1 yet: no speedup is claimed
    (tmp_path / "results.csv").write_text(
        "dataset,n,w,s,detector,mode,total_s,status\n" + "".join(
            f"syn_t,{N},{W},{S},mad,C-1,{v},ok\n" for v in (0.8, 1.0, 1.2)) + f"syn_t,{N},{W},{S},mad,C-1,9,failed\n")
    _, ev = run_and_collect(c, dataset="syn_t", mode="OMP", threads=2, allowOversubscribe=True)
    res = ev[-1][1]["result"]
    assert res["t1Ms"] == pytest.approx(1000.0) and res["speedup"] == pytest.approx(1000.0 / res["ms"])
    assert res["efficiency"] == pytest.approx(res["speedup"] / 2 * 100)
    assert len(c.get("/api/results", params={"mode": "C-1"}).json()) == 4


# ---------------------------------------------------------------- race
def test_race_runs_modes_one_after_another_c1_first(data, tmp_path):
    eng = FakeEngine(delay=0.003, steps=10)
    c, mgr = make_client(data, tmp_path, engine=eng)
    r = c.post("/api/race", json={"modes": ["OMP-2", "OMP-4", "OMP-2", "c-1"], "dataset": "syn_t", "detector": "mad",
                                  "allowOversubscribe": True})
    assert r.status_code == 202 and r.json()["modes"] == ["C-1", "OMP-2", "OMP-4"]     # C-1 first, duplicates dropped
    assert c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"}).status_code == 409      # race holds the lock
    with c.stream("GET", r.json()["stream"]) as s:
        ev = frames(s)
    kinds = [k for k, _ in ev]
    assert kinds[0] == "race_start" and kinds[-1] == "done" and kinds.count("mode_done") == 3
    started = [d["mode"] for k, d in ev if k == "start"]
    assert started == ["C-1", "OMP-2", "OMP-4"]
    # strictly sequential: events of mode k never appear after the start of mode k+1
    order = [d["mode"] for k, d in ev if k in ("start", "progress", "mode_done")]
    firsts = [order.index(m) for m in started]
    lasts = [len(order) - 1 - order[::-1].index(m) for m in started]
    assert all(lasts[i] < firsts[i + 1] for i in range(2))
    assert all(b0 >= a1 for (_, a1), (b0, _) in zip(eng.spans, eng.spans[1:]))                  # DLL calls never overlap
    table = ev[-1][1]["table"]
    assert [row["mode"] for row in table] == ["C-1", "OMP-2", "OMP-4"]
    assert table[0]["speedup"] == pytest.approx(1.0) and table[0]["identical"] is None
    for row in table[1:]:
        assert row["identical"] is True and row["speedup"] == pytest.approx(table[0]["ms"] / row["ms"])
    assert c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"}).status_code == 202      # lock released


def test_race_rejects_bad_mode_and_releases_the_lock(data, tmp_path):
    c, _ = make_client(data, tmp_path)
    r = c.post("/api/race", json={"modes": ["OMP-x"], "dataset": "syn_t"})
    assert r.status_code == 400 and r.json()["code"] == "bad_mode"
    assert c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"}).status_code == 202


def test_run_history_is_capped(data, tmp_path):
    c, mgr = make_client(data, tmp_path, engine=FakeEngine(delay=0, steps=2))
    mgr.cfg.max_runs = 3
    for _ in range(6):
        _, ev = run_and_collect(c, dataset="syn_t", mode="C-1")
        assert ev[-1][0] == "done"
        time.sleep(0.05)
    assert len(mgr.runs) <= 3 + 1
    assert list((tmp_path / "runs").iterdir()) == []


# ---------------------------------------------------------------- real stand-in SDK, Python mode
def test_real_sdk_and_python_modes_end_to_end(data, tmp_path):
    pytest.importorskip("threadwatch.sdk")
    cfg = Settings(data_root=data, runs_root=tmp_path / "runs", results_csv=tmp_path / "r.csv", ui_dir=tmp_path / "x",
                   w=W, s=S, thresh_windows=2_000)
    from fastapi.testclient import TestClient as TC
    app = create_app(cfg)
    c = TC(app)
    x = np.fromfile(data / "syn_t.f32", np.float32)[:3000]                      # keep the Python loop short
    x.tofile(data / "tiny.f32")
    res = {}
    for body in ({"mode": "C-1"}, {"mode": "OMP", "threads": 2, "allowOversubscribe": True}, {"mode": "PY-SEQ"}):
        _, ev = run_and_collect(c, dataset="tiny", detector="zscore", **body)
        assert ev[-1][0] == "done", ev[-1]
        res[ev[-1][1]["result"]["mode"]] = ev[-1][1]["result"]
        assert ev[-1][1]["result"]["countersSum"] == 3000 - W + 1
    assert res["C-1"]["flags"] == res["OMP-2"]["flags"] == res["PY-SEQ"]["flags"]


def test_errors_are_reported_as_events_not_hangs(data, tmp_path):
    class Boom(FakeEngine):
        def compute(self, *a, progress=None, out=None, **k):
            if progress is None:
                return super().compute(*a, progress=progress, out=out, **k)
            raise RuntimeError("kernel exploded")
    c, mgr = make_client(data, tmp_path, engine=Boom())
    _, ev = run_and_collect(c, dataset="syn_t", mode="C-1")
    assert ev[-1][0] == "error" and "kernel exploded" in ev[-1][1]["message"]
    assert c.post("/api/runs", json={"dataset": "syn_t", "mode": "C-1"}).status_code == 202   # lock was released


# ---------------------------------------------------------------- real MPI ranks through the service
MPIEXEC = os.environ.get("TW_MPIEXEC") or shutil.which("mpiexec.openmpi") or shutil.which("mpiexec")


@pytest.mark.skipif(not MPIEXEC, reason="no mpiexec (set TW_MPIEXEC)")
def test_mpi_and_hybrid_runs_from_the_service(data, tmp_path):
    mpiexec = MPIEXEC
    cfg = Settings(data_root=data, runs_root=tmp_path / "runs", results_csv=tmp_path / "r.csv", ui_dir=tmp_path / "x",
                   w=W, s=S, thresh_windows=2_000, mpiexec=mpiexec)
    c = TestClient(create_app(cfg))
    out = {}
    for body in ({"mode": "C-1"}, {"mode": "MPI", "ranks": 3}, {"mode": "HYB", "ranks": 2, "threads": 2}):
        _, ev = run_and_collect(c, dataset="syn_t", detector="mad", allowOversubscribe=True, **body)
        assert ev[-1][0] == "done", ev[-1]
        res = ev[-1][1]["result"]
        out[res["mode"]] = res
        prog = [d for k, d in ev if k == "progress"]
        assert prog[-1]["done"] == NW and sum(prog[-1]["threads"]) == NW
        assert len(prog[-1]["threads"]) == body.get("ranks", 1) * body.get("threads", 1)
    assert out["MPI-3"]["flags"] == out["C-1"]["flags"] == out["HYB-2\u00d72"]["flags"]


@pytest.mark.skipif(not MPIEXEC, reason="no mpiexec (set TW_MPIEXEC)")
def test_mpi_scatter_flags_appear_after_the_gather(data, tmp_path):
    mpiexec = MPIEXEC
    cfg = Settings(data_root=data, runs_root=tmp_path / "runs", results_csv=tmp_path / "r.csv", ui_dir=tmp_path / "x",
                   w=W, s=S, thresh_windows=2_000, mpiexec=mpiexec)
    c = TestClient(create_app(cfg))
    _, ref = run_and_collect(c, dataset="syn_t", detector="mad", mode="C-1")
    _, ev = run_and_collect(c, dataset="syn_t", detector="mad", mode="MPI", ranks=3, inputMode="scatter",
                            allowOversubscribe=True)
    assert ev[-1][0] == "done", ev[-1]
    prog = [d for k, d in ev if k == "progress"]
    assert ev[0][1]["inputMode"] == "scatter" and ev[0][1]["scoresLive"] is False
    assert all(not p["flags"] for p in prog[:-1]) and prog[-1]["flagsTotal"] == ref[-1][1]["result"]["flags"] > 0
    r = c.post("/api/runs", json={"dataset": "syn_t", "mode": "MPI", "ranks": 1, "inputMode": "scatter"})
    assert r.status_code == 422
