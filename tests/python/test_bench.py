"""D-08 - matrix, result file, runner (interleaving, resume, failures, timeout, crash, reruns) and summaries.

The harness logic is tested with fast deterministic NumPy backends; the end-to-end tests at the bottom use the real
SDK wrapper and real `mpiexec` ranks (skipped if no MPI runtime is available).
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from threadwatch.bench import matrix as mx
from threadwatch.bench.rows import (COLUMNS, ResultsFile, make_row, parse_run_id, run_id)
from threadwatch.bench.runner import (EX_TIMEOUT, CBackend, Context, Runner, kill_tree, sample_stack, supervise)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import summarize  # noqa: E402

W = 32


# ------------------------------------------------------------------ fixtures and fakes
def base_cfg():
    cfg = mx.load_config(ROOT / "experiments/matrix.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["window"] = {"w": W, "s": 1}
    cfg["protocol"].update(repeats=3, pause_between_blocks_s=0, fit_sample=300, max_background_cpu_pct=100,
                           allow_oversubscribe=True)
    cfg["modes"]["PY-SEQ"]["max_n"] = {"zscore": "1M", "mad": "1M", "kmeans": "1M", "iforest": "1M"}
    cfg["iforest"]["max_n"] = "1M"
    cfg["extras"] = []
    cfg["timeouts"].update(minimum_s=30)
    return cfg


class NumpyBackend:
    """Same maths as the detectors, vectorised and instant. Optional hooks to simulate trouble."""
    def __init__(self, name="c"):
        self.calls = Counter()
        self.name = name
        self.hook = None                                  # callable(kind, det, p, nth_call) -> may raise or sleep

    def _wins(self, x, w, s):
        nw = (len(x) - w) // s + 1
        return np.lib.stride_tricks.sliding_window_view(np.asarray(x), w)[::s][:nw].astype(np.float64)

    def _ms(self):
        return None if self.name == "py" else 1.0

    def score(self, det, x, w, s, p):
        self.calls["score", det, p] += 1
        if self.hook: self.hook("score", det, p, self.calls["score", det, p])
        win = self._wins(x, w, s)
        if det == "zscore":
            sd = win.std(1, ddof=1)
            sc = np.where(sd > 1e-12, np.abs(win[:, -1] - win.mean(1)) / np.where(sd > 1e-12, sd, 1), 0.0)
        else:
            m = np.median(win, 1)
            sc = np.abs(win[:, -1] - m) / (1.4826 * np.median(np.abs(win - m[:, None]), 1) + 1e-12)
        return sc.astype(np.float32), self._ms()

    def features(self, x, w, s, p):
        self.calls["features", "-", p] += 1
        if self.hook: self.hook("features", "-", p, self.calls["features", "-", p])
        win = self._wins(x, w, s)
        return np.stack([win.mean(1), win.std(1, ddof=1), win.min(1), win.max(1)], 1).astype(np.float32), self._ms()

    def kmeans(self, F, C, scale, p):
        d = ((F[:, None, :].astype(np.float64) - C[None].astype(np.float64)) * scale) ** 2
        return np.sqrt(d.sum(2).min(1)).astype(np.float32), self._ms()


def make_series(root, name, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = (50 + rng.normal(0, 1, n)).astype(np.float32)
    y = np.zeros(n, np.uint8)
    for c in (n // 3, 2 * n // 3):
        x[c:c + 30] += 15
        y[c:c + 30] = 1
    root.mkdir(parents=True, exist_ok=True)
    x.tofile(root / f"{name}.f32")
    y.tofile(root / f"{name}.labels.u8")


@pytest.fixture
def env(tmp_path):
    root, out = tmp_path / "series", tmp_path / "out"
    make_series(root, "tiny_A")
    make_series(root, "tiny_B", 4000, seed=2)
    return root, out


def make_runner(env, cfg, datasets=("tiny_A",), detectors=("zscore",), modes=None, logical=4, c=None, py=None, **kw):
    root, out = env
    cfg = copy.deepcopy(cfg)
    cfg["datasets"], cfg["detectors"] = list(datasets), list(detectors)
    c = c or NumpyBackend("c")
    py = py or NumpyBackend("py")
    ctx = Context(cfg, root, out, c, py)
    cells = mx.build_cells(cfg, logical, ctx.n_of)
    if modes:
        cells = mx.filter_cells(cells, modes=modes)
    results = ResultsFile(out / "results.csv")
    runner = Runner(cfg, ctx, results, out, logical, log=lambda m: None, **kw)
    return runner, cells, results, ctx


def cfg_with(base, **modes):
    cfg = copy.deepcopy(base)
    cfg["modes"] = {k: v for k, v in modes.items()}
    return cfg


C_AND_OMP = dict(**{"PY-SEQ": {"max_n": {"zscore": "1M", "mad": "1M", "kmeans": "1M", "iforest": "1M"}}, "C-1": {},
                    "OMP": {"threads": [2, 4]}})


# ------------------------------------------------------------------ matrix
def test_parse_size_and_auto_threads():
    assert [mx.parse_size(v) for v in ("1M", "2m", "500k", 1234, "1.5M")] == [1_000_000, 2_000_000, 500_000, 1234, 1_500_000]
    assert mx.auto_threads(12) == [2, 4, 8, 12] and mx.auto_threads(8) == [2, 4, 8] and mx.auto_threads(1) == []
    with pytest.raises(ValueError):
        mx.parse_size("lots")


def test_default_matrix_follows_the_spec():
    cfg = mx.load_config(ROOT / "experiments/matrix.yaml")
    cells = mx.build_cells(cfg, 12)
    keys = [c.key for c in cells]
    assert len(keys) == len(set(keys))                                              # unique cells
    by = lambda **f: [c for c in cells if all(getattr(c, k) == v for k, v in f.items())]
    assert {c.dataset for c in by(mode="PY-SEQ", detector="mad")} == {"syn_1M"}      # MAD in Python up to 1M
    assert {c.dataset for c in by(mode="PY-SEQ", detector="zscore")} == {"syn_1M", "syn_2M"}
    assert {c.dataset for c in by(detector="iforest")} == {"syn_1M", "syn_2M"}       # Isolation Forest up to 2M
    assert {c.mode for c in by(detector="iforest")} == {"PY-SEQ", "C-1", "OMP"}      # no MPI for scikit-learn scoring
    assert sorted({c.p for c in by(mode="OMP")}) == [2, 4, 8, 12]
    assert sorted({(c.q, c.p) for c in by(mode="HYB")}) == [(2, 2), (2, 4)]
    assert sorted({c.q for c in by(mode="MPI")}) == [2, 4]
    assert {c.dataset for c in by(mode="MPI-scatter")} == {"syn_2M", "syn_10M"}      # scatter on two sizes
    assert {c.dataset for c in by(tag="second_stride")} == {"syn_5M"} and {c.s for c in by(tag="second_stride")} == {16}
    assert not by(tag="second_stride", mode="PY-SEQ")
    assert all(c.q * c.p <= 12 for c in cells)


def test_cells_are_clipped_to_the_logical_cores():
    cfg = mx.load_config(ROOT / "experiments/matrix.yaml")
    small = mx.build_cells(cfg, 4)
    assert max(c.q * c.p for c in small) <= 4 and not [c for c in small if c.mode == "HYB" and c.p == 4]
    assert sorted({c.p for c in small if c.mode == "OMP"}) == [2, 4]


def test_block_has_c1_first_and_rotates_the_order():
    cfg = mx.load_config(ROOT / "experiments/matrix.yaml")
    blocks = mx.blocks_of(mx.build_cells(cfg, 12))
    b = blocks[0]
    assert b.cells[0].mode == "C-1"
    assert [c.key for c in b.order(0)] == [c.key for c in b.cells]                    # warm-up round: canonical
    first = [b.order(r)[0].key for r in range(1, len(b.cells) + 1)]
    assert len(set(first)) == len(b.cells)                                            # everybody leads exactly once
    last = [b.order(r)[-1].key for r in range(1, len(b.cells) + 1)]
    assert len(set(last)) == len(b.cells)
    assert all(sorted(c.key for c in b.order(r)) == sorted(c.key for c in b.cells) for r in range(6))


def test_timeouts_scale_with_size_and_have_a_floor():
    cfg = mx.load_config(ROOT / "experiments/matrix.yaml")
    cells = {c.key: c for c in mx.build_cells(cfg, 12)}
    mad20 = cells["syn_20M|mad|C-1|w256|s1"]
    assert mad20.timeout_s == pytest.approx(4 * 6.3 * 20)
    assert cells["syn_1M|zscore|C-1|w256|s1"].timeout_s == 120                        # floor
    est = mx.estimate(cfg, list(cells.values()))
    assert est["runs"] == len(cells) * 6 and est["upper_bound_s"] > 3600


# ------------------------------------------------------------------ result file
def test_run_id_roundtrip_and_row_columns(tmp_path):
    assert parse_run_id(run_id("a|b", 3)) == ("a|b", 3, 0) and parse_run_id(run_id("a|b", 3, 2)) == ("a|b", 3, 2)
    c = mx.Cell("tiny_A", 3000, "mad", "OMP", 1, 4, W, 1)
    row = make_row(c, 2, "ok", total_s=2.0, f1=0.5)
    assert list(row) == COLUMNS and row["throughput_sps"] == 1500.0 and row["p_threads"] == 4
    with pytest.raises(KeyError):
        make_row(c, 1, "ok", speedup=3.0)                                             # no derived columns in the raw file


def test_numpy_scalars_are_written_as_plain_numbers(tmp_path):
    """Regression: repr(np.float64(1.0)) is 'np.float64(1.0)' in NumPy 2 and used to corrupt metric columns."""
    rf = ResultsFile(tmp_path / "r.csv")
    c = mx.Cell("tiny_A", 3000, "mad", "C-1", 1, 1, W, 1)
    rf.append(make_row(c, 1, "ok", total_s=np.float64(1.5), f1=np.float32(0.25), event_recall=np.float64("nan"),
                       precision=np.float64(1.0), identical=np.bool_(True), dll_ms=np.int64(7)))
    text = (tmp_path / "r.csv").read_text()
    assert "np." not in text and "float" not in text
    r = rf.rows()[0]
    assert (r["total_s"], r["f1"], r["event_recall"], r["precision"], r["identical"], r["dll_ms"]) == ("1.5", "0.25", "", "1.0", "True", "7")
    assert pd.read_csv(tmp_path / "r.csv")["f1"].dtype == "float64"


def test_results_file_append_resume_and_closed(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c = mx.Cell("tiny_A", 3000, "mad", "C-1", 1, 1, W, 1)
    rf.append(make_row(c, 0, "warmup", total_s=1.0))
    rf.append(make_row(c, 1, "ok", total_s=1.0, identical=True, f1=float("nan"), note='has, comma "quotes"'))
    rf.append(make_row(c, 2, "timeout", note="slow"))
    rows = rf.rows()
    assert [r["status"] for r in rows] == ["warmup", "ok", "timeout"] and rows[1]["f1"] == "" and rows[1]["identical"] == "True"
    assert rows[1]["note"] == 'has, comma "quotes"'
    assert rf.done() == {(c.key, 0, 0), (c.key, 1, 0), (c.key, 2, 0)} and rf.closed() == {(c.key, 0)}
    assert rf.next_batch(c.key) == 1 and rf.next_batch("other") == 0
    assert (tmp_path / "r.csv").read_text().splitlines()[0].split(",") == COLUMNS


def test_rewrite_status_is_atomic_and_keeps_a_backup(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c = mx.Cell("tiny_A", 3000, "mad", "C-1", 1, 1, W, 1)
    for r in range(3):
        rf.append(make_row(c, r, "ok", total_s=1.0 + r))
    before = (tmp_path / "r.csv").read_text()
    n = rf.rewrite_status([run_id(c.key, 1)], "superseded", "superseded by rerun batch 1", tmp_path / "bk")
    assert n == 1
    rows = rf.rows()
    assert [r["status"] for r in rows] == ["ok", "superseded", "ok"] and "rerun batch 1" in rows[1]["note"]
    assert rows[1]["total_s"] == "2.0"                                                # nothing else touched
    [bk] = list((tmp_path / "bk").iterdir())
    assert bk.read_text() == before


# ------------------------------------------------------------------ runner: normal campaign
def test_one_block_is_interleaved_complete_and_equivalent(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2, 4]}})
    runner, cells, results, _ = make_runner(env, cfg, detectors=("zscore",))
    assert [c.label for c in cells] == ["C-1", "OMP-2", "OMP-4"]
    runner.run(cells)
    rows = results.rows()
    reps = int(cfg["protocol"]["repeats"])
    assert len(rows) == len(cells) * (1 + reps)
    for c in cells:
        mine = [r for r in rows if r["run_id"].startswith(c.key + "#")]
        assert [r["status"] for r in mine] == ["warmup"] + ["ok"] * reps and [int(r["repeat"]) for r in mine] == list(range(reps + 1))
        assert all(r["identical"] == "True" for r in mine)                            # same flags as C-1
        assert all(r["f1"] != "" and r["precision"] != "" and r["event_recall"] != "" for r in mine)   # quality in every row
        assert all(float(r["total_s"]) > 0 and r["dll_ms"] != "" for r in mine)
    # interleaved: round by round, never all repeats of one mode back to back
    order = [(int(r["repeat"]), r["mode"] + r["p_threads"]) for r in rows]
    assert [rp for rp, _ in order] == sorted(rp for rp, _ in order)                  # rounds follow each other
    assert len({m for rp, m in order if rp == 1}) == 3
    assert len({m for _, m in order[:3]}) == 3                                       # the first three runs are three different modes
    assert [m for rp, m in order if rp == 1] != [m for rp, m in order if rp == 2]    # rotated order
    assert os.path.exists(env[1] / "thresholds.json") and os.path.exists(env[1] / "campaign.log")
    assert not (env[1] / "current.json").exists()
    assert set(os.listdir(env[1] / "tmp")) == set()                                  # temp files removed


def test_phases_per_detector_and_mode(env):
    cfg = cfg_with(base_cfg(), **C_AND_OMP)
    runner, cells, results, _ = make_runner(env, cfg, detectors=("zscore", "kmeans", "iforest"))
    runner.run(cells)
    df = pd.DataFrame(results.rows())
    pick = lambda det, lab: df[(df.detector == det) & (df["run_id"].str.contains(f"|{lab}|", regex=False)) & (df.status == "ok")].iloc[0]
    z = pick("zscore", "C-1")
    assert z.features_s == "" and z.score_s != "" and float(z.total_s) >= float(z.score_s)
    k = pick("kmeans", "OMP-2")
    assert k.features_s != "" and k.score_s != "" and float(k.total_s) >= float(k.features_s) + float(k.score_s)
    py = pick("kmeans", "PY-SEQ")
    assert py.dll_ms == "" and py.features_s != ""                                  # no DLL time for Python
    i = pick("iforest", "OMP-4")
    assert "n_jobs=4" in i.note and i.features_s != "" and i.identical in ("True", "False")
    assert (df[df["mode"] == "MPI"].shape[0]) == 0


def test_thresholds_are_computed_once_and_reused(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}})
    c = NumpyBackend("c")
    runner, cells, results, ctx = make_runner(env, cfg, c=c)
    runner.run(cells)
    n_calls = sum(c.calls.values())
    thr1 = json.loads((env[1] / "thresholds.json").read_text())
    assert thr1["tiny_A|zscore|32"]["sample_windows"] == 300 and thr1["tiny_A|zscore|32"]["seed"] == 0
    c2 = NumpyBackend("c")
    runner2, cells2, results2, ctx2 = make_runner(env, cfg, c=c2)
    ctx2.prepare("tiny_A", W)
    assert sum(c2.calls.values()) == 0                                               # loaded from disk, nothing recomputed
    assert json.loads((env[1] / "thresholds.json").read_text()) == thr1


def test_fit_sample_is_deterministic():
    x = np.random.default_rng(0).normal(size=5000).astype(np.float32)
    a, b = sample_stack(x, W, 200, 0), sample_stack(x, W, 200, 0)
    assert len(a) == 200 * W and np.array_equal(a, b) and not np.array_equal(a, sample_stack(x, W, 200, 1))
    assert len(sample_stack(x, W, 10**6, 0)) == (5000 - W + 1) * W                   # small series: all windows


# ------------------------------------------------------------------ resume, failures
class Interrupt(BaseException):
    pass


def test_interrupted_campaign_resumes_without_repeats_or_gaps(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}})
    c = NumpyBackend("c")
    seen = {"n": 0}
    def hook(kind, det, p, nth):
        seen["n"] += 1
        if seen["n"] == 5:
            raise Interrupt()                                                        # Ctrl-C / power cut in the middle
    c.hook = hook
    runner, cells, results, _ = make_runner(env, cfg, c=c)
    with pytest.raises(Interrupt):
        runner.run(cells)
    first = results.rows()
    assert 0 < len(first) < len(cells) * 4
    # restart: a fresh process would rebuild everything from the files
    runner2, cells2, results2, _ = make_runner(env, cfg)
    runner2.run(cells2)
    rows = results2.rows()
    ids = [r["run_id"] for r in rows]
    assert len(ids) == len(set(ids)) == len(cells) * 4                              # no duplicate, nothing missing
    assert ids[:len(first)] == [r["run_id"] for r in first]                          # the old rows are untouched
    assert all(r["identical"] == "True" for r in rows)                               # the reference was rebuilt after the restart
    assert runner2.run(cells2) == 0                                                  # a finished campaign does nothing


def test_failed_cell_is_closed_recorded_and_not_retried(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2, 4]}})
    c = NumpyBackend("c")
    def hook(kind, det, p, nth):
        if p == 2 and nth == 3:                                                      # OMP-2 breaks on its third call
            raise RuntimeError("kernel returned TW_E_INTERNAL")
    c.hook = hook
    runner, cells, results, _ = make_runner(env, cfg, c=c)
    runner.run(cells)
    rows = results.rows()
    omp2 = [r for r in rows if "|OMP-2|" in r["run_id"]]
    assert [r["status"] for r in omp2] == ["warmup", "ok", "error"]                  # stopped after the failure
    assert "TW_E_INTERNAL" in omp2[-1]["note"] and omp2[-1]["total_s"] == ""
    assert all(sum(r["status"] == "ok" for r in rows if f"|{lab}|" in r["run_id"]) == 3 for lab in ("C-1", "OMP-4"))
    n = len(rows)
    runner2, cells2, results2, _ = make_runner(env, cfg)                              # restart: the failure is not retried
    runner2.run(cells2)
    assert len(results2.rows()) == n
    cov = summarize.coverage(summarize.load(results.path), cells, 3)
    bad = cov[~cov["complete"]]
    assert list(bad["label"]) == ["OMP-2"] and bad.iloc[0]["reason"].startswith("error at repeat 2")


def test_in_process_timeout_writes_one_row_and_aborts(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}})
    c = NumpyBackend("c")
    c.hook = lambda kind, det, p, nth: time.sleep(1.0) if nth == 2 else None
    exits = []
    runner, cells, results, _ = make_runner(env, cfg, c=c, abort=exits.append)
    cells = [mx.Cell(x.dataset, x.n, x.detector, x.mode, x.q, x.p, x.w, x.s, 0.3) for x in cells]
    runner.run(cells)
    rows = results.rows()
    t = [r for r in rows if r["status"] == "timeout"]
    assert len(t) == 1 and "timeout after" in t[0]["note"] and exits == [EX_TIMEOUT]
    assert [r["status"] for r in rows if r["run_id"].startswith(cells[0].key)] == ["timeout"]   # the late result was not also written
    assert (cells[0].key, 0) in results.closed()


def test_slow_reference_pass_after_resume_cannot_cause_a_false_timeout(env):
    """The watchdog only covers the timed call: the untimed C-1 reference pass after a resume has no timeout of its own."""
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}})
    c = NumpyBackend("c")
    runner, cells, results, _ = make_runner(env, cfg, c=c)
    runner.run([x for x in cells if x.mode == "C-1"][:1])
    c2 = NumpyBackend("c")
    c2.hook = lambda kind, det, p, nth: time.sleep(0.5) if (p == 1) else None       # the reference pass (C-1, p=1) is slow
    exits = []
    runner2, cells2, results2, _ = make_runner(env, cfg, c=c2, abort=exits.append)
    cells2 = [mx.Cell(x.dataset, x.n, x.detector, x.mode, x.q, x.p, x.w, x.s, 0.3) for x in cells2 if x.mode == "OMP"]
    runner2.run(cells2)
    assert exits == [] and not [r for r in results2.rows() if r["status"] == "timeout"]
    assert all(r["identical"] == "True" for r in results2.rows() if "|OMP-2|" in r["run_id"])


def test_supervisor_records_crash_and_restarts(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    rf = ResultsFile(out / "results.csv")
    cell = mx.Cell("tiny_A", 3000, "mad", "OMP", 1, 4, W, 1)
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import json, os, sys, pathlib
        out = pathlib.Path({str(out)!r}); state = out / "attempt"
        n = int(state.read_text()) if state.exists() else 0
        state.write_text(str(n + 1))
        if n == 0:     # first start: die in the middle of a run (segfault-like)
            (out / "current.json").write_text(json.dumps({{"cell": {json.dumps(cell.as_dict())}, "repeat": 2, "batch": 0}}))
            os._exit(139)
        if n == 1:     # second start: a watchdog timeout (the row is already written by the harness)
            os._exit({EX_TIMEOUT})
        sys.exit(0)
    """))
    logs = []
    rc = supervise([sys.executable, str(script)], out, rf, log=logs.append)
    assert rc == 0 and (out / "attempt").read_text() == "3"
    [crash] = rf.rows()
    assert crash["status"] == "crash" and crash["repeat"] == "2" and "0x8b" in crash["note"] and crash["total_s"] == ""
    assert not (out / "current.json").exists()
    # a failure outside any run is not retried forever
    bad = tmp_path / "bad.py"; bad.write_text("import sys; sys.exit(3)")
    assert supervise([sys.executable, str(bad)], out, rf, log=logs.append) == 3


def test_kill_tree_kills_children_too(tmp_path):
    flag = tmp_path / "alive"
    (tmp_path / "grandchild.py").write_text(textwrap.dedent(f"""
        import pathlib, time
        while True:
            pathlib.Path({str(flag)!r}).write_text(str(time.time()))
            time.sleep(0.05)
    """))
    (tmp_path / "child.py").write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, {str(tmp_path / "grandchild.py")!r}])
        time.sleep(60)
    """))
    p = subprocess.Popen([sys.executable, str(tmp_path / "child.py")])
    for _ in range(100):
        if flag.exists(): break
        time.sleep(0.05)
    assert flag.exists()
    kill_tree(p)
    p.wait(5)
    t = flag.read_text(); time.sleep(0.4)
    assert flag.read_text() == t                                                      # the grandchild stopped writing


# ------------------------------------------------------------------ reruns
def test_rerun_supersedes_keeps_everything_and_summaries_use_the_new_batch(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}})
    runner, cells, results, _ = make_runner(env, cfg)
    runner.run(cells)
    n_before = len(results.rows())
    omp = [c for c in cells if c.mode == "OMP"]
    n = runner.rerun(omp, {omp[0].key: "cv 14.2%"})
    rows = results.rows()
    assert n == 4 and len(rows) == n_before + 4                                      # nothing deleted, 1 warm-up + 3 new rows
    old = [r for r in rows if r["run_id"].startswith(omp[0].key + "#") and "@" not in r["run_id"]]
    new = [r for r in rows if r["run_id"].endswith("@1")]
    assert [r["status"] for r in old] == ["superseded"] * 4 and "rerun batch 1 (cv 14.2%)" in old[0]["note"]
    assert [r["status"] for r in new] == ["warmup", "ok", "ok", "ok"] and "rerun batch 1" in new[0]["note"]
    assert os.listdir(env[1] / "backups")
    st = summarize.cell_stats(summarize.load(results.path))
    assert int(st[st.label == "OMP-2"].iloc[0]["runs"]) == 3                         # statistics see the new batch only
    assert runner.rerun(omp, {}) == 4 and any(r["run_id"].endswith("@2") for r in results.rows())


# ------------------------------------------------------------------ summaries
def synthetic(times_by_label, n=1_000_000, ds="syn_1M", det="mad", identical=True):
    rows = []
    for lab, times in times_by_label.items():
        mode, q, p = {"C-1": ("C-1", 1, 1), "OMP-2": ("OMP", 1, 2), "OMP-4": ("OMP", 1, 4), "PY-SEQ": ("PY-SEQ", 1, 1)}[lab]
        c = mx.Cell(ds, n, det, mode, q, p, 256, 1)
        rows.append(make_row(c, 0, "warmup", total_s=times[0], identical=identical, f1=0.4, event_recall=0.5))
        for i, t in enumerate(times, 1):
            rows.append(make_row(c, i, "ok", total_s=t, identical=identical, f1=0.4, event_recall=0.5, dll_ms=t * 1000))
    return rows


def frame(tmp_path, rows):
    rf = ResultsFile(tmp_path / "r.csv")
    for r in rows:
        rf.append(r)
    return summarize.load(rf.path)


def test_cell_stats_speedup_efficiency_cv(tmp_path):
    df = frame(tmp_path, synthetic({"C-1": [10.0, 10.0, 10.0, 10.0, 10.0], "OMP-4": [2.5, 2.5, 2.5, 2.5, 2.5],
                                    "OMP-2": [5.0, 5.0, 5.0, 8.0, 5.0]}))
    st = summarize.cell_stats(df, 10.0).set_index("label")
    assert st.loc["OMP-4", "speedup"] == pytest.approx(4.0) and st.loc["OMP-4", "efficiency"] == pytest.approx(1.0)
    assert st.loc["OMP-2", "median_s"] == 5.0 and st.loc["OMP-2", "min_s"] == 5.0 and st.loc["OMP-2", "max_s"] == 8.0
    assert st.loc["OMP-2", "iqr_s"] == 0.0 and st.loc["OMP-2", "cv_pct"] > 10 and bool(st.loc["OMP-2", "flag_rerun"])
    assert not bool(st.loc["C-1", "flag_rerun"]) and st.loc["C-1", "throughput_sps"] == pytest.approx(1e5)
    assert st.loc["OMP-4", "runs"] == 5                                              # the warm-up is excluded
    assert list(summarize.noisy_cells(df, 10.0)["label"]) == ["OMP-2"]


def test_failed_warmup_and_superseded_rows_are_excluded_from_statistics(tmp_path):
    rows = synthetic({"C-1": [10.0] * 5, "OMP-2": [5.0] * 5})
    rows.append(make_row(mx.Cell("syn_1M", 1_000_000, "mad", "OMP", 1, 2, 256, 1), 6, "timeout", note="slow"))
    df = frame(tmp_path, rows)
    st = summarize.cell_stats(df).set_index("label")
    assert st.loc["OMP-2", "runs"] == 5


def test_drift_is_detected_in_repeated_c1_runs(tmp_path):
    hot = frame(tmp_path, synthetic({"C-1": [10.0, 10.2, 10.6, 11.0, 11.4, 11.8]}))      # slowing down: throttling?
    d = summarize.drift_table(hot)
    assert bool(d.iloc[0]["flag"]) and d.iloc[0]["drift_pct"] > 5
    flat = frame(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, synthetic({"C-1": [10.0, 10.1, 9.9, 10.0, 10.05]}))
    assert not bool(summarize.drift_table(flat).iloc[0]["flag"])


def test_coverage_lists_missing_cells_with_reasons(tmp_path):
    rows = synthetic({"C-1": [10.0] * 5, "OMP-2": [5.0, 5.0, 5.0]})                    # OMP-2: only 3 of 5
    c4 = mx.Cell("syn_1M", 1_000_000, "mad", "OMP", 1, 4, 256, 1)
    rows += [make_row(c4, 0, "warmup", total_s=3.0), make_row(c4, 1, "timeout", note="timeout after 99 s")]
    df = frame(tmp_path, rows)
    cells = [mx.Cell("syn_1M", 1_000_000, "mad", m, 1, p, 256, 1) for m, p in (("C-1", 1), ("OMP", 2), ("OMP", 4), ("OMP", 8))]
    cov = summarize.coverage(df, cells, 5).set_index("label")
    assert bool(cov.loc["C-1", "complete"]) and cov.loc["C-1", "reason"] == ""
    assert cov.loc["OMP-2", "reason"].startswith("incomplete: 3 of 5")
    assert cov.loc["OMP-4", "reason"].startswith("timeout at repeat 1")
    assert cov.loc["OMP-8", "reason"] == "not run yet"


def test_perf_doc_and_csvs_are_generated_from_the_csv(env, tmp_path):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}})
    runner, cells, results, _ = make_runner(env, cfg, datasets=("tiny_A", "tiny_B"))
    runner.run(cells)
    cfg_path = tmp_path / "m.yaml"
    import yaml
    import machine_info
    m = machine_info.collect("gcc -O3 -fopenmp")
    m["logical_cores"] = 4                                       # the machine that ran the campaign (expected cells depend on it)
    (env[1] / "machine.json").write_text(json.dumps(m))
    c2 = copy.deepcopy(cfg); c2["datasets"], c2["detectors"] = ["tiny_A", "tiny_B"], ["zscore"]
    cfg_path.write_text(yaml.safe_dump(c2))
    info = summarize.write_all(tmp_path, cfg_path, env[1], tmp_path / "docs/perf.md")
    assert info["cells_expected"] == 4 and info["cells_complete"] == 4 and info["flagged"] >= 0
    for f in ("summary_zscore.csv", "summary_cells.csv", "coverage.csv", "drift.csv"):
        assert (env[1] / f).exists()
    piv = pd.read_csv(env[1] / "summary_zscore.csv")
    assert list(piv.columns) == ["dataset", "C-1", "OMP-2"] and list(piv["dataset"]) == ["tiny_A", "tiny_B"]
    doc = (tmp_path / "docs/perf.md").read_text()
    for head in ("## Machine", "## Protocol", "## Coverage", "## zscore: median seconds", "## Spread", "## Equivalence with C-1", "## Drift"):
        assert head in doc
    assert "4 of 4 cells" in doc and "No cell is missing" in doc and "gcc -O3 -fopenmp" in doc


def test_perf_doc_without_data_says_so(tmp_path):
    info = summarize.write_all(tmp_path, ROOT / "experiments/matrix.yaml", tmp_path / "out", tmp_path / "docs/perf.md")
    doc = (tmp_path / "docs/perf.md").read_text()
    assert info["rows"] == 0 and "No campaign data yet" in doc and "run_matrix.py" in doc


# ------------------------------------------------------------------ machine record
def test_machine_record_has_what_the_protocol_needs():
    import machine_info
    m = machine_info.collect("gcc -O3")
    for k in ("os", "processor", "logical_cores", "physical_cores", "ram_gb", "python", "packages", "mpi", "dll", "fingerprint",
              "power_plan", "on_battery", "thread_env", "captured_at"):
        assert k in m
    assert m["dll"]["build_note"] == "gcc -O3" and m["packages"]["numpy"] and m["fingerprint"]["logical_cores"] == m["logical_cores"]
    assert machine_info.collect()["fingerprint"] == m["fingerprint"]                # stable between calls


# ------------------------------------------------------------------ end to end: real SDK wrapper, real mpiexec
MPIEXEC = os.environ.get("TW_MPIEXEC") or shutil.which("mpiexec.openmpi") or shutil.which("mpiexec")
needs_sdk = pytest.mark.skipif(not (ROOT / "python/threadwatch/sdk.py").exists() and
                               not __import__("importlib").util.find_spec("threadwatch.sdk"), reason="no SDK wrapper")


@needs_sdk
@pytest.mark.skipif(not MPIEXEC, reason="no mpiexec")
def test_end_to_end_with_real_mpiexec_ranks(env):
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}, "MPI": {"ranks": [2], "scatter_on": ["tiny_A"]},
                                  "HYB": {"shapes": [[2, 2]]}})
    cfg["protocol"].update(repeats=2)
    cfg["mpiexec"] = MPIEXEC
    runner, cells, results, ctx = make_runner(env, cfg, detectors=("zscore", "kmeans"), c=CBackend(), py=NumpyBackend("py"), logical=4)
    assert {c.mode for c in cells} == {"C-1", "OMP", "MPI", "MPI-scatter", "HYB"}
    runner.run(cells)
    df = pd.DataFrame(results.rows())
    assert len(df) == len(cells) * 3 and set(df.status) == {"warmup", "ok"}
    assert (df.identical == "True").all()                                            # OMP, MPI, scatter and HYB equal C-1
    mpi = df[df["mode"].isin(["MPI", "MPI-scatter", "HYB"]) & (df.status == "ok")]
    assert (mpi.comm_s != "").all() and (mpi.q_ranks.astype(int) >= 2).all() and (mpi.f1 != "").all()
    sc = df[df["mode"] == "MPI-scatter"]
    assert len(sc) and (sc.note.str.contains("input=scatter")).all()
    km = df[(df.detector == "kmeans") & df["mode"].isin(["MPI", "HYB"])]
    assert km.note.str.contains("features_s not separable").all()
    assert not (env[1] / "tmp").exists() or not list((env[1] / "tmp").iterdir())


@needs_sdk
@pytest.mark.skipif(not MPIEXEC, reason="no mpiexec")
def test_a_differing_mpi_result_is_reported_not_hidden(env):
    """If an MPI run does not match C-1, `identical` is False and the quality comes from the MPI scores themselves."""
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "MPI": {"ranks": [2]}})
    cfg["protocol"].update(repeats=1)
    cfg["mpiexec"] = MPIEXEC
    runner, cells, results, ctx = make_runner(env, cfg, c=CBackend(), logical=4)
    c1, mpi = cells
    runner.run([c1])
    prep = ctx.prepare("tiny_A", W)
    from threadwatch.bench.runner import BlockState
    st = BlockState(prep)
    st.ref = CBackend().score("zscore", ctx.series("tiny_A"), W, 1, 1)[0].copy()
    st.ref[100] += 5.0                                                               # tamper with the reference
    st.quality = ctx.quality("tiny_A", st.ref, prep.thresholds["zscore"], W, 1)
    block = mx.blocks_of(cells)[0]
    runner.execute_one(mpi, 1, 0, st, block)
    row = [r for r in results.rows() if "|MPI-2|" in r["run_id"]][-1]
    assert row["status"] == "ok" and row["identical"] == "False" and row["f1"] != ""


@needs_sdk
@pytest.mark.skipif(not MPIEXEC, reason="no mpiexec")
def test_run_matrix_command_end_to_end_and_resume(env, tmp_path):
    """The whole campaign from one command, under the supervisor; a second start does nothing."""
    import yaml
    cfg = cfg_with(base_cfg(), **{"C-1": {}, "OMP": {"threads": [2]}, "MPI": {"ranks": [2]}})
    cfg.update(datasets=["tiny_A"], detectors=["zscore"], mpiexec=MPIEXEC)
    cfg["protocol"].update(repeats=2)
    m = tmp_path / "m.yaml"; m.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "campaign"
    env_vars = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]),
                "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1", "OMPI_MCA_rmaps_base_oversubscribe": "1"}
    cmd = [sys.executable, str(ROOT / "experiments/run_matrix.py"), "--matrix", str(m), "--data-root", str(env[0]), "--out", str(out),
           "--logical-cores", "4", "--skip-preflight"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env_vars, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    rows = ResultsFile(out / "results.csv").rows()
    assert len(rows) == 3 * 3 and (out / "machine.json").exists()
    p2 = subprocess.run(cmd, capture_output=True, text=True, env=env_vars, timeout=300)
    assert p2.returncode == 0 and len(ResultsFile(out / "results.csv").rows()) == 9      # resumed: nothing repeated
    dry = subprocess.run(cmd + ["--dry-run"], capture_output=True, text=True, env=env_vars, timeout=60)
    assert "3 cells" in dry.stdout and "holds 9 rows" in dry.stdout
