"""D-07.7 - what does the dashboard cost? Headless run versus the same run with the dashboard connected.

    python experiments/overhead.py --dataset syn_5M --detector mad --mode OMP --threads 4 --runs 10

Headless : the SDK call alone (no progress counters, no monitor, no server).
Dashboard: the same call through the live service, with progress counters, the 10 Hz monitor, flag scanning and a
           client reading the SSE stream (a thread in this process: slightly pessimistic, a browser is another process).
Runs alternate (headless, dashboard, dashboard, headless, ...) after one warm-up of each, so drift cancels out.
Acceptance (D-07): median with dashboard within 2% of the median headless. Never run during a benchmark campaign.
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from threadwatch.service.app import create_app  # noqa: E402
from threadwatch.service.runs import RunRequest, Settings  # noqa: E402

LIMIT_PCT = 2.0


def headless_ms(mgr, req) -> float:
    cfg = mgr.cfg
    x, _ = mgr.store.open(req.dataset)
    model = mgr.store.model(req.dataset) if req.detector == "kmeans" else None
    if req.ranks == 1:
        t0 = perf_counter()
        mgr.engine.compute(req.detector, x, cfg.w, cfg.s, req.threads, model=model)
        return (perf_counter() - t0) * 1000.0
    cmd = [cfg.mpiexec, "-n", str(req.ranks), sys.executable, "-m", "threadwatch.mpi.run", "--mode", "file",
           "--dataset", req.dataset, "--root", str(cfg.data_root), "--detector", req.detector,
           "--threads", str(req.threads), "--w", str(cfg.w), "--s", str(cfg.s), "--allow-oversubscribe"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    rows = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("TW_ROW")]
    if not rows:
        raise RuntimeError(out.stderr[-300:])
    return float(rows[0]["total_s"]) * 1000.0


def dashboard_ms(base: str, body: dict) -> float:
    r = httpx.post(f"{base}/api/runs", json=body, timeout=30)
    if r.status_code != 202:
        raise RuntimeError(f"service refused the run: {r.text}")
    event = None
    with httpx.stream("GET", base + r.json()["stream"], timeout=None) as s:      # the "browser"
        for line in s.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event in ("done", "error"):
                d = json.loads(line[5:])
                if event == "error":
                    raise RuntimeError(d["message"])
                return float(d["result"]["ms"])
    raise RuntimeError("stream ended without a result")


def iqr_pct(v):
    q = st.quantiles(v, n=4) if len(v) >= 4 else [min(v), st.median(v), max(v)]
    return (q[2] - q[0]) / st.median(v) * 100.0


def measure(settings: Settings, body: dict, runs: int, engine=None, log=print) -> dict:
    app = create_app(settings, engine=engine)
    mgr = app.state.mgr
    req = RunRequest(**body)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    try:
        headless_ms(mgr, req); dashboard_ms(base, body)                           # warm-up: page cache, pools, thresholds
        a, b = [], []
        for i in range(runs):
            order = ("h", "d") if i % 2 == 0 else ("d", "h")
            for k in order:
                (a if k == "h" else b).append(headless_ms(mgr, req) if k == "h" else dashboard_ms(base, body))
            log(f"run {i + 1}/{runs}: headless {a[-1]:.1f} ms, dashboard {b[-1]:.1f} ms")
    finally:
        server.should_exit = True
    ma, mb = st.median(a), st.median(b)
    over = (mb / ma - 1.0) * 100.0
    return {"dataset": req.dataset, "detector": req.detector, "mode": req.label, "runs": runs,
            "headlessMs": a, "dashboardMs": b, "medianHeadlessMs": ma, "medianDashboardMs": mb,
            "overheadPct": over, "iqrHeadlessPct": iqr_pct(a), "iqrDashboardPct": iqr_pct(b),
            "limitPct": LIMIT_PCT, "pass": over <= LIMIT_PCT,
            "note": "SSE client runs as a thread in this process (pessimistic); compute time = DLL call wall time"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--detector", default="mad")
    ap.add_argument("--mode", default="OMP", choices=["C-1", "OMP", "MPI", "HYB"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--allow-oversubscribe", action="store_true",
                    help="only to test the script on a small machine: timings with q x p above the cores are not valid")
    ap.add_argument("--out", default="experiments/out/overhead.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cfg = Settings(data_root=root / "data/series", runs_root=root / "runs", results_csv=root / "experiments/out/results.csv",
                   ui_dir=root / "apps/dashboard", mpiexec=a.mpiexec)
    body = {"dataset": a.dataset, "detector": a.detector, "mode": a.mode,
            "threads": a.threads if a.mode in ("OMP", "HYB") else 1, "ranks": a.ranks if a.mode in ("MPI", "HYB") else 1,
            "allowOversubscribe": a.allow_oversubscribe}
    res = measure(cfg, body, a.runs)
    res["oversubscribed"] = a.allow_oversubscribe
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\n{res['mode']} on {a.dataset} ({a.detector}), {a.runs} runs each: median headless {res['medianHeadlessMs']:.1f} ms, "
          f"with dashboard {res['medianDashboardMs']:.1f} ms -> overhead {res['overheadPct']:+.2f}% "
          f"(limit {LIMIT_PCT:.0f}%, IQR {res['iqrHeadlessPct']:.1f}% / {res['iqrDashboardPct']:.1f}%): "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
