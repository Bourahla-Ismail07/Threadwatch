"""D-05.5 / D-05.6: first OpenMP scaling table and the MAD schedule comparison.

For every kernel and thread count: warm-up, then `--repeats` runs (default 5); T1 is re-measured in the
same session; every parallel run is compared with the 1-thread output (np.array_equal). Speedup uses the
Python wall-clock timer (end-to-end); the DLL's own parallel-region time (`c_ms`) is recorded too.
Run on a quiet machine (plugged in, other apps closed, OneDrive paused), WITHOUT pinning to one CPU:
    python experiments/omp_quick_scaling.py                 # syn_5M, about 10 minutes
    python experiments/omp_quick_scaling.py --dataset syn_1M --repeats 3      # quick look
Writes experiments/out/omp-scaling.csv, experiments/out/omp-schedule.csv and docs/omp-notes.md.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # library threads must not disturb the C timings (OMP_NUM_THREADS left alone)

import argparse
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from threadwatch import sdk
from threadwatch.data import load_series
from threadwatch.data.loader import release
from threadwatch.detectors.ml import fit_models

W, S = 256, 1


def cpu_name():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_line():
    import sklearn
    return (f"{cpu_name()}; logical cores {psutil.cpu_count(True)}, physical {psutil.cpu_count(False)}; "
            f"OpenMP cap (sdk.max_threads) {sdk.max_threads()}; Python {sys.version.split()[0]}, "
            f"numpy {np.__version__}, scikit-learn {sklearn.__version__}")


def measure(call, repeats):
    """warm-up, then `repeats` runs: returns (wall_ms list, c_ms list, threads used, last output)"""
    call()
    wall, cms, out, used = [], [], None, None
    for _ in range(repeats):
        t = time.perf_counter()
        out, st = call()
        wall.append((time.perf_counter() - t) * 1000)
        cms.append(st.ms)
        used = st.threads
    return wall, cms, used, out


def md_table(df):
    cols = list(df.columns)
    def cell(v, c):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:.2f}" if c in ("speedup", "speedup_min", "efficiency") else f"{v:,.1f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v, c) for v, c in zip(row, cols)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_5M")
    ap.add_argument("--threads", type=int, nargs="+", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="experiments/out")
    ap.add_argument("--docs", default="docs/omp-notes.md")
    a = ap.parse_args()
    cap = sdk.max_threads()
    threads = sorted({t for t in (a.threads or [1, 2, 4, 8, cap]) if 1 <= t <= cap} | {1})

    xm, _, _ = load_series(a.dataset)
    x = np.array(xm)
    release()
    F = sdk.window_features(x, W, S)[0]
    idx = np.sort(np.random.default_rng(0).choice(len(F), min(100_000, len(F)), replace=False))
    cen, sc, _ = fit_models(F[idx], k=8, seed=0)           # fitted once, outside every timed region
    kernels = {
        "features": lambda t: (lambda: sdk.window_features(x, W, S, threads=t)),
        "zscore": lambda t: (lambda: sdk.zscore(x, W, S, threads=t)),
        "mad": lambda t: (lambda: sdk.mad(x, W, S, threads=t)),
        "kmeans": lambda t: (lambda: sdk.kmeans_score(F, cen, sc, threads=t)),
    }
    rows, base_out = [], {}
    for k, make in kernels.items():
        t1_wall = None
        for t in threads:
            print(f"{k} threads={t} ...", flush=True)
            wall, cms, used, out = measure(make(t), a.repeats)
            if t == 1:
                base_out[k] = out
            same = all(np.array_equal(base_out[k], out2) for out2 in [out])
            med = statistics.median(wall)
            if t == 1:
                t1_wall, t1_min = med, min(wall)
            spread = (max(wall) - min(wall)) / med * 100
            rows.append({"kernel": k, "threads": t, "used": used, "wall_ms": med, "min_ms": min(wall),
                         "spread_pct": spread, "noisy": "YES" if spread > 10 else "",
                         "c_ms": statistics.median(cms), "speedup": t1_wall / med,
                         "speedup_min": t1_min / min(wall), "efficiency": t1_wall / med / used,
                         "identical": bool(same)})
    df = pd.DataFrame(rows)

    sched = []                                              # D-05.6: static vs dynamic,4096 on MAD
    for t in [t for t in threads if t >= 2 and t in (2, 4, 8, cap)]:
        for name in ("static", "dynamic"):
            print(f"mad schedule={name} threads={t} ...", flush=True)
            wall, cms, used, out = measure(lambda: sdk.mad(x, W, S, threads=t, schedule=name), a.repeats)
            med = statistics.median(wall)
            sched.append({"threads": t, "schedule": name, "wall_ms": med, "min_ms": min(wall),
                          "spread_pct": (max(wall) - min(wall)) / med * 100,
                          "identical": bool(np.array_equal(base_out["mad"], out))})
    ds = pd.DataFrame(sched)
    if not ds.empty:
        stat = ds[ds.schedule == "static"].set_index("threads").wall_ms
        ds["dynamic_vs_static"] = [r.wall_ms / stat[r.threads] if r.schedule == "dynamic" else np.nan
                                   for r in ds.itertuples()]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "omp-scaling.csv", index=False)
    ds.to_csv(out / "omp-schedule.csv", index=False)
    doc = ["# OpenMP notes (D-05)", "", f"Machine: {machine_line()}.", "",
           f"Dataset `{a.dataset}`, w={W}, s={S}. Median of {a.repeats} runs after one warm-up; T1 is measured in the same "
           "session. `wall_ms` is the Python timer around the call (end-to-end); `c_ms` is the parallel region measured inside the DLL. "
           "`speedup = T1 / Tp` (wall, medians), `speedup_min` uses the best run of each, `efficiency = speedup / used`. "
           "`used` is the team size the DLL reports (requests above the logical core count are capped). `identical` is "
           "`np.array_equal` of the output against the 1-thread output on every run. Rows with `noisy = YES` (spread above 10%) "
           "were disturbed and should be rerun.", "",
           md_table(df), "",
           "## Schedule comparison on the MAD kernel (D-05.6)", "",
           "`static` = equal contiguous blocks; `dynamic` = chunks of 4096 windows. `dynamic_vs_static` below 1 means dynamic is faster.", "",
           md_table(ds) if not ds.empty else "(needs at least 2 threads)", "",
           "## How to read this on a hybrid CPU", "",
           "Intel 12th gen and newer mix performance cores (P, with hyper-threading) and efficiency cores (E). Logical CPUs 0-7 are the "
           "P-core threads and 8-11 the E-cores on a 4P+4E part, so the speedup curve is expected to bend after 4 threads (SMT siblings share a core) "
           "and again after 8 (E-cores are slower). Compare the curve with the physical (" f"{psutil.cpu_count(False)}) and logical ({psutil.cpu_count(True)}) counts above.", ""]
    Path(a.docs).write_text("\n".join(doc))
    print(df.drop(columns=["c_ms", "speedup_min"]).to_string(index=False))
    print(ds.to_string(index=False))
    if not df.identical.all():
        print("WARNING: some parallel output differs from the 1-thread output")
        sys.exit(1)


if __name__ == "__main__":
    main()
