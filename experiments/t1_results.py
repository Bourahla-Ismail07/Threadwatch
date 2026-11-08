"""D-04.6: single-thread C times (T1), language gain G_lang = T_py / T1, equivalence check.

T_py comes from experiments/out/first-results.csv (D-03 run on this machine); rows missing there are
measured now with the Python detectors. Writes experiments/out/t1-results.csv and docs/t1-results.md.
Run:  python experiments/t1_results.py
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from threadwatch import sdk
from threadwatch.data import load_series
from threadwatch.data.loader import release
from threadwatch.detectors.features import features_py
from threadwatch.detectors.mad import mad_py
from threadwatch.detectors.ml import fit_models, kmeans_score_py
from threadwatch.detectors.zscore import zscore_py

W, S, REPEATS = 256, 1, 5
KERNELS = ("features", "zscore", "mad", "kmeans")


def timed(fn, repeats=None):
    """warm-up once, then median of `repeats` wall-clock runs (ms) and the last result"""
    fn()
    runs, res = [], None
    for _ in range(repeats or REPEATS):
        t = time.perf_counter()
        res = fn()
        runs.append((time.perf_counter() - t) * 1000)
    return res, statistics.median(runs), runs


def machine_line():
    import psutil, sklearn
    return (f"{platform.processor() or platform.machine()}, logical cores {psutil.cpu_count(True)}, "
            f"physical {psutil.cpu_count(False)}, Python {sys.version.split()[0]}, numpy {np.__version__}, "
            f"scikit-learn {sklearn.__version__}")


def md_table(df):
    cols = list(df.columns)
    def cell(v, c):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:,.1f}" if c.endswith("_ms") or c in ("G_lang", "spread_pct") else f"{v:.2e}"
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v, c) for v, c in zip(row, cols)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["syn_1M", "syn_2M", "syn_5M"])
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--first-results", default="experiments/out/first-results.csv")
    ap.add_argument("--out", default="experiments/out")
    ap.add_argument("--docs", default="docs/t1-results.md")
    a = ap.parse_args()
    prev = pd.read_csv(a.first_results) if Path(a.first_results).exists() else None

    rows, equiv = [], {}
    for name in a.datasets:
        x, _, _ = load_series(name)
        x = np.array(x)
        release()
        # models are fitted once, outside every timed region (D-03 rule), on 100,000 sampled windows
        F_c = sdk.window_features(x, W, S)[0]
        idx = np.sort(np.random.default_rng(0).choice(len(F_c), min(100_000, len(F_c)), replace=False))
        cen, sc, _ = fit_models(F_c[idx], k=8, seed=0)
        c_fns = {
            "features": lambda: sdk.window_features(x, W, S),
            "zscore": lambda: sdk.zscore(x, W, S),
            "mad": lambda: sdk.mad(x, W, S),
            "kmeans": lambda: sdk.kmeans_score(sdk.window_features(x, W, S)[0], cen, sc),
        }
        py_fns = {
            "zscore": lambda: zscore_py(x, W, S),
            "mad": lambda: mad_py(x, W, S),
            "kmeans": lambda: kmeans_score_py(features_py(x, W, S), cen, sc),
        }
        for k in KERNELS:
            res, t1, runs = timed(c_fns[k], a.repeats)
            t_py = None
            if k in py_fns and name in ("syn_1M", "syn_2M"):
                if prev is not None and ((prev.dataset == name) & (prev.detector == k)).any():
                    t_py = float(prev[(prev.dataset == name) & (prev.detector == k)].elapsed_ms.iloc[0])
                else:
                    t_py = timed(py_fns[k])[1]
            if k in py_fns and name == "syn_1M":                      # full-series equivalence on syn_1M
                ref, got = py_fns[k](), (res if k == "kmeans" else res[0])
                got = got if isinstance(got, np.ndarray) else got[0]
                equiv[k] = float(np.max(np.abs(got - ref) / (np.abs(ref) + 1e-6)))
            spread = (max(runs) - min(runs)) / t1 * 100
            rows.append({"dataset": name, "kernel": k, "T_py_ms": t_py, "T1_ms": t1,
                         "G_lang": (t_py / t1) if t_py else None, "min_ms": min(runs),
                         "spread_pct": spread, "noisy": "YES" if spread > 10 else "",
                         "runs_ms": str([round(r, 1) for r in runs])})
    df = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "t1-results.csv", index=False)
    doc = ["# T1 results (D-04, C SDK with 1 thread)", "", f"Machine: {machine_line()}.", "",
           f"Window w={W}, stride s={S}. `T1_ms` is the median of {a.repeats} wall-clock runs after one warm-up, measured around the "
           "Python call (includes the output allocation); the individual runs are in `runs_ms`. The `kmeans` kernel "
           "is the full pipeline (C features + C K-Means score), the same work as `T_py` in "
           "[`first-results.md`](first-results.md). `T_py_ms` is read from the D-03 run on this machine; `G_lang = T_py / T1`. "
           "Compiled with `-O3`, no `-ffast-math`; one thread. `spread_pct` = (max - min) / median of the runs; rows with "
           "`noisy = YES` (spread above 10%) were disturbed by something else on the machine and must be rerun before they are used as T1.", "",
           md_table(df.drop(columns="runs_ms")), "",
           "Equivalence with the NumPy reference on the full `syn_1M` (max relative error, tolerance 1e-5; 0 means identical float32 outputs): " +
           ", ".join(f"{k} {v:.1e}" for k, v in equiv.items()) + ".", ""]
    Path(a.docs).write_text("\n".join(doc))
    print(df.drop(columns="runs_ms").to_string(index=False))
    print("equivalence:", equiv)


if __name__ == "__main__":
    main()
