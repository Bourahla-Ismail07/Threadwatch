"""D-06 acceptance: `mpiexec -n 1` reproduces the C-1 time within 5 %.

    python experiments/mpi_n1_check.py --dataset syn_5M --detector mad

Both sides are timed the same way (median of N runs of the DLL call plus the I/O of the mode);
the C-1 side is the plain sdk call, the other side is one MPI rank. Run on an idle machine,
single-thread runs ideally pinned (see D-05 protocol).
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, "python")
import numpy as np  # noqa: E402

from threadwatch import sdk  # noqa: E402
from threadwatch.data.loader import DATA_ROOT  # noqa: E402


def c1_seconds(path, detector, w, s, model, repeats):
    x = np.ascontiguousarray(np.memmap(path, np.float32, "r"))
    def once():
        t = time.perf_counter()
        if detector == "zscore":
            sdk.zscore(x, w, s, threads=1)
        elif detector == "mad":
            sdk.mad(x, w, s, threads=1)
        else:
            f, _ = sdk.window_features(x, w, s, threads=1)
            sdk.kmeans_score(f, model[0], model[1], threads=1)
        return time.perf_counter() - t
    once()
    return st.median(once() for _ in range(repeats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_5M")
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--detector", default="mad", choices=["zscore", "mad", "kmeans"])
    ap.add_argument("--model")
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    path = f"{a.root}/{a.dataset}.f32"
    model = None
    if a.model:
        z = np.load(a.model)
        model = (z["centroids"], z["scale"])
    c1 = c1_seconds(path, a.detector, a.w, a.s, model, a.repeats)
    for mode in ("file", "scatter"):
        cmd = [a.mpiexec, "-n", "1", sys.executable, "-m", "threadwatch.mpi.run", "--mode", mode, "--dataset", a.dataset,
               "--root", a.root, "--detector", a.detector, "--w", str(a.w), "--s", str(a.s),
               "--repeats", str(a.repeats + 1)] + (["--model", a.model] if a.model else [])
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "python"})
        rows = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("TW_ROW")]
        if not rows:
            print(f"{mode}: run failed: {out.stderr.strip()[-200:]}")
            continue
        m = st.median(r["score_s"] for r in rows)          # compute part, comparable to the C-1 call
        dev = abs(m - c1) / c1
        print(f"{mode:8s} MPI-1 compute {m:.3f} s | C-1 {c1:.3f} s | deviation {dev:.1%} | {'PASS' if dev <= a.tol else 'FAIL'} (tol {a.tol:.0%})")


if __name__ == "__main__":
    main()
