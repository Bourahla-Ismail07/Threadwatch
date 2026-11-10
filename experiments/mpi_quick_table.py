"""Fill the communication-versus-computation table of docs/mpi-notes.md from real runs.

    python experiments/mpi_quick_table.py --dataset syn_5M --detector mad > mpi_table.md

One launch per cell: 1 untimed warm-up + 5 timed repeats inside the program; medians are printed.
Run on an idle machine (see D-08 protocol). Cells with q x p above the logical cores are skipped.
"""
import argparse
import json
import statistics as st
import subprocess
import sys

sys.path.insert(0, "python")
from threadwatch.mpi.driver import logical_cores  # noqa: E402

CELLS = [(1, 1), (2, 1), (4, 1), (2, 2), (2, 4)]          # (q ranks, p threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="syn_5M")
    ap.add_argument("--detector", default="mad")
    ap.add_argument("--model", help="kmeans model .npz (python -m threadwatch.mpi.make_model)")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--repeats", type=int, default=6)
    a = ap.parse_args()
    logical = logical_cores()
    print(f"Machine: {logical} logical cores. Dataset {a.dataset}, detector {a.detector}, w=256, s=1. "
          f"Median of {a.repeats - 1} timed runs (seconds, slowest rank).\n")
    print("| run | input | in_s | compute_s | gather_s | comm_s | total_s | comm share |")
    print("|---|---|---|---|---|---|---|---|")
    for mode in ("file", "scatter"):
        for q, p in CELLS:
            if q * p > logical:
                continue
            cmd = [a.mpiexec, "-n", str(q), sys.executable, "-m", "threadwatch.mpi.run", "--mode", mode,
                   "--dataset", a.dataset, "--detector", a.detector, "--threads", str(p), "--repeats", str(a.repeats)]
            if a.model:
                cmd += ["--model", a.model]
            out = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "python"})
            rows = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("TW_ROW")]
            if not rows:
                print(f"| {'MPI-%d' % q if p == 1 else 'HYB-%dx%d' % (q, p)} | {mode} | failed: {out.stderr.strip()[-80:]} |")
                continue
            m = lambda k: st.median(r[k] for r in rows)
            share = m("comm_s") / m("total_s")
            print(f"| {rows[0]['label']} | {mode} | {m('load_s'):.4f} | {m('score_s'):.3f} | {m('gather_s'):.4f} | "
                  f"{m('comm_s'):.4f} | {m('total_s'):.3f} | {share:.1%} |")


if __name__ == "__main__":
    main()
