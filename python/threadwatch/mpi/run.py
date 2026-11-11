"""D-06 command line entry point (one launch = one cell for the D-08 harness).

  mpiexec -n 4 python -m threadwatch.mpi.run --mode file --dataset syn_5M --detector mad --threads 1   # MPI-4
  mpiexec -n 2 python -m threadwatch.mpi.run --mode file --dataset syn_5M --detector mad --threads 2   # HYB-2x2

Rank 0 prints one line per repeat:  TW_ROW {json}
Timing is taken inside the program, between barriers; process start-up is not included.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from ..data.loader import DATA_ROOT
from .driver import DETECTORS, MODES, check_threads, file_length, run


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="threadwatch.mpi.run")
    ap.add_argument("--mode", choices=MODES, default="file", help="file: every rank maps the file; scatter: rank 0 sends slices")
    ap.add_argument("--dataset", required=True, help="series name under --root (<root>/<name>.f32)")
    ap.add_argument("--root", default=str(DATA_ROOT), help="folder holding <dataset>.f32 (default: TW_DATA or data/series)")
    ap.add_argument("--detector", choices=DETECTORS, default="zscore")
    ap.add_argument("--threads", type=int, default=1, help="threads per rank (p); 1 = pure MPI")
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--model", help=".npz with `centroids` and `scale` (kmeans)")
    ap.add_argument("--reference", help=".npy with the C-1 scores; adds `identical` to the row")
    ap.add_argument("--save-scores", help="write the gathered scores to this .npy")
    ap.add_argument("--progress", help="int64 memmap file with q*threads*8 counters (dashboard); must exist")
    ap.add_argument("--scores-out", help="float32 memmap file with nw entries (dashboard); must exist")
    ap.add_argument("--repeats", type=int, default=1, help="timed repeats inside one launch (after one untimed warm-up if > 1)")
    ap.add_argument("--allow-oversubscribe", action="store_true", help="allow q x p above the logical cores (flagged in the row)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    r, q = comm.Get_rank(), comm.Get_size()
    try:
        args = parse_args(argv)
        path = f"{args.root}/{args.dataset}.f32"
        warn = check_threads(q, args.threads, allow_oversubscribe=args.allow_oversubscribe)
        if warn and r == 0:
            print(f"warning: {warn}", file=sys.stderr)
        model = None
        if args.model:
            z = np.load(args.model)
            model = (np.ascontiguousarray(z["centroids"], np.float32), np.ascontiguousarray(z["scale"], np.float32))
        ref = np.load(args.reference) if (args.reference and r == 0) else None

        n = comm.bcast(file_length(path) if r == 0 else None, root=0)
        progress = np.memmap(args.progress, np.int64, "r+") if args.progress else None
        scores_out = np.memmap(args.scores_out, np.float32, "r+") if args.scores_out else None
        label = f"MPI-{q}" if args.threads == 1 else f"HYB-{q}x{args.threads}"
        if args.repeats > 1:                                  # warm-up: page cache, DLL, OpenMP pools
            run(args.mode, path, args.w, args.s, args.threads, args.detector, comm=comm, model=model)

        for rep in range(args.repeats):
            tm, score = run(args.mode, path, args.w, args.s, args.threads, args.detector, comm=comm, model=model,
                            progress=progress, out=scores_out)
            if r != 0:
                continue
            row = {
                "dataset": args.dataset, "n": n, "w": args.w, "s": args.s, "detector": args.detector,
                "mode": "MPI" if args.threads == 1 else "HYB", "label": label, "input_mode": args.mode,
                "q_ranks": q, "p_threads": args.threads, "repeat": rep,
                "load_s": tm["in_s"], "score_s": tm["compute_s"], "gather_s": tm["gather_s"], "comm_s": tm["comm_s"],
                "total_s": tm["total_s"], "dll_ms": tm["dll_ms"], "throughput_sps": n / tm["total_s"],
                "compute_by_rank_s": tm["compute_by_rank"], "status": "ok",
                "note": warn or "",
            }
            if ref is not None:
                row["identical"] = bool(np.array_equal(score, ref))
            print("TW_ROW " + json.dumps(row), flush=True)
            if args.save_scores and rep == args.repeats - 1:
                np.save(args.save_scores, score)
        return 0
    except SystemExit:                                        # argparse --help / usage errors: every rank exits the same way
        raise
    except BaseException:                                     # one rank failing must not leave the others blocked in a collective
        traceback.print_exc()
        sys.stderr.flush()
        comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
