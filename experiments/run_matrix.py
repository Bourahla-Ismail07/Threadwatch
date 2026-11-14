"""D-08 - run the benchmark campaign from one command.

    python experiments/run_matrix.py --dry-run          # list the cells, the cost bound, missing data files
    python experiments/run_matrix.py                    # run (resumes after any interruption)
    python experiments/run_matrix.py --rerun-noisy      # rerun cells whose CV is above the limit (old runs -> superseded)
    python experiments/summarize.py                     # tables and docs/perf.md

Close other applications, plug in the laptop, use the High performance power plan, keep the dashboard off.
The harness runs under a small supervisor: a run that times out or crashes becomes a row and the campaign continues.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments"))

from threadwatch.bench.matrix import blocks_of, build_cells, estimate, filter_cells, load_config  # noqa: E402
from threadwatch.bench.rows import ResultsFile  # noqa: E402
from threadwatch.bench.runner import Context, Runner, supervise  # noqa: E402


def csv_list(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(ROOT / "experiments/matrix.yaml"))
    ap.add_argument("--data-root", default=str(ROOT / "data/series"))
    ap.add_argument("--out", default=str(ROOT / "experiments/out"))
    ap.add_argument("--datasets", help="comma list, e.g. syn_1M,syn_2M")
    ap.add_argument("--detectors", help="comma list, e.g. zscore,mad")
    ap.add_argument("--modes", help="comma list of modes or labels, e.g. C-1,OMP-4,MPI-2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="accepted for tw bench; a run always resumes from results.csv")
    ap.add_argument("--rerun-noisy", action="store_true")
    ap.add_argument("--strict", action="store_true", help="abort if a pre-flight check fails")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--force-machine", action="store_true", help="accept a changed machine configuration on resume")
    ap.add_argument("--no-supervise", action="store_true", help="run in this process (no restart after timeout or crash)")
    ap.add_argument("--logical-cores", type=int, help="override (tests only)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def logical_cores(a) -> int:
    import psutil
    return a.logical_cores or psutil.cpu_count(logical=True) or 1


def selected_cells(a, cfg):
    ctx = Context(cfg, a.data_root, a.out)
    cells = build_cells(cfg, logical_cores(a), ctx.n_of)
    return filter_cells(cells, csv_list(a.datasets), csv_list(a.detectors), csv_list(a.modes)), ctx


def dry_run(a, cfg, cells) -> int:
    est = estimate(cfg, cells)
    print(f"{est['cells']} cells in {est['blocks']} blocks, {est['runs']} runs "
          f"({cfg['protocol']['warmup']} warm-up + {cfg['protocol']['repeats']} timed per cell)")
    print(f"upper bound {est['upper_bound_s'] / 3600:.1f} h (every C mode bounded by its C-1 time; the real time is lower)")
    from collections import Counter
    print("cells by mode:", dict(Counter(c.mode for c in cells)))
    missing = sorted({c.dataset for c in cells if not (Path(a.data_root) / f"{c.dataset}.f32").exists()})
    nolabels = sorted({c.dataset for c in cells if (Path(a.data_root) / f"{c.dataset}.f32").exists()
                       and not (Path(a.data_root) / f"{c.dataset}.labels.u8").exists()})
    if missing:
        print("MISSING data files:", ", ".join(missing), "(generate with the D-02 data script)")
    if nolabels:
        print("no labels (quality columns will be empty):", ", ".join(nolabels))
    done = len(ResultsFile(Path(a.out) / "results.csv").rows())
    print(f"results.csv already holds {done} rows (a run resumes from them)")
    for b in blocks_of(cells)[:3]:
        print("  block", b.key, "->", ", ".join(c.label for c in b.cells))
    return 0


def prepare_campaign(a, cfg) -> bool:
    """machine.json and pre-flight, once per campaign start (not at every restart)."""
    import machine_info
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = machine_info.collect(cfg.get("dll_build_note", ""))
    mp = out / "machine.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if "fingerprint" not in old:                                   # record from another tool or an older format
            stamp = time.strftime("%Y%m%dT%H%M%S")
            mp.rename(out / f"machine.legacy.{stamp}.json")
            print(f"machine.json had no fingerprint (older format): archived as machine.legacy.{stamp}.json, new one written")
            mp.write_text(json.dumps(info, indent=2))
        elif old.get("fingerprint") != info["fingerprint"]:
            diff = {k: (old["fingerprint"].get(k), v) for k, v in info["fingerprint"].items() if old["fingerprint"].get(k) != v}
            if not a.force_machine:
                print("the machine configuration differs from machine.json (campaign started on another setup?):", diff)
                print("use --force-machine to keep going and archive the old record")
                return False
            mp.rename(out / f"machine.{old['captured_at'].replace(':', '')}.json")
            mp.write_text(json.dumps(info, indent=2))
    else:
        mp.write_text(json.dumps(info, indent=2))
    if not a.skip_preflight:
        warns = machine_info.preflight(float(cfg["protocol"].get("max_background_cpu_pct", 15)))
        for w in warns:
            print("PRE-FLIGHT:", w)
        if warns and a.strict:
            print("aborting (--strict)")
            return False
    return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse(argv)
    cfg = load_config(a.matrix)
    cells, ctx = selected_cells(a, cfg)
    if not cells:
        print("no cell selected")
        return 1
    if a.dry_run:
        return dry_run(a, cfg, cells)
    results = ResultsFile(Path(a.out) / "results.csv")

    if not a.child:
        if not prepare_campaign(a, cfg):
            return 2
        if not a.no_supervise:
            return supervise([sys.executable, str(Path(__file__).resolve()), *argv, "--child"], a.out, results)

    runner = Runner(cfg, ctx, results, a.out, logical_cores(a))
    if a.rerun_noisy:
        import pandas as pd
        import summarize
        df = summarize.load(results.path)
        noisy = summarize.noisy_cells(df, float(cfg["protocol"]["cv_flag_pct"]))
        keys = {f"{r.dataset}|{r.detector}|{r.label}|w{r.w}|s{r.s}": f"cv {r.cv_pct:.1f}%" for r in noisy.itertuples()}
        todo = [c for c in cells if c.key in keys]
        print(f"{len(todo)} noisy cells to rerun")
        runner.rerun(todo, keys)
    else:
        runner.run(cells)
    print("done. Next: python experiments/summarize.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
