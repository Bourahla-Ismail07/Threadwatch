"""D-09 - everything from results.csv with one command: indicators, tables, ten figures, docs/results.md.

    python experiments/run_analysis.py [--theme light|dark]

No manual step: rerun it after any change of results.csv. Needs matplotlib and pandas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analysis import indicators, interpret, make_figures, tables  # noqa: E402


def run(results: Path, machine_path: Path, out_dir: Path, docs_path: Path, theme: str = "light", min_large_n: int = interpret.LARGE_N) -> dict:
    df = indicators.load_results(results)
    machine = indicators.load_machine(machine_path)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        status = make_figures.make_all(df, machine, 1, fig_dir, theme, str(out_dir))
        docs_path.write_text("# Results and interpretation (D-09)\n\nNo results yet: `experiments/out/results.csv` is missing or has no `ok` rows. "
                             "Run the D-08 campaign (`python experiments/run_matrix.py`), then `python experiments/run_analysis.py`.\n")
        return {"rows": 0, "figures": status, "verdicts": {}}
    ct = indicators.cell_table(df)
    s0 = indicators.base_stride(df)
    tabs = tables.build_tables(ct, s0)
    written = tables.write_tables(tabs, tab_dir)
    status = make_figures.make_all(ct, machine, s0, fig_dir, theme, str(out_dir))
    v = interpret.verdicts(ct, machine, s0, min_large_n)
    base = ct[ct["s"] == s0]
    protocol = {"w": int(ct["w"].iloc[0]), "runs_min": int(base["runs"].min()), "runs_max": int(base["runs"].max())}
    docs_path.write_text(interpret.render_results(ct, tabs, machine, s0, v, {i: ("ok" if s == "ok" else s) for i, s in status.items()}, protocol))
    return {"rows": len(df), "cells": len(ct), "tables": written, "figures": status, "verdicts": {h: d["result"] for h, d in v.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--min-large-n", type=int, default=interpret.LARGE_N, help="sizes from this on count as large for H1 and H4")
    a = ap.parse_args(argv)
    root = Path(a.root)
    info = run(root / "experiments/out/results.csv", root / "experiments/out/machine.json", root / "experiments/out", root / "docs/results.md",
               a.theme, a.min_large_n)
    print(json.dumps({k: v for k, v in info.items() if k != "tables"}, indent=1))
    missing = [i for i, s in info["figures"].items() if s != "ok"]
    if missing:
        print("figures without data:", ", ".join(f"F{i:02d}" for i in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
