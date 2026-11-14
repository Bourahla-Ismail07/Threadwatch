"""D-08.5 - summary tables, generated from results.csv only (never by hand).

    python experiments/summarize.py                 # writes experiments/out/summary_*.csv and docs/perf.md

Derived numbers (median, spread, speedup, efficiency, throughput) live here, never in the raw file.
Only rows with status == 'ok' count; warm-ups, timeouts, errors, crashes and superseded runs stay in the CSV and are
listed in the coverage table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from threadwatch.bench.matrix import blocks_of, build_cells, load_config  # noqa: E402

GROUP = ["dataset", "n", "w", "s", "detector", "mode", "q_ranks", "p_threads"]
MODE_ORDER = ["PY-SEQ", "C-1", "OMP", "MPI", "MPI-scatter", "HYB"]
DRIFT_FLAG_PCT = 5.0


def label_of(mode: str, q: int, p: int) -> str:
    q, p = int(q), int(p)
    return {"PY-SEQ": "PY-SEQ", "C-1": "C-1", "OMP": f"OMP-{p}", "MPI": f"MPI-{q}", "MPI-scatter": f"MPI-{q}-scatter",
            "HYB": f"HYB-{q}x{p}"}.get(mode, mode)


def load(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"identical": str, "note": str, "status": str})
    df["identical"] = df["identical"].map({"True": True, "False": False})
    df["label"] = [label_of(m, q, p) for m, q, p in zip(df["mode"], df["q_ranks"], df["p_threads"])]
    return df


# ---------------------------------------------------------------------------- statistics
def cell_stats(df: pd.DataFrame, cv_flag_pct: float = 10.0) -> pd.DataFrame:
    """One row per cell (timed ok runs only): median, min, max, IQR, CV, equivalence, quality, speedup, efficiency."""
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for key, g in ok.groupby(GROUP + ["label"], sort=False):
        d = dict(zip(GROUP + ["label"], key))
        t = g["total_s"].to_numpy(float)
        q1, med, q3 = np.percentile(t, [25, 50, 75])
        d.update(runs=len(t), median_s=med, min_s=t.min(), max_s=t.max(), iqr_s=q3 - q1,
                 cv_pct=(t.std(ddof=1) / t.mean() * 100.0) if len(t) > 1 else np.nan,
                 identical_all=(bool(g["identical"].all()) if g["identical"].notna().all() else None),
                 identical_known=int(g["identical"].notna().sum()),
                 f1_median=g["f1"].median(), event_recall_median=g["event_recall"].median(),
                 dll_ms_median=g["dll_ms"].median(), comm_s_median=g["comm_s"].median(),
                 throughput_sps=d["n"] / med)
        rows.append(d)
    st = pd.DataFrame(rows)
    st["flag_rerun"] = st["cv_pct"] > cv_flag_pct
    t1 = st[st["mode"] == "C-1"].set_index(["dataset", "detector", "w", "s"])["median_s"]
    idx = list(zip(st["dataset"], st["detector"], st["w"], st["s"]))
    st["t1_s"] = [t1.get(i, np.nan) for i in idx]
    par = st["mode"] != "PY-SEQ"
    st["speedup"] = np.where(par, st["t1_s"] / st["median_s"], np.nan)            # T1 / Tp, medians
    st["efficiency"] = st["speedup"] / (st["q_ranks"] * st["p_threads"])
    st["order"] = st["mode"].map({m: i for i, m in enumerate(MODE_ORDER)})
    return st.sort_values(["detector", "n", "s", "order", "q_ranks", "p_threads"]).drop(columns="order").reset_index(drop=True)


def noisy_cells(df: pd.DataFrame, cv_flag_pct: float = 10.0) -> pd.DataFrame:
    st = cell_stats(df, cv_flag_pct)
    return st[st["flag_rerun"]] if not st.empty else st


def drift_table(df: pd.DataFrame) -> pd.DataFrame:
    """Thermal / background drift: trend of the repeated C-1 runs of each block, as % of their median over the block."""
    ok = df[(df["status"] == "ok") & (df["mode"] == "C-1")]
    rows = []
    for (ds, det, w, s), g in ok.groupby(["dataset", "detector", "w", "s"]):
        g = g.sort_values("repeat")
        if len(g) < 3:
            continue
        slope = np.polyfit(g["repeat"].to_numpy(float), g["total_s"].to_numpy(float), 1)[0]
        pct = slope * (len(g) - 1) / g["total_s"].median() * 100.0
        rows.append({"dataset": ds, "detector": det, "w": w, "s": s, "runs": len(g), "drift_pct": pct,
                     "flag": pct > DRIFT_FLAG_PCT})
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame, cells, repeats: int) -> pd.DataFrame:
    """Every expected cell with its number of timed runs; if short, the reason."""
    rows = []
    for c in cells:
        g = df[df["run_id"].str.startswith(c.key + "#")] if not df.empty else df
        ok = int((g["status"] == "ok").sum()) if not df.empty else 0
        fail = g[g["status"].isin(["timeout", "error", "crash"])] if not df.empty else g
        sup = int((g["status"] == "superseded").sum()) if not df.empty else 0
        if ok >= repeats:
            reason = ""
        elif len(fail):
            f = fail.iloc[-1]
            reason = f"{f['status']} at repeat {int(f['repeat'])}: {str(f['note'])[:120]}"
        elif ok or (not df.empty and len(g)):
            reason = f"incomplete: {ok} of {repeats} timed runs (campaign interrupted?)"
        else:
            reason = "not run yet"
        rows.append({"dataset": c.dataset, "detector": c.detector, "label": c.label, "w": c.w, "s": c.s,
                     "ok_runs": ok, "required": repeats, "superseded_runs": sup, "complete": ok >= repeats, "reason": reason})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- formatting
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df.empty:
        return "_no data_\n"
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if v != v else floatfmt.format(v)
        return "" if v is None else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def pivot(st: pd.DataFrame, detector: str, value: str) -> pd.DataFrame:
    sub = st[st["detector"] == detector]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.assign(size=[f"{d}" + ("" if s == 1 else f" (s={s})") for d, s in zip(sub["dataset"], sub["s"])])
    cols = list(dict.fromkeys(sub["label"]))
    p = sub.pivot_table(index=["n", "s", "size"], columns="label", values=value, aggfunc="first").reindex(columns=cols)
    return p.reset_index().drop(columns=["n", "s"]).rename(columns={"size": "dataset"})


# ---------------------------------------------------------------------------- writing
def write_all(root: Path, matrix_path: Path, out_dir: Path, doc_path: Path) -> dict:
    cfg = load_config(matrix_path)
    pr = cfg["protocol"]
    df = load(out_dir / "results.csv")
    machine = json.loads((out_dir / "machine.json").read_text()) if (out_dir / "machine.json").exists() else None
    logical = machine["logical_cores"] if machine else (__import__("psutil").cpu_count(logical=True) or 1)
    ns = dict(zip(df["dataset"], df["n"])) if not df.empty else {}          # sizes seen in the results (dataset names need not encode them)
    def n_of(name):
        if name in ns:
            return int(ns[name])
        raise FileNotFoundError(name)
    cells = build_cells(cfg, logical, n_of)
    st = cell_stats(df, pr["cv_flag_pct"]) if not df.empty else pd.DataFrame()
    cov = coverage(df, cells, int(pr["repeats"]))
    dr = drift_table(df) if not df.empty else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for det in cfg.get("detectors", []):
        p = pivot(st, det, "median_s") if not st.empty else pd.DataFrame()
        if not p.empty:
            p.to_csv(out_dir / f"summary_{det}.csv", index=False)
            written.append(f"summary_{det}.csv")
    if not st.empty:
        st.to_csv(out_dir / "summary_cells.csv", index=False)
        written.append("summary_cells.csv")
    cov.to_csv(out_dir / "coverage.csv", index=False)
    written.append("coverage.csv")
    if not dr.empty:
        dr.to_csv(out_dir / "drift.csv", index=False)
        written.append("drift.csv")

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(render_doc(cfg, machine, df, st, cov, dr))
    return {"written": written, "cells_expected": len(cells), "cells_complete": int(cov["complete"].sum()),
            "rows": len(df), "flagged": 0 if st.empty else int(st["flag_rerun"].sum())}


def render_doc(cfg, machine, df, st, cov, dr) -> str:
    pr = cfg["protocol"]
    o = ["# Benchmark results (D-08)\n",
         "Generated by `experiments/summarize.py` from `experiments/out/results.csv`. Do not edit by hand: rerun the script.\n"]
    if machine:
        m = machine
        o.append(f"\n## Machine\n\n{m['processor']}, {m['logical_cores']} logical / {m['physical_cores']} physical cores, "
                 f"{m['ram_gb']} GB RAM, {m['os']}. Python {m['python']}, numpy {m['packages']['numpy']}, "
                 f"scikit-learn {m['packages']['sklearn']}. Power plan: {m['power_plan'] or 'n/a'}. "
                 f"MPI: {m['mpi'].get('library', 'n/a')}. DLL: {m['dll'].get('version') or 'n/a'}"
                 f"{' (' + m['dll']['build_note'] + ')' if m['dll'].get('build_note') else ''}. Captured {m['captured_at']}.\n")
    else:
        o.append("\n## Machine\n\n`machine.json` not found: run `python experiments/machine_info.py --write`.\n")
    o.append(f"\n## Protocol\n\nWindow w={cfg['window']['w']}, stride s={cfg['window']['s']} (second stride on one size where listed). "
             f"Per cell: {pr['warmup']} warm-up run (kept in the CSV, excluded here) and {pr['repeats']} timed runs, modes interleaved "
             f"round by round within each (dataset, detector) block, {pr['pause_between_blocks_s']} s pause between blocks. "
             "`total_s` is end to end (mapping the input included). Medians of the timed runs; speedup = median T1 / median Tp "
             "with T1 the C-1 median of the same dataset, detector, window and stride; efficiency = speedup / (ranks x threads). "
             f"Threshold = 1 - {pr['rho']} quantile of the scores of a fixed {pr['fit_sample']}-window sample, identical in every mode. "
             "`MPI-q-scatter` rows are the scatter input mode; all other MPI rows read the file on every rank. "
             "Isolation Forest: features in C, scoring in scikit-learn (`n_jobs` = the thread parameter), no MPI.\n")
    if df.empty:
        o.append("\n## Results\n\nNo campaign data yet. Run `python experiments/run_matrix.py` (see `--dry-run` for the cost), "
                 "then `python experiments/summarize.py`.\n")
        return "".join(o)

    total = len(cov)
    done = int(cov["complete"].sum())
    o.append(f"\n## Coverage\n\n{done} of {total} cells have at least {pr['repeats']} timed runs. ")
    miss = cov[~cov["complete"]]
    if miss.empty:
        o.append("No cell is missing.\n")
    else:
        o.append("Missing cells and the reason:\n\n" + md_table(miss[["dataset", "detector", "label", "s", "ok_runs", "reason"]]))
    sup = int((df["status"] == "superseded").sum())
    counts = df["status"].value_counts().to_dict()
    o.append(f"\nRows by status: {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}. Failed and superseded runs are never deleted"
             + (f" ({sup} superseded by reruns)." if sup else ".") + "\n")

    for det in cfg.get("detectors", []):
        p = pivot(st, det, "median_s")
        if p.empty:
            continue
        o.append(f"\n## {det}: median seconds (size x mode)\n\n" + md_table(p, "{:.3f}"))
        sp = pivot(st, det, "speedup").drop(columns=["PY-SEQ"], errors="ignore")
        if len(sp.columns) > 1:
            o.append(f"\n{det}: speedup versus C-1 (T1 / Tp)\n\n" + md_table(sp, "{:.2f}"))

    o.append(f"\n## Spread: cells above {pr['cv_flag_pct']}% coefficient of variation\n\n")
    noisy = st[st["flag_rerun"]] if not st.empty else st
    if noisy.empty:
        o.append("None.\n")
    else:
        o.append("These cells should be rerun (`python experiments/run_matrix.py --rerun-noisy`):\n\n"
                 + md_table(noisy[["dataset", "detector", "label", "s", "runs", "median_s", "cv_pct", "min_s", "max_s"]]))
    o.append("\n## Equivalence with C-1\n\n")
    det_kernels = st[st["detector"].isin(["zscore", "mad", "kmeans"]) & st["mode"].isin(["OMP", "MPI", "MPI-scatter", "HYB"])] if not st.empty else st
    bad = det_kernels[det_kernels["identical_all"] == False] if not det_kernels.empty else det_kernels   # noqa: E712
    unk = det_kernels[det_kernels["identical_all"].isna()] if not det_kernels.empty else det_kernels
    o.append(f"{len(det_kernels) - len(bad) - len(unk)} of {len(det_kernels)} OMP/MPI/HYB cells of the deterministic kernels are identical to C-1 in every timed run.\n")
    if len(bad):
        o.append("\n**Not identical (must be investigated, not hidden):**\n\n" + md_table(bad[["dataset", "detector", "label", "identical_known", "f1_median"]]))
    if len(unk):
        o.append("\nIdentity could not be checked for some runs (no C-1 reference):\n\n" + md_table(unk[["dataset", "detector", "label"]]))
    o.append("\n## Drift of the repeated C-1 runs\n\n")
    if dr.empty:
        o.append("Not enough C-1 runs yet.\n")
    else:
        o.append(f"Trend of the timed C-1 runs of each block, as % of their median (flag above {DRIFT_FLAG_PCT:.0f}%: possible thermal throttling or background activity; rerun the block).\n\n"
                 + md_table(dr, "{:.1f}"))
    return "".join(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--matrix", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    info = write_all(root, Path(a.matrix or root / "experiments/matrix.yaml"), root / "experiments/out", root / "docs/perf.md")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
