"""D-09.2/3 - CSV tables under experiments/out/tables/ and the markdown table helper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .indicators import DETERMINISTIC, amdahl_table, compare

PARALLEL = ["OMP", "MPI", "HYB"]


def fmt_n(n: float) -> str:
    n = int(n)
    return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else (f"{n // 1000}k" if n % 1000 == 0 else str(n))


def fmt_approx(n: float) -> str:
    """Readable approximation for an interpolated size: 1.03M, 640k."""
    n = float(n)
    if n != n:
        return "n/a"
    return f"{n / 1e6:.3g}M" if n >= 1e6 else (f"{n / 1e3:.3g}k" if n >= 1e3 else f"{n:.0f}")


def quality_pivot(q: pd.DataFrame) -> pd.DataFrame:
    """Compact quality table: one row per kernel (largest size), one column per mode, 'F1' or 'F1 !=C1' if not identical."""
    if q is None or q.empty:
        return pd.DataFrame()
    big = q[q["n"] == q.groupby("detector")["n"].transform("max")]
    cols = list(dict.fromkeys(big["label"]))
    rows = []
    for det, g in big.groupby("detector", sort=False):
        d = {"detector": det, "n": int(g["n"].iloc[0])}
        for _, r in g.iterrows():
            d[r["label"]] = f"{r['f1']:.3f}" + ("" if r["identical_all"] in ("yes", "") else " (differs)") if r["f1"] == r["f1"] else ""
        rows.append(d)
    return pd.DataFrame(rows, columns=["detector", "n"] + cols)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_no data_\n"

    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if v != v else floatfmt.format(v)
        return "" if v is None else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def vs_c1(row) -> str:
    return compare({"q1": row["q1"], "q3": row["q3"]}, {"q1": row["t1_q1"], "q3": row["t1_q3"]})


def wide(ct: pd.DataFrame, family: str, value: str, s0: int) -> pd.DataFrame:
    sub = ct[(ct["family"] == family) & (ct["s"] == s0)]
    if sub.empty:
        return pd.DataFrame()
    p = sub.pivot_table(index=["detector", "n"], columns="label", values=value, aggfunc="first")
    cols = sorted(p.columns, key=lambda c: int(sub[sub["label"] == c]["workers"].iloc[0]))
    return p[cols].reset_index()


def not_faster(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """Parallel cells that are not clearly faster than C-1: speedup <= 1, or the IQRs overlap."""
    sub = ct[ct["family"].isin(PARALLEL) & (ct["s"] == s0)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["vs_C1"] = [vs_c1(r) for _, r in sub.iterrows()]
    out = sub[(sub["speedup"] <= 1.0) | (sub["vs_C1"] != "faster")]
    return out[["detector", "n", "label", "workers", "t1", "med", "q1", "q3", "speedup", "vs_C1", "comm_share"]].reset_index(drop=True)


def crossover(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """Per (detector, parallel mode): how it compares with C-1 at each size, and where it flips from slower to faster."""
    rows = []
    sub = ct[ct["family"].isin(PARALLEL) & (ct["s"] == s0)]
    for (det, lab), g in sub.groupby(["detector", "label"], sort=False):
        g = g.sort_values("n")
        rel = [vs_c1(r) for _, r in g.iterrows()]
        ns, sp = g["n"].tolist(), g["speedup"].tolist()
        lo = hi = None
        for i in range(len(ns)):
            if rel[i] == "slower":
                later = [j for j in range(i + 1, len(ns)) if rel[j] == "faster"]
                if later:
                    lo, hi = i, later[0]
        est = np.nan
        if lo is not None and sp[hi] != sp[lo]:
            t = (1.0 - sp[lo]) / (sp[hi] - sp[lo])
            est = float(np.exp(np.log(ns[lo]) + t * (np.log(ns[hi]) - np.log(ns[lo]))))
        if lo is not None:
            status = "crossover located"
        elif "slower" in rel:
            status = "slower at measured sizes, crossover above them"
        elif all(r == "faster" for r in rel):
            status = "faster at every measured size (crossover, if any, below the smallest size)"
        else:
            status = "no measurable difference at some sizes"
        rows.append({"detector": det, "label": lab, "sizes": ", ".join(fmt_n(n) for n in ns), "vs_C1": ", ".join(rel),
                     "slower_below": fmt_n(ns[lo]) if lo is not None else "", "faster_from": fmt_n(ns[hi]) if lo is not None else "",
                     "crossover_estimate_n": est, "status": status})
    return pd.DataFrame(rows)


def equal_workers(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """OMP, MPI (file and scatter) and HYB at the same number of workers, compared with the OMP cell of that size."""
    sub = ct[ct["family"].isin(PARALLEL) & (ct["s"] == s0)]
    rows = []
    for (det, n, w), g in sub.groupby(["detector", "n", "workers"]):
        omp = g[g["family"] == "OMP"]
        if omp.empty:
            continue
        o = omp.iloc[0]
        for _, r in g.iterrows():
            rel = "" if r["family"] == "OMP" else compare({"q1": r["q1"], "q3": r["q3"]}, {"q1": o["q1"], "q3": o["q3"]})
            rows.append({"detector": det, "n": n, "workers": w, "label": r["label"], "med_s": r["med"], "q1_s": r["q1"],
                         "q3_s": r["q3"], "vs_OMP_same_workers": rel, "comm_share": r["comm_share"]})
    return pd.DataFrame(rows)


def quality_vs_mode(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """D-09.3: F1 and the identical-to-C-1 flag for every mode."""
    sub = ct[ct["s"] == s0]
    out = sub[["detector", "n", "label", "family", "runs", "f1", "precision", "recall", "event_recall", "identical_all", "identical_known"]].copy()
    out["identical_all"] = out["identical_all"].map(lambda v: "" if v is None or v != v else ("yes" if v else "NO"))
    return out.reset_index(drop=True)


def quality_spread(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """Per (detector, size): F1 range over all modes. For the deterministic kernels it must be 0 (by design, H2)."""
    sub = ct[(ct["s"] == s0) & ct["f1"].notna()]
    rows = []
    for (det, n), g in sub.groupby(["detector", "n"]):
        rows.append({"detector": det, "n": n, "modes": len(g), "f1_min": g["f1"].min(), "f1_max": g["f1"].max(),
                     "f1_range": g["f1"].max() - g["f1"].min(), "deterministic_kernel": det in DETERMINISTIC})
    return pd.DataFrame(rows)


def g_lang_table(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    """Language gain next to the best parallel speedup at the same size; the two are never combined."""
    rows = []
    for (det, n), g in ct[ct["s"] == s0].groupby(["detector", "n"]):
        py = g[g["mode"] == "PY-SEQ"]
        if py.empty:
            continue
        omp = g[g["family"] == "OMP"]
        best = omp.loc[omp["speedup"].idxmax()] if len(omp) else None
        rows.append({"detector": det, "n": n, "T_py_s": py.iloc[0]["med"], "T1_s": py.iloc[0]["t1"], "G_lang": py.iloc[0]["g_lang"],
                     "best_OMP": best["label"] if best is not None else "", "S_p_best": best["speedup"] if best is not None else np.nan})
    return pd.DataFrame(rows)


def stride_effect(ct: pd.DataFrame, s0: int) -> pd.DataFrame:
    other = ct[ct["s"] != s0]
    if other.empty:
        return pd.DataFrame()
    base = ct[ct["s"] == s0].set_index(["detector", "n", "label"])["med"]
    rows = []
    for _, r in other.iterrows():
        rows.append({"detector": r["detector"], "n": r["n"], "s": r["s"], "label": r["label"], "med_s": r["med"],
                     "med_s_at_base_stride": base.get((r["detector"], r["n"], r["label"]), np.nan)})
    return pd.DataFrame(rows)


def build_tables(ct: pd.DataFrame, s0: int) -> Dict[str, pd.DataFrame]:
    if ct.empty:
        return {}
    phase = ct[ct["family"].isin(["MPI", "HYB"]) & (ct["s"] == s0)][
        ["detector", "n", "label", "med", "phase_input", "phase_compute", "phase_gather", "phase_other", "comm_share"]]
    return {
        "indicators": ct.drop(columns=["features_med", "dll_ms_med"]),
        "speedup_omp": wide(ct, "OMP", "speedup", s0), "efficiency_omp": wide(ct, "OMP", "efficiency", s0),
        "karp_flatt_omp": wide(ct, "OMP", "karp_flatt", s0),
        "speedup_mpi": wide(ct, "MPI", "speedup", s0), "speedup_hyb": wide(ct, "HYB", "speedup", s0),
        "throughput": ct[ct["s"] == s0][["detector", "n", "label", "family", "throughput_sps", "throughput_lo", "throughput_hi"]],
        "g_lang": g_lang_table(ct, s0), "quality_vs_mode": quality_vs_mode(ct, s0), "quality_spread": quality_spread(ct, s0),
        "not_faster": not_faster(ct, s0), "crossover": crossover(ct, s0), "equal_workers": equal_workers(ct, s0),
        "phase_breakdown": phase, "amdahl": amdahl_table(ct[ct["s"] == s0]), "stride_effect": stride_effect(ct, s0),
    }


def write_tables(tables: Dict[str, pd.DataFrame], out_dir) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in tables.items():
        if df is not None and not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)
            written.append(f"{name}.csv")
    return written
