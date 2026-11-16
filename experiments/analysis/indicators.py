"""D-09.1 - HPC indicators and the per-cell table built from results.csv.

Formulas (T1 is always the C-1 median of the same dataset, detector, window and stride, never Python):
    speedup      S_p  = T1 / Tp
    efficiency   E_p  = S_p / workers            (workers = ranks x threads)
    Karp-Flatt   e    = (1/S_p - 1/workers) / (1 - 1/workers)      (workers > 1 only)
    throughput        = n / Tp                                       (samples per second)
    language gain G   = T_py / T1                                    (never mixed into S_p)

Medians everywhere; the spread is the interquartile range (IQR). Tp's IQR gives the interval of S_p and E_p
(T1's own spread is kept in the table and used when two cells are compared).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

FAMILY = {"PY-SEQ": "PY-SEQ", "C-1": "C-1", "OMP": "OMP", "MPI": "MPI", "MPI-scatter": "MPI", "HYB": "HYB"}
FAMILY_ORDER = ["PY-SEQ", "C-1", "OMP", "MPI", "HYB"]
DETERMINISTIC = ("zscore", "mad", "kmeans")          # kernels whose output must not depend on threads or ranks (H2)
KEY = ["dataset", "n", "w", "s", "detector"]


# ------------------------------------------------------------------ scalar / vector formulas
def speedup(t1, tp):
    return np.asarray(t1, dtype=float) / np.asarray(tp, dtype=float)


def efficiency(s, workers):
    return np.asarray(s, dtype=float) / np.asarray(workers, dtype=float)


def karp_flatt(s, workers):
    """Experimentally determined serial fraction; NaN where workers <= 1 (undefined)."""
    s, p = np.asarray(s, dtype=float), np.asarray(workers, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (1.0 / s - 1.0 / p) / (1.0 - 1.0 / p)
    return np.where(p > 1, e, np.nan)


def throughput(n, t):
    return np.asarray(n, dtype=float) / np.asarray(t, dtype=float)


def g_lang(t_py, t1):
    return np.asarray(t_py, dtype=float) / np.asarray(t1, dtype=float)


def amdahl_serial_fraction(workers, speedups) -> float:
    """Least-squares serial fraction f of Amdahl's law S = 1 / (f + (1 - f)/p), clipped to [0, 1].
    Linear in f: 1/S - 1/p = f (1 - 1/p), so f = sum(a b) / sum(b b) with a = 1/S - 1/p and b = 1 - 1/p."""
    p, s = np.asarray(workers, dtype=float), np.asarray(speedups, dtype=float)
    m = (p > 1) & np.isfinite(s) & (s > 0)
    if not m.any():
        return float("nan")
    a, b = 1.0 / s[m] - 1.0 / p[m], 1.0 - 1.0 / p[m]
    return float(np.clip((a * b).sum() / (b * b).sum(), 0.0, 1.0))


# ------------------------------------------------------------------ comparing two cells (IQR logic)
def clearly_faster(a_q3: float, b_q1: float) -> bool:
    """A is faster than B beyond the spread: the slowest quarter of A's runs is still faster than B's fastest quarter."""
    return a_q3 < b_q1


def compare(a, b) -> str:
    """'faster' / 'slower' / 'no measurable difference' for two cell rows (needs q1, q3). Overlapping IQRs are never
    called a difference."""
    if a["q3"] < b["q1"]:
        return "faster"
    if a["q1"] > b["q3"]:
        return "slower"
    return "no measurable difference"


# ------------------------------------------------------------------ loading
def label_of(mode: str, q: int, p: int) -> str:
    q, p = int(q), int(p)
    return {"PY-SEQ": "PY-SEQ", "C-1": "C-1", "OMP": f"OMP-{p}", "MPI": f"MPI-{q}", "MPI-scatter": f"MPI-{q}-scatter",
            "HYB": f"HYB-{q}\u00d7{p}"}.get(mode, mode)


def load_results(path) -> pd.DataFrame:
    """Rows with status == 'ok' only (warm-ups, timeouts, errors, crashes and superseded runs are excluded)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"identical": str, "note": str, "status": str})
    df = df[df["status"] == "ok"].copy()
    df["identical"] = df["identical"].map({"True": True, "False": False})
    df["family"] = df["mode"].map(FAMILY)
    df["workers"] = df["q_ranks"] * df["p_threads"]
    df["label"] = [label_of(m, q, p) for m, q, p in zip(df["mode"], df["q_ranks"], df["p_threads"])]
    return df.reset_index(drop=True)


def load_machine(path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def base_stride(df: pd.DataFrame) -> int:
    """The stride used for the main figures: the one with most cells (the matrix has a second stride on one size only)."""
    return int(df.groupby("s").size().idxmax()) if len(df) else 1


# ------------------------------------------------------------------ per-cell table
def _q(p):
    return lambda s: float(np.percentile(s, p))


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell with median, quartiles, T1, speedup (+ IQR interval), efficiency, Karp-Flatt, throughput, phases."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(KEY + ["mode", "q_ranks", "p_threads", "workers", "family", "label"], sort=False)
    ct = g.agg(runs=("total_s", "size"), med=("total_s", "median"), q1=("total_s", _q(25)), q3=("total_s", _q(75)),
               tmin=("total_s", "min"), tmax=("total_s", "max"),
               load_med=("load_s", "median"), features_med=("features_s", "median"), score_med=("score_s", "median"),
               comm_med=("comm_s", "median"), dll_ms_med=("dll_ms", "median"),
               f1=("f1", "median"), precision=("precision", "median"), recall=("recall", "median"),
               event_recall=("event_recall", "median"),
               identical_all=("identical", lambda s: bool(s.all()) if s.notna().all() else None),
               identical_known=("identical", lambda s: int(s.notna().sum()))).reset_index()
    c1 = ct[ct["mode"] == "C-1"].set_index(KEY)[["med", "q1", "q3"]].rename(columns={"med": "t1", "q1": "t1_q1", "q3": "t1_q3"})
    ct = ct.join(c1, on=KEY)
    py = ct[ct["mode"] == "PY-SEQ"].set_index(KEY)["med"].rename("t_py")
    ct = ct.join(py, on=KEY)
    par = ct["family"].isin(["OMP", "MPI", "HYB"])
    ct["speedup"] = np.where(par | (ct["mode"] == "C-1"), speedup(ct["t1"], ct["med"]), np.nan)
    ct["speedup_lo"] = np.where(par, speedup(ct["t1"], ct["q3"]), np.nan)       # slow end of Tp -> low speedup
    ct["speedup_hi"] = np.where(par, speedup(ct["t1"], ct["q1"]), np.nan)
    ct["efficiency"] = np.where(par, efficiency(ct["speedup"], ct["workers"]), np.nan)
    ct["efficiency_lo"] = efficiency(ct["speedup_lo"], ct["workers"])
    ct["efficiency_hi"] = efficiency(ct["speedup_hi"], ct["workers"])
    ct["karp_flatt"] = np.where(par, karp_flatt(ct["speedup"], ct["workers"]), np.nan)
    ct["throughput_sps"] = throughput(ct["n"], ct["med"])
    ct["throughput_lo"] = throughput(ct["n"], ct["q3"])
    ct["throughput_hi"] = throughput(ct["n"], ct["q1"])
    ct["g_lang"] = np.where(ct["mode"] == "PY-SEQ", g_lang(ct["t_py"], ct["t1"]), np.nan)
    # phases of the MPI driver (D-06): input = load_s; compute = score_s; gather = comm_s minus the input in scatter mode
    mpi = ct["family"].isin(["MPI", "HYB"])
    ct["phase_input"] = np.where(mpi, ct["load_med"], np.nan)
    ct["phase_compute"] = np.where(mpi, ct["score_med"], np.nan)
    ct["phase_gather"] = np.where(mpi, ct["comm_med"] - np.where(ct["mode"] == "MPI-scatter", ct["load_med"], 0.0), np.nan)
    ct["phase_other"] = np.where(mpi, np.maximum(ct["med"] - ct["phase_input"] - ct["phase_compute"] - ct["phase_gather"], 0.0), np.nan)
    ct["comm_share"] = np.where(mpi, (ct["phase_gather"] + np.where(ct["mode"] == "MPI-scatter", ct["phase_input"], 0.0)) / ct["med"], np.nan)
    order = ct["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return ct.assign(_o=order).sort_values(["detector", "n", "s", "_o", "q_ranks", "p_threads", "mode"]).drop(columns="_o").reset_index(drop=True)


def amdahl_table(ct: pd.DataFrame) -> pd.DataFrame:
    """Amdahl serial fraction fitted on the OMP cells of each (detector, size)."""
    rows = []
    for (det, n, s), g in ct[ct["family"] == "OMP"].groupby(["detector", "n", "s"]):
        rows.append({"detector": det, "n": n, "s": s, "points": len(g),
                     "serial_fraction": amdahl_serial_fraction(g["workers"], g["speedup"])})
    return pd.DataFrame(rows)
