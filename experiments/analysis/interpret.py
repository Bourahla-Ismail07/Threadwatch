"""D-09.4 - written interpretation per hypothesis, generated from the data by explicit rules.

Nothing here is hand-written about results: every verdict is a function of the cell table, every number in the text is
read from it, and the rules are printed in docs/results.md ("How the verdicts are decided"). Rules in short:

  H1  z-score and MAD, OpenMP, large sizes, thread counts up to the physical cores: supported if the median efficiency is
      >= 70% in every such cell; rejected if in some cell even the fast end of the IQR is < 70%; otherwise inconclusive.
  H2  deterministic kernels (z-score, MAD, K-Means) in every OMP/MPI/HYB cell: supported if all are identical to C-1,
      rejected if any is not, inconclusive if identity was not checked everywhere.
  H3  a parallel mode is "slower" than C-1 only if its IQR lies entirely above C-1's (and "faster" if entirely below);
      overlapping IQRs are "no measurable difference". Supported if a slower-to-faster crossover is located; rejected only if the
      smallest measured size is <= 100k and parallel was clearly faster everywhere; otherwise inconclusive.
  H4  at equal workers and large sizes, MPI (file input) versus OpenMP: supported if MPI is never clearly faster, rejected if
      it is clearly faster in more than half of the comparisons, otherwise inconclusive. Hybrid is reported separately.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import DETERMINISTIC, compare
from .tables import PARALLEL, crossover, equal_workers, fmt_approx, fmt_n, md_table, not_faster, quality_pivot

LARGE_N = 5_000_000
EFF_LIMIT = 0.70
SMALL_FOR_REJECTION = 100_000
H1_KERNELS = ("zscore", "mad")
NAMES = {"zscore": "z-score", "mad": "MAD", "kmeans": "K-Means", "iforest": "Isolation Forest"}

FIGURE_FILES = [
    (1, "Speedup versus threads, per kernel, with the ideal line", "H1"),
    (2, "Efficiency versus threads", "H1"),
    (3, "Throughput versus series size, per mode", "H3"),
    (4, "Time versus size for C-1 and OMP-p (log axes)", "H3"),
    (5, "Karp-Flatt serial fraction versus threads", "H1"),
    (6, "Phase breakdown for MPI and HYB", "H4"),
    (7, "OMP-p versus MPI-q versus HYB at equal total workers", "H4"),
    (8, "Language gain and parallel gain, side by side", "claim hygiene"),
    (9, "schedule(static) versus schedule(dynamic) on MAD", "H1"),
    (10, "F1 per mode", "H2"),
]


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.0f}%"


def large_sizes(df: pd.DataFrame, min_n: int) -> List[int]:
    sizes = sorted(df["n"].unique())
    big = [int(n) for n in sizes if n >= min_n]
    return big or ([int(sizes[-1])] if sizes else [])


def machine_line(m: dict) -> str:
    if not m:
        return "machine record missing (run experiments/machine_info.py --write)"
    return (f"{m.get('processor', '?')}, {m.get('logical_cores', '?')} logical / {m.get('physical_cores', '?')} physical cores, "
            f"{m.get('ram_gb', '?')} GB RAM, {m.get('os', '?')}")


def _verdict(result, evidence, numbers, reading, limit, tables=None, figs=()):
    return {"result": result, "evidence": evidence, "numbers": numbers, "reading": reading, "limit": limit,
            "tables": tables or {}, "figs": list(figs)}


def _runs(df) -> str:
    a, b = int(df["runs"].min()), int(df["runs"].max())
    return str(a) if a == b else f"{a} to {b}"


def kf_trend(ps, es) -> str:
    ps, es = np.asarray(ps, float), np.asarray(es, float)
    m = np.isfinite(es)
    if m.sum() < 2:
        return "not enough points"
    ps, es = ps[m], es[m]
    rel = (es[-1] - es[0]) / max(float(np.mean(np.abs(es))), 1e-9)
    kind = "growing" if rel > 0.25 else ("decreasing" if rel < -0.25 else "roughly constant")
    return kind + (" (only two points: indicative)" if len(es) == 2 else "")


# ---------------------------------------------------------------------------- H1
def hypothesis_1(ct, machine, s0, min_n=LARGE_N) -> dict:
    phys = machine.get("physical_cores")
    omp = ct[(ct["family"] == "OMP") & (ct["s"] == s0) & ct["detector"].isin(H1_KERNELS)]
    figs = [1, 2, 5, 9]
    if omp.empty:
        return _verdict("inconclusive", "no OpenMP results for z-score or MAD", [], "Nothing to read.", "No data.", figs=figs)
    sizes = large_sizes(omp, min_n)
    big = omp[omp["n"].isin(sizes)]
    scope = big[big["p_threads"] <= phys] if phys else big
    numbers, reading = [], []
    if not phys:
        return _verdict("inconclusive", "physical core count unknown (machine.json missing)", [], "H1 is defined up to the physical core count.",
                        "Run experiments/machine_info.py --write.", figs=figs)
    if scope.empty:
        return _verdict("inconclusive", f"no OpenMP cell at or below {phys} threads", [], "Nothing to read.", "No data.", figs=figs)
    below_median = scope[scope["efficiency"] < EFF_LIMIT]
    clearly_below = scope[scope["efficiency_hi"] < EFF_LIMIT]
    result = "supported" if below_median.empty else ("rejected" if len(clearly_below) else "inconclusive")
    pmax = int(scope["p_threads"].max())
    for det in H1_KERNELS:
        g = big[big["detector"] == det]
        if g.empty:
            continue
        n_big = int(g["n"].max())
        gg = g[g["n"] == n_big].sort_values("p_threads")
        parts = [f"p={int(r.p_threads)}: S={r.speedup:.2f}, E={pct(r.efficiency)}" for r in gg.itertuples()]
        numbers.append(f"{NAMES[det]} at {fmt_n(n_big)}: " + "; ".join(parts))
        knee = next((int(r.p_threads) for r in gg.itertuples() if r.efficiency < EFF_LIMIT), None)
        within = gg[gg["p_threads"] <= phys]
        trend = kf_trend(within["p_threads"], within["karp_flatt"])
        if knee is None:
            reading.append(f"{NAMES[det]}: efficiency stays at or above {pct(EFF_LIMIT)} at every measured thread count up to {int(gg['p_threads'].max())}.")
        elif knee > phys:
            reading.append(f"{NAMES[det]}: efficiency falls below {pct(EFF_LIMIT)} only at p={knee}, beyond the {phys} physical cores, "
                           "where threads share cores (hyper-threads); this is expected, not a defect of the kernel.")
        else:
            cause = {"roughly constant": "a roughly constant Karp-Flatt value points to a fixed serial part",
                     "growing": "a growing Karp-Flatt value points to overhead that grows with the thread count (memory bandwidth, scheduling)",
                     }.get(trend.split(" (")[0], "the Karp-Flatt trend is not conclusive")
            reading.append(f"{NAMES[det]}: efficiency falls below {pct(EFF_LIMIT)} at p={knee}, inside the {phys} physical cores; {cause} (likely cause, not tested separately).")
        numbers.append(f"{NAMES[det]} Karp-Flatt trend up to {phys} threads: {trend}")
    if len(below_median):
        numbers.append(f"{len(below_median)} of {len(scope)} cells at or below {phys} threads are under {pct(EFF_LIMIT)}: "
                       + ", ".join(f"{NAMES[r.detector]} {fmt_n(r.n)} p={int(r.p_threads)} ({pct(r.efficiency)})" for r in below_median.head(6).itertuples()))
    tested = sorted(int(p) for p in omp["p_threads"].unique())
    if phys not in tested:
        numbers.append(f"NOTE: the physical core count ({phys}) was not measured; the verdict covers thread counts up to {pmax}. Add {phys} to the OMP thread list to cover it.")
    limit = (f"One machine ({machine_line(machine)}). Kernels: z-score and MAD, sizes {', '.join(fmt_n(n) for n in sizes)}. "
             f"Thread counts measured: {tested}; the physical core count ({phys}) {'was' if phys in tested else 'was not'} measured itself "
             f"(largest tested at or below it: {pmax}). Runs per cell: {_runs(scope)}. "
             "The optional sliding-sum variant is not part of these results.")
    return _verdict(result, "F1, F2, F5 (F9 for the schedule comparison)", numbers, " ".join(reading), limit, figs=figs)


# ---------------------------------------------------------------------------- H2
def hypothesis_2(ct, machine, s0) -> dict:
    sub = ct[ct["family"].isin(PARALLEL) & ct["detector"].isin(DETERMINISTIC) & (ct["s"] == s0)]
    figs = [10]
    if sub.empty:
        return _verdict("inconclusive", "no parallel cell of a deterministic kernel", [], "Nothing to read.", "No data.", figs=figs)
    bad = sub[sub["identical_all"].map(lambda v: v is False)]
    unknown = sub[sub["identical_all"].map(lambda v: v is None or v != v)]
    result = "rejected" if len(bad) else ("inconclusive" if len(unknown) else "supported")
    spread = ct[(ct["s"] == s0) & ct["detector"].isin(DETERMINISTIC) & ct["f1"].notna()].groupby(["detector", "n"])["f1"].agg(lambda s: s.max() - s.min())
    numbers = [f"{len(sub) - len(bad) - len(unknown)} of {len(sub)} OMP/MPI/HYB cells of z-score, MAD and K-Means are identical to C-1 in every timed run",
               f"largest F1 range across all modes of one (kernel, size): {float(spread.max()) if len(spread) else float('nan'):.4f}"]
    if len(bad):
        numbers.append("NOT identical: " + ", ".join(f"{NAMES[r.detector]} {fmt_n(r.n)} {r.label}" for r in bad.head(8).itertuples()))
    if len(unknown):
        numbers.append(f"identity not checked in {len(unknown)} cells (no C-1 reference)")
    ifo = ct[ct["family"].isin(PARALLEL) & (ct["detector"] == "iforest") & (ct["s"] == s0)]
    if len(ifo):
        numbers.append(f"Isolation Forest (library scoring, reported separately): {int(ifo['identical_all'].map(lambda v: v is True).sum())} of {len(ifo)} cells identical")
    reading = ("Identity is a design property: one writer per output slot, no reduction across threads, no fast-math, so the result does not depend on "
               "threads or ranks. It is a check that the implementation keeps that property, not a finding about the data."
               if result == "supported" else
               "At least one parallel result differs from the 1-thread result, which should not happen for these kernels; investigate before using any speedup of that cell.")
    limit = f"Exact comparison (array equality) of every timed run against the C-1 output of the same block. One machine ({machine_line(machine)}); sizes and modes as measured."
    return _verdict(result, "F10 and the quality table", numbers, reading, limit, figs=figs)


# ---------------------------------------------------------------------------- H3
def hypothesis_3(ct, machine, s0) -> dict:
    figs = [3, 4]
    cx = crossover(ct, s0)
    sub = ct[ct["family"].isin(PARALLEL) & (ct["s"] == s0)]
    if sub.empty:
        return _verdict("inconclusive", "no parallel cells", [], "Nothing to read.", "No data.", figs=figs), cx
    nsmall = int(sub["n"].min())
    located = cx[cx["status"] == "crossover located"]
    slower_any = cx[cx["status"].str.startswith("slower")]
    all_faster = cx[cx["status"].str.startswith("faster at every")]
    numbers = [f"smallest measured size: {fmt_n(nsmall)}", f"{len(cx)} (kernel, mode) series: {len(located)} with a located crossover, "
               f"{len(slower_any)} slower at the measured sizes, {len(all_faster)} faster at every measured size"]
    for r in located.head(6).itertuples():
        numbers.append(f"{NAMES[r.detector]} {r.label}: slower than C-1 up to {r.slower_below}, faster from {r.faster_from} "
                       f"(interpolated crossover about {fmt_approx(r.crossover_estimate_n)})")
    if len(located):
        result = "supported"
        reading = ("Below the crossover the parallel run is slower than one thread: starting threads or ranks and splitting the work costs more than "
                   "it saves. The crossover is the measured size where the IQRs stop overlapping and the parallel mode becomes faster; the interpolated value is an estimate between two measured sizes.")
    elif len(slower_any):
        result = "inconclusive"
        reading = ("Some modes are slower than C-1 at the measured sizes but none was clearly faster at a larger size, so the crossover lies above the measured range. "
                   "Parallel overhead is visible; its end point is not.")
    elif nsmall <= SMALL_FOR_REJECTION and len(all_faster) == len(cx):
        result = "rejected"
        reading = f"Every parallel mode was already clearly faster than C-1 at {fmt_n(nsmall)}, so no slow region was found down to that size."
    else:
        result = "inconclusive"
        reading = (f"No parallel mode was clearly slower than C-1 at any measured size, and the smallest size is {fmt_n(nsmall)}: the region where "
                   "overhead dominates, if it exists, lies below the measured range. Add smaller sizes to the matrix (for example 100k, 250k, 500k) to locate it.")
    limit = (f"Sizes measured: {', '.join(fmt_n(n) for n in sorted(sub['n'].unique()))}. 'Slower' and 'faster' require non-overlapping IQRs ({_runs(sub)} runs per cell); overlapping IQRs are reported as no measurable difference. One machine ({machine_line(machine)}).")
    return _verdict(result, "F3, F4 and the not-faster table", numbers, reading, limit, figs=figs), cx


# ---------------------------------------------------------------------------- H4
def hypothesis_4(ct, machine, s0, min_n=LARGE_N) -> dict:
    figs = [6, 7]
    ew = equal_workers(ct, s0)
    if ew.empty:
        return _verdict("inconclusive", "no MPI or hybrid cell with an OpenMP cell at the same worker count", [], "Nothing to read.", "No data.", figs=figs)
    sizes = large_sizes(ew, min_n)
    big = ew[ew["n"].isin(sizes)]
    mpi = big[big["label"].str.match(r"^MPI-\d+$")]
    hyb = big[big["label"].str.startswith("HYB")]
    sca = big[big["label"].str.endswith("scatter")]
    cnt = lambda d, k: int((d["vs_OMP_same_workers"] == k).sum())
    numbers = []
    if len(mpi):
        numbers.append(f"MPI (file input) versus OpenMP at equal workers, sizes {', '.join(fmt_n(n) for n in sizes)}: "
                       f"MPI clearly faster in {cnt(mpi, 'faster')}, clearly slower in {cnt(mpi, 'slower')}, no measurable difference in {cnt(mpi, 'no measurable difference')} of {len(mpi)} comparisons")
        share = mpi["comm_share"].median()
        numbers.append(f"median share of the MPI run spent in input and gather: {pct(share)}")
    if len(sca):
        numbers.append(f"scatter input versus OpenMP: clearly slower in {cnt(sca, 'slower')} of {len(sca)} comparisons")
    if len(hyb):
        numbers.append(f"hybrid versus OpenMP at equal workers: clearly faster in {cnt(hyb, 'faster')}, clearly slower in {cnt(hyb, 'slower')}, "
                       f"no measurable difference in {cnt(hyb, 'no measurable difference')} of {len(hyb)} comparisons")
    for det in [d for d in NAMES if d in set(ct["detector"])]:
        g = ct[(ct["detector"] == det) & ct["family"].isin(PARALLEL) & (ct["s"] == s0)]
        if g.empty:
            continue
        g = g[g["n"] == g["n"].max()]
        b = g.loc[g["med"].idxmin()]
        numbers.append(f"fastest parallel configuration, {NAMES[det]} at {fmt_n(b['n'])}: {b['label']} ({b['med']:.3g} s, S={b['speedup']:.2f})")
    if not len(mpi):
        result, reading = "inconclusive", "No MPI cell can be compared with an OpenMP cell at the same worker count."
    else:
        wins, total = cnt(mpi, "faster"), len(mpi)
        result = "supported" if wins == 0 else ("rejected" if wins > total / 2 else "inconclusive")
        reading = ("On one machine every rank shares the same memory and cores, and MPI adds input and gather work that OpenMP does not have; "
                   f"here MPI was {'never' if wins == 0 else 'sometimes'} clearly faster than OpenMP at equal workers. ")
        if len(hyb):
            hw, hl = cnt(hyb, "faster"), cnt(hyb, "slower")
            reading += (f"Hybrid beat OpenMP clearly in {hw} and lost clearly in {hl} comparisons, so hybrid "
                        + ("helps only for some shapes: the shape (ranks x threads) has to be tuned." if hw and hl else
                           "did not beat OpenMP in these measurements." if not hw else "beat OpenMP wherever it was measured."))
    limit = (f"Equal-workers comparisons exist only for the shapes in the matrix (MPI-2, MPI-4; HYB-2x2, HYB-2x4). One machine ({machine_line(machine)}). "
             "The ranks run on the same host: this says nothing about a cluster, where memory is not shared and the network replaces shared memory.")
    return _verdict(result, "F6, F7 and the equal-workers table", numbers, reading, limit, figs=figs)


# ---------------------------------------------------------------------------- not faster
def explain_not_faster(ct, machine, s0, limit_rows=6) -> str:
    nf = not_faster(ct, s0)
    sub = ct[ct["family"].isin(PARALLEL) & (ct["s"] == s0)]
    if nf.empty:
        return (f"No parallel cell was slower than or indistinguishable from C-1 among the {len(sub)} measured; the smallest size is "
                f"{fmt_n(sub['n'].min()) if len(sub) else 'n/a'}. Any region where overhead wins lies below the measured sizes.\n")
    phys = machine.get("physical_cores")
    out = [f"{len(nf)} of {len(sub)} parallel cells are not clearly faster than C-1 (speedup at most 1, or overlapping IQRs). The lowest speedups:\n"]
    for r in nf.sort_values("speedup").head(limit_rows).itertuples():
        why = []
        if r.comm_share == r.comm_share and r.comm_share is not None:
            why.append(f"input and gather take {pct(r.comm_share)} of the run")
        if phys and r.workers > phys:
            why.append(f"{int(r.workers)} workers exceed the {phys} physical cores")
        if not why:
            why.append(f"each thread gets about {r.n / r.workers:,.0f} windows, so start-up and memory traffic weigh against the work")
        out.append(f"- {NAMES[r.detector]}, {fmt_n(r.n)}, {r.label}: S={r.speedup:.2f} ({r.vs_C1}); likely cause: {'; '.join(why)} (not tested separately).")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------- results.md
def verdicts(ct, machine, s0, min_n=LARGE_N) -> Dict[str, dict]:
    h3, _ = hypothesis_3(ct, machine, s0)
    return {"H1": hypothesis_1(ct, machine, s0, min_n), "H2": hypothesis_2(ct, machine, s0), "H3": h3, "H4": hypothesis_4(ct, machine, s0, min_n)}


TITLES = {"H1": ("OpenMP efficiency", "z-score and MAD keep parallel efficiency above 70% up to the physical core count."),
          "H2": ("Equivalence", "parallel outputs are identical to the 1-thread outputs, so F1 does not change."),
          "H3": ("Size", "below a threshold size parallel overhead makes a run slower than one thread; the crossover is measured."),
          "H4": ("MPI", "on a single machine MPI does not beat OpenMP; hybrid helps only when ranks x threads is tuned.")}


def render_results(ct, tables, machine, s0, v, figure_status, protocol: dict) -> str:
    o = ["# Results and interpretation (D-09)\n",
         "Generated by `python experiments/run_analysis.py` from `experiments/out/results.csv`. Do not edit by hand.\n"]
    runs = str(protocol["runs_min"]) if protocol["runs_min"] == protocol["runs_max"] else f"{protocol['runs_min']} to {protocol['runs_max']}"
    o.append(f"\n**Machine:** {machine_line(machine)}. **Protocol:** w={protocol['w']}, s={s0}; "
             f"{runs} timed runs per cell (median, IQR as spread); T1 is the C-1 median of the same size and detector; "
             "speedup and efficiency are never computed against Python.\n")
    o.append("\n## Verdicts\n\n| | Hypothesis | Result |\n|---|---|---|\n" + "".join(
        f"| {h} | {TITLES[h][0]}: {TITLES[h][1]} | **{v[h]['result']}** |\n" for h in ("H1", "H2", "H3", "H4")))
    for h in ("H1", "H2", "H3", "H4"):
        d = v[h]
        o.append(f"\n## {h} - {TITLES[h][0]}\n\n**Result:** {d['result']}\n\n**Evidence:** {d['evidence']}.\n\n" + "".join(f"- {x}\n" for x in d["numbers"]))
        o.append(f"\n**Reading:** {d['reading']}\n\n**Limit:** {d['limit']}\n")
        for f in d["figs"]:
            if figure_status.get(f) == "ok":
                o.append(f"\n![F{f:02d}](../experiments/out/figures/F{f:02d}.png)\n")
            else:
                o.append(f"\n_F{f:02d}: {figure_status.get(f, 'not generated')}_\n")
    o.append("\n## Where parallel is not faster\n\n" + explain_not_faster(ct, machine, s0))
    nf = tables.get("not_faster")
    if nf is not None and not nf.empty:
        o.append("\n" + md_table(nf.sort_values("speedup").head(15), "{:.3g}"))
    o.append("\n## Quality versus mode (F1 and identity with C-1)\n\nF1 is identical across modes for the deterministic kernels by design (H2); it is not a finding about the data.\n\n")
    q = tables.get("quality_vs_mode")
    if q is not None and not q.empty:
        o.append("F1 at the largest size of each kernel (\"differs\" = not identical to C-1):\n\n" + md_table(quality_pivot(q)) +
                 "\nAll sizes, precision, recall and event recall: `experiments/out/tables/quality_vs_mode.csv`.\n")
    o.append("\n## Language gain and parallel gain\n\n`G_lang = T_py / T1` is the effect of the language; `S_p = T1 / Tp` is the effect of parallelism. They are separate.\n\n"
             + md_table(tables.get("g_lang"), "{:.2f}"))
    o.append("\n## Figures\n\n" + "".join(
        f"- F{i:02d} ({h}): {t} - {'`experiments/out/figures/F%02d.png`' % i if figure_status.get(i) == 'ok' else figure_status.get(i, 'not generated')}\n"
        for i, t, h in FIGURE_FILES))
    o.append(f"""
## How the verdicts are decided

Every verdict is computed from the data by these rules (`experiments/analysis/interpret.py`); a result is *inconclusive* when the data
cannot settle the question.

- **Medians and spread.** Medians of the timed runs; the interquartile range (IQR) is the spread. Two cells are different only if their IQRs do not overlap.
- **H1.** z-score and MAD, OpenMP, sizes of at least {fmt_n(LARGE_N)} (else the largest size), thread counts up to the physical core count. Supported if the median efficiency
  is at least {pct(EFF_LIMIT)} in every such cell; rejected if in some cell even the fast end of the IQR is below it; inconclusive otherwise.
- **H2.** Deterministic kernels (z-score, MAD, K-Means), every OMP, MPI and HYB cell: supported if all are identical to C-1, rejected if any is not, inconclusive if identity was not checked everywhere.
- **H3.** A parallel mode is *slower* than C-1 only if its IQR lies entirely above C-1's, *faster* if entirely below. Supported if a slower-to-faster crossover is located;
  rejected only if the smallest size is at most {fmt_n(SMALL_FOR_REJECTION)} and every parallel mode was clearly faster; inconclusive otherwise.
- **H4.** At equal workers and large sizes, MPI with file input against OpenMP: supported if MPI is never clearly faster, rejected if it is clearly faster in more than half of the comparisons, inconclusive otherwise. Hybrid and scatter are reported alongside.
- **Causes** in the readings are plausible explanations drawn from the numbers (physical cores, Karp-Flatt trend, communication share); they are marked as not tested separately.
- **Efficiency beyond the physical cores** is lower by construction (threads share cores); the physical core count is marked on the plots.
""")
    return "".join(o)
