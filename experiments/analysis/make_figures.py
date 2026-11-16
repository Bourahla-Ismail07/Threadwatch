"""D-09.2 - the ten figures, script-generated from results.csv (no manual editing).

    python experiments/analysis/make_figures.py        (or: python experiments/run_analysis.py for everything)

Style: brand colours and marker shapes per mode (never colour alone: also a label), medians with interquartile-range
error bars, linear axes unless the caption says log, every caption states machine, window, stride and number of runs.
A figure whose data is missing is still written, as a placeholder that says what is missing.
"""
from __future__ import annotations

import logging
import math
import textwrap
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from .indicators import FAMILY_ORDER  # noqa: E402
from .interpret import EFF_LIMIT, NAMES  # noqa: E402
from .tables import crossover, fmt_n  # noqa: E402

THEMES = {
    "light": dict(bg="#FFFFFF", fg="#0D1117", muted="#4B5668", grid="#D5DCE6", seq="#5B6675", omp="#0F766E", mpi="#1D4ED8",
                  hybrid="#6D28D9", alert="#B91C1C", warn="#92400E"),
    "dark": dict(bg="#0D1117", fg="#E6EDF3", muted="#8B98A9", grid="#263040", seq="#9AA6B2", omp="#2DD4BF", mpi="#60A5FA",
                 hybrid="#A78BFA", alert="#F87171", warn="#FBBF24"),
}
MARKER = {"PY-SEQ": "s", "C-1": "o", "OMP": "^", "MPI": "D", "HYB": "h"}          # square, circle, triangle, diamond, hexagon
COLORKEY = {"PY-SEQ": "seq", "C-1": "seq", "OMP": "omp", "MPI": "mpi", "HYB": "hybrid"}
DETECTORS = ["zscore", "mad", "kmeans", "iforest"]
HEAD = {"weight": "bold"}                                                  # "semibold" when Inter is installed (set in _setup_font)
SCHEDULE_CSV = "schedule_mad.csv"     # D-05.6 output contract: dataset,n,threads,schedule,repeat,total_s  (schedule: static | dynamic)


class Ctx:
    def __init__(self, ct, machine, s0, theme="light", outdir="experiments/out/figures", data_dir="experiments/out"):
        self.ct, self.machine, self.s0, self.T = ct, machine, s0, THEMES[theme]
        self.outdir, self.data_dir = Path(outdir), Path(data_dir)
        self.phys = machine.get("physical_cores")
        self.base = ct[ct["s"] == s0] if len(ct) else ct                  # empty table: placeholders only
        self.w = int(ct["w"].iloc[0]) if len(ct) else 256
        self.runs = (int(self.base["runs"].min()), int(self.base["runs"].max())) if len(self.base) else (0, 0)

    def color(self, family, shade=1.0):
        c = np.array(to_rgb(self.T[COLORKEY[family]]))
        bg = np.array(to_rgb(self.T["bg"]))
        return tuple(bg + (c - bg) * (0.35 + 0.65 * shade))          # shade 0 = pale, 1 = full colour

    def caption(self, scale: str, bars: bool = True) -> str:
        m = self.machine
        mach = (f"{m.get('processor', 'machine n/a')}, {m.get('logical_cores', '?')} logical / {m.get('physical_cores', '?')} physical cores"
                if m else "machine record missing")
        r = f"{self.runs[0]}" if self.runs[0] == self.runs[1] else f"{self.runs[0]} to {self.runs[1]}"
        return f"Machine: {mach}. Window w={self.w}, stride s={self.s0}. Median of {r} timed runs{'; error bars = interquartile range' if bars else ''}. Scale: {scale}."


# ------------------------------------------------------------------ helpers
def _setup_font():
    """Inter if installed (heading weight: the heaviest of semibold/bold it really has), else DejaVu Sans bold."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)     # the fallback font lacks some weights: harmless
    from matplotlib.font_manager import FontProperties
    HEAD["weight"] = "bold"
    try:
        font_manager.findfont("Inter", fallback_to_default=False)
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
        return
    plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
    for w in ("semibold", "bold"):
        try:
            path = font_manager.findfont(FontProperties(family="Inter", weight=w), fallback_to_default=False)
            if w in Path(path).stem.lower().replace("-", ""):
                HEAD["weight"] = w
                return
        except Exception:
            continue
    HEAD["weight"] = "normal"                                           # only a regular cut available


def _style_axes(ax, T):
    ax.set_facecolor(T["bg"])
    for sp in ax.spines.values():
        sp.set_color(T["grid"])
    ax.tick_params(colors=T["muted"], labelsize=8)
    ax.xaxis.label.set_color(T["fg"]); ax.yaxis.label.set_color(T["fg"]); ax.title.set_color(T["fg"])
    ax.grid(True, color=T["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def detectors_present(df) -> list:
    have = set(df["detector"].unique()) if len(df) else set()
    return [d for d in DETECTORS if d in have]


def _panels(n, ctx, width=5.2, height=3.7):
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows + 0.9), squeeze=False)
    fig.patch.set_facecolor(ctx.T["bg"])
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    for ax in flat[:n]:
        _style_axes(ax, ctx.T)
    return fig, flat[:n]


def _finish(fig, ctx, name, title, scale, bars=True):
    fig.suptitle(title, color=ctx.T["fg"], fontsize=11, fontweight=HEAD["weight"], x=0.01, ha="left")
    fig.text(0.01, 0.01, textwrap.fill(ctx.caption(scale, bars), 150), color=ctx.T["muted"], fontsize=7, va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    ctx.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(ctx.outdir / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return "ok"


def _placeholder(ctx, name, title, message):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    fig.patch.set_facecolor(ctx.T["bg"]); ax.set_facecolor(ctx.T["bg"]); ax.axis("off")
    ax.text(0.5, 0.55, "No data", ha="center", color=ctx.T["fg"], fontsize=16, fontweight=HEAD["weight"])
    ax.text(0.5, 0.38, textwrap.fill(message, 80), ha="center", va="top", color=ctx.T["muted"], fontsize=9)
    fig.suptitle(title, color=ctx.T["fg"], fontsize=11, fontweight=HEAD["weight"], x=0.01, ha="left")
    ctx.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(ctx.outdir / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"no data: {message}"


def _phys_line(ax, ctx, text=True):
    if ctx.phys:
        ax.axvline(ctx.phys, color=ctx.T["warn"], linestyle=":", linewidth=1.2)
        if text:
            ax.text(ctx.phys, ax.get_ylim()[1], f" {ctx.phys} physical cores", color=ctx.T["warn"], fontsize=7, va="top", ha="left")


def _err(med, lo, hi):
    return [np.maximum(np.asarray(med) - np.asarray(lo), 0), np.maximum(np.asarray(hi) - np.asarray(med), 0)]


def _size_shades(sizes):
    return {n: (i + 1) / len(sizes) for i, n in enumerate(sizes)}


def _omp_by_size(ctx, det):
    return ctx.base[(ctx.base["detector"] == det) & (ctx.base["family"] == "OMP")]


# ------------------------------------------------------------------ F1, F2, F5: OpenMP scaling
def _scaling(ctx, name, title, ycol, ylabel, y1, ideal, ylim0=True, hline=None, p_min=1):
    dets = [d for d in detectors_present(ctx.base) if not _omp_by_size(ctx, d).empty]
    if not dets:
        return _placeholder(ctx, name, title, "no OpenMP cells in results.csv")
    fig, axes = _panels(len(dets), ctx)
    for ax, det in zip(axes, dets):
        g = _omp_by_size(ctx, det)
        sizes = sorted(g["n"].unique())
        sh = _size_shades(sizes)
        pmax = int(g["p_threads"].max())
        if ideal is not None:
            xs = np.arange(max(p_min, 1), pmax + 1)
            ax.plot(xs, ideal(xs), "--", color=ctx.T["muted"], linewidth=1, label="ideal")
        for n in sizes:
            gg = g[g["n"] == n].sort_values("p_threads")
            xs, ys = list(gg["p_threads"]), list(gg[ycol])
            lo, hi = list(gg[ycol + "_lo"]) if ycol + "_lo" in gg else ys, list(gg[ycol + "_hi"]) if ycol + "_hi" in gg else ys
            if y1 is not None and p_min <= 1:                                    # p = 1 is C-1: speedup 1, efficiency 100%
                xs, ys, lo, hi = [1] + xs, [y1] + ys, [y1] + lo, [y1] + hi
            c = ctx.color("OMP", sh[n])
            ax.errorbar(xs, ys, yerr=_err(ys, lo, hi), color=c, marker=MARKER["OMP"], markersize=6, linewidth=1.6, capsize=2,
                        label=f"OMP-p, n = {fmt_n(n)}")
        ax.set_title(NAMES[det], fontsize=10, loc="left")
        ax.set_xlabel("OpenMP threads p"); ax.set_ylabel(ylabel)
        if ylim0:
            ax.set_ylim(bottom=0)
        ax.set_xticks(sorted(set(int(p) for p in g["p_threads"]) | ({1} if p_min <= 1 else set())))
        if hline:
            ax.axhline(hline[0], color=ctx.T["alert"], linestyle="--", linewidth=1); ax.text(ax.get_xlim()[0], hline[0], f" {hline[1]}", color=ctx.T["alert"], fontsize=7, va="bottom")
        _phys_line(ax, ctx)
        ax.legend(fontsize=7, frameon=False, labelcolor=ctx.T["fg"], loc="upper left" if ycol == "karp_flatt" else "best")
    return _finish(fig, ctx, name, title, "linear axes")


def f01(ctx):
    return _scaling(ctx, "F01.png", "F1 - OpenMP speedup versus threads (reference: C-1)", "speedup", "Speedup S_p = T1 / Tp (x)", 1.0, lambda x: x)


def f02(ctx):
    c = ctx
    ctx = Ctx.__new__(Ctx); ctx.__dict__.update(c.__dict__)
    ctx.base = c.base.assign(efficiency_pct=c.base["efficiency"] * 100, efficiency_pct_lo=c.base["efficiency_lo"] * 100,
                             efficiency_pct_hi=c.base["efficiency_hi"] * 100)
    return _scaling(ctx, "F02.png", "F2 - OpenMP efficiency versus threads", "efficiency_pct", "Efficiency E_p = S_p / p (%)", 100.0,
                    lambda x: np.full(len(x), 100.0), hline=(EFF_LIMIT * 100, "70%"))


def f05(ctx):
    return _scaling(ctx, "F05.png", "F5 - Karp-Flatt serial fraction versus threads", "karp_flatt", "Karp-Flatt e (serial fraction, 1 = all serial)", None, None,
                    ylim0=False, p_min=2)


# ------------------------------------------------------------------ F3, F4: size
def _label_style(ctx, g):
    """Family colour with a shade by worker count; marker by family."""
    out = {}
    for fam in FAMILY_ORDER:
        labs = g[g["family"] == fam].sort_values("workers")["label"].drop_duplicates().tolist()
        for i, lab in enumerate(labs):
            out[lab] = (fam, ctx.color(fam, (i + 1) / len(labs) if len(labs) > 1 else 1.0))
    return out


def f03(ctx):
    name, title = "F03.png", "F3 - Throughput versus series size, per mode"
    dets = detectors_present(ctx.base)
    if not dets or ctx.base["n"].nunique() < 1:
        return _placeholder(ctx, name, title, "no results")
    fig, axes = _panels(len(dets), ctx)
    for ax, det in zip(axes, dets):
        g = ctx.base[ctx.base["detector"] == det]
        st = _label_style(ctx, g)
        for lab, (fam, c) in st.items():
            gg = g[g["label"] == lab].sort_values("n")
            ys = gg["throughput_sps"] / 1e6
            ax.errorbar(gg["n"], ys, yerr=_err(ys, gg["throughput_lo"] / 1e6, gg["throughput_hi"] / 1e6), color=c, marker=MARKER[fam],
                        markersize=5, linewidth=1.3, capsize=2, linestyle="--" if lab.endswith("scatter") else "-", label=lab)
        ax.set_xscale("log"); ax.set_title(NAMES[det], fontsize=10, loc="left")
        ax.set_xlabel("Series size n (samples)"); ax.set_ylabel("Throughput (M samples/s)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, frameon=False, ncol=2, labelcolor=ctx.T["fg"])
    return _finish(fig, ctx, name, title, "log x axis (size), linear y axis")


def f04(ctx):
    name, title = "F04.png", "F4 - Time versus size: C-1 and OpenMP (log-log)"
    dets = detectors_present(ctx.base)
    g0 = ctx.base[ctx.base["family"].isin(["C-1", "OMP"])]
    if g0.empty:
        return _placeholder(ctx, name, title, "no C-1 or OpenMP cells")
    dets = [d for d in dets if d in set(g0["detector"])]
    cx = crossover(ctx.ct, ctx.s0)
    fig, axes = _panels(len(dets), ctx)
    for ax, det in zip(axes, dets):
        g = g0[g0["detector"] == det]
        st = _label_style(ctx, g)
        for lab, (fam, c) in st.items():
            gg = g[g["label"] == lab].sort_values("n")
            ax.errorbar(gg["n"], gg["med"], yerr=_err(gg["med"], gg["q1"], gg["q3"]), color=c, marker=MARKER[fam], markersize=5,
                        linewidth=1.3, capsize=2, label=lab)
        for r in cx[(cx["detector"] == det) & (cx["status"] == "crossover located") & cx["label"].str.startswith("OMP")].head(2).itertuples():
            ax.axvline(r.crossover_estimate_n, color=ctx.T["alert"], linestyle=":", linewidth=1)
            ax.text(r.crossover_estimate_n, ax.get_ylim()[1], f" {r.label} crossover ~{fmt_n(r.crossover_estimate_n)}", color=ctx.T["alert"], fontsize=7, va="top")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(NAMES[det], fontsize=10, loc="left")
        ax.set_xlabel("Series size n (samples)"); ax.set_ylabel("Time Tp (s)")
        ax.legend(fontsize=6, frameon=False, ncol=2, labelcolor=ctx.T["fg"])
    return _finish(fig, ctx, name, title, "log x and log y axes (small differences look smaller than they are)")


# ------------------------------------------------------------------ F6, F7: MPI and hybrid
def _size_for(ctx, det, families):
    g = ctx.base[(ctx.base["detector"] == det) & ctx.base["family"].isin(families)]
    return int(g["n"].max()) if len(g) else None


def f06(ctx):
    name, title = "F06.png", "F6 - Phase breakdown for MPI and hybrid runs"
    g0 = ctx.base[ctx.base["family"].isin(["MPI", "HYB"])]
    dets = [d for d in detectors_present(ctx.base) if d in set(g0["detector"])]
    if not dets:
        return _placeholder(ctx, name, title, "no MPI or hybrid cells")
    fig, axes = _panels(len(dets), ctx, width=5.6)
    for ax, det in zip(axes, dets):
        sc = g0[(g0["detector"] == det) & (g0["mode"] == "MPI-scatter")]
        n = int(sc["n"].max()) if len(sc) else _size_for(ctx, det, ["MPI", "HYB"])      # a size where file and scatter can be compared
        g = g0[(g0["detector"] == det) & (g0["n"] == n)].sort_values(["family", "workers", "mode"], key=lambda s: s.map({"MPI": 0, "HYB": 1}) if s.name == "family" else s)
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for col, hatch, lab in (("phase_input", "//", "input"), ("phase_compute", "", "compute"), ("phase_gather", "xx", "gather"), ("phase_other", ".", "other")):
            v = g[col].fillna(0).to_numpy()
            cols = [ctx.color(f) if col == "phase_compute" else ctx.color(f, 0.45) for f in g["family"]]
            ax.bar(x, v, bottom=bottom, color=cols, hatch=hatch, edgecolor=ctx.T["bg"], linewidth=0.5, label=lab)
            bottom += v
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.plot(xi, r["med"] * 1.04, marker=MARKER[r["family"]], color=ctx.color(r["family"]), markersize=6, linestyle="none")
            ax.text(xi, r["med"] * 1.12, f"{100 * r['comm_share']:.0f}% comm", ha="center", fontsize=6, color=ctx.T["muted"])
        ax.set_xticks(x); ax.set_xticklabels(g["label"], rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{NAMES[det]}, n = {fmt_n(n)}", fontsize=10, loc="left")
        ax.set_ylabel("Time (s)"); ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        from matplotlib.patches import Patch
        grey = ctx.T["muted"]
        proxies = [Patch(facecolor="none", edgecolor=grey, hatch=h, label=l) for h, l in (("//", "input"), ("", "compute"), ("xx", "gather"), (".", "other"))]
        proxies[1] = Patch(facecolor=grey, edgecolor=grey, label="compute")
        ax.legend(handles=proxies, fontsize=7, frameon=False, labelcolor=ctx.T["fg"], ncol=2, loc="upper right")
    return _finish(fig, ctx, name, title, "linear axes; 'comm' = input (scatter only) plus gather as a share of the run; 'other' = barriers and the rest", bars=False)


def f07(ctx):
    name, title = "F07.png", "F7 - OpenMP versus MPI versus hybrid at equal total workers"
    par = ctx.base[ctx.base["family"].isin(["OMP", "MPI", "HYB"])]
    dets = [d for d in detectors_present(ctx.base) if d in set(par[par["family"].isin(["MPI", "HYB"])]["detector"])]
    if not dets:
        return _placeholder(ctx, name, title, "no MPI or hybrid cells")
    fig, axes = _panels(len(dets), ctx, width=5.8)
    for ax, det in zip(axes, dets):
        n = _size_for(ctx, det, ["MPI", "HYB"])
        g = par[(par["detector"] == det) & (par["n"] == n)]
        workers = sorted(w for w in g["workers"].unique() if len(g[(g["workers"] == w)]["family"].unique()) > 1)
        pos, ticks = 0.0, []
        for w in workers:
            gg = g[g["workers"] == w].sort_values(["family", "mode"], key=lambda s: s.map({f: i for i, f in enumerate(FAMILY_ORDER)}) if s.name == "family" else s)
            start = pos
            for _, r in gg.iterrows():
                ax.bar(pos, r["med"], width=0.8, color=ctx.color(r["family"]), hatch="//" if r["label"].endswith("scatter") else "",
                       edgecolor=ctx.T["bg"], yerr=_err([r["med"]], [r["q1"]], [r["q3"]]), error_kw=dict(ecolor=ctx.T["fg"], capsize=2, lw=1))
                ax.plot(pos, r["q3"] * 1.04, marker=MARKER[r["family"]], color=ctx.color(r["family"]), markersize=6, linestyle="none")
                ax.text(pos, 0, r["label"], rotation=90, ha="center", va="bottom", fontsize=6, color=ctx.T["bg"])
                pos += 1
            ticks.append(((start + pos - 1) / 2, f"{int(w)} workers"))
            pos += 0.8
        ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=8)
        ax.set_title(f"{NAMES[det]}, n = {fmt_n(n)}", fontsize=10, loc="left"); ax.set_ylabel("Time (s)")
    return _finish(fig, ctx, name, title, "linear axes; hatched = scatter input mode; shorter is better")


# ------------------------------------------------------------------ F8: language gain versus parallel gain
def f08(ctx):
    name, title = "F08.png", "F8 - Language gain G_lang versus parallel speedup S_p"
    rows = []
    for det in detectors_present(ctx.base):
        g = ctx.base[ctx.base["detector"] == det]
        py = g[g["mode"] == "PY-SEQ"]
        if py.empty:
            continue
        n = int(py["n"].max())
        omp = g[(g["family"] == "OMP") & (g["n"] == n)]
        rows.append((det, n, float(py[py["n"] == n].iloc[0]["g_lang"]), omp.loc[omp["speedup"].idxmax()] if len(omp) else None))
    if not rows:
        return _placeholder(ctx, name, title, "no PY-SEQ cells, so G_lang cannot be computed")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(ctx.T["bg"]); _style_axes(ax, ctx.T)
    x = np.arange(len(rows)); wd = 0.36
    for i, (det, n, gl, best) in enumerate(rows):
        ax.bar(i - wd / 2, gl, wd, color=ctx.color("C-1"), hatch="//", edgecolor=ctx.T["bg"], label="G_lang = T_py / T1 (C versus Python)" if i == 0 else None)
        ax.text(i - wd / 2, gl, f"{gl:.1f}x", ha="center", va="bottom", fontsize=8, color=ctx.T["fg"])
        if best is not None:
            ax.bar(i + wd / 2, best["speedup"], wd, color=ctx.color("OMP"), edgecolor=ctx.T["bg"],
                   label="S_p = T1 / Tp (best OpenMP)" if i == 0 else None,
                   yerr=_err([best["speedup"]], [best["speedup_lo"]], [best["speedup_hi"]]), error_kw=dict(ecolor=ctx.T["fg"], capsize=2, lw=1))
            ax.text(i + wd / 2, best["speedup_hi"], f"{best['speedup']:.1f}x ({best['label']})", ha="center", va="bottom", fontsize=7, color=ctx.T["fg"])
    ax.set_xticks(x); ax.set_xticklabels([f"{NAMES[d]}\nn = {fmt_n(n)}" for d, n, *_ in rows], fontsize=8)
    ax.set_ylabel("Gain (x)"); ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.legend(fontsize=8, frameon=False, labelcolor=ctx.T["fg"], loc="upper left")
    return _finish(fig, ctx, name, title, "linear y axis; the two gains answer different questions and are never multiplied or added")


# ------------------------------------------------------------------ F9: schedule(static) versus schedule(dynamic)
def schedule_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    need = {"dataset", "n", "threads", "schedule", "total_s"}
    if not need <= set(d.columns):
        raise ValueError(f"{path.name} must have columns {sorted(need)} (+ repeat)")
    n = d["n"].max()
    d = d[d["n"] == n]
    g = d.groupby(["threads", "schedule"])["total_s"]
    out = g.agg(med="median", q1=lambda s: np.percentile(s, 25), q3=lambda s: np.percentile(s, 75), runs="size").reset_index()
    out["n"] = n
    return out


def f09(ctx):
    name, title = "F09.png", "F9 - schedule(static) versus schedule(dynamic, 4096) on MAD"
    try:
        t = schedule_table(ctx.data_dir / SCHEDULE_CSV)
    except ValueError as e:
        return _placeholder(ctx, name, title, str(e))
    if t.empty:
        return _placeholder(ctx, name, title, f"experiments/out/{SCHEDULE_CSV} is missing. It is the output of the D-05.6 experiment, with columns "
                            "dataset, n, threads, schedule (static or dynamic), repeat, total_s. It is not produced by the D-08 matrix.")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.patch.set_facecolor(ctx.T["bg"]); _style_axes(ax, ctx.T)
    for sched, ls, mk in (("static", "-", "^"), ("dynamic", "--", "v")):
        g = t[t["schedule"] == sched].sort_values("threads")
        ax.errorbar(g["threads"], g["med"], yerr=_err(g["med"], g["q1"], g["q3"]), color=ctx.color("OMP", 1.0 if sched == "static" else 0.55),
                    marker=mk, linestyle=ls, capsize=2, linewidth=1.5, label=f"schedule({sched})")
    ax.set_xlabel("OpenMP threads p"); ax.set_ylabel("Time (s)"); ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(t["threads"].unique())); ax.legend(fontsize=8, frameon=False, labelcolor=ctx.T["fg"])
    _phys_line(ax, ctx)
    ctx.runs = (int(t["runs"].min()), int(t["runs"].max()))
    return _finish(fig, ctx, name, f"{title} (n = {fmt_n(t['n'].iloc[0])})", "linear axes; data from the D-05.6 experiment")


# ------------------------------------------------------------------ F10: F1 per mode
def f10(ctx):
    name, title = "F10.png", "F10 - F1 per mode"
    g = ctx.base[ctx.base["f1"].notna()]
    if g.empty:
        return _placeholder(ctx, name, title, "no F1 values (labels missing?)")
    order = (g.drop_duplicates("label").assign(_f=lambda d: d["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
             .sort_values(["_f", "workers", "label"])["label"].tolist())
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor(ctx.T["bg"]); _style_axes(ax, ctx.T)
    styles = ["-", "--", "-.", ":"]
    for i, det in enumerate(detectors_present(g)):
        gd = g[g["detector"] == det]
        gd = gd[gd["n"] == gd.groupby("label")["n"].transform("max")].set_index("label")      # largest size at which each mode was measured
        gd = gd.loc[sorted(gd.index, key=order.index)]
        xs = [order.index(l) for l in gd.index]
        ys = gd["f1"].to_numpy()
        ax.plot(xs, ys, color=ctx.T["muted"], linestyle=styles[i % 4], linewidth=1, label=NAMES[det], zorder=1)
        for x, (lab, r) in zip(xs, gd.iterrows()):
            same = r["identical_all"] is True
            ax.plot(x, r["f1"], marker=MARKER[r["family"]], markersize=7, color=ctx.color(r["family"]), markerfacecolor=ctx.color(r["family"]) if same else ctx.T["bg"],
                    markeredgewidth=1.5, linestyle="none", zorder=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0); ax.set_ylabel("F1 (point-wise)")
    ax.legend(fontsize=7, frameon=False, labelcolor=ctx.T["fg"], loc="lower left", title="filled marker = identical to C-1; hollow = not identical or not checked",
              title_fontsize=7)
    ax.get_legend().get_title().set_color(ctx.T["muted"])
    return _finish(fig, ctx, name, title, "linear y axis from 0 to 1; each point is at the largest size where that mode was measured; a flat line is by design for the deterministic kernels (H2), not a finding about the data", bars=False)


FIGURES = {1: f01, 2: f02, 3: f03, 4: f04, 5: f05, 6: f06, 7: f07, 8: f08, 9: f09, 10: f10}


def make_all(ct, machine, s0, outdir, theme="light", data_dir="experiments/out") -> Dict[int, str]:
    _setup_font()
    ctx = Ctx(ct, machine, s0, theme, outdir, data_dir)
    status: Dict[int, str] = {}
    for i, fn in FIGURES.items():
        if ct.empty:
            status[i] = _placeholder(ctx, f"F{i:02d}.png", f"F{i}", "no results in results.csv yet")
            continue
        try:
            status[i] = fn(ctx)
        except Exception as e:                                       # a broken figure must not stop the others, and must be visible
            plt.close("all")
            status[i] = _placeholder(ctx, f"F{i:02d}.png", f"F{i}", f"figure failed: {type(e).__name__}: {e}")
    return status
