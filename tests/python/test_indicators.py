"""D-09 - indicator formulas, the cell table, the verdict rules, the figures and the one-command regeneration.

Synthetic results are written through the real D-08 row writer (threadwatch.bench.rows), so these tests also pin down
the D-08 -> D-09 file contract (status values, column names, mode names).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from threadwatch.bench.matrix import Cell
from threadwatch.bench.rows import ResultsFile, make_row

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from analysis import indicators as ind  # noqa: E402
from analysis import interpret, make_figures, tables  # noqa: E402
import run_analysis  # noqa: E402


# ------------------------------------------------------------------ synthetic results with known physics
SIZES = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
RATE = {"zscore": 0.43, "mad": 6.2, "kmeans": 0.45, "iforest": 5.0}
PYRATE = {"zscore": 2.1, "mad": 9.2, "kmeans": 2.8, "iforest": 5.7}
F1 = {"zscore": 0.62, "mad": 0.71, "kmeans": 0.55, "iforest": 0.48}
DETS = ("zscore", "mad", "kmeans", "iforest")


def build(path, serial=0.03, hyper_penalty=0.45, phys=6, mpi_comm=0.25, mpi_clear_win=False, nondeterministic=False,
          repeats=5, noise=0.01, sizes=SIZES, seed=0, detectors=DETS, small_penalty=True, with_mpi=True, with_py=True, threads=(2, 4, 8, 12)):
    rng = np.random.default_rng(seed)
    rf = ResultsFile(path)

    def emit(cell, t, ident=True, inp=0.0, comm=0.0):
        rf.append(make_row(cell, 0, "warmup", total_s=t * 1.3, identical=True, f1=F1[cell.detector]))
        for r in range(1, repeats + 1):
            tt = t * (1 + rng.normal(0, noise))
            ph = {}
            if cell.mode in ("MPI", "HYB", "MPI-scatter"):
                i_, c_ = inp * tt, comm * tt
                ph = dict(load_s=i_, score_s=tt - i_ - c_ - 0.02 * tt, comm_s=c_ + (i_ if cell.mode == "MPI-scatter" else 0.0))
            rf.append(make_row(cell, r, "ok", total_s=tt, f1=F1[cell.detector], precision=0.6, recall=0.7, event_recall=0.9,
                               identical=ident, dll_ms=tt * 1000, **ph))

    for det in detectors:
        for n in sizes:
            t1 = RATE[det] * n / 1e6
            c = lambda mode, q=1, p=1: Cell(f"syn_{n}", n, det, mode, q, p, 256, 1)
            emit(c("C-1"), t1)
            if with_py and n <= 2_000_000 and not (det == "mad" and n > 1_000_000):
                emit(c("PY-SEQ"), PYRATE[det] * n / 1e6)

            def tp(workers):
                par = serial * t1 + (1 - serial) * t1 / min(workers, phys) * (1 + (hyper_penalty * (workers - phys) / phys if workers > phys else 0))
                return par + (0.125 * t1 if (small_penalty and n <= 1_000_000 and workers >= 8) else 0)
            for p in threads:
                emit(c("OMP", 1, p), tp(p), ident=not (nondeterministic and det == "zscore" and p == 8))
            if det == "iforest" or not with_mpi:
                continue
            for q in (2, 4):
                t = tp(q) * (0.6 if mpi_clear_win else 1.0) + mpi_comm * tp(q) + (0.4 * t1 if (small_penalty and n <= 1_000_000) else 0)
                emit(c("MPI", q, 1), t, comm=mpi_comm * tp(q) / t)
                if n in (2_000_000, 10_000_000):
                    emit(c("MPI-scatter", q, 1), t * 1.15, comm=0.1, inp=0.12)
            for q, p in ((2, 2), (2, 4)):
                emit(c("HYB", q, p), tp(q * p) * 1.08 + mpi_comm * tp(q * p) * 0.8, comm=0.1)
    return path


MACHINE = {"processor": "Test CPU", "logical_cores": 12, "physical_cores": 6, "ram_gb": 16, "os": "TestOS", "power_plan": "High performance",
           "mpi": {}, "dll": {}, "packages": {}}


@pytest.fixture
def synth(tmp_path):
    p = tmp_path / "results.csv"
    build(p)
    return p


def cell_table_of(path):
    return ind.cell_table(ind.load_results(path))


def verdicts_of(path, machine=MACHINE):
    ct = cell_table_of(path)
    return interpret.verdicts(ct, machine, 1), ct


# ------------------------------------------------------------------ D-09.1 formulas
def test_spec_examples_speedup_efficiency_karp_flatt():
    s = ind.speedup(8.0, 2.0)
    assert s == 4.0 and ind.efficiency(s, 4) == 1.0
    assert ind.karp_flatt(s, 4) == pytest.approx(0.0)                                  # ideal scaling: no serial part


def test_karp_flatt_matches_the_formula_and_is_undefined_for_one_worker():
    s, p = 3.0, 4
    assert ind.karp_flatt(s, p) == pytest.approx((1 / 3 - 1 / 4) / (1 - 1 / 4))
    assert np.isnan(ind.karp_flatt(1.0, 1))
    out = ind.karp_flatt(np.array([1.0, 2.0, 3.5]), np.array([1, 2, 4]))
    assert np.isnan(out[0]) and np.isfinite(out[1:]).all()


def test_karp_flatt_recovers_a_fixed_serial_fraction():
    f = 0.1
    p = np.array([2, 4, 8, 16])
    s = 1 / (f + (1 - f) / p)                                                          # Amdahl: e equals f at every p
    assert ind.karp_flatt(s, p) == pytest.approx(f)
    assert ind.amdahl_serial_fraction(p, s) == pytest.approx(f)


def test_throughput_and_language_gain_are_separate_quantities():
    assert ind.throughput(10_000_000, 2.0) == 5_000_000
    assert ind.g_lang(9.0, 3.0) == 3.0 and ind.speedup(3.0, 1.0) == 3.0                # G = T_py / T1; S = T1 / Tp


def test_iqr_comparison_never_calls_overlap_a_difference():
    a, b = {"q1": 1.0, "q3": 1.2}, {"q1": 2.0, "q3": 2.4}
    assert ind.compare(a, b) == "faster" and ind.compare(b, a) == "slower"
    assert ind.compare({"q1": 1.0, "q3": 2.1}, b) == "no measurable difference"        # IQRs overlap
    assert ind.compare({"q1": 1.0, "q3": 2.0}, {"q1": 2.0, "q3": 3.0}) == "no measurable difference"   # touching is not separated


# ------------------------------------------------------------------ cell table
def test_only_ok_rows_count_and_labels_follow_the_d08_names(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c = Cell("syn_1M", 1_000_000, "mad", "OMP", 1, 4, 256, 1)
    rf.append(make_row(c, 0, "warmup", total_s=99.0))
    for i, t in enumerate([2.0, 2.1, 1.9], 1):
        rf.append(make_row(c, i, "ok", total_s=t, identical=True, f1=0.5))
    rf.append(make_row(c, 4, "timeout", note="slow"))
    rf.append(make_row(c, 5, "superseded", total_s=50.0))
    rf.append(make_row(c, 6, "crash"))
    df = ind.load_results(rf.path)
    assert len(df) == 3 and set(df["label"]) == {"OMP-4"} and df["identical"].all()
    assert ind.label_of("MPI-scatter", 4, 1) == "MPI-4-scatter" and ind.label_of("HYB", 2, 4) == "HYB-2\u00d74"
    assert ind.load_results(tmp_path / "missing.csv").empty


def test_cell_table_uses_c1_as_t1_never_python(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    mk = lambda mode, q, p: Cell("syn_1M", 1_000_000, "mad", mode, q, p, 256, 1)
    for mode, q, p, t in (("PY-SEQ", 1, 1, 9.0), ("C-1", 1, 1, 8.0), ("OMP", 1, 4, 2.0), ("MPI", 2, 1, 4.0)):
        for i in range(1, 6):
            rf.append(make_row(mk(mode, q, p), i, "ok", total_s=t, identical=True, f1=0.5))
    ct = ind.cell_table(ind.load_results(rf.path)).set_index("label")
    assert ct.loc["OMP-4", "t1"] == 8.0 and ct.loc["OMP-4", "speedup"] == 4.0 and ct.loc["OMP-4", "efficiency"] == 1.0
    assert ct.loc["OMP-4", "karp_flatt"] == pytest.approx(0.0)
    assert ct.loc["MPI-2", "speedup"] == 2.0 and ct.loc["MPI-2", "workers"] == 2 and ct.loc["MPI-2", "efficiency"] == 1.0
    assert ct.loc["PY-SEQ", "g_lang"] == pytest.approx(9.0 / 8.0) and np.isnan(ct.loc["PY-SEQ", "speedup"])   # separate from S_p
    assert np.isnan(ct.loc["C-1", "karp_flatt"]) and ct.loc["C-1", "speedup"] == 1.0
    assert ct.loc["OMP-4", "throughput_sps"] == pytest.approx(1_000_000 / 2.0)


def test_one_slow_outlier_does_not_move_the_headline_number(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c = Cell("syn_1M", 1_000_000, "mad", "C-1", 1, 1, 256, 1)
    for i, t in enumerate([2.0, 2.0, 2.0, 2.0, 30.0], 1):
        rf.append(make_row(c, i, "ok", total_s=t))
    row = ind.cell_table(ind.load_results(rf.path)).iloc[0]
    assert row["med"] == 2.0 and row["tmax"] == 30.0 and row["q3"] == 2.0


def test_missing_c1_gives_nan_not_a_crash(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c = Cell("syn_1M", 1_000_000, "mad", "OMP", 1, 4, 256, 1)
    for i in range(1, 4):
        rf.append(make_row(c, i, "ok", total_s=2.0))
    ct = ind.cell_table(ind.load_results(rf.path))
    assert np.isnan(ct.iloc[0]["speedup"]) and np.isnan(ct.iloc[0]["efficiency"])


def test_phase_breakdown_separates_input_compute_gather_for_file_and_scatter(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    f = Cell("syn_1M", 1_000_000, "mad", "MPI", 2, 1, 256, 1)
    s = Cell("syn_1M", 1_000_000, "mad", "MPI-scatter", 2, 1, 256, 1)
    for i in range(1, 4):
        rf.append(make_row(f, i, "ok", total_s=10.0, load_s=0.1, score_s=8.0, comm_s=1.0))          # file: comm_s = gather only
        rf.append(make_row(s, i, "ok", total_s=10.0, load_s=1.0, score_s=8.0, comm_s=1.5))          # scatter: comm_s = input + gather
    ct = ind.cell_table(ind.load_results(rf.path)).set_index("label")
    assert ct.loc["MPI-2", "phase_gather"] == pytest.approx(1.0) and ct.loc["MPI-2", "comm_share"] == pytest.approx(0.1)
    assert ct.loc["MPI-2-scatter", "phase_gather"] == pytest.approx(0.5) and ct.loc["MPI-2-scatter", "comm_share"] == pytest.approx(0.15)
    assert ct.loc["MPI-2-scatter", "phase_other"] == pytest.approx(0.5)


# ------------------------------------------------------------------ tables
def test_crossover_is_bracketed_and_interpolated(synth):
    ct = cell_table_of(synth)
    cx = tables.crossover(ct, 1)
    r = cx[(cx.detector == "mad") & (cx.label == "MPI-2")].iloc[0]
    assert r["status"] == "crossover located" and r["slower_below"] == "1M" and r["faster_from"] == "2M"
    assert 1_000_000 < r["crossover_estimate_n"] < 2_000_000
    omp = cx[(cx.detector == "mad") & (cx.label == "OMP-4")].iloc[0]
    assert omp["status"].startswith("faster at every")


def test_not_faster_table_lists_small_size_mpi_cells(synth):
    nf = tables.not_faster(cell_table_of(synth), 1)
    assert len(nf) and (nf["n"] == 1_000_000).all() and set(nf["label"]) == {"MPI-2"} | set(nf["label"]) and (nf["speedup"] <= 1.0).any()
    assert set(nf["vs_C1"]) <= {"slower", "no measurable difference"}


def test_equal_workers_compares_with_the_omp_cell_of_the_same_worker_count(synth):
    ew = tables.equal_workers(cell_table_of(synth), 1)
    g = ew[(ew.detector == "mad") & (ew.n == 20_000_000)]
    assert set(g[g.workers == 4]["label"]) == {"OMP-4", "MPI-4", "HYB-2\u00d72"}
    assert g[g.label == "MPI-4"].iloc[0]["vs_OMP_same_workers"] == "slower"
    assert g[g.label == "OMP-4"].iloc[0]["vs_OMP_same_workers"] == ""


def test_quality_tables(synth):
    ct = cell_table_of(synth)
    q = tables.quality_vs_mode(ct, 1)
    assert set(q["identical_all"]) <= {"yes", "NO", ""}
    sp = tables.quality_spread(ct, 1)
    assert (sp[sp.deterministic_kernel]["f1_range"] == 0).all()
    piv = tables.quality_pivot(q)
    assert list(piv["detector"]) and "OMP-4" in piv.columns


def test_g_lang_table_never_mixes_the_gains(synth):
    gl = tables.g_lang_table(cell_table_of(synth), 1)
    r = gl[gl.detector == "mad"].iloc[0]
    assert r["G_lang"] == pytest.approx(r["T_py_s"] / r["T1_s"]) and r["S_p_best"] > 1 and r["n"] == 1_000_000


# ------------------------------------------------------------------ verdicts
def test_h1_supported_when_efficiency_holds_up_to_the_physical_cores(synth):
    v, _ = verdicts_of(synth)
    h = v["H1"]
    assert h["result"] == "supported"
    assert any("p=4" in n and "E=9" in n for n in h["numbers"])                       # ~90% at 4 threads
    assert "beyond the 6 physical cores" in h["reading"] and "was not measured" in h["limit"]
    assert any("NOTE" in n and "physical core count (6)" in n for n in h["numbers"])


def test_h1_rejected_when_efficiency_is_clearly_below_70_inside_the_physical_cores(tmp_path):
    build(tmp_path / "r.csv", serial=0.35)
    v, _ = verdicts_of(tmp_path / "r.csv")
    assert v["H1"]["result"] == "rejected" and "inside the 6 physical cores" in v["H1"]["reading"]


def test_h1_inconclusive_when_the_spread_reaches_the_limit(tmp_path):
    """Median efficiency just under 70%, but the fast end of the IQR above it: not enough to reject."""
    rf = ResultsFile(tmp_path / "r.csv")
    n = 5_000_000
    for det in ("zscore", "mad"):
        mk = lambda mode, p: Cell(f"syn_{n}", n, det, mode, 1, p, 256, 1)
        for i in range(1, 6):
            rf.append(make_row(mk("C-1", 1), i, "ok", total_s=8.0, identical=True, f1=0.5))
            rf.append(make_row(mk("OMP", 2), i, "ok", total_s=4.0, identical=True, f1=0.5))                    # E = 100%
            rf.append(make_row(mk("OMP", 4), i, "ok", total_s=[2.5, 2.7, 2.9, 3.1, 3.3][i - 1], identical=True, f1=0.5))   # E = 69%
    ct = ind.cell_table(ind.load_results(rf.path))
    p4 = ct[ct.p_threads == 4]
    assert (p4.efficiency < 0.70).all() and (p4.efficiency_hi >= 0.70).all()            # the situation the rule is about
    assert interpret.hypothesis_1(ct, MACHINE, 1)["result"] == "inconclusive"


def test_h1_needs_the_physical_core_count_and_omp_data(synth, tmp_path):
    ct = cell_table_of(synth)
    assert interpret.hypothesis_1(ct, {}, 1)["result"] == "inconclusive"
    no_omp = ct[ct.family != "OMP"]
    assert interpret.hypothesis_1(no_omp, MACHINE, 1)["result"] == "inconclusive"


def test_h2_supported_and_rejected(tmp_path, synth):
    v, _ = verdicts_of(synth)
    assert v["H2"]["result"] == "supported" and "by design" in v["H2"]["reading"] or "design property" in v["H2"]["reading"]
    build(tmp_path / "bad.csv", nondeterministic=True)
    v2, _ = verdicts_of(tmp_path / "bad.csv")
    assert v2["H2"]["result"] == "rejected" and any("NOT identical" in n and "z-score" in n and "OMP-8" in n for n in v2["H2"]["numbers"])


def test_h2_inconclusive_when_identity_was_not_checked(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    c1, c4 = (Cell("syn_1M", 1_000_000, "mad", m, 1, p, 256, 1) for m, p in (("C-1", 1), ("OMP", 4)))
    for i in range(1, 6):
        rf.append(make_row(c1, i, "ok", total_s=8.0, f1=0.5))
        rf.append(make_row(c4, i, "ok", total_s=2.0, f1=0.5))                         # identical column left empty
    ct = ind.cell_table(ind.load_results(rf.path))
    assert interpret.hypothesis_2(ct, MACHINE, 1)["result"] == "inconclusive"


def test_h3_supported_when_a_crossover_is_located(synth):
    v, _ = verdicts_of(synth)
    h = v["H3"]
    assert h["result"] == "supported" and any("interpolated crossover about" in n for n in h["numbers"])
    assert "slower than one thread" in h["reading"]


def test_h3_inconclusive_when_the_measured_sizes_are_too_large_to_see_the_overhead(tmp_path):
    build(tmp_path / "r.csv", sizes=[5_000_000, 10_000_000, 20_000_000], detectors=("mad", "zscore"))
    v, _ = verdicts_of(tmp_path / "r.csv")
    assert v["H3"]["result"] == "inconclusive" and "below the measured range" in v["H3"]["reading"]
    assert "100k" in v["H3"]["reading"]                                                # tells the user what to add


def test_h3_rejected_only_with_small_sizes_and_no_slow_region(tmp_path):
    build(tmp_path / "r.csv", sizes=[50_000, 5_000_000, 10_000_000], detectors=("mad",), small_penalty=False, mpi_comm=0.02)
    v, ct = verdicts_of(tmp_path / "r.csv")
    assert v["H3"]["result"] == "rejected", v["H3"]["numbers"]


def test_h3_inconclusive_when_parallel_is_slower_but_no_larger_size_wins(tmp_path):
    rf = ResultsFile(tmp_path / "r.csv")
    for n, t1, tp in ((1_000_000, 1.0, 1.4), (2_000_000, 2.0, 2.6)):
        for mode, p, t in (("C-1", 1, t1), ("OMP", 4, tp)):
            for i in range(1, 6):
                rf.append(make_row(Cell(f"syn_{n}", n, "mad", mode, 1, p, 256, 1), i, "ok", total_s=t * (1 + 0.01 * i), identical=True, f1=0.5))
    ct = ind.cell_table(ind.load_results(rf.path))
    h, _ = interpret.hypothesis_3(ct, MACHINE, 1)
    assert h["result"] == "inconclusive" and "crossover lies above the measured range" in h["reading"]


def test_h4_supported_when_mpi_never_beats_openmp_and_rejected_when_it_usually_does(tmp_path, synth):
    v, _ = verdicts_of(synth)
    h = v["H4"]
    assert h["result"] == "supported" and "MPI clearly faster in 0" in " ".join(h["numbers"])
    assert any(n.startswith("fastest parallel configuration, MAD at 20M") for n in h["numbers"])
    build(tmp_path / "w.csv", mpi_clear_win=True, mpi_comm=0.0, small_penalty=False)
    v2, _ = verdicts_of(tmp_path / "w.csv")
    assert v2["H4"]["result"] == "rejected"


def test_h4_overlapping_iqrs_are_not_a_win_for_mpi(tmp_path):
    build(tmp_path / "r.csv", mpi_comm=0.0, noise=0.2, seed=5, small_penalty=False, detectors=("mad",))
    ct = cell_table_of(tmp_path / "r.csv")
    ew = tables.equal_workers(ct, 1)
    assert (ew["vs_OMP_same_workers"] == "no measurable difference").any()
    assert interpret.hypothesis_4(ct, MACHINE, 1)["result"] in ("supported", "inconclusive")


def test_every_verdict_is_one_of_the_three_allowed_values(synth):
    v, _ = verdicts_of(synth)
    assert set(v) == {"H1", "H2", "H3", "H4"} and all(d["result"] in ("supported", "rejected", "inconclusive") for d in v.values())
    assert all(d["numbers"] and d["reading"] and d["limit"] and d["figs"] for d in v.values())


# ------------------------------------------------------------------ figures
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_all_ten_figures_regenerate_from_the_csv(synth, tmp_path, theme):
    ct = cell_table_of(synth)
    out = tmp_path / "figs"
    status = make_figures.make_all(ct, MACHINE, 1, out, theme, str(tmp_path))
    assert sorted(status) == list(range(1, 11))
    for i in range(1, 11):
        f = out / f"F{i:02d}.png"
        assert f.exists() and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and f.stat().st_size > 5_000
    assert [i for i, s in status.items() if s != "ok"] == [9]                          # F9 needs the D-05.6 schedule data
    assert "schedule_mad.csv" in status[9] and "not produced by the D-08 matrix" in status[9]


def test_f9_is_drawn_when_the_schedule_data_exists_and_rejects_a_bad_file(synth, tmp_path):
    ct = cell_table_of(synth)
    rows = [{"dataset": "syn_5M", "n": 5_000_000, "threads": p, "schedule": s, "repeat": r, "total_s": 30 / p * (1.0 if s == "static" else 0.95) * (1 + 0.01 * r)}
            for p in (1, 2, 4, 8) for s in ("static", "dynamic") for r in range(5)]
    pd.DataFrame(rows).to_csv(tmp_path / "schedule_mad.csv", index=False)
    assert make_figures.make_all(ct, MACHINE, 1, tmp_path / "f", "light", str(tmp_path))[9] == "ok"
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "schedule_mad.csv", index=False)
    st = make_figures.make_all(ct, MACHINE, 1, tmp_path / "g", "light", str(tmp_path))[9]
    assert st.startswith("no data") and "columns" in st


def test_figures_without_mpi_or_python_cells_degrade_to_placeholders_not_crashes(tmp_path):
    build(tmp_path / "r.csv", with_mpi=False, with_py=False, detectors=("mad",))
    ct = cell_table_of(tmp_path / "r.csv")
    st = make_figures.make_all(ct, MACHINE, 1, tmp_path / "f", "light", str(tmp_path))
    assert st[1] == st[2] == st[3] == st[4] == st[5] == st[10] == "ok"
    assert st[6].startswith("no data") and st[7].startswith("no data") and st[8].startswith("no data")
    assert all((tmp_path / "f" / f"F{i:02d}.png").exists() for i in range(1, 11))


def test_captions_state_machine_window_stride_runs_and_scale(synth):
    ctx = make_figures.Ctx(cell_table_of(synth), MACHINE, 1)
    cap = ctx.caption("log x and log y axes")
    assert "Test CPU" in cap and "12 logical / 6 physical" in cap and "w=256" in cap and "s=1" in cap
    assert "5 timed runs" in cap and "interquartile range" in cap and "log x and log y" in cap


def test_brand_markers_and_colours_per_mode():
    assert make_figures.MARKER == {"PY-SEQ": "s", "C-1": "o", "OMP": "^", "MPI": "D", "HYB": "h"}
    light = make_figures.THEMES["light"]
    assert (light["omp"], light["mpi"], light["hybrid"], light["alert"]) == ("#0F766E", "#1D4ED8", "#6D28D9", "#B91C1C")
    dark = make_figures.THEMES["dark"]
    assert (dark["seq"], dark["omp"], dark["mpi"], dark["hybrid"]) == ("#9AA6B2", "#2DD4BF", "#60A5FA", "#A78BFA")


# ------------------------------------------------------------------ one command, results.md
def test_one_command_writes_tables_figures_and_results_md(synth, tmp_path):
    root = tmp_path / "proj"
    out = root / "experiments/out"
    out.mkdir(parents=True)
    (out / "results.csv").write_text(synth.read_text())
    (out / "machine.json").write_text(json.dumps(MACHINE))
    cmd = [sys.executable, str(ROOT / "experiments/run_analysis.py"), "--root", str(root)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-600:]
    assert "figures without data: F09" in p.stdout
    md = (root / "docs/results.md").read_text()
    for h in ("H1", "H2", "H3", "H4"):
        assert f"## {h} - " in md
    for part in ("**Result:**", "**Evidence:**", "**Reading:**", "**Limit:**", "## Where parallel is not faster", "## Quality versus mode",
                 "## Language gain and parallel gain", "## How the verdicts are decided", "../experiments/out/figures/F01.png"):
        assert part in md
    assert (out / "figures/F10.png").exists() and (out / "tables/indicators.csv").exists() and (out / "tables/not_faster.csv").exists()
    first = md
    subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    assert (root / "docs/results.md").read_text().replace("\r", "") == first.replace("\r", "")     # same input, same document


def test_no_results_gives_placeholders_and_an_honest_document(tmp_path):
    root = tmp_path / "proj"
    (root / "experiments/out").mkdir(parents=True)
    info = run_analysis.run(root / "experiments/out/results.csv", root / "experiments/out/machine.json", root / "experiments/out", root / "docs/results.md")
    assert info["rows"] == 0 and (root / "experiments/out/figures/F01.png").exists()
    assert "No results yet" in (root / "docs/results.md").read_text()


def test_results_md_shows_a_case_where_parallel_is_not_faster_and_explains_it(synth, tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "machine.json").write_text(json.dumps(MACHINE))
    run_analysis.run(synth, out / "machine.json", out, tmp_path / "results.md")
    md = (tmp_path / "results.md").read_text()
    seg = md.split("## Where parallel is not faster")[1].split("## Quality")[0]
    assert "not clearly faster than C-1" in seg and "likely cause" in seg and "MPI-2" in seg and "| speedup |" in seg
