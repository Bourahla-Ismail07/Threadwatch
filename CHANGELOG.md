# Changelog

## D-05
- D-05: OpenMP kernels with per-thread counters, output identical for 1..max threads, first scaling table
- Thread requests are capped at the logical core count; `tw_mad_ex` adds a static/dynamic schedule choice.

## D-04
- D-04: C SDK kernels (threads=1) with ctypes wrapper, equivalence < 1e-5 vs NumPy, T1 and G_lang recorded
- MAD quickselect uses a three-way partition, so flat or repeated-value windows stay linear.

## D-03
- D-03: Python sequential detectors (z-score, MAD, K-Means, Isolation Forest), metrics, first F1 and T_py table

## D-02
- D-02: literature review, NAB/SMD conversion, labelled synthetic generator (1M to 20M), data sheet

## D-01
- D-01: project sheet, repo scaffold, hello DLL and hello MPI checks passing
- Added `experiments/cpu_info.py` and `tests/python/test_cores.py` to record logical vs physical cores.

## D-06
- D-06: MPI file and scatter drivers with overlap rule, hybrid runs, output identical to C-1
