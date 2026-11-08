# Changelog

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
