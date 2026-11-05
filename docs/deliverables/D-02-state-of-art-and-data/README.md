# D-02 - State of the art and datasets

| | |
|---|---|
| **Teacher's item** | État de l'art et description des données |
| **Depends on** | D-01 |
| **Unblocks** | D-03 |

## What this part is responsible for

A short, honest literature review that positions the project, and the data the whole project runs on: two real datasets for detection quality, and a labelled synthetic generator for scaling tests. Everything is stored in one simple binary format so every later stage reads it the same way.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| Python + NumPy | synthetic generator, binary writing, memory-mapped loading |
| pandas | read NAB and SMD files, summary statistics |
| matplotlib | sample plots for the data sheet |
| pytest | generator tests (determinism, label consistency) |

## Tasks

| Task | What |
|---|---|
| D-02.1 | Literature review: 4 to 6 sources actually read (list below) and a one-page synthesis |
| D-02.2 | Download and convert NAB and SMD series to `float32` + labels |
| D-02.3 | Synthetic generator with injected labelled anomalies |
| D-02.4 | Loader (`load_series(name)` returns a memory-mapped array and labels) |
| D-02.5 | Data sheet: source, size, sampling, licence, anomaly rate, one plot per dataset |

## Literature review: what to cover

1. Time-series anomaly detection (statistical and machine-learning families).
2. Isolation Forest (the original paper) and robust statistics (median, MAD).
3. Parallel performance evaluation with OpenMP and MPI (a textbook or survey chapter).
4. Amdahl's law and the Karp-Flatt metric.
5. The NAB benchmark paper and the SMD (OmniAnomaly) paper.
6. One recent paper on parallel or accelerated anomaly detection.

Cite only what you read. Positioning sentence to write at the end: what existing work already shows, and what a controlled benchmark of the same kernel under five execution modes adds.

## Binary format

- `data/series/<name>.f32`: little-endian `float32`, one value per sample.
- `data/series/<name>.labels.u8`: one byte per sample, `1` = anomaly.
- `data/series/<name>.json`: `{"name", "source", "n", "sampling", "seed", "generator_version"}`.

## Pseudo code: synthetic generator

```python
def generate(n, seed=7, period_day=1440, rate=0.001):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 50 + 0.00001 * t                                   # slow trend
    base += 10 * np.sin(2 * np.pi * t / period_day)           # daily cycle
    base += 4 * np.sin(2 * np.pi * t / (7 * period_day))      # weekly cycle
    noise = ar1(n, phi=0.7, sigma=1.0, rng=rng)               # AR(1), vectorised with lfilter
    x = base + noise
    y = np.zeros(n, dtype=np.uint8)

    n_events = int(n * rate / 50) + 1                         # average event length ~ 50
    starts = rng.integers(1000, n - 1000, size=n_events)
    for s in starts:
        kind = rng.choice(["spike", "level", "variance"], p=[0.5, 0.3, 0.2])
        if kind == "spike":      L = 1;                    x[s] += rng.choice([-1, 1]) * rng.uniform(6, 12)
        elif kind == "level":    L = rng.integers(30, 301); x[s:s+L] += rng.choice([-1, 1]) * rng.uniform(4, 8)
        else:                    L = rng.integers(30, 301); x[s:s+L] += rng.normal(0, 4, size=L)
        y[s:s+L] = 1
    return x.astype(np.float32), y
```

Generate in blocks for the largest sizes (for example 5M samples per block with a continuing phase) to keep peak memory low, and write straight to the `.f32` file.

Sizes to create: `syn_1M`, `syn_2M`, `syn_5M`, `syn_10M`, `syn_20M` (and `syn_50M` if memory allows).

## Loader

```python
def load_series(name, root="data/series"):
    x = np.memmap(f"{root}/{name}.f32", dtype=np.float32, mode="r")
    y = np.memmap(f"{root}/{name}.labels.u8", dtype=np.uint8, mode="r")
    meta = json.load(open(f"{root}/{name}.json"))
    assert len(x) == len(y) == meta["n"]
    return x, y, meta
```

Pass `np.ascontiguousarray(x)` or the memmap itself to the DLL; a memmap is already contiguous.

## Files to create

- `python/threadwatch/data/{__init__.py, synthetic.py, nab.py, smd.py, loader.py}`
- `tests/python/test_data.py`
- `docs/data.md` (the data sheet), `report/state-of-the-art.md` (exported to PDF for sending)

## Tests and acceptance

- The same seed gives the same series (hash of the `.f32` file is identical).
- `len(labels) == len(series)` for every dataset; anomaly rate within 20% of the requested rate.
- All five synthetic sizes exist; NAB and SMD selections are converted and load.
- The data sheet lists source, size, sampling interval, licence, anomaly rate and one plot per dataset; sizes and licences are taken from the downloaded files, not from memory.
- The literature review cites only sources that were read.

## Pitfalls

- NAB labels are given as anomaly **windows** in a separate JSON file, not per point; convert them to per-sample labels and document how.
- SMD is multivariate (38 metrics per machine); pick metrics explicitly and write down which, rather than averaging.
- NaN or missing values in real data must be handled (forward fill) and counted in the data sheet.
- `np.memmap` on Windows locks the file while open; close it before regenerating.
- Do not tile a real series to reach 50M samples and report F1 on it; scaling runs are for timing only.

## Changelog line

`D-02: literature review, NAB/SMD conversion, labelled synthetic generator (1M to 20M), data sheet`
