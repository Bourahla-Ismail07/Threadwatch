# D-09 - Analysis, graphs and interpretation

| | |
|---|---|
| **Teacher's item** | Graphiques et interprétation |
| **Depends on** | D-08 |
| **Unblocks** | D-10 |

## What this part is responsible for

Turning `results.csv` into the HPC indicators, the figures, and a written reading of each hypothesis (H1 to H4), including the cases where parallelism does not help.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| pandas | aggregate runs, compute speedup, efficiency, throughput, Karp-Flatt |
| matplotlib | all figures (script-generated, no manual editing) |
| NumPy | curve fits (Amdahl) if used |
| pytest | tests for the indicator formulas |

## Tasks

| Task | What |
|---|---|
| D-09.1 | Indicator functions (speedup, efficiency, throughput, Karp-Flatt, `G_lang`) with tests |
| D-09.2 | Figures (list below) |
| D-09.3 | Quality versus mode table (F1 and identical flags) |
| D-09.4 | Written interpretation per hypothesis |

## Figures

| # | Figure | Shows | Hypothesis |
|---|---|---|---|
| F1 | Speedup versus threads, per kernel, with the ideal line | scaling shape and saturation | H1 |
| F2 | Efficiency versus threads | where efficiency drops below 70% | H1 |
| F3 | Throughput (samples/s) versus series size, per mode | when parallel pays off | H3 |
| F4 | Time versus size for C-1 and OMP-p (log axes) | the crossover size | H3 |
| F5 | Karp-Flatt `e` versus threads | fixed serial part or growing overhead | H1 |
| F6 | Phase breakdown (input, compute, gather) for MPI and HYB | cost of communication | H4 |
| F7 | OMP-p versus MPI-q versus HYB at equal total workers | single-machine comparison | H4 |
| F8 | Language gain and parallel gain, side by side | `G_lang` versus `S_p` | claim hygiene |
| F9 | `schedule(static)` versus `schedule(dynamic)` on MAD | load balancing | H1 |
| F10 | F1 per mode (flat line) | quality unchanged | H2 |

Style rules: axes labelled with units, the same colour and marker per mode as the brand guide, medians with error bars (interquartile range), log scale only where stated, a caption saying machine, window, stride and number of runs.

## Pseudo code

```python
df = pd.read_csv("experiments/out/results.csv").query("status == 'ok'")
t = df.groupby(["detector", "n", "mode", "q_ranks", "p_threads"]).total_s.median().reset_index()

t1 = t[t["mode"] == "C-1"].set_index(["detector", "n"]).total_s
t["T1"] = t.set_index(["detector", "n"]).index.map(t1)
t["speedup"] = t["T1"] / t["total_s"]
t["workers"] = t["q_ranks"] * t["p_threads"]
t["efficiency"] = t["speedup"] / t["workers"]
t["karp_flatt"] = (1 / t["speedup"] - 1 / t["workers"]) / (1 - 1 / t["workers"])     # workers > 1 only
t["throughput_sps"] = t["n"] / t["total_s"]
```

## Interpretation template (one block per hypothesis)

```
H1 - OpenMP efficiency
Result:        supported / rejected / inconclusive
Evidence:      F1, F2, F5 (numbers: speedup at 4 and 8 threads, efficiency, Karp-Flatt trend)
Reading:       one paragraph: where it saturates and the likely cause (physical cores, memory, serial part)
Limit:         one machine, which kernels, how many runs
```

## Files to create

- `experiments/analysis/{indicators.py, make_figures.py, tables.py}`
- `experiments/out/figures/F01.png … F10.png` and `experiments/out/tables/*.csv`
- `tests/python/test_indicators.py`
- `docs/results.md` (tables, figures, interpretation)

## Tests and acceptance

- Indicator tests: with `T1 = 8` and `Tp = 2` at `p = 4`, speedup is 4 and efficiency is 1; Karp-Flatt is 0 for ideal scaling.
- Every figure regenerates from the CSV with one command and no manual step.
- Each of H1 to H4 is marked supported, rejected or inconclusive, with its figures and numbers.
- At least one case where parallel is not faster (small sizes) is shown and explained.
- Axes carry units; captions state machine and protocol.

## Pitfalls

- **Speedup against the wrong baseline:** the reference is C-1, not Python. Show `G_lang` separately (F8).
- **Efficiency with hyper-threads:** `p` above the physical core count lowers efficiency by construction; mark the physical core line on the plots.
- **Medians versus means:** use medians and show the spread; one slow outlier should not move a headline number.
- **Over-reading small differences:** if the interquartile ranges overlap, say "no measurable difference".
- **Quality claims:** F1 identical for deterministic kernels is by design (H2); do not present it as a finding about the data.
- **Log axes hiding small gaps:** state the scale in the caption.

## Changelog line

`D-09: indicators, ten figures from results.csv, interpretation of H1 to H4`
