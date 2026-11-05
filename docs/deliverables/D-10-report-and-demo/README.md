# D-10 - Final report and demonstration

| | |
|---|---|
| **Teacher's item** | Rapport final et démonstration |
| **Depends on** | D-07, D-09 |
| **Unblocks** | nothing (final) |

## What this part is responsible for

The final report, a demonstration that runs from a clean checkout, and the release of the repository. It reuses the structure the course brief gives for the group report: introduction, related work, materials and methods, results, discussion, conclusion.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| Word or Docs | the report (exported to PDF) |
| Dashboard (D-07) | the live part of the demonstration |
| matplotlib figures (D-09) | included in the report |
| Git tag + GitHub release | the final version |
| `make` | the one-command build and demo scripts |

## Tasks

| Task | What |
|---|---|
| D-10.1 | Report skeleton from the template below, filled from existing documents |
| D-10.2 | Results and discussion from `docs/results.md` |
| D-10.3 | Demonstration script and a clean-checkout test |
| D-10.4 | Release: README, changelog summary, tag `v1.0.0` |

## Report structure

| Section | Content | Source |
|---|---|---|
| Abstract | filled version of the draft abstract | `project-theme-and-abstract.md` §4 |
| Introduction | why fast anomaly detection matters for monitoring; question and contribution | theme doc §1 to §3 |
| Related work | anomaly detection in time series; parallel approaches | D-02 review |
| Materials and methods | data, windows, four detectors, DLL design, OpenMP, MPI, hybrid, protocol, metrics | architecture + implementation docs, D-02, D-08 |
| Results | tables and figures F1 to F10 | D-09 |
| Discussion | H1 to H4 outcomes, overhead versus gain, crossover size, speed versus quality, limits | D-09 |
| Conclusion | when parallelism pays off for monitoring; limits; future work (GPU, cluster, streaming) | all |
| Appendix | machine description, matrix, reproduction commands | D-08 |

Rules: every number comes from `results.csv`; every figure from a script; cite only sources actually read; state the limits plainly (one machine, a few datasets, statistical and tree-based detectors only).

## Demonstration script (about 10 minutes)

1. **Problem (1 min):** a long server-metric series, anomalies to detect, why time matters.
2. **Baseline (1 min):** Python detector on 1M samples, time shown.
3. **SDK (1 min):** the DLL and its header; the same detector in C, one thread (T1).
4. **Live run (4 min):** start the dashboard race view on a 5M series: C-1, OMP-4, OMP-8, MPI-4, HYB-2×4. Point at the thread lanes, the core bars, the throughput curves and the final speedup and efficiency cards.
5. **Results (2 min):** F1, F3 and F7 from the report; one case where parallel is not faster.
6. **Conclusion (1 min):** when it pays off, what stays unchanged (F1), the limits.

A fallback screen recording of the same run is kept in case the machine behaves differently on the day.

## Files to create

- `report/report.md` (and exported `report/report.pdf`)
- `docs/demo.md` (script, commands, expected screens)
- `Makefile` targets: `make build`, `make test`, `make demo`, `make bench`
- `README.md` (what it is, how to build, how to run, results summary)
- `CHANGELOG.md` with a project summary of H1 to H4

## Tests and acceptance

- A clean checkout on the same machine builds with `make build`, passes `make test`, and starts the dashboard with `make demo`.
- The report contains every section above, every figure caption states machine and protocol, and all numbers match `results.csv`.
- Each hypothesis has an outcome and a limit.
- The demonstration runs within the planned time without manual fixes.
- Release tag `v1.0.0` exists and the README matches the actual commands.

## Pitfalls

- Numbers copied by hand drift from the data; generate tables from the CSV.
- A live demo on a different power state (battery) gives different timings; use mains power and the same power plan as the campaign.
- Do not claim "real-time" without latency numbers; use throughput and per-window latency from the data.
- Keep the report to what was measured: no claims about GPUs, clusters or production use.

## Changelog line

`D-10: final report, demonstration script, release v1.0.0`
