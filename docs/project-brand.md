# Threadwatch — Brand Guide

> Check name availability (GitHub org/repo, PyPI, domain) before committing. Fallback names: **Lanewatch**, **Strandwatch**.

## Identity

- **Name:** Threadwatch (threads + monitoring)
- **Tagline:** *Same detections, in a fraction of the time.*
- **Elevator line:** A C/OpenMP/MPI anomaly-detection SDK for server metrics, benchmarked against Python and scikit-learn, with a live dashboard that shows how each approach runs.
- **Personality:** precise, calm, engineering-first. Benchmark honesty over marketing.

## Voice

- Say what was measured, on which machine, with units. Never "real-time" without a latency number. Never a speedup without its T1 beside it.
- Short sentences. Units always shown (ms, samples/s, MB, x speedup, % efficiency).
- UI copy examples: "OMP-4 · 3.4x vs C-1 · efficiency 85%" · "Throughput 12.4 M samples/s" · "F1 0.91, identical to sequential".

## Colour (dark-first)

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#0D1117` | background |
| `--panel` | `#151B23` | panels |
| `--line` | `#263040` | borders, grid lines |
| `--text` | `#E6EDF3` | primary text |
| `--muted` | `#8B98A9` | secondary text |
| `--seq` | `#9AA6B2` | sequential / baseline modes (PY-SEQ, C-1) |
| `--omp` | `#2DD4BF` | OpenMP |
| `--mpi` | `#60A5FA` | MPI |
| `--hybrid` | `#A78BFA` | MPI + OpenMP |
| `--alert` | `#F87171` | anomaly flagged |
| `--warn` | `#FBBF24` | warning, imbalance |
| `--ok` | `#4ADE80` | check passed |

Light theme: backgrounds `#F6F8FB` / `#FFFFFF`, text `#0D1117`, accents darkened for AA contrast (`--omp #0F766E`, `--mpi #1D4ED8`, `--hybrid #6D28D9`, `--alert #B91C1C`). Never rely on colour alone: every mode also has a text label and a marker shape.

| Mode | Colour | Marker |
|---|---|---|
| PY-SEQ | `--seq` | square |
| C-1 | `--seq` | circle |
| OMP-p | `--omp` | triangle |
| MPI-q | `--mpi` | diamond |
| HYB-q×p | `--hybrid` | hexagon |

## Typography

- UI: **Inter** · Data/code: **JetBrains Mono** · Headings: Inter 600, tight tracking.
- Numbers in tables and panels use tabular figures and monospace.

## Logo concept

One horizontal line (the series) splits into parallel strands (the threads) that meet again at a single dot (the anomaly). Monochrome favicon: three short parallel strokes and a dot. Simple hand-authored SVG, no existing logos or mascots.

## UI principles

1. **Measured, not claimed:** every speed number shows its mode, thread count and reference (T1).
2. **Show the machinery:** per-thread lanes and per-core load are always visible, because the point is to see how the work is shared.
3. **Fair timing:** the dashboard never runs two modes at the same time; reported numbers come from headless runs.
4. **Responsive:** panels stack under 768 px.
5. Accessibility: contrast AA, keyboard navigable, `prefers-reduced-motion` respected.

## Naming conventions

| Thing | Convention |
|---|---|
| DLL and header | `threadwatch.dll`, `threadwatch.h`; exported functions prefixed `tw_` |
| Python package | `threadwatch`, submodules `threadwatch.data`, `.detectors`, `.metrics`, `.mpi`, `.bench`, `.service` |
| Dashboard | `threadwatch-ui` (`apps/dashboard/`) |
| Git tags | `v0.1.0` … semantic versioning |
| Commits | `D-05: add OpenMP MAD kernel` (prefix with deliverable ID) |
| Branches | `d-05-openmp-kernels` |

## README badges / header

`Threadwatch` · tagline · test status · licence · "benchmark prototype, not for production monitoring" notice.
