# D-11 - Rust CLI (`tw`)

| | |
|---|---|
| **Teacher's item** | extra, optional, usable in the demonstration |
| **Depends on** | D-07 (progress file contract), D-08 (benchmark runner) |
| **Unblocks** | nothing (last deliverable) |

## What this part is responsible for

A single fast command-line tool, `tw`, that runs the whole project from the terminal: check the toolchain, build the DLL, run the tests, run one detection with a live view, race several modes, run the benchmark campaign and print the results, with clear animations. It **orchestrates and displays only**: it starts `make`, `pytest`, `python` and `mpiexec` as processes and reads files. It never computes detections and it changes none of the logic of D-01 to D-10.

## Performance rules (what "best performance" means here)

1. **The CLI must not disturb what it launches.** No animation or live drawing runs during a timed benchmark cell: `tw bench` is quiet by design and updates one status line only between cells.
2. **Low overhead:** start-up under 50 ms, near 0% CPU when idle, under 20 MB of memory, drawing at 10 Hz at most.
3. **No async runtime:** `std::thread` and channels are enough; the work is waiting on child processes and a 10 Hz poll.
4. **Read-only and lock-free:** the live view maps `progress.bin` read-only; it never writes to anything a run is using.
5. **Release profile:** `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `strip = true`.
6. **Optional native runner (D-11.8):** `libloading` calls the DLL directly, avoiding Python start-up for quick runs. Its numbers are never mixed into the benchmark tables.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| Rust (stable) + cargo | the CLI, one crate `tw-cli`, binary `tw` |
| `clap` (derive) | commands, flags, help, shell completions |
| `cliclack` | interactive choice menus (arrow keys, multi-select, text input with validation), in the style of the `create-vite` / `create-svelte` prompts; `inquire` is an equivalent alternative |
| `num_cpus` | physical and logical core counts for the thread menu |
| `indicatif` | spinners and progress bars (build, tests, race) |
| `ratatui` + `crossterm` | live run view: thread lanes, throughput, core bars |
| `memmap2` | read-only view of `runs/<id>/progress.bin` |
| `serde`, `serde_json`, `csv` | run metadata, JSON output, results tables |
| `anyhow` | error handling with context |
| `ctrlc`, `which` | clean Ctrl-C, find `python`, `make`, `mpiexec` |
| `libloading` (optional) | native runner |
| `assert_cmd`, `tempfile` | integration tests |
| Existing stack | `make`, `pytest`, `python -m threadwatch...`, `mpiexec`, the DLL, `results.csv` |

The CLI starts everything as external processes, so its Rust toolchain is independent of the C toolchain: use the MSVC target (Visual Studio Build Tools) or `stable-x86_64-pc-windows-gnu`, whichever installs more easily.

## Commands

| Command | What it does | Animation |
|---|---|---|
| `tw doctor` | checks `gcc`, 64-bit Python, `make`, `mpiexec`, `mpi4py`, DLL loads, thread count | checklist ticks one by one |
| `tw build` | runs `make`, shows compiler errors on failure | spinner, elapsed time |
| `tw test [--sdk \| --py \| --mpi \| --all]` | runs `pytest` and the `mpiexec` tests, prints a summary | progress bar from pytest output, then pass/fail table |
| `tw run -d syn_5M --detector mad --mode omp -t 4` | one run with a live view | thread lanes, throughput curve, core bars |
| `tw race -d syn_5M --detector mad --modes c1,omp4,mpi4,hyb2x2` | modes one after another on the same data | one lane block per mode, final speedup table |
| `tw bench [--resume] [--quiet]` | runs the D-08 matrix | status line only between cells; nothing while timing |
| `tw results [--detector mad]` | prints summary tables from `results.csv` | none |
| `tw serve` | starts the dashboard service (D-07) and prints the URL | spinner until ready |
| `tw version` | CLI, DLL and Python package versions | none |
| `tw help [command]` | grouped overview, or the help of one command (same as `tw <command> --help`) | none, instant |
| `tw completions <shell>` | prints shell completions (PowerShell, bash, zsh, fish) | none |

Running `tw` with no command opens a main menu (Doctor, Build, Test, Run, Race, Bench, Results, Serve, Help). Any parameter not given as a flag is asked in a menu (next section). A flag always wins over a menu.

Global flags: `--no-anim`, `--plain` (no colour, no cursor control), `--json` (machine-readable result), `--yes` (accept defaults, never prompt), `-q`. Animation is off automatically when output is not a terminal, when `NO_COLOR` is set, or when `TW_REDUCED_MOTION=1`.

## Help

`tw help`, `tw -h` and `tw --help` print a grouped overview; `tw help <command>` and `tw <command> --help` print the help of one command. Help is static and instant (no animation, no menu, no child process), uses the brand colours, and drops colour with `NO_COLOR` or when output is not a terminal.

```
Threadwatch CLI  (tw 1.0.0)
Same detections, in a fraction of the time.

USAGE
  tw <command> [options]        run with no command for a menu

SETUP
  doctor        Check the toolchain: gcc, Python 64-bit, make, MPI, the DLL
  build         Build threadwatch.dll with make

VERIFY
  test          Run the SDK, Python and MPI tests

RUN
  run           One detection run with a live view
  race          Several modes one after another on the same data
  bench         The full benchmark campaign (quiet while timing)

INSPECT
  results       Print summary tables from results.csv
  serve         Start the web dashboard
  version       Show CLI, DLL and package versions
  help          This screen, or help for one command
  completions   Print shell completions

GLOBAL OPTIONS
  --no-anim     No animations
  --plain       No colour and no cursor control
  --json        Machine-readable output
  --yes         Accept defaults, never prompt
  -q, --quiet   Print only the result
  -h, --help    Help

EXAMPLES
  tw run -d syn_5M --detector mad --mode omp -t 4
  tw race -d syn_5M --detector mad --modes c1,omp4,mpi4,hyb2x2
  tw bench --resume --quiet
  tw test --sdk --py

EXIT CODES
  0 success   1 a step failed   2 usage error or missing flags   130 cancelled

ENVIRONMENT
  NO_COLOR            no colour
  TW_REDUCED_MOTION=1 no animation
  Config file         tw.toml in the project folder; last choices in ~/.config/tw/last.json
```

Per-command help follows the same layout: one line of purpose, usage, options with their defaults and allowed values, two examples, and the equivalent menu step (for example `tw run --help` lists `--dataset`, `--detector`, `--mode`, `-t/--threads`, `--ranks`, `-w/--window`, `-s/--stride`). A mistyped command prints the closest match (`did you mean 'bench'?`) with exit code 2.

Implementation: `clap` generates the per-command help and the suggestions; the overview uses a custom `help_template` with `styles` for the colours, so it stays in one place and cannot drift from the commands.

## Interactive menus

Menus appear only in a terminal, only for parameters you did not pass, and only **before** a run starts, so they cost nothing while timing. Arrow keys move, Enter confirms, Space ticks in multi-select, Esc cancels, typing filters long lists.

```
┌  tw run
│
◆  Dataset
│  ● syn_5M        5,000,000 samples · synthetic · labelled
│  ○ syn_10M       10,000,000 samples · synthetic · labelled
│  ○ syn_20M       20,000,000 samples · synthetic · labelled
│  ○ nab_cpu_utc2  4,032 samples · NAB real server metric · labelled
│
◆  Detector
│  ○ Rolling z-score   (C, OpenMP)
│  ● Rolling MAD       (C, OpenMP)
│  ○ K-Means distance  (C, OpenMP)
│  ○ Isolation Forest  (C features + scikit-learn)
│
◆  Mode
│  ○ PY-SEQ   Python baseline
│  ○ C-1      C, 1 thread (reference T1)
│  ● OMP-p    OpenMP threads
│  ○ MPI-q    MPI processes
│  ○ HYB-q×p  MPI + OpenMP
│
◆  Threads
│  ○ 1    ○ 2    ● 4    ○ 8 (physical cores)    ○ 16 (logical cores)    ○ Custom…
│
◆  Window length (w)  256
◆  Stride (s)         1
│
◇  Summary
│  syn_5M · MAD · OMP-4 · w=256 · s=1
│  Equivalent command: tw run -d syn_5M --detector mad --mode omp -t 4
│
◆  Start the run?  ● Yes  ○ No
└
```

| Menu | Type | Choices | Validation |
|---|---|---|---|
| Dataset | select | every `data/series/*.json` found, with size, source and labelled flag | the `.f32` and `.labels.u8` files exist and match `n` |
| Detector | select | z-score, MAD, K-Means, Isolation Forest | Isolation Forest above the cap (default 2M) asks for confirmation |
| Mode | select | PY-SEQ, C-1, OMP-p, MPI-q, HYB-q×p | PY-SEQ above its cap (default 2M) asks for confirmation |
| Threads (p) | select + custom | 1, 2, 4, 8, … up to the logical core count; physical and logical counts labelled | integer from 1 to the logical count; above it asks for confirmation (oversubscription) |
| Ranks (q) | select + custom | 2, 4, … (only for MPI and HYB) | `q × p` at most the logical count; above it asks for confirmation |
| MPI input | select | file, scatter (only for MPI and HYB) | none |
| Window, stride | text with default | default 256 and 1 | integers; `w >= 2`, `s >= 1`, `w <= n` |
| Race modes | multi-select | C-1, OMP-2, OMP-4, OMP-8, MPI-2, MPI-4, HYB-2×2, HYB-2×4 | at least one; `q × p` rule per entry |
| Bench matrix | multi-select | sizes, detectors, modes (from `matrix.yaml`) | at least one of each |
| Tests to run | multi-select | SDK, Python, MPI | at least one |

Behaviour:

- **Defaults remember your last choice** (`~/.config/tw/last.json`, or the platform equivalent): the menu opens on what you picked last time.
- **Always shows the equivalent flags**, so a menu run can be repeated in a script or put in the report's reproduction commands.
- **Dataset list is read from `data/series/*.json`** by the CLI itself (no Python call), so the menu opens instantly.
- **Not a terminal or `--yes`:** no menu is shown; missing required parameters are an error that names the missing flags (`missing: --dataset, --detector`).
- **Cancel:** Esc or Ctrl-C at any menu exits with code 130 and starts nothing.

## Look (terminal version of the brand)

- Colours in truecolor ANSI with the brand tokens: baseline `#9AA6B2`, OpenMP `#2DD4BF`, MPI `#60A5FA`, hybrid `#A78BFA`, alert `#F87171`, ok `#4ADE80`.
- Every mode has a marker as well as a colour: PY-SEQ ■, C-1 ●, OMP ▲, MPI ◆, HYB ⬢. ASCII fallbacks (`[]`, `o`, `^`, `<>`, `#`) when the terminal is not UTF-8.
- Banner: the logo (one line splitting into strands that meet at a dot) drawn in under 600 ms, skipped by any key, `--no-anim`, or a non-terminal.
- Numbers use fixed width, units always shown: `OMP-4 · 3.5x vs C-1 · efficiency 88%`.

## Tasks

| Task | What |
|---|---|
| D-11.1 | Thin Python entry `python -m threadwatch.runner`: wraps existing functions, accepts `--progress <path>`, prints one JSON result line. No logic change |
| D-11.2 | Crate scaffold, `tw.toml` config (paths, defaults), `tw doctor`, `tw version` |
| D-11.3 | Process runner: spawn, stream stdout and stderr line by line, exit codes, Ctrl-C cleanup |
| D-11.4 | `tw build` and `tw test` with spinner, progress bar and summary |
| D-11.5 | `tw run` live view reading `progress.bin` |
| D-11.6 | `tw race`, `tw bench`, `tw results`, `tw serve` |
| D-11.7 | Performance pass: release profile, overhead measurements, plain and JSON modes |
| D-11.8 | Optional native runner with `libloading` |
| D-11.10 | Help: grouped overview, per-command help with examples, exit codes and environment variables, shell completions |
| D-11.9 | Interactive menus: dataset, detector, mode, threads, ranks, window, stride, multi-selects, remembered defaults, equivalent-flags summary |

## Contracts the CLI relies on

- `runs/<id>/progress.bin`: `int64` counters, thread `t` at index `t * 8` (64-byte stride), as defined in `system-architecture.md` §8. File length is `slots * 64` bytes.
- `runs/<id>/meta.json`: written by `threadwatch.runner` before the run starts: `{"label": "OMP-4", "windows": 4999745, "slots": 4, "ranks": 1, "threads": 4}`.
- Result line from `threadwatch.runner`: `{"mode": "OMP-4", "elapsedMs": 2741.0, "dllMs": 2733.2, "f1": 0.91, "identical": true}`.
- `experiments/out/results.csv` columns as in D-08.

If `tw run` is used for a run started by the dashboard service (D-07), it can attach read-only to the same `progress.bin`.

## Pseudo code

```rust
// progress.rs
pub struct Progress { map: memmap2::Mmap, slots: usize }
impl Progress {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };           // read-only mapping
        Ok(Self { slots: map.len() / 64, map })
    }
    pub fn done(&self, t: usize) -> i64 {
        let p = self.map.as_ptr() as *const i64;
        unsafe { std::ptr::read_volatile(p.add(t * 8)) }           // written by another process
    }
    pub fn total(&self) -> i64 { (0..self.slots).map(|t| self.done(t)).sum() }
}

// runner.rs: stream a child process without blocking the UI
pub fn spawn_streaming(cmd: &mut Command, tx: Sender<Line>) -> anyhow::Result<Child> {
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    for (stream, kind) in [(child.stdout.take().unwrap(), Kind::Out), (child.stderr.take().unwrap(), Kind::Err)] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines().map_while(Result::ok) { let _ = tx.send(Line { kind, text: line }); }
        });
    }
    Ok(child)
}

// run_view.rs: 10 Hz, no busy loop
loop {
    terminal.draw(|f| draw(f, &state))?;                           // lanes, curve, cores
    if crossterm::event::poll(Duration::from_millis(100))? { handle_key(&mut state)?; }   // sleeps up to 100 ms
    state.update(progress.done_all(), cpu_sample());               // cheap reads
    if child.try_wait()?.is_some() { break; }
}

// test.rs: parse pytest -v lines for live progress, then the junit summary
// "tests/python/test_sdk_equivalence.py::test_zscore PASSED [ 45%]"
fn parse_pytest_line(s: &str) -> Option<(String, Outcome, u8)> { /* regex-free split on ' ' and '[' */ }
```

Per-core CPU in the live view comes from the run's own events when the dashboard service is used, or from a lightweight sampler (`sysinfo` crate, refreshed at 2 Hz) when running standalone.

## Files to create

- `crates/tw-cli/Cargo.toml` (release profile as above)
- `crates/tw-cli/src/{main.rs, cli.rs, config.rs, doctor.rs, runner.rs, progress.rs, test_cmd.rs, build_cmd.rs, run_cmd.rs, race_cmd.rs, bench_cmd.rs, results_cmd.rs, ui/{mod.rs, theme.rs, spinner.rs, lanes.rs, banner.rs}}.rs`
- `crates/tw-cli/src/prompts/{mod.rs, datasets.rs, params.rs, last.rs}`
- `crates/tw-cli/tests/{parse_pytest.rs, progress_file.rs, cli_smoke.rs, prompts_logic.rs}`
- `python/threadwatch/runner.py`, `tests/python/test_runner_entry.py`
- `tw.toml.example`, `docs/cli.md` (commands, screenshots, performance numbers)

## Tests and acceptance

- `tw doctor` passes on the development machine and fails with a clear message when a tool is missing.
- `tw test --all` exit code equals the combined exit code of `pytest` and the `mpiexec` tests; the summary counts match pytest's.
- With output redirected to a file, no ANSI escape codes appear and no animation is drawn.
- `tw run` on `syn_5M` shows one lane per thread; the lanes add up to the total windows done.
- **Overhead:** median run time with the live view is within 2% of the same run started directly (10 runs each, same mode and data).
- `tw bench` produces a `results.csv` with the same columns and the same `identical == True` results as the D-08 command; during timed cells the CLI uses less than 0.5% of one core.
- Start-up (`tw version`) under 50 ms; idle CPU under 1% during a 60 s `tw run`; memory under 20 MB.
- Ctrl-C stops the CLI and leaves no `python` or `mpiexec` process behind.
- **Menus:** `tw run` with no flags in a terminal asks dataset, detector, mode, threads (and ranks for MPI and HYB), window and stride, shows a summary with the equivalent command, and starts only after confirmation.
- Flags override menus: `tw run -d syn_5M --detector mad --mode omp -t 4 --yes` shows no menu.
- **Help:** `tw help`, `tw -h` and `tw --help` exit with 0 and list every command in its group; `tw help run` equals `tw run --help`; every command's help shows at least one example; a mistyped command suggests the closest one and exits with 2.
- Help output contains no ANSI codes when `NO_COLOR` is set or output is redirected, and prints in under 50 ms.
- `tw completions powershell` (and bash, zsh, fish) prints a script that completes command names and flags.
- With input redirected or `--yes`, no menu appears and missing required flags are reported by name with exit code 2.
- The thread menu lists the logical and physical core counts of the machine; entering a value above the logical count asks for confirmation; `q × p` above the logical count asks for confirmation.
- The dataset menu lists exactly the series in `data/series/` and hides entries whose files are missing or inconsistent.
- The last choices are offered as defaults on the next run.
- Unit tests: pytest line parser (passed, failed, skipped, xfail), progress reader on a temporary file, config loading.

## Pitfalls

- **Animations steal CPU from the run:** a spinner on a busy 8-thread machine can cost a core. Keep drawing at 10 Hz and never animate during timed benchmark cells.
- **Orphan processes:** killing `tw` does not kill `mpiexec` and its Python children on Windows. Track child PIDs, and kill the process tree on exit (`taskkill /T /F` on Windows).
- **Terminals:** the legacy Windows console does not render truecolor or block characters well. Recommend Windows Terminal, keep the ASCII fallback and `--plain`.
- **Buffered child output:** Python buffers stdout when piped; pass `-u` or set `PYTHONUNBUFFERED=1`, or the progress looks frozen.
- **Reading a file being written:** the counters are plain `int64` stores; a torn read is not possible for aligned 8-byte values on x86-64, but treat the numbers as informational only.
- **The file must exist:** wait for `meta.json` and `progress.bin` with a short retry loop instead of failing at once.
- **Help drifting from reality:** do not write the command list by hand twice. Generate the overview from the same `clap` definitions that parse the commands, and add a test that every subcommand appears in `tw help`.
- **Prompts and raw mode:** finish every menu before the live view starts; two libraries cannot own the terminal at the same time. Restore the terminal on Esc, Ctrl-C and panics (use a guard that resets raw mode on drop).
- **Menus never run in unattended runs:** `tw bench` in a script or scheduled task must use flags or `--yes`, or it would wait forever for a keypress.
- **Core counts:** `available_parallelism` can be lower than the hardware count under CPU limits; show both it and `num_cpus::get_physical()`.
- **Stale dataset entries:** read the metadata file and check that the data files exist, or the menu offers a series that fails on start.
- **Do not duplicate logic:** if a rule exists in Python (thresholds, metrics), call Python; the CLI never reimplements it.
- **Benchmark comparability:** results from the optional native runner (D-11.8) go in a separate file and are never merged into the D-08 tables.

## Changelog line

`D-11: Rust CLI tw (doctor, build, test, run, race, bench, results, serve, help) with interactive menus and live view, start-up < 50 ms, overhead < 2%`
