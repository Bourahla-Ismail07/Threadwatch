# Threadwatch

Parallel anomaly detection in long server-metric time series with OpenMP, MPI and hybrid parallelism (C SDK as a DLL, called from Python).
Project sheet: [`report/project-sheet.md`](report/project-sheet.md) · Docs: [`docs/`](docs/implementation-docs.md)

## Setup (Windows)

| Step | Where | Command |
|---|---|---|
| 1. Compiler | MSYS2 UCRT64 shell | `pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-libgomp` |
| 2. MPI | Windows | Install MS-MPI **runtime** (`msmpisetup.exe`) and **SDK** (`msmpisdk.msi`) |
| 3. Python | PowerShell, python.org Python 3.10+ (64-bit, not the MSYS2 one) | `python -m venv .venv` then `.venv\Scripts\Activate.ps1` |
| 4. Packages | PowerShell, venv active | `pip install numpy pandas scikit-learn scipy matplotlib mpi4py fastapi uvicorn psutil pytest ruff` then `pip install -e .` |
| 5. Build DLL | MSYS2 UCRT64 shell | `make build/threadwatch.dll` |
| 6. Test | PowerShell, venv active | `python -m pytest tests/python -q` |
| 7. Hello MPI | PowerShell, venv active | `mpiexec -n 2 python experiments/hello_mpi.py` |

Notes:
- `cannot find -lgomp` at the link step: install `mingw-w64-ucrt-x86_64-libgomp` (step 1).
- Python 3.14 from MSYS2 cannot build `mpi4py`; use the python.org build, which has a prebuilt wheel.
- `make test` works only where `make` and `python` share a PATH; from PowerShell use step 6.

## Data (D-02)

```powershell
python -m threadwatch.data all      # clones NAB and OmniAnomaly into data/raw, converts, generates syn_1M..syn_20M, rebuilds docs/data.md
```

`data/` is not versioned. Details, licences and plots: [`docs/data.md`](docs/data.md). Literature review (draft): [`report/state-of-the-art.md`](report/state-of-the-art.md).

## Python baseline (D-03)

```powershell
python experiments/first_results.py --datasets syn_1M syn_2M nab_ec2_cpu_utilization_24ae8d nab_ec2_cpu_utilization_5f5533
```

Writes `experiments/out/first-results.csv` and [`docs/first-results.md`](docs/first-results.md) (F1 table and `T_py`).

## Machine: logical vs physical cores

`tw_max_threads()` returns the number of **logical** cores; the physical count can be lower. Record both:

```powershell
python experiments/cpu_info.py
```

It writes `experiments/out/machine.json`. Result for this machine:

| tw_max_threads() | logical cores | physical cores |
|---|---|---|
| 12 | 12 | 8 |
