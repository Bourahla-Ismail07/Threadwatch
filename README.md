# Threadwatch

Parallel anomaly detection in long server-metric time series with OpenMP, MPI and hybrid parallelism (C SDK as a DLL, called from Python).
Project sheet: [`report/project-sheet.md`](report/project-sheet.md) · Full docs: [`docs/`](docs/implementation-docs.md)

## Setup (Windows)

| Step | Where | Command |
|---|---|---|
| 1. Compiler | MSYS2 UCRT64 shell | `pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-libgomp` |
| 2. MPI | Windows | Install MS-MPI **runtime** (`msmpisetup.exe`) and **SDK** (`msmpisdk.msi`) |
| 3. Python | PowerShell, python.org Python 3.10+ (64-bit, not the MSYS2 one) | `python -m venv .venv` then `.venv\Scripts\Activate.ps1` |
| 4. Packages | PowerShell, venv active | `pip install numpy pandas scikit-learn matplotlib mpi4py fastapi uvicorn psutil pytest ruff` then `pip install -e .` |
| 5. Build DLL | MSYS2 UCRT64 shell | `make build/threadwatch.dll` |
| 6. Test | PowerShell, venv active | `python -m pytest tests/python -q` |
| 7. Hello MPI | PowerShell, venv active | `mpiexec -n 2 python experiments/hello_mpi.py` |

Notes:
- If the link step says `cannot find -lgomp`, install `mingw-w64-ucrt-x86_64-libgomp` (step 1).
- Python 3.14 from MSYS2 cannot build `mpi4py`; use the python.org build, which has a prebuilt wheel.
- `make test` works only where `make` and `python` are on the same PATH; from PowerShell use step 6.

## Machine: logical vs physical cores

`tw_max_threads()` returns the number of **logical** cores. On a hyper-threaded CPU the physical count is lower, and H1 is stated up to the *physical* count, so both are recorded:

```powershell
python experiments/cpu_info.py
```

It prints e.g. `tw_max_threads()=8, logical=8, physical=4` and writes `experiments/out/machine.json`. Result for this machine:

| tw_max_threads() | logical cores | physical cores |
|---|---|---|
| 12 | 12 | 8 |
