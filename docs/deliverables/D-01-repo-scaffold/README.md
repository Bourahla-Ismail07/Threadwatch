# D-01 - Project sheet and repository scaffold

| | |
|---|---|
| **Teacher's item** | Fiche de projet |
| **Depends on** | nothing |
| **Unblocks** | every other deliverable |

## What this part is responsible for

Two things: the one-to-two page project sheet the teacher expects, and a repository where C, OpenMP, Python and MPI build and talk to each other from the first minute. The riskiest setup steps (DLL loading on Windows, MPI) are tested here so they cannot surprise you later.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| Git / GitHub | versioning, one branch per deliverable |
| MSYS2 UCRT64 + GCC | builds the DLL |
| `make` | one-command build and test |
| OpenMP | the "hello threads" check inside the DLL |
| Python 3.10+ (64-bit), venv | runs the checks; must match the 64-bit compiler |
| `ctypes` | loads the DLL |
| MS-MPI + `mpi4py` | the "hello ranks" check |
| pytest | smoke tests |

## Tasks

| Task | What |
|---|---|
| D-01.1 | Project sheet (1 to 2 pages): title, problem, research question, objectives, hypotheses, scope, metrics, deliverable list. Source text: `project-theme-and-abstract.md` |
| D-01.2 | Repository, folders, `.gitignore`, `Makefile`, `pyproject.toml` |
| D-01.3 | "Hello DLL": `tw_version()` and `tw_max_threads()` called from Python |
| D-01.4 | "Hello MPI": two ranks each load the DLL and print their rank |

## Step 0 - Check what you have

```bash
git --version
python --version          # 3.10 or newer, 64-bit
python -c "import struct; print(struct.calcsize('P') * 8)"   # must print 64
gcc --version             # from MSYS2 UCRT64
mpiexec -help | head -3   # MS-MPI
```

Install the compiler (MSYS2 UCRT64 shell):

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make
```

Install MS-MPI (runtime and SDK) from Microsoft, then:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install numpy pandas scikit-learn matplotlib mpi4py fastapi uvicorn psutil pytest ruff
```

## Step 1 - Repo and folders

```bash
mkdir threadwatch && cd threadwatch
git init -b main
mkdir -p sdk/include sdk/src sdk/tests python/threadwatch tests/python apps/dashboard \
         experiments/out docs/deliverables docs/design report data/raw data/series
touch experiments/out/.gitkeep
printf "data/\n.venv/\nbuild/\n__pycache__/\n*.dll\n*.o\nruns/\n" > .gitignore
```

## Step 2 - Hello DLL

`sdk/include/threadwatch.h` (first version):

```c
#ifndef THREADWATCH_H
#define THREADWATCH_H
#include <stdint.h>
#ifdef _WIN32
  #define TW_API __declspec(dllexport)
#else
  #define TW_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
TW_API int tw_version(void);
TW_API int tw_max_threads(void);
TW_API int tw_hello_threads(int32_t threads, int32_t* seen);  /* counts threads that actually ran */
#ifdef __cplusplus
}
#endif
#endif
```

`sdk/src/common.c`:

```c
#include <omp.h>
#include "threadwatch.h"
int tw_version(void) { return 100; }          /* 1.0.0 */
int tw_max_threads(void) { return omp_get_max_threads(); }
int tw_hello_threads(int32_t threads, int32_t* seen) {
    if (!seen || threads < 1) return -1;
    int count = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp atomic
        count++;
    }
    *seen = count;
    return 0;
}
```

`Makefile`:

```make
CC      = gcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -fopenmp -Isdk/include
SRC     = $(wildcard sdk/src/*.c)
build/threadwatch.dll: $(SRC) sdk/include/threadwatch.h
	mkdir -p build
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) -lm
test: build/threadwatch.dll
	python -m pytest tests/python -q
```

Build from the MSYS2 UCRT64 shell: `make -f Makefile build/threadwatch.dll`.

## Step 3 - Load it from Python

`python/threadwatch/lib.py` (first version):

```python
import ctypes as C, os, pathlib

def load(path=None, mingw_bin=r"C:\msys64\ucrt64\bin"):
    root = pathlib.Path(__file__).resolve().parents[2]
    dll = pathlib.Path(path) if path else root / "build" / "threadwatch.dll"
    if os.name == "nt" and os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)        # finds libgomp-1.dll, libwinpthread-1.dll
    lib = C.CDLL(str(dll))                     # CDLL releases the GIL during calls
    lib.tw_version.restype = C.c_int
    lib.tw_max_threads.restype = C.c_int
    lib.tw_hello_threads.argtypes = [C.c_int32, C.POINTER(C.c_int32)]
    lib.tw_hello_threads.restype = C.c_int
    return lib
```

`tests/python/test_hello.py`:

```python
import ctypes as C
from threadwatch.lib import load

def test_hello_dll():
    lib = load()
    assert lib.tw_version() == 100
    assert lib.tw_max_threads() >= 1

def test_threads_run():
    lib = load()
    seen = C.c_int32(0)
    assert lib.tw_hello_threads(4, C.byref(seen)) == 0
    assert seen.value == 4
```

## Step 4 - Hello MPI

```python
# experiments/hello_mpi.py
from mpi4py import MPI
from threadwatch.lib import load
comm = MPI.COMM_WORLD
lib = load()
print(f"rank {comm.Get_rank()}/{comm.Get_size()} dll v{lib.tw_version()} max_threads={lib.tw_max_threads()}")
```

Run: `mpiexec -n 2 python experiments/hello_mpi.py`.

## Files to create

- `docs/` (copy of these documents), `README.md`, `CHANGELOG.md`, `.gitignore`, `Makefile`, `pyproject.toml`
- `sdk/include/threadwatch.h`, `sdk/src/common.c`
- `python/threadwatch/{__init__.py, lib.py}`, `tests/python/test_hello.py`, `experiments/hello_mpi.py`
- `report/project-sheet.md` (exported to PDF for sending)

## Tests and acceptance

- `make test` passes: `tw_version()` and `tw_max_threads()` answer from Python; 4 requested threads give 4 threads seen.
- `mpiexec -n 2 python experiments/hello_mpi.py` prints two lines, one per rank.
- The project sheet fits 1 to 2 pages and lists title, problem, objectives, hypotheses, scope, metrics.
- The build runs from a clean checkout with one command.

## Pitfalls

- **32-bit / 64-bit mismatch:** a 64-bit Python cannot load a 32-bit DLL (error "not a valid Win32 application"). Check both.
- **`FileNotFoundError` when loading a DLL that exists:** a dependency is missing. Use `os.add_dll_directory` for the MSYS2 `bin` folder, or copy `libgomp-1.dll` and `libwinpthread-1.dll` next to your DLL.
- **`tw_max_threads()` returns the number of logical cores:** the physical core count is lower on hyper-threaded CPUs. Record both with `psutil.cpu_count(logical=False)`.
- **`mpi4py` wheel not found:** install the MS-MPI SDK first, or run MPI under WSL with Open MPI.

## Changelog line

`D-01: project sheet, repo scaffold, hello DLL and hello MPI checks passing`
