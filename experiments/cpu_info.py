"""Record logical vs physical cores and the OpenMP thread count of the DLL.

Run from the repo root (venv active):  python experiments/cpu_info.py
Writes experiments/out/machine.json and prints one line to paste in the report.
"""
import json
import pathlib
import platform

import psutil
from threadwatch.lib import load

lib = load()
info = {
    "cpu": platform.processor(),
    "tw_max_threads": lib.tw_max_threads(),
    "logical_cores": psutil.cpu_count(logical=True),
    "physical_cores": psutil.cpu_count(logical=False),
    "tw_version": lib.tw_version(),
}
out = pathlib.Path(__file__).parent / "out" / "machine.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(info, indent=2))
print(
    f"tw_max_threads()={info['tw_max_threads']}, "
    f"logical={info['logical_cores']}, physical={info['physical_cores']}"
)
