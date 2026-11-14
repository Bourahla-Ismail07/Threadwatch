"""D-08.1 - machine record (experiments/out/machine.json) and pre-flight checks of the protocol.

    python experiments/machine_info.py            # print the record and the pre-flight warnings
    python experiments/machine_info.py --write    # write experiments/out/machine.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def _ver(mod: str):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return None


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def power_plan():
    """Windows: the active power scheme name; None elsewhere or if unavailable."""
    if platform.system() != "Windows":
        return None
    out = _run(["powercfg", "/getactivescheme"])
    if not out:
        return None
    return out.split("(", 1)[1].rsplit(")", 1)[0] if "(" in out else out


def dll_info(cfg_note: str = ""):
    info = {"build_note": cfg_note or None, "version": None, "path": None, "sha256": None}
    try:
        from threadwatch import sdk
        v = getattr(sdk, "version", None)
        info["version"] = v() if callable(v) else v
        lib = getattr(sdk, "lib", None) or getattr(sdk, "_lib", None)
        path = getattr(lib, "_name", None)
        if path and Path(path).exists():
            info["path"] = str(path)
            info["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def mpi_info():
    try:
        from mpi4py import MPI
        return {"mpi4py": _ver("mpi4py"), "library": MPI.Get_library_version().strip().splitlines()[0]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def collect(dll_build_note: str = "") -> dict:
    import psutil
    freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    bat = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    omp_env = {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "GOMP_", "MKL_", "OPENBLAS_", "KMP_"))}
    info = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": {"current": round(freq.current), "max": round(freq.max)} if freq else None,
        "ram_gb": round(mem.total / 2**30, 1),
        "power_plan": power_plan(),
        "on_battery": (not bat.power_plugged) if bat is not None else None,
        "python": sys.version.split()[0],
        "packages": {m: _ver(m) for m in ("numpy", "scipy", "pandas", "sklearn", "threadpoolctl", "psutil", "joblib", "yaml")},
        "mpi": mpi_info(),
        "dll": dll_info(dll_build_note),
        "thread_env": omp_env,
        "git": {"commit": commit, "dirty": bool(dirty)} if commit else None,
    }
    # what identifies "the same configuration": a resumed campaign must not silently change any of these
    info["fingerprint"] = {
        "processor": info["processor"], "logical_cores": info["logical_cores"], "ram_gb": info["ram_gb"],
        "python": info["python"], "numpy": info["packages"]["numpy"], "sklearn": info["packages"]["sklearn"],
        "dll_sha256": info["dll"].get("sha256"), "dll_version": info["dll"].get("version"),
    }
    return info


def preflight(max_background_cpu_pct: float = 15.0, probe_s: float = 2.0, dashboard_port: int = 8000) -> list:
    """Protocol item 1 as warnings (never silent): mains power, power plan, idle machine, dashboard off."""
    import psutil
    import socket
    warns = []
    bat = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if bat is not None and not bat.power_plugged:
        warns.append("running on battery: plug in the laptop")
    plan = power_plan()
    if plan is not None and not any(k in plan.lower() for k in ("high performance", "ultimate", "haute performance")):
        warns.append(f"power plan is {plan!r}: use High performance")
    busy = psutil.cpu_percent(interval=probe_s)
    if busy > max_background_cpu_pct:
        top = sorted(psutil.process_iter(["name", "cpu_percent"]), key=lambda p: p.info["cpu_percent"] or 0, reverse=True)[:3]
        warns.append(f"background CPU load {busy:.0f}% (limit {max_background_cpu_pct:.0f}%); busiest: "
                     + ", ".join(f"{p.info['name']} {p.info['cpu_percent']:.0f}%" for p in top))
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", dashboard_port)) == 0:
            warns.append(f"something listens on port {dashboard_port} (the dashboard?): keep it off during the campaign")
    return warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments/out/machine.json"))
    ap.add_argument("--note", default="", help="DLL build note, e.g. 'gcc 15 -O3 -fopenmp, no -ffast-math'")
    a = ap.parse_args(argv)
    info = collect(a.note)
    info["preflight_warnings"] = preflight()
    print(json.dumps(info, indent=2))
    if a.write:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(info, indent=2))
        print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
