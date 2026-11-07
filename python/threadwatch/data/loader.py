"""Binary series format shared by every stage.

data/series/<name>.f32        little-endian float32, one value per sample
data/series/<name>.labels.u8  one byte per sample, 1 = anomaly
data/series/<name>.json       {"name", "source", "n", "sampling", "seed", "generator_version", ...}

Windows note: np.memmap locks the file while it is open. Call release(x) (or `del x, y`
and gc.collect()) before regenerating a series.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
DATA_ROOT = pathlib.Path(os.environ.get("TW_DATA", _REPO / "data" / "series"))

_META_KEYS = ("name", "source", "n", "sampling", "seed", "generator_version")


def _paths(name: str, root):
    root = pathlib.Path(root)
    return root / f"{name}.f32", root / f"{name}.labels.u8", root / f"{name}.json"


def write_series(name, x, y, meta, root=DATA_ROOT):
    """Write a whole series (used by NAB/SMD; the synthetic generator streams instead)."""
    x = np.ascontiguousarray(x, dtype="<f4")
    y = np.ascontiguousarray(y, dtype=np.uint8)
    if len(x) != len(y):
        raise ValueError("series and labels differ in length")
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    fx, fy, fm = _paths(name, root)
    x.tofile(fx)
    y.tofile(fy)
    write_meta(name, len(x), meta, root)


def write_meta(name, n, meta, root=DATA_ROOT):
    full = {"name": name, "n": int(n), "source": None, "sampling": None,
            "seed": None, "generator_version": None}
    full.update(meta)
    full["name"], full["n"] = name, int(n)
    _, _, fm = _paths(name, root)
    fm.write_text(json.dumps(full, indent=2))


def load_series(name, root=DATA_ROOT):
    """Return (x, y, meta); x and y are read-only memory maps (already contiguous)."""
    fx, fy, fm = _paths(name, root)
    meta = json.loads(fm.read_text())
    x = np.memmap(fx, dtype="<f4", mode="r")
    y = np.memmap(fy, dtype=np.uint8, mode="r")
    if not (len(x) == len(y) == meta["n"]):
        raise ValueError(f"{name}: lengths differ (x={len(x)}, y={len(y)}, meta={meta['n']})")
    return x, y, meta


def release(*arrays):
    """Close memory maps so Windows lets you overwrite the files."""
    for a in arrays:
        mm = getattr(a, "_mmap", None)
        if mm is not None:
            mm.close()


def list_series(root=DATA_ROOT, prefix=""):
    root = pathlib.Path(root)
    return sorted(p.stem for p in root.glob(f"{prefix}*.json"))
