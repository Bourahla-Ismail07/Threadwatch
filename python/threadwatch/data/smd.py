"""Convert SMD (Server Machine Dataset, OmniAnomaly) to the Threadwatch binary format.

SMD has 28 machines x 38 unnamed metrics. We use the labelled TEST split and ONE metric per
machine, chosen by a written rule (not by averaging):
  metric = the dimension that appears in the most anomaly segments of that machine's
           interpretation_label file (ties -> lowest index). Dimensions are 1-based there.
The chosen metric is stored in the .json (field "metric", 0-based column) and in docs/data.md.
Sampling interval: 1 minute (as described by the dataset authors).
"""
from __future__ import annotations

import collections
import pathlib

import numpy as np

from .loader import DATA_ROOT, write_series


def pick_metric(interp_file, n_cols=38):
    counts = collections.Counter()
    for line in pathlib.Path(interp_file).read_text().splitlines():
        if ":" in line:
            counts.update(int(d) - 1 for d in line.split(":")[1].split(","))
    if not counts:
        return 0
    best = max(counts.values())
    return min(c for c, k in counts.items() if k == best and 0 <= c < n_cols)


def convert(smd_dir, root=DATA_ROOT):
    smd_dir = pathlib.Path(smd_dir)
    done = []
    for f in sorted((smd_dir / "test").glob("machine-*.txt")):
        m = f.stem
        data = np.loadtxt(f, delimiter=",", dtype=np.float64)
        labels = np.loadtxt(smd_dir / "test_label" / f.name, dtype=np.uint8)
        col = pick_metric(smd_dir / "interpretation_label" / f.name, data.shape[1])
        x = data[:, col]
        n_missing = int(np.isnan(x).sum())
        if n_missing:
            x = np.where(np.isnan(x), np.nan, x)
            idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
            np.maximum.accumulate(idx, out=idx)
            x = x[idx]
        write_series("smd_" + m, x, labels, {
            "source": f"SMD {m} (test split)", "sampling": "1 min", "metric": int(col),
            "n_metrics_in_file": int(data.shape[1]), "n_missing_filled": n_missing})
        done.append(("smd_" + m, len(x), float(labels.mean()), col, n_missing))
    return done
