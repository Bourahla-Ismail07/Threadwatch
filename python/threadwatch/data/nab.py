"""Convert NAB (Numenta Anomaly Benchmark) series to the Threadwatch binary format.

Selection: every CSV of realAWSCloudwatch/ (AWS CloudWatch server metrics). Other folders can
be added with the `folders` argument.
Labels: NAB gives anomaly WINDOWS (start, end timestamps) in labels/combined_windows.json.
A sample is labelled 1 when its timestamp lies inside any window, bounds included.
Missing values (NaN) are forward-filled; a leading NaN is back-filled. Counts are returned.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .loader import DATA_ROOT, write_series

DEFAULT_FOLDERS = ("realAWSCloudwatch",)


def convert_file(csv_path, windows):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ts = df["timestamp"].to_numpy()
    v = df["value"].astype("float64")
    n_missing = int(v.isna().sum())
    v = v.ffill().bfill()
    y = np.zeros(len(df), dtype=np.uint8)
    for start, end in windows:
        y[(ts >= np.datetime64(pd.Timestamp(start))) & (ts <= np.datetime64(pd.Timestamp(end)))] = 1
    step = pd.Series(ts).diff().dropna().median()
    return v.to_numpy(), y, n_missing, step


def convert(nab_dir, root=DATA_ROOT, folders=DEFAULT_FOLDERS):
    nab_dir = pathlib.Path(nab_dir)
    labels = json.loads((nab_dir / "labels" / "combined_windows.json").read_text())
    done = []
    for folder in folders:
        for csv in sorted((nab_dir / "data" / folder).glob("*.csv")):
            key = f"{folder}/{csv.name}"
            x, y, n_missing, step = convert_file(csv, labels[key])
            name = "nab_" + csv.stem
            write_series(name, x, y, {
                "source": f"NAB {key}", "sampling": f"{int(pd.Timedelta(step).total_seconds() // 60)} min",
                "n_windows": len(labels[key]), "n_missing_filled": n_missing})
            done.append((name, len(x), float(y.mean()), n_missing))
    return done
