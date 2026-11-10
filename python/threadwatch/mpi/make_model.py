"""Fit the K-Means model once and save it for `threadwatch.mpi.run --detector kmeans --model`.

  python -m threadwatch.mpi.make_model --dataset syn_5M --out data/kmeans_syn_5M.npz

Same recipe as D-03: 100k-window sample, k = 8, seed 0, features [mean, std, min, max].
"""
import argparse

import numpy as np

from .. import sdk
from ..data.loader import DATA_ROOT
from ..detectors.ml import fit_models


def main(argv=None):
    ap = argparse.ArgumentParser(prog="threadwatch.mpi.make_model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    x = np.memmap(f"{a.root}/{a.dataset}.f32", np.float32, "r")
    feats, _ = sdk.window_features(np.ascontiguousarray(x), a.w, a.s, threads=1)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(feats), size=min(a.sample, len(feats)), replace=False))
    cen, scale, _ = fit_models(feats[idx], k=a.k, seed=a.seed)
    np.savez(a.out, centroids=cen, scale=scale)
    print(f"saved {a.out}: centroids {cen.shape}, scale {scale.shape}")


if __name__ == "__main__":
    main()
