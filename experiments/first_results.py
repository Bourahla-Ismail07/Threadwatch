"""Generate baseline quality and timing tables for available datasets.

Expected loader contract from D-02: load_series(name) -> (x, labels, metadata).
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadwatch.detectors.zscore import zscore_py
from threadwatch.detectors.mad import mad_py
from threadwatch.detectors.features import features_py
from threadwatch.detectors.ml import fit_models,kmeans_score_py,isolation_score_py,threshold
from threadwatch.metrics import evaluate
from threadwatch.bench.timing import time_call

def run_one(x,y,w=256,s=1,seed=0,rho=.001,repeats=3):
    nw=(len(x)-w)//s+1
    rng=np.random.default_rng(seed)
    sample_idx=np.sort(rng.choice(nw,size=min(100_000,nw),replace=False))
    F_sample=features_py(x,w,s)[sample_idx]
    centroids,scale,iso=fit_models(F_sample,k=8,seed=seed)
    methods={
      "zscore":lambda:zscore_py(x,w,s),
      "mad":lambda:mad_py(x,w,s),
      "kmeans":lambda:kmeans_score_py(features_py(x,w,s),centroids,scale),
      "isolation_forest":lambda:isolation_score_py(features_py(x,w,s),iso),
    }
    rows=[]
    for name,fn in methods.items():
        scores,timing=time_call(fn,warmup=0,repeats=repeats)
        cut=threshold(scores[sample_idx],rho)
        metrics=evaluate(scores>=cut,y,w,s)
        rows.append({"detector":name,**metrics,"elapsed_ms":timing["median_ms"],"runs_ms":json.dumps(timing["runs_ms"])})
    return pd.DataFrame(rows)

def md_table(df):
    """Markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    def cell(v, c):
        if isinstance(v, float):
            return f"{v:,.1f}" if c.endswith("_ms") else f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v, c) for v, c in zip(row, cols)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def machine_line():
    import sklearn, psutil
    return (f"{platform.processor() or platform.machine()}, logical cores {psutil.cpu_count(True)}, "
            f"physical {psutil.cpu_count(False)}, Python {sys.version.split()[0]}, numpy {np.__version__}, "
            f"scikit-learn {sklearn.__version__}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--datasets",nargs="+",default=["syn_1M","syn_2M"])
    ap.add_argument("--root",default="data/series")
    ap.add_argument("--out",default="experiments/out")
    ap.add_argument("--window",type=int,default=256); ap.add_argument("--stride",type=int,default=1)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--docs",default="docs/first-results.md")
    args=ap.parse_args()
    from threadwatch.data.loader import load_series
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    frames=[]
    for name in args.datasets:
        x,y,meta=load_series(name,args.root)
        frame=run_one(x,y,args.window,args.stride,repeats=args.repeats); frame.insert(0,"dataset",name); frames.append(frame)
    result=pd.concat(frames,ignore_index=True); result.to_csv(out/"first-results.csv",index=False)
    (out/"first-results.md").write_text(md_table(result.drop(columns="runs_ms")))
    print(result.to_string(index=False))
    doc = ["# First results (D-03, Python sequential baseline)", "",
           f"Machine: {machine_line()}.", "",
           f"Window w={args.window}, stride s={args.stride}, threshold = 1 - rho quantile of the scores of a fixed "
           "100,000-window fit sample (rho = 0.001, seed 0). Models are fitted once, outside the timed region. "
           "`elapsed_ms` is the median of the runs listed in `runs_ms` (see the CSV). For `kmeans` and "
           "`isolation_forest` the time includes feature extraction. Everything runs on one thread (`n_jobs=1`). "
           "`elapsed_ms` of `syn_1M` and `syn_2M` is the language baseline T_py for D-04.", "",
           md_table(result.drop(columns="runs_ms")), "",
           "Labels are point-wise; a window is placed at the index of its last sample. Series with no labelled "
           "anomaly have undefined F1 (reported 0)."]
    Path(args.docs).write_text("\n".join(doc))
if __name__=="__main__": main()
