"""python -m threadwatch.data {fetch,synth,nab,smd,sheet,all}"""
import argparse
import pathlib
import subprocess

from . import nab, smd, synthetic
from .loader import DATA_ROOT
from .sheet import RAW, build

REPOS = {"NAB": "https://github.com/numenta/NAB.git",
         "OmniAnomaly": "https://github.com/NetManAIOps/OmniAnomaly.git"}


def main(argv=None):
    p = argparse.ArgumentParser(prog="threadwatch.data")
    p.add_argument("cmd", choices=["fetch", "synth", "nab", "smd", "sheet", "all"])
    p.add_argument("--sizes", nargs="*", default=["syn_1M", "syn_2M", "syn_5M", "syn_10M", "syn_20M"])
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    todo = ["fetch", "synth", "nab", "smd", "sheet"] if a.cmd == "all" else [a.cmd]
    for c in todo:
        if c == "fetch":
            RAW.mkdir(parents=True, exist_ok=True)
            for name, url in REPOS.items():
                if not (RAW / name).exists():
                    subprocess.run(["git", "clone", "--depth", "1", url, str(RAW / name)], check=True)
        elif c == "synth":
            for s in a.sizes:
                m = synthetic.generate_to_disk(s, synthetic.SIZES[s], seed=a.seed)
                print(f"{s}: n={synthetic.SIZES[s]:,} rate={m['rate_actual']:.5f} events={m['n_events']}")
        elif c == "nab":
            for r in nab.convert(RAW / "NAB"):
                print("nab", r)
        elif c == "smd":
            for r in smd.convert(RAW / "OmniAnomaly" / "ServerMachineDataset"):
                print("smd", r)
        elif c == "sheet":
            print("wrote", build())


if __name__ == "__main__":
    main()
