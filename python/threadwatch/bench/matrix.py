"""D-08.2 - the experiment matrix: cells, blocks, interleaved order, cost bound.

A *cell* is one configuration (dataset, detector, mode, ranks, threads, window, stride, input mode).
A *block* groups the cells that share data and detector; the modes of a block are interleaved round by round
(warm-up round, then the timed rounds), so slow drift (heat, background activity) hits every mode alike.
Blocks run one after another, with a pause between them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

DETECTORS = ("zscore", "mad", "kmeans", "iforest")
MODES = ("PY-SEQ", "C-1", "OMP", "MPI", "HYB", "MPI-scatter")
C_MODES = ("C-1", "OMP", "MPI", "HYB", "MPI-scatter")


def parse_size(v) -> int:
    """'2M' -> 2_000_000, '500k' -> 500_000, 1000 -> 1000."""
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", str(v).strip())
    if not m:
        raise ValueError(f"bad size {v!r}")
    return int(float(m[1]) * {"": 1, "k": 1_000, "m": 1_000_000}[m[2].lower()])


@dataclass(frozen=True)
class Cell:
    dataset: str
    n: int
    detector: str
    mode: str
    q: int = 1                      # MPI ranks
    p: int = 1                      # threads per rank (OpenMP threads, n_jobs, or 1)
    w: int = 256
    s: int = 1
    timeout_s: float = 600.0
    tag: str = ""                   # name of the matrix part that produced the cell (main, second_stride, ...)

    @property
    def input_mode(self) -> str:
        return "scatter" if self.mode == "MPI-scatter" else ("file" if self.mode in ("MPI", "HYB") else "")

    @property
    def label(self) -> str:
        return {"PY-SEQ": "PY-SEQ", "C-1": "C-1", "OMP": f"OMP-{self.p}", "MPI": f"MPI-{self.q}",
                "MPI-scatter": f"MPI-{self.q}-scatter", "HYB": f"HYB-{self.q}x{self.p}"}[self.mode]

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.detector}|{self.label}|w{self.w}|s{self.s}"

    @property
    def block_key(self) -> Tuple[str, str, int, int]:
        return (self.dataset, self.detector, self.w, self.s)

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in ("dataset", "n", "detector", "mode", "q", "p", "w", "s", "timeout_s", "tag")}


@dataclass
class Block:
    key: Tuple[str, str, int, int]
    cells: List[Cell] = field(default_factory=list)

    def order(self, rnd: int) -> List[Cell]:
        """Cell order of round `rnd`: canonical in the warm-up round, then rotated by one position per round,
        so no mode is always first (right after a pause) or always last (hottest)."""
        if rnd == 0 or len(self.cells) < 2:
            return list(self.cells)
        k = rnd % len(self.cells)
        return self.cells[k:] + self.cells[:k]


# ---------------------------------------------------------------------------
def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def auto_threads(logical: int) -> List[int]:
    """2, 4, 8, ... up to the logical cores, and the logical core count itself."""
    out, t = [], 2
    while t <= logical:
        out.append(t)
        t *= 2
    if logical > 1 and logical not in out:
        out.append(logical)
    return out


def cost_bound_s(cfg: dict, mode: str, detector: str, n: int) -> float:
    """Upper bound of one run in seconds. PY-SEQ uses the measured Python rate; every C mode is bounded by the
    C-1 rate (a parallel mode is not slower than C-1 unless it is broken). The rates come from matrix.yaml."""
    t = cfg["timeouts"]
    rate = (t["py_seconds_per_msamples"] if mode == "PY-SEQ" else t["c1_seconds_per_msamples"])[detector]
    return rate * n / 1e6


def build_cells(cfg: dict, logical_cores: int, n_of: Optional[Callable[[str], int]] = None) -> List[Cell]:
    """All cells of the matrix, in canonical order. `n_of(dataset)` gives the sample count (file size); when the
    file does not exist yet (dry run), the size is read from the name (syn_5M -> 5,000,000)."""
    def n_default(name: str) -> int:
        m = re.search(r"_(\d+(?:\.\d+)?[kKmM])$", name)
        if not m:
            raise ValueError(f"cannot infer the size of {name!r}; the file must exist")
        return parse_size(m[1])

    def size(name: str) -> int:
        if n_of:
            try:
                return n_of(name)
            except (FileNotFoundError, OSError):
                pass
        return n_default(name)

    modes_cfg, tcfg = cfg["modes"], cfg["timeouts"]
    only_dets = cfg.get("detectors", list(DETECTORS))
    parts = [{"name": "main", "datasets": cfg["datasets"], "window": cfg["window"], "skip_modes": []}]
    for ex in cfg.get("extras", []) or []:
        parts.append({"name": ex["name"], "datasets": ex["datasets"], "window": ex["window"],
                      "skip_modes": ex.get("skip_modes", [])})

    cells: List[Cell] = []
    for part in parts:
        w, s = part["window"]["w"], part["window"]["s"]
        for ds in part["datasets"]:
            n = size(ds)
            for det in only_dets:
                allowed = (cfg.get("iforest", {}).get("modes") if det == "iforest" else None)
                max_n = None
                if det == "iforest":
                    max_n = parse_size(cfg["iforest"]["max_n"])

                def add(mode, q=1, p=1):
                    if mode in part["skip_modes"] or (allowed is not None and mode not in allowed):
                        return
                    if q * p > logical_cores:
                        return
                    if mode == "PY-SEQ":
                        lim = modes_cfg["PY-SEQ"]["max_n"].get(det)
                        if lim is not None and n > parse_size(lim):
                            return
                    if max_n is not None and n > max_n:
                        return
                    bound = cost_bound_s(cfg, mode, det, n)
                    timeout = max(float(tcfg["minimum_s"]), float(tcfg["factor"]) * bound)
                    cells.append(Cell(ds, n, det, mode, q, p, w, s, timeout, part["name"]))

                add("C-1")                                      # first: its output is the reference of the block
                if "PY-SEQ" in modes_cfg:
                    add("PY-SEQ")
                if "OMP" in modes_cfg:
                    th = modes_cfg["OMP"].get("threads", "auto")
                    for p in (auto_threads(logical_cores) if th == "auto" else th):
                        add("OMP", 1, int(p))
                if "MPI" in modes_cfg:
                    for q in modes_cfg["MPI"]["ranks"]:
                        add("MPI", int(q), 1)
                        if ds in (modes_cfg["MPI"].get("scatter_on") or []):
                            add("MPI-scatter", int(q), 1)
                if "HYB" in modes_cfg:
                    for q, p in modes_cfg["HYB"]["shapes"]:
                        add("HYB", int(q), int(p))
    seen = set()
    for c in cells:
        if c.key in seen:
            raise ValueError(f"duplicate cell {c.key}")
        seen.add(c.key)
    return cells


def blocks_of(cells: Sequence[Cell]) -> List[Block]:
    blocks: Dict[tuple, Block] = {}
    for c in cells:
        blocks.setdefault(c.block_key, Block(c.block_key)).cells.append(c)
    return list(blocks.values())


def filter_cells(cells: Sequence[Cell], datasets=None, detectors=None, modes=None) -> List[Cell]:
    def ok(c):
        return ((not datasets or c.dataset in datasets) and (not detectors or c.detector in detectors)
                and (not modes or c.mode in modes or c.label in modes))
    return [c for c in cells if ok(c)]


def estimate(cfg: dict, cells: Sequence[Cell]) -> dict:
    """Rough upper bound of the whole campaign (see cost_bound_s) and the number of runs."""
    pr = cfg["protocol"]
    per = pr["warmup"] + pr["repeats"]
    sec = sum(cost_bound_s(cfg, c.mode, c.detector, c.n) * per for c in cells)
    pauses = len(blocks_of(cells)) * float(pr.get("pause_between_blocks_s", 0))
    return {"cells": len(cells), "blocks": len(blocks_of(cells)), "runs": len(cells) * per,
            "upper_bound_s": sec + pauses}
