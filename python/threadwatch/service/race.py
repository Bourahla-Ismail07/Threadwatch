"""D-07.6 - race view: chosen modes one after another on the same data, in a fixed order.

They never run at the same time: they would compete for the same cores and spoil the timings.
C-1 always runs first, because every speedup and efficiency in the summary is relative to it.
"""
from __future__ import annotations

import threading
import uuid
from typing import List

from pydantic import BaseModel, ConfigDict

from .runs import ApiError, EventLog, RunManager, request_from_token

DEFAULT_ORDER = {"C-1": 0, "PY-SEQ": 1}


class RaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modes: List[str]
    dataset: str
    detector: str = "mad"
    allowOversubscribe: bool = False


class Race:
    def __init__(self, mgr: RunManager, req: RaceRequest):
        self.mgr, self.req = mgr, req
        self.id = uuid.uuid4().hex[:10]
        self.log = EventLog()
        self.done = threading.Event()
        # fixed order: C-1 first (reference and T1), then PY-SEQ, then the others as given; C-1 is added if missing; duplicates dropped
        seen, tokens = set(), []
        for t in ["C-1"] + list(req.modes):
            key = t.strip().upper().replace("\u00d7", "X")
            if key not in seen:
                seen.add(key)
                tokens.append(key)
        tokens.sort(key=lambda t: DEFAULT_ORDER.get(t, 2))              # stable: keeps the user's order for the rest
        self.requests = [request_from_token(t, req.dataset, req.detector, req.allowOversubscribe) for t in tokens]
        for r in self.requests:
            mgr.validate(r)                                              # fail early, before anything runs
        self.table: List[dict] = []
        self._released = False

    def _release(self) -> None:
        if not self._released:
            self._released = True
            self.mgr.release()

    def start(self) -> None:
        self.mgr.races[self.id] = self
        threading.Thread(target=self._work, name=f"race-{self.id}", daemon=True).start()

    def _forward(self, ev: dict) -> None:
        ev = dict(ev)
        ev["raceId"] = self.id
        if ev["type"] in ("done", "error"):
            ev["type"] = "mode_done" if ev["type"] == "done" else "mode_error"
        self.log.append(ev)

    def _work(self) -> None:
        try:
            self.log.append({"type": "race_start", "raceId": self.id, "dataset": self.req.dataset,
                             "detector": self.req.detector,
                             "modes": [{"label": r.label, "kind": r.mode, "ranks": r.ranks, "threads": r.threads}
                                       for r in self.requests]})
            ref, t1 = None, None
            for r in self.requests:
                run = self.mgr.create(r, sink=self._forward, keep_scores=(r.mode == "C-1"),
                                      ref_scores=ref, t1_ms=t1)
                run.execute()                                            # synchronous: one mode at a time
                if run.result:
                    self.table.append(run.result)
                    if r.mode == "C-1":
                        ref, t1 = run.scores, run.result["ms"]
                else:
                    self.table.append({"mode": run.label, "kind": r.mode, "ranks": r.ranks, "threads": r.threads,
                                       "error": run.error})
            for row in self.table:                                       # speedup and efficiency versus C-1 (T1)
                if t1 and row.get("ms"):
                    row["t1Ms"], row["speedup"] = t1, t1 / row["ms"]
                    row["efficiency"] = row["speedup"] / (row["ranks"] * row["threads"]) * 100.0
            self._release()                                              # before the terminal event (see Run._finish)
            self.log.append({"type": "done", "raceId": self.id, "table": self.table})
        except Exception as e:
            self._release()
            self.log.append({"type": "error", "raceId": self.id, "code": "race_failed",
                             "message": f"{type(e).__name__}: {e}"})
        finally:
            self._release()
            self.done.set()
            self.log.close()
