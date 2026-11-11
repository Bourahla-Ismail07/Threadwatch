"""D-07.3 - HTTP API and SSE. The page is served by the same app, so there is no CORS to configure.

    uvicorn threadwatch.service.app:app --port 8000        (from the repo root)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from threadwatch.mpi.driver import logical_cores
from .race import Race, RaceRequest
from .runs import ApiError, EventLog, RunManager, RunRequest, Settings

TERMINAL = ("done", "error")
KEEPALIVE_S = 15.0


def sse_events(log: EventLog, last_id: int = -1, keepalive_s: float = KEEPALIVE_S):
    """Yield SSE frames; a comment line every `keepalive_s` seconds keeps proxies from closing the stream."""
    idx = last_id + 1
    while True:
        events, closed = log.read(idx, keepalive_s)
        if not events:
            if closed:
                return
            yield ": keepalive\n\n"
            continue
        for ev in events:
            yield f"id: {idx}\nevent: {ev.get('type', 'progress')}\ndata: {json.dumps(ev, separators=(',', ':'))}\n\n"
            idx += 1
            if ev.get("type") in TERMINAL:
                return


def stream_response(log: EventLog, request: Request, keepalive_s: float = KEEPALIVE_S) -> StreamingResponse:
    last = request.headers.get("last-event-id")
    start = int(last) if last and last.lstrip("-").isdigit() else -1
    return StreamingResponse(
        sse_events(log, start, keepalive_s), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


def create_app(settings: Optional[Settings] = None, engine=None, py_engine=None, keepalive_s: float = KEEPALIVE_S) -> FastAPI:
    cfg = settings or Settings()
    mgr = RunManager(cfg, engine=engine, py_engine=py_engine)
    app = FastAPI(title="Threadwatch live service", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.mgr = mgr

    @app.exception_handler(ApiError)
    async def _api_error(_, e: ApiError):
        return JSONResponse({"code": e.code, "message": e.message}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_, e: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:])}: {err['msg']}" for err in e.errors())
        return JSONResponse({"code": "bad_request", "message": msg}, status_code=422)

    @app.get("/api/health")
    def health():
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
        except ImportError:
            physical = None
        version = None
        try:
            from threadwatch import sdk
            version = getattr(sdk, "version", lambda: None)()
        except Exception:
            pass
        return {"status": "ok", "dllVersion": version, "logicalCores": logical_cores(), "physicalCores": physical,
                "window": cfg.w, "stride": cfg.s, "mpiexec": bool(shutil.which(cfg.mpiexec)),
                "busy": mgr._busy.locked()}

    @app.get("/api/datasets")
    def datasets():
        return mgr.store.list()

    @app.post("/api/runs", status_code=202)
    def post_run(req: RunRequest):
        run = mgr.start(req)
        return {"runId": run.id, "mode": run.label, "stream": f"/api/runs/{run.id}/stream"}

    @app.get("/api/runs/{run_id}/stream")
    def run_stream(run_id: str, request: Request):
        return stream_response(mgr.get(run_id).log, request, keepalive_s)

    @app.post("/api/race", status_code=202)
    def post_race(req: RaceRequest):
        mgr.acquire()
        try:
            race = Race(mgr, req)
        except Exception:
            mgr.release()
            raise
        race.start()
        return {"raceId": race.id, "modes": [r.label for r in race.requests], "stream": f"/api/race/{race.id}/stream"}

    @app.get("/api/race/{race_id}/stream")
    def race_stream(race_id: str, request: Request):
        race = mgr.races.get(race_id)
        if race is None:
            raise ApiError(404, "no_race", f"race {race_id!r} not found")
        return stream_response(race.log, request, keepalive_s)

    @app.get("/api/results")
    def results(dataset: Optional[str] = None, detector: Optional[str] = None, mode: Optional[str] = None,
                limit: int = 2000):
        try:
            with open(cfg.results_csv, newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return []
        rows = [r for r in rows if (not dataset or r.get("dataset") == dataset)
                and (not detector or r.get("detector") == detector) and (not mode or r.get("mode") == mode)]
        return rows[-max(1, min(limit, 20000)):]

    if cfg.ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=cfg.ui_dir, html=True), name="ui")
    return app


def _default_app() -> FastAPI:
    root = Path(os.environ.get("THREADWATCH_ROOT", "."))
    return create_app(Settings(data_root=root / "data/series", runs_root=root / "runs",
                               results_csv=root / "experiments/out/results.csv", ui_dir=root / "apps/dashboard"))


def __getattr__(name):                      # `uvicorn threadwatch.service.app:app` builds the app on first use,
    if name == "app":                       # so importing this module (tests, tools) has no side effects
        globals()["app"] = _default_app()
        return globals()["app"]
    raise AttributeError(name)
