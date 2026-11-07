# D-07 - Live dashboard

| | |
|---|---|
| **Teacher's item** | extra, shown in the demonstration |
| **Depends on** | D-05 (counters), D-06 for MPI panels |
| **Unblocks** | D-10 |

## What this part is responsible for

A small web page that shows, while a run executes, how each approach works under the hood: progress, throughput, how the work is shared between threads, per-core CPU load and the anomalies as they are found. It must not disturb the measurements it shows: monitoring costs under 2% of run time, and reported benchmark numbers always come from headless runs.

## Tech stack used here

| Tool | Role in this deliverable |
|---|---|
| FastAPI + uvicorn | `POST /api/runs`, SSE stream, static files |
| Python thread | monitor loop that reads counters every 100 ms |
| `np.memmap` | progress buffer shared with the DLL and with MPI ranks |
| psutil | per-core CPU percentages |
| HTML + CSS | layout, brand tokens, light and dark themes |
| vanilla JavaScript (ES modules, `EventSource`) | SSE client, state, rendering |
| Canvas 2D | throughput curve, thread lanes, series strip |

## Tasks

| Task | What |
|---|---|
| D-07.1 | Run manager: start a run in a worker thread, create the progress memmap |
| D-07.2 | Monitor thread: read counters, CPU, new flags; emit events |
| D-07.3 | HTTP API and SSE (`/api/runs`, `/api/runs/{id}/stream`, `/api/race`, `/api/results`) |
| D-07.4 | Page shell: header, controls, mode panels, theme switch |
| D-07.5 | Panels: progress, throughput curve, thread lanes, core bars, series strip, final card |
| D-07.6 | Race view: run chosen modes one after another on the same data |
| D-07.7 | Overhead measurement: headless run versus dashboard run |

## Panels (one per mode)

| Panel part | Data | Why it is there |
|---|---|---|
| Header: mode, colour, marker, elapsed | `mode`, `t` | identifies the approach (OMP-4, HYB-2×2) |
| Progress bar and windows done | `done`, `total` | basic liveness |
| Throughput curve | `throughputSps` over time | shows ramp-up and saturation |
| Thread lanes | `threads[]` (windows done per thread) | one bar per thread; uneven bars show load imbalance |
| Core bars | `cpu[]` | shows whether cores are really busy |
| Series strip | decimated series + `flags[]` | the anomalies appearing as the run advances |
| Final card | `ms`, speedup vs T1, efficiency, F1 | the result, with its reference |

The design reference is `docs/design/threadwatch-template.html` (static mock-up with simulated data).

## Pseudo code

```python
# service/runs.py
class Run:
    def __init__(self, req):
        self.id = new_id(); self.req = req
        self.slots = req.ranks * req.threads
        self.progress = np.memmap(f"runs/{self.id}/progress.bin", np.int64, "w+", shape=(self.slots * 8,))
        self.events = queue.Queue()

    def start(self):
        threading.Thread(target=self._work, daemon=True).start()
        threading.Thread(target=self._monitor, daemon=True).start()

    def _work(self):
        if self.req.ranks == 1:
            out, st = sdk.compute(self.req.detector, self.x, W, S, self.req.threads, progress=self.progress)
        else:
            subprocess.run(["mpiexec", "-n", str(self.req.ranks), "python", "-m", "threadwatch.mpi.run",
                            "--progress", self.progress.filename, ...])
        self.result = summarise(out, st); self.done.set()

    def _monitor(self):
        psutil.cpu_percent(percpu=True)                     # prime the counter
        last_total, last_t = 0, time.perf_counter()
        while not self.done.is_set():
            time.sleep(0.1)
            per_thread = self.progress[::8].tolist()
            total = sum(per_thread); now = time.perf_counter()
            self.events.put({"t": now - self.t0, "mode": self.label, "done": total, "total": self.nw,
                             "throughputSps": (total - last_total) * S / (now - last_t),
                             "threads": per_thread, "cpu": psutil.cpu_percent(percpu=True),
                             "flags": self._new_flags(per_thread)})
            last_total, last_t = total, now

# SSE endpoint
@app.get("/api/runs/{run_id}/stream")
def stream(run_id: str):
    def gen():
        while True:
            ev = runs[run_id].events.get()
            yield f"event: {ev.get('type', 'progress')}\ndata: {json.dumps(ev)}\n\n"
            if ev.get("type") == "done": break
    return StreamingResponse(gen(), media_type="text/event-stream")
```

New flags: with `schedule(static)`, thread `t` owns windows starting at `t·q + min(t, r)` (`q = nw / threads`, `r = nw % threads`). The monitor scans only the part of the score array that thread has newly written since the last poll and reports scores above the detector's threshold (at most 50 per event). In MPI scatter mode flags appear after the gather; in file mode ranks write into a shared score memmap, so flags are live.

Browser side:

```javascript
const es = new EventSource(`/api/runs/${id}/stream`);
es.addEventListener("progress", e => panel.update(JSON.parse(e.data)));
es.addEventListener("done", e => { panel.finish(JSON.parse(e.data)); es.close(); });
```

Draw on `requestAnimationFrame`, not on every event, and keep a fixed-length history buffer (for example 300 points) for the curves.

## Race view

`POST /api/race {modes: ["C-1", "OMP-4", "OMP-8", "MPI-4", "HYB-2x4"], dataset, detector}` runs the modes **one after another** on the same data, in a fixed order. Each panel fills in turn, and a summary table with speedup and efficiency versus C-1 appears at the end. They never run at the same time, because they would compete for the same cores and spoil the timings.

## Files to create

- `python/threadwatch/service/{app.py, runs.py, monitor.py, race.py}`
- `apps/dashboard/{index.html, style.css, app.js, panels.js, charts.js}`
- `tests/python/test_service.py` (API and SSE tests with FastAPI `TestClient`)
- `docs/dashboard.md` (how to start, screenshots)
- `experiments/overhead.py` (headless versus dashboard timing)

## Tests and acceptance

- All five modes (PY-SEQ is shown as progress only, then C-1, OMP-p, MPI-q, HYB-q×p) run from the page on a 5M-sample series and show live counters.
- **Overhead:** median run time with the dashboard connected is within 2% of the headless run (10 runs each, same mode and data).
- Thread lanes show one bar per thread and add up to the total windows done.
- The page works in light and dark themes, is keyboard navigable (controls reachable with Tab, visible focus), respects `prefers-reduced-motion` (no animated curves, values still update), and stacks panels under 768 px.
- Modes are distinguishable without colour (label and marker shape).
- Closing the browser tab does not stop or slow a run.

## Pitfalls

- **GIL:** `ctypes.CDLL` releases the GIL during the call, so the monitor thread keeps running. `ctypes.PyDLL` does not; do not use it.
- **Dashboard disturbing timings:** poll at 10 Hz, not faster; do not parse large arrays in the monitor (read only the new slice); never run the dashboard during benchmark campaigns.
- **`psutil.cpu_percent(percpu=True)`** returns meaningless values on its first call; call it once before the loop.
- **SSE through proxies:** disable buffering and send a comment line every 15 seconds to keep the connection alive.
- **CORS:** serve the page from the same FastAPI app so no CORS setup is needed.
- **Large series in the strip:** decimate (min/max per pixel column) instead of drawing millions of points.
- **Stale runs:** delete `runs/<id>/` after completion, and cap the number of stored runs.

## Changelog line

`D-07: live dashboard (FastAPI SSE + vanilla JS): thread lanes, core load, throughput, race view; overhead < 2%`
