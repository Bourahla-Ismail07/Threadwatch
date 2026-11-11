// Controls, SSE client, state. Drawing happens on requestAnimationFrame, never per event.
import { Panel, marker, kindStyle } from "./panels.js";
import { fmtRate } from "./charts.js";

const $ = (s) => document.querySelector(s);
const grid = $("#grid"), msg = $("#msg"), status = $("#status"), summary = $("#summary");
let panels = new Map();          // label -> Panel
let source = null;               // active EventSource
let raf = 0;
let health = { logicalCores: 4, mpiexec: true };

// ---------------------------------------------------------------- theme
const root = document.documentElement;
function setTheme(t) { root.dataset.theme = t; try { localStorage.setItem("tw-theme", t); } catch { /* private mode */ } panels.forEach((p) => p.redraw()); }
let saved = null; try { saved = localStorage.getItem("tw-theme"); } catch { /* ignore */ }
root.dataset.theme = saved === "light" ? "light" : "dark";            // brand: dark first; the choice is remembered
$("#theme").addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));

// ---------------------------------------------------------------- helpers
const say = (t) => { status.textContent = t; };
function showError(t) { msg.textContent = t; msg.hidden = !t; }
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.message || `${r.status} ${r.statusText}`);
  return body;
}
function setBusy(b) {
  for (const id of ["#run", "#race"]) $(id).disabled = b;
  $("#busy").hidden = !b;
}
function loop() {
  panels.forEach((p) => p.render());
  raf = requestAnimationFrame(loop);
}
const startLoop = () => { if (!raf) raf = requestAnimationFrame(loop); };
const stopLoop = () => { cancelAnimationFrame(raf); raf = 0; panels.forEach((p) => p.render(true)); };
function closeSource() { if (source) { source.close(); source = null; } }
function addPanel(label, kind) {
  const p = new Panel(label, kind);
  panels.set(label, p); grid.appendChild(p.el);
  return p;
}
function reset() { closeSource(); panels = new Map(); grid.textContent = ""; summary.hidden = true; summary.textContent = ""; showError(""); }
const kindOf = (label) => (label === "PY-SEQ" ? "PY-SEQ" : label === "C-1" ? "C-1" : label.startsWith("OMP") ? "OMP" : label.startsWith("MPI") ? "MPI" : "HYB");

// ---------------------------------------------------------------- stream wiring
function listen(url, handlers, onEnd) {
  const es = new EventSource(url);
  source = es;
  for (const [name, fn] of Object.entries(handlers)) es.addEventListener(name, (e) => fn(JSON.parse(e.data)));
  es.onerror = () => {                           // EventSource reconnects by itself (with Last-Event-ID); only report a dead stream
    if (es.readyState === EventSource.CLOSED) { showError("Connection lost. The run continues on the server; reload to replay it."); onEnd(); }
  };
  return es;
}

// ---------------------------------------------------------------- single run
async function runOne() {
  reset();
  const mode = $("#mode").value;
  const body = {
    dataset: $("#dataset").value, detector: $("#detector").value, mode,
    threads: mode === "OMP" || mode === "HYB" ? +$("#threads").value : 1,
    ranks: mode === "MPI" || mode === "HYB" ? +$("#ranks").value : 1,
    allowOversubscribe: $("#oversub").checked,
  };
  if (mode === "MPI" || mode === "HYB") body.inputMode = $("#input").value;
  setBusy(true);
  try {
    const r = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const panel = addPanel(r.mode, mode);
    startLoop();
    const end = () => { closeSource(); stopLoop(); setBusy(false); };
    listen(r.stream, {
      start: (e) => panel.start(e),
      progress: (e) => panel.update(e),
      done: (e) => { panel.finish(e.result); say(`${r.mode} finished in ${(e.result.ms / 1000).toFixed(2)} seconds`); end(); },
      error: (e) => { panel.fail(e.message); showError(e.message); end(); },
    }, end);
  } catch (err) { showError(err.message); setBusy(false); }
}

// ---------------------------------------------------------------- race
function raceModes() { return [...document.querySelectorAll("#racemodes input:checked")].map((i) => i.value); }

async function runRace() {
  reset();
  setBusy(true);
  try {
    const r = await api("/api/race", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modes: raceModes(), dataset: $("#dataset").value, detector: $("#detector").value, allowOversubscribe: $("#oversub").checked }) });
    startLoop();
    const end = () => { closeSource(); stopLoop(); setBusy(false); };
    const get = (e) => panels.get(e.mode);
    listen(r.stream, {
      race_start: (e) => e.modes.forEach((m) => addPanel(m.label, m.kind)),
      start: (e) => get(e)?.start(e),
      progress: (e) => get(e)?.update(e),
      mode_done: (e) => { get(e)?.finish(e.result); say(`${e.mode} finished`); },
      mode_error: (e) => { get(e)?.fail(e.message); },
      done: (e) => { renderSummary(e.table); say("Race finished"); end(); },
      error: (e) => { showError(e.message); end(); },
    }, end);
  } catch (err) { showError(err.message); setBusy(false); }
}

function renderSummary(table) {
  summary.hidden = false;
  const rows = table.map((r) => {
    if (r.error) return `<tr><th scope="row"></th><td colspan="6" class="bad">failed: ${esc(r.error.message)}</td></tr>`;
    return `<tr><th scope="row"></th><td class="num">${(r.ms / 1000).toFixed(2)} s</td><td class="num">${r.speedup != null ? r.speedup.toFixed(2) + "\u00d7" : "-"}</td>` +
      `<td class="num">${r.efficiency != null ? r.efficiency.toFixed(0) + "%" : "-"}</td><td class="num">${fmtRate(r.throughputSps)} samples/s</td>` +
      `<td class="num">${r.f1 != null ? r.f1.toFixed(2) : "-"}</td><td>${r.kind === "C-1" ? "reference" : r.identical === true ? "<b class='ok'>identical</b>" : r.identical === false ? "<b class='bad'>differs</b>" : "-"}</td></tr>`;
  });
  summary.innerHTML = `<h2>Summary (speedup and efficiency are relative to C-1 on the same data)</h2>
    <div class="tablewrap"><table><thead><tr><th>Mode</th><th>Time</th><th>Speedup vs C-1</th><th>Efficiency</th><th>Throughput</th><th>F1</th><th>Output</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  summary.querySelectorAll("tbody tr").forEach((tr, i) => {
    const th = tr.querySelector("th"), row = table[i];
    th.append(marker(row.kind), document.createTextNode(" " + row.mode));
  });
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- controls
function syncMode() {
  const m = $("#mode").value;
  $("#threads-wrap").hidden = !(m === "OMP" || m === "HYB");
  $("#ranks-wrap").hidden = !(m === "MPI" || m === "HYB");
  $("#input-wrap").hidden = !(m === "MPI" || m === "HYB");
}
$("#mode").addEventListener("change", syncMode);
$("#run").addEventListener("click", runOne);
$("#race").addEventListener("click", runRace);
new ResizeObserver(() => panels.forEach((p) => p.redraw())).observe(grid);

async function init() {
  syncMode();
  try {
    health = await api("/api/health");
    $("#cores").textContent = `${health.logicalCores} logical cores`;
    const ds = await api("/api/datasets");
    $("#dataset").innerHTML = "";
    for (const d of ds) { const o = document.createElement("option"); o.value = d.name; o.textContent = `${d.name} \u00b7 ${d.length.toLocaleString("en")} samples${d.labelled ? "" : " (no labels)"}`; $("#dataset").appendChild(o); }
    if (!ds.length) showError("No series found in data/series. Generate data first (D-02).");
    const cores = health.logicalCores;
    document.querySelectorAll("#racemodes input").forEach((i) => {
      const m = /^(?:OMP|MPI)-(\d+)$|^HYB-(\d+)x(\d+)$/.exec(i.value);
      const need = m ? (m[1] ? +m[1] : m[2] * m[3]) : 1;
      i.checked = i.value === "C-1" || (["OMP-4", "MPI-4", "HYB-2x2"].includes(i.value) && need <= cores);
      if (need > cores) i.title = `needs ${need} logical cores, this machine has ${cores}`;
      if (/^MPI|^HYB/.test(i.value) && !health.mpiexec) { i.disabled = true; i.checked = false; i.title = "mpiexec not found"; }
    });
    if (!health.mpiexec) for (const v of ["MPI", "HYB"]) $(`#mode option[value=${v}]`).disabled = true;
    setBusy(!!health.busy);
  } catch (err) { showError(`Service not reachable: ${err.message}`); }
}
init();
