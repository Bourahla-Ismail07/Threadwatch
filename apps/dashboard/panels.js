// One panel per mode: header, progress, throughput curve, thread lanes, core bars, series strip, final card.
import { drawStrip, drawThroughput, fmtRate } from "./charts.js";

const KIND = {
  "PY-SEQ": { color: "--seq", marker: "square" },
  "C-1": { color: "--seq", marker: "circle" },
  OMP: { color: "--omp", marker: "triangle" },
  MPI: { color: "--mpi", marker: "diamond" },
  HYB: { color: "--hybrid", marker: "hex" },
};
export const kindStyle = (kind) => KIND[kind] || KIND["C-1"];

const HISTORY = 300;                         // fixed-length history for the curve
const MAX_FLAGS_KEPT = 3000;
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmtInt = (n) => Number(n).toLocaleString("en");
const secs = (ms) => (ms / 1000).toFixed(ms < 10000 ? 2 : 1);

export function marker(kind) {
  const m = document.createElement("span");
  m.className = `mk ${kindStyle(kind).marker}`;
  m.setAttribute("aria-hidden", "true");
  m.style.setProperty("--c", `var(${kindStyle(kind).color})`);
  return m;
}

export class Panel {
  constructor(label, kind) {
    this.label = label; this.kind = kind;
    this.state = "queued";
    this.hist = []; this.flags = []; this.dirty = true; this.last = null; this.counters = [];
    this.el = document.createElement("section");
    this.el.className = "panel queued";
    this.el.setAttribute("aria-label", `Panel ${label}`);
    this.el.style.setProperty("--c", `var(${kindStyle(kind).color})`);
    this.el.innerHTML = `
      <div class="ph"><div class="mode"><span data-k="mk"></span><span data-k="label"></span></div>
        <span class="mono small muted" data-k="el">queued</span></div>
      <div class="bar" role="progressbar" aria-label="Progress of ${label}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i data-k="bar"></i></div>
      <div class="row small"><span class="muted">windows done</span><span class="mono" data-k="done">0</span></div>
      <div class="row small"><span class="muted">throughput</span><span class="mono" data-k="rate">0 samples/s</span></div>
      <h2>Throughput over time</h2><canvas data-k="tp" role="img" aria-label="Throughput curve of ${label}"></canvas>
      <h2>Thread lanes <span class="muted" data-k="lanehint"></span></h2><div class="lanes" data-k="lanes"></div>
      <h2>Core load (%)</h2><div class="cores" data-k="cores"></div>
      <h2>Series and anomalies <span class="muted" data-k="flagn"></span></h2>
      <canvas class="strip" data-k="strip" role="img" aria-label="Series with flagged anomalies for ${label}"></canvas>
      <div class="card" data-k="card" tabindex="-1"></div>`;
    this.q = (k) => this.el.querySelector(`[data-k="${k}"]`);
    this.q("mk").replaceWith(marker(kind));
    this.q("label").textContent = label;
    this._pct = -1;
  }

  start(ev) {
    this.state = "running";
    this.info = ev;
    this.el.classList.remove("queued");
    this.hist = []; this.flags = []; this.counters = new Array(ev.slots).fill(0);
    this.q("el").textContent = "0.0 s";
    this.q("done").textContent = `0 / ${fmtInt(ev.nw)}`;
    const lanes = this.q("lanes"); lanes.textContent = "";
    this.laneEls = [];
    const per = ev.threads;
    for (let k = 0; k < ev.slots; k++) {
      const row = document.createElement("div"); row.className = "lane";
      const name = ev.ranks > 1 ? `R${Math.floor(k / per)}·T${k % per}` : `T${k}`;
      row.innerHTML = `<span class="mono muted"></span><div class="bar"><i></i></div><span class="mono"></span>`;
      row.children[0].textContent = name;
      lanes.appendChild(row);
      this.laneEls.push({ fill: row.querySelector("i"), n: row.children[2] });
    }
    this.q("lanehint").textContent = ev.slots > 1 ? "(windows done per thread; uneven bars = imbalance)" : "(one thread)";
    this.q("flagn").textContent = ev.scoresLive ? "" : "(flags appear when the run ends)";
    this.q("card").classList.remove("on");
    this.dirty = true;
  }

  update(ev) {
    this.last = ev;
    this.counters = ev.threads;
    this.hist.push(ev.throughputSps);
    if (this.hist.length > HISTORY) this.hist.shift();
    for (const f of ev.flags) if (this.flags.length < MAX_FLAGS_KEPT) this.flags.push(f.i);
    this.dirty = true;
  }

  finish(res) {
    this.state = "done";
    this.result = res;
    this.el.classList.add("finished");
    const card = this.q("card");
    const bits = [`Time <b>${secs(res.ms)} s</b>`];
    if (res.speedup != null) {
      bits.push(`speedup <b>${res.speedup.toFixed(1)}\u00d7</b> vs C-1 (T1 ${secs(res.t1Ms)} s)`);
      bits.push(`efficiency <b>${res.efficiency.toFixed(0)}%</b>`);
    } else if (res.kind === "C-1") {
      bits.push("this is the T1 reference");
    } else {
      bits.push("no C-1 reference yet: run C-1 first");
    }
    let line2 = `Throughput <b>${fmtRate(res.throughputSps)} samples/s</b> \u00b7 ${fmtInt(res.flags)} flags`;
    if (res.f1 != null) line2 += ` \u00b7 F1 <b>${res.f1.toFixed(2)}</b>`;
    if (res.identical === true) line2 += ` \u00b7 <b class="ok">identical to C-1</b>`;
    if (res.identical === false) line2 += ` \u00b7 <b class="bad">differs from C-1</b>`;
    card.innerHTML = `${bits.join(" \u00b7 ")}<br>${line2}`;
    card.classList.add("on");
    this.q("el").textContent = `${secs(res.ms)} s`;
    this.dirty = true;
    this.render(true);
  }

  fail(msg) {
    this.state = "error";
    this.el.classList.remove("queued");
    const card = this.q("card");
    card.textContent = `Failed: ${msg}`;
    card.classList.add("on", "err");
    this.q("el").textContent = "failed";
  }

  /** Called from requestAnimationFrame. Numbers always update; curves are drawn live unless reduced motion. */
  render(force = false) {
    if (!this.dirty && !force) return;
    this.dirty = false;
    const ev = this.last, info = this.info;
    if (ev && info) {
      const pct = Math.min(100, Math.floor((ev.done / ev.total) * 100));
      this.q("bar").style.width = `${(ev.done / ev.total) * 100}%`;
      if (pct !== this._pct) { this._pct = pct; this.q("bar").parentElement.setAttribute("aria-valuenow", pct); }
      this.q("done").textContent = `${fmtInt(ev.done)} / ${fmtInt(ev.total)}`;
      if (this.state === "running") {
        this.q("el").textContent = `${ev.t.toFixed(1)} s`;
        this.q("rate").textContent = `${fmtRate(ev.throughputSps)} samples/s`;
      }
      info.blocks.forEach(([, len], k) => {
        const c = this.counters[k] || 0, l = this.laneEls[k];
        if (!l) return;
        l.fill.style.width = `${Math.min(100, (c / Math.max(1, len)) * 100)}%`;
        l.n.textContent = fmtInt(c);
      });
      this._cores(ev.cpu);
      if (ev.flagsTotal) this.q("flagn").textContent = `(${fmtInt(ev.flagsTotal)} flagged)`;
    }
    if (!reduced() || this.state === "done" || force) this._draw();
  }

  _cores(cpu) {
    const box = this.q("cores");
    if (box.children.length !== cpu.length) {
      box.textContent = "";
      cpu.forEach((_, i) => { const b = document.createElement("i"); b.title = `core ${i}`; box.appendChild(b); });
    }
    cpu.forEach((v, i) => { box.children[i].style.height = `${Math.max(3, v)}%`; box.children[i].title = `core ${i}: ${v}%`; });
  }

  _draw() {
    drawThroughput(this.q("tp"), this.hist, kindStyle(this.kind).color, HISTORY);
    if (this.info) {
      drawStrip(this.q("strip"), {
        strip: this.info.strip, n: this.info.n, nw: this.info.nw, blocks: this.info.blocks,
        counters: this.counters, flags: this.flags, labelBins: this.info.labelBins, colorVar: kindStyle(this.kind).color,
      });
    }
  }

  redraw() {                                  // theme switch or resize
    this.dirty = true;
    if (reduced() && this.state !== "done") return;      // reduced motion: curves are drawn when the run ends
    this._draw();
  }
}
