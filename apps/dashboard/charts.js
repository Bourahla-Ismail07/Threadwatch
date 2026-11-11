// Canvas 2D drawing: throughput curve and series strip. No data handling here, only pixels.

export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Size the canvas backing store to its CSS size times devicePixelRatio; returns a 2D context in device pixels. */
function setup(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  return { g, w, h, dpr };
}

export function fmtRate(sps) {
  if (!isFinite(sps) || sps <= 0) return "0";
  if (sps >= 1e6) return (sps / 1e6).toFixed(sps >= 1e7 ? 1 : 2) + " M";
  if (sps >= 1e3) return (sps / 1e3).toFixed(0) + " k";
  return sps.toFixed(0);
}

/** Throughput over time. `hist` is a fixed-length history (newest last, at most `capacity` points):
 *  it always spans the full width, so a short run is as readable as a long one. */
export function drawThroughput(cv, hist, colorVar, capacity = 300) {
  const { g, w, h, dpr } = setup(cv);
  const color = cssVar(colorVar), muted = cssVar("--muted"), line = cssVar("--line");
  g.strokeStyle = line; g.lineWidth = dpr; g.beginPath();
  for (let k = 1; k < 4; k++) { const y = Math.round((h * k) / 4) + 0.5; g.moveTo(0, y); g.lineTo(w, y); }
  g.stroke();
  if (!hist.length) return;
  const max = Math.max(...hist, 1) * 1.15, pad = 4 * dpr;
  g.strokeStyle = color; g.lineWidth = 2.5 * dpr; g.lineJoin = "round"; g.beginPath();
  hist.forEach((v, i) => {
    const x = (i / Math.max(1, hist.length - 1)) * w;
    const y = h - pad - (v / max) * (h - 2 * pad);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
  g.fillStyle = muted; g.font = `${11 * dpr}px ${cssVar("--mono") || "monospace"}`;
  g.fillText(`max ${fmtRate(max / 1.15)} samples/s`, 6 * dpr, 13 * dpr);
}

/**
 * Series strip: min/max per pixel column (decimated by the server), labelled events as a shaded band,
 * how far each thread block has got as thin bands along the bottom, and flags as circles with a stem.
 * o = {strip:{min,max}, n, nw, blocks:[[start,len]], counters:[], flags:[sampleIdx], labelBins:[col], colorVar}
 */
export function drawStrip(cv, o) {
  const { g, w, h, dpr } = setup(cv);
  const cols = o.strip.min.length, color = cssVar(o.colorVar);
  const bandH = Math.min(5, Math.max(2, Math.floor(18 / Math.max(1, o.blocks.length)))) * dpr;
  const bandsH = o.blocks.length * (bandH + dpr) + dpr;
  const top = 14 * dpr, bottom = h - bandsH - 2 * dpr;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < cols; i++) { lo = Math.min(lo, o.strip.min[i]); hi = Math.max(hi, o.strip.max[i]); }
  if (!(hi > lo)) { hi = lo + 1; }
  const yOf = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
  const colW = w / cols;

  g.fillStyle = cssVar("--warn"); g.globalAlpha = 0.18;                    // labelled anomaly events
  for (const c of o.labelBins) g.fillRect(c * colW, 0, Math.max(colW, dpr), bottom + dpr);
  g.globalAlpha = 1;

  g.fillStyle = cssVar("--seq");                                            // the series itself
  for (let i = 0; i < cols; i++) {
    const y1 = yOf(o.strip.max[i]), y2 = yOf(o.strip.min[i]);
    g.fillRect(i * colW, y1, Math.max(colW, dpr), Math.max(dpr, y2 - y1));
  }

  o.blocks.forEach(([start, len], k) => {                                   // per-thread progress bands
    const y = bottom + dpr * 2 + k * (bandH + dpr);
    const x0 = (start / o.nw) * w, wFull = (len / o.nw) * w;
    g.fillStyle = cssVar("--line"); g.fillRect(x0, y, wFull, bandH);
    g.fillStyle = color; g.fillRect(x0, y, wFull * Math.min(1, (o.counters[k] || 0) / Math.max(1, len)), bandH);
  });

  g.strokeStyle = cssVar("--alert"); g.fillStyle = cssVar("--alert"); g.lineWidth = dpr;
  const r = 3 * dpr;
  for (const i of o.flags) {                                                // flags: stem + circle
    const x = (i / o.n) * w;
    g.beginPath(); g.moveTo(x, top + r); g.lineTo(x, bottom); g.stroke();
    g.beginPath(); g.arc(x, top, r, 0, 2 * Math.PI); g.fill();
  }
}
