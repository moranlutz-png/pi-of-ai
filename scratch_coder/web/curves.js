// curves.js — the loss curve and the per-layer gradient norms over training, read
// from the loss.jsonl the trainers already write (carried inside inspect.json).
//
// The honest part: at four layers the gradient spread is mild and non-monotonic —
// L0..L3 sit within an order of magnitude of each other. That is NOT the textbook
// "gradients vanish in the early layers", and the caption must not claim it is.
// Four layers is not deep enough to vanish, which is itself worth a student seeing.

import { finiteNum } from './inspect.js';

const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const polyline = (points, stroke, w = 1.7) =>
  el('polyline', { points, fill: 'none', stroke, 'stroke-width': w, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });

function scaler(xs, ys, W, H, pad) {
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  return {
    sx: (x) => pad + (xmax > xmin ? (x - xmin) / (xmax - xmin) : 0.5) * (W - 2 * pad),
    sy: (y) => H - pad - (ymax > ymin ? (y - ymin) / (ymax - ymin) : 0.5) * (H - 2 * pad),
    ymin, ymax,
  };
}

function axisLabels(svg, sc, W, H, pad, fmt) {
  svg.appendChild(el('line', { x1: pad, y1: H - pad, x2: W - pad, y2: H - pad, stroke: 'var(--edge2)', 'stroke-width': 1 }));
  const top = el('text', { x: pad, y: pad - 6, fill: 'var(--dimmer)', 'font-size': 10 }); top.textContent = fmt(sc.ymax);
  const bot = el('text', { x: pad, y: H - pad + 13, fill: 'var(--dimmer)', 'font-size': 10 }); bot.textContent = fmt(sc.ymin);
  svg.append(top, bot);
}

export function renderCurves(root, training) {
  const lossSvg = root.querySelector('#lossCurve');
  const gradSvg = root.querySelector('#gradCurve');
  const legend = root.querySelector('#gradLegend');
  const note = root.querySelector('#curvesNote');

  const rows = (training || []).filter((r) => finiteNum(r.val_loss) != null);
  if (rows.length < 2) {
    note.textContent = 'Not enough logged points yet — run train.py (or train_forever.py), then re-export.';
    return;
  }

  const W = 460, H = 190, pad = 30;

  // --- loss curve ---
  lossSvg.setAttribute('viewBox', `0 0 ${W} ${H}`); lossSvg.innerHTML = '';
  const lsc = scaler(rows.map((r) => r.iter), rows.map((r) => r.val_loss), W, H, pad);
  axisLabels(lossSvg, lsc, W, H, pad, (v) => v.toFixed(2));
  lossSvg.appendChild(polyline(rows.map((r) => `${lsc.sx(r.iter).toFixed(1)},${lsc.sy(r.val_loss).toFixed(1)}`).join(' '), 'hsl(205 72% 62%)', 2));

  // --- per-layer gradient norms ---
  gradSvg.setAttribute('viewBox', `0 0 ${W} ${H}`); gradSvg.innerHTML = ''; legend.innerHTML = '';
  const gradRows = rows.filter((r) => Array.isArray(r.layer_norms) && r.layer_norms.length);
  if (gradRows.length >= 2) {
    const nL = gradRows[0].layer_norms.length;
    const gsc = scaler(gradRows.map((r) => r.iter), gradRows.flatMap((r) => r.layer_norms), W, H, pad);
    axisLabels(gradSvg, gsc, W, H, pad, (v) => v.toExponential(1));
    const last = gradRows[gradRows.length - 1].layer_norms;
    for (let l = 0; l < nL; l++) {
      const hue = (l * 62) % 360;
      gradSvg.appendChild(polyline(gradRows.map((r) => `${gsc.sx(r.iter).toFixed(1)},${gsc.sy(r.layer_norms[l]).toFixed(1)}`).join(' '), `hsl(${hue} 62% 62%)`));
      const chip = document.createElement('span'); chip.className = 'gradchip';
      chip.innerHTML = `<i style="background:hsl(${hue} 62% 62%)"></i>L${l} ${last[l].toExponential(1)}`;
      legend.appendChild(chip);
    }
    note.textContent = 'The four layers sit within an order of magnitude of each other — a mild, non-monotonic spread, not the textbook vanishing-gradient picture. Four layers is not deep enough to vanish.';
  } else {
    note.textContent = 'Loss is logged; gradient norms need a run that logged them (train.py does now).';
  }
}
