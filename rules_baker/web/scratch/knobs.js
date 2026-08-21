// knobs.js — the Knob Matrix. One cell per weight tensor, in forward-pass order,
// sized by parameter count (√-scaled, or head.weight drowns every LayerNorm at a
// 100x ratio), grouped and hued by where it sits in the stack. A view toggle
// re-skins the SAME layout with the trained statistics, the random-init ones, or
// the difference — the difference view answers "what did training change?".

import { finiteNum } from './inspect.js';

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

const KIND_HUE = { embedding: 265, block: 205, output: 140 };

// The number a cell shows in each view, plus its label. Returns v:null when the
// file has no value for it — the cell renders dark rather than as a false zero.
function cellStat(name, view, trained, random) {
  const T = trained && trained[name], R = random && random[name];
  if (view === 'random') {
    const v = R ? finiteNum(R.std) : null;
    return { v, label: v == null ? 'no value' : `std ${v.toFixed(4)}` };
  }
  if (view === 'diff') {
    const tv = T ? finiteNum(T.std) : null, rv = R ? finiteNum(R.std) : null;
    if (tv == null || rv == null) return { v: null, label: 'no value' };
    const d = tv - rv;
    return { v: Math.abs(d), label: `Δstd ${d >= 0 ? '+' : ''}${d.toFixed(4)}` };
  }
  const v = T ? finiteNum(T.std) : null;
  return { v, label: v == null ? 'no value' : `std ${v.toFixed(4)}` };
}

export function renderKnobs(root, model, view = 'trained') {
  const arch = model.arch || model;             // inspect.json has .arch; arch.json IS the arch
  const trained = model.trained || null, random = model.random || null;
  const hasValues = !!(trained && random);

  // Largest tensor sets the size scale; largest value sets the brightness scale.
  let maxParams = 1, maxStat = 0;
  for (const g of arch.groups) for (const t of g.tensors) {
    maxParams = Math.max(maxParams, t.params);
    if (hasValues) { const s = cellStat(t.name, view, trained, random).v; if (s != null) maxStat = Math.max(maxStat, s); }
  }
  const size = (p) => Math.round(24 + (Math.sqrt(p) / Math.sqrt(maxParams)) * 66);  // 24..90px

  root.innerHTML = '';
  root.dataset.view = view;

  for (const g of arch.groups) {
    const hue = KIND_HUE[g.kind] ?? 210;
    const share = arch.total_params ? (100 * g.params / arch.total_params).toFixed(1) : '?';
    const col = document.createElement('div'); col.className = 'kgroup';
    col.innerHTML = `<div class="kglabel">${esc(g.name)} <span class="kgshare">${esc(share)}%</span></div>`;
    const cells = document.createElement('div'); cells.className = 'kcells';

    for (const t of g.tensors) {
      const st = hasValues ? cellStat(t.name, view, trained, random)
                           : { v: null, label: 'export a checkpoint to see values' };
      const s = size(t.params);
      const lit = (st.v != null && maxStat > 0) ? 22 + 46 * (st.v / maxStat) : 15;  // dark = no value
      const cell = document.createElement('div');
      cell.className = 'knob';
      cell.style.width = cell.style.height = s + 'px';
      cell.style.background = `hsl(${hue} 45% ${lit}%)`;
      cell.style.borderColor = `hsl(${hue} 45% ${Math.min(lit + 18, 82)}%)`;
      cell.title = `${t.name}\n${t.shape.join(' × ')} · ${t.params.toLocaleString()} params\n${t.role}\n${st.label}`;
      cells.appendChild(cell);
    }
    col.appendChild(cells);
    root.appendChild(col);
  }

  // The causal mask: in the stack, but a buffer, not a parameter — shown, marked.
  for (const b of (arch.buffers || [])) {
    const col = document.createElement('div'); col.className = 'kgroup';
    col.innerHTML = `<div class="kglabel">buffer <span class="kgshare">not learned</span></div>`;
    const cells = document.createElement('div'); cells.className = 'kcells';
    const cell = document.createElement('div');
    cell.className = 'knob buffer';
    cell.style.width = cell.style.height = '48px';
    cell.title = `${b.name}\n${b.shape.join(' × ')}\n${b.role}`;
    cells.appendChild(cell);
    col.appendChild(cells);
    root.appendChild(col);
  }
}
