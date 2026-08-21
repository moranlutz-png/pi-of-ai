// knobs.js — the model drawn as the ROUTE data takes through it, left to right.
// Text enters on the left as characters and is carried through embedding, each
// transformer block, and the output head, to a guess at the next character. Every
// station says what it does and shows the weight tensors it is built from (sized by
// parameter count). The trained / random / difference toggle recolours those tensors
// so you can see what training changed at each step of the route.

import { finiteNum } from './inspect.js';

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const glyph = (c) => ({ ' ': '␠', '\n': '↵', '\t': '⇥', '\r': '␍' }[c] ?? c);

const HUE = { input: 38, embedding: 265, block: 205, output: 140, predict: 38 };

function cellStat(name, view, trained, random) {
  const T = trained && trained[name], R = random && random[name];
  if (view === 'random') { const v = R ? finiteNum(R.std) : null; return { v, label: v == null ? 'no value' : `std ${v.toFixed(4)}` }; }
  if (view === 'diff') {
    const tv = T ? finiteNum(T.std) : null, rv = R ? finiteNum(R.std) : null;
    if (tv == null || rv == null) return { v: null, label: 'no value' };
    const d = tv - rv; return { v: Math.abs(d), label: `Δstd ${d >= 0 ? '+' : ''}${d.toFixed(4)}` };
  }
  const v = T ? finiteNum(T.std) : null;
  return { v, label: v == null ? 'no value' : `std ${v.toFixed(4)}` };
}

export function renderKnobs(root, model, view = 'trained') {
  const arch = model.arch || model;
  const trained = model.trained || null, random = model.random || null;
  const hasValues = !!(trained && random);

  let maxParams = 1, maxStat = 0;
  for (const g of arch.groups) for (const t of g.tensors) {
    maxParams = Math.max(maxParams, t.params);
    if (hasValues) { const s = cellStat(t.name, view, trained, random).v; if (s != null) maxStat = Math.max(maxStat, s); }
  }
  const size = (p) => Math.round(13 + (Math.sqrt(p) / Math.sqrt(maxParams)) * 40);   // 13..53px

  root.innerHTML = '';
  root.dataset.view = view;
  const flow = document.createElement('div'); flow.className = 'flow';

  const arrow = () => { const a = document.createElement('div'); a.className = 'flow-arrow'; a.textContent = '→'; flow.appendChild(a); };

  const station = (title, kind, blurb, extraHtml, group) => {
    const s = document.createElement('div'); s.className = `station st-${kind}`;
    const hue = HUE[kind] ?? 210;
    s.style.borderTopColor = `hsl(${hue} 48% 48%)`;
    let knobs = '';
    if (group) {
      const cells = group.tensors.map((t) => {
        const st = hasValues ? cellStat(t.name, view, trained, random) : { v: null, label: 'export a checkpoint to see values' };
        const px = size(t.params);
        const lit = (st.v != null && maxStat > 0) ? 22 + 46 * (st.v / maxStat) : 15;
        return `<span class="knob" title="${esc(t.name)}\n${t.shape.join(' × ')} · ${t.params.toLocaleString()} params\n${esc(t.role || '')}\n${st.label}"`
          + ` style="width:${px}px;height:${px}px;background:hsl(${hue} 45% ${lit}%);border-color:hsl(${hue} 45% ${Math.min(lit + 18, 82)}%)"></span>`;
      }).join('');
      const share = arch.total_params ? `${(100 * group.params / arch.total_params).toFixed(1)}%` : '';
      knobs = `<div class="st-knobs">${cells}</div><div class="st-meta">${group.tensors.length} tensors · ${share}</div>`;
    }
    s.innerHTML = `<div class="st-title">${esc(title)}</div><div class="st-blurb">${esc(blurb)}</div>${extraHtml || ''}${knobs}`;
    flow.appendChild(s);
  };

  // Input — the text being traced, as characters.
  const inputText = (document.getElementById('flowInput')?.value) ?? 'def ';
  const chips = [...inputText].slice(0, 24).map((c) => `<span class="tok">${esc(glyph(c))}</span>`).join('');
  station('Input', 'input', 'Your text arrives as separate characters — one token each.',
    `<div class="st-toks">${chips || '<span class="st-meta">type text above</span>'}</div>`);
  arrow();

  for (const g of arch.groups) {
    let title, blurb;
    if (g.kind === 'embedding') { title = 'Embedding'; blurb = 'Each character becomes a list of 128 numbers, plus where it sits in the text.'; }
    else if (g.kind === 'output') { title = 'Output head'; blurb = 'Turns the final numbers into a score for every possible next character.'; }
    else { const i = g.index, n = arch.config.n_layer; title = `Block ${i}`; blurb = `Look back at earlier characters (attention), then think it over (MLP). Layer ${i + 1} of ${n}.`; }
    station(title, g.kind, blurb, '', g);
    arrow();
  }

  // Prediction — the model's next-character guess.
  station('Next char', 'predict', "The model's best guess for the character that comes next.",
    '<div class="st-toks"><span class="tok pred">?</span></div>');

  root.appendChild(flow);

  // The causal mask is in the model but not on this route — a fixed helper, said once.
  if (arch.buffers && arch.buffers.length) {
    const note = document.createElement('div'); note.className = 'buffer-note';
    note.textContent = `Not on the route: ${arch.buffers[0].name} — the causal mask (${arch.buffers[0].role}). It's fixed, never trained.`;
    root.appendChild(note);
  }
}
