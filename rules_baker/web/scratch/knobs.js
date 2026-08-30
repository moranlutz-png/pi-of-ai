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

// The last traced character's real embedding, drawn as a strip of numbers: purple = positive,
// blue = negative, brightness = magnitude. Makes "becomes a list of numbers" something you see.
function embStrip(vec) {
  const N = 24, step = vec.length / N;
  let mx = 1e-6; for (const v of vec) mx = Math.max(mx, Math.abs(v));
  let cells = '';
  for (let i = 0; i < N; i++) {
    const v = vec[Math.floor(i * step)] || 0, a = Math.min(1, Math.abs(v) / mx);
    const col = v >= 0 ? `rgba(150,130,240,${(0.12 + 0.82 * a).toFixed(2)})` : `rgba(90,160,240,${(0.12 + 0.82 * a).toFixed(2)})`;
    cells += `<span class="emb-cell" style="background:${col}"></span>`;
  }
  return cells;
}

export function renderKnobs(root, model, view = 'trained', trace = null) {
  const arch = model.arch || model;
  const trained = model.trained || null, random = model.random || null;
  const hasValues = !!(trained && random);

  let maxParams = 1, maxStat = 0;
  for (const g of arch.groups) for (const t of g.tensors) {
    maxParams = Math.max(maxParams, t.params);
    if (hasValues) { const s = cellStat(t.name, view, trained, random).v; if (s != null) maxStat = Math.max(maxStat, s); }
  }
  const size = (p) => Math.round(8 + (Math.sqrt(p) / Math.sqrt(maxParams)) * 18);   // 8..26px — secondary to the data strip

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
      knobs = `<div class="st-knobs">${cells}</div><div class="st-meta">weights · ${share} of the model</div>`;
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
    let title, blurb, extra = '';
    if (g.kind === 'embedding') { title = 'Embedding'; blurb = `Each character becomes a list of ${arch.config.n_embd} numbers, plus where it sits in the text.`;
      if (trace && trace.embVec) extra = `<div class="emb-demo"><span class="tok">${esc(glyph(trace.lastChar))}</span><span class="emb-arrow">→</span><span class="emb-strip" title="the real embedding of “${esc(glyph(trace.lastChar))}”">${embStrip(trace.embVec)}</span></div>`; }
    else if (g.kind === 'output') { title = 'Output head'; blurb = 'Turns the final numbers into a score for every possible next character.'; }
    else { const i = g.index, n = arch.config.n_layer; title = `Block ${i}`; blurb = `Look back at earlier characters (attention), then think it over (MLP). Layer ${i + 1} of ${n}.`;
      if (trace && trace.blockVecs && trace.blockVecs[i]) extra = `<div class="emb-demo"><span class="emb-lbl">so far</span><span class="emb-strip" title="the running vector after block ${i}">${embStrip(trace.blockVecs[i])}</span></div>`; }
    station(title, g.kind, blurb, extra, g);
    arrow();
  }

  // Prediction — the model's real next-character guess for the traced text.
  let predExtra = '<div class="st-toks"><span class="tok pred">?</span></div>';
  if (trace && trace.top && trace.top.length) {
    const top0 = trace.top[0].p || 1;
    predExtra = '<div class="pred-list">' + trace.top.slice(0, 4).map((o, i) =>
      `<div class="pred-row${i === 0 ? ' top' : ''}"><span class="tok pred">${esc(glyph(o.char))}</span>`
      + `<span class="pred-bar"><span class="pred-fill" style="width:${Math.round((o.p / top0) * 100)}%"></span></span>`
      + `<span class="pred-p">${Math.round(o.p * 100)}%</span></div>`).join('') + '</div>';
  }
  station('Next char', 'predict', 'The model reads all of that and guesses what comes next.', predExtra);

  root.appendChild(flow);

  // The causal mask is in the model but not on this route — a fixed helper, said once.
  if (arch.buffers && arch.buffers.length) {
    const note = document.createElement('div'); note.className = 'buffer-note';
    note.textContent = `Not on the route: ${arch.buffers[0].name} — the causal mask (${arch.buffers[0].role}). It's fixed, never trained.`;
    root.appendChild(note);
  }
}
