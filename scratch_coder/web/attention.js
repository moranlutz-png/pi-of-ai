// attention.js — every head's attention, no sampling. One small heatmap per
// (layer, head), drawn to <canvas> (a 4×4 grid of T×T heatmaps is thousands of
// cells — that many SVG rects locks the tab).
//
// Each layer is a row of heatmaps with an info panel to its right: a one-line
// description of what every head in that row tends to do, and — when you point at
// any square — a plain-English reading of that exact square. A cell (row i, column
// j) is how much the character at position i looked at the character at position j
// while being processed. The matrix is lower-triangular: a character can only look
// at itself and what came before it.

const glyph = (c) => ({ ' ': '␠', '\n': '↵', '\t': '⇥', '\r': '␍' }[c] ?? c);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const gl = (c) => esc(glyph(c));

// Characterise a head from its attention matrix: does it mostly look at the current
// character, the previous one, the first one, or spread its attention out? Averaged
// over every query position (skipping position 0, which can only see itself).
function describeHead(m, T) {
  if (T < 2) return 'only one character — nothing to compare yet';
  let self = 0, prev = 0, first = 0, dist = 0, n = 0;
  for (let i = 1; i < T; i++) {
    self += m[i][i]; prev += m[i][i - 1]; first += m[i][0];
    let d = 0; for (let j = 0; j <= i; j++) d += (i - j) * m[i][j];
    dist += d; n++;
  }
  self /= n; prev /= n; first /= n; dist /= n;
  const cands = [
    { k: 'the current character', v: self },
    { k: 'the previous character', v: prev },
    { k: 'the first character', v: first },
  ].sort((a, b) => b.v - a.v);
  const top = cands[0];
  if (top.v < 0.34) return `spread out · looks ~${dist.toFixed(1)} characters back on average`;
  return `mostly ${top.k} (${Math.round(top.v * 100)}%)`;
}

export function renderAttention(root, attn, chars) {
  root.innerHTML = '';
  if (!attn || !attn.length) return;
  const L = attn.length, H = attn[0].length, T = attn[0][0].length;
  const cell = Math.max(11, Math.min(30, Math.floor(300 / T)));
  const PAD = Math.round(cell * 0.72) + 3;   // room for the character tick labels

  for (let l = 0; l < L; l++) {
    const row = document.createElement('div'); row.className = 'attnrow';
    const heads = document.createElement('div'); heads.className = 'attnheads';

    // Info panel to the RIGHT of this layer's row of heads.
    const info = document.createElement('div'); info.className = 'attninfo';
    let infoHtml = `<div class="attninfo-h">Layer ${l} · ${H} heads</div>`;
    for (let h = 0; h < H; h++) infoHtml += `<div class="attnhead-line"><b>head ${h}</b> — ${esc(describeHead(attn[l][h], T))}</div>`;
    info.innerHTML = infoHtml;
    const read = document.createElement('div'); read.className = 'attnread';
    read.innerHTML = 'Hover a square to read it in plain English.';
    info.appendChild(read);

    for (let h = 0; h < H; h++) {
      const wrap = document.createElement('div'); wrap.className = 'attncell';
      const lbl = document.createElement('div'); lbl.className = 'attnlbl';
      lbl.textContent = `head ${h}`;
      const cv = document.createElement('canvas');
      cv.width = PAD + T * cell; cv.height = PAD + T * cell;
      const ctx = cv.getContext('2d');
      const m = attn[l][h];

      const draw = (hi = -1, hj = -1) => {
        ctx.clearRect(0, 0, cv.width, cv.height);
        // Character ticks: top row = the character being looked AT (column j),
        // left column = the character doing the looking (row i).
        ctx.font = `${Math.min(cell - 3, 12)}px ui-monospace, monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        for (let k = 0; k < T; k++) {
          ctx.fillStyle = (k === hj) ? '#ffcf6b' : '#8f8f8f';
          ctx.fillText(glyph(chars[k]), PAD + k * cell + cell / 2, PAD / 2);
          ctx.fillStyle = (k === hi) ? '#ffcf6b' : '#8f8f8f';
          ctx.fillText(glyph(chars[k]), PAD / 2, PAD + k * cell + cell / 2);
        }
        for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
          const w = m[i][j];                       // 0..1 within a causal row
          ctx.fillStyle = `hsl(205 70% ${Math.round(8 + w * 72)}%)`;
          ctx.fillRect(PAD + j * cell, PAD + i * cell, cell - 1, cell - 1);
        }
        if (hi >= 0) {
          ctx.strokeStyle = '#ffcf6b'; ctx.lineWidth = 1.5;
          ctx.strokeRect(PAD + hj * cell + 0.5, PAD + hi * cell + 0.5, cell - 2, cell - 2);
        }
      };
      draw();

      cv.onmousemove = (e) => {
        const r = cv.getBoundingClientRect();
        const sc = cv.width / r.width;
        const j = Math.floor(((e.clientX - r.left) * sc - PAD) / cell);
        const i = Math.floor(((e.clientY - r.top) * sc - PAD) / cell);
        if (i >= 0 && i < T && j >= 0 && j <= i) {
          const w = m[i][j];
          draw(i, j);
          read.innerHTML = `While processing <b>“${gl(chars[i])}”</b> (position ${i}), `
            + `<b>head ${h}</b> put <b>${(w * 100).toFixed(0)}%</b> of its attention on `
            + `<b>“${gl(chars[j])}”</b> (position ${j})${i === j ? ' — itself' : ''}.`;
        }
      };
      cv.onmouseleave = () => { draw(); read.innerHTML = 'Hover a square to read it in plain English.'; };

      wrap.appendChild(lbl); wrap.appendChild(cv);
      heads.appendChild(wrap);
    }
    row.appendChild(heads); row.appendChild(info);
    root.appendChild(row);
  }
}
