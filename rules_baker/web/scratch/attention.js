// attention.js — every head's attention, no sampling. One small heatmap per
// (layer, head), all on one screen, drawn to <canvas> (a 4×4 grid of T×T heatmaps
// is thousands of cells — that many SVG rects locks the tab).
//
// To make them readable rather than cryptic, each heatmap now carries the actual
// characters along both axes, and pointing at any square reads it out in plain
// words. A cell (row i, column j) is how much the character at position i looked at
// the character at position j while the model was processing it. The matrix is
// lower-triangular — a character can only look at itself and what came before it.

const glyph = (c) => ({ ' ': '␠', '\n': '↵', '\t': '⇥', '\r': '␍' }[c] ?? c);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const gl = (c) => esc(glyph(c));

export function renderAttention(root, attn, chars) {
  root.innerHTML = '';
  if (!attn || !attn.length) return;
  const L = attn.length, H = attn[0].length, T = attn[0][0].length;
  const cell = Math.max(11, Math.min(30, Math.floor(300 / T)));
  const PAD = Math.round(cell * 0.72) + 3;   // room for the character tick labels

  // A shared caption that reads out whatever square you point at, in plain words.
  const cap = document.createElement('div'); cap.className = 'attncap';
  cap.innerHTML = 'Point at any square below to read it in plain English.';
  root.appendChild(cap);

  for (let l = 0; l < L; l++) {
    const row = document.createElement('div'); row.className = 'attnrow';
    for (let h = 0; h < H; h++) {
      const wrap = document.createElement('div'); wrap.className = 'attncell';
      const lbl = document.createElement('div'); lbl.className = 'attnlbl';
      lbl.textContent = `layer ${l} · head ${h}`;
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
          cap.innerHTML = `In <b>layer ${l}, head ${h}</b>: while processing <b>“${gl(chars[i])}”</b> `
            + `(position ${i}), the model put <b>${(w * 100).toFixed(0)}%</b> of this head's attention on `
            + `<b>“${gl(chars[j])}”</b> (position ${j})${i === j ? ' — itself' : ''}.`;
        }
      };
      cv.onmouseleave = () => { draw(); cap.innerHTML = 'Point at any square below to read it in plain English.'; };

      wrap.appendChild(lbl); wrap.appendChild(cv);
      row.appendChild(wrap);
    }
    root.appendChild(row);
  }
}
