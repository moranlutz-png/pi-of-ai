// attention.js — every head's attention, no sampling. One small heatmap per
// (layer, head), all of them on one screen. Rendered to <canvas>: a 4×4 grid of
// T×T heatmaps is thousands of cells, and that many SVG rects locks the tab.
//
// A cell (i, j) is how much position i attended to position j. The matrix is
// lower-triangular — a token can only look at itself and what came before.

export function renderAttention(root, attn, chars) {
  root.innerHTML = '';
  if (!attn || !attn.length) return;
  const L = attn.length, H = attn[0].length, T = attn[0][0].length;
  const cell = Math.max(5, Math.min(22, Math.floor(200 / T)));

  for (let l = 0; l < L; l++) {
    const row = document.createElement('div'); row.className = 'attnrow';
    for (let h = 0; h < H; h++) {
      const wrap = document.createElement('div'); wrap.className = 'attncell';
      const lbl = document.createElement('div'); lbl.className = 'attnlbl';
      lbl.textContent = `L${l}·H${h}`;
      const cv = document.createElement('canvas');
      cv.width = T * cell; cv.height = T * cell;
      cv.title = `layer ${l}, head ${h} — row = query position, column = attended-to position`;
      const ctx = cv.getContext('2d');
      const m = attn[l][h];
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const w = m[i][j];                       // 0..1 within a causal row
        ctx.fillStyle = `hsl(205 65% ${Math.round(7 + w * 74)}%)`;
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
      wrap.appendChild(lbl); wrap.appendChild(cv);
      row.appendChild(wrap);
    }
    root.appendChild(row);
  }
}
