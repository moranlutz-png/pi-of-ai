// embedding.js — the token embedding projected to 2D, drawn as the characters
// themselves. The lesson is *which characters cluster*, so the glyphs go where the
// dots would (a legend mapping 101 dots to 101 characters is a worse version of
// that). The neighbour list is computed in the FULL space, not the projection, so
// it is the check on the picture: when the plot and the list disagree, the list is
// right, and that disagreement is worth showing rather than hiding.

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');

// Whitespace has no visible glyph; show a stand-in so a cluster of spaces/newlines
// (which is exactly what a code model learns first) is legible.
function glyphLabel(ch) {
  return { ' ': '␠', '\n': '↵', '\t': '⇥', '\r': '␍' }[ch] ?? ch;
}

export function renderEmbedding(root, emb) {
  const svg = root.querySelector('#embed');
  const varEl = root.querySelector('#embedVar');
  const nbrEl = root.querySelector('#embedNeighbours');
  const pts = emb.points, nbrs = emb.neighbours || {};

  const W = 460, H = 460, pad = 26;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (v) => pad + (xmax > xmin ? (v - xmin) / (xmax - xmin) : 0.5) * (W - 2 * pad);
  const sy = (v) => pad + (ymax > ymin ? (v - ymin) / (ymax - ymin) : 0.5) * (H - 2 * pad);

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = pts.map((p) => {
    const nb = (nbrs[p.char] || []).map((n) => glyphLabel(n.char)).join(' ');
    return `<text x="${sx(p.x).toFixed(1)}" y="${sy(p.y).toFixed(1)}" class="glyph"`
      + ` data-char="${esc(p.char)}" data-neighbours="${esc(nb)}">${esc(glyphLabel(p.char))}</text>`;
  }).join('');

  varEl.textContent = `these two directions carry ${emb.varianceExplainedPct}% of the variation`
    + ' — two components of a 128-dimensional space, so most of it is not in this picture.';

  function showNeighbours(ch) {
    const list = nbrs[ch] || [];
    nbrEl.innerHTML = `<div class="nbrhead">nearest to <b>${esc(glyphLabel(ch))}</b><br>(cosine, full space)</div>`
      + list.map((n) => `<div class="nbrrow"><span class="nbrch">${esc(glyphLabel(n.char))}</span>`
        + `<span class="nbrcos">${Number(n.cos).toFixed(3)}</span></div>`).join('');
  }

  svg.querySelectorAll('.glyph').forEach((g) => {
    const pick = () => {
      svg.querySelectorAll('.glyph').forEach((x) => x.classList.remove('sel'));
      g.classList.add('sel');
      showNeighbours(g.dataset.char);
    };
    g.addEventListener('mouseenter', () => showNeighbours(g.dataset.char));
    g.addEventListener('click', pick);
  });

  // Open on a letter if present, so the panel is never empty.
  const seed = pts.find((p) => p.char === 'e') || pts[0];
  if (seed) showNeighbours(seed.char);
}
