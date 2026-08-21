// embedding.js — the token embedding as a 3D graph you can orbit. Each character is
// a node placed where its learned embedding lands (projected to 3D), and lines join
// each character to its nearest neighbours in the FULL space — the connections the
// model has learned between characters. Drag to look around, scroll to zoom.
//
// Hand-rolled on <canvas>, no three.js: this build has no dependencies and must run
// offline on a locked-down Chromebook, and a from-scratch project drawing its own 3D
// is rather the point. 101 nodes and a few hundred edges is nothing for a canvas.

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const glyph = (c) => ({ ' ': '␠', '\n': '↵', '\t': '⇥', '\r': '␍' }[c] ?? c);
const avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export function renderEmbedding(root, emb) {
  const canvas = root.querySelector('#embed');
  const varEl = root.querySelector('#embedVar');
  const nbrEl = root.querySelector('#embedNeighbours');
  const pts = emb.points, nbrs = emb.neighbours || {}, ideal = emb.idealNeighbours || {};

  // Centre the cloud on the origin and scale it to radius ~1.
  const cx = avg(pts.map((p) => p.x)), cy = avg(pts.map((p) => p.y)), cz = avg(pts.map((p) => p.z ?? 0));
  const P = pts.map((p) => ({ char: p.char, v: [p.x - cx, p.y - cy, (p.z ?? 0) - cz] }));
  // Normalise by the FARTHEST node so radius 1 is the outermost point. Paired with
  // the projection scale below, that guarantees every node stays inside the canvas
  // through the entire spin — the graph never clips off the edge as it tumbles.
  let norm = 1e-6;
  for (const p of P) norm = Math.max(norm, Math.hypot(...p.v));
  P.forEach((p) => { p.v = p.v.map((c) => c / norm); });
  const idxOf = {}; P.forEach((p, i) => { idxOf[p.char] = i; });

  // Edges: each character to each of its neighbours, deduped.
  const seen = new Set(), edges = [];
  for (const p of P) for (const n of (nbrs[p.char] || [])) {
    const j = idxOf[n.char]; if (j == null) continue;
    const a = Math.min(idxOf[p.char], j), b = Math.max(idxOf[p.char], j), k = a + '-' + b;
    if (!seen.has(k)) { seen.add(k); edges.push([a, b, Number(n.cos)]); }   // cosine is symmetric
  }
  let cosMin = Infinity, cosMax = -Infinity;
  for (const e of edges) { cosMin = Math.min(cosMin, e[2]); cosMax = Math.max(cosMax, e[2]); }
  const cosRange = (cosMax - cosMin) || 1;

  varEl.textContent = `these three directions carry ${emb.varianceExplainedPct}% of the variation — drag to orbit, scroll to zoom`;

  // The canvas fills its box (the full-width graph section); we render at the box's
  // actual pixel size so the graph is as large as the space allows, and re-fit on
  // resize. Scale is tied to the smaller side so the graph fills the height and never
  // clips as it spins — a round point cloud leaves some horizontal margin by nature.
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ctx = canvas.getContext('2d');
  const DIST = 3.4;
  let W = 640, H = 640, CX = 320, CY = 320, FOV = 909;
  function size() {
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    CX = W / 2; CY = H / 2;
    // Fill the box: scale so the outermost node's orbit reaches ~1px from the nearer
    // edge. A radius-1 node's peak projected offset over all rotations is 0.3078·FOV
    // (the max of √(1−z²)/(DIST−z) at z=1/DIST), so this puts that peak at min/2 − 1.
    FOV = (Math.min(W, H) / 2 - 1) / 0.3078;
  }
  size();
  window.addEventListener('resize', size);

  let rotX = -0.35, rotY = 0.6, zoom = 1, dragging = false, lastX = 0, lastY = 0, autoRotate = true, hover = -1, byStrength = true;

  function rot([x, y, z]) {
    const cy1 = Math.cos(rotY), sy1 = Math.sin(rotY);
    const x1 = x * cy1 + z * sy1, z1 = -x * sy1 + z * cy1;
    const cx1 = Math.cos(rotX), sx1 = Math.sin(rotX);
    return [x1, y * cx1 - z1 * sx1, y * sx1 + z1 * cx1];
  }
  function project(v) { const r = rot(v); const s = (FOV * zoom) / (DIST - r[2]); return { x: CX + r[0] * s, y: CY - r[1] * s, depth: r[2] }; }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const pr = P.map((p) => project(p.v));
    for (const [a, b, cos] of edges) {
      const pa = pr[a], pb = pr[b], t = ((pa.depth + pb.depth) / 2 + 1) / 2;
      const hot = hover === a || hover === b;
      let alpha, w;
      if (byStrength) { const nc = (cos - cosMin) / cosRange; alpha = (0.03 + 0.55 * nc) * (0.45 + 0.55 * t); w = 0.5 + 1.3 * nc; }
      else { alpha = 0.05 + 0.14 * t; w = 0.6; }
      ctx.strokeStyle = hot ? 'rgba(120,180,255,.6)' : `rgba(200,210,230,${alpha})`;
      ctx.lineWidth = hot ? 1.4 : w;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }
    const order = P.map((_, i) => i).sort((i, j) => pr[i].depth - pr[j].depth);
    for (const i of order) {
      const p = pr[i], t = (p.depth + 1) / 2, hot = hover === i;
      ctx.globalAlpha = hot ? 1 : 0.32 + 0.6 * t;
      ctx.fillStyle = hot ? '#cfe4ff' : '#e9e9e9';
      ctx.font = `${hot ? 700 : 500} ${8 + 8 * t}px ui-monospace, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(glyph(P[i].char), p.x, p.y);
      ctx.globalAlpha = 1;
    }
    canvas.__pr = pr;
  }

  // Idle spin: a touch faster, and tumbling across two axes (Y faster than X) so
  // you see the graph from every side rather than just spinning on the spot.
  function loop() { if (autoRotate && !dragging) { rotY += 0.0034; rotX += 0.0013; } draw(); root.__embRaf = requestAnimationFrame(loop); }
  cancelAnimationFrame(root.__embRaf); loop();

  canvas.onmousedown = (e) => { dragging = true; autoRotate = false; lastX = e.clientX; lastY = e.clientY; };
  window.addEventListener('mouseup', () => { dragging = false; });
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    if (dragging) { rotY += (e.clientX - lastX) * 0.01; rotX += (e.clientY - lastY) * 0.01; lastX = e.clientX; lastY = e.clientY; return; }
    const sc = W / (rect.width || W);
    const mx = (e.clientX - rect.left) * sc, my = (e.clientY - rect.top) * sc, pr = canvas.__pr || [];
    let best = -1, bd = 256;
    for (let i = 0; i < pr.length; i++) { const dx = pr[i].x - mx, dy = pr[i].y - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
    if (best !== hover) { hover = best; if (best >= 0) showNeighbours(P[best].char); }
  };
  canvas.onwheel = (e) => { e.preventDefault(); zoom = Math.max(0.4, Math.min(4, zoom * (e.deltaY < 0 ? 1.1 : 0.9))); };

  const strengthToggle = document.getElementById('embByStrength');
  if (strengthToggle) { byStrength = strengthToggle.checked; strengthToggle.onchange = () => { byStrength = strengthToggle.checked; }; }

  // Two columns: what the model has learned so far, and what a perfectly trained
  // model should recover — the corpus's own context structure, exported alongside.
  function showNeighbours(ch) {
    // The ideal set for this character — the neighbours a perfect model should find.
    // A model-now neighbour that is also in this set is one the model already got
    // right, so its character and value turn green (correctness, not decoration).
    const idealList = ideal[ch] || [];
    const idealChars = new Set(idealList.map((n) => n.char));
    // Fade each row by its strength within its own column — brightest is the closest
    // neighbour, dimming down the list — the same brighter-is-stronger cue as the lines.
    // A correct match overrides the fade to full strength so it stands out.
    const rows = (list, matchSet) => {
      const arr = list || [], cs = arr.map((n) => Number(n.cos));
      const lo = Math.min(...cs), rng = (Math.max(...cs) - lo) || 1;
      return arr.map((n) => {
        const hit = matchSet && matchSet.has(n.char);
        const op = hit ? '1' : (0.4 + 0.6 * ((Number(n.cos) - lo) / rng)).toFixed(3);
        return `<div class="nbrrow${hit ? ' match' : ''}" style="opacity:${op}"><span class="nbrch">${esc(glyph(n.char))}</span><span class="nbrcos">${Number(n.cos).toFixed(3)}</span></div>`;
      }).join('');
    };
    nbrEl.innerHTML = `<div class="nbrhead">nearest to <b>${esc(glyph(ch))}</b> · cosine, full space</div>`
      + `<div class="nbrcompare">`
      +   `<div class="nbrcol"><div class="nbrcolhead">model now</div>${rows(nbrs[ch], idealChars)}</div>`
      +   `<div class="nbrcol ideal"><div class="nbrcolhead">ideal</div>${rows(idealList)}</div>`
      + `</div>`;
  }
  const seed = P.find((p) => p.char === 'e') || P[0];
  if (seed) showNeighbours(seed.char);
}
