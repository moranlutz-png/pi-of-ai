// gpt.js — the from-scratch GPT's forward pass, ported straight from model.py and
// kept name-for-name (tok_emb, pos_emb, c_attn, c_proj, ln1/ln2, mlp.c_fc/c_proj,
// ln_f, head) so a student can read the two side by side and check the translation.
// It runs in the browser on plain Float32Arrays; with {collectAttention:true} it
// returns every head's full T×T attention. That is only affordable because the
// model is tiny — 0.84M params, 3.4MB, the whole thing held for one keystroke.
//
// The three parts worth pointing at are their own functions: the causal mask
// (softmax runs only over j<=i), the 1/sqrt(head_dim) scale, and the softmax.

const scale = (headDim) => 1 / Math.sqrt(headDim);              // model.py: 1/math.sqrt(head)

function layernorm(x, w, b) {                                   // nn.LayerNorm, eps 1e-5
  const n = x.length; let m = 0;
  for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
  const inv = 1 / Math.sqrt(v + 1e-5), o = new Float32Array(n);
  for (let i = 0; i < n; i++) o[i] = (x[i] - m) * inv * w[i] + b[i];
  return o;
}
function linear(x, W, b, inD, outD) {                           // nn.Linear: W is [outD, inD]
  const o = new Float32Array(outD);
  for (let r = 0; r < outD; r++) { let s = b ? b[r] : 0; const base = r * inD; for (let i = 0; i < inD; i++) s += x[i] * W[base + i]; o[r] = s; }
  return o;
}
function erf(x) {                                               // for exact GELU (|err|<1.5e-7)
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function gelu(a) { for (let i = 0; i < a.length; i++) { const x = a[i]; a[i] = 0.5 * x * (1 + erf(x / Math.SQRT2)); } return a; }

// Causal softmax: over the first (i+1) scores only; positions j>i stay 0, which is
// exactly what masked_fill(-inf) + softmax produces in model.py.
function softmaxCausal(scores, i) {
  let mx = -Infinity; for (let j = 0; j <= i; j++) if (scores[j] > mx) mx = scores[j];
  let sum = 0; for (let j = 0; j <= i; j++) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
  for (let j = 0; j <= i; j++) scores[j] /= sum;
}

// Standard normal (Box–Muller), for the "add noise" weight-surgery op.
function gaussian() {
  let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class ScratchGPT {
  constructor(cfg, weights) { this.cfg = cfg; this.W = weights; }

  static async load(dir = '.') {
    const inspect = await (await fetch(`${dir}/inspect.json`, { cache: 'no-store' })).json();
    if (!inspect.weights) throw new Error('no weights block in inspect.json — re-export with --weights');
    const resp = await fetch(`${dir}/${inspect.weights.file}`, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`weights.bin not found (${resp.status}) — re-export with python export_inspect.py --weights`);
    const buf = await resp.arrayBuffer();
    if (buf.byteLength !== inspect.weights.byteLength)
      throw new Error(`weights.bin is ${buf.byteLength} bytes, inspect.json expects ${inspect.weights.byteLength} — mismatched export`);
    const all = new Float32Array(buf), W = {};
    for (const t of inspect.weights.tensors) W[t.name] = all.subarray(t.offset, t.offset + t.count);
    const g = new ScratchGPT(inspect.config, W);
    g.vocab = (inspect.embedding && inspect.embedding.points) ? inspect.embedding.points.map((p) => p.char) : null;
    // For weight surgery: every W[name] is a view into `all`, so editing them edits the
    // live forward pass. Keep the whole buffer and a pristine copy so restore() is exact.
    g._all = all; g._orig = Float32Array.from(all); g.edited = false;
    return g;
  }

  // --- weight surgery ---------------------------------------------------------
  // Change the loaded weights IN MEMORY so generation and attention react at once.
  // Nothing here writes a file — restore() copies the pristine buffer back. The CLI
  // twin that DOES persist is edit_weights.py.
  _names(target) {
    const all = Object.keys(this.W);
    if (target === 'all') return all;
    if (target === 'embedding') return all.filter((n) => n.startsWith('tok_emb') || n.startsWith('pos_emb'));
    if (target === 'output') return all.filter((n) => n.startsWith('ln_f') || n.startsWith('head'));
    if (target.startsWith('block:')) { const l = target.slice(6); return all.filter((n) => n.startsWith(`blocks.${l}.`)); }
    return [];
  }
  _std(a) { let m = 0; for (let i = 0; i < a.length; i++) m += a[i]; m /= a.length;
    let v = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - m; v += d * d; } return Math.sqrt(v / a.length); }
  addNoise(target, sigma) {   // add sigma * (this tensor's std) * N(0,1) to each weight
    for (const n of this._names(target)) { const a = this.W[n], s = sigma * this._std(a); for (let i = 0; i < a.length; i++) a[i] += s * gaussian(); }
    this.edited = true;
  }
  scaleW(target, factor) { for (const n of this._names(target)) { const a = this.W[n]; for (let i = 0; i < a.length; i++) a[i] *= factor; } this.edited = true; }
  ablate(target) {
    // A block: zero its two output projections so it adds nothing to the residual
    // stream — an identity pass-through, i.e. "remove this layer" without a reshape.
    // Anything else: hard-zero every weight in it.
    if (target.startsWith('block:')) {
      const l = target.slice(6);
      for (const nm of [`blocks.${l}.attn.c_proj.weight`, `blocks.${l}.attn.c_proj.bias`,
                        `blocks.${l}.mlp.c_proj.weight`, `blocks.${l}.mlp.c_proj.bias`]) if (this.W[nm]) this.W[nm].fill(0);
    } else { for (const n of this._names(target)) this.W[n].fill(0); }
    this.edited = true;
  }
  restore() { this._all.set(this._orig); this.edited = false; }

  encode(text) {
    if (!this.vocab) throw new Error('no vocab in inspect.json');
    const stoi = {}; this.vocab.forEach((c, i) => { stoi[c] = i; });
    return [...text].map((c) => (c in stoi ? stoi[c] : 0)).slice(0, this.cfg.block_size);
  }

  // Whole prompt at once. collectAttention -> attn[layer][head][T][T], masked at 0.
  forward(ids, { collectAttention = false } = {}) {
    const { n_embd: C, n_head: H, n_layer: L, vocab_size: V } = this.cfg;
    const hd = C / H, T = ids.length, W = this.W, sc = scale(hd);
    const te = W['tok_emb.weight'], pe = W['pos_emb.weight'];

    const x = [];
    for (let t = 0; t < T; t++) { const row = new Float32Array(C); for (let i = 0; i < C; i++) row[i] = te[ids[t] * C + i] + pe[t * C + i]; x.push(row); }

    const attn = collectAttention ? [] : null;
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}.`;
      const q = [], k = [], v = [];
      for (let t = 0; t < T; t++) {
        const h = layernorm(x[t], W[p + 'ln1.weight'], W[p + 'ln1.bias']);
        const qkv = linear(h, W[p + 'attn.c_attn.weight'], W[p + 'attn.c_attn.bias'], C, 3 * C);
        q.push(qkv.subarray(0, C)); k.push(qkv.subarray(C, 2 * C)); v.push(qkv.subarray(2 * C, 3 * C));
      }
      const y = []; for (let t = 0; t < T; t++) y.push(new Float32Array(C));
      const layerAttn = collectAttention ? [] : null;
      for (let hh = 0; hh < H; hh++) {
        const off = hh * hd, headMat = collectAttention ? [] : null;
        for (let i = 0; i < T; i++) {
          const scores = new Float32Array(T);
          for (let j = 0; j <= i; j++) { let s = 0; for (let d = 0; d < hd; d++) s += q[i][off + d] * k[j][off + d]; scores[j] = s * sc; }
          softmaxCausal(scores, i);
          for (let j = 0; j <= i; j++) { const w = scores[j]; for (let d = 0; d < hd; d++) y[i][off + d] += w * v[j][off + d]; }
          if (collectAttention) headMat.push(Array.from(scores));
        }
        if (collectAttention) layerAttn.push(headMat);
      }
      if (collectAttention) attn.push(layerAttn);
      for (let t = 0; t < T; t++) { const ao = linear(y[t], W[p + 'attn.c_proj.weight'], W[p + 'attn.c_proj.bias'], C, C); for (let i = 0; i < C; i++) x[t][i] += ao[i]; }
      for (let t = 0; t < T; t++) {
        const h2 = layernorm(x[t], W[p + 'ln2.weight'], W[p + 'ln2.bias']);
        const m1 = gelu(linear(h2, W[p + 'mlp.c_fc.weight'], W[p + 'mlp.c_fc.bias'], C, 4 * C));
        const m2 = linear(m1, W[p + 'mlp.c_proj.weight'], W[p + 'mlp.c_proj.bias'], 4 * C, C);
        for (let i = 0; i < C; i++) x[t][i] += m2[i];
      }
    }
    const xf = layernorm(x[T - 1], W['ln_f.weight'], W['ln_f.bias']);
    return { logits: linear(xf, W['head.weight'], null, C, V), attn };
  }

  decode(ids) { return ids.map((i) => this.vocab[i]).join(''); }

  // Draw one next-token id from the final logits — the browser twin of sample_big.py,
  // with a few standard quality knobs on top of temperature:
  //   topK          — only consider the K most likely characters
  //   topP           — nucleus: keep the smallest set of characters covering P of the
  //                    probability mass (cuts the long tail without a hard K)
  //   repeatPenalty  — divide the logit of any character in `recent`, to discourage the
  //                    low-temperature loops; pass `recent` already stripped of
  //                    whitespace so indentation and newlines are never penalised.
  // exp((l−max)/t) is softmax(l/t) with the max subtracted for numerical stability.
  sampleNext(logits, temperature = 0.8, topK = 0, { topP = 1, repeatPenalty = 1, recent = null } = {}) {
    const V = logits.length, t = Math.max(1e-6, temperature);
    const L = Float32Array.from(logits);              // copy — never mutate the caller's logits
    if (repeatPenalty !== 1 && recent) {
      for (const id of new Set(recent)) L[id] = L[id] > 0 ? L[id] / repeatPenalty : L[id] * repeatPenalty;
    }
    const pool = (topK > 0 && topK < V)
      ? Array.from({ length: V }, (_, i) => i).sort((a, b) => L[b] - L[a]).slice(0, topK)
      : Array.from({ length: V }, (_, i) => i);
    let mx = -Infinity; for (const i of pool) if (L[i] > mx) mx = L[i];
    let sum = 0; const ps = new Float64Array(pool.length);
    for (let k = 0; k < pool.length; k++) { const e = Math.exp((L[pool[k]] - mx) / t); ps[k] = e; sum += e; }
    for (let k = 0; k < ps.length; k++) ps[k] /= sum;
    if (topP < 1) {                                   // nucleus filter over the pool
      const ord = Array.from({ length: pool.length }, (_, k) => k).sort((a, b) => ps[b] - ps[a]);
      let cum = 0, n = ord.length;
      for (let k = 0; k < ord.length; k++) { cum += ps[ord[k]]; if (cum >= topP) { n = k + 1; break; } }
      const keep = ord.slice(0, n);
      let ks = 0; for (const k of keep) ks += ps[k];
      let r = Math.random() * ks;
      for (const k of keep) { r -= ps[k]; if (r <= 0) return pool[k]; }
      return pool[keep[keep.length - 1]];
    }
    let r = Math.random();
    for (let k = 0; k < pool.length; k++) { r -= ps[k]; if (r <= 0) return pool[k]; }
    return pool[pool.length - 1];
  }
}
