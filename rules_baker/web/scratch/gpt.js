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
    return g;
  }

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
}
