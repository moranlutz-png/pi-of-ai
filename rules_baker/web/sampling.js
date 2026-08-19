/* sampling.js — what the sampler does to the model's opinion.
 *
 * wllama's getLogits() hands back the softmax of the raw logits: the model's
 * unmodified distribution over the next token. Temperature, top-k and top-p are
 * applied AFTER that, inside llama.cpp, and are never visible.
 *
 * That invisibility is the thing worth teaching. Re-implementing the three
 * transforms here — as pure functions over a plain array — lets the UI show the
 * before and the after side by side, so a temperature slider visibly moves one
 * column and not the other.
 *
 * These are the same operations llama.cpp performs, in the same order it
 * performs them (temperature, then top-k, then top-p). They are not a
 * reimplementation of the sampler used for real generation — that stays inside
 * WASM. This is a mirror, for showing the work.
 */

export const SAMPLING_DEFAULTS = Object.freeze({ temperature: 0.2, topK: 40, topP: 0.9 });

/** Temperature rescales the logits before the softmax. We only have post-softmax
 *  probabilities, so recover the logits with log(p), divide, and re-normalise —
 *  mathematically identical for our purposes and avoids needing the raw logits. */
export function applyTemperature(dist, temperature) {
  const t = Number(temperature);
  if (!isFinite(t) || t <= 0) {
    // Temperature 0 is greedy: all mass on the single most likely token.
    const best = dist.reduce((a, b) => (b.p > a.p ? b : a), dist[0]);
    return dist.map(d => ({ ...d, p: d === best ? 1 : 0 }));
  }
  if (t === 1) return dist.map(d => ({ ...d }));
  const scaled = dist.map(d => ({ ...d, _l: Math.log(Math.max(d.p, 1e-12)) / t }));
  // Not Math.max(...scaled.map(...)): spreading into a function call passes
  // every element as a positional argument, and engines throw
  // "RangeError: Maximum call stack size exceeded" once the array gets into
  // the tens of thousands — well within real vocabulary sizes (32k-152k+).
  // reduce() has no such limit.
  const max = scaled.reduce((m, d) => Math.max(m, d._l), -Infinity);
  const exps = scaled.map(d => ({ ...d, _e: Math.exp(d._l - max) }));
  const sum = exps.reduce((s, d) => s + d._e, 0) || 1;
  return exps.map(({ _l, _e, ...rest }) => ({ ...rest, p: _e / sum }));
}

/** Keep the k most likely, drop the rest, re-normalise. */
export function applyTopK(dist, k) {
  const n = parseInt(k, 10);
  if (!isFinite(n) || n <= 0 || n >= dist.length) return dist.map(d => ({ ...d }));
  const kept = [...dist].sort((a, b) => b.p - a.p).slice(0, n);
  return renormalise(kept);
}

/** Keep the smallest set whose probabilities sum to p, re-normalise. */
export function applyTopP(dist, p) {
  const target = Number(p);
  if (!isFinite(target) || target <= 0 || target >= 1) return dist.map(d => ({ ...d }));
  const sorted = [...dist].sort((a, b) => b.p - a.p);
  const kept = [];
  let cum = 0;
  for (const d of sorted) {
    kept.push(d);
    cum += d.p;
    if (cum >= target) break;        // inclusive: the token that crosses stays
  }
  return renormalise(kept);
}

function renormalise(dist) {
  const sum = dist.reduce((s, d) => s + d.p, 0) || 1;
  return dist.map(d => ({ ...d, p: d.p / sum }));
}

/** All three, in llama.cpp's order. Returns a NEW array, sorted most likely
 *  first, so a caller can render it beside the untouched original. */
export function applySampling(dist, opts = {}) {
  if (!Array.isArray(dist) || !dist.length) return [];
  const { temperature, topK, topP } = { ...SAMPLING_DEFAULTS, ...opts };
  let out = applyTemperature(dist, temperature);
  out = applyTopK(out, topK);
  out = applyTopP(out, topP);
  return out.sort((a, b) => b.p - a.p);
}
