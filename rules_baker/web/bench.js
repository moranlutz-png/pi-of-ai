/* bench.js — what this model actually does on THIS machine.
 *
 * The published numbers for a small model are measured on hardware nobody in a
 * classroom owns. The question that matters to a student on a Chromebook is not
 * "how fast is Qwen 0.5B" but "how fast is it here, on this, right now" — and
 * whether it stays that fast after two minutes.
 *
 * Four things worth measuring, in rising order of how often they are ignored:
 *
 *   TTFT      time to first token — the wait before anything appears, which is
 *             what the user actually experiences as "slow".
 *   DECODE    sustained tokens per second once it is going.
 *   THERMAL   whether decode DECAYS across repeated runs. Laptops and phones
 *             throttle under sustained load; a single short benchmark hides it
 *             completely, which is why published figures flatter thin hardware.
 *   MEMORY    heap headroom, where the browser will tell us.
 *
 * The thermal figure is the reason this file exists. Everything else is already
 * visible in the status line.
 */

const KEY = 'pi-of-ai:bench';

// Deliberately dull and short. A benchmark prompt that invites a long creative
// answer measures the sampler's mood as much as the hardware.
export const BENCH_PROMPT = 'Count from one to twenty, separated by commas.';

function read() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '{}');
    return (raw && typeof raw === 'object' && !Array.isArray(raw)) ? raw : {};
  } catch (_) { return {}; }
}

function write(all) {
  try { localStorage.setItem(KEY, JSON.stringify(all)); return true; }
  catch (_) { return false; }
}

export function savedRuns() { return read(); }

export function clearRuns() {
  try { localStorage.removeItem(KEY); } catch (_) {}
}

/** Coarse device fingerprint, so results from a phone and a workstation don't
 *  get averaged into a number describing neither. */
export function deviceKey() {
  const n = navigator;
  return [
    (n.userAgentData && n.userAgentData.platform) || n.platform || 'unknown',
    (n.hardwareConcurrency || '?') + 'c',
    (n.deviceMemory ? n.deviceMemory + 'gb' : 'mem?'),
  ].join('/');
}

/** Heap headroom, where the browser reports it. Chrome-only, and it measures
 *  the JS heap rather than WASM memory — so it is a hint, not an accounting. */
export function memorySnapshot() {
  const m = performance.memory;
  if (!m) return null;
  return {
    usedMb: Math.round(m.usedJSHeapSize / 1048576),
    limitMb: Math.round(m.jsHeapSizeLimit / 1048576),
    headroomPct: Math.round(100 * (1 - m.usedJSHeapSize / m.jsHeapSizeLimit)),
  };
}

/**
 * Run the benchmark.
 *
 * `runOnce` must be a function the caller supplies that performs ONE generation
 * and reports timings — bench.js deliberately knows nothing about wllama or
 * Ollama, so the same benchmark measures both without special-casing either.
 * It should resolve to { ttftMs, tokens, totalMs }.
 *
 * `rounds` defaults to 3 because one round cannot show thermal decay and five
 * takes long enough on a Chromebook that nobody waits for it.
 */
export async function runBenchmark({ runOnce, rounds = 3, onProgress } = {}) {
  const results = [];
  const memBefore = memorySnapshot();

  for (let i = 0; i < rounds; i++) {
    if (onProgress) onProgress({ round: i + 1, rounds });
    const r = await runOnce();
    const secs = r.totalMs / 1000;
    results.push({
      round: i + 1,
      ttftMs: Math.round(r.ttftMs),
      tokens: r.tokens,
      tokensPerSecond: secs > 0 ? Math.round((r.tokens / secs) * 10) / 10 : 0,
    });
  }

  const rates = results.map(r => r.tokensPerSecond).filter(x => x > 0);
  const first = rates[0] || 0;
  const last = rates[rates.length - 1] || 0;
  // Positive means it got SLOWER over the run. Below ~5% is noise on any real
  // machine; above ~15% something is thermally or power limited.
  const decayPct = first > 0 ? Math.round(((first - last) / first) * 1000) / 10 : 0;

  return {
    at: new Date().toISOString(),
    device: deviceKey(),
    rounds: results,
    bestTokensPerSecond: rates.length ? Math.max(...rates) : 0,
    medianTtftMs: median(results.map(r => r.ttftMs)),
    decayPct,
    throttling: decayPct >= 15 ? 'likely' : decayPct >= 5 ? 'possible' : 'none detected',
    memoryBefore: memBefore,
    memoryAfter: memorySnapshot(),
  };
}

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : Math.round((s[m - 1] + s[m]) / 2);
}

/** Keep the newest result per model+device pair. */
export function saveRun(modelKey, result) {
  const all = read();
  all[`${modelKey}@@${result.device}`] = result;
  write(all);
  return result;
}
