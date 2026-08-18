# Observability Tier 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app show what the model is actually doing — the candidate tokens behind each choice, what the sampler does to them, how text becomes tokens — and stop it stating things about itself that are untrue.

**Architecture:** Two new pure modules (`sampling.js`, `gguf.js`) that are fully verifiable in the console with no model loaded, plus wiring in `index.html`. Nothing forks wllama; everything uses its published API.

**Tech Stack:** Vanilla ES modules, no build step, no dependencies. wllama 2.x.

**Spec:** `docs/specs/2026-08-18-observability-and-honesty-design.md`

## Global Constraints

- **No build step.** `index.html` loads one `<script type="module">`; new modules are imported from it.
- **No test framework.** Verification is in-browser against `http://localhost:8123`, run from the console. Start it from the checkout you are editing: `cd rules_baker/web && python3 serve.py 8123`. A server rooted in a *different* checkout will pass every check against code you did not write.
- **Every value interpolated into `innerHTML` goes through the existing `esc()` helper**, which escapes quotes as well as angle brackets.
- **`#output` is a transcript.** Never assign to it; append, or clear via `clearTranscript()`.
- **Existing helpers — use them, do not reimplement:** `$(id)`, `esc(str)`, `tp(text, cls)` (terminal print), `complete(messages, nPredict, onText)`, `runtimeReady()`, `setBusy`/`clearBusy`.
- **wllama API facts, verified against `@wllama/wllama@2`'s `wllama.d.ts`:**
  - `getLogits(topK?: number) => Promise<{token: number, p: number}[]>` — **softmax of raw logits, before sampling**
  - `samplingInit(config: SamplingConfig, pastTokens?: number[]) => Promise<void>`
  - `tokenize(text, special?) => Promise<number[]>`
  - `detokenize(tokens, returnString: true) => Promise<string>`
  - `kvClear() => Promise<void>`, `kvRemove(nKeep, nDiscard) => Promise<void>`
- **The app is CPU-only.** wllama runs WASM SIMD + threads. There is no GPU inference.

---

### Task 1: Stop the app lying about itself

Two hardcoded strings in Settings are false. Both are one-line fixes, and they set the tone for everything else in this plan.

**Files:**
- Modify: `rules_baker/web/index.html` — the `fillRuntime` IIFE

**Interfaces:**
- Consumes: nothing
- Produces: nothing later tasks rely on

- [ ] **Step 1: Fix the context window row**

Find in `fillRuntime`:

```js
    + row('Context window', '1024 tokens')
```

Every load in this file passes `n_ctx: 4096` (three call sites: `loadModelFromUrl`, `loadModel` for a dropped file, `loadModel` for a stored blob). The 1024 is stale. Replace with:

```js
    // Must track the n_ctx passed at load time — every loadModel call in this
    // file uses 4096. A hardcoded figure here silently drifts from reality.
    + row('Context window', `${N_CTX.toLocaleString()} tokens`)
```

- [ ] **Step 2: Introduce the constant it now reads**

Immediately above the `fillRuntime` IIFE, add:

```js
// One source of truth for the context length. It was written out four times —
// three loadModel calls and a Settings row — and the Settings copy had drifted
// to 1024 while the loads used 4096.
const N_CTX = 4096;
```

Then replace the literal in all three load sites so they read from it:

```js
{ n_ctx: N_CTX }
```

and in the URL-load call, `{ n_ctx: N_CTX, progressCallback: ... }`.

- [ ] **Step 3: Fix the WebGPU row**

`'gpu' in navigator` is true on most machines, and the row currently renders a bare `available`, which reads as "the GPU is doing the work". It is not — inference is CPU-only.

In `fillRuntime`:

```js
    + row('WebGPU', webgpu ? 'available — not used (CPU inference)' : 'no')
```

And in the dashboard device strip (search for `col('WebGPU'`):

```js
    + col('WebGPU', webgpu ? 'not used' : 'no', '');
```

Note the third argument drops from `'ok'` to `''` — the green styling is what made it read as a capability in use.

- [ ] **Step 4: Verify**

Reload `http://localhost:8123`, then in the console:

```js
document.getElementById('setNav').click();
await new Promise(r => setTimeout(r, 200));
const rows = [...document.querySelectorAll('#setRuntime .set-row')].map(r => r.textContent);
console.log(rows.find(r => r.includes('Context')));   // expect "4,096 tokens"
console.log(rows.find(r => r.includes('WebGPU')));    // expect "not used"
document.getElementById('setClose').click();
console.log('strip:', document.getElementById('device').textContent.includes('not used'));
```

Expected: context reads `4,096 tokens`, WebGPU says `not used`, strip check `true`.

- [ ] **Step 5: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Stop Settings reporting a context window and a GPU it doesn't use"
```

---

### Task 2: `sampling.js` — the sampler, in JS, as a pure function

**Files:**
- Create: `rules_baker/web/sampling.js`

**Interfaces:**
- Consumes: nothing
- Produces: `applySampling(dist, {temperature, topK, topP})` → `{token, p}[]`; `SAMPLING_DEFAULTS`

- [ ] **Step 1: Create the module**

```js
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

export const SAMPLING_DEFAULTS = { temperature: 0.2, topK: 40, topP: 0.9 };

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
  const max = Math.max(...scaled.map(d => d._l));
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
  const { temperature, topK, topP } = { ...SAMPLING_DEFAULTS, ...opts };
  let out = applyTemperature(dist, temperature);
  out = applyTopK(out, topK);
  out = applyTopP(out, topP);
  return out.sort((a, b) => b.p - a.p);
}
```

- [ ] **Step 2: Verify the maths in the browser**

Server should already be running. Open `http://localhost:8123` and run:

```js
const S = await import('./sampling.js');
const d = [{token:1,p:0.5},{token:2,p:0.25},{token:3,p:0.15},{token:4,p:0.10}];
const sum = (x) => x.reduce((s,d)=>s+d.p,0);

const hot = S.applySampling(d, { temperature: 2.0, topK: 0, topP: 1 });
const cold = S.applySampling(d, { temperature: 0.1, topK: 0, topP: 1 });
console.log('hot flattens:', hot[0].p < 0.5, hot[0].p.toFixed(3));
console.log('cold sharpens:', cold[0].p > 0.5, cold[0].p.toFixed(3));
console.log('both still sum to 1:', Math.abs(sum(hot)-1) < 1e-9, Math.abs(sum(cold)-1) < 1e-9);

const k2 = S.applyTopK(d, 2);
console.log('top-k keeps 2, renormalised:', k2.length === 2, Math.abs(sum(k2)-1) < 1e-9);

const p6 = S.applyTopP(d, 0.6);
console.log('top-p 0.6 keeps 2 (0.5 then 0.75 crosses):', p6.length === 2);

const greedy = S.applySampling(d, { temperature: 0 });
console.log('temp 0 is greedy:', greedy[0].p === 1 && greedy[0].token === 1);

console.log('input not mutated:', d[0].p === 0.5);
```

Expected: every check `true`. `hot[0].p` below 0.5, `cold[0].p` above 0.5.

- [ ] **Step 3: Commit**

```bash
git add rules_baker/web/sampling.js
git commit -m "Add a pure JS mirror of llama.cpp's sampler"
```

---

### Task 3: One temperature control, and make it real

`/temp` in the terminal writes `termState.temp`, which nothing on the generation path reads — `wasmComplete` reads `$('temp').value` from the Playground dropdown. So the terminal's temperature command has never done anything.

**Files:**
- Modify: `rules_baker/web/index.html` — `termState`, the `/temp` case, `wasmComplete`

**Interfaces:**
- Consumes: `SAMPLING_DEFAULTS` from Task 2
- Produces: `currentSampling()` → `{temperature, topK, topP}` — Task 8 renders from this

- [ ] **Step 1: Import**

Below the existing `import { runChain } from './chain.js';` line:

```js
import { applySampling, SAMPLING_DEFAULTS } from './sampling.js';
```

- [ ] **Step 2: Add one accessor both paths read**

Add immediately after the `termState` declaration:

```js
// The single place sampling settings are read from. /temp used to write
// termState.temp, which nothing on the generation path ever read — the command
// reported a change and changed nothing. Both the terminal and the Playground
// now resolve through here.
function currentSampling() {
  return {
    temperature: parseFloat($('temp').value),
    topK: SAMPLING_DEFAULTS.topK,
    topP: SAMPLING_DEFAULTS.topP,
  };
}
```

- [ ] **Step 3: Make `/temp` drive the real control**

Replace the `/temp` case in `termRun`:

```js
      case 'temp': {
        const v = parseFloat(arg);
        if (isNaN(v)) { tp('usage: /temp 0.2', 'errmsg'); break; }
        // Write to the control the generation path actually reads. The dropdown
        // has fixed options, so snap to the nearest rather than silently
        // accepting a value that will not apply.
        const opts = [...$('temp').options].map(o => parseFloat(o.value));
        const near = opts.reduce((b, c) => Math.abs(c - v) < Math.abs(b - v) ? c : b, opts[0]);
        $('temp').value = String(near);
        termState.temp = near;
        tp(near === v ? `temperature = ${near}`
                      : `temperature = ${near}  (nearest available to ${v})`, 'sysmsg');
        break;
      }
```

- [ ] **Step 4: Have `wasmComplete` read the accessor**

In `wasmComplete`, replace:

```js
    sampling: { temp: parseFloat($('temp').value), top_p: 0.9 },
```

with:

```js
    sampling: (() => { const s = currentSampling();
      return { temp: s.temperature, top_k: s.topK, top_p: s.topP }; })(),
```

- [ ] **Step 5: Verify**

```js
document.getElementById('temp').value = '0.2';
const inp = document.getElementById('termIn');
const run = (v) => { inp.value = v; inp.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true})); };
run('/temp 0.8');
await new Promise(r => setTimeout(r, 200));
console.log('dropdown followed:', document.getElementById('temp').value === '0.8');
run('/temp 0.37');
await new Promise(r => setTimeout(r, 200));
console.log('snapped to nearest:', document.getElementById('temp').value === '0.5');
console.log('terminal said so:', document.getElementById('termOut').textContent.includes('nearest available'));
```

Expected: all three `true`.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Make /temp actually change the temperature"
```

---

### Task 4: Stop degenerate repetition

A small model looping `and and and` until `maxtok` is not a crash, it is a wasted run. Detect it and stop.

Do **not** implement this by purging context — that is silent trimming, which `docs/specs/2026-08-17-multi-turn-chat-design.md` forbids: an explicit stop is understandable, silent amnesia is not.

**Files:**
- Create: `rules_baker/web/repetition.js`
- Modify: `rules_baker/web/index.html` — `wasmComplete`

**Interfaces:**
- Consumes: nothing
- Produces: `looksDegenerate(text)` → `{repeating: boolean, phrase: string|null, times: number}`

- [ ] **Step 1: Create the detector**

```js
/* repetition.js — spotting a model that has got stuck.
 *
 * Small models fall into loops: the same word, phrase or line emitted until the
 * token budget runs out. It is not an error and nothing crashes, so nothing
 * stops it — the run just burns its budget producing nothing.
 *
 * Detection is on the emitted TEXT rather than on token ids, because the same
 * stall shows up at three different granularities and a token-level check
 * catches only the narrowest one:
 *   - a single token repeated  ("the the the")
 *   - a phrase cycling         ("I can help. I can help.")
 *   - a whole line looping     (common in code output)
 */

const MIN_TAIL = 40;        // don't judge a reply until there is something to judge
const REPEATS_TO_STOP = 4;  // three is a stylistic flourish; four is a stall

/** Does the tail of `text` consist of one short unit repeated? */
export function looksDegenerate(text) {
  const s = String(text || '');
  if (s.length < MIN_TAIL) return { repeating: false, phrase: null, times: 0 };

  // Look only at the tail: a reply that legitimately repeated something early
  // and then moved on is not stuck.
  const tail = s.slice(-400);

  // Try progressively longer units. 1-3 words catches token loops, longer
  // catches sentence and line loops.
  for (const unit of splitUnits(tail)) {
    const times = trailingRepeats(tail, unit);
    if (times >= REPEATS_TO_STOP) {
      return { repeating: true, phrase: unit.trim(), times };
    }
  }
  return { repeating: false, phrase: null, times: 0 };
}

function splitUnits(tail) {
  const words = tail.trim().split(/\s+/);
  const units = [];
  for (const n of [1, 2, 3, 5, 8]) {
    if (words.length >= n * REPEATS_TO_STOP) units.push(words.slice(-n).join(' ') + ' ');
  }
  const lines = tail.split('\n').filter(Boolean);
  if (lines.length >= REPEATS_TO_STOP) units.push(lines[lines.length - 1] + '\n');
  return units;
}

/** How many times does `unit` repeat, back to back, at the end of `tail`? */
function trailingRepeats(tail, unit) {
  if (!unit.trim()) return 0;
  const norm = (x) => x.toLowerCase().replace(/\s+/g, ' ');
  const u = norm(unit);
  let t = norm(tail);
  let count = 0;
  while (t.endsWith(u) && count < 64) {
    t = t.slice(0, -u.length);
    count++;
  }
  return count;
}
```

- [ ] **Step 2: Verify**

```js
const R = await import('./repetition.js');
console.log('single word loop:', R.looksDegenerate('ok so ' + 'the '.repeat(20)).repeating === true);
console.log('phrase loop:', R.looksDegenerate('sure. ' + 'I can help. '.repeat(8)).repeating === true);
console.log('line loop:', R.looksDegenerate('x\n' + 'print(1)\n'.repeat(9)).repeating === true);
console.log('normal prose is fine:', R.looksDegenerate(
  'The function reverses a string by slicing it with a negative step, which is the idiomatic approach in Python and runs in linear time.').repeating === false);
console.log('short text is not judged:', R.looksDegenerate('hi hi hi').repeating === false);
console.log('three repeats allowed:', R.looksDegenerate('well ' + 'no '.repeat(3) + 'and then something else entirely happened here').repeating === false);
const hit = R.looksDegenerate('and ' + 'again '.repeat(12));
console.log('reports what and how many:', hit.phrase === 'again', hit.times >= 4);
```

Expected: every check `true`.

- [ ] **Step 3: Wire it into generation**

In `wasmComplete`, inside the `onNewToken` callback, after `nTok++; full = currentText;`:

```js
      // Check occasionally rather than per token — the scan is cheap but not
      // free, and a loop is still a loop eight tokens later.
      if (nTok % 8 === 0 && opt?.abortSignal) {
        const rep = looksDegenerate(currentText);
        if (rep.repeating) {
          degenerate = rep;           // reported after the loop, not from inside it
          opt.abortSignal();
        }
      }
```

Declare `let degenerate = null;` beside `let nTok = 0, full = '';` at the top of `wasmComplete`, and after the `createChatCompletion` call resolves:

```js
  if (degenerate) {
    // Said out loud. A run that stopped early and does not say so looks like a
    // model that simply had nothing more to add.
    tp(`[stopped: repeating "${degenerate.phrase}" ${degenerate.times}× — the model is stuck, not finished]`, 'errmsg');
  }
```

Add the import beside the others:

```js
import { looksDegenerate } from './repetition.js';
```

- [ ] **Step 4: Verify against a real model**

Load the smallest model and provoke a loop:

```js
document.getElementById('url').value =
  'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf';
document.getElementById('loadBtn').click();
```

Wait for the load, then in the terminal type: `say the word banana over and over forever`

Expected: generation stops early with the `[stopped: repeating …]` line rather than running to the token cap. If the model does not loop on that prompt, try `/temp 0.1` first — low temperature makes loops far more likely. Report which prompt produced it.

- [ ] **Step 5: Commit**

```bash
git add rules_baker/web/repetition.js rules_baker/web/index.html
git commit -m "Stop generation when the model starts repeating itself"
```

---

### Task 5: Clear the KV cache

There is no cache reset anywhere in the app. New chat clears the transcript and the history array; the model's KV cache keeps the old conversation's state.

**Files:**
- Modify: `rules_baker/web/index.html` — `newChatBtn.onclick`, and a terminal command

**Interfaces:**
- Consumes: nothing
- Produces: nothing later tasks rely on

- [ ] **Step 1: Clear on New chat**

In `newChatBtn.onclick`, immediately after `clearHistory(); clearTranscript();`:

```js
  // The transcript and the history array are ours; the KV cache is the model's,
  // and nothing was clearing it. On a 4GB Chromebook a 4096-token cache is real
  // memory. Wrapped because kvClear only exists once a model is loaded.
  try { if (activeRuntime === 'wasm' && wllama) await wllama.kvClear(); } catch (_) {}
```

- [ ] **Step 2: Add a terminal command**

Add to `TERM_COMMANDS`, after the `status` entry:

```js
  { name: 'flush',   args: '',            desc: 'clear the model\'s KV cache' },
```

And a case in `termRun`:

```js
      case 'flush': {
        if (activeRuntime !== 'wasm' || !wllama) { tp('no in-browser model loaded', 'errmsg'); break; }
        try {
          await wllama.kvClear();
          // Precise about what was freed. A student who reads "memory released"
          // and then sees 270MB still resident would reasonably think it failed.
          tp('KV cache cleared — the conversation\'s working memory. The weights stay loaded.', 'sysmsg');
        } catch (e) { tp('could not clear cache: ' + e.message, 'errmsg'); }
        break;
      }
```

- [ ] **Step 3: Verify**

With a model loaded:

```js
const inp = document.getElementById('termIn');
inp.value = '/flush';
inp.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
await new Promise(r => setTimeout(r, 400));
console.log(document.getElementById('termOut').textContent.slice(-140));
```

Expected: `KV cache cleared — the conversation's working memory. The weights stay loaded.` and no console error. Then generate again and confirm it still works — a cleared cache must not break the next generation.

- [ ] **Step 4: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Clear the model's KV cache on New chat and via /flush"
```

---

### Task 6: `gguf.js` — read the header, and judge it

**Files:**
- Create: `rules_baker/web/gguf.js`

**Interfaces:**
- Consumes: nothing
- Produces: `readGgufHeader(blob)` → `Promise<{ok, arch, name, layers, embedding, contextLength, heads, quant, tensorCount, kvCount, truncated, problems[]}>`

- [ ] **Step 1: Create the parser**

```js
/* gguf.js — what a .gguf says about itself, and whether to believe it.
 *
 * A dropped model file currently shows six "unverifiable" fields in the
 * passport, because everything the app knows about models comes from
 * models.json — metadata a human typed. The file itself carries the truth in
 * its header, and reading it turns those six unknowns into facts.
 *
 * It doubles as a gate. GGUF parser bugs are a real vulnerability class, this
 * app loads .gguf from arbitrary user-supplied URLs, and llama.cpp will parse
 * whatever it is handed. WASM contains the blast radius — an overflow inside
 * wllama's linear memory cannot reach the host — so this is defence in depth
 * rather than a hole being plugged. Cheap defence, given we are reading the
 * header anyway.
 *
 * Format: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count, then
 * kv_count pairs of (string key, u32 value_type, value). All little-endian.
 */

const MAGIC = 0x46554747;            // "GGUF" read as LE u32

// GGUF metadata value types.
const T = { U8:0, I8:1, U16:2, I16:3, U32:4, I32:5, F32:6, BOOL:7, STR:8, ARR:9,
            U64:10, I64:11, F64:12 };

// Enough for the header on every model this project ships. The tokenizer's
// vocabulary lives in the metadata and can run to megabytes, so a parse that
// runs off the end is reported as truncated rather than treated as corrupt.
const HEADER_BYTES = 8 * 1024 * 1024;

// llama.cpp's file_type enum, trimmed to the ones seen in the wild here.
const FILE_TYPES = {
  0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1', 7:'Q8_0', 8:'Q5_0', 9:'Q5_1',
  10:'Q2_K', 11:'Q3_K_S', 12:'Q3_K_M', 13:'Q3_K_L', 14:'Q4_K_S', 15:'Q4_K_M',
  16:'Q5_K_S', 17:'Q5_K_M', 18:'Q6_K', 19:'IQ2_XXS', 20:'IQ2_XS',
};

class Reader {
  constructor(buf) { this.v = new DataView(buf); this.o = 0; this.end = buf.byteLength; }
  need(n) { if (this.o + n > this.end) throw new RangeError('truncated'); }
  u8()  { this.need(1); return this.v.getUint8(this.o++); }
  u32() { this.need(4); const x = this.v.getUint32(this.o, true); this.o += 4; return x; }
  i32() { this.need(4); const x = this.v.getInt32(this.o, true); this.o += 4; return x; }
  f32() { this.need(4); const x = this.v.getFloat32(this.o, true); this.o += 4; return x; }
  f64() { this.need(8); const x = this.v.getFloat64(this.o, true); this.o += 8; return x; }
  u64() { this.need(8); const x = this.v.getBigUint64(this.o, true); this.o += 8; return Number(x); }
  str() {
    const n = this.u64();
    if (n > 1 << 22) throw new RangeError('implausible string length');
    this.need(n);
    const s = new TextDecoder().decode(new Uint8Array(this.v.buffer, this.o, n));
    this.o += n;
    return s;
  }
  value(type) {
    switch (type) {
      case T.U8: case T.I8:   return this.u8();
      case T.U16: case T.I16: { this.need(2); const x = this.v.getUint16(this.o, true); this.o += 2; return x; }
      case T.U32:             return this.u32();
      case T.I32:             return this.i32();
      case T.F32:             return this.f32();
      case T.BOOL:            return this.u8() !== 0;
      case T.STR:             return this.str();
      case T.U64: case T.I64: return this.u64();
      case T.F64:             return this.f64();
      case T.ARR: {
        const itemType = this.u32();
        const n = this.u64();
        // Vocabularies are huge and we never need their contents — skip the
        // values but keep the cursor exact, or every later key misparses.
        for (let i = 0; i < n; i++) this.value(itemType);
        return `[${n} items]`;
      }
      default: throw new RangeError('unknown value type ' + type);
    }
  }
}

export async function readGgufHeader(blob) {
  const problems = [];
  let buf;
  try {
    buf = await blob.slice(0, Math.min(HEADER_BYTES, blob.size)).arrayBuffer();
  } catch (e) {
    return fail(['could not read the file: ' + (e.message || e)]);
  }
  if (buf.byteLength < 24) return fail(['file is too small to be a GGUF']);

  const r = new Reader(buf);
  if (r.u32() !== MAGIC) return fail(['not a GGUF file — the magic bytes are wrong']);

  const version = r.u32();
  if (version < 1 || version > 10) problems.push(`unexpected GGUF version ${version}`);

  const tensorCount = r.u64();
  const kvCount = r.u64();
  // Sanity gate. Real models are in the hundreds-to-thousands of tensors; a
  // header claiming millions is either corrupt or hostile, and either way we
  // should not hand it to the parser.
  if (tensorCount > 100000) problems.push(`implausible tensor count (${tensorCount})`);
  if (kvCount > 10000) problems.push(`implausible metadata count (${kvCount})`);

  const kv = {};
  let truncated = false;
  try {
    for (let i = 0; i < kvCount; i++) {
      const key = r.str();
      const type = r.u32();
      kv[key] = r.value(type);
    }
  } catch (e) {
    // Running out of buffer is expected for models with large vocabularies and
    // is NOT corruption — say which it was.
    if (e instanceof RangeError && e.message === 'truncated') truncated = true;
    else problems.push('malformed metadata: ' + (e.message || e));
  }

  const arch = kv['general.architecture'] || null;
  const g = (suffix) => (arch ? kv[`${arch}.${suffix}`] : undefined);
  const ft = kv['general.file_type'];

  const layers = g('block_count') ?? null;
  const embedding = g('embedding_length') ?? null;
  const contextLength = g('context_length') ?? null;

  if (layers !== null && (layers < 1 || layers > 512)) problems.push(`implausible layer count (${layers})`);
  if (embedding !== null && (embedding < 1 || embedding > 65536)) problems.push(`implausible embedding size (${embedding})`);

  return {
    ok: problems.length === 0,
    version, tensorCount, kvCount, truncated, problems,
    arch,
    name: kv['general.name'] || null,
    layers, embedding, contextLength,
    heads: g('attention.head_count') ?? null,
    quant: ft !== undefined ? (FILE_TYPES[ft] || `type ${ft}`) : null,
  };
}

function fail(problems) {
  return { ok: false, problems, version: null, tensorCount: null, kvCount: null,
           truncated: false, arch: null, name: null, layers: null, embedding: null,
           contextLength: null, heads: null, quant: null };
}
```

- [ ] **Step 2: Verify against a real file and against junk**

```js
const G = await import('./gguf.js');

// A real one, fetched as a Blob.
const url = 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf';
const real = await (await fetch(url)).blob();
const h = await G.readGgufHeader(real);
console.log('real file:', h);
console.log('arch read:', h.arch === 'llama');
console.log('layers read:', h.layers === 30);
console.log('quant read:', h.quant === 'Q4_K_M');
console.log('no problems:', h.problems.length === 0);

// Junk must be rejected, not parsed.
const junk = new Blob([new Uint8Array(4096)]);
const j = await G.readGgufHeader(junk);
console.log('junk rejected:', j.ok === false, j.problems[0]);

// A file that is too short.
const tiny = await G.readGgufHeader(new Blob([new Uint8Array(8)]));
console.log('tiny rejected:', tiny.ok === false);
```

Expected: the real file reports `arch: "llama"`, a layer count, `quant: "Q4_K_M"`, no problems. Junk reports `not a GGUF file — the magic bytes are wrong`. Report the actual layer count you observe rather than assuming 30.

- [ ] **Step 3: Commit**

```bash
git add rules_baker/web/gguf.js
git commit -m "Add a GGUF header reader that also refuses implausible files"
```

---

### Task 7: Use the header — honest local files, and a gate

**Files:**
- Modify: `rules_baker/web/index.html` — `loadLocal`, `renderPassport`
- Modify: `rules_baker/web/provenance.js` — `buildPassport`'s `local` branch

**Interfaces:**
- Consumes: `readGgufHeader(blob)` from Task 6
- Produces: passports for local files carrying real `arch`, `layers`, `params`-adjacent fields

- [ ] **Step 1: Read the header before loading**

In `loadLocal`, replace the guard block with:

```js
async function loadLocal(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.gguf')) { status.innerHTML = `<span style="color:var(--bad)">✗ Not a .gguf file.</span>`; return; }

  // Read the header BEFORE handing the bytes to wllama. Two reasons: it turns
  // the passport's unknowns into facts, and it refuses obviously bad files
  // rather than passing them to a C++ parser.
  status.textContent = 'Reading the file header…';
  const header = await readGgufHeader(file);
  if (!header.ok && header.problems.length) {
    status.innerHTML = `<span style="color:var(--bad)">✗ This file did not pass the header check: `
      + `${esc(header.problems.join('; '))}. Not loading it.</span>`;
    return;
  }

  let kept = null;
  pendingLocalFile = { name: file.name, size: file.size, header };
```

Leave the rest of `loadLocal` unchanged.

- [ ] **Step 2: Add the import**

```js
import { readGgufHeader } from './gguf.js';
```

- [ ] **Step 3: Let the passport use it**

In `provenance.js`, replace the `if (local)` branch of `buildPassport`:

```js
  if (local) {
    const h = local.header || {};
    // What the file itself says. Anything the header did not carry stays
    // unverifiable — the list shrinks, it does not disappear, and the
    // difference between "read from the file" and "typed into a catalogue" is
    // the whole point.
    const unverifiable = ['training data', 'evaluation methodology'];
    if (!h.arch) unverifiable.push('architecture');
    if (!h.layers) unverifiable.push('layer count');
    unverifiable.push('licence');            // never present in a GGUF header

    return {
      generatedAt: now,
      source: 'local-file',
      name: h.name || local.name,
      sizeBytes: local.size,
      license: null,
      licenceReview: licenceReview(null),
      base: null,
      arch: h.arch || null,
      layers: h.layers || null,
      hidden: h.embedding || null,
      ctx: h.contextLength || null,
      quant: h.quant || null,
      params: null,
      headerRead: !!h.arch,
      unverifiable,
      runtime, measured: measured || null,
    };
  }
```

- [ ] **Step 4: Show the new fields**

In `renderPassport`, add to the `pp-grid` block, after the `Context` field:

```js
    + field('Quantisation', p.quant)
    + field('Layers', p.layers)
```

And after the licence note, when the header was read:

```js
    + (p.headerRead
        ? `<div class="pp-unk" style="color:var(--dim)">These figures were read from the file's own header, `
          + `not from a catalogue.</div>`
        : '')
```

- [ ] **Step 5: Verify with a real dropped file**

Download a small GGUF to disk, then drop it onto the app's drop zone (or use `browse…`). Confirm:

```js
console.log(document.getElementById('passport').textContent);
```

Expected: the passport shows a real architecture, layer count and quantisation for a file that previously showed none, and the "can't be verified" list is shorter — `licence`, `training data`, `evaluation methodology` — rather than six entries. Report the observed text.

Then rename any non-GGUF file to `.gguf` and drop it. Expected: refused with `did not pass the header check: not a GGUF file — the magic bytes are wrong`, and wllama is never called.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html rules_baker/web/provenance.js
git commit -m "Read a dropped model's header instead of admitting ignorance"
```

---

### Task 8: The generation panel — candidates, sampler, tokens

The payoff. Three views of the same moment: what the model thinks, what the sampler does to it, and how the text became tokens.

**Files:**
- Modify: `rules_baker/web/index.html` — markup near `#playground`, CSS, and a new render function

**Interfaces:**
- Consumes: `applySampling` (Task 2), `currentSampling()` (Task 3), wllama's `getLogits`, `tokenize`, `detokenize`
- Produces: nothing later tasks rely on

- [ ] **Step 1: Add the markup**

Immediately after the `#output` div in the Playground section:

```html
      <div id="inspect" class="passport" hidden>
        <div class="pp-head">
          <span class="pp-title">What the model was choosing between</span>
          <button type="button" class="ghost pp-dl" id="inspectClose"
                  style="padding:4px 10px; font-size:11.5px">Hide</button>
        </div>
        <div id="inspectBody"></div>
      </div>
```

- [ ] **Step 2: Add the styles**

After the `.pp-unk` rules:

```css
  .cand { display:grid; grid-template-columns:1fr 54px 1fr 54px; gap:4px 10px;
    align-items:center; font:12px var(--mono); }
  .cand .h { font-size:10px; letter-spacing:.07em; text-transform:uppercase;
    color:var(--dimmer); font-family:inherit; }
  .cand .tokv { color:var(--ink); overflow:hidden; text-overflow:ellipsis; white-space:pre; }
  .cand .barwrap { background:rgba(255,255,255,.06); border-radius:3px; height:13px; position:relative; }
  .cand .bar { background:rgba(255,255,255,.55); height:100%; border-radius:3px; }
  .cand .pct { color:var(--dim); text-align:right; font-variant-numeric:tabular-nums; }
  .cand .gone { opacity:.32; }
  .toks { display:flex; flex-wrap:wrap; gap:3px; margin-top:4px; }
  .tok { font:11.5px var(--mono); padding:1px 5px; border-radius:4px;
    background:rgba(255,255,255,.07); color:var(--ink); white-space:pre; }
  .tok .id { color:var(--dimmer); margin-left:5px; font-size:10px; }
```

- [ ] **Step 3: Render the two distributions**

Add before `renderActivity`:

```js
// ---- generation inspector ----
// getLogits() returns the model's distribution BEFORE sampling — that is what
// its doc comment says and it is easy to miss. Rendering only that would give
// us a temperature control that visibly does nothing, which teaches the exact
// opposite of the point. So: two columns, raw and sampled, from the same moment.
async function renderInspector() {
  const box = $('inspect'), body = $('inspectBody');
  if (activeRuntime !== 'wasm' || !wllama) {
    body.innerHTML = `<div class="pp-note">Only available on the in-browser runtime — `
      + `Ollama does not expose the candidate list here.</div>`;
    box.hidden = false;
    return;
  }
  let raw;
  try {
    raw = await wllama.getLogits(8);
  } catch (e) {
    body.innerHTML = `<div class="pp-note">No candidates available yet — generate something first.</div>`;
    box.hidden = false;
    return;
  }
  if (!raw || !raw.length) {
    body.innerHTML = `<div class="pp-note">No candidates available yet — generate something first.</div>`;
    box.hidden = false;
    return;
  }

  const s = currentSampling();
  const sampled = applySampling(raw, s);
  const kept = new Set(sampled.map(d => d.token));
  const pieces = await Promise.all(raw.map(d => wllama.detokenize([d.token], true).catch(() => '?')));
  const pct = (p) => (p * 100).toFixed(1) + '%';
  const bar = (p) => `<div class="barwrap"><div class="bar" style="width:${Math.round(p * 100)}%"></div></div>`;

  const rows = raw.map((d, i) => {
    const after = sampled.find(x => x.token === d.token);
    const cls = kept.has(d.token) ? '' : ' gone';
    return `<div class="tokv${cls}">${esc(JSON.stringify(pieces[i]).slice(1, -1))}</div>`
      + `<div class="pct${cls}">${pct(d.p)}</div>`
      + `<div class="${cls ? 'gone' : ''}">${after ? bar(after.p) : '<span class="pct">cut</span>'}</div>`
      + `<div class="pct${cls}">${after ? pct(after.p) : '—'}</div>`;
  }).join('');

  body.innerHTML =
      `<div class="cand">`
    +   `<div class="h">Token</div><div class="h">Model</div>`
    +   `<div class="h">After sampling</div><div class="h">&nbsp;</div>`
    +   rows
    + `</div>`
    + `<div class="pp-unk" style="margin-top:9px">Left is the model's own distribution. `
    + `Right is the same moment after temperature ${s.temperature}, top-k ${s.topK}, top-p ${s.topP}. `
    + `Change the temperature and only the right column moves — the model's opinion never changes, `
    + `only what the sampler does with it.</div>`;
  box.hidden = false;
}

$('inspectClose').onclick = () => { $('inspect').hidden = true; };
```

- [ ] **Step 4: Show it after each generation**

At the end of `handleGenerate`, immediately before `refreshUsage();`:

```js
  renderInspector();     // the candidates behind the last token generated
```

- [ ] **Step 5: Add the token inspector command**

Add to `TERM_COMMANDS`, after `flush`:

```js
  { name: 'tokens?', args: '<text>',      desc: 'show how text splits into tokens' },
```

Note the trailing `?` keeps it distinct from the existing `/tokens` setting command. Add a case in `termRun`:

```js
      case 'tokens?': {
        if (activeRuntime !== 'wasm' || !wllama) { tp('no in-browser model loaded', 'errmsg'); break; }
        if (!arg) { tp('usage: /tokens? strawberry', 'errmsg'); break; }
        try {
          const ids = await wllama.tokenize(arg);
          const pieces = await Promise.all(ids.map(id => wllama.detokenize([id], true).catch(() => '?')));
          tp(`${ids.length} token${ids.length === 1 ? '' : 's'} for ${arg.length} characters:`, 'sysmsg');
          tp(pieces.map((p, i) => `${JSON.stringify(p)}·${ids[i]}`).join('  '), 'gen');
        } catch (e) { tp('tokenize failed: ' + e.message, 'errmsg'); }
        break;
      }
```

- [ ] **Step 6: Verify with a model loaded**

```js
// Tokeniser — the classic demonstration.
const inp = document.getElementById('termIn');
const run = (v) => { inp.value = v; inp.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true})); };
run('/tokens? strawberry');
await new Promise(r => setTimeout(r, 500));
console.log(document.getElementById('termOut').textContent.slice(-160));
```

Expected: `strawberry` splits into more than one token, each shown with its id. Report the actual split.

Then generate something in the Playground and check the inspector:

```js
console.log('inspector shown:', !document.getElementById('inspect').hidden);
console.log(document.getElementById('inspectBody').textContent.slice(0, 260));
```

Expected: a table of candidate tokens with two percentage columns. Now change temperature and re-render:

```js
document.getElementById('temp').value = '0.8';
await renderInspector?.();   // if not reachable, generate again at 0.8 instead
```

Expected: the left column is unchanged and the right column is flatter. **This is the check that matters** — if both columns move, the raw distribution is being re-sampled somewhere and the lesson is broken. Report both columns at 0.1 and at 0.8.

- [ ] **Step 7: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Show the candidate tokens, and what the sampler does to them"
```

---

## Self-review

**Spec coverage** (items 1, 2, 4, 5, 6 of `2026-08-18-observability-and-honesty-design.md`):

| Spec requirement | Task |
|---|---|
| Top-5 log-probs | 8 |
| Sampling knobs, live | 2, 3, 8 |
| Tokenizer view | 8 |
| Raw-vs-sampled two columns (the trap) | 2, 8 |
| GGUF header parsing → honest metadata | 6, 7 |
| GGUF validation as a gate | 6, 7 |
| WebGPU label honesty | 1 |
| KV cache clear | 5 |
| Degenerate repetition stop | 4 |
| Not purging context to achieve it | 4 (explicit note) |

Spec item 3 (the NaN breaker) is Tier 2 and out of scope here — it is Python-side and shares no files with this plan.

**Two pre-existing bugs are fixed in passing**, each inside the task already editing that line: the Settings context row said 1024 while every load passes 4096 (Task 1), and `/temp` wrote a variable the generation path never read (Task 3). Neither is in the spec; both were found while verifying it.

**Placeholders:** none. Every step carries the code to write and the console snippet to verify it.

**Type consistency:** `applySampling(dist, {temperature, topK, topP})` returns `{token, p}[]` and is called that way in Task 8. `currentSampling()` returns `{temperature, topK, topP}` — defined in Task 3, consumed in Task 8. `readGgufHeader(blob)` returns the object destructured in Task 7. `looksDegenerate(text)` returns `{repeating, phrase, times}`, used with those names in Task 4. `pendingLocalFile` gains a `header` field in Task 7 and is read by `buildPassport`'s `local` branch in the same task.

**Ordering:** Tasks 2 and 6 create pure modules verifiable with no model loaded, so they can be checked instantly. Tasks 3, 4, 5, 7 and 8 need a loaded model — SmolLM2 135M (100MB) is the fastest, and its URL is in the Task 4 snippet.

**Known risk:** Task 8's key assertion — that changing temperature moves only the sampled column — depends on `getLogits()` returning the pre-sampling distribution, which its doc comment states but which has not been confirmed against a running model. If both columns move, stop and report rather than adjusting the UI to hide it; the whole feature rests on that distinction.
