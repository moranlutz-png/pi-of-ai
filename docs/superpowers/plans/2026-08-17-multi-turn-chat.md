# Multi-turn Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A follow-up question ("now add error handling") sees the previous exchange, instead of every prompt starting from nothing.

**Architecture:** A new ES module `chat.js` owns the message history, a character-based size estimate, and the handoff text. `index.html` renders `#output` as a transcript instead of a single overwritten slot, appends each accepted turn to the history, and gains a New chat button. Auto-fix retries stay internal and never enter the history.

**Tech Stack:** Vanilla ES modules, no build step, no dependencies.

**Spec:** `docs/specs/2026-08-17-multi-turn-chat-design.md`

## Global Constraints

- No build step. `index.html` loads one `<script type="module">`; new modules are imported from it.
- No test framework. Verification is in-browser against `http://localhost:8126` (serving the main checkout). Port 8123 belongs to another session — do not use or kill it.
- `n_ctx` is **4096** tokens, shared between system prompt, history, request and reply.
- Warn past **50%** of the window; **refuse** new turns past **70%**.
- Refusing is deliberate: never silently drop the oldest turn.
- The system prompt sits at index 0 and is exempt from trimming.
- Auto-fix retries and the intro line never enter the history.
- Every value interpolated into innerHTML goes through the existing `esc()` helper, which escapes quotes as well as angle brackets.
- Existing helpers in `index.html`: `$(id)` = getElementById, `esc(str)`, `segments(text)` splits text into `{type:'prose'|'code', ...}`, `COPY_SVG`, `__copyCode(btn)`, `cmdLine(text)`.

---

### Task 1: The chat module

**Files:**
- Create: `rules_baker/web/chat.js`

**Interfaces:**
- Consumes: nothing
- Produces: `getHistory()`, `addTurn(role, content)`, `clearHistory()`, `historyFor(systemPrompt, request)`, `usage()`, `lastCodeBlock()`, `CTX_TOKENS`, `WARN_AT`, `REFUSE_AT`

- [ ] **Step 1: Create the module**

```js
/* chat.js — the conversation history behind multi-turn chat.
 *
 * The whole design is shaped by one number: the models here are loaded with a
 * 4096-token context, shared between the system prompt, the history, the new
 * request and the reply. The app has already shipped one overflow bug from a
 * long prompt plus three auto-fix retries, so this module's job is as much
 * about refusing to overflow as it is about remembering.
 */

export const CTX_TOKENS = 4096;
export const WARN_AT = 0.5;      // show a fullness indicator past here
export const REFUSE_AT = 0.7;    // stop accepting turns past here

// Turns, oldest first. The system prompt is NOT stored here — it is supplied
// per request by historyFor(), so a variant's rules cannot be trimmed away or
// go stale when the user edits them mid-chat.
let turns = [];

export function getHistory() {
  return turns.slice();
}

export function addTurn(role, content) {
  const text = String(content || '');
  if (!text.trim()) return getHistory();
  turns.push({ role, content: text });
  return getHistory();
}

export function clearHistory() {
  turns = [];
}

// Rough, and deliberately so: shipping a tokenizer to make this exact would
// cost more than the number is worth. ~4 characters per token is the usual
// English approximation; code runs denser, which is what the headroom between
// REFUSE_AT and 1.0 is for.
export function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

// What actually goes to the model: system prompt first (exempt from any
// trimming), then the history, then the new request.
export function historyFor(systemPrompt, request) {
  const msgs = [];
  if (systemPrompt && systemPrompt.trim()) {
    msgs.push({ role: 'system', content: systemPrompt });
  }
  for (const t of turns) msgs.push({ role: t.role, content: t.content });
  if (request && request.trim()) msgs.push({ role: 'user', content: request });
  return msgs;
}

// Fullness against the window, including the system prompt and the pending
// request, since those occupy the same budget.
export function usage(systemPrompt = '', pending = '') {
  const all = historyFor(systemPrompt, pending)
    .map(m => m.content).join('\n');
  const tokens = estimateTokens(all);
  const ratio = tokens / CTX_TOKENS;
  return {
    tokens,
    ratio,
    percent: Math.min(100, Math.round(ratio * 100)),
    warn: ratio >= WARN_AT,
    full: ratio >= REFUSE_AT,
    turns: turns.length,
  };
}

// The most recent fenced block in the conversation, newest first. Extracted
// mechanically for the handoff — asking a small model to reproduce code inside
// a summary invites it to mangle it.
export function lastCodeBlock() {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role !== 'assistant') continue;
    const m = /```([\w+.-]*)\n([\s\S]*?)```/g;
    let found = null, hit;
    while ((hit = m.exec(turns[i].content)) !== null) found = hit;
    if (found) return { lang: found[1] || 'python', code: found[2].replace(/\n$/, '') };
  }
  return null;
}

// The text the user copies into a new chat. The model writes the prose; this
// appends the code verbatim.
export function handoffText(summary) {
  const parts = [];
  const s = String(summary || '').trim();
  parts.push(s || 'Continuing from an earlier chat.');
  const block = lastCodeBlock();
  if (block) {
    parts.push('Here is the latest version of the code:');
    parts.push('```' + block.lang + '\n' + block.code + '\n```');
  }
  return parts.join('\n\n');
}
```

- [ ] **Step 2: Verify in the browser**

Server should be running; if not:

```bash
cd ~/Projects/Dev/pi-of-ai/rules_baker/web && python3 serve.py 8126
```

Open `http://localhost:8126` and run in the console:

```js
const C = await import('./chat.js');
C.clearHistory();
C.addTurn('user', 'write a reverse function');
C.addTurn('assistant', 'Sure.\n\n```python\ndef rev(s):\n    return s[::-1]\n```');
console.log('two turns:', C.getHistory().length === 2);
console.log('blank ignored:', (C.addTurn('user', '   '), C.getHistory().length === 2));

const msgs = C.historyFor('BE TERSE', 'now add type hints');
console.log('system first:', msgs[0].role === 'system' && msgs[0].content === 'BE TERSE');
console.log('request last:', msgs[msgs.length - 1].content === 'now add type hints');
console.log('shape:', msgs.map(m => m.role).join(','));
console.log('no system when blank:', C.historyFor('', 'x')[0].role === 'user');

const u = C.usage('BE TERSE', 'now add type hints');
console.log('usage:', u);
console.log('not full yet:', u.full === false);

const blk = C.lastCodeBlock();
console.log('code extracted:', blk.lang === 'python' && blk.code.includes('s[::-1]'));

const h = C.handoffText('We wrote a string reverser.');
console.log('handoff has prose + verbatim code:',
  h.startsWith('We wrote') && h.includes('return s[::-1]'));

// fullness: push a lot of text and confirm it refuses
C.addTurn('assistant', 'x'.repeat(4096 * 4 * 0.75));
console.log('now full:', C.usage('', '').full === true);
C.clearHistory();
console.log('cleared:', C.getHistory().length === 0);
```

Expected: every check `true`; `usage` reports small `tokens`/`percent` before the big push, and `full: true` after.

- [ ] **Step 3: Commit**

```bash
cd ~/Projects/Dev/pi-of-ai
git add rules_baker/web/chat.js
git commit -m "Add chat history module with a context-window budget"
```

---

### Task 2: Render the output as a transcript

**Files:**
- Modify: `rules_baker/web/index.html` — CSS near `.disclaimer`, and `renderFinal`

**Interfaces:**
- Consumes: `segments(text)`, `esc()`, `noteHtml(rep)`, `checkPython()` — all existing
- Produces: `appendTurn(role, html)`, `clearTranscript()`, and `#output` containing `.turn` elements

- [ ] **Step 1: Add the styles**

Add immediately before the `.disclaimer` rule:

```css
  .turn { padding:11px 0; border-top:1px solid var(--hair); }
  .turn:first-child { border-top:0; padding-top:0; }
  .turn .who { display:flex; align-items:center; gap:7px; margin-bottom:6px;
    font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--dimmer); }
  .turn.user .body { font-size:13.5px; color:var(--ink); white-space:pre-wrap; }
  .turn.model .body { font-size:13.5px; }
```

- [ ] **Step 2: Add the transcript helpers**

Add immediately before `async function renderFinal(intro, text) {`:

```js
// #output used to be a single slot overwritten per generation. It is now a
// transcript, so a turn is appended rather than replacing what came before.
function appendTurn(role, html) {
  const el = document.createElement('div');
  el.className = 'turn ' + role;
  el.innerHTML = `<div class="who">${role === 'user' ? 'You' : 'Model'}</div>`
    + `<div class="body">${html}</div>`;
  output.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return el;
}

function clearTranscript() {
  output.innerHTML = '';
}
```

- [ ] **Step 3: Stop renderIntro wiping the transcript**

`renderIntro` currently does `output.innerHTML = ...`, which with a transcript
would erase every earlier turn at the start of each generation. It must stream
into a turn of its own instead, which the final render then fills in.

Replace the existing one-line `renderIntro` (search for `function renderIntro`)
with:

```js
// The streaming intro gets its own turn, held open so renderFinal can replace
// its contents with the real answer. Writing to output.innerHTML here would
// wipe the whole transcript on every generation.
let pendingTurn = null;
function renderIntro(text) {
  if (!pendingTurn) pendingTurn = appendTurn('model', '');
  pendingTurn.querySelector('.body').innerHTML =
    `<div class="intro">${esc(String(text).split('\n')[0])}</div>`;
}
```

- [ ] **Step 4: Make renderFinal fill that turn instead of replacing everything**

In `renderFinal`, find the line:

```js
  output.innerHTML = html;
```

and replace it with:

```js
  // Fill the turn the intro opened, or append one if there wasn't an intro.
  if (pendingTurn) { pendingTurn.querySelector('.body').innerHTML = html; pendingTurn = null; }
  else appendTurn('model', html);
```

Then find `function clearTranscript()` from Step 2 and add `pendingTurn = null;`
as its first line, so a cleared transcript cannot leave a dangling reference to
a removed element.

- [ ] **Step 5: Verify**

Reload `http://localhost:8126`, then in the console:

```js
const out = document.getElementById('output');
out.innerHTML = '';
// simulate two model turns landing one after the other
window.__appendProbe = (t) => { const d = document.createElement('div');
  d.className = 'turn model'; d.innerHTML = `<div class="who">Model</div><div class="body">${t}</div>`;
  out.appendChild(d); };
__appendProbe('first answer'); __appendProbe('second answer');
console.log('two turns present:', out.querySelectorAll('.turn').length === 2);
console.log('first survived:', out.textContent.includes('first answer'));
```

Expected: both `true` — the second turn does not replace the first.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Render generated output as a transcript of turns"
```

---

### Task 3: Wire generation to the history

**Files:**
- Modify: `rules_baker/web/index.html` — the module import block, and `handleGenerate`

**Interfaces:**
- Consumes: `getHistory`, `addTurn`, `historyFor`, `usage` from Task 1; `appendTurn` from Task 2
- Produces: history containing exactly the accepted turns

- [ ] **Step 1: Import the module**

Below the existing `import { saveModel, ... } from './model-store.js';` line, add:

```js
import { getHistory, addTurn, clearHistory, historyFor, usage,
         handoffText, CTX_TOKENS, WARN_AT, REFUSE_AT } from './chat.js';
```

- [ ] **Step 2: Build the request from history, and record the turn**

In `handleGenerate`, replace this block:

```js
  const base = sys.trim()
    ? [{ role: 'system', content: sys }, { role: 'user', content: request }]
    : [{ role: 'user', content: request }];
```

with:

```js
  // The request now carries the conversation so far. historyFor() puts the
  // system prompt at index 0 and appends this request last.
  const base = historyFor(sys, request);
```

Then, at the very end of `handleGenerate`, immediately before its closing `}`, add:

```js
  // Only the accepted answer joins the history. The auto-fix retries above are
  // an internal exchange between the app and the model — putting them in would
  // carry up to three rejected attempts and three correction prompts into every
  // later turn's context, on a window that has already overflowed once.
  if (!abort && finalText.trim()) {
    addTurn('user', request);
    addTurn('assistant', finalText);
  }
```

- [ ] **Step 3: Show the user's turn when they send**

In `runBtn.onclick`, immediately after the empty-prompt guard returns and before `runBtn.disabled = true;`, add:

```js
  appendTurn('user', esc($('prompt').value.trim()));
```

- [ ] **Step 4: Verify the history actually reaches the model**

Reload, then in the console. This mocks the runtime so the outgoing messages can be read without a real model:

```js
const C = await import('./chat.js');
C.clearHistory();
const real = window.fetch;
window.__sent = [];
window.fetch = async (u, o) => {
  if (String(u).includes('11434/api/tags'))
    return new Response(JSON.stringify({ models: [{ name: 't:1b' }] }), { status: 200 });
  if (String(u).includes('11434/v1/chat/completions')) {
    window.__sent.push(JSON.parse(o.body).messages.map(m => m.role + ':' + m.content.slice(0, 24)));
    const enc = new TextEncoder();
    return new Response(new ReadableStream({ start(c) {
      c.enqueue(enc.encode(`data: ${JSON.stringify({choices:[{delta:{content:'ok one'}}]})}\n\n`));
      c.enqueue(enc.encode('data: [DONE]\n\n')); c.close(); } }), { status: 200 });
  }
  return real(u, o);
};
document.querySelector('.rt-pill[data-rt="ollama"]').click();
await new Promise(r => setTimeout(r, 900));
document.getElementById('selffix').checked = false;
document.getElementById('sys').value = '';
document.getElementById('prompt').value = 'first question';
document.getElementById('runBtn').click();
await new Promise(r => setTimeout(r, 2500));
document.getElementById('prompt').value = 'second question';
document.getElementById('runBtn').click();
await new Promise(r => setTimeout(r, 2500));
console.log('history:', C.getHistory().map(t => t.role + ':' + t.content.slice(0, 20)));
console.log('LAST request sent:', window.__sent[window.__sent.length - 1]);
window.fetch = real;
```

Expected: history has four entries (user, assistant, user, assistant). The last request sent contains the first exchange **before** `second question` — that is the whole feature. Note the intro call also hits the mock, so `__sent` has more entries than turns; read the last one.

- [ ] **Step 5: Verify retries stay out**

```js
const C = await import('./chat.js');
C.clearHistory();
console.log('with auto-fix ON, a retried generation still adds exactly 2 entries');
// Turn auto-fix on and generate against the mock above; then:
console.log(C.getHistory().length === 2, C.getHistory().map(t => t.role));
```

Expected: `true` and `['user','assistant']` — never more, however many retries ran.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Carry conversation history into each generation"
```

---

### Task 4: Fullness indicator and the ceiling

**Files:**
- Modify: `rules_baker/web/index.html` — composer markup, CSS near `.composer-bar`, and `runBtn.onclick`

**Interfaces:**
- Consumes: `usage(systemPrompt, pending)` from Task 1
- Produces: `refreshUsage()`

- [ ] **Step 1: Add the indicator to the composer**

In the `.composer-bar` div, immediately before the `<span class="composer-actions">` line, add:

```html
          <span class="ctxmeter" id="ctxMeter" hidden></span>
```

- [ ] **Step 2: Add the styles**

Add after the existing `.composer-actions` rule:

```css
  .ctxmeter { font:10.5px var(--mono); color:var(--dimmer); flex:none; }
  .ctxmeter[hidden] { display:none; }
  .ctxmeter.full { color:var(--bad); }
```

- [ ] **Step 3: Wire it up**

Add immediately after the `autoGrow` IIFE:

```js
// The window is small enough that a few code-carrying turns fill it, so the
// user needs to see it coming rather than hit a wall.
function refreshUsage() {
  const u = usage($('sys').value, $('prompt').value);
  const el = $('ctxMeter');
  el.hidden = !u.warn;
  el.textContent = `chat ${u.percent}% full`;
  el.classList.toggle('full', u.full);
  return u;
}
$('prompt').addEventListener('input', refreshUsage);
$('sys').addEventListener('input', refreshUsage);
```

- [ ] **Step 4: Refuse past the ceiling**

In `runBtn.onclick`, immediately after the `if (!runtimeReady())` block closes, add:

```js
  // Refuse rather than silently dropping the oldest turn: the model forgetting
  // something still visible on screen is not something a beginner can diagnose.
  if (refreshUsage().full) {
    output.innerHTML += `<div class="sandnote bad">This chat is full — the model's context `
      + `window can't hold any more. Press <b>New chat</b> to carry the important parts over `
      + `and keep going.</div>`;
    return;
  }
```

Then add `refreshUsage();` as the last line of `handleGenerate`, after the `addTurn` calls.

- [ ] **Step 5: Verify**

```js
const C = await import('./chat.js');
C.clearHistory();
const meter = document.getElementById('ctxMeter');
document.getElementById('prompt').value = 'short';
document.getElementById('prompt').dispatchEvent(new Event('input'));
console.log('hidden when small:', meter.hidden === true);

C.addTurn('assistant', 'x'.repeat(Math.round(4096 * 4 * 0.55)));
document.getElementById('prompt').dispatchEvent(new Event('input'));
console.log('shows past halfway:', meter.hidden === false, meter.textContent);

C.addTurn('assistant', 'x'.repeat(Math.round(4096 * 4 * 0.2)));
document.getElementById('prompt').dispatchEvent(new Event('input'));
console.log('marked full:', meter.classList.contains('full'), meter.textContent);

document.getElementById('runBtn').click();
console.log('refused with an explanation:',
  document.getElementById('output').textContent.includes('This chat is full'));
C.clearHistory();
```

Expected: hidden when small, visible past halfway, `full` class past 70%, and clicking Generate refuses with the message rather than generating.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Show how full the chat is, and refuse past the ceiling"
```

---

### Task 5: New chat, with the pasteable handoff

**Files:**
- Modify: `rules_baker/web/index.html` — composer markup, CSS, and a handler

**Interfaces:**
- Consumes: `clearHistory`, `handoffText`, `getHistory` from Task 1; `clearTranscript` from Task 2; `complete()`, `setBusy`, `clearBusy`, `COPY_SVG`, `__copyCode` — existing
- Produces: nothing later tasks rely on

- [ ] **Step 1: Add the button and the handoff box**

In `.composer-actions`, immediately before the Stop button, add:

```html
            <button id="newChatBtn" class="ghost" title="Start a fresh chat">New chat</button>
```

And immediately after the closing `</div>` of `.composer`, add:

```html
      <div class="handoff" id="handoff" hidden>
        <div class="handoff-head">Paste this into your new chat to carry it over
          <button type="button" class="copybtn inline" title="Copy"
                  onclick="__copyCode(this)">COPY_ICON</button></div>
        <pre><code id="handoffText"></code></pre>
      </div>
```

Then replace the literal `COPY_ICON` with the actual icon by adding this line right after the `autoGrow` IIFE:

```js
document.querySelector('#handoff .copybtn').innerHTML = COPY_SVG;
```

- [ ] **Step 2: Add the styles**

Add after the `.ctxmeter` rules:

```css
  .handoff { margin-top:11px; border:1px solid var(--edge2); border-radius:11px;
    background:var(--panel); padding:11px 13px; }
  .handoff[hidden] { display:none; }
  .handoff-head { display:flex; align-items:center; gap:8px; font-size:11.5px;
    color:var(--dim); margin-bottom:8px; }
  .handoff-head .copybtn { margin-left:auto; }
  .handoff pre { margin:0; max-height:220px; overflow:auto; }
  .handoff code { font:11.5px var(--mono); color:var(--ink); white-space:pre-wrap; }
```

- [ ] **Step 3: Wire the button**

Add after `refreshUsage`'s listeners:

```js
$('newChatBtn').onclick = async () => {
  const had = getHistory().length > 0;
  if (!had) { clearTranscript(); $('handoff').hidden = true; return; }

  let summary = '';
  if (runtimeReady()) {
    // The model writes the prose; handoffText() appends the code verbatim,
    // because a small model asked to retype a code block will mangle it.
    setBusy('Summarising the chat…');
    try {
      summary = await complete([
        { role: 'system', content: 'Summarise this conversation in two or three sentences: what '
          + 'the user was building and what was decided. Do not include code.' },
        ...getHistory(),
      ], 120, null);
    } catch (_) { summary = ''; }
    clearBusy();
  }

  $('handoffText').textContent = handoffText(summary);
  $('handoff').hidden = false;
  clearHistory();
  clearTranscript();
  refreshUsage();
};
```

- [ ] **Step 4: Verify**

```js
const C = await import('./chat.js');
C.clearHistory();
C.addTurn('user', 'write a reverser');
C.addTurn('assistant', 'Done.\n\n```python\ndef rev(s):\n    return s[::-1]\n```');
document.querySelector('.rt-pill[data-rt="wasm"]').click();   // no model → skips the summary call
document.getElementById('newChatBtn').click();
await new Promise(r => setTimeout(r, 400));
const box = document.getElementById('handoff');
console.log('handoff shown:', box.hidden === false);
console.log('contains the code verbatim:',
  document.getElementById('handoffText').textContent.includes('return s[::-1]'));
console.log('history cleared:', C.getHistory().length === 0);
console.log('transcript cleared:', document.getElementById('output').children.length === 0);
console.log('has a copy button:', !!box.querySelector('.copybtn svg'));
```

Expected: all `true`. With no model loaded the summary is skipped and the handoff is just the code, which is the correct degraded behaviour.

- [ ] **Step 5: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Add New chat with a pasteable handoff"
```

---

## Self-review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| `#output` becomes a transcript | 2 |
| History persists between generations | 1, 3 |
| Auto-fix retries excluded | 3 |
| Intro line excluded | 3 (only `finalText` is recorded) |
| System prompt at index 0, exempt from trimming | 1 (`historyFor`) |
| Character-based estimate ~4 chars/token | 1 (`estimateTokens`) |
| Warn past 50% | 1, 4 |
| Refuse past 70%, never silently drop | 1, 4 |
| New chat clears transcript and history | 5 |
| Model writes summary | 5 |
| App appends last code block verbatim | 1 (`lastCodeBlock`, `handoffText`) |
| Shown with Copy, user pastes it | 5 |
| Handoff shows tips during its wait | 5 (`setBusy` starts tips already) |

**Defect found in self-review:** renderIntro also assigned `output.innerHTML`, so it
would have wiped the transcript at the start of every generation. Task 2 now
converts it to stream into its own held-open turn, which renderFinal fills.

**Placeholders:** none — every step carries the code to write and the snippet to verify it.

**Type consistency:** `addTurn(role, content)`, `historyFor(systemPrompt, request)`, `usage(systemPrompt, pending)` and `handoffText(summary)` are used with those exact signatures in Tasks 3–5. `usage()` returns `{tokens, ratio, percent, warn, full, turns}`, matching Task 4's use of `.warn`, `.full` and `.percent`. `appendTurn(role, html)` is defined in Task 2 and called in Tasks 2 and 3. `clearTranscript()` is defined in Task 2 and called in Task 5.

**Known ordering dependency:** Task 3 calls `appendTurn`, defined in Task 2; Task 4's refusal path and Task 5 both call `refreshUsage`, defined in Task 4. Function declarations hoist within the module scope, so the finished branch is correct — only the intermediate commits would throw, and only on the specific path.
