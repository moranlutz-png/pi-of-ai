# Model Variants (Rung 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a beginner save a named set of plain-English house rules bound to a base model, and apply it in one click — customising behaviour without touching weights.

**Architecture:** A new ES module `variants.js` owns storage and CRUD. `index.html` gains a sidebar section, an editor popup reusing the existing small-modal pattern, and wiring that applies a variant by loading its base model and writing its rules into the visible System prompt box. No new generation path — a variant is exactly equivalent to the student typing those rules themselves.

**Tech Stack:** Vanilla ES modules, no build step. `localStorage` for persistence. Served by `rules_baker/web/serve.py`.

**Spec:** `docs/specs/2026-08-16-model-variants-design.md`

## Global Constraints

- No build step. `index.html` loads one `<script type="module">` at line 576; `variants.js` is imported from it.
- No test framework exists in this repo. Verification is in-browser against `http://localhost:8123`, run from the browser console.
- Rules MUST be written into the visible `#sys` textarea, never applied invisibly. This is the spec's central decision.
- Variants MUST NOT be rendered inside `Custom GGUFs` — those are real `.gguf` files.
- `index.html` is 2147 lines. New logic goes in `variants.js`; only wiring goes in `index.html`.
- Storage key: `pi-of-ai:variants`. Existing keys in use: `pi-of-ai:pinned`, `pi-of-ai:recent`, `pi-of-ai:measured-tokps`, `pi-of-ai:load-mbps`.
- Every `localStorage` access must be wrapped in try/catch — the app already assumes private-mode browsers may throw.

---

### Task 1: Variant storage module

**Files:**
- Create: `rules_baker/web/variants.js`

**Interfaces:**
- Consumes: nothing
- Produces: `listVariants()`, `getVariant(id)`, `saveVariant(v)`, `deleteVariant(id)`, `exportVariantsJson()`, `importVariantsJson(text)`, `STARTER_RULES`

- [ ] **Step 1: Create the module**

```js
/* variants.js — named sets of house rules bound to a base model.
 *
 * A variant is Rung 1 of customisation: the rules are sent as the system
 * prompt on every request. The weights are untouched. Rung 2 (baking the same
 * rules into the weights) reuses this exact rules array as the teacher
 * instruction, which is why the shape is kept simple and portable.
 */

const KEY = 'pi-of-ai:variants';

// Seeded into a new variant so the editor never opens blank. Condensed from
// rules_baker/data_gen/rules/example_rules.md, which is the format the Rung 2
// pipeline already parses.
export const STARTER_RULES = [
  'Private helper functions must start with a single underscore.',
  'Never use a bare except; catch specific exception types.',
  'All function signatures must have complete type hints.',
  'Use the module logger, never print(), for anything diagnostic.',
];

function readAll() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '[]');
    return Array.isArray(raw) ? raw.filter(isVariant) : [];
  } catch (_) {
    return [];
  }
}

function writeAll(list) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
    return true;
  } catch (_) {
    return false;            // private mode — caller keeps working in memory
  }
}

function isVariant(v) {
  return v && typeof v === 'object'
    && typeof v.id === 'string'
    && typeof v.name === 'string'
    && Array.isArray(v.rules);
}

export function listVariants() {
  return readAll().sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
}

export function getVariant(id) {
  return readAll().find(v => v.id === id) || null;
}

// Upserts. Assigns id and createdAt when absent, so callers can pass a bare
// object for a new variant and the saved one back for an edit.
export function saveVariant(input) {
  const list = readAll();
  const v = {
    id: input.id || (crypto.randomUUID ? crypto.randomUUID() : 'v' + Date.now()),
    name: String(input.name || '').trim() || 'Untitled variant',
    rules: (input.rules || []).map(r => String(r).trim()).filter(Boolean),
    baseModelUrl: String(input.baseModelUrl || ''),
    temp: Number(input.temp) || 0.2,
    maxTokens: Number(input.maxTokens) || 256,
    createdAt: input.createdAt || Date.now(),
  };
  const i = list.findIndex(x => x.id === v.id);
  if (i >= 0) list[i] = v; else list.push(v);
  writeAll(list);
  return v;
}

export function deleteVariant(id) {
  writeAll(readAll().filter(v => v.id !== id));
}

export function exportVariantsJson() {
  return JSON.stringify({ kind: 'pi-of-ai:variants', version: 1, variants: listVariants() }, null, 2);
}

// Accepts either the wrapper written by exportVariantsJson or a bare array.
// Existing ids are kept and overwritten, so re-importing your own file is
// idempotent rather than producing duplicates.
export function importVariantsJson(text) {
  let parsed;
  try { parsed = JSON.parse(text); } catch (_) { return { added: 0, skipped: 0, error: 'Not valid JSON' }; }
  const incoming = Array.isArray(parsed) ? parsed : (parsed && parsed.variants);
  if (!Array.isArray(incoming)) return { added: 0, skipped: 0, error: 'No variants in that file' };
  let added = 0, skipped = 0;
  for (const v of incoming) {
    if (!isVariant(v)) { skipped++; continue; }
    saveVariant(v);
    added++;
  }
  return { added, skipped, error: null };
}
```

- [ ] **Step 2: Verify it loads and round-trips**

Start the server if not running:

```bash
cd ~/Projects/Dev/pi-of-ai/rules_baker/web && python3 serve.py 8123
```

Open `http://localhost:8123` and run in the console:

```js
const V = await import('./variants.js');
localStorage.removeItem('pi-of-ai:variants');
const a = V.saveVariant({ name: 'Strict typing', rules: ['Always use type hints', ''], baseModelUrl: 'x', temp: 0.5, maxTokens: 512 });
console.log('saved:', a.id && a.rules.length === 1);          // blank rule dropped
console.log('listed:', V.listVariants().length === 1);
V.saveVariant({ ...a, name: 'Renamed' });
console.log('upserted not duplicated:', V.listVariants().length === 1, V.listVariants()[0].name === 'Renamed');
const json = V.exportVariantsJson();
localStorage.removeItem('pi-of-ai:variants');
console.log('import:', V.importVariantsJson(json), V.listVariants().length === 1);
console.log('bad import:', V.importVariantsJson('not json').error);
V.deleteVariant(a.id);
console.log('deleted:', V.listVariants().length === 0);
```

Expected: every check `true`, bad import reports `Not valid JSON`.

- [ ] **Step 3: Commit**

```bash
cd ~/Projects/Dev/pi-of-ai
git add rules_baker/web/variants.js
git commit -m "Add variant storage module"
```

---

### Task 2: Make the Temp/Tokens dropdowns reflect programmatic changes

**Files:**
- Modify: `rules_baker/web/index.html` — inside `makeDropdown`, after `render();`

**Interfaces:**
- Consumes: nothing
- Produces: setting `#temp`/`#maxtok` `.value` then dispatching `change` updates the visible pill

`makeDropdown` builds the pill label inside a closure and only re-renders on
click. Applying a variant sets `select.value` in code, so without this the pill
keeps showing the old value while generation uses the new one — a silent
mismatch between what is displayed and what is sent.

- [ ] **Step 1: Confirm the bug first**

Run in the console on `http://localhost:8123`:

```js
const before = document.querySelector('.pills .dd-btn .lbl').textContent;
document.getElementById('temp').value = '0.8';
document.getElementById('temp').dispatchEvent(new Event('change'));
console.log('label before:', before, '| after:', document.querySelector('.pills .dd-btn .lbl').textContent);
```

Expected: both read the same (e.g. `Temp · 0.2`) — the label did not follow.

- [ ] **Step 2: Add the listener**

Find `render();` near the end of `makeDropdown` (immediately before
`document.addEventListener('click', ...)`) and add below it:

```js
  // Keep the pill in step when the value is set in code — applying a variant
  // does exactly that, and a stale label would misreport what is being sent.
  sel.addEventListener('change', render);
```

- [ ] **Step 3: Verify the fix**

Reload and run the Step 1 snippet again.
Expected: label after reads `Temp · 0.8`.

- [ ] **Step 4: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Keep dropdown labels in step with programmatic value changes"
```

---

### Task 3: Sidebar section and variant rendering

**Files:**
- Modify: `rules_baker/web/index.html:427` (markup, above the `Custom GGUFs` seclabel)
- Modify: `rules_baker/web/index.html` — CSS block, near `.modelnote`
- Modify: `rules_baker/web/index.html:908` `renderModelTree` — call the new renderer

**Interfaces:**
- Consumes: `listVariants()` from Task 1
- Produces: `renderVariants()`; rows carry `data-variant-id`

- [ ] **Step 1: Add the markup**

Insert immediately **above** the existing `<div class="seclabel">Custom GGUFs</div>` line:

```html
    <div class="seclabel">Your variants</div>
    <div id="variantTree"></div>
    <div class="navlink" id="newVariantNav"><span data-ic="plus"></span>New variant</div>

```

- [ ] **Step 2: Add the styles**

Add after the `.modelnote` rule:

```css
  /* A variant is rules, not weights — it must not read as a model. */
  .varitem { display:flex; align-items:center; gap:7px; padding:4px 8px; border-radius:6px;
    cursor:pointer; color:var(--dim); }
  .varitem:hover { background:rgba(255,255,255,.045); color:var(--ink); }
  .varitem.active { background:var(--panel2); color:var(--ink); }
  .varitem .nm { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .varitem .rulecount { font:10px var(--mono); color:var(--dimmer); }
  .varitem .edit { opacity:0; color:var(--dimmer); font-size:11px; }
  .varitem:hover .edit { opacity:1; }
```

- [ ] **Step 3: Import the module and render**

At the top of the `<script type="module">` block, below the existing `import { Wllama }` line:

```js
import { listVariants, getVariant, saveVariant, deleteVariant,
         exportVariantsJson, importVariantsJson, STARTER_RULES } from './variants.js';
```

Add this function immediately before `function renderModelTree(models) {`:

```js
// Rendered separately from the model tree: variants are rules, not GGUFs, and
// mixing them into Custom GGUFs would teach exactly the wrong thing.
function renderVariants() {
  const list = listVariants();
  $('variantTree').innerHTML = list.length
    ? list.map(v => `<div class="varitem" data-variant-id="${esc(v.id)}" `
        + `title="${esc(v.rules.join(' · '))}">`
        + `<span class="nm">${esc(v.name)}</span>`
        + `<span class="rulecount">${v.rules.length} rule${v.rules.length === 1 ? '' : 's'}</span>`
        + `<span class="edit">edit</span></div>`).join('')
    : '<div class="modelnote">None yet — a variant is a saved set of house rules, '
      + 'sent with every request.</div>';
}
```

Call it at the end of `renderModelTree`, immediately before the closing `}`:

```js
  renderVariants();
```

- [ ] **Step 4: Verify**

Reload, then in the console:

```js
const V = await import('./variants.js');
V.saveVariant({ name: 'Strict typing', rules: ['Always use type hints', 'No bare except'], baseModelUrl: 'x' });
location.reload();
```

After reload:

```js
console.log('section present:', [...document.querySelectorAll('.seclabel')].map(s => s.textContent));
console.log('rendered:', document.querySelectorAll('#variantTree .varitem').length === 1);
console.log('rule count shown:', document.querySelector('.varitem .rulecount').textContent === '2 rules');
console.log('not inside custom ggufs:', !document.querySelector('#customTree .varitem'));
```

Expected: sections include `Your variants`; all checks `true`.

- [ ] **Step 5: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Add Your variants sidebar section"
```

---

### Task 4: Applying a variant

**Files:**
- Modify: `rules_baker/web/index.html` — after `renderVariants`

**Interfaces:**
- Consumes: `getVariant(id)`; `#sys`, `#sysWrap`, `#temp`, `#maxtok`, `urlIn`, `loadBtn`
- Produces: `applyVariant(id)`

- [ ] **Step 1: Add the apply function**

```js
// Applying a variant writes its rules into the VISIBLE system prompt box. It
// deliberately does not use a hidden channel: the student needs to see the
// rules occupying the context window, because removing that cost is the whole
// point of baking them into the weights later.
function applyVariant(id) {
  const v = getVariant(id);
  if (!v) return;

  document.querySelectorAll('.varitem').forEach(el =>
    el.classList.toggle('active', el.dataset.variantId === id));

  $('sys').value = v.rules.join('\n');
  $('sysWrap').hidden = false;                       // reveal, don't hide the mechanism
  const sign = document.querySelector('#sysToggle .sign');
  if (sign) sign.textContent = '−';

  $('temp').value = String(v.temp);
  $('temp').dispatchEvent(new Event('change'));      // needs Task 2
  $('maxtok').value = String(v.maxTokens);
  $('maxtok').dispatchEvent(new Event('change'));

  const target = v.baseModelUrl
    && document.querySelector(`.modelitem[data-url="${CSS.escape(v.baseModelUrl)}"]`);
  if (!target) {
    status.innerHTML = `<span style="color:var(--bad)">“${esc(v.name)}” points at a model that `
      + `is no longer in the library. Its rules are applied — pick a model to load.</span>`;
    return;
  }
  target.click();                                    // selects it and fills the URL box
  if (loadedUrl !== v.baseModelUrl) loadBtn.click();
}

document.querySelector('.sidebar').addEventListener('click', (e) => {
  const row = e.target.closest('.varitem');
  if (!row) return;
  if (e.target.closest('.edit')) openVariantEditor(row.dataset.variantId);
  else applyVariant(row.dataset.variantId);
});
```

- [ ] **Step 2: Verify (without triggering a model download)**

```js
const V = await import('./variants.js');
localStorage.setItem('pi-of-ai:variants', '[]');
const url = document.querySelector('.modelitem').dataset.url;
V.saveVariant({ name: 'Terse', rules: ['Be terse', 'No comments'], baseModelUrl: url, temp: 0.8, maxTokens: 512 });
location.reload();
```

After reload — note this DOES start a model load, so use a model already cached, or expect the download:

```js
document.querySelector('.varitem').click();
console.log('rules in the visible box:', document.getElementById('sys').value);
console.log('prompt section revealed:', document.getElementById('sysWrap').hidden === false);
console.log('temp applied:', document.getElementById('temp').value === '0.8');
console.log('temp label follows:', document.querySelector('.pills .dd-btn .lbl').textContent);
console.log('row marked active:', document.querySelector('.varitem.active') !== null);
```

Expected: two rules on separate lines, section revealed, temp `0.8`, label reads `Temp · 0.8`, row active.

Then verify the missing-model path:

```js
const V = await import('./variants.js');
V.saveVariant({ name: 'Orphan', rules: ['x'], baseModelUrl: 'https://example.invalid/none.gguf' });
location.reload();
```

```js
[...document.querySelectorAll('.varitem')].find(r => r.textContent.includes('Orphan')).click();
console.log('warned, did not throw:', document.getElementById('status').textContent);
```

Expected: message naming the variant; no console error.

- [ ] **Step 3: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Apply a variant: load its base model and reveal its rules"
```

---

### Task 5: The variant editor popup

**Files:**
- Modify: `rules_baker/web/index.html:515` — add a modal beside `#ollamaModal`
- Modify: `rules_baker/web/index.html` — editor logic after `applyVariant`

**Interfaces:**
- Consumes: `saveVariant`, `deleteVariant`, `exportVariantsJson`, `importVariantsJson`, `STARTER_RULES`
- Produces: `openVariantEditor(id)` — called by Task 4's click handler and the `New variant` nav link

- [ ] **Step 1: Add the modal markup**

Insert immediately **before** `<div id="ollamaModal" class="modal" hidden>`:

```html
  <!-- ============ VARIANT EDITOR (small popup) ============ -->
  <div id="varModal" class="modal" hidden>
    <div class="modal-panel small">
      <button class="modal-close" id="varClose" title="Close">&times;</button>
      <h2 class="modal-title" id="varTitle">New variant</h2>
      <div class="rt-body">
        <label class="varlabel" for="varName">Name</label>
        <input type="text" id="varName" class="url" placeholder="e.g. Strict typing" />

        <label class="varlabel" for="varRules">House rules — one per line</label>
        <textarea id="varRules" rows="6"></textarea>
        <div class="dnote" id="varRuleNote"></div>

        <label class="varlabel" for="varBase">Base model</label>
        <select id="varBase" class="varselect"></select>

        <div class="varrow">
          <span><label class="varlabel" for="varTemp">Temperature</label>
            <select id="varTemp" class="varselect">
              <option value="0.1">0.1</option><option value="0.2" selected>0.2</option>
              <option value="0.5">0.5</option><option value="0.8">0.8</option>
            </select></span>
          <span><label class="varlabel" for="varTokens">Max tokens</label>
            <select id="varTokens" class="varselect">
              <option value="128">128</option><option value="256" selected>256</option>
              <option value="512">512</option>
            </select></span>
        </div>

        <div class="dnote" style="margin-top:10px">These rules are sent with every request, so
          they take up room in the context window. Baking them into the weights instead is what
          <b>Bake</b> will do.</div>
      </div>
      <div class="dactions">
        <button type="button" class="act" id="varSave">Save variant</button>
        <button type="button" class="ghost" id="varBake" disabled
          title="Rung 2 — not built yet">Bake into weights</button>
        <button type="button" class="ghost" id="varDelete" hidden>Delete</button>
      </div>
      <div class="dactions" style="margin-top:8px">
        <button type="button" class="backbtn" id="varExport">Export all</button>
        <button type="button" class="backbtn" id="varImport">Import…</button>
        <input type="file" id="varImportFile" accept=".json,application/json" hidden />
      </div>
    </div>
  </div>
```

- [ ] **Step 2: Add the styles**

Add after the `.varitem` rules from Task 3:

```css
  .varlabel { display:block; margin:11px 0 4px; font-size:11px; letter-spacing:.05em;
    text-transform:uppercase; color:var(--dimmer); }
  .varselect { background:var(--panel); color:var(--ink); border:1px solid var(--edge2);
    border-radius:7px; padding:5px 9px; font:inherit; font-size:12.5px; }
  .varrow { display:flex; gap:16px; }
  #varModal textarea { width:100%; }
```

- [ ] **Step 3: Add the editor logic**

```js
const varModal = $('varModal');
let editingId = null;

function populateBasePicker(selected) {
  const opts = [...document.querySelectorAll('.modelitem')].map(el =>
    `<option value="${esc(el.dataset.url)}"${el.dataset.url === selected ? ' selected' : ''}>`
    + `${esc(el.querySelector('.nm').textContent.trim())}</option>`).join('');
  $('varBase').innerHTML = opts || '<option value="">No models available</option>';
}

function updateRuleNote() {
  const rules = $('varRules').value.split('\n').map(r => r.trim()).filter(Boolean);
  // Rough: ~4 characters per token. Enough to warn, not precise enough to quote.
  const approxTokens = Math.round($('varRules').value.length / 4);
  $('varRuleNote').textContent = `${rules.length} rule${rules.length === 1 ? '' : 's'} · `
    + `roughly ${approxTokens} tokens of the 4096-token window, on every request.`;
}

function openVariantEditor(id) {
  editingId = id || null;
  const v = id ? getVariant(id) : null;
  $('varTitle').textContent = v ? 'Edit variant' : 'New variant';
  $('varName').value = v ? v.name : '';
  $('varRules').value = (v ? v.rules : STARTER_RULES).join('\n');
  populateBasePicker(v ? v.baseModelUrl : (document.querySelector('.modelitem.active') || {}).dataset?.url);
  $('varTemp').value = String(v ? v.temp : 0.2);
  $('varTokens').value = String(v ? v.maxTokens : 256);
  $('varDelete').hidden = !v;
  updateRuleNote();
  varModal.hidden = false;
  $('varName').focus();
}

$('varRules').addEventListener('input', updateRuleNote);
$('newVariantNav').onclick = () => openVariantEditor(null);
$('varClose').onclick = () => { varModal.hidden = true; };
varModal.addEventListener('click', (e) => { if (e.target === varModal) varModal.hidden = true; });
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !varModal.hidden) varModal.hidden = true;
});

$('varSave').onclick = () => {
  saveVariant({
    id: editingId,
    name: $('varName').value,
    rules: $('varRules').value.split('\n'),
    baseModelUrl: $('varBase').value,
    temp: parseFloat($('varTemp').value),
    maxTokens: parseInt($('varTokens').value, 10),
  });
  varModal.hidden = true;
  renderVariants();
};

$('varDelete').onclick = () => {
  if (editingId) deleteVariant(editingId);
  varModal.hidden = true;
  renderVariants();
};

$('varExport').onclick = () => {
  const blob = new Blob([exportVariantsJson()], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pi-of-ai-variants.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

$('varImport').onclick = () => $('varImportFile').click();
$('varImportFile').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const res = importVariantsJson(await file.text());
  renderVariants();
  $('varRuleNote').textContent = res.error
    ? `Import failed: ${res.error}`
    : `Imported ${res.added} variant${res.added === 1 ? '' : 's'}.`;
  e.target.value = '';
};
```

- [ ] **Step 4: Verify**

```js
localStorage.setItem('pi-of-ai:variants', '[]');
location.reload();
```

After reload:

```js
document.getElementById('newVariantNav').click();
console.log('opens with starter rules:', document.getElementById('varRules').value.split('\n').length === 4);
console.log('note shown:', document.getElementById('varRuleNote').textContent);
console.log('bake disabled:', document.getElementById('varBake').disabled === true);
console.log('base picker filled:', document.querySelectorAll('#varBase option').length > 1);
document.getElementById('varName').value = 'My rules';
document.getElementById('varSave').click();
console.log('saved and listed:', document.querySelectorAll('#variantTree .varitem').length === 1);

// edit path
document.querySelector('.varitem .edit').click();
console.log('edit title:', document.getElementById('varTitle').textContent === 'Edit variant');
console.log('delete visible when editing:', document.getElementById('varDelete').hidden === false);
document.getElementById('varDelete').click();
console.log('deleted:', document.querySelectorAll('#variantTree .varitem').length === 0);
```

Expected: all `true`; the note reads a rule count and an approximate token cost.

- [ ] **Step 5: Verify export downloads a valid file**

Click **Export all** with at least one variant saved, then confirm the
downloaded `pi-of-ai-variants.json` parses and contains a `variants` array.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Add the variant editor popup"
```

---

## Self-review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| Data model with id/name/rules/baseModelUrl/temp/maxTokens/createdAt | 1 |
| `localStorage` under `pi-of-ai:variants` | 1 |
| Export / Import | 1 (logic), 5 (buttons) |
| Loads base model on selection | 4 |
| Writes rules into the visible System prompt box | 4 |
| Applies temp and maxTokens | 4 (needs 2) |
| Missing base model warns rather than failing silently | 4 |
| `Your variants` section above Custom GGUFs | 3 |
| Marker distinguishing a variant from a model | 3 (`.varitem`, rule count) |
| Editor popup reusing `.modal-panel.small` | 5 |
| Starter rules pre-filled | 1 (`STARTER_RULES`), 5 |
| Disabled Bake button with a note | 5 |
| Honesty line about the context window | 5 |
| Rule-length indication (spec Risks) | 5 (`updateRuleNote`) |

**Placeholders:** none — every step carries the code to write and the snippet to verify it.

**Type consistency:** `renderVariants`, `applyVariant`, `openVariantEditor`, `updateRuleNote`, `populateBasePicker` are each defined once and referenced by those exact names. `STARTER_RULES` is exported in Task 1 and consumed in Task 5. `data-variant-id` is written in Task 3 and read in Task 4.

**Known ordering dependency:** Task 4 depends on Task 2. Applying a variant sets `#temp`/`#maxtok` in code, and without Task 2 the pill labels keep the old text while generation uses the new values.
