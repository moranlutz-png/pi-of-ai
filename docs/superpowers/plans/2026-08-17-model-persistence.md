# Model Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `.gguf` loaded from disk survives a page reload and reappears under Custom GGUFs, instead of dying with the tab.

**Architecture:** A new ES module `model-store.js` owns an IndexedDB object store holding the file Blob plus metadata. `index.html` saves after a successful local load, renders stored models in the sidebar, and loads them back. No change to how wllama consumes a file — it already accepts a `File`/`Blob`, and that is exactly what comes back out of the store.

**Tech Stack:** Vanilla ES modules, IndexedDB, no build step, no dependencies.

**Spec:** `docs/specs/2026-08-17-bake-into-weights-design.md` (the "Return" section)

## Why this is its own plan

Rung 2's spec covers three independently useful pieces: notebook generation, model persistence, and the compare view. This plan is persistence only. It is worth shipping before the rest because a dropped `.gguf` is lost on reload **today** — anyone hand-baking a model already hits it — and both other pieces depend on a baked model surviving.

## Global Constraints

- No build step. `index.html` loads one `<script type="module">`; new modules are imported from it.
- No test framework. Verification is in-browser against `http://localhost:8123`, run from the browser console.
- Every storage access wrapped in try/catch — the app assumes private-mode browsers may throw.
- Storage failures must be **reported, not silent**. The spec names this explicitly: "eviction under storage pressure must fail visibly rather than silently."
- Measured on the dev machine: quota 3.2GB, `navigator.storage.persisted()` returns **false** — storage is evictable unless persistence is requested. A 4GB Chromebook will have a smaller quota.
- Existing localStorage keys, do not collide: `pi-of-ai:variants`, `pi-of-ai:pinned`, `pi-of-ai:recent`, `pi-of-ai:measured-tokps`, `pi-of-ai:load-mbps`.
- Existing helpers in `index.html`: `$(id)` = getElementById, `esc(str)` = HTML-escape (escapes quotes as well as angle brackets). Use them; do not reimplement.

---

### Task 1: The store module

**Files:**
- Create: `rules_baker/web/model-store.js`

**Interfaces:**
- Consumes: nothing
- Produces: `saveModel(file)`, `listModels()`, `getModelBlob(id)`, `deleteModel(id)`, `storageInfo()`, `requestPersistence()`

- [ ] **Step 1: Create the module**

```js
/* model-store.js — keeps a locally-loaded .gguf across reloads.
 *
 * Without this a model dropped into the page lives only in memory: reload,
 * or let a Chromebook sleep, and it is gone. After a 40-minute bake that is
 * brutal, which is why Rung 2 depends on it.
 *
 * IndexedDB rather than localStorage (which is string-only and far too small)
 * or the Cache API (which wants Requests/Responses, not files). A Blob round-
 * trips through IndexedDB unchanged, and wllama accepts a Blob directly, so
 * nothing has to re-encode a 270MB file.
 */

const DB_NAME = 'pi-of-ai';
const DB_VERSION = 1;
const STORE = 'models';

function openDb() {
  return new Promise((resolve, reject) => {
    let req;
    try { req = indexedDB.open(DB_NAME, DB_VERSION); }
    catch (e) { reject(new Error('IndexedDB unavailable: ' + (e.message || e))); return; }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('could not open the model store'));
    req.onblocked = () => reject(new Error('model store blocked by another tab'));
  });
}

function tx(db, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const store = t.objectStore(STORE);
    let out;
    try { out = fn(store); } catch (e) { reject(e); return; }
    t.oncomplete = () => resolve(out && out.result !== undefined ? out.result : out);
    t.onerror = () => reject(t.error || new Error('store transaction failed'));
    t.onabort = () => reject(t.error || new Error('store transaction aborted'));
  });
}

// Derived from the filename so re-saving the same file replaces rather than
// duplicates it — a student who re-downloads a bake shouldn't get two copies.
function idFor(name) {
  return 'gguf:' + String(name).toLowerCase();
}

export async function saveModel(file) {
  if (!file || !file.name) return { ok: false, error: 'no file' };
  let db;
  try { db = await openDb(); } catch (e) { return { ok: false, error: e.message }; }
  const record = {
    id: idFor(file.name),
    name: file.name,
    size: file.size,
    savedAt: Date.now(),
    blob: file,
  };
  try {
    await tx(db, 'readwrite', (s) => s.put(record));
    return { ok: true, id: record.id };
  } catch (e) {
    // QuotaExceededError lands here. Report it — a silent failure would let a
    // student believe a 40-minute bake was safely stored when it was not.
    const quota = /quota/i.test(e.name || '') || /quota/i.test(e.message || '');
    return { ok: false, error: quota ? 'not enough browser storage for this model' : (e.message || 'save failed') };
  } finally { db.close(); }
}

// Metadata only — never pulls the Blobs, so rendering the sidebar stays cheap.
export async function listModels() {
  let db;
  try { db = await openDb(); } catch (_) { return []; }
  try {
    const all = await tx(db, 'readonly', (s) => s.getAll());
    return (all || [])
      .map(({ id, name, size, savedAt }) => ({ id, name, size, savedAt }))
      .sort((a, b) => (a.savedAt || 0) - (b.savedAt || 0));
  } catch (_) { return []; }
  finally { db.close(); }
}

export async function getModelBlob(id) {
  let db;
  try { db = await openDb(); } catch (_) { return null; }
  try {
    const rec = await tx(db, 'readonly', (s) => s.get(id));
    return rec ? rec.blob : null;
  } catch (_) { return null; }
  finally { db.close(); }
}

export async function deleteModel(id) {
  let db;
  try { db = await openDb(); } catch (e) { return { ok: false, error: e.message }; }
  try { await tx(db, 'readwrite', (s) => s.delete(id)); return { ok: true }; }
  catch (e) { return { ok: false, error: e.message || 'delete failed' }; }
  finally { db.close(); }
}

export async function storageInfo() {
  if (!navigator.storage || !navigator.storage.estimate) return null;
  try {
    const { quota = 0, usage = 0 } = await navigator.storage.estimate();
    const persisted = navigator.storage.persisted ? await navigator.storage.persisted() : false;
    return { quota, usage, free: Math.max(0, quota - usage), persisted };
  } catch (_) { return null; }
}

// Without this the browser may evict a stored model under disk pressure. It
// can refuse, so the result is reported rather than assumed.
export async function requestPersistence() {
  if (!navigator.storage || !navigator.storage.persist) return false;
  try { return await navigator.storage.persist(); } catch (_) { return false; }
}
```

- [ ] **Step 2: Verify the whole surface in the browser**

Server should already be running; if not:

```bash
cd ~/Projects/Dev/pi-of-ai/rules_baker/web && python3 serve.py 8123
```

Open `http://localhost:8123` and run in the console:

```js
const S = await import('./model-store.js');
const fake = new File([new Uint8Array(2 * 1024 * 1024)], 'test-bake.gguf');
console.log('save:', await S.saveModel(fake));
const list = await S.listModels();
console.log('listed:', list.length === 1, list[0].name, list[0].size === 2097152);
console.log('metadata only (no blob):', list[0].blob === undefined);
const blob = await S.getModelBlob(list[0].id);
console.log('blob back, right size:', blob instanceof Blob && blob.size === 2097152);
console.log('re-save replaces:', (await S.saveModel(fake), (await S.listModels()).length === 1));
console.log('storage:', await S.storageInfo());
console.log('delete:', await S.deleteModel(list[0].id), (await S.listModels()).length === 0);
console.log('missing id returns null:', (await S.getModelBlob('nope')) === null);
console.log('no-file guard:', await S.saveModel(null));
```

Expected: every check `true`; `storageInfo()` reports a quota; `saveModel(null)` returns `{ok:false, error:'no file'}`.

- [ ] **Step 3: Commit**

```bash
cd ~/Projects/Dev/pi-of-ai
git add rules_baker/web/model-store.js
git commit -m "Add IndexedDB store for locally-loaded models"
```

---

### Task 2: Save on load, and ask for persistence

**Files:**
- Modify: `rules_baker/web/index.html` — the module's import block, and `loadLocal`

**Interfaces:**
- Consumes: `saveModel(file)`, `requestPersistence()` from Task 1
- Produces: a saved record after any successful local load

- [ ] **Step 1: Import the module**

Below the existing `import { listVariants, ... } from './variants.js';` line, add:

```js
import { saveModel, listModels, getModelBlob, deleteModel, storageInfo,
         requestPersistence } from './model-store.js';
```

- [ ] **Step 2: Save after a successful load**

Replace the whole existing `loadLocal` function with this version:

```js
async function loadLocal(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.gguf')) { status.innerHTML = `<span style="color:var(--bad)">✗ Not a .gguf file.</span>`; return; }
  let kept = null;
  await withLoad(file.name, async () => {
    status.textContent = `Reading ${file.name} (${(file.size/1048576).toFixed(0)} MB) into WASM…`;
    await wllama.loadModel([file], { n_ctx: 4096 });
    // Only after the model actually loads — never store a file that failed,
    // and never make the user wait on storage before they can use it.
    kept = await saveModel(file);
  });
  // Reported AFTER withLoad, not inside it: withLoad writes its own success
  // line to #status once its callback resolves, which would erase anything set
  // from within.
  if (kept && kept.ok) {
    await requestPersistence();
    renderStoredModels();
  } else if (kept) {
    status.innerHTML += ` <span class="note">Couldn't keep it for next time: ${esc(kept.error)}.</span>`;
  }
}
```

- [ ] **Step 3: Verify**

Reload `http://localhost:8123`, then in the console:

```js
const S = await import('./model-store.js');
await S.saveModel(new File([new Uint8Array(1024)], 'probe.gguf'));   // stand-in for a real load
console.log('stored:', (await S.listModels()).map(m => m.name));
console.log('persistence granted:', await navigator.storage.persisted());
await S.deleteModel('gguf:probe.gguf');
```

Expected: `['probe.gguf']`. Persistence may report `false` — Chrome grants it based on engagement heuristics, so log the value rather than asserting it.

- [ ] **Step 4: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Keep a locally-loaded model after it loads successfully"
```

---

### Task 3: Show stored models in the sidebar

**Files:**
- Modify: `rules_baker/web/index.html` — markup near `#customTree`, CSS near `.varitem`, and render logic near `renderVariants`

**Interfaces:**
- Consumes: `listModels()`, `getModelBlob(id)`, `deleteModel(id)` from Task 1
- Produces: `renderStoredModels()` — called by Task 2 after a save

- [ ] **Step 1: Add the container**

Immediately after the existing `<div id="customTree"></div>` line, add:

```html
    <div id="storedTree"></div>
```

- [ ] **Step 2: Add the styles**

Add after the existing `.varitem` rules:

```css
  .storeditem { display:flex; align-items:center; gap:7px; padding:4px 8px; border-radius:6px;
    cursor:pointer; color:var(--dim); }
  .storeditem:hover { background:rgba(255,255,255,.045); color:var(--ink); }
  .storeditem .nm { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .storeditem .sz { font:10px var(--mono); color:var(--dimmer); }
  .storeditem .del { opacity:0; color:var(--dimmer); font-size:11px; }
  .storeditem:hover .del { opacity:1; }
```

- [ ] **Step 3: Render them, and load one on click**

Add immediately before `function renderVariants() {`:

```js
// Models kept from a previous session. Rendered from metadata only — the Blobs
// stay in IndexedDB until one is actually clicked.
async function renderStoredModels() {
  const models = await listModels();
  $('storedTree').innerHTML = models.map(m =>
    `<div class="storeditem" data-model-id="${esc(m.id)}" title="${esc(m.name)}">`
    + `<span class="bk">${ICONS.laptop}</span>`
    + `<span class="nm">${esc(m.name)}</span>`
    + `<span class="sz">${(m.size / 1048576).toFixed(0)} MB</span>`
    + `<span class="del">remove</span></div>`).join('');
}

document.querySelector('.sidebar').addEventListener('click', async (e) => {
  const row = e.target.closest('.storeditem');
  if (!row) return;
  const id = row.dataset.modelId;
  if (e.target.closest('.del')) {
    const res = await deleteModel(id);
    if (!res.ok) status.innerHTML = `<span style="color:var(--bad)">✗ Couldn't remove: ${esc(res.error)}</span>`;
    renderStoredModels();
    return;
  }
  const blob = await getModelBlob(id);
  if (!blob) {
    status.innerHTML = `<span style="color:var(--bad)">✗ That model is no longer in browser storage — `
      + `it may have been evicted. Load the file again.</span>`;
    renderStoredModels();
    return;
  }
  const name = row.querySelector('.nm').textContent;
  withLoad(name, () => {
    status.textContent = `Reading ${name} from browser storage…`;
    return wllama.loadModel([blob], { n_ctx: 4096 });
  });
});
```

- [ ] **Step 4: Render on startup**

Find the unconditional `renderVariants();` call that runs before `fetch('models.json')` and add below it:

```js
renderStoredModels();
```

- [ ] **Step 5: Verify**

```js
const S = await import('./model-store.js');
await S.saveModel(new File([new Uint8Array(3 * 1024 * 1024)], 'my-bake.gguf'));
location.reload();
```

After the reload:

```js
const row = document.querySelector('#storedTree .storeditem');
console.log('survived reload:', !!row);
console.log('name + size:', row.querySelector('.nm').textContent, row.querySelector('.sz').textContent);
row.querySelector('.del').click();
await new Promise(r => setTimeout(r, 300));
console.log('removed:', document.querySelectorAll('#storedTree .storeditem').length === 0);
```

Expected: the row survives the reload showing `my-bake.gguf` and `3 MB`; clicking `remove` deletes it.

- [ ] **Step 6: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Show models kept from a previous session in the sidebar"
```

---

### Task 4: Show what storage is being used

**Files:**
- Modify: `rules_baker/web/index.html` — the Settings modal's runtime pane

**Interfaces:**
- Consumes: `storageInfo()` from Task 1
- Produces: nothing later tasks rely on

A student whose model was evicted needs to know storage was the reason, and one about to bake needs to know whether there is room.

- [ ] **Step 1: Render it**

Add immediately after `renderStoredModels`'s closing brace:

```js
// Surfaced because eviction is invisible otherwise: a model simply stops being
// there, with nothing to explain why.
async function renderStorageNote() {
  const el = $('storageNote');
  if (!el) return;
  const info = await storageInfo();
  if (!info) { el.textContent = 'This browser does not report a storage quota.'; return; }
  const gb = (n) => (n / 1e9).toFixed(1);
  el.innerHTML = `Browser storage: <b>${gb(info.usage)}GB used</b> of ${gb(info.quota)}GB `
    + `(${gb(info.free)}GB free).<br>`
    + (info.persisted
      ? 'Marked persistent — models are kept until you remove them.'
      : 'Not marked persistent, so the browser may evict stored models if disk runs low.');
}
```

- [ ] **Step 2: Add the element and call it**

In the Settings modal, inside the runtime pane, immediately after the line containing `<div class="set-rows" id="setRuntime"></div>`, add:

```html
            <p class="note" id="storageNote" style="line-height:1.7; margin-top:12px"></p>
```

Then find where `openSettings` is defined and add `renderStorageNote();` as the first line of its body, so the figures are current each time the modal opens.

- [ ] **Step 3: Verify**

```js
document.getElementById('setNav').click();
await new Promise(r => setTimeout(r, 200));
console.log(document.getElementById('storageNote').textContent);
document.getElementById('setClose').click();
```

Expected: a line reading used/quota/free in GB, plus a sentence about persistence.

- [ ] **Step 4: Commit**

```bash
git add rules_baker/web/index.html
git commit -m "Report browser storage usage in Settings"
```

---

## Self-review

**Spec coverage** (the spec's "Return" section and its storage risk):

| Requirement | Task |
|---|---|
| A dropped `.gguf` survives a reload | 2, 3 |
| It reappears under Custom GGUFs | 3 |
| Storage failure reported, not silent | 1 (quota detection), 2 (message), 3 (delete + evicted-blob message) |
| Eviction is visible | 3 (missing-blob path), 4 (persistence state) |
| Request persistence | 2 |

Not covered here, and correctly so — they belong to the other two sub-plans: notebook generation, `training.json` loss curve, and the compare view.

**Placeholders:** none. Every step carries the code to write and the snippet to verify it.

**Type consistency:** `saveModel` returns `{ok, id}` or `{ok, error}`; `deleteModel` returns `{ok}` or `{ok, error}`; both are destructured that way in Tasks 2 and 3. `listModels()` returns metadata objects without `blob`, which Task 3 relies on. `storageInfo()` returns `{quota, usage, free, persisted}`, matching Task 4's use. `renderStoredModels` is defined in Task 3 and called from Task 2 — Task 2 lands first, so between the two commits that call is a forward reference resolved by hoisting, exactly as the Rung 1 plan handled the same situation.

**Known ordering note:** Task 2 calls `renderStoredModels()`, defined in Task 3. Function declarations hoist within the module scope, so the finished branch is correct; only the single commit between them would throw, and only on a successful local load.
