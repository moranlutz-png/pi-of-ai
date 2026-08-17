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
    t.oncomplete = () => resolve(out instanceof IDBRequest ? out.result : out);
    t.onerror = () => reject((out instanceof IDBRequest && out.error) || t.error || new Error('store transaction failed'));
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
