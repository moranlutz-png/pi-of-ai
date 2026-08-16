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
