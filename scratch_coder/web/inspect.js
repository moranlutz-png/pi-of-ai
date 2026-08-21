// inspect.js — parse and validate what export_inspect.py wrote. Validate hard,
// never throw: the caller puts the reason on screen. Same argument training-log.js
// makes in the other build — read it before changing this one.

export const INSPECT_KIND = 'pi-of-ai:scratch-inspect';

// Number(null) is 0, and a 0.0 statistic reads as "exactly zero" when it is really
// "absent". Every value that comes off the JSON goes through this first.
export function finiteNum(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

// Full inspect file: architecture + trained/random statistics + embedding.
export function parseInspect(text) {
  let d;
  try { d = JSON.parse(text); }
  catch (e) { return { ok: false, error: 'not valid JSON: ' + e.message }; }
  if (!d || typeof d !== 'object') return { ok: false, error: 'empty file' };
  if (d.kind !== INSPECT_KIND) return { ok: false, error: `not a ${INSPECT_KIND} file (kind=${d.kind})` };
  if (!d.arch || !Array.isArray(d.arch.groups) || d.arch.groups.length === 0)
    return { ok: false, error: 'no arch.groups in the file' };
  if (!d.trained || !d.random) return { ok: false, error: 'missing trained/random statistics' };
  return { ok: true, data: d };
}

// Structure-only fallback (arch_map.py --json). Lets the page draw the model with
// no checkpoint at all — every tensor and its shape, values dark until an export.
export function parseArch(text) {
  let d;
  try { d = JSON.parse(text); }
  catch (e) { return { ok: false, error: 'not valid JSON: ' + e.message }; }
  if (!d || !Array.isArray(d.groups) || d.groups.length === 0)
    return { ok: false, error: 'no groups in arch.json' };
  return { ok: true, data: d };
}
