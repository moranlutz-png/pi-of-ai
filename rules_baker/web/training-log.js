/* training-log.js — reading the JSON a bake produces.
 *
 * The notebook emits two files: the model, and this. The model is the point;
 * this is the evidence. A loss curve drawn from the student's OWN run is worth
 * more than a stock illustration of one, because the question it answers is
 * "did MY training actually do anything", and a picture from a textbook cannot
 * answer that.
 *
 * Everything here is defensive. The file arrives by drag-and-drop from a
 * student's Downloads folder, so it may be the wrong JSON entirely, a
 * half-written file from a Colab session that dropped, or something hand-edited
 * into nonsense. None of those may break the page — but none of them may be
 * silently rendered as an empty chart either, which would read as "your bake
 * did nothing" when the truth is "this file is not a training log".
 */

export const TRAINING_LOG_KIND = 'pi-of-ai:training-log';

/* Deliberately narrow. Number() is far too willing: Number(null), Number(''),
 * Number(false) and Number([]) are all 0 — a finite number — so a null loss
 * would sail through as a perfect 0.0 and draw the single most misleading point
 * a loss curve can have, since zero reads as "learned perfectly". Only a real
 * finite number, or a non-empty string that parses to one, counts. */
function finiteNum(v) {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/**
 * Parse and validate. Returns { ok:true, log } or { ok:false, error } — never
 * throws, so a caller can put the reason on screen rather than in the console.
 *
 * `text` is the raw file contents.
 */
export function parseTrainingLog(text) {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch (_) {
    return { ok: false, error: 'that file is not valid JSON' };
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { ok: false, error: 'that file does not contain a training log' };
  }
  // Checked explicitly rather than inferred from a `loss` key being present:
  // plenty of unrelated JSON has one, and guessing produces a confident chart
  // of somebody's unrelated data.
  if (raw.kind !== TRAINING_LOG_KIND) {
    return { ok: false, error: 'that JSON is not a Pi-of-AI training log' };
  }

  const points = Array.isArray(raw.loss) ? raw.loss : [];
  const loss = [];
  for (const p of points) {
    if (!p || typeof p !== 'object') continue;
    const step = finiteNum(p.step);
    const value = finiteNum(p.loss);
    // NaN and Infinity are exactly what a diverged run writes, and they would
    // silently poison the min/max the chart scales to.
    if (step === null || value === null) continue;
    loss.push({ step, loss: value });
  }
  if (loss.length < 2) {
    return { ok: false, error: 'this log has no loss history to draw — the run may not have finished' };
  }
  loss.sort((a, b) => a.step - b.step);

  return {
    ok: true,
    log: {
      variant: String(raw.variant || 'Unnamed variant'),
      slug: String(raw.slug || ''),
      bakedOn: String(raw.bakedOn || ''),
      baseModel: String(raw.baseModel || 'unknown'),
      target: String(raw.target || ''),
      quant: String(raw.quant || ''),
      rules: Array.isArray(raw.rules) ? raw.rules.map(String) : [],
      examples: finiteNum(raw.examples),
      epochs: finiteNum(raw.epochs),
      finalLoss: finiteNum(raw.finalLoss),
      ggufFile: String(raw.ggufFile || ''),
      loss,
    },
  };
}

/**
 * Turn the points into an SVG polyline in a 0..w / 0..h box.
 *
 * Returned as plain numbers so the caller owns the markup — this module never
 * builds HTML, which keeps the escaping question in one place.
 */
export function lossPath(loss, w, h, pad = 4) {
  const values = loss.map(p => p.loss);
  let lo = Math.min(...values), hi = Math.max(...values);
  // A perfectly flat curve has zero range, which would divide by zero and put
  // every point at NaN. Centre it instead — flat is a real, reportable result.
  if (!(hi > lo)) { hi = lo + 1; lo -= 1; }
  const x0 = loss[0].step, x1 = loss[loss.length - 1].step;
  const span = x1 - x0 || 1;
  return loss.map(p => {
    const x = pad + ((p.step - x0) / span) * (w - pad * 2);
    const y = pad + (1 - (p.loss - lo) / (hi - lo)) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}
