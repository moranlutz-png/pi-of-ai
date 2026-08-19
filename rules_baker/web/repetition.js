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
  // `unit` always carries a trailing separator (see splitUnits), but real
  // streamed text almost never ends ON a separator — the last emitted token
  // is content, not the space or newline after it. Pad so the final in-progress
  // instance still lines up with `unit`, instead of only ever matching text
  // that happens to end mid-whitespace.
  if (!t.endsWith(' ')) t += ' ';
  let count = 0;
  while (t.endsWith(u) && count < 64) {
    t = t.slice(0, -u.length);
    count++;
  }
  return count;
}
