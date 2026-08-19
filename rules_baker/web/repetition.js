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
const REPEATS_TO_STOP = 4;  // three is a stylistic flourish; four is a stall (word/phrase units)
// Closing-brace and `return` sequences legitimately stack a few deep in real
// code, so a repeated line needs a higher bar than a repeated word before it
// counts as a stall: six identical substantive lines in a row is not structure.
const LINE_REPEATS_TO_STOP = 6;

/** Does the tail of `text` consist of one short unit repeated? */
export function looksDegenerate(text) {
  const s = String(text || '');
  if (s.length < MIN_TAIL) return { repeating: false, phrase: null, times: 0 };

  // Look only at the tail: a reply that legitimately repeated something early
  // and then moved on is not stuck.
  const tail = s.slice(-400);

  // Try progressively longer units. 1-3 words catches token loops, longer
  // catches sentence and line loops. Each unit carries the repeat count that
  // applies to it — word/phrase units use REPEATS_TO_STOP, the line unit uses
  // the stricter LINE_REPEATS_TO_STOP.
  for (const unit of splitUnits(tail)) {
    const times = trailingRepeats(tail, unit.text);
    if (times >= unit.minRepeats) {
      return { repeating: true, phrase: unit.text.trim(), times };
    }
  }
  return { repeating: false, phrase: null, times: 0 };
}

function splitUnits(tail) {
  const words = tail.trim().split(/\s+/);
  const units = [];
  for (const n of [1, 2, 3, 5, 8]) {
    if (words.length >= n * REPEATS_TO_STOP) {
      units.push({ text: words.slice(-n).join(' ') + ' ', minRepeats: REPEATS_TO_STOP });
    }
  }
  const lines = tail.split('\n').filter(Boolean);
  if (lines.length >= LINE_REPEATS_TO_STOP) {
    const lastLine = lines[lines.length - 1];
    // A repeated line only means something if it carries real content — skip
    // structural lines like `}`, `)`, `];` which stack legitimately in nested
    // JS/TS/JSON/C-like code and carry no meaning when repeated.
    const alnumCount = (lastLine.match(/[a-z0-9]/gi) || []).length;
    if (alnumCount >= 3) {
      units.push({ text: lastLine + '\n', minRepeats: LINE_REPEATS_TO_STOP });
    }
  }
  return units;
}

/** How many times does `unit` repeat, back to back, at the end of `tail`? */
function trailingRepeats(tail, unit) {
  if (!unit.trim()) return 0;
  // Collapse horizontal whitespace runs only — a newline is a real boundary,
  // not interchangeable with a space. A word/phrase unit is joined with, and
  // padded by, a space; the line unit is joined with, and padded by, a
  // newline (see splitUnits). Blurring the two together would let a line
  // separated by newlines (e.g. four `}` lines, or `return 1` once per line)
  // satisfy a plain word/phrase unit's space-separated pattern and fire the
  // lower, word-level bar instead of the line-level one it should be judged
  // against.
  const norm = (x) => x.toLowerCase().replace(/[ \t]+/g, ' ');
  const u = norm(unit);
  let t = norm(tail);
  // `unit` always carries a trailing separator (see splitUnits), but real
  // streamed text almost never ends ON a separator — the last emitted token
  // is content, not the space or newline after it. Pad with that same
  // separator so the final in-progress instance still lines up with `unit`,
  // instead of only ever matching text that happens to end mid-whitespace.
  const sep = u.slice(-1);
  if (!t.endsWith(sep)) t += sep;
  let count = 0;
  while (t.endsWith(u) && count < 64) {
    t = t.slice(0, -u.length);
    count++;
  }
  return count;
}
