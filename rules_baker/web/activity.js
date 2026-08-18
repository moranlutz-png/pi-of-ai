/* activity.js — the practice heatmap's data.
 *
 * Nothing in this app recorded WHEN anything happened. Variants, pins, recents
 * and measured speeds are all "what", never "when", so a student had no way to
 * see their own practice accumulating — which for a teaching kit is most of the
 * motivation. This keeps one counter per calendar day.
 *
 * A day counts if the student did anything at all: generated, loaded a model,
 * baked, or edited a variant. One square per day, shaded by how much.
 *
 * localStorage rather than IndexedDB: this is a few hundred bytes a year of
 * plain integers, and it must survive the same eviction that can take a stored
 * .gguf. Losing your model is bad; losing your streak as well would be worse.
 */

const KEY = 'pi-of-ai:activity';

// Local date, not ISO/UTC — toISOString() would roll the day over at 01:00 for
// anyone east of UTC, so an evening session in Berlin would land on tomorrow.
function dayKey(d) {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function read() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '{}');
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
    // Drop anything that isn't a date->positive-integer pair, so one bad write
    // (or a hand-edited store) can't make the grid throw on render.
    const out = {};
    for (const [k, v] of Object.entries(raw)) {
      if (/^\d{4}-\d{2}-\d{2}$/.test(k) && Number.isFinite(+v) && +v > 0) out[k] = Math.floor(+v);
    }
    return out;
  } catch (_) {
    return {};                       // private mode, or corrupt — start empty
  }
}

function write(map) {
  try { localStorage.setItem(KEY, JSON.stringify(map)); return true; }
  catch (_) { return false; }        // private mode — the app keeps working
}

/** Count one thing the student did today. Cheap and idempotent-ish by design:
 *  callers fire it freely and the day is what matters, not the exact tally. */
export function recordActivity(when = new Date()) {
  const map = read();
  const k = dayKey(when);
  map[k] = (map[k] || 0) + 1;
  write(map);
  return map[k];
}

export function activityMap() {
  return read();
}

export function clearActivity() {
  try { localStorage.removeItem(KEY); } catch (_) {}
}

/** Years that have data, plus the current year, newest first — the dropdown. */
export function activityYears(today = new Date()) {
  const years = new Set([today.getFullYear()]);
  for (const k of Object.keys(read())) years.add(+k.slice(0, 4));
  return [...years].sort((a, b) => b - a);
}

// Shading thresholds. Fixed rather than quartiles of the user's own data: with
// quartiles a quiet week would light up as darkly as a heavy one, which reads
// as progress that isn't there.
function levelFor(count) {
  if (!count) return 0;
  if (count <= 2) return 1;
  if (count <= 5) return 2;
  if (count <= 9) return 3;
  return 4;
}

/**
 * The grid, as data — index.html renders it.
 *
 * Columns are weeks and rows are days, GitHub-style, so the first column starts
 * on the Sunday on or before Jan 1 and may carry a few empty cells for the tail
 * of the previous December. For the current year the grid stops at today rather
 * than drawing four months of empty future.
 */
export function heatmapModel(year, today = new Date()) {
  const map = read();
  const isCurrentYear = year === today.getFullYear();

  const start = new Date(year, 0, 1);
  const end = isCurrentYear ? new Date(today.getFullYear(), today.getMonth(), today.getDate())
                            : new Date(year, 11, 31);

  // Back up to the Sunday that starts the first column.
  const cursor = new Date(start);
  cursor.setDate(cursor.getDate() - cursor.getDay());

  const weeks = [];
  const monthLabels = [];
  let lastMonth = -1;
  let total = 0, activeDays = 0, daysInRange = 0;

  while (cursor <= end) {
    const week = [];
    for (let dow = 0; dow < 7; dow++) {
      const inRange = cursor >= start && cursor <= end;
      if (inRange) {
        const key = dayKey(cursor);
        const count = map[key] || 0;
        week.push({ date: key, count, level: levelFor(count) });
        total += count;
        daysInRange++;
        if (count > 0) activeDays++;
        // Label a column with a month name the first time that month appears.
        if (cursor.getMonth() !== lastMonth && cursor.getDate() <= 7) {
          lastMonth = cursor.getMonth();
          monthLabels.push({ week: weeks.length, month: cursor.getMonth() });
        }
      } else {
        week.push(null);             // padding before Jan 1 / after today
      }
      cursor.setDate(cursor.getDate() + 1);
    }
    weeks.push(week);
  }

  return { year, weeks, monthLabels, total, activeDays, days: daysInRange };
}
