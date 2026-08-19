/* audit.js — a record of what the model was asked and what came back.
 *
 * The governance question nobody can answer about a self-hosted model is
 * "prove it did what you say it did, last Tuesday." Hosted APIs keep this for
 * you. Run the weights yourself and the record simply does not exist unless
 * you make it.
 *
 * Two deliberate constraints:
 *
 * PROMPTS ARE HASHED, NOT STORED. An audit trail that quietly accumulates
 * everything a student typed is a privacy incident waiting to happen, and in a
 * classroom it would be an indefensible one. A hash still answers the question
 * an auditor actually asks — was this the same request, how often, in what
 * order — without keeping the text.
 *
 * TRUNCATION IS COUNTED, NEVER SILENT. localStorage is finite, so old entries
 * roll off. A log that drops records without saying so is worse than no log,
 * because it looks complete. The dropped count is part of the export.
 */

const KEY = 'pi-of-ai:audit';
const MAX_ENTRIES = 500;

function read() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || '{}');
    return {
      entries: Array.isArray(raw.entries) ? raw.entries : [],
      dropped: Number.isFinite(+raw.dropped) ? +raw.dropped : 0,
    };
  } catch (_) {
    return { entries: [], dropped: 0 };
  }
}

function write(log) {
  try { localStorage.setItem(KEY, JSON.stringify(log)); return true; }
  catch (_) { return false; }        // private mode — the app keeps running
}

/** SHA-256, first 16 hex chars. Enough to match requests, useless for recovering
 *  them. Needs a secure context, which localhost counts as. */
async function hashPrompt(text) {
  try {
    const bytes = new TextEncoder().encode(String(text || ''));
    const buf = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(buf)].slice(0, 8)
      .map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (_) {
    return null;                     // no crypto available — record the absence
  }
}

/**
 * Record one generation. Everything here is a fact about the run, never its
 * content: which model, how long, how many tokens, what the sandbox made of
 * the code, and whether the user stopped it.
 *
 * outcome is one of: completed | stopped | stopped-repeating | failed.
 * stopped-repeating is distinct from stopped — the model stalled in a
 * repetition loop and the app cut it short, which is not the same event as
 * the user pressing Stop.
 */
export async function recordGeneration(f) {
  const log = read();
  log.entries.push({
    ts: new Date().toISOString(),
    model: f.model || null,
    runtime: f.runtime || null,
    promptSha256: await hashPrompt(f.prompt),
    promptChars: String(f.prompt || '').length,
    systemPromptChars: String(f.systemPrompt || '').length,
    outputTokens: f.outputTokens ?? null,
    seconds: f.seconds != null ? Math.round(f.seconds * 10) / 10 : null,
    attempts: f.attempts ?? 1,          // auto-fix retries that actually ran
    sandbox: f.sandbox || 'not-run',    // ok | problem | not-run
    outcome: f.outcome || 'completed',  // completed | stopped | stopped-repeating | failed
  });

  if (log.entries.length > MAX_ENTRIES) {
    const over = log.entries.length - MAX_ENTRIES;
    log.entries.splice(0, over);
    log.dropped += over;
  }
  write(log);
  return log.entries.length;
}

export function auditSummary() {
  const log = read();
  const first = log.entries[0];
  return {
    count: log.entries.length,
    dropped: log.dropped,
    since: first ? first.ts : null,
    stopped: log.entries.filter(e => e.outcome === 'stopped').length,
    stoppedRepeating: log.entries.filter(e => e.outcome === 'stopped-repeating').length,
    failed: log.entries.filter(e => e.outcome === 'failed').length,
    sandboxProblems: log.entries.filter(e => e.sandbox === 'problem').length,
  };
}

/** The exportable artifact. Self-describing, so it still means something to
 *  somebody who has never seen this app. */
export function auditExport() {
  const log = read();
  return {
    schema: 'pi-of-ai/audit-log/1',
    exportedAt: new Date().toISOString(),
    notice: 'Prompts are stored as SHA-256 prefixes, never as text. '
          + 'Entries beyond the retention limit are dropped and counted in droppedEntries.',
    retentionLimit: MAX_ENTRIES,
    droppedEntries: log.dropped,
    entryCount: log.entries.length,
    entries: log.entries,
  };
}

export function clearAudit() {
  try { localStorage.removeItem(KEY); } catch (_) {}
}
