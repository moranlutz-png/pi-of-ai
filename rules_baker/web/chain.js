/* chain.js — a deliberately small agent loop, with its guards on show.
 *
 * This is not an agent framework. It is the smallest thing that can actually
 * fail the way agent frameworks fail, running on a model too small to hide it.
 *
 * Every production agent harness converges on the same three guards, and they
 * are always described as edge cases. On a 135M model they are not edge cases,
 * they are Tuesday — which is exactly why this belongs in a teaching kit:
 *
 *   STEP BUDGET   a hard ceiling on steps. Without one, "keep going until done"
 *                 means "keep going", because a small model rarely emits a
 *                 clean stop. This is the guard that costs real money in
 *                 production and the one people add last.
 *
 *   LOOP DETECT   models repeat themselves. The same action twice in a row is
 *                 usually a stall, not progress. Detecting it needs a notion of
 *                 "the same", which is where naive implementations go wrong —
 *                 exact string match misses paraphrase, so this normalises.
 *
 *   VISIBLE STATE the scratchpad the steps read and write. Agent bugs are
 *                 almost always state bugs, and they are invisible unless the
 *                 state is shown at every step rather than only at the end.
 *
 * The lesson lands when a chain hits a guard, which on a small model it usually
 * will. A run that trips the loop detector after three identical steps teaches
 * more than one that happens to succeed.
 */

export const STOP_REASONS = {
  done: 'the model said it was finished',
  budget: 'hit the step budget',
  loop: 'repeated itself',
  aborted: 'stopped by you',
  error: 'a step failed',
};

/** Normalised for comparison: case, punctuation and whitespace removed, so a
 *  model that rephrases the same action is still caught repeating itself. */
function fingerprint(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 160);
}

/**
 * Run the loop.
 *
 * `step` is supplied by the caller and does one turn: given the goal and the
 * state so far, return { action, output, done }. chain.js never talks to a
 * model itself, so the same guards work on any runtime.
 *
 * `onEvent` receives every guard decision as it happens — the point is that the
 * guards are watchable, not that they are silent.
 */
export async function runChain({ goal, step, maxSteps = 6, onEvent = () => {}, shouldAbort = () => false } = {}) {
  const state = { goal, steps: [], notes: [] };
  const seen = [];
  let stop = null;

  for (let i = 1; i <= maxSteps; i++) {
    if (shouldAbort()) { stop = 'aborted'; break; }

    onEvent({ type: 'step-start', n: i, of: maxSteps });

    let result;
    try {
      result = await step({ goal, state, n: i });
    } catch (e) {
      state.notes.push(`step ${i} failed: ${e.message || e}`);
      onEvent({ type: 'error', n: i, message: e.message || String(e) });
      stop = 'error';
      break;
    }

    const action = String(result?.action || '').trim();
    const entry = { n: i, action, output: result?.output ?? null };
    state.steps.push(entry);
    onEvent({ type: 'step-done', ...entry });

    // Loop guard BEFORE the done check: a model that has started repeating
    // itself will also happily claim it is finished, and the honest report is
    // that it stalled rather than that it succeeded.
    const fp = fingerprint(action);
    if (fp && seen.includes(fp)) {
      const firstAt = seen.indexOf(fp) + 1;
      onEvent({ type: 'loop', n: i, firstAt, action });
      stop = 'loop';
      break;
    }
    seen.push(fp);

    if (result?.done) { stop = 'done'; break; }
  }

  if (!stop) {
    stop = 'budget';
    onEvent({ type: 'budget', maxSteps });
  }

  return {
    goal,
    stopped: stop,
    reason: STOP_REASONS[stop],
    stepsRun: state.steps.length,
    maxSteps,
    // Reported plainly: a chain that ran out of budget did NOT finish the job,
    // and a harness that presents the last output as an answer is lying about
    // what happened.
    completed: stop === 'done',
    state,
  };
}
