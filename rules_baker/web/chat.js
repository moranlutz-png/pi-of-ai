/* chat.js — the conversation history behind multi-turn chat.
 *
 * The whole design is shaped by one number: the models here are loaded with a
 * 4096-token context, shared between the system prompt, the history, the new
 * request and the reply. The app has already shipped one overflow bug from a
 * long prompt plus three auto-fix retries, so this module's job is as much
 * about refusing to overflow as it is about remembering.
 */

export const CTX_TOKENS = 4096;
export const WARN_AT = 0.5;      // show a fullness indicator past here
export const REFUSE_AT = 0.7;    // stop accepting turns past here

// Turns, oldest first. The system prompt is NOT stored here — it is supplied
// per request by historyFor(), so a variant's rules cannot be trimmed away or
// go stale when the user edits them mid-chat.
let turns = [];

export function getHistory() {
  return turns.slice();
}

export function addTurn(role, content) {
  const text = String(content || '');
  if (!text.trim()) return getHistory();
  turns.push({ role, content: text });
  return getHistory();
}

export function clearHistory() {
  turns = [];
}

// Rough, and deliberately so: shipping a tokenizer to make this exact would
// cost more than the number is worth. ~4 characters per token is the usual
// English approximation; code runs denser, which is what the headroom between
// REFUSE_AT and 1.0 is for.
export function estimateTokens(text) {
  return Math.ceil(String(text || '').length / 4);
}

// What actually goes to the model: system prompt first (exempt from any
// trimming), then the history, then the new request.
export function historyFor(systemPrompt, request) {
  const msgs = [];
  if (systemPrompt && systemPrompt.trim()) {
    msgs.push({ role: 'system', content: systemPrompt });
  }
  for (const t of turns) msgs.push({ role: t.role, content: t.content });
  if (request && request.trim()) msgs.push({ role: 'user', content: request });
  return msgs;
}

// Fullness against the window, including the system prompt and the pending
// request, since those occupy the same budget.
export function usage(systemPrompt = '', pending = '') {
  const all = historyFor(systemPrompt, pending)
    .map(m => m.content).join('\n');
  const tokens = estimateTokens(all);
  const ratio = tokens / CTX_TOKENS;
  return {
    tokens,
    ratio,
    percent: Math.min(100, Math.round(ratio * 100)),
    warn: ratio >= WARN_AT,
    full: ratio >= REFUSE_AT,
    turns: turns.length,
  };
}

// The most recent fenced block in the conversation, newest first. Extracted
// mechanically for the handoff — asking a small model to reproduce code inside
// a summary invites it to mangle it.
//
// A reply truncated at the token limit mid-block still RENDERS as a code block
// on screen — index.html's segments()/bodyHtml() parse by ``` parity, not by
// requiring a closing fence. If this function only matched closed fences, a
// truncated answer would be invisible to it and the handoff would walk back to
// an older, stale block while labelling it "the latest version". So this must
// use the same parity split as the renderer, not a regex that requires a close.
export function lastCodeBlock() {
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role !== 'assistant') continue;
    const parts = turns[i].content.split('```');
    // Same parity as segments()/bodyHtml() in index.html: parts[1], parts[3], …
    // are code. An EVEN parts.length means the trailing ``` never closed, so
    // the last element (an odd index) is the open, truncated block. An ODD
    // parts.length means every fence closed, so the last COMPLETE code block
    // is the second-to-last element (the final element is trailing prose).
    const idx = parts.length % 2 === 0 ? parts.length - 1 : parts.length - 2;
    if (idx < 1) continue;   // no ``` in this turn at all
    const raw = parts[idx];
    const m = /^([\w+.-]*)\n([\s\S]*)$/.exec(raw);
    const lang = (m && m[1]) || 'python';
    const code = (m ? m[2] : raw).replace(/\n$/, '');
    return { lang, code };
  }
  return null;
}

// The text the user copies into a new chat. The model writes the prose; this
// appends the code verbatim.
export function handoffText(summary) {
  const parts = [];
  const s = String(summary || '').trim();
  parts.push(s || 'Continuing from an earlier chat.');
  const block = lastCodeBlock();
  if (block) {
    parts.push('Here is the latest version of the code:');
    parts.push('```' + block.lang + '\n' + block.code + '\n```');
  }
  return parts.join('\n\n');
}
