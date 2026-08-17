# Multi-turn chat — design

**Status:** approved, not yet implemented
**Date:** 2026-08-17

## Problem

Every generation starts from nothing. Asking "now add error handling" produces a
fresh answer to that sentence alone, because the model never sees what came
before. The obvious follow-up to any generated code is impossible.

## The constraint that shapes everything

`n_ctx` is **4096 tokens** — roughly 3000 words, shared between the system
prompt, the whole history, the new request and the reply. A code-carrying turn
can be several hundred tokens on its own.

This is not hypothetical: the app already shipped a bug where a long prompt plus
three auto-fix retries overran the window and died with "Running out of context
cache". History management is therefore the design, not a detail of it.

## What a chat is

`#output` stops being a single slot overwritten per generation and becomes a
**transcript**: alternating user and model turns, newest at the bottom, each
model turn keeping its own sandbox results and its disclaimer.

A `messages` array persists between generations instead of being rebuilt. Two
things are deliberately excluded:

- **Auto-fix retries.** These are an internal exchange between the app and the
  model — "your code has this problem, fix it". Only the final accepted answer
  enters the history. Including them would put up to three rejected attempts and
  three correction prompts into every subsequent turn's context.
- **The intro line.** It is UI narration, not part of the conversation.

The system prompt sits at position zero and is **exempt from trimming**. A
variant's rules silently ceasing to apply mid-conversation would be a
particularly confusing failure, since the rules are still visible in the box.

## The limit

A running estimate of history size is kept against the window. Estimation is
character-based (~4 characters per token) — approximate, but the alternative is
shipping a tokenizer, and the number only needs to be right enough to warn.

- Past **~50%**: a quiet indicator shows how full the chat is.
- Past **~70%**: the app refuses new turns and says the chat is full.

**Refusing rather than dropping the oldest turn is deliberate.** Silent trimming
means the model forgets something the user can still see on screen, and a
beginner has no way to work out why. An explicit stop is understandable; silent
amnesia is not.

## New chat, and the handoff

**New chat** clears the transcript. Before it does, it offers a handoff:

1. The model is asked for a short summary of what the chat covered and what was
   decided.
2. The app appends the most recent code block **verbatim**, extracted
   mechanically rather than requested from the model.
3. Both are shown in a box with a Copy button, labelled so it is clear the user
   pastes it into the new chat themselves.

Two decisions worth keeping:

**The user pastes it; the app does not inject it.** A summary the user reads
before using cannot silently poison the next chat. Given the models here — 135M
to 2B — a wrong summary is likely enough that hiding it would be a real hazard,
and a beginner would not catch it.

**The code is extracted, not summarised.** Asking a small model to reproduce a
code block inside prose invites it to mangle it. The prose is the model's job;
the code is the app's.

## Explicitly not building

- Persisting chats across reloads
- Multiple saved chats, or switching between them
- Automatic summarisation without the user seeing it

All are reasonable later. None is needed for follow-up questions to work.

## Behaviour

1. Sending a message appends `{role: 'user'}` to history and renders a user turn.
2. Generation runs as now — including auto-fix, which stays internal.
3. The accepted answer appends `{role: 'assistant'}` and renders a model turn
   with its sandbox notes.
4. The fullness indicator updates.
5. At the limit, the composer refuses and points at New chat.

## Risks

- **Small models handle multi-turn badly.** A 135M model may ignore history or
  repeat itself. That is a property of the model, not a bug — but the UI should
  not imply the model is more capable than it is.
- **The token estimate is approximate.** Character-based counting can
  under-estimate for dense code. The 70% ceiling leaves room for that; if
  overflows still occur, lower the ceiling rather than adding a tokenizer.
- **A summary costs a generation.** On WASM that is a real wait. The handoff
  should show the same tips shown during any other wait.

## Testing

No test framework; verification is in-browser against the running app.

1. A second message includes the first exchange — confirmed by inspecting the
   outgoing request.
2. Auto-fix retries do not appear in the history of the next turn.
3. The system prompt stays at position zero after several turns.
4. The transcript renders every turn, with sandbox notes on model turns only.
5. The fullness indicator appears past the halfway mark.
6. At the ceiling, sending is refused with an explanation rather than failing.
7. New chat produces a summary containing the last code block verbatim.
8. New chat clears both the transcript and the history array.
9. A chat under a variant still sends the variant's rules on turn five.

## Files touched

| File | Change |
|---|---|
| `rules_baker/web/chat.js` *(new)* | History array, token estimate, trimming policy, handoff text |
| `rules_baker/web/index.html` | Transcript rendering, New chat button, fullness indicator, handoff box |

`variants.js` and `model-store.js` are unchanged.
