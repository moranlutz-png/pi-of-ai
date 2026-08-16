# Model Variants (Rung 1) — design

**Status:** approved, not yet implemented
**Date:** 2026-08-16

## Problem

A beginner who wants the model to behave differently has no route in the app.
The only real mechanism the project offers is `rules_baker`, which needs a GPU,
Python, a config file and a runbook. "Construct a GGUF" is not a beginner task.

## Scope

This spec covers **Rung 1 only**: customising behaviour through saved house
rules, applied at inference time. It changes no weights.

**Rung 2** — baking those same rules into the weights via a generated Colab
notebook — is deliberately out of scope and specified separately. Rung 1 is
designed so its output feeds Rung 2 unchanged: the rules a student writes here
become the teacher instruction there.

### Constraint carried forward to Rung 2: the lesson is one hour

Students get a single one-hour lesson. The Rung 2 sketch — teacher-generated
dataset, LoRA train, merge, build `llama.cpp`, quantise, download ~0.4–1.7GB,
re-upload to the browser — does not fit inside that, and a dropped Colab
session means starting over with no time left. Idle waiting is the expensive
resource, not compute.

This is not a Rung 1 problem, but recording it here because it probably
invalidates the obvious Rung 2 design. Directions worth testing when Rung 2 is
specified:

- **Ship the adapter, not a merged model.** A rank-32 LoRA on a 0.5B base is
  tens of megabytes rather than hundreds. Removes most of the download and the
  merge step. Needs checking against what wllama can load — Ollama supports
  adapters directly, wllama may not.
- **Skip the quantise step.** Converting to F16 GGUF is pure Python; only
  quantising to Q4 needs a compiled `llama.cpp` binary. Dropping the compile
  removes the slowest and most fragile install, at the cost of a larger file.
- **Bake the smallest model, not the best one.** SmolLM2 135M trains in a
  fraction of the time and produces a file small enough to move over school
  wifi. A worse model that finishes inside the lesson teaches more than a better
  one that doesn't.
- **Start the job first, teach during the wait.** If the bake is kicked off in
  the first ten minutes, the waiting is lesson time rather than dead time.

### Explicitly not building

- Sharing variants by URL or short code
- Applying variants to Ollama models
- Any weight modification

Both of the first two are reasonable later; neither is needed to prove the
shape.

## Why rules, and why visible

The rules are the same artefact at both rungs. At Rung 1 they are the system
prompt. At Rung 2 they are what the teacher model sees while generating training
data, and what gets stripped from the student's prompt. One input, two rungs.

The editor **shows the rules in the System prompt box** rather than applying
them invisibly. This is the point of the feature, not an implementation detail:
the student watches their rules consume the context window on every request.
That is the cost `rules_baker` exists to remove, and if Rung 1 hides it, Rung 2
has nothing to contrast against.

## Data model

A variant is a named set of rules bound to a base model:

```js
{
  id: string,            // generated, stable
  name: string,          // user-supplied, shown in the sidebar
  rules: string[],       // one plain-English rule per entry
  baseModelUrl: string,  // must match a models.json entry's url
  temp: number,
  maxTokens: number,
  createdAt: number      // epoch ms
}
```

Persisted to `localStorage` under `pi-of-ai:variants` as an array.

`localStorage` is per-browser and cleared with site data, so **Export** writes
the array as a `.json` download and **Import** reads one back. That covers the
loss case and lets a teacher hand a starter variant to a class as a file.

## Behaviour

Selecting a variant:

1. Loads its base model if not already loaded (reusing the existing load path).
2. Writes `rules.join('\n')` into the System prompt textarea and expands the
   collapsed section so it is visible.
3. Applies `temp` and `maxTokens` to the existing controls.

No new generation path. Generation continues to read the System prompt box, so a
variant is exactly equivalent to the student having typed those rules
themselves — which is both simpler and more honest.

If `baseModelUrl` no longer matches any entry in `models.json`, the variant is
shown with a warning and selecting it prompts for a new base model rather than
failing silently.

## UI

**Sidebar** — a new `Your variants` section above `Custom GGUFs`.

Variants are not `.gguf` files and must not sit inside Custom GGUFs. Each row
carries a marker distinguishing it from a model, because a variant that looks
like a baked model teaches the wrong thing.

**Editor** — a popup reusing the existing small-modal pattern
(`.modal-panel.small`, as the Ollama setup popup does):

- Name
- Rules, one per line
- Base model picker, populated from `models.json`
- Temperature and max tokens
- Save / Delete / Export / Import
- A disabled **Bake into weights** button with a "coming soon" note

The Bake button is present and disabled deliberately: it tells the student the
next rung exists and what it is for, rather than leaving Rung 1 looking like the
whole story.

**Starter rules** — the editor opens pre-filled from
`rules_baker/data_gen/rules/example_rules.md`, which already exists and is
already in the format the Rung 2 pipeline parses. This solves the blank-page
problem without inventing content, and keeps one source for the examples.

**Honesty line** — the editor states plainly: *rules are sent with every
request; to bake them into the weights instead, see Bake.*

## Files touched

| File | Change |
|---|---|
| `rules_baker/web/index.html` | Sidebar section, editor popup, variant store, selection wiring |
| `rules_baker/web/variants.js` *(new)* | Variant CRUD, storage, import/export — kept out of `index.html`, which is already 2147 lines |

No change to `models.json`, `pyworker.js`, or anything under `scratch_coder/`.

## Testing

No test framework exists in this repo; the established approach in this codebase
is in-browser verification against the running app. For each of these, verify in
the served page:

1. Creating a variant persists it and it survives a reload.
2. Selecting it loads the base model, fills the System prompt box with the
   rules, and applies temp/tokens.
3. A generation made under a variant sends the rules as a `system` message —
   confirmed by inspecting the outgoing request.
4. Export produces valid JSON; importing it recreates the variant exactly.
5. Deleting removes it from both the sidebar and storage.
6. A variant whose `baseModelUrl` is absent from `models.json` warns rather than
   breaking.
7. Variants are visually distinct from models in the sidebar.

## Risks

- **`localStorage` limits.** Variants are small text, so the practical ceiling
  is thousands of them. Export exists for durability regardless.
- **Rules crowding the context window.** The window is 4096 tokens; a long rule
  set measurably reduces room for the request and reply. The editor should show
  a rough rule-length indication rather than let a student silently fill the
  window.
- **Mistaking Rung 1 for Rung 2.** Mitigated by naming, the marker, the honesty
  line, and the visible System prompt — but this is the risk most worth
  guarding, because the project's central claim is the difference between the
  two.
