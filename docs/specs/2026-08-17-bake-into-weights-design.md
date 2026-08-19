# Bake Into Weights (Rung 2) — design

**Status:** implemented 2026-08-19 — `bake.js`, `bake-template.ipynb`,
`training-log.js`, and the bake/compare popups in `index.html`. The notebook's
own training run is unverified: it needs a Colab GPU, which no in-browser check
can supply.
**Date:** 2026-08-17
**Follows:** `docs/specs/2026-08-16-model-variants-design.md` (Rung 1, shipped)

## Problem

Rung 1 lets a student write house rules and apply them as a system prompt. The
rules work, but they occupy the context window on every request — which is the
cost `rules_baker` exists to remove.

Rung 2 puts the same rules into the weights. Nothing in the app can do that: the
browser cannot train, and `rules_baker` needs a GPU, Python and a runbook. The
student must end up with a real GGUF without meeting any of that.

## Binding constraints

**The lesson is one hour.** Idle waiting is the scarce resource, not compute. A
design that technically works but overruns produces nothing, because a dropped
Colab session leaves no time to retry.

**The browser needs a merged GGUF.** wllama exposes no `lora` or `adapter`
surface — checked against its published type definitions — so shipping a small
adapter instead of a merged model is not available for the browser. Adapters
remain viable for the optional Ollama runtime; that is a fast path for capable
machines, not for the Chromebook target.

**The repo stays private.** Colab can open notebooks from a private GitHub repo
only if each student authorises Colab and has repo access — unworkable for a
class. So the notebook is generated client-side and downloaded, and GitHub is
not involved.

**Nothing gets installed.** Colab runs in a browser tab and Google supplies the
GPU, so a locked-down Chromebook can do this. Credentials and cost sit with the
student's own Google account.

## Flow

### Outward

`Bake into weights` in the variant editor — currently a disabled signpost —
becomes live and opens a popup:

1. **Pick a target.** Three options, each with an honest time estimate:

   | Target | Size | Notes |
   |---|---|---|
   | SmolLM2 135M, F16 | ~270MB | **Default.** No quantise step |
   | SmolLM2 135M, Q4 | ~100MB | Smallest download; needs a `llama.cpp` build |
   | Qwen2.5-Coder 0.5B, Q4 | ~400MB | Best comparison; slowest, most likely to overrun |

   The default is the safe path, so it is the one a student falls into rather
   than the one they must know to choose. The slower two carry a warning.

   The F16 default exists because converting to GGUF is pure Python while
   quantising to Q4 needs a compiled `llama.cpp` — the slowest and most
   failure-prone step in the notebook. Dropping it trades file size for
   finishing inside the lesson.

2. **Generate the notebook.** A template file in `web/` is filled in with the
   variant's rules, the chosen base model and quant, then downloaded as
   `.ipynb`. The template stays an editable, testable file: fixing the training
   code never means shipping new JavaScript.

3. **Show the four steps.** Open Colab → upload the notebook → Runtime > Run all
   → return with two files.

### What the notebook does

Everything, in one run: generates a dataset from the rules using a teacher model
on Colab's GPU, trains the LoRA, merges it, converts to GGUF, and emits

- **`<variant-slug>-<YYYY-MM-DD>.gguf`** — the baked model
- **`<variant-slug>-<YYYY-MM-DD>.json`** — loss history and run metadata

The filenames must be **unique per bake**, not a fixed `model.gguf`. The model
store from sub-project 1 keys records on filename, deliberately, so that
re-saving the same file replaces rather than duplicates it. A constant filename
turns that into a trap: the second bake silently overwrites the first, and a
student loses a model they spent most of a lesson producing, with no warning.

Deriving the name from the variant and the date fixes it without touching the
store's key scheme — which would otherwise need changing, and changing it after
students have stored data means a migration. `strict-typing-2026-08-17.gguf`
also tells them which rules produced it, which `model.gguf` never could.

Baking the *same* variant twice in one day still lands on the same name. That is
usually what you want — the second run is a retry of the first — so it replaces
rather than accumulating near-identical models. But it must not be silent: when
a save would replace an existing record, the app says which model it is about to
replace and lets the student rename instead. Losing a bake is the failure this
whole section exists to prevent, and "you asked for it" is no comfort when a
lesson's work disappears.

It plots the loss inline as it trains, so the wait has something to watch.

The teacher/student split is the one `rules_baker` already implements: the
teacher sees the rules and writes compliant output; the stored training prompt
has the rules stripped, so the student model learns to obey without being told.

### Return

Both files drop into the app.

**The `.gguf` is persisted.** Today a dropped model is held only in memory and
is lost on reload — after a 40-minute bake that is brutal, and a sleeping
Chromebook would do it. It goes into browser storage and reappears under Custom
GGUFs on the next visit.

**The training log renders the loss curve** — the student's own run, not a stock
illustration.

### The payoff: Compare

Same prompt, **no system prompt**, base model and baked model side by side.

The base ignores the house rules because it never saw them. The baked one obeys
because they are in its weights. That contrast is the project's central claim in
one screen, and it is why Rung 1 insisted on showing the rules in the system
prompt box — so this moment has something to land against.

Two views prove different things and both are needed: the loss curve shows
training happened; the comparison shows *your rules* were learned.

## Explicitly not building

- **Automatic rule-checking.** Marking each rule obeyed/not needs a judge model
  or the Python sandbox, and would be wrong often enough to mislead a beginner
  who cannot yet tell.
- **Sharing baked models between students.**
- **Adapter-only output for the Ollama runtime.** Viable, but it splits the flow
  in two for the minority of users on capable machines.

## Risks

- **Colab drops the session.** Mitigated only by finishing fast — hence the
  small default. The notebook should checkpoint so a re-run resumes rather than
  restarts.
- **A bad bake is undiagnosable.** The comparison shows *that* it failed, not
  *why*: bad rules, weak teacher, and too-few steps look identical to a
  beginner. Not solvable within Rung 2 — see Future.
- **Browser storage limits.** A 270MB model is large for `localStorage`-class
  storage; persistence needs the Cache API or IndexedDB, and eviction under
  storage pressure must fail visibly rather than silently.
- **Teacher quality caps student quality.** A weak teacher produces a weak
  dataset and no amount of training fixes it.

## Future

**An API that reads the work and says what failed.** Given the rules, the
generated dataset and the before/after outputs, a model could tell the student
*why* a bake disappointed — rules too vague to learn, dataset too small,
training too short, or a rule the teacher itself never obeyed. That closes the
one gap this design cannot, and it is the natural Rung 3.

## Testing

No test framework; verification is in-browser against the running app.

1. The bake popup generates a valid `.ipynb` — parses as JSON, contains the
   variant's rules and the chosen base model.
2. Each of the three targets produces a notebook naming the right model and
   quant.
3. A dropped `.gguf` survives a reload and reappears under Custom GGUFs.
4. A dropped training log renders a loss curve matching its data.
5. Compare sends the identical prompt with no system message to both models —
   confirmed by inspecting both outgoing requests.
6. Storage failure on persist is reported, not silent.
7. A malformed training log is rejected without breaking the page.
8. Two bakes of different variants produce two distinct stored models, not one
   overwriting the other.

## Files touched

| File | Change |
|---|---|
| `rules_baker/web/bake.js` *(new)* | Notebook generation, target definitions |
| `rules_baker/web/bake-template.ipynb` *(new)* | The Colab notebook, editable as a real file |
| `rules_baker/web/model-store.js` *(new)* | Persisting a baked GGUF in browser storage |
| `rules_baker/web/index.html` | Bake popup, compare view, loss curve, drop handling |

`variants.js` is unchanged — the rules array it already stores is exactly what
the notebook consumes.
