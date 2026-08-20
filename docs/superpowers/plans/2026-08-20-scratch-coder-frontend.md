# Scratch-Coder Front End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `scratch_coder` the thing it has never had — a way to *look* at the model. Not a prettier CLI: a page where a student can see every weight tensor in the stack, watch attention completely rather than sampled, and put a trained model beside a freshly-random one and see which parts actually differ.

**Architecture:** A new static page at `scratch_coder/web/`, mirroring `rules_baker/web/` exactly — no build step, no bundler, one `index.html` plus small ES modules, served by a copy of `serve.py`. A new Python exporter turns a checkpoint into JSON the page reads. The exporter is the join that `arch_map.py` was already written to expect.

**Tech Stack:** Vanilla ES modules, inline SVG and `<canvas>`. Python + torch on the export side only. No dependencies on either side.

**Spec:** `docs/specs/2026-08-18-observability-and-honesty-design.md`, item 11.

---

## What exists today, and what this plan could not see

**Already in the repo, and load-bearing:**

- `scratch_coder/arch_map.py` — the tensor map, computed from the architecture with no torch and no checkpoint. It already emits `groups[].tensors[]` with `name`, `shape`, `params`, `role`, plus `buffers` for the causal mask. Its own docstring names this UI: *"The Knob Matrix in the web UI needs to draw one cell per weight tensor... Weight values are a separate question, answered later by an exporter that does have torch and does have a checkpoint. The tensor names emitted here match PyTorch's `state_dict()` keys exactly, so the two join cleanly."* **This plan builds that exporter.** Do not re-derive the layout in JS; fetch it from `arch_map.py`.
- `scratch_coder/guards.py` — `layer_grad_norms(model)` and `format_layer_norms(norms)` already exist and are already called each eval by both trainers. Rung 4 renders numbers that are being computed today and thrown away.
- `scratch_coder/train_forever.py` — already appends `data_big/loss.jsonl`, one JSON object per eval (`iter`, `val_loss`, `best`, `elapsed_s`, `params`). Rung 4 has its data source already written.

**Verified on macOS, 2026-08-20**, so the plan rests on observed behaviour rather than assumption:

| Fact | Observed |
|---|---|
| `prepare_data.py` + `train.py` on CPU | 500 iters in **30s**, val loss **4.770 → 1.575** |
| `train.py` model shape | 4 layers, 4 heads, `n_embd` 128, block 128, vocab **101** → **835,584** params, 53 tensors |
| `data/ckpt.pt` size | **3.6 MB** (fp32 `state_dict` + `cfg`) |
| `arch_map.py` defaults | 5 layers, 6 heads, `n_embd` 192, block 160 → **2.29M** params |

**The two shapes differ, and that is a trap.** `arch_map.py`'s `DEFAULTS` mirror `train_forever.py`, not `train.py`. A UI that draws `arch_map.py --json` with its defaults and then loads a `train.py` checkpoint renders a five-block stack with four blocks of data in it. **The shape must come from the checkpoint's own `cfg`**, and `arch_map.py` must be called with those values. Step 0.1 below is that call.

**What this plan could not see.** The user reports part of this item is already built on a Windows machine. Nothing of it is reachable from this clone — `git log --all`, `git ls-remote origin`, and `git stash list` show no `scratch_coder` UI file in any branch, tag, or stash; the only trace is `arch_map.py`'s forward-reference to a "Knob Matrix" that does not exist here. So this plan is written to **merge with** that work, not to replace it:

- The rung boundaries are the merge points. If the Windows machine already has the exporter, skip Rung 0 and keep its JSON schema — adopt whatever it emits and adjust the readers, rather than renaming its fields to match this document.
- Before starting, push the Windows branch and diff it against this plan. Where they disagree about a *name*, the existing code wins. Where they disagree about *behaviour*, the spec wins (`CLAUDE.md`: specs and plans are the authority; the spec wins over the plan).
- The one thing worth defending in a merge is the honesty section at the bottom. It is the part most likely to have been skipped, and the part the spec is most specific about.

---

## Global Constraints

- **No build step.** `scratch_coder/web/index.html` loads one `<script type="module">`; new modules sit beside it and are imported from it. Same rule as `rules_baker/web/`.
- **No test framework** (`CLAUDE.md`). Verification is in a browser, from the console, against the checkout under test. Serve it on **port 8125** — `8123` is the rules_baker page and reusing it means one page shadows the other and you verify the wrong checkout.
- **Two builds stay independent.** Nothing in `scratch_coder/web/` may import from `rules_baker/web/`, and vice versa. The README describes them as independent builds sharing a philosophy; a shared module would make that false. Copying ~40 lines of `serve.py` is the correct cost.
- **Escape everything interpolated into `innerHTML`.** Copy `esc()` from `rules_baker/web/index.html` (it escapes quotes as well as angle brackets). Tensor names come from a `.pt` file, which is user-supplied input.
- **The page must work with no checkpoint.** `arch_map.py` needs neither torch nor a trained model, so the structural view — every tensor, its shape, its parameter share — renders from architecture alone. A student who has not trained yet still sees the model. Values light up when a checkpoint is exported.
- **Never draw a number the file does not contain.** `training-log.js` in the other build is the precedent: `Number(null)` is `0`, and a `0.0` loss reads as "learned perfectly". Reuse its `finiteNum()` discipline for every value read from JSON.
- **Visual changes need a screenshot or a computed-style read**, not a `textContent` check (`CLAUDE.md`). An unconstrained inline SVG renders at ~300×150 and blows out its row, which no DOM-text assertion catches.

---

### Task 0: The exporter — `scratch_coder/export_inspect.py`

The join `arch_map.py` was written to expect. Reads a checkpoint, writes one JSON file the page can fetch. Runs where torch is; the page never sees a `.pt`.

**Files:**
- Create: `scratch_coder/export_inspect.py`
- Modify: `.gitignore` — add `scratch_coder/web/inspect.json`

**Interfaces:**
- Consumes: `data/ckpt.pt` or `data_big/ckpt.pt`, `arch_map.tensor_map()`
- Produces: `scratch_coder/web/inspect.json`, the only contract the page depends on

- [ ] **Step 0.1: Take the shape from the checkpoint, not from `arch_map`'s defaults**

```python
ckpt = torch.load(path, map_location="cpu")
cfg = ckpt["cfg"]          # both trainers save cfg.__dict__ beside the state_dict
# arch_map's DEFAULTS mirror train_forever.py. A train.py checkpoint is a
# different shape, and drawing one against the other's layout silently renders
# a stack with a block missing.
amap = tensor_map(cfg["vocab_size"], cfg["block_size"],
                  cfg["n_layer"], cfg["n_head"], cfg["n_embd"])
```

- [ ] **Step 0.2: Emit per-tensor statistics, not per-tensor weights**

Per tensor: `mean`, `std`, `absMax`, `l2`, `fracNearZero`. The full weights are Rung 3's problem; the Knob Matrix needs a single number per cell and nothing larger should be in this file.

Compute the same statistics for a **freshly initialised** model at the same config and the same seed, and emit both. The trained-vs-random comparison is the starkest honest demo in the spec, and it is only free if both halves are exported together — asking a student to train a second model to get the baseline is not free.

```python
random_model = GPT(GPTConfig(**cfg))     # untrained, same shape
```

- [ ] **Step 0.3: Emit the embedding matrix in full, projected to 2D**

`tok_emb.weight` is `[vocab_size, n_embd]` — 101 × 128 at `train.py`'s shape. Small enough to ship whole, and it is the one place the spec promises real visible structure. Project to 2D with PCA computed in numpy (no sklearn: a power-iteration on the covariance of a 101×128 matrix is about fifteen lines and one fewer dependency). Emit both the 2D points labelled with their characters, and the top-k nearest neighbours per character by cosine distance.

- [ ] **Step 0.4: Write the file, and say what it does not contain**

```python
{
  "kind": "pi-of-ai:scratch-inspect",   # checked explicitly by the reader
  "version": 1,
  "config": {...}, "totalParams": N,
  "checkpoint": {"path": ..., "iter": ..., "valLoss": ..., "sizeBytes": ...},
  "arch": amap,                          # arch_map.py's output, verbatim
  "trained": {"<tensor name>": {...stats}},
  "random":  {"<tensor name>": {...stats}},
  "embedding": {"points": [...], "neighbours": {...}, "varianceExplainedPct": ...},
  "unverifiable": [
    "whether a 2D projection preserves the neighbourhoods it appears to show",
    "what a weight statistic means for behaviour — these are shapes, not explanations"
  ]
}
```

- [ ] **Step 0.5: Verify**

```bash
cd scratch_coder && python prepare_data.py && python train.py   # ~30s on CPU
python export_inspect.py --ckpt data/ckpt.pt
python -c "
import json; d = json.load(open('web/inspect.json'))
print(d['kind'], d['config']['n_layer'], 'blocks,', f\"{d['totalParams']:,}\", 'params')
print('tensors:', sum(len(g['tensors']) for g in d['arch']['groups']))
print('trained/random keys match:', set(d['trained']) == set(d['random']))
print('embedding points:', len(d['embedding']['points']))
"
```

Expected against a `train.py` checkpoint: `pi-of-ai:scratch-inspect 4 blocks, 835,584 params`, **53** tensors (2 embedding + 4×12 block + 3 output), keys match `True`, **101** embedding points. If it prints 5 blocks, Step 0.1 was skipped.

- [ ] **Step 0.6: Commit**

```bash
git add scratch_coder/export_inspect.py .gitignore
git commit -m "Export a checkpoint as something a page can read"
```

---

### Task 1: The page, and the Knob Matrix

**Files:**
- Create: `scratch_coder/web/index.html`, `scratch_coder/web/serve.py`, `scratch_coder/web/inspect.js`, `scratch_coder/web/knobs.js`

**Interfaces:**
- Consumes: `inspect.json`
- Produces: `parseInspect(text)` → `{ok, data}` / `{ok, error}`, the shape every later rung reads through

- [ ] **Step 1.1: `serve.py`, on port 8125**

Copy `rules_baker/web/serve.py`, change the default port to **8125**, and drop the COOP/COEP headers — those exist for SharedArrayBuffer and WASM threads, which this page does not use. Say so in a comment, or the next person will copy them back.

- [ ] **Step 1.2: `inspect.js` — validate hard, never throw**

```js
export const INSPECT_KIND = 'pi-of-ai:scratch-inspect';
export function parseInspect(text) { /* → {ok:true, data} | {ok:false, error} */ }
```

Check `kind` explicitly. Reject a missing or empty `arch.groups`. Run every statistic through a `finiteNum()` clone. Return a reason, never an exception — the caller puts it on screen. This is the same argument `training-log.js` makes at the top of its file; read it before writing this one.

- [ ] **Step 1.3: `knobs.js` — one cell per tensor**

Render `arch.groups` in forward-pass order: embedding, then each block, then output. One cell per tensor, sized by `params` (a square-root scale, or `head.weight` at 12,928 params drowns every LayerNorm's 128 in the same picture (a 101x ratio)). Colour by `kind`. Hover shows `role`, which `arch_map.py` already wrote for exactly this.

Show the causal-mask **buffer** in the layout, marked as not learned. `arch_map.py` went out of its way to emit it separately with the comment *"it is not trained and must not be counted"* — a matrix that appears in the stack but not in the parameter count is a genuinely interesting thing for a student to notice.

- [ ] **Step 1.4: Trained versus random, side by side**

A toggle over the same layout: **trained**, **random**, **difference**. The difference view is the payload — it answers "what did training actually change?" and the honest answer at 4 layers is "much less than you would guess, and not evenly".

- [ ] **Step 1.5: Verify**

```bash
cd scratch_coder/web && python3 serve.py 8125
```

```js
// http://localhost:8125 console
const cells = document.querySelectorAll('#knobs .knob');
console.log('cells:', cells.length);                    // expect 53 + 1 buffer = 54
const el = cells[0];
const r = el.getBoundingClientRect();
console.log('first cell box:', r.width, r.height);      // expect > 0, not 300x150
console.log('has role tooltip:', !!el.title);           // expect true
document.getElementById('viewDiff').click();
await new Promise(r => setTimeout(r, 100));
console.log('diff view active:', document.getElementById('knobs').dataset.view); // "diff"
```

Then **take a screenshot**. A cell count is not a picture, and this task's whole output is a picture.

- [ ] **Step 1.6: Verify the no-checkpoint path**

```bash
mv web/inspect.json /tmp/ && open http://localhost:8125
```

Expected: the matrix still draws from architecture alone, with a visible line saying no checkpoint has been exported and how to make one. Not an empty page, and not a console error. Move it back afterwards.

- [ ] **Step 1.7: Commit**

```bash
git add scratch_coder/web/
git commit -m "Draw every tensor in the stack, trained beside random"
```

---

### Task 2: Embedding neighbourhoods

The spec is explicit that this is one of only two things that shows real structure. Build it before anything flashier.

**Files:** Create `scratch_coder/web/embedding.js`; modify `index.html`

- [ ] **Step 2.1: Scatter the 2D projection, labelled with the characters themselves**

Not dots — the glyphs. The lesson is *which characters cluster*, and a legend mapping 101 dots to 101 characters is a worse version of writing them where the dots are.

- [ ] **Step 2.2: Print the variance explained, next to the plot**

Two components of a 128-dimensional space usually capture a small fraction of it. Saying so is the difference between a plot and a claim. If it is 18%, the label reads "these two directions carry 18% of the variation" — which is also the honest answer to a student asking why two obviously-similar characters sit far apart.

- [ ] **Step 2.3: Nearest neighbours as a list, beside the plot**

Cosine neighbours in the **full** space, not the projection. This is the check on the picture: when the plot and the list disagree, the list is right, and that disagreement is worth showing rather than hiding.

- [ ] **Step 2.4: Verify**

```js
const pts = document.querySelectorAll('#embed text');
console.log('glyphs:', pts.length);                     // expect 101
console.log('variance label:', document.getElementById('embedVar').textContent);
// pick a digit and read its neighbours — expect other digits to rank high
console.log(document.querySelector('[data-char="7"]').dataset.neighbours);
```

Expected: digits neighbour digits, lowercase letters neighbour lowercase letters, whitespace neighbours whitespace. **If they do not, report that.** A model trained for 500 iterations may genuinely not have separated them yet, and "not yet" is the correct finding, not a bug to tune away. Re-run after `train_forever.py` and compare.

- [ ] **Step 2.5: Commit**

---

### Task 3: The live forward pass — attention, complete

The spec's claim is that at this size you can render attention *completely* rather than sampled. That needs the attention matrices for whatever the student typed, which means the forward pass has to happen somewhere the student's typing is. The browser has no torch, so the forward pass gets written in JS.

**This is the rung that earns the "small enough to see all the way through" framing.** 0.84M parameters is 3.4MB as fp32 — a page can hold the whole model. `model.py` is 125 lines; a JS port is not a research project.

**Files:** Create `scratch_coder/web/gpt.js`, `scratch_coder/web/attention.js`; modify `export_inspect.py` (a `--weights` flag) and `index.html`

- [ ] **Step 3.1: Export the weights as a binary sidecar, not as JSON**

`web/weights.bin`, fp32 little-endian, tensors concatenated in `arch_map.py`'s order, with the offsets recorded in `inspect.json`. JSON-encoding 840,000 floats produces a ~10MB text file that must be parsed as text; the binary is 3.4MB and arrives as an `ArrayBuffer`. Record a checksum and the config, and have the loader refuse a `weights.bin` whose config does not match the `inspect.json` beside it.

- [ ] **Step 3.2: `gpt.js` — the forward pass, ported straight from `model.py`**

Keep the function and variable names from `model.py`. This file's second job is to be read next to the Python by a student who wants to check the translation, and that only works if the two look alike.

Ship the causal mask, the `1/sqrt(head_dim)` scale and the softmax as their own small functions. They are the parts worth pointing at.

- [ ] **Step 3.3: Return attention, not just logits**

```js
forward(idx, {collectAttention: true}) // → {logits, attn: [layer][head][T][T], acts: [layer][T][C]}
```

At 4 layers × 4 heads × 128 × 128 that is 262,144 floats — a few megabytes held for one keystroke, and the reason this is possible at all is that the model is tiny. Say that in a comment; it is the spec's argument in one line.

- [ ] **Step 3.4: `attention.js` — every head, no sampling**

A grid: one heatmap per (layer, head), all of them, on one screen. Rendered to `<canvas>`, not SVG — 16 heatmaps of 16,384 cells each is 262,144 SVG rects and a locked tab.

- [ ] **Step 3.5: Verify against the Python, not against intuition**

The only verification that means anything here is agreement with `model.py`. Export the attention for one fixed prompt from Python, and compare.

```bash
python export_inspect.py --ckpt data/ckpt.pt --attn-probe "def hello(" --out /tmp/probe.json
```

```js
// console, same prompt typed into the page
const probe = await (await fetch('/probe.json')).json();
const mine = window.__lastForward.attn;
let worst = 0;
probe.attn.forEach((L,l) => L.forEach((H,h) => H.forEach((row,i) => row.forEach((v,j) => {
  worst = Math.max(worst, Math.abs(v - mine[l][h][i][j]));
}))));
console.log('largest disagreement with PyTorch:', worst);   // expect < 1e-4
```

Expected: under `1e-4`. Report the actual number. Anything above `1e-3` means the port is wrong — most often the mask, the scale, or a transposed `c_attn` split — and a pretty heatmap of the wrong matrix is worse than no heatmap.

- [ ] **Step 3.6: Verify the page still works without `weights.bin`**

Rungs 1 and 2 must not regress into requiring a 3.4MB download. Move `weights.bin` away, reload, confirm the Knob Matrix and the embedding plot still render and the attention panel says why it is empty.

- [ ] **Step 3.7: Commit**

---

### Task 4: Gradient norms and the loss curve

The cheapest rung in the plan: both trainers already compute this and throw it away.

**Files:** Modify `scratch_coder/train.py`, `scratch_coder/train_forever.py`; create `scratch_coder/web/curves.js`

- [ ] **Step 4.1: Write the per-layer norms to the log that already exists**

`train_forever.py` already appends `data_big/loss.jsonl` each eval. Add the `layer_grad_norms(model)` list it is already calling for the console line. `train.py` prints them and logs nothing — give it the same JSONL.

- [ ] **Step 4.2: Render both curves, and be careful what you claim**

The loss curve, and per-layer gradient norms over time. **Observed here at 500 iterations, 4 layers:** `L0:2.0e-01 L1:1.4e-01 L2:1.3e-01 L3:1.6e-01`. That is a mild, non-monotonic spread — it is *not* the textbook picture of vanishing gradients, and a UI captioned "watch gradients vanish in the early layers" would be pointing at numbers that say something else. Label the axes, plot the lines, and let a student read four numbers that differ by less than an order of magnitude. Four layers is not deep enough to vanish, which is itself worth saying.

- [ ] **Step 4.3: Verify**

```bash
cd scratch_coder && python train.py && wc -l data/loss.jsonl
```

```js
const pts = document.querySelectorAll('#lossCurve polyline')[0].getAttribute('points').split(' ');
console.log('loss points:', pts.length);
console.log('grad lines:', document.querySelectorAll('#gradCurve polyline').length); // expect n_layer
```

- [ ] **Step 4.4: Commit**

---

### Task 5: `layers/` — the template for a custom layer

**Files:** Create `scratch_coder/layers/README.md`, `scratch_coder/layers/example_layer.py`; modify `model.py`

- [ ] **Step 5.1: One extension point, and only one**

A registry `model.py` consults for the block's MLP. Not a plugin system — a dict, a name in the config, and a template file with a working example that does something visibly different. The point is that a student can change the architecture and see the Knob Matrix change shape, which requires exactly one seam, not a framework.

- [ ] **Step 5.2: Make `arch_map.py` honest about a custom layer**

`arch_map.py` hardcodes `model.py`'s tensor list. A custom layer breaks that silently — the map describes a model that is no longer being built. Have it detect a non-default layer and say it cannot describe it, rather than describing the wrong thing.

- [ ] **Step 5.3: Verify** — train 50 iterations with the example layer, export, confirm the Knob Matrix redraws with the new tensors and that `arch_map.py` flags what it cannot map.

- [ ] **Step 5.4: Commit**

---

## The honesty section — do not cut this

The spec gives one explicit warning for this item, and it is the kind that gets dropped in a merge:

> **trained transformer weights rendered as a raw grid look like noise.** Not structured colour — noise. Do not promise a student swirls that never appear.

So:

- **There is no raw-weight-grid view in this plan.** Not because it is hard — it is the easiest thing here — but because it teaches the opposite of the truth. If it gets built anyway, it must be captioned with what it is: a picture of noise, which is what a trained weight matrix genuinely looks like, and the interesting question is why that is.
- **What does show structure**, and is therefore what got the rungs: per-layer and per-tensor norms (Rung 1), embedding neighbourhoods (Rung 2), attention (Rung 3), trained-versus-random (Rung 1, the toggle).
- **The starkest honest demo is trained versus random, side by side.** It is Rung 1 for that reason, not because it is the easiest to build.
- **Every projection is a lie of some size.** The 2D embedding plot must print its variance-explained, and the neighbour list must be computed in the full space, so the student has the means to catch the picture being wrong.
- **The model stays useless at coding, and the page must not imply otherwise.** The sample this checkout produced after 500 iterations, in full:

  ```
  def the results ondulse or a ction (proms`)*******k)
         #                 -     # Wlass arks con is None, cange, the contar tup.
  ```

  That is the product working. A page that shows attention maps and gradient curves around output like that is at risk of looking like a debugging tool for a model that is nearly there. It is not nearly there and it never will be — `README.md` already says so, and the page should too, in the same words.

## Not building, and why

- **A raw weight-grid heatmap.** See above.
- **Training in the browser.** No torch, no autograd, and Pyodide does not carry PyTorch. Rung 3 runs a forward pass, which is a different and much smaller claim.
- **Sharing modules with `rules_baker/web/`.** Two independent builds; a shared import makes that untrue for the sake of forty lines.
- **A checkpoint uploader in the page.** `.pt` is a pickle, and unpickling arbitrary uploads is remote code execution. The exporter runs where the checkpoint already is, and the page only ever reads JSON and a float array.

## Open question for the merge

`arch_map.py` says "Knob Matrix", so a UI with that name may already exist on the Windows machine. If it does, its JSON schema replaces Task 0's — keep the existing field names and adapt the readers. The rung order in this plan is an argument about what to build first, not a claim on names.
