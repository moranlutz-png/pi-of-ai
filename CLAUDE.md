# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Pi-of-AI is a teaching kit: the smallest, most transparent thing that still shows how AI actually
works, running locally — including on a locked-down 4GB school Chromebook, in the browser. The
target device is a real constraint, not an aspiration. When a design choice trades size or
simplicity for capability, the small side usually wins.

Two independent builds share the repo and the philosophy:

- **`rules_baker/`** — bake a project's house rules into a small model's *weights* via QLoRA, so
  the rules stop eating the context window on every request. The teacher model *sees* the rules and
  writes compliant output; the stored student prompt has the rules **stripped**, so the student
  learns to obey silently. This asymmetry is the whole idea — preserve it in any change to the data
  pipeline.
- **`scratch_coder/`** — a nanoGPT-style character-level model written from scratch and trained
  from random init. It will never be a useful coder; that is the point. It exists so a student can
  watch structure emerge from noise.

Heavy work (training/baking) happens once on a GPU or free Colab. Inference runs anywhere on CPU.
"Runs on any computer" is a claim about inference only.

## Commands

```bash
# Browser runtime page (the main UI). Serves rules_baker/web/ with COOP/COEP,
# which is what enables SharedArrayBuffer -> WASM threads -> faster wllama.
cd rules_baker/web && python3 serve.py 8123     # default port is 8123

# rules_baker — generate training data against a LOCAL teacher endpoint
cd rules_baker
python data_gen/generate_dataset.py --config configs/qwen_coder_0_5b_chromebook.yaml
python eval/eval_rules.py                        # score rule compliance
python export/export_gguf.py                     # merge + convert for the browser

# scratch_coder — trains on the machine's own Python stdlib
cd scratch_coder
python train_forever.py                          # resumable; Ctrl-C and rerun continues
python sample_big.py "def "

pip install -r rules_baker/requirements.txt      # data-gen/eval only; training deps are commented
                                                  # out and installed on the GPU box (see README)
```

Training needs CUDA (Linux or WSL2) — see `rules_baker/train/BAKE_RUNBOOK.md` and
`rules_baker/train/WSL2_CUDA_SETUP.md`.

## Testing

**There is no test framework, and adding one is not implied by a task.** Verification is
in-browser against the running page, using the browser console. Plans and specs in `docs/` spell
out the exact snippets to run and the expected observed values.

Consequences worth internalising:

- A change is not verified until it has been *run in a browser* and the observed output reported.
  Reading the code and reasoning that it should work is not verification.
- Verify against the checkout you actually changed. If a server is already on the port, confirm
  what directory it is rooted in before trusting it — serving the wrong checkout produces
  convincing false passes.
- Visual changes need a screenshot or a computed-style read, not just a `textContent` check. A
  missing CSS rule renders an unconstrained inline SVG at ~300×150 and blows out a row, which no
  DOM-text assertion will catch.
- Port matters for browser storage: `localhost:8123` and `localhost:8126` are separate origins with
  separate IndexedDB, so parallel sessions on different ports cannot corrupt each other's data.

## Architecture

### The browser runtime (`rules_baker/web/`)

No build step, no bundler, no dependencies. `index.html` is a single ~2900-line file holding the
markup, all CSS, and one `<script type="module">`; small ES modules sit beside it and are imported
from that script. Adding a module means adding an import line, not a build config.

- `index.html` — UI, sidebar model tree, command palette (Cmd/Ctrl+K), settings modal, runtime
  switching, generation loop.
- `variants.js` — named sets of house rules bound to a base model (localStorage). **Rung 1** of
  customisation: rules are sent as the system prompt, weights untouched. The rules array shape is
  deliberately simple and portable because **Rung 2** (baking) reuses it verbatim as the teacher
  instruction.
- `model-store.js` — IndexedDB store keeping a locally-loaded `.gguf` across reloads. Records are
  keyed on lowercased filename so re-saving the same file replaces rather than duplicates it.
- `pyworker.js` — Pyodide in a Web Worker; offline after first load. Two ops: `run` (execute,
  capture stdout) and `check` (scope-aware syntax + undefined-name analysis, then execute).
- `serve.py` — static server that sets COOP/COEP. `coi-serviceworker.js` is the fallback for
  static hosts that cannot set headers (GitHub Pages); the page still works single-threaded.

**Two runtimes, one generation path.** `activeRuntime` is either `wasm` (wllama, in-tab) or
`ollama` (a detected local Ollama). Both funnel through one completion call. A recurring class of
bug is the *mismatch*: the model loaded in one runtime is not the one that will actually generate.
When touching model selection or loading, check both runtimes.

**Error handling convention.** Storage and runtime helpers never throw — they return result objects
(`{ok, id}` / `{ok, error}`) or `null`, and callers surface failures in the `#status` line. Storage
failures must be **visible**: a student who loses a 40-minute bake to silent eviction has no way to
know why. Keep new failure paths loud.

**Shared helpers already exist** — `$(id)`, `esc(str)` (escapes quotes as well as angle brackets),
`withLoad(label, fn)` (disables controls, times the load, writes the status line). Use them rather
than reimplementing. Note that `withLoad` writes its own success line to `#status` when its
callback resolves, so anything set from *inside* the callback gets overwritten — report from
after it returns.

**Browser storage keys in use** (do not collide): `pi-of-ai:variants`, `pi-of-ai:pinned`,
`pi-of-ai:recent`, `pi-of-ai:measured-tokps`, `pi-of-ai:load-mbps`, and IndexedDB database
`pi-of-ai`, object store `models`.

### The baking pipeline (`rules_baker/`)

`data_gen/generate_dataset.py` (teacher writes compliant output, rules stripped from the stored
prompt) → `train/train_lora.py` (QLoRA) → `export/export_gguf.py` (merge + convert) →
the browser loads the GGUF. Config lives in `configs/*.yaml`; `qwen_coder_0_5b_chromebook.yaml` is
the small/default path, `qwen_coder_7b.yaml` the capable one.

The browser needs a **merged** GGUF: wllama exposes no LoRA/adapter surface, so shipping a small
adapter is not an option for the browser path (it remains viable for Ollama).

## Working in this repo

**Specs and plans are the authority.** `docs/specs/` holds approved designs; `docs/superpowers/plans/`
holds the implementation plans that argue from them. When a plan and a spec conflict, the spec wins.
When a plan's own code turns out to be wrong, the plan's stated *intent* and its verification
snippets are the tiebreaker — plans have shipped with real defects in their code blocks.

Features are built in **rungs**: each rung is independently useful and ships on its own. Rung 1 of
customisation is rules-as-system-prompt; Rung 2 is baking those same rules into weights. Later rungs
depend on earlier ones staying intact.

Large binaries (checkpoints, GGUFs, corpora) are **not** committed — they are regenerable, and
trained models live on the Hugging Face Hub (`huggingface/` holds the cards and upload script).

This is a teaching codebase that students read. Comments explaining *why* a choice was made are
part of the product, and the existing code is written that way — match it.
