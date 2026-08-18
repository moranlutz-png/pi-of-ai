# Observability and honesty — design

**Status:** approved, not yet implemented
**Date:** 2026-08-18
**Origin:** a review of ~38 proposed features across seven themed lists. This spec
is what survived; the rejections at the bottom matter as much as the build list,
because several of them look attractive and are not physically possible.

## The constraint that decides almost everything

This project has **two lanes**, and they have opposite properties. Nearly every
proposal that had to be rejected was a lane-2 feature aimed at lane 1.

| | Browser lane | Scratch lane |
|---|---|---|
| Engine | wllama — llama.cpp compiled to WASM | `scratch_coder`, PyTorch, ours |
| Models | Real ones (135M–2B GGUF) | 4 layers, 4 heads, 128-dim, char-level |
| Tensor access | **None** | Total |
| Character | A sealed appliance with a config panel | A glass box |

wllama's documented API is: **completions and embeddings** (high level);
**(de)tokenization, KV cache control, and sampling control** (low level). It does
**not** expose logits, hidden states, attention matrices, or per-layer weights,
and it cannot run a custom architecture. Anything needing those is either a
llama.cpp fork with new WASM bindings — a C++ project, not a feature — or it
belongs in `scratch_coder`, where it is free.

Verify against wllama's published type definitions before building anything that
assumes an API surface. The Rung 2 spec already set this precedent when it
checked for LoRA support and found none.

## The through-line

The differentiator is not capability, it is **honesty**. Nothing else in the
teaching-AI space is built around telling a student the truth about what small
models cannot do. The existing work already leans this way — a passport that
names what cannot be verified, an audit log that hashes rather than hoards, a
datasheet that reports its own keep rate, an agent loop that admits it did not
finish. Everything below extends that.

---

## Build list

### 1. The generation panel — sampling knobs, log-probs, tokens

**The highest-value item in the whole review.** Three features that are weak
apart and excellent together, all within the documented API.

- **Top-5 log-probs.** At each generated token, show the top five candidates and
  their percentages. This is the demo that makes "it predicts the next token"
  stop being a slogan.
- **Sampling knobs.** Temperature, top-k, top-p exposed live (wllama has
  sampling control; llama.cpp supports Mirostat too). Changing temperature and
  watching the distribution flatten or spike is the payoff.
- **Tokenizer view.** `tokenize()` is in the low-level API. Show how a word
  splits — that `strawberry` becomes several tokens is the cheapest good
  explanation of why models miscount its letters.

Verify log-prob availability in wllama first. If it is not exposed there, scope
the feature to the **Ollama runtime**, whose OpenAI-compatible endpoint supports
`logprobs`, and say so in the UI rather than faking it where it is unavailable.

This also feeds the accuracy work: when a model states something confidently and
wrongly, the interesting question is what the runner-up tokens were and how thin
the margin was.

### 2. GGUF header parsing, with validation

One job, two payoffs.

**Honest metadata.** GGUF headers carry tensor shapes, layer counts, embedding
dimensions and quantisation level, and parsing them in JS is doable. Today a
dropped `.gguf` shows six *unverifiable* fields in the model passport because
`models.json` is metadata typed by hand. Read the header and those become facts
resolved from the artifact — the same argument as provenance.

**A safety gate.** GGUF parser vulnerabilities are a documented class; malformed
headers causing heap overflows in llama.cpp have been found before. This app
loads `.gguf` from arbitrary user-supplied URLs, which is the exposure. Validate
tensor counts, layer dimensions and offsets before handing untrusted bytes to
the parser; reject implausible values and say why.

Note WASM meaningfully limits the blast radius — an overflow inside wllama's
linear memory cannot reach the host — so this is defence in depth, not a hole.
It costs almost nothing once the parser exists.

### 3. NaN and exploding-gradient breaker

**Confirmed absent** from `scratch_coder/train_forever.py`, `scratch_coder/train.py`
and `rules_baker/train/train_lora.py` — no `isnan`, no `isfinite`, no
`clip_grad_norm_` anywhere.

Reframe the harm: NaN does not lock up hardware. It poisons every subsequent
weight update while training *appears to continue normally*. In a one-hour
lesson that is the worst possible failure — forty minutes of Colab, a GGUF that
downloads and loads fine and emits garbage, with no signal that it broke. The
Rung 2 spec names "a bad bake is undiagnosable" as an unsolved risk; this is one
concrete cause.

Check `torch.isfinite(loss)` each step, add gradient clipping, abort loudly with
the step number. Log per-layer gradient norms alongside — cheap, and it turns
vanishing/exploding gradients into something a student can watch rather than a
phrase they are told.

### 4. Fix the WebGPU label

`index.html` reports **"WebGPU: yes"** in green on the dashboard and "available"
in Settings. Both are literally true (`'gpu' in navigator`) and both mislead: a
student reading a green "yes" on a page about running models locally will
conclude the GPU is doing the work. It is not — wllama runs on CPU via WASM SIMD
and threads. Every token is computed on the CPU.

Change to something like *"WebGPU: available — not used; inference runs on the
CPU"*. One string, removes a real misconception, and for a project built on
honesty it is the cheapest possible win.

### 5. KV cache clear

There is currently **no cache reset anywhere in the app**. New chat clears the
transcript and the history array, but the model's KV cache in WASM keeps holding
the previous conversation's state. wllama exposes KV cache control, so this is
directly buildable, and on a 4GB Chromebook reclaiming a 4096-token cache is
real memory.

Label it for what it does — clears the conversation's working memory, not the
weights. A student clicking "release memory" and seeing 270MB still resident
would reasonably conclude it did nothing.

### 6. Degenerate repetition stop

A small model looping `and and and` until `maxtok` is not a crash, it is a
wasted run. `repeat_penalty` is reachable through wllama's sampling control, and
an n-gram repeat detector that stops early is cheap.

**Do not** implement the originally proposed "purge context arrays" remedy — it
is silent trimming, which the multi-turn chat spec forbids in so many words:
silent amnesia is not understandable, an explicit stop is.

### 7. Colab OOM survival

Colab hands out whatever GPU it feels like — T4, L4, A100 — and a batch size
tuned for one OOMs on another. A catch-OOM-halve-batch-retry loop turns a dead
session into a slower one. Given a dropped session leaves no time to retry
inside a one-hour lesson, this is a meaningful save. Frame it as "survive
whatever GPU you got", not as a scheduler.

### 8. Finish what Rung 2 already specced

Two sub-projects from `2026-08-17-bake-into-weights-design.md` remain unbuilt:
the **loss curve** rendered from `training.json`, and the **compare view** (same
prompt, no system prompt, base vs baked side by side). Both are specced and
approved; they need implementing, not designing.

### 9. Ship the adapter for Ollama

`train_lora.py` already saves a LoRA adapter and `export_gguf.py` merges it into
a GGUF. The browser needs the merged file because wllama has no adapter API —
but **Ollama can load adapters directly**. Shipping the adapter alongside the
merged model is 50MB instead of 500MB for Ollama users. The Rung 2 spec flagged
this and deferred it.

### 10. Documents → seeds

`generate_dataset.py` is already a local synthetic training-pair generator. The
worthwhile extension is a different *input*: ingest a folder of user documents,
clean them, and feed them as seeds. It should emit prompt/response pairs via the
teacher, **not** tokens — tokenisation happens at training time, and a folder of
tokens is not a trainable dataset.

Expect the datasheet's keep rate to drop with messy input. That is the honest
signal working, not a bug.

### 11. Give `scratch_coder` a visual front end

**The largest item, and a separate project.** Eight rejected proposals needed
tensor access; all of them are free in PyTorch. `scratch_coder` should become the
interpretability product — currently framed as "it will not become a useful
coder, that's the point", the stronger identity is **the model small enough to
see all the way through**.

At 4 layers, 4 heads and 128 tokens of context you can render attention
*completely* rather than sampled, which is better teaching than a large model's
map, not worse.

Carries: attention maps, hidden-state activation canvas, gradient norms,
trained-vs-random weight comparison, and a `layers/` directory with a template
for custom layers.

One honesty warning: **trained transformer weights rendered as a raw grid look
like noise.** Not structured colour — noise. Do not promise a student swirls that
never appear. What does show real structure: embedding neighbourhoods (similar
characters genuinely cluster, renderable as a 2D projection) and per-layer weight
norms. The starkest honest demo is trained versus freshly-random, side by side.

Cost to name: `scratch_coder` has no UI today — it is `train.py` and `sample.py`
at a CLI. This means building one.

---

## Explicitly not building, and why

These were proposed and rejected. Recorded so they are not re-litigated.

- **Live parameter expansion (2.3M → 1B at runtime).** You cannot grow a trained
  model's parameter count and have the result mean anything — the new weights are
  untrained, so a working small model becomes a broken large one. Every research
  method for model growth (Net2Net, LiGO) requires training after expansion. A 1B
  model's weights are ~2GB that must come from somewhere.
- **Partial weight execution.** Transformer layers are sequential; you cannot run
  "only the tuned segment" and get sense out. Random weights produce noise, not
  degraded output. The teaching version — zeroed vs random vs trained, compared —
  belongs in `scratch_coder`.
- **Concurrent dual runtimes.** Two models resident doubles RAM on a device chosen
  because memory is scarce. The compare view solves the same user-visible need
  sequentially.
- **Hot-reload of inference or sampler code.** The code you would swap is compiled
  into WASM and unreachable. You would build a reload mechanism for the shell,
  which is the part that rarely changes.
- **Custom sampler algorithms.** Expose llama.cpp's knobs; you cannot replace its
  samplers from JS.
- **Browser-side quantisation.** Needs llama.cpp's `quantize` binary. Belongs in
  the Colab notebook, where a compiler exists — the bake spec already made this
  call and defaulted to F16 for exactly this reason.
- **Thermal telemetry (core temps, clock rates, NPU).** No browser exposes these,
  and none will. The honest proxy already exists: `bench.js` infers throttling
  from measured decay across runs.
- **VRAM layer-offload slider.** wllama runs on CPU; there are no GPU layers to
  allocate. Real for Ollama (`num_gpu`), but Ollama already auto-tunes it well.
- **Embedded code editor for engine internals.** Inference and sampler code is in
  WASM; PyTorch does not run in Pyodide, so `scratch_coder` layers cannot be
  edited in-browser either. An editor for *sandbox* Python is a different and
  smaller feature.
- **A sandbox execution boundary.** Already present by construction —
  `pyworker.js` runs Pyodide in a Web Worker with no host filesystem access. One
  caveat worth documenting: `loadPackagesFromImports` fetches packages from a
  CDN, so an `import` in model output can trigger a network request.

## Testing

No test framework; verification is in-browser against the running app, per
`CLAUDE.md`. Start the server from the checkout under test:

```bash
cd rules_baker/web && python3 serve.py 8123
```

A change is not verified until it has been run in a browser and the observed
value reported. Visual changes need a screenshot or a computed-style read.
