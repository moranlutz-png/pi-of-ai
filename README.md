# Pi-of-AI — the Raspberry Pi of AI

Local, offline, tinkerable AI you can actually understand. Cheap, hackable,
private, and small enough to run on almost any computer — even a locked-down
4 GB school Chromebook, straight in the browser.

The goal isn't a bigger model. It's the opposite: the smallest, most
transparent thing that still teaches **how AI actually works** — and does
something genuinely useful along the way.

> **Status:** early, honest, and a work in progress. Nothing here pretends to be
> a frontier model. It's a teaching kit and a set of small, real tools.

---

## What's inside

This repo has two independent builds that share one philosophy — *small, local,
yours*:

### 1. `rules_baker/` — bake your house style into a model's weights
Fine-tunes a small open model (Qwen2.5-Coder) so it silently obeys a project's
coding rules **without** those rules eating up the context window on every
request.

The trick: the **teacher** model *sees* the rules and writes compliant code, but
the stored **student** training prompt has the rules **stripped out** — so the
student learns to obey the house style on its own. Clean context, no rule-bloat,
no lag.

- Runs entirely against a **local** OpenAI-compatible endpoint (Ollama / vLLM /
  llama.cpp) — nothing leaves the room.
- Bakes once on a GPU (or free Colab), then runs **CPU-only** anywhere — including
  **in the browser via WebAssembly** (`rules_baker/web/`, powered by wllama).
- See [`rules_baker/README.md`](rules_baker/README.md) and the
  [bake runbook](rules_baker/train/BAKE_RUNBOOK.md).

### 2. `scratch_coder/` — a coding model where *every weight is ours*
A tiny nanoGPT-style, character-level GPT written from scratch (`model.py` — our
own attention / MLP / blocks) and trained from random init on real Python source.
Nothing downloaded; every parameter trained here.

It will **not** become a useful coder — that's the point. Watch it climb from
random noise to real Python *structure* (`def`, `self`, indentation, matched
brackets, docstrings) with invented-but-plausible words, and you've seen, end to
end, what a language model learns about *form* and what it can never learn about
*meaning*. See [`scratch_coder/README.md`](scratch_coder/README.md).

**It ships in sizes.** The same from-scratch model is trained at several
parameter counts, so you can watch capability grow with scale:

| Tier | Params | Trained on | For |
|---|---|---|---|
| **Featherweight** | ~0.5M | 31 MB stdlib | anything, fastest |
| **Chromebook** | ~1.2M | 31 MB stdlib | the sweet spot |
| **Laptop** | ~1.9M | 31 MB stdlib | sharper output |
| **Max** | ~6.4M | 80 MB stdlib + site-packages | the any-PC ceiling |
| **Ultra** | ~14.3M | 128 MB Python ecosystem | powerful-PC showcase (57 MB `weights.bin`, slow in-browser) |

The lesson these tiers teach together: **more scale buys convincing *syntax* —
type hints, OOP, argument unpacking — but never crosses into *semantics*** (it
still references variables it never declared). Data matters more than size: the
2 MB → 31 MB jump helped far more than doubling the parameters.

---

## See it in your browser (no build step, no GPU)

The interactive **inspector** is the heart of the teaching kit. It runs the
from-scratch model's real forward pass in JavaScript, so you can watch it think:

```bash
cd rules_baker/web && python3 serve.py 8123      # open http://localhost:8123
```

- **Sidebar → Scratch-Coder** — the inspector. Pick a size from the model-picker
  dropdown, then explore:
  - **Generate** — type a prompt and it continues it character by character, live.
  - **Embedding neighbourhoods** — an orbitable 3D graph of what each character's
    learned vector is near, with "what a perfect model *should* recover" beside it.
  - **Live attention** — every head's attention heatmap over your prompt, labelled
    with the actual characters and readable in plain English on hover.
  - **Training curves**, the **source code** it's built from, and the **training
    data** it learned from — all in the page.
- **Library** — every model as a card, including each Scratch-Coder tier; "Use"
  makes it the active model so you can generate with it on the dashboard, just
  like a downloaded GGUF. The `rules_baker` browser runtime (wllama in-tab, or a
  detected local Ollama) lives here too.

The Scratch-Coder page is also served standalone from `scratch_coder/web/`.

---

## Two phases, one idea

The heavy part (training / baking) happens **once**, on a GPU or a free cloud
notebook. The result runs **anywhere** on CPU. "Runs on any computer" is about
*inference* — and that's the part that has to be accessible.

Training the from-scratch model needs CUDA (a modest GPU is plenty — the tiers
above were trained on a 4 GB GTX 1050 Ti in minutes each). Inference — including
the whole browser inspector — is pure CPU/JS.

---

## Quick start

```bash
# --- run the browser inspector (above) ---
cd rules_baker/web && python3 serve.py 8123

# --- from-scratch coder: build a corpus, then train a tier ---
cd scratch_coder
python prepare_data.py                                   # ~31 MB of stdlib Python (ASCII)
python train.py --n-layer 6 --n-head 4 --n-embd 128 --iters 10000   # the ~1.2M "Chromebook" tier
python export_inspect.py --weights                       # write inspect.json + weights.bin for the page

#   train.py flags: --n-layer/--n-head/--n-embd/--block (size), --lr (lower for
#   bigger models), --data (alternate corpus dir), --iters, --tag (per-tier
#   checkpoint), --resume (continue a checkpoint). Checkpoints save atomically.
#   Bigger corpus:  python prepare_data.py --out-dir data_max --max-bytes 80000000 --site-packages
python sample_big.py "def "                              # or watch it from the CLI

# --- rules-baker: generate training data against a local teacher (e.g. Ollama) ---
cd rules_baker
python data_gen/generate_dataset.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

## What's committed

- **The Scratch-Coder tier weights ARE committed** (`*/web/tiers/*/weights.bin`,
  ~100 MB total across the five tiers) so the browser inspector works out of the box on a fresh clone —
  no GPU, no re-train. Deliberately overriding the usual "no binaries" rule
  because being runnable everywhere is the whole point of *this* model.
- **Everything else regenerable is not committed** — the rules-baker's merged
  GGUFs and its corpora, and the from-scratch model's raw training corpora
  (`data*/`). The baked GGUFs are meant to live on the Hugging Face Hub; see
  `huggingface/` and each build's README for links.

## License

MIT — see [LICENSE](LICENSE). Contributions and forks welcome; that's the whole
idea.
