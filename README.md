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
  **in the browser via WebAssembly** (`web/index.html`, powered by wllama).
- See [`rules_baker/README.md`](rules_baker/README.md) and the
  [bake runbook](rules_baker/train/BAKE_RUNBOOK.md).

### 2. `scratch_coder/` — a coding model where *every weight is ours*
A tiny nanoGPT-style, character-level GPT written from scratch (`model.py` — our
own attention/MLP/blocks) and trained from random init on real Python source.
Nothing downloaded; every parameter trained here.

It will **not** become a useful coder — that's the point. Watch it climb from
random noise to real Python *structure* (defs, `self`, indentation, matched
brackets) with invented words, and you've seen, end to end, what a language model
learns and what it can't. See [`scratch_coder/README.md`](scratch_coder/README.md).

---

## Two phases, one idea

The heavy part (training / baking) happens **once**, on a GPU or a free cloud
notebook. The result runs **anywhere** on CPU. "Runs on any computer" is about
*inference* — and that's the part that has to be accessible.

---

## Quick start

Each build is self-contained Python with a short README. In brief:

```bash
# from-scratch coder — trains on your machine's own Python stdlib
cd scratch_coder
python train_forever.py        # resumable; Ctrl-C and rerun to continue
python sample_big.py "def "     # watch what it has learned

# rules-baker — generate training data against a local teacher (e.g. Ollama)
cd rules_baker
python data_gen/generate_dataset.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

Large binaries (trained checkpoints, GGUF models, corpora) are **not** committed
here — they're regenerable, and the trained models are meant to live on the
Hugging Face Hub. See each build's README for the current model links.

## License

MIT — see [LICENSE](LICENSE). Contributions and forks welcome; that's the whole
idea.
