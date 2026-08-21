# Rules-Baker 🔥

> **Gap 2 — The "System Prompt" Bloat killer.** Bake your project's architecture
> rules and house style directly into a small open-weight model's *weights* via
> QLoRA, so your IDE context window stays clean and fast. No rule-bloat, no VRAM
> lag, no cloud. Part of the **Raspberry Pi of AI** stack.

## The core idea

Instead of shoving a 4,000-token `STYLE_GUIDE.md` into every prompt, you bake the
rules once into a tiny **LoRA adapter** (~50–200MB). At inference the system prompt
is one line — the rules live in the weights.

The training trick:

```
  TEACHER model  ──sees your rules──▶  writes compliant code
                                              │
                        (strip the rules out of the stored prompt)
                                              ▼
  STUDENT example:   bare task  ──▶  compliant code
```

The student learns to obey **silently**. Everything runs on a local
OpenAI-compatible endpoint (Ollama / vLLM / llama.cpp) — nothing leaves the room.

## 🎯 Locked target: runs in a browser on a school Chromebook

The mission is teaching — make AI as accessible as the Raspberry Pi made coding.
School Chromebooks (4GB RAM, no GPU, Linux/Play-Store disabled by admin policy)
are the floor, and the only runtime they can always run is **the browser**. So
the finished model runs **in a web page via WebAssembly** — no install, no admin
rights, fully offline after one download. Default config:
`configs/qwen_coder_0_5b_chromebook.yaml`.

| | Floor (Chromebook) | Laptop tier |
|---|---|---|
| Model | **Qwen2.5-Coder-0.5B** | Qwen2.5-Coder-1.5B |
| GGUF @ Q4_K_M | ~0.4 GB | ~1 GB |
| Free RAM needed | ~1.5 GB | ~3 GB |
| Runtime | **wllama** (llama.cpp → WASM, CPU) | same + WebGPU when present |

0.5B is a *feature*: rules-baking is a narrow, pattern-based task a tiny model can
learn, and "a model small enough to live in a browser tab, learning your rules,
with the eval harness proving it worked" is the whole lesson. `wllama` loads our
existing GGUF unchanged — no new export format.

## Two phases — only ONE needs a GPU

The "runs on nearly any computer" promise is about **running** the finished model,
not **baking** it. Don't confuse them:

| | **Baking** (make the adapter) | **Running** (use it) |
|---|---|---|
| When | Once, by you | Constantly, by anyone |
| Hardware | A CUDA GPU (or ~1hr rented) | **CPU-only is fine** — laptop, mini-PC, Pi 5 |
| Ships? | No — it's the oven | Yes — the quantized GGUF is the product |

**No GPU?** Pick a 1B–3B base (`Qwen2.5-Coder-1.5B`, `Llama-3.2-1B/3B`) — QLoRA-trains
on a 6–8GB card or free Colab, and runs at Q4 on almost any CPU. Edit one line in the config.

## Data quality gates (hardening)

The generator drops bad synthetic data instead of training on it. Tunable in the
`quality:` block of the config:

- **`require_parse`** — anything that fails `ast.parse` is dropped/retried.
- **`min_code_chars`** — trivially short/empty generations are dropped.
- **`dedup`** — near-duplicate code (whitespace-normalized hash) is dropped.
- **`max_retries`** — rejected examples are re-asked before giving up.
- **`eval_holdout_frac`** — reserves a slice of seeds for eval ONLY, written to
  `datasets/eval_seeds.txt`. Training never sees them, so your compliance score
  isn't inflated by testing on tasks you trained on.

Every drop is counted and printed at the end (keep-rate + reasons) — nothing is dropped silently.

## Pipeline

| Stage | Script | Runs on | Output |
|-------|--------|---------|--------|
| 1. Generate data | `data_gen/generate_dataset.py` | CPU (anywhere) | `datasets/rules_sft.jsonl` |
| 2. Train QLoRA | `train/train_lora.py` | CUDA GPU (16–24GB) | LoRA adapter |
| 3a. Export merged | `export/export_gguf.py` | CUDA GPU | quantized GGUF (~500MB) |
| 3b. Export adapter | `export/export_adapter.py` | CPU (anywhere) | GGUF adapter (~50MB) |
| 4. Serve | `export/Modelfile` / `Modelfile.adapter` | CPU/GPU edge | local `/v1` API |
| ✅ Eval | `eval/eval_rules.py` | CPU (anywhere) | compliance % (before/after) |

### Two ways to ship the same bake

Training produces an **adapter** — a delta against the base model, not a model.
What you do with it depends on who is running it:

| | Merged GGUF | Adapter |
|---|---|---|
| Size | ~500MB (whole model) | ~50MB (the change only) |
| Browser (wllama) | **required** | cannot load it |
| Ollama | works | works, and is the better deal |
| Ten sets of rules | ten whole models | one base + ten deltas |

The browser has no choice: wllama exposes no adapter API, so the LoRA has to be
folded into the weights and the whole thing shipped. Ollama applies an adapter on
top of a stock base at load time, so an Ollama user downloads a tenth as much —
and the base stays shared across every set of rules you ever bake.

One sharp edge, worth knowing before you hand it to a class: an adapter is only
meaningful against the exact weights it was trained on, and **Ollama does not
check**. `ollama create` accepts a mismatched adapter and reports success; the
failure arrives the first time you generate, as the model server exiting
(observed on Ollama 0.32.13). So generate once before you trust a build, and
leave `export.ollama_base_model` in the config alone unless you also changed
`student.base_model`.

## Seeds from your own documents

The seed list is the other half of what shapes a bake: the rules say *how* to
write, the seeds say *what to write about*. `data_gen/seeds/task_seeds.txt` ships
68 generic coding tasks. If you would rather the student practised on the shape
of work your team actually does, point the ingester at a folder of your own
documents:

```bash
python data_gen/documents_to_seeds.py --config configs/qwen_coder_0_5b_chromebook.yaml \
                                      --docs ~/handbook
```

It cleans the markup and the code out, splits what is left into blocks, and asks
the same local teacher to turn each block into concrete coding tasks. Output is
**seeds** — the teacher still writes every training pair downstream. It is
deliberately not tokens: tokenisation happens at training time from text, and a
folder of tokens is a dataset with the labels thrown away.

**Seeds must not mention style.** This is the one thing to understand before
using it. The whole bake works because the student never sees a rule — so a seed
like "write a logger that follows our naming convention" puts the rule back into
the prompt, and the model learns to look for the mention instead of applying the
style to everything. The teacher is instructed not to write such tasks and the
output is gated again afterwards, and those drops are counted as
`seed_mentions_style`.

**Where the keep rate falls.** Every drop is counted by reason into a
`from_documents.ingestsheet.json` beside the seeds, for the same reason the
datasheet reports its own. Observed on this repo's own `docs/` folder: with
`--no-teacher`, which uses cleaned excerpts as candidate seeds unchanged, the
keep rate is **0%** — prose describes work, it is not phrased as work, and every
candidate is dropped as `seed_not_a_task`, `seed_quotes_code` or
`seed_is_a_fragment`. With the teacher it is high, because the messiness has
already been absorbed upstream. Downstream, in `generate_dataset.py` against a
`qwen2.5-coder:1.5b` teacher, 15 document-derived seeds kept **87.9%** against
**100%** for 15 curated ones — a real drop, but a small sample, and the honest
signal mostly shows up in the ingest sheet rather than the datasheet.

Read the seed file before you bake on it. The gates can tell a task from a
sentence; they cannot tell a good task from a dull one.

## What you tinker with (the "Tinker Factor")

- **`data_gen/rules/example_rules.md`** — replace with YOUR rules. Each `## RULE <id>`
  block becomes one atomic, checkable rule. This file *is* the product config.
- **`data_gen/seeds/task_seeds.txt`** — the coding tasks the student practices on.
- **`configs/qwen_coder_7b.yaml`** — swap the base model, teacher, LoRA rank, epochs.
- **`eval/eval_rules.py`** — add a `check_<rule_id>()` to make a new rule measurable.

Swap `Qwen2.5-Coder-7B` for `Phi-4-mini` or `Llama-3.1-8B` by editing one line.

## Quickstart

**1. Generate the dataset** (needs a teacher model served locally, e.g. `ollama pull qwen2.5-coder:32b-instruct`):

```bash
pip install -r requirements.txt
python data_gen/generate_dataset.py --config configs/qwen_coder_7b.yaml
```

**2. Baseline eval — score the UNTUNED model first** (so you can prove the bake worked):

```bash
python eval/eval_rules.py --config configs/qwen_coder_7b.yaml --model qwen2.5-coder:7b-instruct
```

**3. Train the adapter** — full step-by-step in [`train/BAKE_RUNBOOK.md`](train/BAKE_RUNBOOK.md)
(Colab or WSL2). To turn a local Windows+NVIDIA box into an offline baking station,
see [`train/WSL2_CUDA_SETUP.md`](train/WSL2_CUDA_SETUP.md):

```bash
pip install unsloth
python train/train_lora.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

**4. Export + serve.** Merged, for the browser or for Ollama:

```bash
python export/export_gguf.py --config configs/qwen_coder_7b.yaml
ollama create qwen-coder-housestyle -f export/Modelfile
```

Or just the adapter, if you are serving through Ollama — a tenth of the size, and
no GPU needed for this step:

```bash
python export/export_adapter.py --config configs/qwen_coder_7b.yaml
ollama create qwen-coder-housestyle-adapter -f outputs/qwen_coder_7b_rules/adapter/Modelfile
```

**5. Post-bake eval — compare against the baseline:**

```bash
python eval/eval_rules.py --config configs/qwen_coder_7b.yaml --model qwen-coder-housestyle
```

If step 5 beats step 2, the rules are in the weights. That delta is your whole thesis, proven.

## Repo layout

```
rules_baker/
├── configs/            # one YAML per model — the swap-a-base-model surface
├── data_gen/
│   ├── rules/          # YOUR house-style rules (the config that matters)
│   ├── seeds/          # coding tasks to synthesize around
│   └── generate_dataset.py
├── train/train_lora.py # Unsloth QLoRA
├── eval/eval_rules.py  # objective rule-compliance scorer
├── export/             # merged GGUF, adapter GGUF, and both Ollama Modelfiles
└── datasets/           # generated SFT data lands here
```
