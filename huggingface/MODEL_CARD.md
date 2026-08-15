---
license: mit
library_name: gguf
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
  - code
  - qwen2.5-coder
  - rules-baker
  - pi-of-ai
  - gguf
  - qlora
---

# Pi-of-AI · Rules-Baker (smoke-test model)

Part of **[Pi-of-AI](https://github.com/moranlutz-png/pi-of-ai)** — the "Raspberry
Pi of AI." This is the Rules-Baker experiment: a small model with a project's
house-style coding rules **baked into its weights**, so it obeys them with an
empty system prompt (no rule-bloat in the context window).

## ⚠️ What this checkpoint is

`smoke_housestyle.gguf` is a **pipeline smoke-test**, not a finished model. It was
LoRA-trained on a tiny *mock* dataset purely to prove the train → merge → GGUF →
run-in-browser loop end to end. It **overfits the mock template** and does not yet
follow arbitrary tasks. The real, task-following bake comes from Stage 3+ teacher
data (see the repo roadmap). Published here for transparency and to demonstrate
the closing-the-loop test — treat it as a demo, not a tool.

- **Base model:** Qwen2.5-Coder-0.5B-Instruct
- **Method:** LoRA (vanilla PEFT), merged, converted to GGUF (`q8_0`, ~506 MB)
- **Runtime:** llama.cpp / Ollama, or **in-browser via wllama** (WebAssembly, CPU)

## The idea (why bake rules into weights?)

The **teacher** model sees your coding rules and writes compliant code; the stored
**student** training prompt has the rules **stripped out**. The student learns to
obey silently — so at inference time your IDE context stays clean and fast.

## Use it in the browser

Open `rules_baker/web/index.html` from the repo, then drag-drop this `.gguf` in.
Same tiny prompt, rules obeyed — that's the closing-the-loop test.

## License

MIT. See the [project repo](https://github.com/moranlutz-png/pi-of-ai).
