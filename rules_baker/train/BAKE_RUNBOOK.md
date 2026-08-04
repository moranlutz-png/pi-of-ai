# Stage 5 — Bake Runbook 🔥

Turn your dataset into a droppable `.gguf`. This is the **one-time GPU step**. The
output is a quantized model file you drag into `web/index.html` — no GPU ever
needed again to *run* it.

Baking a **0.5B** is tiny: minutes on a free Colab T4, and it fits on any 6GB+ GPU.

---

## ⚡ TL;DR

```
1. Get a GPU (free Google Colab, or local WSL2/Linux + NVIDIA).
2. Upload the rules_baker/ folder (must contain datasets/rules_sft.jsonl).
3. pip install unsloth
4. python train/train_lora.py  --config configs/qwen_coder_0_5b_chromebook.yaml
5. python export/export_gguf.py --config configs/qwen_coder_0_5b_chromebook.yaml
6. Download outputs/qwen_coder_0_5b_rules/gguf/*.gguf
7. Drag it into http://localhost:8123  →  ask a coding task  →  watch the rules apply.
```

> **Smoke-bake first.** You already have a 45-example mock `datasets/rules_sft.jsonl`
> from Stage 2. Bake *that* first to prove the whole train→export→drop-in loop end
> to end (the model won't be smart, but the pipeline will be verified). Then do
> Stage 3 for real data and re-bake. De-risk the machinery before the data.

---

## Path A — Google Colab (recommended if you have no GPU)

**1. New notebook → Runtime → Change runtime type → T4 GPU.**

**2. Upload the project.** Zip `rules_baker/` locally first:

```powershell
# PowerShell (your Windows machine)
Compress-Archive -Path "$env:USERPROFILE\pi-of-ai\rules_baker\*" -DestinationPath "$env:USERPROFILE\Desktop\rules_baker.zip" -Force
```

Then in a **Colab cell**:

```python
# Colab cell
from google.colab import files
up = files.upload()                      # pick rules_baker.zip
!unzip -q rules_baker.zip -d rules_baker
%cd rules_baker
```

**3. Install Unsloth** (Colab cell):

```python
# Colab cell
!pip install -q unsloth
```

**4. Train** (Colab cell):

```python
# Colab cell
!python train/train_lora.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

**5. Export to GGUF** (Colab cell):

```python
# Colab cell
!python export/export_gguf.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

**6. Download the `.gguf`** (Colab cell):

```python
# Colab cell
import glob
from google.colab import files
path = glob.glob("outputs/qwen_coder_0_5b_rules/gguf/*.gguf")[0]
print("downloading", path)
files.download(path)
```

---

## Path B — Local WSL2 / Linux (NVIDIA GPU)

Unsloth needs CUDA; it does **not** run on bare Windows. Use WSL2 (Ubuntu) or a
Linux box. From inside WSL2, your Windows files are under `/mnt/c/...`.

```bash
# WSL2 / Linux bash
cd /mnt/c/Users/<you>/pi-of-ai/rules_baker

python -m venv .venv && source .venv/bin/activate
pip install unsloth

python train/train_lora.py  --config configs/qwen_coder_0_5b_chromebook.yaml
python export/export_gguf.py --config configs/qwen_coder_0_5b_chromebook.yaml

ls -lh outputs/qwen_coder_0_5b_rules/gguf/*.gguf   # your baked model
```

The `.gguf` is already on your Windows drive (via `/mnt/c`), so `web/index.html`
can load it directly.

---

## 7. Verify — the closing-the-loop test

1. Start the page on your machine:

```powershell
# PowerShell (your Windows machine)
py "$env:USERPROFILE\pi-of-ai\rules_baker\web\serve.py"
```

2. Open <http://localhost:8123>.
3. First load the **stock** Chromebook 0.5B preset, ask the default request → it
   ignores your house rules.
4. Now **drag your baked `.gguf`** onto the drop zone, ask the **same** request
   with the same tiny system prompt → the rules should now apply.
5. Then run `eval/eval_rules.py` against both for the objective compliance delta
   (Stage 7). If baked > stock, the rules are in the weights. Done.

---

## Knobs worth knowing (in `configs/qwen_coder_0_5b_chromebook.yaml`)

| Knob | Default | When to change |
|------|---------|----------------|
| `student.base_model` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Swap to `-1.5B-Instruct` for the laptop tier |
| `student.load_in_4bit` | `true` | Set **false** for 0.5B — it fits in fp16 easily and trains a touch better |
| `train.epochs` | `4` | More epochs help tiny models learn a narrow rule set; watch for over-repetition |
| `export.gguf_quant` | `q4_k_m` | `q8_0` for the laptop tier (bigger, slightly better) |

---

## Troubleshooting (the two things that actually break)

**GGUF export fails / llama.cpp won't compile.** `save_pretrained_gguf` builds
llama.cpp under the hood and sometimes chokes. Fallback — merge to 16-bit, then
convert by hand:

```bash
# WSL2 / Colab bash  — run in Python first to merge:
python -c "from unsloth import FastLanguageModel; m,t=FastLanguageModel.from_pretrained('outputs/qwen_coder_0_5b_rules/adapter', load_in_4bit=False); m.save_pretrained_merged('merged_16bit', t, save_method='merged_16bit')"

git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py merged_16bit --outfile model-f16.gguf --outtype f16
cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build -j --config Release
./llama.cpp/build/bin/llama-quantize model-f16.gguf qwen-coder-housestyle-mini.Q4_K_M.gguf q4_k_m
```

**`SFTTrainer` / `SFTConfig` throws an unexpected-keyword error.** `trl` changes
its API between releases. Unsloth pins a compatible `trl`, so prefer a fresh
`pip install unsloth` in a clean environment. If it still fails, pin explicitly:
`pip install "trl==0.11.4" "transformers>=4.44"` and re-run.

**Out of memory** (unlikely at 0.5B): drop `train.batch_size` to 1 and raise
`grad_accum`, or set `load_in_4bit: true`.
