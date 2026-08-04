# WSL2 + CUDA — Local Baking Station Runbook 🐧⚡

Turn this Windows machine into a **fully-offline Unsloth baking station** using the
**GTX 1050 Ti (4 GB, Pascal, compute 6.1)**. After a one-time setup, you bake real
models locally with no cloud, no Colab.

> **Reality check for this card.** 4 GB + Pascal is great for **0.5B** (recommended)
> and OK for **1.5B** (via QLoRA). It will **not** fit 7B — use Colab for those.
> Two Pascal quirks handled below: **no bf16** (fp16 only) and **bitsandbytes is
> happier avoided at this size** (we train 0.5B in plain fp16).

The setup needs internet and pulls several GB (Ubuntu ~0.5 GB, torch+CUDA ~2.5 GB,
model ~1 GB). Baking itself is offline once cached.

---

## ⚠️ The one rule that breaks everyone

Install the NVIDIA driver **on Windows only**. **NEVER** install an NVIDIA driver
*inside* Ubuntu. WSL2 uses a special GPU stub that forwards to the Windows driver;
a Linux driver clobbers it and `nvidia-smi` stops working. If you ever ran
`apt install nvidia-driver-xxx` in WSL, that's the bug.

Your Windows driver already works (`nvidia-smi` showed the 1050 Ti). If it's old,
update it from GeForce Experience or nvidia.com — nothing else driver-wise.

---

## 1 · Install WSL2 + Ubuntu

**PowerShell (Run as Administrator):**

```powershell
wsl --install -d Ubuntu
```

This enables the WSL + Virtual Machine Platform features, installs Ubuntu, and sets
WSL2 as default. **Reboot when prompted.** On first launch Ubuntu asks you to create
a UNIX username + password (your own local Linux account — pick anything you'll
remember).

Confirm you're on version **2** (not 1):

**PowerShell:**

```powershell
wsl -l -v
```

`VERSION` must read `2`. If it says `1`, run `wsl --set-version Ubuntu 2`.

---

## 2 · Verify the GPU is visible inside Linux

**Ubuntu (WSL2) bash:**

```bash
nvidia-smi
```

You should see **NVIDIA GeForce GTX 1050 Ti**. If "command not found" or no devices
appear, jump to Troubleshooting — do not proceed until this works.

---

## 3 · Base tooling

**Ubuntu (WSL2) bash:**

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip build-essential git
```

> CUDA toolkit (`nvcc`) is **optional** — PyTorch/Unsloth pip wheels bundle the CUDA
> runtime, so you don't need it just to bake. Skip it unless you plan to compile
> custom kernels. (If you ever do, use NVIDIA's **`wsl-ubuntu`** CUDA repo variant,
> which installs the toolkit *without* a driver.)

---

## 4 · Install Unsloth in a clean venv

**Ubuntu (WSL2) bash:**

```bash
cd ~
python3 -m venv rbenv && source rbenv/bin/activate
pip install --upgrade pip
pip install unsloth
```

Verify torch actually sees the GPU:

**Ubuntu (WSL2) bash:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect something like: `2.x.x True NVIDIA GeForce GTX 1050 Ti`. If `False`, see
Troubleshooting (almost always WSL1, or a stray Linux driver).

---

## 5 · Tune the config for this card

Edit `configs/qwen_coder_0_5b_chromebook.yaml` and set:

```yaml
student:
  load_in_4bit: false      # 0.5B fits in 4 GB in fp16; avoids bitsandbytes-on-Pascal
```

Why: Pascal (6.1) predates good 4-bit kernel support. A 0.5B in fp16 is tiny
(~1 GB weights + a little for LoRA), so QLoRA buys you nothing here and only risks
bitsandbytes headaches. For **1.5B**, do the opposite — keep `load_in_4bit: true`
so it fits 4 GB. bf16 is unsupported on Pascal; Unsloth will pick **fp16**
automatically, so you don't set anything for that.

---

## 6 · Bake for real

Your Windows files live under `/mnt/c`, so you edit on Windows and bake in Linux
against the same files.

**Ubuntu (WSL2) bash:**

```bash
cd /mnt/c/Users/<you>/pi-of-ai/rules_baker
python train/train_lora.py  --config configs/qwen_coder_0_5b_chromebook.yaml
python export/export_gguf.py --config configs/qwen_coder_0_5b_chromebook.yaml
```

The GGUF lands in `outputs/qwen_coder_0_5b_rules/gguf/` — already on your Windows
drive via `/mnt/c`, ready for the web page.

---

## 7 · Verify — drop into the page

**PowerShell (Windows):**

```powershell
py "$env:USERPROFILE\pi-of-ai\rules_baker\web\serve.py"
```

Open <http://localhost:8123>, drag your baked `.gguf` onto the drop zone, and run
the stock-vs-baked A/B from `BAKE_RUNBOOK.md`. Then score it with
`eval/eval_rules.py` for the objective compliance delta (Stage 7).

---

## GTX 1050 Ti quick reference

| Model | Setting | Fits 4 GB? |
|-------|---------|-----------|
| 0.5B  | `load_in_4bit: false` (fp16 LoRA) | ✅ easily — **recommended** |
| 1.5B  | `load_in_4bit: true` (QLoRA)       | ✅ tight but works |
| 7B+   | —                                   | ❌ use Colab |

---

## Troubleshooting

- **`nvidia-smi` missing / no devices in WSL** → update the Windows NVIDIA driver;
  run `wsl --update` (PowerShell); confirm WSL2 with `wsl -l -v`. Never install a
  Linux driver in WSL.
- **`torch.cuda.is_available()` is False** → you're on WSL1 (`wsl --set-version
  Ubuntu 2`), or a Linux NVIDIA driver got installed (remove it), or the Windows
  driver is too old.
- **`bitsandbytes` errors / warnings on Pascal** → set `load_in_4bit: false` for
  0.5B (recommended anyway).
- **CUDA out of memory (4 GB)** → drop `train.batch_size` to 1, raise `grad_accum`;
  stay on 0.5B, not 1.5B.
- **First run seems to hang** → Unsloth compiles Triton kernels on first use; give
  it a minute.
- **GGUF export fails** → use the manual `convert_hf_to_gguf.py` fallback in
  `BAKE_RUNBOOK.md`.
- **WSL disk keeps growing** → HF models cache in `~/.cache/huggingface` inside the
  WSL vhdx. To cache on the Windows drive instead:
  `export HF_HOME=/mnt/c/Users/<you>/hf_cache` (slower disk I/O, but
  keeps the vhdx small).
