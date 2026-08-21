"""
bake_gpu.py  —  Windows/GPU LoRA bake of the house-style rules into the tiny student.

This is the Windows-friendly counterpart to train_lora.py (which uses Unsloth, and
Unsloth is Linux-only). It uses plain PEFT + transformers, trains a LoRA on the
rules_sft.jsonl chat data, MERGES the adapter back into the base weights, and saves a
merged fp16 model ready for GGUF conversion.

Design choices for a 4GB Pascal card (GTX 1050 Ti):
    * fp16, NOT 4-bit  — bitsandbytes is flaky on Pascal; 0.5B in fp16 fits easily.
    * gradient checkpointing + small batch — keeps VRAM well under 4GB.
    * HF cache forced onto D: — keeps the ~1GB base-model download off the tiny C:.

RUN IN YOUR OWN TERMINAL (survives session teardowns; the bake takes ~10-20 min):

    D:\\rbgpu\\Scripts\\python.exe train/bake_gpu.py

Options: --max-seq 1024 (lower to 512 if you OOM), --batch 2 (lower to 1 if you OOM).

Outputs (under the config's train.output_dir):
    <output_dir>/merged/     the merged fp16 model  -> convert to GGUF next
    <output_dir>_adapter/    the LoRA adapter alone (small; keep for reproducibility)
"""
from __future__ import annotations

import os
# Keep the ~1GB base-model download off the cramped C: drive. Set before HF imports.
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_chat_examples(jsonl_path: Path) -> list[list[dict]]:
    """Read the SFT file: one JSON object per line, each with a `messages` list."""
    rows: list[list[dict]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        msgs = obj.get("messages")
        if msgs:
            rows.append(msgs)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU LoRA bake (Windows/PEFT).")
    ap.add_argument("--config", type=Path, default=Path("configs/qwen_coder_0_5b_chromebook.yaml"))
    ap.add_argument("--max-seq", type=int, default=1024, help="lower to 512 if VRAM is tight")
    ap.add_argument("--batch", type=int, default=2, help="lower to 1 if you OOM")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA. Run with D:\\rbgpu\\Scripts\\python.exe (the GPU venv).")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    base = cfg["student"]["base_model"]
    data_path = root / cfg["output_jsonl"]
    out_dir = root / cfg["train"]["output_dir"]
    merged_dir = out_dir / "merged"
    adapter_dir = out_dir.parent / (out_dir.name + "_adapter")

    # --- tokenizer + data ---------------------------------------------------
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    chats = load_chat_examples(data_path)
    print(f"{len(chats)} training examples from {data_path.name}")
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in chats]

    def _tok(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, max_length=args.max_seq)

    ds = Dataset.from_dict({"text": texts}).map(_tok, batched=True, remove_columns=["text"])

    # --- model + LoRA -------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()          # required so LoRA grads flow with checkpointing
    model.config.use_cache = False

    lc = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    # --- train --------------------------------------------------------------
    t = cfg["train"]
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["learning_rate"]),
        warmup_steps=t["warmup_steps"],
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        seed=t["seed"],
        report_to="none",
        optim="adamw_torch",
    )
    Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    # --- save adapter, then merge into the base weights ---------------------
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    print(f"adapter -> {adapter_dir}")

    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    print(f"merged model -> {merged_dir}")
    print("\nNEXT: convert the merged model to GGUF, then re-run eval_rules.py against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
