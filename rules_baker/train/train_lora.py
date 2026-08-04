"""
train_lora.py  —  QLoRA fine-tune with Unsloth (Gap 2: bake rules into weights).

Runs on a single consumer GPU (16-24GB). The output is a small LoRA adapter
(~50-200MB) that a developer can ship, swap, and stack — the "cartridge" for a
given company's house style. Base model stays untouched and reusable.

Usage (Linux / WSL2 with CUDA — Unsloth needs a CUDA GPU):
    python train/train_lora.py --config configs/qwen_coder_7b.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("rules_baker.train")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="QLoRA fine-tune for rules-baking.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[1]

    # Imports are deferred so the file can be inspected without a GPU/CUDA present.
    from unsloth import FastLanguageModel  # noqa: PLC0415
    from unsloth.chat_templates import get_chat_template  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from trl import SFTTrainer, SFTConfig  # noqa: PLC0415

    s, lora, tr = cfg["student"], cfg["lora"], cfg["train"]

    # --- 1. Load 4-bit base + attach LoRA adapters --------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=s["base_model"],
        max_seq_length=s["max_seq_length"],
        load_in_4bit=s["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=tr["seed"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # --- 2. Load the generated dataset and render with the chat template ----
    ds_path = root / cfg["output_jsonl"]
    dataset = load_dataset("json", data_files=str(ds_path), split="train")
    # Our generator already emits {"messages":[{role,content}...]} — the exact shape
    # apply_chat_template wants — so no standardize_sharegpt conversion is needed.

    def _format(batch: dict[str, Any]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in batch["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(_format, batched=True, remove_columns=dataset.column_names)
    logger.info("Training on %d examples", len(dataset))

    # --- 3. Train -----------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=s["max_seq_length"],
        args=SFTConfig(
            per_device_train_batch_size=tr["batch_size"],
            gradient_accumulation_steps=tr["grad_accum"],
            warmup_steps=tr["warmup_steps"],
            num_train_epochs=tr["epochs"],
            learning_rate=tr["learning_rate"],
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=tr["seed"],
            output_dir=tr["output_dir"],
            report_to="none",
        ),
    )
    trainer.train()

    # --- 4. Save the adapter (the shippable "house-style cartridge") --------
    out = Path(tr["output_dir"]) / "adapter"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Saved LoRA adapter -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
