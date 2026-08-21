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
import gc
import logging
import math
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("rules_baker.train")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure.

    Matched on the message rather than the exception class on purpose:
    torch.cuda.OutOfMemoryError only exists from torch 2.x, and on older builds
    the identical condition arrives as a plain RuntimeError. Both say "out of
    memory", and Colab hands out whatever torch its image happens to carry.
    """
    return "out of memory" in str(exc).lower()


def shrink_batch(batch_size: int, grad_accum: int) -> tuple[int, int] | None:
    """Halve the micro-batch, double the accumulation.

    The doubling is the point. Halving the batch alone would quietly change the
    effective batch size, and therefore the training dynamics, so a run that
    survived an OOM would no longer be the run the config described — a silent
    substitution of one experiment for another. Doubling accumulation keeps
    batch_size * grad_accum identical: same maths, more steps, less memory.

    Returns None when the micro-batch is already 1 and there is nothing left
    to give.
    """
    if batch_size <= 1:
        return None
    effective = batch_size * grad_accum
    new_batch = batch_size // 2
    # Derived from the effective batch rather than just doubling. Doubling is
    # exact only when the batch size is a power of two: it would take 5x2 to
    # 2x4, quietly dropping the effective batch from 10 to 8 — the silent
    # substitution this function exists to avoid. Deriving keeps it at 2x5.
    # An odd effective batch (7x1) still cannot be preserved exactly; the
    # caller logs what was actually achieved rather than what was intended.
    return new_batch, max(1, round(effective / new_batch))


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
    import torch  # noqa: PLC0415
    from transformers import TrainerCallback  # noqa: PLC0415

    class HaltOnPoisonedLoss(TrainerCallback):
        """Stop the moment the loss stops being a finite number.

        A NaN does not raise and does not halt training. It flows into every
        subsequent weight update while the progress bar advances normally, and
        the run ends by saving an adapter that loads without complaint and
        produces garbage. In a one-hour lesson that is the worst available
        failure: forty minutes of Colab spent, a plausible-looking artifact,
        and nothing anywhere recording when it broke.

        The Rung 2 spec names "a bad bake is undiagnosable" as an open risk.
        This is one concrete cause of it, closed by refusing to continue and
        naming the step.
        """

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            value = (logs or {}).get("loss")
            if value is None or math.isfinite(value):
                return
            control.should_training_stop = True
            raise RuntimeError(
                f"loss became {value} at step {state.global_step} — aborting the bake.\n"
                "Training did not crash; it was stopped. Left alone it would have run to "
                "completion and saved an adapter that merges, converts and loads fine while "
                "generating noise.\n"
                "Usual cause is a learning rate too high for this batch size. Halve "
                f"train.learning_rate (currently {tr['learning_rate']}) and run again."
            )

    # Colab hands out whatever GPU it feels like — T4, L4, A100 — and a batch
    # size tuned on one OOMs on another. Rather than lose the session, halve the
    # micro-batch and double accumulation until it fits: the effective batch
    # size, and therefore the result, stays exactly what the config asked for.
    # A dropped session leaves no time to retry inside a one-hour lesson, so a
    # slower run that finishes beats a fast one that dies.
    batch_size, grad_accum = tr["batch_size"], tr["grad_accum"]
    effective = batch_size * grad_accum

    def build_trainer(bs: int, ga: int):
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=s["max_seq_length"],
            args=SFTConfig(
                per_device_train_batch_size=bs,
                gradient_accumulation_steps=ga,
                warmup_steps=tr["warmup_steps"],
                num_train_epochs=tr["epochs"],
                learning_rate=tr["learning_rate"],
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                # Explicit rather than relying on the framework default. Clipping
                # does not repair a NaN that has already happened — nothing does.
                # It prevents the enormous update that usually causes one.
                max_grad_norm=1.0,
                lr_scheduler_type="linear",
                seed=tr["seed"],
                output_dir=tr["output_dir"],
                report_to="none",
            ),
            callbacks=[HaltOnPoisonedLoss()],
        )

    while True:
        logger.info("Training with micro-batch %d x grad-accum %d (effective batch %d)",
                    batch_size, grad_accum, batch_size * grad_accum)
        trainer = build_trainer(batch_size, grad_accum)
        try:
            trainer.train()
            break
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            smaller = shrink_batch(batch_size, grad_accum)
            if smaller is None:
                logger.error(
                    "Out of memory at a micro-batch of 1 — there is nothing left to halve. "
                    "This GPU cannot hold %s at %d tokens. Try a smaller base model or a "
                    "shorter max_seq_length.", s["base_model"], s["max_seq_length"])
                raise
            batch_size, grad_accum = smaller
            achieved = batch_size * grad_accum
            if achieved == effective:
                logger.warning(
                    "Out of GPU memory. Retrying at micro-batch %d with grad-accum %d — "
                    "effective batch stays %d, so the result is unchanged; it will just "
                    "take more steps.", batch_size, grad_accum, effective)
            else:
                logger.warning(
                    "Out of GPU memory. Retrying at micro-batch %d with grad-accum %d. "
                    "Effective batch is now %d rather than %d — it could not be preserved "
                    "exactly at this size, so this is no longer quite the run the config "
                    "described.", batch_size, grad_accum, achieved, effective)
            # Release the old trainer's allocations before rebuilding, or the
            # retry OOMs on the memory the failed attempt is still holding.
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

    # --- 4. Save the adapter (the shippable "house-style cartridge") --------
    out = Path(tr["output_dir"]) / "adapter"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Saved LoRA adapter -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
