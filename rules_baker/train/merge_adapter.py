"""
merge_adapter.py  —  merge a saved LoRA adapter back into the base weights.

Separate from bake_gpu.py so a successful train isn't lost if the merge/save step
fails (e.g. out of disk). Loads the base + the saved adapter, merges, and writes an
fp16 model ready for GGUF conversion.

    D:\\rbgpu\\Scripts\\python.exe train/merge_adapter.py --out <dir>
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/qwen_coder_0_5b_chromebook.yaml"))
    ap.add_argument("--out", type=Path, required=True, help="where to write the merged model")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    base = cfg["student"]["base_model"]
    adapter = root / "outputs" / (Path(cfg["train"]["output_dir"]).name + "_adapter")

    print(f"base:    {base}")
    print(f"adapter: {adapter}")
    print(f"out:     {args.out}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter))
    merged = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    tok.save_pretrained(str(args.out))
    print(f"\nmerged fp16 model written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
