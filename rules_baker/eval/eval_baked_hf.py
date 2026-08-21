"""
eval_baked_hf.py  —  score a merged HF model on the held-out seeds, on GPU.

Same test as eval_rules.py (bare task, NO rules in the prompt, same deterministic
checkers) but generates locally with transformers instead of hitting Ollama — so we
can score the freshly-baked model without a GGUF conversion first. Reuses the
CHECKERS from eval_rules.py so the "before" (stock) and "after" (baked) numbers are
directly comparable.

    D:\\rbgpu\\Scripts\\python.exe eval/eval_baked_hf.py [model_dir]
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_rules import CHECKERS, _extract_code  # noqa: E402  (reuse the exact checkers)

SYSTEM = "You are a helpful senior Python engineer."


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/qwen_coder_0_5b_chromebook.yaml").read_text(encoding="utf-8"))
    model_dir = sys.argv[1] if len(sys.argv) > 1 else str(root / cfg["train"]["output_dir"] / "merged")

    seeds_file = root / "datasets" / "eval_seeds.txt"
    seeds = [
        ln.strip() for ln in seeds_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    print(f"Model: {model_dir}\nEvaluating on {len(seeds)} held-out seeds\n")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16).to("cuda").eval()

    def generate(task: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return _extract_code(tok.decode(gen, skip_special_tokens=True))

    passed: Counter[str] = Counter()
    checked: Counter[str] = Counter()
    for i, task in enumerate(seeds, 1):
        code = generate(task)
        for rid, fn in CHECKERS.items():
            checked[rid] += 1
            if fn(code):
                passed[rid] += 1
        print(f"  [{i}/{len(seeds)}] done")

    print(f"\n=== Rule compliance for BAKED model over {len(seeds)} tasks ===")
    tp = tc = 0
    for rid in CHECKERS:
        p, c = passed[rid], checked[rid]
        tp += p
        tc += c
        print(f"  {rid:32s} {p:3d}/{c:<3d}  {(100*p/c if c else 0):5.1f}%")
    print(f"  {'OVERALL':32s} {tp:3d}/{tc:<3d}  {(100*tp/tc if tc else 0):5.1f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
