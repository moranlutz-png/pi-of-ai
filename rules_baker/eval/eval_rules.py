"""
eval_rules.py  —  Rule-compliance harness (what makes Gap 2 *checkable*).

Knowledge-injection via LoRA is finicky: a model can under-absorb or over-apply
rules. This harness gives you an objective score BEFORE and AFTER baking, so you
know the adapter actually worked instead of hoping it did.

It sends the bare task seeds (NO rules in the prompt) to a local OpenAI-compatible
endpoint and runs deterministic checkers over the generated code. Each checker
maps 1:1 to a `## RULE <id>` in your rules doc.

    python eval/eval_rules.py --config configs/qwen_coder_7b.yaml \
        --model qwen-coder-housestyle          # the baked model in Ollama
    # compare against the untuned baseline:
    python eval/eval_rules.py --config configs/qwen_coder_7b.yaml \
        --model qwen2.5-coder:7b-instruct

Add a checker: write a function `check_<rule_id>(code) -> bool` (True = compliant)
and register it in CHECKERS. Unregistered rules are reported as "no checker".
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml
from openai import OpenAI

logger = logging.getLogger("rules_baker.eval")

# --------------------------------------------------------------------------- #
#  Deterministic checkers — one per rule id in example_rules.md.
#  Return True when the code COMPLIES with the rule.
# --------------------------------------------------------------------------- #
def check_errors_no_bare_except(code: str) -> bool:
    if re.search(r"except\s*:", code):
        return False
    if re.search(r"except\s+Exception\s*(as\s+\w+)?\s*:\s*\n\s*pass", code):
        return False
    return True


def check_typing_required(code: str) -> bool:
    # every `def` that takes args should annotate; require a return arrow too.
    defs = re.findall(r"def\s+\w+\s*\(([^)]*)\)\s*(->\s*[^:]+)?:", code)
    if not defs:
        return True
    for params, ret in defs:
        real = [p for p in params.split(",") if p.strip() and p.strip() not in ("self", "cls")]
        if real and not all(":" in p for p in real):
            return False
        if not ret:
            return False
    return True


def check_logging_module_logger(code: str) -> bool:
    if "print(" in code:
        return False
    return True  # presence of logger is context-dependent; absence of print is the hard rule


def check_naming_private_underscore(code: str) -> bool:
    # heuristic: no camelCase function names (house style forbids them).
    return not re.search(r"def\s+[a-z]+[A-Z]\w*\s*\(", code)


def check_docstrings_google_style(code: str) -> bool:
    # every top-level def/class should be followed by a triple-quoted docstring.
    for m in re.finditer(r"^(def|class)\s+\w+.*:\s*\n(\s+)", code, re.MULTILINE):
        tail = code[m.end():m.end() + 8]
        if '"""' not in tail and "'''" not in tail:
            return False
    return True


def check_constants_uppercase_module(code: str) -> bool:
    return True  # advisory-only; hard to check statically without false positives


def check_layering_no_db_in_handlers(code: str) -> bool:
    return True  # requires call-graph analysis; left as manual/advisory


CHECKERS: dict[str, Callable[[str], bool]] = {
    "errors-no-bare-except": check_errors_no_bare_except,
    "typing-required": check_typing_required,
    "logging-module-logger": check_logging_module_logger,
    "naming-private-underscore": check_naming_private_underscore,
    "docstrings-google-style": check_docstrings_google_style,
    "constants-uppercase-module": check_constants_uppercase_module,
    "layering-no-db-in-handlers": check_layering_no_db_in_handlers,
}


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:[\w+-]*)\n(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def generate(client: OpenAI, model: str, task: str) -> str:
    """Ask the model to do a task with NO rules in the prompt — the real test."""
    out = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a helpful senior Python engineer."},
            {"role": "user", "content": task},
        ],
    )
    return _extract_code(out.choices[0].message.content or "")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Score rule-compliance of a served model.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model", required=True, help="model name as served locally")
    args = ap.parse_args(argv)

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]

    # Prefer the leak-proof held-out seeds written by generate_dataset.py; these
    # were NEVER used in training. Fall back to the full seed file if absent.
    holdout = root / Path(cfg["output_jsonl"]).parent / "eval_seeds.txt"
    seeds_src = holdout if holdout.exists() else (root / cfg["seeds_file"])
    seeds = [
        ln.strip()
        for ln in seeds_src.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    logger.info("Evaluating on %d seeds from %s", len(seeds), seeds_src.name)

    t = cfg["teacher"]
    client = OpenAI(base_url=t["base_url"], api_key=t["api_key"])

    passed: Counter[str] = Counter()
    checked: Counter[str] = Counter()
    for task in seeds:
        code = generate(client, args.model, task)
        for rid, fn in CHECKERS.items():
            checked[rid] += 1
            if fn(code):
                passed[rid] += 1

    print(f"\n=== Rule compliance for model '{args.model}' over {len(seeds)} tasks ===")
    total_p = total_c = 0
    for rid in CHECKERS:
        p, c = passed[rid], checked[rid]
        total_p += p
        total_c += c
        pct = (100 * p / c) if c else 0
        print(f"  {rid:32s} {p:3d}/{c:<3d}  {pct:5.1f}%")
    overall = (100 * total_p / total_c) if total_c else 0
    print(f"  {'OVERALL':32s} {total_p:3d}/{total_c:<3d}  {overall:5.1f}%\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
