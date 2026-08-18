"""
generate_dataset.py  —  Rules-Baker synthetic data generator (Gap 2), hardened.

Core idea (the whole trick):
    The TEACHER model sees your rules and writes rule-compliant code.
    The STUDENT training example has the rules STRIPPED OUT of the prompt.
    -> the student learns to obey your house style *silently*, so at inference
       time your IDE context window stays clean and fast. No rule-bloat, no lag.

Two kinds of examples are produced:
    positive  : ordinary coding request  -> compliant code (rules never shown)
    revision  : rule-violating code       -> corrected code (teaches self-repair)

HARDENING (this version):
    * ast.parse gate      — any generation that isn't valid Python is dropped/retried.
    * min-length gate      — trivially short/empty outputs are dropped.
    * dedup                — near-duplicate code (normalized-whitespace hash) is dropped.
    * retry loop           — rejected examples are re-asked up to max_retries times.
    * leak-proof split     — SEEDS are split into train/eval; we generate ONLY from the
                             train seeds and write the eval seeds out for eval_rules.py,
                             so you never test on a task you trained on.
    * drop accounting      — every rejection is counted and logged. Nothing is dropped
                             silently (a silent cap reads as "clean data" when it isn't).

Everything runs against a LOCAL OpenAI-compatible endpoint (Ollama / vLLM /
llama.cpp). Nothing leaves the room.

Usage:
    py data_gen/generate_dataset.py --config configs/qwen_coder_7b.yaml
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

logger = logging.getLogger("rules_baker.datagen")

RULE_HEADER_RE = re.compile(r"^##\s+RULE\s+(?P<id>[\w-]+)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------- #
#  Inputs
# --------------------------------------------------------------------------- #
def parse_rules(rules_path: Path) -> list[dict[str, str]]:
    """Parse a rules markdown file into a list of atomic {id, text} rules.

    Args:
        rules_path: Path to the markdown rules file.

    Returns:
        A list of dicts with keys ``id`` and ``text``.

    Raises:
        ValueError: If no `## RULE` blocks are found.
    """
    raw = rules_path.read_text(encoding="utf-8")
    matches = list(RULE_HEADER_RE.finditer(raw))
    if not matches:
        raise ValueError(f"No '## RULE <id>' blocks found in {rules_path}")

    rules: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        rules.append({"id": m.group("id"), "text": raw[start:end].strip()})
    logger.info("Parsed %d rules from %s", len(rules), rules_path.name)
    return rules


def load_seeds(seeds_path: Path) -> list[str]:
    """Load coding-task seeds, skipping blanks and #-comments."""
    lines = seeds_path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def split_seeds(
    seeds: list[str], holdout_frac: float, rng: random.Random
) -> tuple[list[str], list[str]]:
    """Deterministically split seeds into (train, eval_holdout).

    The eval holdout is what eval_rules.py scores on, so training never sees it.
    """
    shuffled = seeds[:]
    rng.shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * holdout_frac)) if holdout_frac > 0 else 0
    eval_seeds = shuffled[:n_eval]
    train_seeds = shuffled[n_eval:]
    if not train_seeds:  # tiny seed lists: never leave training empty
        train_seeds, eval_seeds = shuffled, []
    return train_seeds, eval_seeds


# --------------------------------------------------------------------------- #
#  Quality gates
# --------------------------------------------------------------------------- #
def _extract_code(text: str) -> str:
    """Pull the first fenced code block out of a response, else return raw text."""
    m = FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _normalized_hash(code: str) -> str:
    """Whitespace-insensitive hash for dedup (collapses runs of blank/space)."""
    norm = re.sub(r"\s+", " ", code).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def validate_code(code: str, q: dict[str, Any]) -> str | None:
    """Return a rejection reason string if the code fails a quality gate, else None."""
    if not code or len(code) < q["min_code_chars"]:
        return "too_short"
    if q["require_parse"]:
        try:
            ast.parse(code)
        except SyntaxError:
            return "parse_error"
    return None


# --------------------------------------------------------------------------- #
#  Teacher calls
# --------------------------------------------------------------------------- #
def _render_rules_block(rules: list[dict[str, str]]) -> str:
    return "\n".join(f"- ({r['id']}) {r['text']}" for r in rules)


def _chat(client: OpenAI, cfg: dict[str, Any], system: str, user: str) -> str | None:
    """Single chat completion against the local teacher endpoint."""
    t = cfg["teacher"]
    try:
        out = client.chat.completions.create(
            model=t["model"],
            temperature=t["temperature"],
            max_tokens=t["max_tokens"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return out.choices[0].message.content
    except Exception:  # noqa: BLE001 — surface, don't crash the whole run
        logger.exception("Teacher call failed")
        return None


def teacher_positive(
    client: OpenAI, cfg: dict[str, Any], task: str, rules: list[dict[str, str]]
) -> dict[str, str] | None:
    """POSITIVE example: bare task -> compliant code (rules hidden from student)."""
    system = (
        "You are a senior engineer. Write Python code that STRICTLY follows every "
        "one of the following internal house-style rules. Do NOT mention the rules "
        "or explain them — just produce clean code that silently obeys them.\n\n"
        f"HOUSE RULES:\n{_render_rules_block(rules)}\n\n"
        "Respond with exactly one fenced ```python code block and nothing else."
    )
    resp = _chat(client, cfg, system, task)
    if resp is None:
        return None
    code = _extract_code(resp)
    return {
        "kind": "positive",
        "rule_ids": ",".join(r["id"] for r in rules),
        "user": task,                    # BARE task — rules absent. That is the point.
        "code": code,
        "assistant": f"```python\n{code}\n```",
    }


def teacher_revision(
    client: OpenAI, cfg: dict[str, Any], task: str, rules: list[dict[str, str]], q: dict[str, Any]
) -> dict[str, str] | None:
    """REVISION example: rule-violating code -> corrected code (teaches self-repair)."""
    rules_block = _render_rules_block(rules)
    bad_system = (
        "You are generating training data. Write Python code for the task below that "
        "VIOLATES the following rules (naturally, as a careless junior might). "
        f"Rules to violate:\n{rules_block}\n\n"
        "Respond with exactly one fenced ```python code block and nothing else."
    )
    bad = _chat(client, cfg, bad_system, task)
    if bad is None:
        return None
    bad_code = _extract_code(bad)
    # The "before" snippet must itself be coherent code, or the fix example is junk.
    if validate_code(bad_code, q) is not None:
        return None

    fix_system = (
        "You are a senior engineer enforcing internal house-style rules. Rewrite the "
        "user's code so it fully complies with EVERY rule below. Keep behavior "
        "identical. Do not explain — return only the corrected code.\n\n"
        f"HOUSE RULES:\n{rules_block}\n\n"
        "Respond with exactly one fenced ```python code block and nothing else."
    )
    fix_user = f"Task was: {task}\n\nFix this code:\n```python\n{bad_code}\n```"
    good = _chat(client, cfg, fix_system, fix_user)
    if good is None:
        return None
    good_code = _extract_code(good)
    return {
        "kind": "revision",
        "rule_ids": ",".join(r["id"] for r in rules),
        "user": f"Review and fix this code to match our house style:\n```python\n{bad_code}\n```",
        "code": good_code,
        "assistant": f"```python\n{good_code}\n```",
    }


# --------------------------------------------------------------------------- #
#  Orchestration with retries + dedup + accounting
# --------------------------------------------------------------------------- #
def _attempt(
    kind: str, client: OpenAI, cfg: dict[str, Any], task: str,
    rules: list[dict[str, str]], q: dict[str, Any], seen: set[str], drops: Counter[str],
) -> dict[str, str] | None:
    """One retry loop for a single desired example. Returns a kept record or None."""
    for _ in range(1 + q["max_retries"]):
        rec = (
            teacher_positive(client, cfg, task, rules)
            if kind == "positive"
            else teacher_revision(client, cfg, task, rules, q)
        )
        if rec is None:
            drops["teacher_null"] += 1
            continue
        reason = validate_code(rec["code"], q)
        if reason is not None:
            drops[reason] += 1
            continue
        if q["dedup"]:
            h = _normalized_hash(rec["code"])
            if h in seen:
                drops["duplicate"] += 1
                continue
            seen.add(h)
        rec.pop("code", None)  # strip the scratch field before storing
        return rec
    return None


def _existing_hashes(out_path: Path) -> tuple[set[str], int]:
    """Rebuild the dedup set + count from an existing JSONL so resumes don't duplicate."""
    seen, n = set(), 0
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asst = rec["messages"][1]["content"]
            m = FENCE_RE.search(asst)
            seen.add(_normalized_hash((m.group(1) if m else asst).strip()))
            n += 1
    return seen, n


def build_dataset(cfg: dict[str, Any], out_path: Path, progress_path: Path):
    """Generate from TRAIN seeds, writing each kept example to disk IMMEDIATELY.

    Fully resumable: completed seeds are recorded in ``progress_path`` and skipped
    on restart, and the dedup set is rebuilt from the existing JSONL. A teardown
    mid-run loses at most the seed in flight, never the whole dataset.
    """
    root = Path(__file__).resolve().parents[1]
    rules = parse_rules(root / cfg["rules_file"])
    all_seeds = load_seeds(root / cfg["seeds_file"])

    t, q = cfg["teacher"], cfg["quality"]
    rng = random.Random(cfg["train"]["seed"])
    train_seeds, eval_seeds = split_seeds(all_seeds, q["eval_holdout_frac"], rng)

    done = set(progress_path.read_text(encoding="utf-8").splitlines()) if progress_path.exists() else set()
    seen, kept = _existing_hashes(out_path)
    logger.info("Seeds: %d total -> %d train / %d eval | resuming: %d seeds done, %d examples on disk",
                len(all_seeds), len(train_seeds), len(eval_seeds), len(done), kept)

    lo, hi = t["rules_per_example"]
    drops: Counter[str] = Counter()
    out_f = out_path.open("a", encoding="utf-8")
    prog_f = progress_path.open("a", encoding="utf-8")
    try:
        for task in train_seeds:
            if task in done:
                continue
            for kind, cnt in (("positive", t["n_positive_per_seed"]),
                              ("revision", t["n_revision_per_seed"])):
                for _ in range(cnt):
                    k = rng.randint(lo, min(hi, len(rules)))
                    rec = _attempt(kind, client=OPENAI_CLIENT, cfg=cfg, task=task,
                                  rules=rng.sample(rules, k), q=q, seen=seen, drops=drops)
                    if rec:
                        out_f.write(json.dumps(to_sft_messages(rec), ensure_ascii=False) + "\n")
                        out_f.flush()
                        kept += 1
            prog_f.write(task + "\n"); prog_f.flush()
            logger.info("task done: %-50s kept=%d dropped=%d", task[:50], kept, sum(drops.values()))
    finally:
        out_f.close(); prog_f.close()
    return kept, eval_seeds, drops


def to_sft_messages(rec: dict[str, str]) -> dict[str, Any]:
    """Convert an internal record into the chat-messages schema Unsloth expects."""
    return {
        "messages": [
            {"role": "user", "content": rec["user"]},
            {"role": "assistant", "content": rec["assistant"]},
        ],
        "meta": {"kind": rec["kind"], "rule_ids": rec["rule_ids"]},
    }


# The teacher client is a module-level singleton so _attempt stays call-site clean.
OPENAI_CLIENT: OpenAI = None  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    global OPENAI_CLIENT
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Generate a hardened rules-baked SFT dataset.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    t = cfg["teacher"]
    OPENAI_CLIENT = OpenAI(base_url=t["base_url"], api_key=t["api_key"])

    out_path = root / cfg["output_jsonl"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_name(out_path.stem + ".progress.txt")

    kept, eval_seeds, drops = build_dataset(cfg, out_path, progress_path)

    # Write the leak-proof eval holdout for eval_rules.py (idempotent).
    eval_path = out_path.parent / "eval_seeds.txt"
    eval_path.write_text(
        "# Held-out eval seeds — NOT used in training. Auto-generated.\n"
        + "\n".join(eval_seeds) + "\n",
        encoding="utf-8",
    )

    dropped = sum(drops.values())

    # ---- datasheet -------------------------------------------------------
    # A trained model's weights are auditable only as far as its data is, and
    # synthetic data is the case where nobody can reconstruct provenance after
    # the fact: the teacher gets upgraded, the rules get edited, the seeds get
    # extended, and the dataset on disk no longer explains itself. Written here
    # because this is the only moment all of it is known at once.
    #
    # Every drop is reported. A keep-rate is the difference between "the teacher
    # produced clean data" and "we discarded most of what it produced", and a
    # dataset that omits it reads as the former regardless of which it was.
    datasheet_path = out_path.with_name(out_path.stem + ".datasheet.json")
    datasheet = {
        "schema": "pi-of-ai/datasheet/1",
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dataset": {
            "path": out_path.name,
            "examplesKept": kept,
            "examplesDropped": dropped,
            "keepRatePct": round(100 * kept / (kept + dropped), 1) if kept + dropped else None,
            "dropReasons": dict(drops) or {},
        },
        "generation": {
            "method": "synthetic, teacher-student with rule stripping",
            "note": (
                "The teacher saw the house rules and wrote compliant code. The stored "
                "student prompt has the rules removed, so the student learns to obey "
                "without being told. Training prompts therefore do NOT contain the rules."
            ),
            "teacherModel": t.get("model"),
            "teacherEndpoint": t.get("base_url"),
            "temperature": cfg.get("generation", {}).get("temperature"),
            "maxTokens": cfg.get("generation", {}).get("max_tokens"),
            "config": args.config.name,
        },
        "rules": {
            "source": cfg.get("rules_file"),
            "count": len(parse_rules(root / cfg["rules_file"])) if cfg.get("rules_file") else None,
        },
        "splits": {
            "evalSeedsHeldOut": len(eval_seeds),
            "leakProof": True,
            "note": "Generation used train seeds only; eval seeds were never generated from.",
        },
        "provenance": {
            "humanAuthored": False,
            "licence": (
                "Inherits the teacher model's licence and acceptable-use terms. "
                "Synthetic output does not clear the terms of the model that produced it — "
                "check before redistributing this dataset or anything trained on it."
            ),
            "containsPersonalData": False,
            "unverifiable": [
                "what the teacher model was itself trained on",
                "whether teacher outputs reproduce memorised training data",
            ],
        },
    }
    datasheet_path.write_text(json.dumps(datasheet, indent=2) + "\n", encoding="utf-8")

    logger.info("=" * 60)
    logger.info("KEPT %d examples total on disk", kept)
    logger.info("DROPPED %d  -> %s", dropped, dict(drops) or "none")
    if kept + dropped:
        logger.info("keep-rate %.1f%%", 100 * kept / (kept + dropped))
    logger.info("dataset  -> %s (incremental, resumable)", out_path)
    logger.info("eval set -> %s (%d held-out seeds)", eval_path, len(eval_seeds))
    logger.info("=" * 60)

    if kept == 0:
        logger.error("No examples kept. Is the teacher endpoint up at %s ?", t["base_url"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
