"""
documents_to_seeds.py  —  turn a folder of your own documents into task seeds.

WHAT THIS IS
------------
`generate_dataset.py` already makes training pairs; what it needs from you is a
list of coding tasks to make them *about*. `data_gen/seeds/task_seeds.txt` ships
68 generic ones. This script produces that file from documents you already have
— an internal handbook, an architecture note, an onboarding guide — so the
student practises on the shape of work your team actually does.

    documents/  ->  cleaned excerpts  ->  task seeds  ->  generate_dataset.py

Note where the pipeline is joined: this produces **seeds**, and the teacher
still writes every prompt/response pair downstream. It deliberately does not
produce tokens. Tokenisation happens at training time from text, and a folder of
tokens is not a trainable dataset — it is a dataset with the labels thrown away.

THE CONSTRAINT THAT SHAPES THE PROMPT
-------------------------------------
Seeds must stay **rule-agnostic**, and documents are full of rules. That tension
is the whole difficulty here.

The bake works by asymmetry: the teacher sees the house rules and writes
compliant code, and the stored student prompt has the rules stripped out, so the
student learns to obey without being told. If a seed says "write a logger that
follows our naming convention", the rule is back in the prompt — and the student
learns to recognise requests that talk about style rather than to apply style to
everything. That is the one failure mode that would quietly undo the project's
central trick, so the teacher is told not to produce such tasks and the output is
gated again afterwards in case it does anyway.

HONEST EXPECTATIONS
-------------------
Real documents are messy, and the keep rate here will be far below what the
curated seed list gets. That is the gates working, not the tool failing. Every
drop is counted by reason and written to an ingest sheet next to the output, for
the same reason the datasheet reports its keep rate: a seed file that does not
say how much was discarded reads as "the documents were full of good tasks"
regardless of whether they were.

USAGE
-----
    # needs the same local teacher endpoint generate_dataset.py uses
    python data_gen/documents_to_seeds.py --config configs/qwen_coder_0_5b_chromebook.yaml \\
                                          --docs ~/handbook

    # deterministic only: no teacher, no network, excerpts used as-is. Lower
    # quality, but it shows you exactly what the cleaner extracted.
    python data_gen/documents_to_seeds.py --config configs/... --docs ~/handbook --no-teacher
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import yaml
from openai import OpenAI

# Same directory, so this resolves when run as `python data_gen/documents_to_seeds.py`.
# Imported rather than copied: _chat owns the contract with the teacher endpoint
# (timeouts, failure returning None rather than raising) and there should be one
# of those, not two that drift.
from generate_dataset import _chat, _normalized_hash  # noqa: PLC2701

logger = logging.getLogger("rules_baker.docseeds")

DEFAULTS: dict[str, Any] = {
    "extensions": [".md", ".txt", ".rst", ".markdown"],
    "max_file_bytes": 2_000_000,
    "excerpt_min_words": 6,
    "excerpt_max_words": 120,
    "seed_min_words": 4,
    "seed_max_words": 30,
    "tasks_per_excerpt": 3,
    "max_seeds": 200,
}

FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t).*$", re.MULTILINE)
HTML_RE = re.compile(r"<[^>]+>")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MD_MARKS_RE = re.compile(r"[*_`#>]+")
LIST_MARK_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

# The verbs the shipped seed list opens with, plus the obvious neighbours. A seed
# that does not start with one of these is not phrased as a task — most often it
# is a sentence of prose the teacher echoed back. Matching on the opening word is
# crude, but it is inspectable and it is counted, which a cleverer classifier
# with the same error rate would not be.
TASK_VERBS = {
    "add", "build", "compute", "convert", "create", "decode", "detect", "encode",
    "extract", "filter", "find", "format", "generate", "group", "handle",
    "implement", "load", "merge", "normalize", "normalise", "parse", "process",
    "produce", "read", "remove", "render", "resolve", "retry", "return", "run",
    "sanitize", "sanitise", "serialize", "serialise", "sort", "split", "store",
    "stream", "transform", "truncate", "update", "validate", "write",
}

# Markers of a line that is quoting code or pointing into a document rather than
# describing a job: identifiers with call parens, CSS/attribute selectors, paths,
# URLs, braces. These survive the verb check easily — "Add render() below the
# import" opens with a fine verb — and they are the single most common thing a
# technical document is full of.
CODEISH_RE = re.compile(
    r"\(\)|[{}]|https?://|\S+\.(?:js|py|html|json|md|yaml|yml|css|ts)\b|(?:^|\s)[.#][A-Za-z][\w-]*"
)

# Words that mean the task has started talking about *style*, which is exactly
# what a seed must never do — see the note at the top of this file.
STYLE_WORDS = (
    "house style", "coding style", "style guide", "convention", "naming",
    "type hint", "docstring", "lint", "pep 8", "pep8", "formatting",
    "underscore", "camelcase", "snake_case", "our rules", "the rules",
    "code review", "best practice",
)


# --------------------------------------------------------------------------- #
#  Reading and cleaning
# --------------------------------------------------------------------------- #
def iter_documents(root: Path, extensions: list[str], max_bytes: int,
                   drops: Counter[str]) -> Iterator[tuple[Path, str]]:
    """Yield (path, raw text) for every readable document under ``root``."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stat().st_size > max_bytes:
            drops["file_too_large"] += 1
            logger.warning("skipping %s (%.1f MB, over the limit)",
                           path, path.stat().st_size / 1e6)
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            drops["file_not_text"] += 1
            continue
        yield path, raw.decode("utf-8", errors="replace")


def clean_text(raw: str) -> str:
    """Strip the markup and the code, leaving the prose.

    Code blocks go first and deliberately. They are examples of the answer, not
    descriptions of the work, and feeding them in as tasks would ask the teacher
    to write the code that is already sitting in front of it.
    """
    text = FRONT_MATTER_RE.sub("", raw)
    text = FENCE_RE.sub("\n", text)
    text = INDENTED_CODE_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)      # keep the link text, drop the URL
    text = HTML_RE.sub(" ", text)
    return text


def split_units(text: str) -> list[str]:
    """Split cleaned text into excerpt-sized units, one per block.

    Blocks, not sentences: a heading plus the paragraph under it says more than
    either alone, and an excerpt is only ever read by the teacher, which handles
    a little surrounding context better than a sentence with its subject removed.
    """
    units, current = [], []
    for line in text.splitlines():
        if TABLE_ROW_RE.match(line):
            continue                        # tables are reference data, not work
        stripped = LIST_MARK_RE.sub("", line).strip()
        stripped = MD_MARKS_RE.sub("", stripped).strip()
        if not stripped:
            if current:
                units.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        units.append(" ".join(current))
    return [re.sub(r"\s+", " ", u).strip() for u in units if u.strip()]


def gate_excerpt(unit: str, cfg: dict[str, Any]) -> str | None:
    """Rejection reason for an excerpt, or None to keep it."""
    words = unit.split()
    if len(words) < cfg["excerpt_min_words"]:
        return "excerpt_too_short"
    if len(words) > cfg["excerpt_max_words"]:
        return "excerpt_too_long"
    # A block with almost no letters is a badge row, a divider, or a path list.
    letters = sum(c.isalpha() for c in unit)
    if letters < len(unit) * 0.5:
        return "excerpt_not_prose"
    return None


# --------------------------------------------------------------------------- #
#  Excerpt -> tasks, via the teacher
# --------------------------------------------------------------------------- #
def teacher_tasks(client: OpenAI, cfg: dict[str, Any], excerpt: str, n: int) -> list[str]:
    """Ask the local teacher for ``n`` coding tasks grounded in one excerpt."""
    system = (
        "You turn a team's internal documentation into short practice coding tasks.\n\n"
        f"Given an excerpt, write exactly {n} Python coding tasks that a developer on "
        "that team might plausibly be given. Requirements:\n"
        "- One task per line. No numbering, no bullets, no explanation, no blank lines.\n"
        "- Each starts with an imperative verb: Write, Implement, Create, Parse, "
        "Validate, Build, Compute, Convert...\n"
        "- Each must stand alone. Someone who has never read the excerpt must be able "
        "to do it without asking what it refers to.\n"
        "- NEVER mention coding style, conventions, naming, formatting, type hints, "
        "docstrings, linting, or the document itself. Describe only the WORK. This is "
        "the important one: these tasks are used to teach a model to apply a house "
        "style unprompted, and a task that mentions style teaches it to look for the "
        "mention instead.\n"
        "- Keep each under 25 words."
    )
    resp = _chat(client, cfg, system, f"Excerpt:\n{excerpt}")
    if resp is None:
        return []
    return [ln.strip() for ln in resp.splitlines() if ln.strip()]


def gate_seed(seed: str, cfg: dict[str, Any], seen: set[str]) -> tuple[str | None, str]:
    """Return (rejection reason or None, cleaned seed)."""
    s = LIST_MARK_RE.sub("", seed).strip()
    s = MD_MARKS_RE.sub("", s).strip().strip('"')
    if not s:
        return "seed_empty", s
    words = s.split()
    if len(words) < cfg["seed_min_words"]:
        return "seed_too_short", s
    if len(words) > cfg["seed_max_words"]:
        return "seed_too_long", s
    if words[0].lower().rstrip(":,") not in TASK_VERBS:
        return "seed_not_a_task", s
    if s.endswith(":"):
        # "Add this beside the others:" — a sentence introducing a code block it
        # has been separated from. On its own it asks for nothing.
        return "seed_is_a_fragment", s
    if CODEISH_RE.search(s):
        return "seed_quotes_code", s
    low = s.lower()
    if any(w in low for w in STYLE_WORDS):
        # Not a formatting quibble. See the module docstring: a seed that names
        # the style teaches the student to wait to be told.
        return "seed_mentions_style", s
    h = _normalized_hash(s)
    if h in seen:
        return "seed_duplicate", s
    seen.add(h)
    return None, s


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def build_seeds(cfg: dict[str, Any], docs: Path, opts: dict[str, Any],
                client: OpenAI | None) -> tuple[list[str], Counter[str], list[str]]:
    """Walk the documents and return (seeds, drop counts, source file names)."""
    drops: Counter[str] = Counter()
    seen: set[str] = set()
    seeds: list[str] = []
    sources: list[str] = []

    for path, raw in iter_documents(docs, opts["extensions"], opts["max_file_bytes"], drops):
        sources.append(str(path.relative_to(docs)))
        units = split_units(clean_text(raw))
        kept_here = 0
        for unit in units:
            if len(seeds) >= opts["max_seeds"]:
                break
            reason = gate_excerpt(unit, opts)
            if reason:
                drops[reason] += 1
                continue
            if client is None:
                # Deterministic mode: the excerpt IS the candidate seed. It will
                # usually fail seed_not_a_task, which is the honest outcome —
                # prose is not a task list, and pretending otherwise would put
                # sentences into the pipeline dressed as work.
                candidates = [unit]
            else:
                candidates = teacher_tasks(client, cfg, unit, opts["tasks_per_excerpt"])
                if not candidates:
                    drops["teacher_null"] += 1
                    continue
            for cand in candidates:
                reason, cleaned = gate_seed(cand, opts, seen)
                if reason:
                    drops[reason] += 1
                    continue
                seeds.append(cleaned)
                kept_here += 1
        logger.info("%-50s %d excerpts -> %d seeds (%d total)",
                    str(path.relative_to(docs))[:50], len(units), kept_here, len(seeds))
        if len(seeds) >= opts["max_seeds"]:
            logger.warning("Hit --max-seeds (%d); stopping here. %d document(s) were read.",
                           opts["max_seeds"], len(sources))
            break
    return seeds, drops, sources


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Turn a folder of documents into task seeds.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--docs", required=True, type=Path, help="Folder of documents to ingest.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Seed file to write. Default: data_gen/seeds/from_documents.txt")
    ap.add_argument("--no-teacher", action="store_true",
                    help="Skip the teacher; emit cleaned excerpts as candidates.")
    ap.add_argument("--max-seeds", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]

    opts = dict(DEFAULTS)
    opts.update(cfg.get("documents") or {})
    if args.max_seeds is not None:
        opts["max_seeds"] = args.max_seeds
    opts["extensions"] = [e.lower() if e.startswith(".") else "." + e.lower()
                          for e in opts["extensions"]]

    if not args.docs.is_dir():
        logger.error("--docs %s is not a directory", args.docs)
        return 1

    client = None
    if not args.no_teacher:
        t = cfg["teacher"]
        client = OpenAI(base_url=t["base_url"], api_key=t["api_key"])
        logger.info("Teacher: %s at %s", t["model"], t["base_url"])
    else:
        logger.warning("Running without the teacher — excerpts are used as candidate "
                       "seeds unchanged, and most will be dropped as not-a-task.")

    seeds, drops, sources = build_seeds(cfg, args.docs, opts, client)

    out_path = args.out or (root / "data_gen" / "seeds" / "from_documents.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dropped = sum(drops.values())
    out_path.write_text(
        "# Task seeds generated from documents by data_gen/documents_to_seeds.py.\n"
        f"# Source folder: {args.docs}\n"
        f"# {len(sources)} document(s) read, {len(seeds)} seeds kept, {dropped} candidates dropped.\n"
        "# Read them before you bake on them. They are only as good as the documents,\n"
        "# and the gates cannot tell a dull task from an interesting one.\n"
        "# One coding task per line. Blank lines and #-comments are ignored.\n\n"
        + "\n".join(seeds) + "\n",
        encoding="utf-8",
    )

    # ---- ingest sheet ----------------------------------------------------
    # The same argument as the dataset's datasheet: a seed file that does not
    # record what it discarded reads as if the documents were full of good tasks.
    # Written beside the seeds so the two travel together.
    sheet_path = out_path.with_name(out_path.stem + ".ingestsheet.json")
    sheet_path.write_text(json.dumps({
        "schema": "pi-of-ai/ingestsheet/1",
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": {
            "folder": str(args.docs),
            "documentsRead": len(sources),
            "files": sources,
            "extensions": opts["extensions"],
        },
        "seeds": {
            "path": out_path.name,
            "kept": len(seeds),
            "dropped": dropped,
            "keepRatePct": round(100 * len(seeds) / (len(seeds) + dropped), 1)
            if len(seeds) + dropped else None,
            "dropReasons": dict(drops) or {},
        },
        "method": {
            "teacher": None if client is None else cfg["teacher"].get("model"),
            "note": (
                "Documents were cleaned of markup and code, split into blocks, and each "
                "surviving block was turned into coding tasks by the local teacher. Seeds "
                "are deliberately rule-agnostic: any candidate naming style, conventions "
                "or formatting was dropped, because a seed that mentions the house style "
                "teaches the student to wait to be told about it."
            ),
            "emits": "task seeds (text), not tokens",
        },
        "unverifiable": [
            "whether the tasks are representative of the documents they came from",
            "whether the teacher invented detail the documents do not support",
        ],
    }, indent=2) + "\n", encoding="utf-8")

    logger.info("=" * 60)
    logger.info("KEPT %d seeds from %d document(s)", len(seeds), len(sources))
    logger.info("DROPPED %d -> %s", dropped, dict(drops) or "none")
    if len(seeds) + dropped:
        logger.info("keep-rate %.1f%%", 100 * len(seeds) / (len(seeds) + dropped))
    logger.info("seeds  -> %s", out_path)
    logger.info("sheet  -> %s", sheet_path)
    logger.info("Point seeds_file at it in your config, then run generate_dataset.py.")
    logger.info("=" * 60)

    if not seeds:
        if client is None:
            logger.error(
                "No seeds kept. Expected without --no-teacher on most real documents: "
                "prose describes work, it is not phrased as work. Drop the flag and let "
                "the teacher turn these %d excerpts into tasks.", sum(drops.values()))
        else:
            logger.error("No seeds kept. Either the folder has no prose documents, or the "
                         "teacher endpoint is down at %s.", cfg["teacher"]["base_url"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
