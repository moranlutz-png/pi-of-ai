"""
fork_tier.py — fork a Scratch-Coder tier into your own editable experiment.

The browser can't train (no CUDA in a tab), so this is the CLI side of "make a
copy of a tier and take it further". Two things you can do with a fork:

  * CONTINUE — copy a tier's trained weights and keep teaching it (train.py
    --resume picks up where it left off). Good for "train it further".
  * SCRATCH  — copy only the tier's architecture and the code, at random init,
    and train it yourself so you can watch it learn from nothing again.

Either way you get a self-contained folder under experiments/<name>/ with its
OWN copy of the model code (model.py, train.py, guards.py, ...). Edit the
architecture or the training loop there and it changes only your fork — the
tiers stay put. The fork reads a shared corpus (--data) but writes its
checkpoint into its own folder (--out-dir .), so it never clobbers a tier.

Examples:
  python fork_tier.py --list
  python fork_tier.py --tier ultra --name my-ultra              # continue (default)
  python fork_tier.py --tier chromebook --name from-zero --scratch
  python fork_tier.py --blank --name tiny --arch 4x4x96 --block 128

A "continue" fork prefers the tier's real checkpoint (.pt, with optimizer
state) if it's on disk; if it isn't (e.g. a fresh clone, or a corpus dir that
was cleared), it rebuilds a weights-only checkpoint from the committed
weights.bin — so continuing works anywhere the repo does.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_TIERS = HERE / "web" / "tiers"
EXP = HERE / "experiments"

# Code the fork needs to train, sample, and export on its own. Copied (not
# referenced) so edits to the architecture live in the fork, not the shared tree.
CODE_FILES = ["model.py", "train.py", "guards.py", "arch_map.py",
              "sample_big.py", "export_inspect.py"]
CODE_DIRS = ["layers"]   # the pluggable-MLP registry model.py can load with --mlp

# Which corpus each tier trained on, and how to rebuild it if it's not on disk.
# (data_max / data_ultra are big and gitignored, so a clone won't have them.)
CORPUS = {
    "featherweight": {"dir": "data", "regen": "python prepare_data.py"},
    "chromebook":    {"dir": "data", "regen": "python prepare_data.py"},
    "laptop":        {"dir": "data", "regen": "python prepare_data.py"},
    "max":   {"dir": "data_max",
              "regen": "python prepare_data.py --out-dir data_max --max-bytes 80000000 --site-packages"},
    "ultra": {"dir": "data_ultra",
              "regen": "python prepare_data.py --out-dir data_ultra --max-bytes 128000000 --site-packages"},
}


def load_manifest() -> list:
    try:
        return json.loads((WEB_TIERS / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return []


def load_tier(tier_id: str) -> dict:
    """The tier's architecture and checkpoint meta, from its inspect.json."""
    p = WEB_TIERS / tier_id / "inspect.json"
    if not p.exists():
        raise SystemExit(f"no inspect.json for tier {tier_id!r} at {p}\n"
                         f"  (regenerate it with: python export_inspect.py --weights "
                         f"--ckpt <ckpt> --out-dir web/tiers/{tier_id} --label {tier_id.title()})")
    return json.loads(p.read_text(encoding="utf-8"))


def arch_of(inspect: dict) -> dict:
    c = inspect["config"]
    block = c.get("block_size", c.get("block"))
    return {"n_layer": c["n_layer"], "n_head": c["n_head"],
            "n_embd": c["n_embd"], "block": block, "vocab_size": c["vocab_size"]}


def suggested_lr(n_embd: int) -> str:
    # Bigger models train unstably at 3e-3 (the small-tier default) — the Ultra
    # build needed 1e-3 to keep the first layer's gradients from dwarfing the rest.
    return "1e-3" if n_embd >= 320 else "3e-3"


def cmd_lines(corpus_rel: str, tag: str, arch: dict, resume: bool, iters: int) -> str:
    lr = suggested_lr(arch["n_embd"])
    parts = [f"python train.py --data {corpus_rel} --out-dir . --tag {tag}",
             f"  --n-layer {arch['n_layer']} --n-head {arch['n_head']} "
             f"--n-embd {arch['n_embd']} --block {arch['block']}",
             f"  --lr {lr} --iters {iters}" + (" --resume" if resume else "")]
    return " \\\n".join(parts)


def rebuild_ckpt_from_weights(tier_id: str, inspect: dict, arch: dict, dest: Path) -> None:
    """Reconstruct a training checkpoint from the committed weights.bin, using the
    tensor manifest inspect.json already carries. No optimizer state (Adam just
    restarts), but the weights are exact — so --resume keeps teaching these weights."""
    import numpy as np
    import torch
    from model import GPT, GPTConfig

    wman = inspect.get("weights", {}).get("tensors")
    if not wman:
        raise SystemExit(f"tier {tier_id!r} has no weights manifest in inspect.json "
                         f"(re-export it with --weights) — can't rebuild a checkpoint")
    blob = np.fromfile(WEB_TIERS / tier_id / "weights.bin", dtype="<f4")
    cfg = GPTConfig(vocab_size=arch["vocab_size"], block_size=arch["block"],
                    n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"])
    sd = GPT(cfg).state_dict()
    for t in wman:
        name, off, cnt = t["name"], t["offset"], t["count"]
        sd[name] = torch.from_numpy(blob[off:off + cnt].reshape(sd[name].shape).copy())
    ck = inspect.get("checkpoint", {})
    torch.save({"model": sd, "cfg": cfg.__dict__,
                "iter": int(ck.get("iter") or 0), "best": float(ck.get("valLoss") or 1e9)}, dest)


def copy_code(dst: Path) -> None:
    for f in CODE_FILES:
        src = HERE / f
        if src.exists():
            shutil.copy2(src, dst / f)
    for d in CODE_DIRS:
        src = HERE / d
        if src.is_dir():
            shutil.copytree(src, dst / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def do_list() -> int:
    man = load_manifest()
    if not man:
        print("no tiers found under web/tiers/")
        return 0
    print("tiers you can fork:\n")
    for t in man:
        try:
            a = arch_of(load_tier(t["id"]))
            ashape = f"{a['n_layer']}L·{a['n_head']}H·{a['n_embd']}d·blk{a['block']}"
        except SystemExit:
            ashape = "(no inspect.json)"
        pt = corpus_ckpt_path(t["id"])
        has = "ckpt on disk" if pt and pt.exists() else "rebuild from weights.bin"
        print(f"  {t['id']:<14} {t.get('params', 0)/1e6:>6.2f}M  {ashape:<22} continue: {has}")
    print("\nfork one with:  python fork_tier.py --tier <id> --name <your-name> [--scratch]")
    return 0


def corpus_ckpt_path(tier_id: str):
    """Where the tier's real .pt would live, if it's still on disk."""
    c = CORPUS.get(tier_id)
    if not c:
        return None
    try:
        fname = load_tier(tier_id).get("checkpoint", {}).get("path")
    except SystemExit:
        return None
    return (HERE / c["dir"] / fname) if fname else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fork a Scratch-Coder tier into an editable experiment.")
    ap.add_argument("--list", action="store_true", help="list the tiers you can fork, then exit")
    ap.add_argument("--tier", help="tier id to fork (see --list)")
    ap.add_argument("--blank", action="store_true", help="no tier — just the base code at a chosen --arch")
    ap.add_argument("--name", help="name for the fork (folder under experiments/)")
    ap.add_argument("--scratch", action="store_true", help="train from random init (default is continue)")
    ap.add_argument("--arch", help="for --blank: LxHxE, e.g. 6x6x192 (layers x heads x embd)")
    ap.add_argument("--block", type=int, default=128, help="for --blank: context length (default 128)")
    ap.add_argument("--iters", type=int, help="starting --iters in the generated command")
    ap.add_argument("--force", action="store_true", help="overwrite an existing fork of the same name")
    args = ap.parse_args()

    if args.list:
        return do_list()
    if not args.name:
        raise SystemExit("--name is required (the fork's folder under experiments/)")
    if not args.tier and not args.blank:
        raise SystemExit("pick a source: --tier <id> (see --list) or --blank --arch LxHxE")

    dst = EXP / args.name
    if dst.exists():
        if not args.force:
            raise SystemExit(f"{dst} already exists — pass --force to overwrite it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # --- resolve architecture + corpus + mode ------------------------------------
    if args.blank:
        if not args.arch:
            raise SystemExit("--blank needs --arch LxHxE, e.g. --arch 4x4x96")
        L, H, E = (int(x) for x in args.arch.lower().split("x"))
        arch = {"n_layer": L, "n_head": H, "n_embd": E, "block": args.block, "vocab_size": None}
        corpus_dir, regen = "data", CORPUS["chromebook"]["regen"]
        mode, resume, source = "scratch", False, "blank"
        inspect = None
    else:
        inspect = load_tier(args.tier)
        arch = arch_of(inspect)
        corpus_dir = CORPUS.get(args.tier, {"dir": "data"})["dir"]
        regen = CORPUS.get(args.tier, {}).get("regen", "python prepare_data.py")
        mode = "scratch" if args.scratch else "continue"
        resume = not args.scratch
        source = args.tier

    corpus_rel = f"../../{corpus_dir}"
    corpus_exists = (HERE / corpus_dir).is_dir()
    iters = args.iters or (2000 if resume else 8000)
    tag = args.name

    # --- lay down the fork -------------------------------------------------------
    copy_code(dst)

    # A "continue" fork needs a starting checkpoint named ckpt_<tag>.pt in the fork,
    # so `train.py --resume` finds it. Prefer the real .pt (optimizer state and all);
    # fall back to rebuilding it from the committed weights.bin.
    ckpt_note = ""
    if resume:
        dest_ckpt = dst / f"ckpt_{tag}.pt"
        real = corpus_ckpt_path(args.tier)
        if real and real.exists():
            shutil.copy2(real, dest_ckpt)
            ckpt_note = f"copied {real.relative_to(HERE)} (with optimizer state)"
        else:
            rebuild_ckpt_from_weights(args.tier, inspect, arch, dest_ckpt)
            ckpt_note = "rebuilt from web/tiers/%s/weights.bin (weights only, no optimizer)" % args.tier

    # config.json — a machine-readable record of what this fork is.
    (dst / "config.json").write_text(json.dumps({
        "name": args.name, "source": source, "mode": mode, "arch": arch,
        "corpus": corpus_dir, "resume": resume, "start_iters": iters,
    }, indent=2), encoding="utf-8")

    # Ready-to-run commands for both shells.
    train_cmd = cmd_lines(corpus_rel, tag, arch, resume, iters)
    (dst / "train.sh").write_text("#!/usr/bin/env bash\nset -e\n" + train_cmd + "\n", encoding="utf-8")
    (dst / "train.ps1").write_text(train_cmd.replace(" \\\n", " `\n") + "\n", encoding="utf-8")

    # FORK_README.md — what it is and exactly what to run.
    warn = "" if corpus_exists else (
        f"\n> ⚠️  The corpus `{corpus_dir}/` isn't on disk. Rebuild it first:\n>\n"
        f">     {regen}\n")
    vocab_line = f"- vocab: {arch['vocab_size']}\n" if arch["vocab_size"] else ""
    readme = f"""# Fork: {args.name}

{'Continuing' if resume else 'From-scratch copy of'} **{source}** — {arch['n_layer']} layers, {arch['n_head']} heads, n_embd {arch['n_embd']}, block {arch['block']}.
{vocab_line}
This folder has its **own copy of the model code** (`model.py`, `train.py`, …).
Change the architecture or the training loop here and it affects only this fork.
{warn}
## Train it

```bash
cd experiments/{args.name}
{train_cmd}
```

- reads the shared corpus at `{corpus_rel}` (doesn't copy or clobber it)
- writes `ckpt_{tag}.pt` + `loss_{tag}.jsonl` here in the fork
{('- starting checkpoint: ' + ckpt_note) if resume else '- starts from random init'}
- edit `model.py` first if you want a different architecture (then use `--scratch`-style
  training; a shape change means the old weights can't be resumed and it starts fresh)

## Watch it / sample

```bash
python sample_big.py "def "
```

## Turn it into a browsable tier

From the **main** `scratch_coder/` dir:

```bash
python export_inspect.py --weights --ckpt experiments/{args.name}/ckpt_{tag}.pt \\
  --out-dir web/tiers/{args.name} --label {args.name.title()}
```

then add `{{ "id": "{args.name}", "label": "{args.name.title()}", "params": … }}` to
`web/tiers/manifest.json` (and the mirror) so it shows up in the inspector and Library.
"""
    (dst / "FORK_README.md").write_text(readme, encoding="utf-8")

    # --- report ------------------------------------------------------------------
    print(f"forked {source} -> {dst.relative_to(HERE)}  ({mode})")
    if resume:
        print(f"  checkpoint: {ckpt_note}")
    if not corpus_exists:
        print(f"  [!] corpus {corpus_dir}/ missing - rebuild: {regen}")
    print(f"\nnext:\n  cd experiments/{args.name}\n  # edit model.py if you like, then:\n"
          f"  bash train.sh        # or:  ./train.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
