"""
arch_map.py — the model's tensor map, computed from the architecture alone.

WHAT THIS IS FOR
----------------
The Knob Matrix in the web UI needs to draw one cell per weight tensor: what it
is, how big it is, where it sits in the stack. That layout is fully determined
by the config in `model.py` — you do not need a trained checkpoint to know it.

So this module answers "what does the model look like?" with zero dependencies:
no torch, no numpy, pure stdlib. It runs on a Chromebook, in CI, or here.

Weight *values* are a separate question, answered later by an exporter that does
have torch and does have a checkpoint. The tensor names emitted here match
PyTorch's `state_dict()` keys exactly, so the two join cleanly.

USAGE
-----
    python arch_map.py                    # current model (train_forever.py's config)
    python arch_map.py --json map.json    # write the map for the UI
    python arch_map.py --target 1e9       # what shape reaches 1B parameters?

NOTE ON nn.Linear
-----------------
PyTorch stores a Linear's weight as [out_features, in_features] — transposed
from how the maths is usually written. Shapes below follow PyTorch so they line
up with a real state_dict.
"""
from __future__ import annotations

import argparse
import json
import sys

# Defaults mirror train_forever.py — the model actually being trained.
DEFAULTS = dict(vocab_size=100, block_size=160, n_layer=5, n_head=6, n_embd=192)


def tensor_map(vocab_size: int, block_size: int, n_layer: int, n_head: int,
               n_embd: int) -> dict:
    """Every parameter tensor in the model, in forward-pass order.

    Mirrors model.py exactly. `kind` is a hint for the UI's colour grouping;
    `role` is the one-line human explanation shown on hover.
    """
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) must divide by n_head ({n_head})")

    groups = []

    groups.append({
        "name": "embedding",
        "kind": "embedding",
        "tensors": [
            {"name": "tok_emb.weight", "shape": [vocab_size, n_embd],
             "role": "one row per character in the vocabulary"},
            {"name": "pos_emb.weight", "shape": [block_size, n_embd],
             "role": "one row per position in the context window"},
        ],
    })

    for i in range(n_layer):
        groups.append({
            "name": f"block.{i}",
            "kind": "block",
            "index": i,
            "tensors": [
                {"name": f"blocks.{i}.ln1.weight", "shape": [n_embd], "role": "norm scale, pre-attention"},
                {"name": f"blocks.{i}.ln1.bias", "shape": [n_embd], "role": "norm shift, pre-attention"},
                {"name": f"blocks.{i}.attn.c_attn.weight", "shape": [3 * n_embd, n_embd],
                 "role": "query, key and value projections fused into one matmul"},
                {"name": f"blocks.{i}.attn.c_attn.bias", "shape": [3 * n_embd], "role": "q/k/v bias"},
                {"name": f"blocks.{i}.attn.c_proj.weight", "shape": [n_embd, n_embd],
                 "role": "mixes the attention heads back together"},
                {"name": f"blocks.{i}.attn.c_proj.bias", "shape": [n_embd], "role": "output projection bias"},
                {"name": f"blocks.{i}.ln2.weight", "shape": [n_embd], "role": "norm scale, pre-MLP"},
                {"name": f"blocks.{i}.ln2.bias", "shape": [n_embd], "role": "norm shift, pre-MLP"},
                {"name": f"blocks.{i}.mlp.c_fc.weight", "shape": [4 * n_embd, n_embd],
                 "role": "widens to 4x — where most of the thinking happens"},
                {"name": f"blocks.{i}.mlp.c_fc.bias", "shape": [4 * n_embd], "role": "widening bias"},
                {"name": f"blocks.{i}.mlp.c_proj.weight", "shape": [n_embd, 4 * n_embd],
                 "role": "narrows back to the residual width"},
                {"name": f"blocks.{i}.mlp.c_proj.bias", "shape": [n_embd], "role": "narrowing bias"},
            ],
        })

    groups.append({
        "name": "output",
        "kind": "output",
        "tensors": [
            {"name": "ln_f.weight", "shape": [n_embd], "role": "final norm scale"},
            {"name": "ln_f.bias", "shape": [n_embd], "role": "final norm shift"},
            {"name": "head.weight", "shape": [vocab_size, n_embd],
             "role": "scores every character as the next one (no bias)"},
        ],
    })

    # Fill in per-tensor and per-group counts.
    total = 0
    for g in groups:
        g_total = 0
        for t in g["tensors"]:
            n = 1
            for d in t["shape"]:
                n *= d
            t["params"] = n
            g_total += n
        g["params"] = g_total
        total += g_total

    # The causal mask is a registered buffer, not a parameter — it is not
    # trained and must not be counted. Recorded so the UI can show it honestly.
    buffers = [{
        "name": "blocks.*.attn.mask",
        "shape": [1, 1, block_size, block_size],
        "per_block_elements": block_size * block_size,
        "role": "lower-triangular causal mask — a fixed buffer, not learned",
    }]

    return {
        "config": {"vocab_size": vocab_size, "block_size": block_size,
                   "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
                   "head_dim": n_embd // n_head},
        "total_params": total,
        "groups": groups,
        "buffers": buffers,
    }


def block_params(n_embd: int) -> int:
    """Parameters in one transformer block. Useful for scaling arithmetic."""
    e = n_embd
    return (2 * e) + (e * 3 * e + 3 * e) + (e * e + e) + (2 * e) \
        + (e * 4 * e + 4 * e) + (4 * e * e + e)


# Width-to-depth ratio of real transformers: GPT-2 small is 768/12 = 64,
# GPT-3 is 12288/96 = 128. Shapes far outside this band hit the same parameter
# count but train badly — a 1266-layer stack is arithmetically valid and
# useless. Suggestions are filtered to the band, then ranked by closeness.
ASPECT_MIN, ASPECT_MAX = 32, 192


def suggest_shapes(target: float, vocab_size: int, block_size: int,
                   candidates=(256, 384, 512, 768, 1024, 1280, 1600, 2048)) -> list:
    """Buildable configs that land nearest a target parameter count.

    The block term dominates (~12 * n_layer * n_embd^2), so for each candidate
    width we solve for the depth that gets closest, drop shapes with an
    implausible width:depth ratio, then report the error.
    """
    out = []
    for e in candidates:
        per = block_params(e)
        fixed = vocab_size * e + block_size * e + 2 * e + e * vocab_size
        n_layer = max(1, round((target - fixed) / per))

        aspect = e / n_layer
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # n_embd must divide evenly by the head count.
        heads = next((h for h in (16, 12, 8, 6, 4) if e % h == 0), None)
        if heads is None:
            continue

        total = fixed + n_layer * per
        out.append({"n_embd": e, "n_layer": n_layer, "n_head": heads,
                    "aspect": round(aspect, 1), "total_params": total,
                    "error_pct": 100 * (total - target) / target})
    out.sort(key=lambda r: abs(r["error_pct"]))
    return out


def _human(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tensor map for the scratch_coder GPT.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=v)
    ap.add_argument("--json", metavar="PATH", help="write the map as JSON (- for stdout)")
    ap.add_argument("--target", type=float, help="suggest configs near this parameter count")
    args = ap.parse_args()

    m = tensor_map(args.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd)

    if args.json:
        text = json.dumps(m, indent=2)
        if args.json == "-":
            print(text)
        else:
            with open(args.json, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"wrote {args.json}  ({m['total_params']:,} params)")
        return 0

    c = m["config"]
    print(f"scratch_coder GPT — {_human(m['total_params'])} parameters "
          f"({m['total_params']:,})")
    print(f"  n_layer={c['n_layer']}  n_head={c['n_head']}  n_embd={c['n_embd']}  "
          f"head_dim={c['head_dim']}  block_size={c['block_size']}  vocab={c['vocab_size']}\n")

    for g in m["groups"]:
        share = 100 * g["params"] / m["total_params"]
        print(f"  {g['name']:<12} {g['params']:>12,}  {share:5.1f}%  "
              f"({len(g['tensors'])} tensors)")

    if args.target:
        print(f"\nShapes near {_human(int(args.target))} parameters:")
        rows = suggest_shapes(args.target, args.vocab_size, args.block_size)
        if not rows:
            print("  no buildable shape in the candidate widths — widen the list.")
        for r in rows[:5]:
            print(f"  n_embd={r['n_embd']:<5} n_layer={r['n_layer']:<3} "
                  f"n_head={r['n_head']:<3} aspect={r['aspect']:<6} "
                  f"-> {_human(r['total_params']):>8} ({r['error_pct']:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
