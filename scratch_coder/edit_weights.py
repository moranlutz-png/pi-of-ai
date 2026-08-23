"""
edit_weights.py — surgery on a trained checkpoint, the CLI twin of the inspector's
"Weight surgery" panel. The panel edits weights in the browser's memory (nothing
saved); this writes a new checkpoint you keep, so you can re-export it to a tier and
study the damage — or feed it back into train.py to heal.

Named operations (all leave the shapes intact, so the model still runs):
  zero-layer   turn a block into a no-op pass-through (zero its output projections)
  add-noise    add sigma * (each tensor's std) * N(0,1)   [--sigma]
  scale        multiply the target's weights by a factor  [--factor]
  prune        zero the smallest-magnitude fraction of the target [--frac]

Target grammar (same as the panel):  all | embedding | output | block:N

Examples:
  python edit_weights.py --ckpt data/ckpt.pt --op zero-layer --layer 3
  python edit_weights.py --ckpt data/ckpt.pt --op add-noise --target all --sigma 0.1
  python edit_weights.py --ckpt data/ckpt.pt --op scale --target block:2 --factor 0.5
  python edit_weights.py --ckpt data/ckpt.pt --op prune --target all --frac 0.2

Then see it in the browser:
  python export_inspect.py --weights --ckpt data/ckpt_edited.pt --out-dir web/tiers/<id> --label <Label>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def names_for(sd: dict, target: str) -> list[str]:
    keys = list(sd.keys())
    if target == "all":
        return keys
    if target == "embedding":
        return [k for k in keys if k.startswith("tok_emb") or k.startswith("pos_emb")]
    if target == "output":
        return [k for k in keys if k.startswith("ln_f") or k.startswith("head")]
    if target.startswith("block:"):
        l = target.split(":", 1)[1]
        return [k for k in keys if k.startswith(f"blocks.{l}.")]
    raise SystemExit(f"unknown target {target!r} — use: all | embedding | output | block:N")


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit a trained checkpoint's weights (surgery).")
    ap.add_argument("--ckpt", type=Path, required=True, help="checkpoint to edit (e.g. data/ckpt.pt)")
    ap.add_argument("--op", required=True, choices=["zero-layer", "add-noise", "scale", "prune"])
    ap.add_argument("--target", default="all", help="all | embedding | output | block:N (default all)")
    ap.add_argument("--layer", type=int, help="shorthand for --target block:N (used by zero-layer)")
    ap.add_argument("--sigma", type=float, default=0.1, help="add-noise: noise std as a fraction of each tensor's std")
    ap.add_argument("--factor", type=float, default=0.5, help="scale: multiply weights by this")
    ap.add_argument("--frac", type=float, default=0.2, help="prune: fraction of smallest-magnitude weights to zero")
    ap.add_argument("--hard", action="store_true", help="zero-layer: zero EVERY tensor in the block, not just outputs")
    ap.add_argument("--out", type=Path, help="output checkpoint (default: <ckpt>_edited.pt)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"no checkpoint at {args.ckpt}")
    if args.layer is not None:
        args.target = f"block:{args.layer}"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck   # tolerate a bare state_dict
    torch.manual_seed(1337)
    touched, zeroed = [], 0

    if args.op == "zero-layer":
        if not args.target.startswith("block:"):
            raise SystemExit("zero-layer needs a block target: --layer N (or --target block:N)")
        l = args.target.split(":", 1)[1]
        if args.hard:
            targets = names_for(sd, args.target)
        else:
            targets = [f"blocks.{l}.attn.c_proj.weight", f"blocks.{l}.attn.c_proj.bias",
                       f"blocks.{l}.mlp.c_proj.weight", f"blocks.{l}.mlp.c_proj.bias"]
        for k in targets:
            if k in sd:
                sd[k].zero_(); touched.append(k)
        note = f"block {l} " + ("hard-zeroed" if args.hard else "disabled (output projections zeroed -> no-op)")

    elif args.op == "add-noise":
        for k in names_for(sd, args.target):
            t = sd[k]
            t.add_(torch.randn_like(t) * (args.sigma * t.float().std().item()))
            touched.append(k)
        note = f"added sigma={args.sigma} * std noise to {args.target}"

    elif args.op == "scale":
        for k in names_for(sd, args.target):
            sd[k].mul_(args.factor); touched.append(k)
        note = f"scaled {args.target} by {args.factor}"

    elif args.op == "prune":
        keys = names_for(sd, args.target)
        pool = torch.cat([sd[k].abs().flatten().float() for k in keys])
        thr = torch.quantile(pool, args.frac).item()
        for k in keys:
            mask = sd[k].abs() < thr
            zeroed += int(mask.sum().item()); sd[k][mask] = 0; touched.append(k)
        note = f"pruned {args.frac:.0%} smallest-|w| in {args.target} (|w| < {thr:.4g}), {zeroed:,} weights zeroed"

    out = args.out or args.ckpt.with_name(args.ckpt.stem + "_edited.pt")
    if "model" in ck:
        ck["model"] = sd
    torch.save(ck if "model" in ck else sd, out)

    print(f"{note}")
    print(f"  {len(touched)} tensors touched -> {out}")
    print(f"  see it: python export_inspect.py --weights --ckpt {out} --out-dir web/tiers/<id> --label <Label>")
    print(f"  heal it: python train.py --resume  (train.py keeps teaching these weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
