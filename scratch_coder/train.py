"""
train.py — train the from-scratch coder from random noise.

Watch the val loss fall and the samples go from gibberish -> Python-shaped text.
Every weight here started as random numbers; the only thing that makes it "know"
Python is this loop. Runs on CPU (slow but works) or GPU if available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from guards import (DEFAULT_CLIP, clip_and_measure, format_layer_norms,
                    is_poisoned, layer_grad_norms, poisoned_report)
from model import GPT, GPTConfig

D = Path(__file__).resolve().parent / "data"

# --- config (small so it trains on a laptop CPU in a few minutes) ------------
BLOCK = 128
BATCH = 32
ITERS = 500
EVAL_EVERY = 100
LR = 3e-3
GRAD_CLIP = DEFAULT_CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

ap = argparse.ArgumentParser(description="Train the from-scratch coder from random noise.")
ap.add_argument("--mlp", default="default",
                help="block MLP from the registry in layers/ (default: the built-in MLP)")
ap.add_argument("--resume", action="store_true",
                help="continue from data/ckpt.pt (keep teaching) instead of starting from random")
ap.add_argument("--iters", type=int, default=ITERS, help="iterations to run this invocation")
# Capacity knobs — defaults keep the fast ~0.84M teaching model; bump for a bigger one.
ap.add_argument("--n-layer", type=int, default=4, help="transformer blocks (deeper = more capacity)")
ap.add_argument("--n-head", type=int, default=4, help="attention heads (must divide n-embd)")
ap.add_argument("--n-embd", type=int, default=128, help="embedding width")
ap.add_argument("--block", type=int, default=BLOCK, help="context length in characters")
ap.add_argument("--tag", default="", help="suffix for checkpoint/loss files so tiers don't clash")
ap.add_argument("--data", default="data", help="data directory (train.bin/val.bin/meta.pkl) to train on")
ap.add_argument("--out-dir", default="", help="where checkpoint/loss files go (default: the --data dir); "
                "a fork points this at its own folder so it reads a shared corpus without clobbering it")
ap.add_argument("--batch", type=int, default=BATCH, help="batch size (lower it to fit a bigger model in VRAM)")
ap.add_argument("--lr", type=float, default=LR, help="learning rate (lower it for bigger models, e.g. 1e-3)")
ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant",
                help="LR schedule: 'constant' (the old behaviour) or 'cosine' with a warmup then decay to --min-lr")
ap.add_argument("--warmup", type=int, default=200, help="cosine schedule: linear warmup steps from 0 up to --lr")
ap.add_argument("--min-lr", type=float, default=0.0, help="cosine schedule: floor to decay to (0 -> --lr/10)")
ap.add_argument("--grad-accum", type=int, default=1,
                help="accumulate this many micro-batches per optimizer step — a bigger EFFECTIVE batch "
                     "(batch x grad-accum) for steadier gradients, without the VRAM of a bigger raw batch")
ap.add_argument("--fresh-schedule", action="store_true",
                help="run the cosine schedule relative to THIS invocation's start, not absolute iter — "
                     "for a continued-training phase (resume a checkpoint, then train on more data with "
                     "its own warmup + decay). Without it, --resume continues the original schedule.")
ap.add_argument("--sched-start", type=int, default=-1,
                help="(with --fresh-schedule) the iter the phase's schedule began — pass this on a "
                     "RESUME so the cosine continues instead of re-warming up (default: this run's start)")
ap.add_argument("--sched-total", type=int, default=-1,
                help="(with --fresh-schedule) the phase's TOTAL length in iters (default: --iters)")
args = ap.parse_args()
BLOCK = args.block   # get_batch reads these module globals, so set them before that runs
BATCH = args.batch
LR = args.lr
MIN_LR = args.min_lr if args.min_lr > 0 else LR / 10   # cosine floor
D = Path(__file__).resolve().parent / args.data
# Checkpoints/logs go beside the data by default, but --out-dir splits them off so a
# fork can read a shared corpus (--data) while writing its own checkpoint here.
OUT = (Path(__file__).resolve().parent / args.out_dir) if args.out_dir else D
OUT.mkdir(parents=True, exist_ok=True)
# Per-tier files when --tag is given, so several sizes can be trained side by side.
CKPT = OUT / (f"ckpt_{args.tag}.pt" if args.tag else "ckpt.pt")
LOSS_LOG = OUT / (f"loss_{args.tag}.jsonl" if args.tag else "loss.jsonl")

meta = pickle.load(open(D / "meta.pkl", "rb"))
itos, stoi = meta["itos"], meta["stoi"]
train_ids = np.fromfile(D / "train.bin", dtype=np.uint16)
val_ids = np.fromfile(D / "val.bin", dtype=np.uint16)


def get_batch(split: str):
    d = train_ids if split == "train" else val_ids
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(d[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=args.block,
                n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, mlp=args.mlp)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device: {device} | params: {n_params/1e6:.2f}M | vocab: {cfg.vocab_size}")

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# --- resume (keep teaching the same model) — opt-in; default is still from random --
start_iter, best = 0, 1e9
if args.resume and CKPT.exists():
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    if ck.get("cfg") == cfg.__dict__:
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])     # Adam momentum too, so it picks up smoothly
        start_iter = int(ck.get("iter", 0)); best = float(ck.get("best", 1e9))
        print(f"resumed from iter {start_iter} (best val {best:.3f})")
    else:
        print("checkpoint config differs from this model — starting from random instead")


@torch.no_grad()
def val_loss() -> float:
    model.eval()
    losses = [model(*get_batch("val"))[1].item() for _ in range(5)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(prompt: str = "def ", n: int = 240) -> str:
    model.eval()
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(ids, n, temperature=0.8, top_k=40)[0].tolist()
    model.train()
    return "".join(itos[i] for i in out)


t0 = time.time()
# Loss and per-layer gradient norms, one JSON object per eval — the data source
# the web UI's loss/gradient curves read (train_forever.py writes the same shape).
# A fresh run starts a clean file; a resume appends so the curve continues.
if not (args.resume and CKPT.exists()):
    LOSS_LOG.write_text("", encoding="utf-8")


def save():
    # iter/best/opt so a later --resume picks up exactly where this left off.
    # Atomic: write a temp file then rename over CKPT, so a process killed mid-save
    # (e.g. a session ending) can never truncate the good checkpoint — the worst case
    # is the previous eval's checkpoint survives intact, and --resume continues from it.
    tmp = CKPT.with_name(CKPT.name + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg.__dict__, "iter": it, "best": best}, tmp)
    os.replace(tmp, CKPT)


end_iter = start_iter + args.iters


def lr_at(step):
    """Constant (the old behaviour), or a linear warmup then a cosine decay to MIN_LR —
    the standard schedule that lets a bigger model settle into a lower loss than a flat
    rate reaches: high early to move fast, low late to fine-tune without overshooting."""
    if args.schedule != "cosine":
        return LR
    # --fresh-schedule: measure from this phase's start over this phase's length, so a
    # continued run gets its own warmup + decay. Default: absolute, so an interrupted run
    # resumed with the remaining --iters continues the one original schedule.
    if args.fresh_schedule:
        ref = args.sched_start if args.sched_start >= 0 else start_iter   # phase's true start (resume-proof)
        span = args.sched_total if args.sched_total >= 0 else args.iters
        s = step - ref
    else:
        s, span = step, end_iter
    if s < args.warmup:
        return LR * (s + 1) / max(1, args.warmup)
    prog = min(1.0, (s - args.warmup) / max(1, span - args.warmup))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))


print(f"schedule: {args.schedule} (lr {LR:g}"
      + (f" -> {MIN_LR:g}, warmup {args.warmup})" if args.schedule == "cosine" else ")")
      + f" | grad-accum {args.grad_accum} -> effective batch {BATCH * args.grad_accum}")
# The norm from the last completed step, so that if the next one blows up we can
# report what the gradients were doing just before it did.
last_grad_norm = float("nan")
it = start_iter
try:
    for it in range(start_iter, end_iter + 1):
        if it % EVAL_EVERY == 0:
            vl = val_loss(); best = min(best, vl)
            norms = layer_grad_norms(model) if it > start_iter else []
            save()   # checkpoint at every eval, so a killed long run keeps its progress
            print(f"iter {it:4d} | val loss {vl:.3f} | best {best:.3f} | lr {lr_at(it):.2e} | {time.time()-t0:.0f}s")
            if it > start_iter:
                print(f"           grads {format_layer_norms(norms)}")
            with LOSS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"iter": it, "val_loss": round(vl, 5), "best": round(best, 5),
                                     "lr": round(lr_at(it), 7),
                                     "elapsed_s": round(time.time() - t0, 1), "params": n_params,
                                     "layer_norms": [round(float(n), 6) for n in norms]}) + "\n")
        if it >= end_iter:
            break
        for g in opt.param_groups:   # apply the LR schedule for this step
            g["lr"] = lr_at(it)

        # One optimizer step over grad-accum micro-batches: a bigger EFFECTIVE batch for
        # steadier gradients at the VRAM of a single micro-batch. accum=1 is the old path.
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            xb, yb = get_batch("train")
            _, loss = model(xb, yb)
            # Checked BEFORE backward(): a NaN loss is in the weights one line later, and
            # from then on every iteration trains on nothing while still printing as though
            # it were working. .item() costs a device sync, far cheaper than finding out at the end.
            lv = loss.item()
            if is_poisoned(lv):
                raise SystemExit(poisoned_report(it, lv, last_grad_norm))
            (loss / args.grad_accum).backward()   # scale so the accumulated grad is the mean
        last_grad_norm = clip_and_measure(model, GRAD_CLIP)
        opt.step()
except KeyboardInterrupt:
    print("\nstopping — saving checkpoint…")

save()
print(f"\nsaved checkpoint -> {CKPT} (iter {it}, best val {best:.3f})")
print("\n===== sample from the from-scratch model =====")
print(sample("def ", 300))
print("==============================================")
