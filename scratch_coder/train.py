"""
train.py — train the from-scratch coder from random noise.

Watch the val loss fall and the samples go from gibberish -> Python-shaped text.
Every weight here started as random numbers; the only thing that makes it "know"
Python is this loop. Runs on CPU (slow but works) or GPU if available.
"""
from __future__ import annotations

import argparse
import json
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
args = ap.parse_args()

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


cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=BLOCK, n_layer=4, n_head=4, n_embd=128, mlp=args.mlp)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device: {device} | params: {n_params/1e6:.2f}M | vocab: {cfg.vocab_size}")

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# --- resume (keep teaching the same model) — opt-in; default is still from random --
CKPT = D / "ckpt.pt"
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
LOSS_LOG = D / "loss.jsonl"
if not (args.resume and CKPT.exists()):
    LOSS_LOG.write_text("", encoding="utf-8")


def save():
    # iter/best/opt so a later --resume picks up exactly where this left off.
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg.__dict__, "iter": it, "best": best}, CKPT)


end_iter = start_iter + args.iters
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
            print(f"iter {it:4d} | val loss {vl:.3f} | best {best:.3f} | {time.time()-t0:.0f}s")
            if it > start_iter:
                print(f"           grads {format_layer_norms(norms)}")
            with LOSS_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"iter": it, "val_loss": round(vl, 5), "best": round(best, 5),
                                     "elapsed_s": round(time.time() - t0, 1), "params": n_params,
                                     "layer_norms": [round(float(n), 6) for n in norms]}) + "\n")
        if it >= end_iter:
            break
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        # Checked BEFORE backward(): a NaN loss is in the weights one line later,
        # and from then on every iteration trains on nothing while still printing
        # as though it were working. .item() costs a device sync each step, which
        # on a 2.3M-parameter model is far cheaper than finding out at the end.
        lv = loss.item()
        if is_poisoned(lv):
            raise SystemExit(poisoned_report(it, lv, last_grad_norm))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = clip_and_measure(model, GRAD_CLIP)
        opt.step()
except KeyboardInterrupt:
    print("\nstopping — saving checkpoint…")

save()
print(f"\nsaved checkpoint -> {CKPT} (iter {it}, best val {best:.3f})")
print("\n===== sample from the from-scratch model =====")
print(sample("def ", 300))
print("==============================================")
