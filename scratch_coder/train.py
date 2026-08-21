"""
train.py — train the from-scratch coder from random noise.

Watch the val loss fall and the samples go from gibberish -> Python-shaped text.
Every weight here started as random numbers; the only thing that makes it "know"
Python is this loop. Runs on CPU (slow but works) or GPU if available.
"""
from __future__ import annotations

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


cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=BLOCK, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device: {device} | params: {n_params/1e6:.2f}M | vocab: {cfg.vocab_size}")

opt = torch.optim.AdamW(model.parameters(), lr=LR)


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
# The norm from the last completed step, so that if the next one blows up we can
# report what the gradients were doing just before it did.
last_grad_norm = float("nan")
for it in range(ITERS + 1):
    if it % EVAL_EVERY == 0:
        print(f"iter {it:4d} | val loss {val_loss():.3f} | {time.time()-t0:.0f}s")
        if it:
            print(f"           grads {format_layer_norms(layer_grad_norms(model))}")
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

torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, D / "ckpt.pt")
print(f"\nsaved checkpoint -> {D / 'ckpt.pt'}")
print("\n===== sample from the from-scratch model =====")
print(sample("def ", 300))
print("==============================================")
