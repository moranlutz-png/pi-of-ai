"""
train_forever.py — a continuous training daemon for the from-scratch coder.

Trains on a LARGE corpus of real, human-written Python (the highest-quality
"content" for learning to code), checkpoints as it goes, prints samples so you
watch it improve, and RESUMES from the last checkpoint if restarted. Leave it
running; it keeps getting better (structurally) until it hits the capacity
ceiling of its size.

    py train_forever.py          # start (or resume) continuous training
    Ctrl-C to stop; run again to pick up where it left off.

Honest note: more data + more time makes the STRUCTURE sharper (real words,
matched brackets, valid-looking syntax). It will not learn to reason — that needs
a bigger model, not more training. Bump N_EMBD / N_LAYER below for more capacity.
"""
from __future__ import annotations

import json
import pickle
import sys
import sysconfig
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch

from guards import (DEFAULT_CLIP, clip_and_measure, format_layer_norms,
                    is_poisoned, layer_grad_norms, poisoned_report)
from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_big"
DATA.mkdir(exist_ok=True)

# --- model capacity (bigger than the demo, so more training actually helps) --
BLOCK, N_LAYER, N_HEAD, N_EMBD = 160, 5, 6, 192
BATCH, LR = 32, 3e-3
GRAD_CLIP = DEFAULT_CLIP
EVAL_EVERY, SAMPLE_EVERY, MAX_ITERS = 200, 1000, 100_000
CORPUS_MB = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


def build_corpus() -> None:
    """Gather up to CORPUS_MB of real Python from the stdlib (recursively)."""
    lib = Path(sys.executable).resolve().parent / "Lib"
    if not (lib / "os.py").exists():
        lib = Path(sysconfig.get_path("stdlib"))   # portable stdlib location
    texts, total, cap = [], 0, CORPUS_MB * 1_000_000
    for path in sorted(lib.rglob("*.py")):
        if "test" in str(path).lower():   # skip test files — noisier, less canonical
            continue
        try:
            t = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        texts.append(t); total += len(t)
        if total >= cap:
            break
    data = "\n\n".join(texts)[:cap]
    chars = sorted(set(data))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in data], dtype=np.uint16)
    n = int(0.9 * len(ids))
    ids[:n].tofile(DATA / "train.bin")
    ids[n:].tofile(DATA / "val.bin")
    with open(DATA / "meta.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": {i: c for c, i in stoi.items()}, "vocab_size": len(chars)}, f)
    print(f"built corpus: {len(data):,} chars, vocab {len(chars)}", flush=True)


if not (DATA / "train.bin").exists():
    build_corpus()

meta = pickle.load(open(DATA / "meta.pkl", "rb"))
itos, stoi = meta["itos"], meta["stoi"]
train_ids = np.fromfile(DATA / "train.bin", dtype=np.uint16)
val_ids = np.fromfile(DATA / "val.bin", dtype=np.uint16)


def get_batch(split: str):
    d = train_ids if split == "train" else val_ids
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(d[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=BLOCK, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD)
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# --- resume if a compatible checkpoint exists -------------------------------
CKPT = DATA / "ckpt.pt"
start_iter, best = 0, 1e9
if CKPT.exists():
    ck = torch.load(CKPT, map_location=device)
    if ck["cfg"] == cfg.__dict__:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_iter, best = ck["iter"], ck.get("best", 1e9)
        print(f"resumed from iter {start_iter} (best val {best:.3f})", flush=True)

n_params = sum(p.numel() for p in model.parameters())
print(f"device: {device} | params: {n_params/1e6:.2f}M | "
      f"vocab: {cfg.vocab_size} | starting at iter {start_iter}", flush=True)


@torch.no_grad()
def val_loss() -> float:
    model.eval()
    v = sum(model(*get_batch("val"))[1].item() for _ in range(8)) / 8
    model.train()
    return v


@torch.no_grad()
def sample(prompt: str = "def ", n: int = 260) -> str:
    model.eval()
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(ids, n, temperature=0.8, top_k=40)[0].tolist()
    model.train()
    return "".join(itos[i] for i in out)


def save(it: int, best_val: float) -> None:
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "cfg": cfg.__dict__, "iter": it, "best": best_val}, CKPT)


# Loss history for the UI's loss curve. Appended as JSON Lines so a run that is
# Ctrl-C'd (or killed) keeps every point written so far, and resuming a run just
# continues the file — the curve survives across sessions like the checkpoint.
LOSS_LOG = DATA / "loss.jsonl"


def log_loss(it: int, val: float, best_val: float, norms: list[float]) -> None:
    rec = {"iter": it, "val_loss": round(val, 5), "best": round(best_val, 5),
           "elapsed_s": round(time.time() - t0, 1), "params": n_params,
           "layer_norms": [round(float(n), 6) for n in norms]}
    try:
        with LOSS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as e:
        # Never let logging take down a long training run.
        print(f"  [loss-log] write failed: {e}", flush=True)


t0 = time.time()
# The norm from the last completed step, so a blow-up can report what the
# gradients were doing immediately before it.
last_grad_norm = float("nan")
try:
    for it in count(start_iter):
        if it >= MAX_ITERS:
            break
        if it % EVAL_EVERY == 0:
            vl = val_loss(); best = min(best, vl)
            norms = layer_grad_norms(model) if it > start_iter else []
            save(it, best)
            log_loss(it, vl, best, norms)
            print(f"iter {it:6d} | val loss {vl:.3f} | best {best:.3f} | {time.time()-t0:.0f}s", flush=True)
            if it > start_iter:
                print(f"         grads {format_layer_norms(norms)}", flush=True)
        if it % SAMPLE_EVERY == 0 and it > start_iter:
            print("  sample:", repr(sample("def ", 120)[:120]), flush=True)
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        # Checked BEFORE backward(), for the reason this whole guard exists: a
        # NaN does not stop the loop. It lands in the weights on the next line
        # and every iteration after it trains on nothing, while the counter
        # keeps climbing and the samples keep printing. This run is designed to
        # be left going for hours, which is exactly how a poisoned one goes
        # unnoticed the longest.
        lv = loss.item()
        if is_poisoned(lv):
            # Deliberately does NOT save. The checkpoint on disk is from the
            # last eval, before the poisoning; writing the current weights over
            # it would destroy the only good copy — and this trainer's whole
            # promise is that you can restart and pick up where you left off.
            print(poisoned_report(it, lv, last_grad_norm), flush=True)
            print(f"\nCheckpoint at {CKPT} is untouched and still good.", flush=True)
            raise SystemExit(1)

        opt.zero_grad(set_to_none=True); loss.backward()
        last_grad_norm = clip_and_measure(model, GRAD_CLIP)
        opt.step()
except KeyboardInterrupt:
    print("\nstopping — saving checkpoint…", flush=True)
    save(it, best)
    print(f"saved at iter {it}. Run again to resume.", flush=True)
