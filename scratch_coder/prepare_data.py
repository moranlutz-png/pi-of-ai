"""
prepare_data.py — build a training corpus of REAL Python code (offline).

We teach the from-scratch model to code by feeding it actual Python source. The
easiest fully-offline corpus is the Python standard library already on this
machine. Character-level: we map every distinct character to an integer.
"""
from __future__ import annotations

import pickle
import sys
import sysconfig
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

# The Python standard library — thousands of .py files, already on disk, offline.
LIB = Path(sys.executable).resolve().parent / "Lib"
if not (LIB / "os.py").exists():
    # Fallback: ask the interpreter where its stdlib lives (portable across OSes).
    LIB = Path(sysconfig.get_path("stdlib"))

MAX_BYTES = 2_000_000   # ~2 MB of Python is plenty for a tiny char model

texts, total = [], 0
for path in sorted(LIB.glob("*.py")):
    try:
        t = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    texts.append(t)
    total += len(t)
    if total >= MAX_BYTES:
        break

data = "\n\n".join(texts)[:MAX_BYTES]
print(f"source dir : {LIB}")
print(f"corpus     : {len(data):,} characters from {len(texts)} files")

chars = sorted(set(data))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
ids = np.array([stoi[c] for c in data], dtype=np.uint16)

n = int(0.9 * len(ids))
ids[:n].tofile(OUT / "train.bin")
ids[n:].tofile(OUT / "val.bin")
with open(OUT / "meta.pkl", "wb") as f:
    pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": len(chars)}, f)

print(f"vocab size : {len(chars)} distinct characters")
print(f"train/val  : {n:,} / {len(ids) - n:,} characters")
print(f"wrote      : {OUT / 'train.bin'}, val.bin, meta.pkl")
