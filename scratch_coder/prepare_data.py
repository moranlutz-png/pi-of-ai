"""
prepare_data.py — build a training corpus of REAL Python code (offline).

We teach the from-scratch model to code by feeding it actual Python source. The
easiest fully-offline corpus is the Python standard library already on this
machine. Character-level: we map every distinct character to an integer.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import sysconfig
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description="Build a char-level Python corpus from the stdlib.")
ap.add_argument("--out-dir", default="data", help="where to write train.bin/val.bin/meta.pkl")
ap.add_argument("--max-bytes", type=int, default=32_000_000, help="corpus size cap")
ap.add_argument("--site-packages", action="store_true",
                help="also include installed third-party code — more data, noisier (for the Max tier)")
cli = ap.parse_args()

OUT = Path(__file__).resolve().parent / cli.out_dir
OUT.mkdir(parents=True, exist_ok=True)

# The Python standard library — thousands of .py files, already on disk, offline.
LIB = Path(sys.executable).resolve().parent / "Lib"
if not (LIB / "os.py").exists():
    # Fallback: ask the interpreter where its stdlib lives (portable across OSes).
    LIB = Path(sysconfig.get_path("stdlib"))

MAX_BYTES = cli.max_bytes

texts, total = [], 0
# Recursive (rglob), so subpackages (json/, email/, importlib/, unittest/, ...) and the
# stdlib's own test suites all count — it's all real, idiomatic Python. By default only
# site-packages is skipped (installed third-party code); --site-packages includes it for
# the Max tier, where sheer volume matters more than staying purely canonical.
for path in sorted(LIB.rglob("*.py")):
    if not cli.site_packages and "site-packages" in str(path).lower():
        continue
    try:
        t = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    texts.append(t)
    total += len(t)
    if total >= MAX_BYTES:
        break

data = "\n\n".join(texts)[:MAX_BYTES]
# Keep it ASCII. The stdlib's Unicode test data drags in hundreds of rare characters
# (accents, symbols, whole other scripts) that a tiny char model would only ever see a
# handful of times — bloating the vocab and wasting capacity. Python source is ASCII
# anyway; keep printable ASCII plus newline and tab, drop the rest.
data = "".join(c for c in data if c == "\n" or c == "\t" or 32 <= ord(c) <= 126)
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
