"""
prepare_data.py — build a training corpus of REAL Python code (offline).

We teach the from-scratch model to code by feeding it actual Python source. The
easiest fully-offline corpus is the Python standard library already on this
machine. Character-level: we map every distinct character to an integer.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import random
import sys
import sysconfig
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description="Build a char-level Python corpus from the stdlib.")
ap.add_argument("--out-dir", default="data", help="where to write train.bin/val.bin/meta.pkl")
ap.add_argument("--max-bytes", type=int, default=32_000_000, help="corpus size cap")
ap.add_argument("--site-packages", action="store_true",
                help="also include installed third-party code — more data, noisier (for the Max tier)")
ap.add_argument("--extra-lib", action="append", default=[], metavar="DIR",
                help="additional Python Lib dir(s) to scan beyond this interpreter's — repeatable, "
                     "so several installs' code can be pooled for a bigger corpus")
ap.add_argument("--shuffle", action="store_true",
                help="shuffle files before the byte cap, so the corpus is a diverse cross-section "
                     "of the whole pool rather than the alphabetically-first slice")
ap.add_argument("--seed", type=int, default=1337, help="shuffle seed (reproducible corpora)")
cli = ap.parse_args()

OUT = Path(__file__).resolve().parent / cli.out_dir
OUT.mkdir(parents=True, exist_ok=True)

# The Python standard library — thousands of .py files, already on disk, offline.
LIB = Path(sys.executable).resolve().parent / "Lib"
if not (LIB / "os.py").exists():
    # Fallback: ask the interpreter where its stdlib lives (portable across OSes).
    LIB = Path(sysconfig.get_path("stdlib"))

MAX_BYTES = cli.max_bytes

# Pool the .py files from this interpreter's Lib plus any --extra-lib dirs, so several
# installs' code (stdlib + site-packages + a torch venv, say) can feed one big corpus.
# Recursive (rglob), so subpackages and the stdlib's own test suites all count. By default
# site-packages is skipped; --site-packages includes it, where sheer volume matters.
libs = [LIB] + [Path(x) for x in cli.extra_lib]
paths, seen_paths = [], set()
for lib in libs:
    if not lib.exists():
        print(f"skip (missing): {lib}"); continue
    for path in lib.rglob("*.py"):
        if not cli.site_packages and "site-packages" in str(path).lower():
            continue
        key = str(path.resolve()).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key); paths.append(path)

# Shuffle for a diverse cross-section (else the byte cap just takes the alphabetical-first
# slice — all 'a'..'m' packages); sorted otherwise, to keep the small tiers reproducible.
random.Random(cli.seed).shuffle(paths) if cli.shuffle else paths.sort()

texts, total, seen_hashes = [], 0, set()
for path in paths:
    try:
        t = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    h = hashlib.md5(t.encode("utf-8", "ignore")).digest()   # drop exact dupes (same file in two installs)
    if h in seen_hashes:
        continue
    seen_hashes.add(h)
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
