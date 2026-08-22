"""
export_web.py  —  export the trained Scratch-Coder checkpoint for the browser.

Writes two files the web app loads:
    rules_baker/web/scratch/model.json   config + vocab + a tensor manifest
    rules_baker/web/scratch/model.bin    all weights, raw float32, concatenated

The JS side (scratchgpt.js) rebuilds our GPT from these and runs it 100% in the
browser — our own architecture, our own weights, nothing downloaded.

Re-run this anytime to refresh the web model with the latest training progress:
    D:\\rbgpu\\Scripts\\python.exe export_web.py     (or C:\\rb\\Scripts\\python.exe)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_big"
OUT = HERE.parent / "rules_baker" / "web" / "scratch"
OUT.mkdir(parents=True, exist_ok=True)

ck = torch.load(DATA / "ckpt.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]                      # vocab_size, block_size, n_layer, n_head, n_embd, ...
sd = ck["model"]
meta = pickle.load(open(DATA / "meta.pkl", "rb"))
itos = meta["itos"]                  # {int: char}

# Tensor order MUST match the load order in scratchgpt.js.
names = ["tok_emb.weight", "pos_emb.weight"]
for i in range(cfg["n_layer"]):
    p = f"blocks.{i}."
    names += [
        p + "ln1.weight", p + "ln1.bias",
        p + "attn.c_attn.weight", p + "attn.c_attn.bias",
        p + "attn.c_proj.weight", p + "attn.c_proj.bias",
        p + "ln2.weight", p + "ln2.bias",
        p + "mlp.c_fc.weight", p + "mlp.c_fc.bias",
        p + "mlp.c_proj.weight", p + "mlp.c_proj.bias",
    ]
names += ["ln_f.weight", "ln_f.bias", "head.weight"]

manifest, offset = [], 0
with open(OUT / "model.bin", "wb") as f:
    for n in names:
        arr = sd[n].detach().to(torch.float32).contiguous().cpu().numpy().ravel()
        f.write(arr.tobytes())
        manifest.append({"name": n, "offset": offset, "size": int(arr.size), "shape": list(sd[n].shape)})
        offset += int(arr.size)

model_json = {
    "config": {k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")},
    "itos": [itos[i] for i in range(cfg["vocab_size"])],   # char for each token id
    "manifest": manifest,
    "iter": int(ck.get("iter", 0)),
    "best_val": float(ck.get("best", 0.0)),
}
json.dump(model_json, open(OUT / "model.json", "w", encoding="utf-8"), ensure_ascii=True)

params = sum(m["size"] for m in manifest)
print(f"exported {len(names)} tensors · {params/1e6:.2f}M params · {offset*4/1e6:.1f} MB")
print(f"  iter {model_json['iter']} · best val {model_json['best_val']:.3f}")
print(f"  -> {OUT}")
