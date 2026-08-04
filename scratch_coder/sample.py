"""
sample.py — generate code from the trained from-scratch model.

    py sample.py "def fibonacci"
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

from model import GPT, GPTConfig

D = Path(__file__).resolve().parent / "data"
device = "cuda" if torch.cuda.is_available() else "cpu"

meta = pickle.load(open(D / "meta.pkl", "rb"))
ckpt = torch.load(D / "ckpt.pt", map_location=device)
cfg = GPTConfig(**ckpt["cfg"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "def "
ids = torch.tensor([[meta["stoi"].get(c, 0) for c in prompt]], dtype=torch.long, device=device)
out = model.generate(ids, max_new_tokens=400, temperature=0.8, top_k=40)[0].tolist()
print("".join(meta["itos"][i] for i in out))
