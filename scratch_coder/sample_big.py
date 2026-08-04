"""Sample from the continuously-trained model without disturbing the trainer."""
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import torch

from model import GPT, GPTConfig

D = Path(__file__).resolve().parent / "data_big"
tmp = D / "ckpt_read.pt"
shutil.copy(D / "ckpt.pt", tmp)   # copy so we don't race the trainer's write

meta = pickle.load(open(D / "meta.pkl", "rb"))
ck = torch.load(tmp, map_location="cpu", weights_only=False)
cfg = GPTConfig(**ck["cfg"])
m = GPT(cfg)
m.load_state_dict(ck["model"])
m.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "def "
ids = torch.tensor([[meta["stoi"].get(c, 0) for c in prompt]], dtype=torch.long)
out = m.generate(ids, 320, temperature=0.8, top_k=40)[0].tolist()
print(f"[iter {ck['iter']} | best val loss {ck.get('best'):.3f}]")
print("".join(meta["itos"][i] for i in out))
tmp.unlink(missing_ok=True)
