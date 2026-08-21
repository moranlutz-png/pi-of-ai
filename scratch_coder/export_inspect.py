"""
export_inspect.py — turn a scratch_coder checkpoint into JSON the web page reads.

The join arch_map.py was written to expect. This side has torch and a checkpoint;
the page never sees a .pt (a pickle is remote code execution waiting to happen).
It writes per-tensor *statistics* — never the weights themselves (that is the
attention rung's problem) — plus the same statistics for a freshly-random model of
the same shape (the trained-vs-random demo), plus the token embedding projected to
2D so the page can show which characters cluster.

    python export_inspect.py                      # data/ckpt.pt by default
    python export_inspect.py --ckpt data_big/ckpt.pt

The shape comes from the checkpoint's own cfg, NOT arch_map's defaults: those
mirror train_forever.py, and drawing a train.py checkpoint against them silently
renders a five-block stack holding four blocks of data.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from arch_map import tensor_map
from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
OUT = HERE / "web" / "inspect.json"


def tensor_stats(arr: np.ndarray) -> dict:
    """One number per cell for the Knob Matrix — nothing larger belongs in this file."""
    a = arr.ravel().astype(np.float64)
    absa = np.abs(a)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "absMax": float(absa.max()),
        "l2": float(np.sqrt(float((a * a).sum()))),
        "fracNearZero": float((absa < 1e-3).mean()),
    }


def stats_for(state_dict: dict, names: list[str]) -> dict:
    return {n: tensor_stats(state_dict[n].detach().cpu().numpy()) for n in names}


def pca_2d(mat: np.ndarray, iters: int = 300):
    """Project rows of `mat` to 2D. Power iteration on the covariance, top-2 by
    deflation — about fifteen lines and one fewer dependency than sklearn."""
    x = mat - mat.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / x.shape[0]                 # [C, C]
    total_var = float(np.trace(cov))
    rng = np.random.default_rng(0)
    comps, evals, c = [], [], cov.copy()
    for _ in range(2):
        v = rng.standard_normal(c.shape[0]); v /= np.linalg.norm(v)
        for _ in range(iters):
            v = c @ v
            nrm = np.linalg.norm(v)
            if nrm == 0:
                break
            v /= nrm
        lam = float(v @ c @ v)
        comps.append(v); evals.append(lam)
        c = c - lam * np.outer(v, v)             # deflate the found component
    pts = x @ np.stack(comps, axis=1)            # [V, 2]
    var_pct = 100.0 * (sum(evals) / total_var) if total_var > 0 else 0.0
    return pts, var_pct


def neighbours(emb: np.ndarray, itos: list[str], k: int = 6) -> dict:
    """Top-k cosine neighbours per character, computed in the FULL space — the
    check on the 2D picture. When the plot and this list disagree, this is right."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    out = {}
    for i in range(emb.shape[0]):
        idx = np.argsort(-sim[i])[:k]
        out[itos[i]] = [{"char": itos[int(j)], "cos": round(float(sim[i, j]), 3)} for j in idx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a checkpoint as something a page can read.")
    ap.add_argument("--ckpt", type=Path, default=HERE / "data" / "ckpt.pt")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"no checkpoint at {args.ckpt} — train one first (python train.py)")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    sd = ck["model"]

    # Step 0.1 — the layout comes from THIS checkpoint's shape.
    amap = tensor_map(cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"])
    names = [t["name"] for g in amap["groups"] for t in g["tensors"]]

    # Step 0.2 — trained stats, and a fresh random model at the same shape and seed.
    trained = stats_for(sd, names)
    torch.manual_seed(1337)
    random_stats = stats_for(GPT(GPTConfig(**cfg)).state_dict(), names)

    # Step 0.3 — the embedding matrix, projected, with full-space neighbours.
    meta = pickle.load(open(args.ckpt.parent / "meta.pkl", "rb"))
    itos = [meta["itos"][i] for i in range(cfg["vocab_size"])]
    emb = sd["tok_emb.weight"].detach().cpu().numpy()
    pts, var_pct = pca_2d(emb)
    points = [{"char": itos[i], "x": round(float(pts[i, 0]), 4), "y": round(float(pts[i, 1]), 4)}
              for i in range(len(itos))]

    out = {
        "kind": "pi-of-ai:scratch-inspect",
        "version": 1,
        "config": amap["config"],
        "totalParams": amap["total_params"],
        "checkpoint": {
            # null, not 0 — train.py saves neither, and a fabricated "iter 0, val
            # 0.000" reads as a perfectly-trained model. The page omits nulls.
            "path": args.ckpt.name,
            "iter": (int(ck["iter"]) if "iter" in ck else None),
            "valLoss": (float(ck["best"]) if "best" in ck else None),
            "sizeBytes": int(args.ckpt.stat().st_size),
        },
        "arch": amap,
        "trained": trained,
        "random": random_stats,
        "embedding": {"points": points, "neighbours": neighbours(emb, itos),
                      "varianceExplainedPct": round(var_pct, 1)},
        "unverifiable": [
            "whether a 2D projection preserves the neighbourhoods it appears to show",
            "what a weight statistic means for behaviour — these are shapes, not explanations",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=True)
    print(f"{out['kind']}  {cfg['n_layer']} blocks, {amap['total_params']:,} params, {len(names)} tensors")
    print(f"  embedding points: {len(points)} · variance explained: {var_pct:.1f}%")
    print(f"  trained/random keys match: {set(trained) == set(random_stats)}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
