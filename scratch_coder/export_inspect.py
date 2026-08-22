"""
export_inspect.py — turn a scratch_coder checkpoint into JSON the web page reads.

The join arch_map.py was written to expect. This side has torch and a checkpoint;
the page never sees a .pt (a pickle is remote code execution waiting to happen).

Writes:
  * web/inspect.json  — per-tensor statistics (never the weights), the same for a
                        freshly-random model of the same shape (trained-vs-random),
                        and the token embedding projected to 2D.
  * web/weights.bin   — with --weights: fp32 little-endian, tensors concatenated in
                        arch_map order, offsets recorded in inspect.json. The
                        attention rung's forward pass reads this.
  * <probe>.json      — with --attn-probe TEXT: the real model's attention for one
                        prompt, the reference the JS forward pass is checked against.

    python export_inspect.py                       # inspect.json for data/ckpt.pt
    python export_inspect.py --weights             # + weights.bin for the attention rung
    python export_inspect.py --attn-probe "def hello(" --out /tmp/probe.json

The shape always comes from the checkpoint's own cfg, NOT arch_map's defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import zlib
from pathlib import Path

import numpy as np
import torch

from arch_map import tensor_map
from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
OUT = HERE / "web" / "inspect.json"
WEIGHTS = HERE / "web" / "weights.bin"


def tensor_stats(arr: np.ndarray) -> dict:
    """One number per cell for the Knob Matrix — nothing larger belongs here."""
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


def pca_3d(mat: np.ndarray, iters: int = 300):
    """Project rows to 3D for the orbitable graph. Power iteration on the covariance,
    top-3 components by deflation — about fifteen lines and one fewer dependency than
    sklearn. Returns the points and the percent of total variance the 3 axes carry."""
    x = mat - mat.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / x.shape[0]
    total_var = float(np.trace(cov))
    rng = np.random.default_rng(0)
    comps, evals, c = [], [], cov.copy()
    for _ in range(3):
        v = rng.standard_normal(c.shape[0]); v /= np.linalg.norm(v)
        for _ in range(iters):
            v = c @ v
            nrm = np.linalg.norm(v)
            if nrm == 0:
                break
            v /= nrm
        lam = float(v @ c @ v)
        comps.append(v); evals.append(lam)
        c = c - lam * np.outer(v, v)
    pts = x @ np.stack(comps, axis=1)
    var_pct = 100.0 * (sum(evals) / total_var) if total_var > 0 else 0.0
    return pts, var_pct


def neighbours(emb: np.ndarray, itos: list[str], k: int = 6) -> dict:
    """Top-k cosine neighbours per character, in the FULL space — the check on the
    2D picture. When the plot and this list disagree, this is right."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    out = {}
    for i in range(emb.shape[0]):
        idx = np.argsort(-sim[i])[:k]
        out[itos[i]] = [{"char": itos[int(j)], "cos": round(float(sim[i, j]), 3)} for j in idx]
    return out


def ideal_neighbours(itos: list[str], corpus: Path, k: int = 6) -> dict:
    """What the neighbours WOULD be for a perfectly trained model — read from the data,
    not the weights. A char model is trained only to predict the next character, so it
    is pushed to place characters that appear in the same contexts near each other. We
    take that target straight from the corpus: each character's profile is how often
    every other character sits immediately before and immediately after it, and the
    cosine between two profiles is how interchangeably the text uses them. That
    distribution is the ideal the embedding is climbing toward, so these are the
    neighbours a fully trained model should recover — shown beside what it has so far."""
    ids = np.fromfile(corpus, dtype=np.uint16).astype(np.int64)
    V = len(itos)
    left = np.zeros((V, V)); right = np.zeros((V, V))
    a, b = ids[:-1], ids[1:]
    np.add.at(right, (a, b), 1.0)   # b follows a
    np.add.at(left,  (b, a), 1.0)   # a precedes b
    profile = np.concatenate([left, right], axis=1)          # each char: [who precedes | who follows]
    norm = profile / (np.linalg.norm(profile, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    out = {}
    for i in range(V):
        idx = np.argsort(-sim[i])[:k]
        out[itos[i]] = [{"char": itos[int(j)], "cos": round(float(sim[i, j]), 3)} for j in idx]
    return out


def read_loss_log(path: Path) -> list:
    """The loss + per-layer gradient-norm history the trainers append (loss.jsonl).
    Rides along in inspect.json because the page is served from web/ only and cannot
    reach the data dir. A half-written last line during a live run is skipped, not
    fatal."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def arch_from_state_dict(sd: dict, cfg: dict) -> dict:
    """Build the tensor map from an actual checkpoint — for a custom layer that
    arch_map cannot describe. Real tensor names and shapes, grouped the same way
    (embedding, each block, output) so the Knob Matrix draws exactly what is there."""
    import re

    emb = {"name": "embedding", "kind": "embedding", "tensors": []}
    out = {"name": "output", "kind": "output", "tensors": []}
    blocks: dict = {}
    for name, t in sd.items():
        entry = {"name": name, "shape": list(t.shape), "params": int(t.numel()), "role": "custom layer tensor"}
        m = re.match(r"blocks\.(\d+)\.", name)
        if m:
            i = int(m.group(1))
            blocks.setdefault(i, {"name": f"block.{i}", "kind": "block", "index": i, "tensors": []})
            blocks[i]["tensors"].append(entry)
        elif name.startswith(("tok_emb", "pos_emb")):
            emb["tensors"].append(entry)
        else:
            out["tensors"].append(entry)
    groups = [emb] + [blocks[i] for i in sorted(blocks)] + [out]
    total = 0
    for g in groups:
        g["params"] = sum(e["params"] for e in g["tensors"])
        total += g["params"]
    return {
        "config": {"vocab_size": cfg["vocab_size"], "block_size": cfg["block_size"],
                   "n_layer": cfg["n_layer"], "n_head": cfg["n_head"], "n_embd": cfg["n_embd"],
                   "head_dim": cfg["n_embd"] // cfg["n_head"]},
        "total_params": total, "groups": groups,
        "buffers": [{"name": "blocks.*.attn.mask",
                     "shape": [1, 1, cfg["block_size"], cfg["block_size"]],
                     "per_block_elements": cfg["block_size"] ** 2,
                     "role": "lower-triangular causal mask — a fixed buffer, not learned"}],
    }


def write_weights(sd: dict, names: list[str]) -> dict:
    """fp32 little-endian, tensors concatenated in arch_map order. The manifest
    (offsets in floats) plus a crc32 go into inspect.json so gpt.js can refuse a
    weights.bin that does not belong to it."""
    manifest, offset, chunks = [], 0, []
    for n in names:
        a = sd[n].detach().to(torch.float32).contiguous().cpu().numpy().ravel().astype("<f4")
        chunks.append(a.tobytes())
        manifest.append({"name": n, "offset": offset, "count": int(a.size)})
        offset += int(a.size)
    blob = b"".join(chunks)
    WEIGHTS.write_bytes(blob)
    return {"file": "weights.bin", "dtype": "float32-le", "totalFloats": offset,
            "byteLength": len(blob), "crc32": zlib.crc32(blob) & 0xffffffff, "tensors": manifest}


def attn_probe(sd: dict, cfg: dict, itos: dict, prompt: str) -> dict:
    """The real model's attention for `prompt`, the reference gpt.js is checked
    against. CausalSelfAttention.forward is monkeypatched with a copy of its own
    math that stashes `att` — and a logits-drift check confirms the copy matches
    the unpatched model, so the reference is model.py's behaviour, not a fork."""
    import model as M
    net = GPT(GPTConfig(**cfg)); net.load_state_dict(sd); net.eval()
    stoi = {itos[i]: i for i in range(cfg["vocab_size"])}
    ids = [stoi.get(c, 0) for c in prompt][:cfg["block_size"]]
    xb = torch.tensor([ids], dtype=torch.long)

    captured, orig = [], M.CausalSelfAttention.forward

    def patched(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        captured.append(att[0].detach().cpu().numpy().tolist())   # [n_head, T, T]
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    with torch.no_grad():
        logits_ref, _ = net(xb)                # unpatched = the source of truth
        M.CausalSelfAttention.forward = patched
        try:
            logits_patched, _ = net(xb)
        finally:
            M.CausalSelfAttention.forward = orig
    drift = float((logits_patched - logits_ref).abs().max())

    return {"prompt": prompt, "ids": ids, "chars": [itos[i] for i in ids],
            "config": {k: cfg[k] for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")},
            "attn": captured, "logitsDrift": drift}


SRC_FILES = ["model.py", "train.py", "prepare_data.py", "sample_big.py"]


def source_files() -> list:
    """The Python the model is actually written and trained in — the from-scratch GPT,
    the training loop, how the corpus is built, and how it samples. Shipped so the page
    can show the real implementation next to the weights it produced. Small; embedded."""
    out = []
    for name in SRC_FILES:
        p = HERE / name
        if p.exists():
            out.append({"name": name, "code": p.read_text(encoding="utf-8")})
    return out


def write_corpus(train_bin: Path, itos: dict, web_dir: Path) -> dict:
    """Decode the training split back to text and drop it beside the page (corpus.txt),
    so the training-data view can show the exact characters the model learned from. It
    is regenerable and large (~1.8 MB), so it is gitignored, like weights.bin."""
    ids = np.fromfile(train_bin, dtype=np.uint16)
    text = "".join(itos[int(i)] for i in ids)
    (web_dir / "corpus.txt").write_text(text, encoding="utf-8", newline="")
    return {"file": "corpus.txt", "chars": len(text)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a checkpoint as something a page can read.")
    ap.add_argument("--ckpt", type=Path, default=HERE / "data" / "ckpt.pt")
    ap.add_argument("--weights", action="store_true", help="also write web/weights.bin")
    ap.add_argument("--attn-probe", metavar="TEXT", help="write an attention reference for a prompt, then exit")
    ap.add_argument("--out", type=Path, help="output path for --attn-probe (default web/probe.json)")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"no checkpoint at {args.ckpt} — train one first (python train.py)")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = ck["cfg"], ck["model"]
    meta = pickle.load(open(args.ckpt.parent / "meta.pkl", "rb"))
    itos = meta["itos"]

    if args.attn_probe is not None:
        probe = attn_probe(sd, cfg, itos, args.attn_probe)
        out_path = args.out or (HERE / "web" / "probe.json")
        json.dump(probe, open(out_path, "w", encoding="utf-8"), ensure_ascii=True)
        print(f"attn probe {args.attn_probe!r} -> {out_path}")
        print(f"  {len(probe['ids'])} tokens · logits drift patched-vs-real: {probe['logitsDrift']:.2e}")
        return 0

    mlp = cfg.get("mlp", "default")
    if mlp == "default":
        amap = tensor_map(cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"])
        custom = None
    else:
        # arch_map cannot describe a custom MLP; take the real tensors from the ckpt.
        amap = arch_from_state_dict(sd, cfg)
        custom = {"mlp": mlp, "note": f"arch_map cannot describe the custom MLP {mlp!r}; shapes taken from the checkpoint"}
    names = [t["name"] for g in amap["groups"] for t in g["tensors"]]

    trained = stats_for(sd, names)
    torch.manual_seed(1337)
    random_stats = stats_for(GPT(GPTConfig(**cfg)).state_dict(), names)

    itos_list = [itos[i] for i in range(cfg["vocab_size"])]
    emb = sd["tok_emb.weight"].detach().cpu().numpy()
    pts, var_pct = pca_3d(emb)
    points = [{"char": itos_list[i], "x": round(float(pts[i, 0]), 4),
               "y": round(float(pts[i, 1]), 4), "z": round(float(pts[i, 2]), 4)}
              for i in range(len(itos_list))]

    out = {
        "kind": "pi-of-ai:scratch-inspect",
        "version": 1,
        "config": amap["config"],
        "totalParams": amap["total_params"],
        "customLayer": custom,
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
        "embedding": {"points": points, "neighbours": neighbours(emb, itos_list),
                      "idealNeighbours": ideal_neighbours(itos_list, args.ckpt.parent / "train.bin"),
                      "varianceExplainedPct": round(var_pct, 1)},
        "training": read_loss_log(args.ckpt.parent / "loss.jsonl"),
        "source": source_files(),
        "corpus": write_corpus(args.ckpt.parent / "train.bin", itos, OUT.parent),
        "unverifiable": [
            "whether a 2D projection preserves the neighbourhoods it appears to show",
            "what a weight statistic means for behaviour — these are shapes, not explanations",
        ],
    }
    if args.weights:
        out["weights"] = write_weights(sd, names)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=True)
    print(f"{out['kind']}  {cfg['n_layer']} blocks, {amap['total_params']:,} params, {len(names)} tensors")
    print(f"  embedding points: {len(points)} · variance explained: {var_pct:.1f}%")
    if args.weights:
        print(f"  weights.bin: {out['weights']['byteLength'] / 1e6:.2f} MB ({out['weights']['totalFloats']:,} floats)")
    print(f"  -> {OUT}")

    # The rules_baker app embeds a same-origin copy of the inspector (a section in
    # its sidebar). Mirror the generated files there so it stays in sync. The user
    # chose to merge the two builds; this keeps that copy honest without a second
    # export command to remember.
    mirror = HERE.parent / "rules_baker" / "web" / "scratch"
    if mirror.is_dir():
        import shutil
        shutil.copy2(OUT, mirror / "inspect.json")
        if args.weights and WEIGHTS.exists():
            shutil.copy2(WEIGHTS, mirror / "weights.bin")
        corpus_txt = OUT.parent / "corpus.txt"
        if corpus_txt.exists():
            shutil.copy2(corpus_txt, mirror / "corpus.txt")
        print(f"  mirrored -> {mirror}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
