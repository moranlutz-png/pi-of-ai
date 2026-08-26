"""Validate the ESP32 approach on the PC: run the SAME int8 weight-only forward pass
(same quantization + maths as pi_gpt.ino) and generate. If this reads like Python, the
quantizer and the on-device algorithm are correct; the C port does identical maths."""
import json, numpy as np
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "web" / "tiers" / "featherweight"
insp = json.load(open(T / "inspect.json"))
cfg = insp["config"]; V, C, L, H, BLK = cfg["vocab_size"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["block_size"]
HD = C // H
blob = np.fromfile(T / "weights.bin", dtype="<f4")
man = {t["name"]: (t["offset"], t["count"]) for t in insp["weights"]["tensors"]}
g = lambda n: blob[man[n][0]:man[n][0] + man[n][1]].astype(np.float32)
chars = [p["char"] for p in insp["embedding"]["points"]]; stoi = {c: i for i, c in enumerate(chars)}

def qi8(w):  # symmetric per-tensor int8, then dequant (exactly what the board stores/uses)
    s = (np.max(np.abs(w)) or 1e-8) / 127.0
    return np.clip(np.round(w / s), -127, 127).astype(np.int8).astype(np.float32) * s

# int8 the block matmul weights (dequantized copies), float the rest — mirrors quantize.py
W = {n: g(n) for n in man}
for l in range(L):
    for suf in ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"):
        W[f"blocks.{l}.{suf}"] = qi8(g(f"blocks.{l}.{suf}"))

def ln(x, w, b):
    m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + 1e-5) * w + b
def lin(x, w, b, inD, outD):
    return (w.reshape(outD, inD) @ x) + (b if b is not None else 0)

kc = [np.zeros((BLK, C), np.float32) for _ in range(L)]; vc = [np.zeros((BLK, C), np.float32) for _ in range(L)]
def step(tok, t):
    x = W["tok_emb.weight"].reshape(V, C)[tok] + W["pos_emb.weight"].reshape(BLK, C)[t]
    for l in range(L):
        p = f"blocks.{l}."
        h = ln(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
        qkv = lin(h, W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"], C, 3 * C)
        q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]
        kc[l][t] = k; vc[l][t] = v
        y = np.zeros(C, np.float32)
        for hh in range(H):
            o = hh * HD
            sc = (kc[l][:t + 1, o:o + HD] @ q[o:o + HD]) / np.sqrt(HD)
            sc = np.exp(sc - sc.max()); sc /= sc.sum()
            y[o:o + HD] = sc @ vc[l][:t + 1, o:o + HD]
        x = x + lin(y, W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"], C, C)
        h = ln(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        m = lin(h, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"], C, 4 * C)
        m = 0.5 * m * (1 + np.vectorize(lambda z: __import__("math").erf(z / 2 ** 0.5))(m))
        x = x + lin(m, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"], 4 * C, C)
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return lin(x, W["head.weight"], None, C, V)

rng = np.random.default_rng(0)
prompt = "def "; ids = [stoi[c] for c in prompt]; t = 0; logits = None
for c in ids: logits = step(stoi[chars[c]] if False else c, t); t += 1
out = prompt
for _ in range(180):
    if t >= BLK: break
    p = np.exp((logits - logits.max()) / 0.8); p /= p.sum()
    nxt = int(rng.choice(V, p=p)); out += chars[nxt]; logits = step(nxt, t); t += 1
print("=== INT8 forward (what the ESP32 will run) generating from 'def ' ===")
print(out)
