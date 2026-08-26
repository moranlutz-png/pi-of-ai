"""De-risk Rung 3: quantize the 28.5M model to group-wise INT4 (GGUF Q4-style, 32-weight
groups, one scale each) and generate. If it still reads like Python, INT4 is viable for
the board; then we build the binary export + flash loader. Reads the trained checkpoint."""
import pickle, numpy as np, torch, math
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data_esp32"
ck = torch.load(D / "ckpt_esp32.pt", map_location="cpu", weights_only=False)
sd = ck["model"]; cfg = ck["cfg"]
V, C, L, H, BLK = cfg["vocab_size"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["block_size"]
HD = C // H
meta = pickle.load(open(D / "meta.pkl", "rb")); itos = meta["itos"]; stoi = meta["stoi"]
g = lambda n: sd[n].detach().numpy().astype(np.float32).ravel()

def qi4(w, G=32):   # group-wise int4 symmetric, returns dequantized (what the board reconstructs)
    n = len(w); pad = (-n) % G
    wp = np.concatenate([w, np.zeros(pad, np.float32)]).reshape(-1, G)
    amax = np.max(np.abs(wp), axis=1, keepdims=True); amax[amax == 0] = 1e-8
    s = amax / 7.0
    deq = (np.clip(np.round(wp / s), -7, 7) * s).reshape(-1)[:n]
    return deq

W, worst = {}, 0.0
for n in sd:
    w = g(n)
    if n.endswith((".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight")):
        d = qi4(w); worst = max(worst, float(np.max(np.abs(w - d)) / (np.max(np.abs(w)) or 1e-8)))
        W[n] = d
    else:
        W[n] = w

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
        q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]; kc[l][t] = k; vc[l][t] = v
        y = np.zeros(C, np.float32)
        for hh in range(H):
            o = hh * HD
            sc = (kc[l][:t + 1, o:o + HD] @ q[o:o + HD]) / np.sqrt(HD)
            sc = np.exp(sc - sc.max()); sc /= sc.sum(); y[o:o + HD] = sc @ vc[l][:t + 1, o:o + HD]
        x = x + lin(y, W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"], C, C)
        h = ln(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        m = lin(h, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"], C, 4 * C)
        m = 0.5 * m * (1 + np.vectorize(lambda z: math.erf(z / 2 ** 0.5))(m))
        x = x + lin(m, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"], 4 * C, C)
    x = ln(x, W["ln_f.weight"], W["ln_f.bias"])
    return lin(x, W["head.weight"], None, C, V)

print(f"28.5M model | worst int4 group rel error: {worst*100:.1f}%")
rng = np.random.default_rng(0); prompt = "def "; t = 0; logits = None
for c in prompt: logits = step(stoi[c], t); t += 1
out = prompt
for _ in range(60):
    if t >= BLK: break
    p = np.exp((logits - logits.max()) / 0.7); p /= p.sum()
    nxt = int(rng.choice(V, p=p)); out += itos[nxt]; logits = step(nxt, t); t += 1
print("=== INT4 28.5M generating from 'def ' ===\n" + out)
