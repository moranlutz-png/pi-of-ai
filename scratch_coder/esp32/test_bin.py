"""Read model.bin with the SAME parsing pi_gpt_big.ino will use, run the forward, and
generate — so the binary format is proven correct before writing any C."""
import struct, numpy as np, math
from pathlib import Path

data = (Path(__file__).resolve().parent / "model.bin").read_bytes()
off = 0
def a4():
    global off
    off = (off + 3) & ~3
def i32():
    global off; v = struct.unpack_from("<i", data, off)[0]; off += 4; return v
def f32(n):
    global off; a4(); v = np.frombuffer(data, "<f4", n, off).copy(); off += 4 * n; return v
def i4():
    global off; a4(); N = i32(); ng = i32(); a4()
    scales = np.frombuffer(data, "<f4", ng, off).copy(); off += 4 * ng; a4()
    nb = (N + 1) // 2; packed = np.frombuffer(data, np.uint8, nb, off).copy(); off += nb
    lo = packed & 0x0F; hi = (packed >> 4) & 0x0F
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8); hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    q = np.empty(N, np.float32); q[0::2] = lo[:len(q[0::2])]; q[1::2] = hi[:len(q[1::2])]
    deq = np.empty(N, np.float32)
    for gi in range(ng): s = scales[gi]; b = gi * GROUP; deq[b:b + GROUP] = q[b:b + GROUP] * s
    return deq

assert data[:4] == b"PIG4"; off = 4
ver, L, H, C, BLK, V, GROUP = (i32() for _ in range(7))
chars = [chr(b) for b in data[off:off + V]]; off += V; a4()
W = {}
W["tok_emb.weight"] = f32(V * C); W["pos_emb.weight"] = f32(BLK * C)
W["ln_f.weight"] = f32(C); W["ln_f.bias"] = f32(C); W["head.weight"] = f32(V * C)
for l in range(L):
    p = f"blocks.{l}."
    W[p + "ln1.weight"] = f32(C); W[p + "ln1.bias"] = f32(C)
    for suf, outD in (("attn.c_attn", 3 * C), ("attn.c_proj", C), ("mlp.c_fc", 4 * C), ("mlp.c_proj", C)):
        W[p + suf + ".weight"] = i4(); W[p + suf + ".bias"] = f32(outD)
    W[p + "ln2.weight"] = f32(C); W[p + "ln2.bias"] = f32(C)
print(f"parsed model.bin: {L}L/{H}H/{C}d block {BLK} vocab {V} group {GROUP} | consumed {off}/{len(data)} bytes")

HD = C // H; stoi = {c: i for i, c in enumerate(chars)}
def ln(x, w, b): m = x.mean(); v = ((x - m) ** 2).mean(); return (x - m) / np.sqrt(v + 1e-5) * w + b
def lin(x, w, b, inD, outD): return (w.reshape(outD, inD) @ x) + (b if b is not None else 0)
kc = [np.zeros((BLK, C), np.float32) for _ in range(L)]; vc = [np.zeros((BLK, C), np.float32) for _ in range(L)]
def step(tok, t):
    x = W["tok_emb.weight"].reshape(V, C)[tok] + W["pos_emb.weight"].reshape(BLK, C)[t]
    for l in range(L):
        p = f"blocks.{l}."
        h = ln(x, W[p + "ln1.weight"], W[p + "ln1.bias"]); qkv = lin(h, W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"], C, 3 * C)
        q, k, v = qkv[:C], qkv[C:2 * C], qkv[2 * C:]; kc[l][t] = k; vc[l][t] = v; y = np.zeros(C, np.float32)
        for hh in range(H):
            o = hh * HD; sc = (kc[l][:t + 1, o:o + HD] @ q[o:o + HD]) / np.sqrt(HD)
            sc = np.exp(sc - sc.max()); sc /= sc.sum(); y[o:o + HD] = sc @ vc[l][:t + 1, o:o + HD]
        x = x + lin(y, W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"], C, C)
        h = ln(x, W[p + "ln2.weight"], W[p + "ln2.bias"]); m = lin(h, W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"], C, 4 * C)
        m = 0.5 * m * (1 + np.vectorize(lambda z: math.erf(z / 2 ** 0.5))(m))
        x = x + lin(m, W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"], 4 * C, C)
    return lin(ln(x, W["ln_f.weight"], W["ln_f.bias"]), W["head.weight"], None, C, V)

rng = np.random.default_rng(0); prompt = "def "; t = 0; logits = None
for c in prompt: logits = step(stoi[c], t); t += 1
out = prompt
for _ in range(60):
    if t >= BLK: break
    p = np.exp((logits - logits.max()) / 0.7); p /= p.sum(); nxt = int(rng.choice(V, p=p)); out += chars[nxt]; logits = step(nxt, t); t += 1
print("=== from model.bin (what the board will run) ===\n" + out)
