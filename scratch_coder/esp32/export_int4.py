"""
export_int4.py — pack a trained tier into model.bin: group-wise INT4 for the block
matmul weights, FP32 for the small precision-sensitive tensors. This binary is flashed
to a dedicated partition and memory-mapped by pi_gpt_big.ino (too big to compile in).

    python export_int4.py --tier ultra          # -> esp32/model.bin

Format (little-endian, every FP32 section 4-byte aligned so mmap reads stay aligned):
  header: char[4]"PIG4"; int32 version, n_layer, n_head, n_embd, block, vocab, group
  uint8 vocab_chars[vocab]                        (padded to 4)
  FP32 tok_emb[V*C], pos_emb[BLOCK*C], ln_f_w[C], ln_f_b[C], head_w[V*C]
  per layer: FP32 ln1_w[C], ln1_b[C]
             {c_attn, c_proj, fc, proj}: int4 block then FP32 bias[outD]
             FP32 ln2_w[C], ln2_b[C]
  int4 block = int32 N, int32 n_groups, FP32 scales[n_groups], uint8 packed[(N+1)/2] (pad 4)
  packed: two signed-4bit (two's complement, clipped -7..7) per byte, low nibble first.
"""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="ultra")
ap.add_argument("--group", type=int, default=32)
ap.add_argument("--out", type=Path, default=HERE / "model.bin")
a = ap.parse_args()

T = HERE.parent / "web" / "tiers" / a.tier
insp = json.loads((T / "inspect.json").read_text(encoding="utf-8"))
cfg = insp["config"]; V, C, L, H, BLK = cfg["vocab_size"], cfg["n_embd"], cfg["n_layer"], cfg["n_head"], cfg["block_size"]
blob = np.fromfile(T / "weights.bin", dtype="<f4")
man = {t["name"]: (t["offset"], t["count"]) for t in insp["weights"]["tensors"]}
g = lambda n: blob[man[n][0]:man[n][0] + man[n][1]].astype(np.float32)
chars = [p["char"] for p in insp["embedding"]["points"]]

buf = bytearray()
def align4():
    while len(buf) % 4: buf.append(0)
def put_f32(arr): align4(); buf.extend(np.asarray(arr, dtype="<f4").tobytes())
def put_i32(v): buf.extend(struct.pack("<i", v))
def put_i4(w):
    w = np.asarray(w, dtype=np.float32).ravel(); N = len(w); G = a.group
    pad = (-N) % G
    wp = np.concatenate([w, np.zeros(pad, np.float32)]).reshape(-1, G)
    amax = np.max(np.abs(wp), axis=1, keepdims=True); amax[amax == 0] = 1e-8
    scales = (amax / 7.0).astype(np.float32).ravel()
    q = np.clip(np.round(wp / (amax / 7.0)), -7, 7).astype(np.int8).ravel()[:N]
    align4(); put_i32(N); put_i32(len(scales))
    align4(); buf.extend(scales.tobytes())
    lo = (q[0::2] & 0x0F)
    hi = (q[1::2] & 0x0F) if N % 2 == 0 else np.append(q[1::2] & 0x0F, 0)
    packed = (lo | (hi << 4)).astype(np.uint8)
    align4(); buf.extend(packed.tobytes())

# header
buf.extend(b"PIG4")
for v in (1, L, H, C, BLK, V, a.group): put_i32(v)
buf.extend(bytes(ord(c) for c in chars)); align4()
# fp32 top-level
put_f32(g("tok_emb.weight")); put_f32(g("pos_emb.weight"))
put_f32(g("ln_f.weight")); put_f32(g("ln_f.bias")); put_f32(g("head.weight"))
# layers
for l in range(L):
    p = f"blocks.{l}."
    put_f32(g(p + "ln1.weight")); put_f32(g(p + "ln1.bias"))
    for suf in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
        put_i4(g(p + suf + ".weight")); put_f32(g(p + suf + ".bias"))
    put_f32(g(p + "ln2.weight")); put_f32(g(p + "ln2.bias"))

a.out.write_bytes(buf)
mb = len(buf) / 1048576
print(f"tier {a.tier}: {L}L/{H}H/{C}d block {BLK} vocab {V}, int4 group {a.group}")
print(f"wrote {a.out}  ({mb:.2f} MB)")
