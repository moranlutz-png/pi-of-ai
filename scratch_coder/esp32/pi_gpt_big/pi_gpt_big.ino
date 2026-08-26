// pi_gpt_big.ino — a bigger Scratch-Coder tier (e.g. Ultra 14.3M) on the ESP32-S3,
// with the model too big to compile in: it's flashed to a "model" partition and
// MEMORY-MAPPED here, weights read straight from flash (INT4, group-wise). Same
// gpt.js forward pass as pi_gpt.ino; activations + KV cache live in PSRAM.
//
// Setup:
//   1. Copy partitions.csv into this sketch folder (it's here already).
//   2. Tools: ESP32S3 Dev Module | PSRAM: OPI | USB CDC On Boot: Enabled |
//      Flash: 16MB | Partition Scheme: "Custom" (uses partitions.csv).
//   3. Upload this sketch (the app).
//   4. Flash the weights to the model partition (offset 0x310000) — from a terminal:
//        py -m pip install esptool
//        py -m esptool --chip esp32s3 --port COM9 write_flash 0x310000 ..\model.bin
//   5. Open Serial Monitor @115200, type a prompt, watch it generate (slow — it's big).

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "esp_partition.h"

#define MAXL 32
static const uint8_t *BASE, *P;
static int NL, NH, NE, BK, VC, GR, HD;
static const char *VOCABC;
static const float *tok_emb, *pos_emb, *ln_f_w, *ln_f_b, *head_w;
struct I4 { const float *scales; const uint8_t *packed; };
static struct { const float *ln1_w, *ln1_b, *cattn_b, *cproj_b, *fc_b, *proj_b, *ln2_w, *ln2_b;
                I4 cattn, cproj, fc, proj; } LY[MAXL];

static void al() { size_t o = (size_t)(P - BASE); o = (o + 3) & ~((size_t)3); P = BASE + o; }
static int32_t ri() { int32_t v; memcpy(&v, P, 4); P += 4; return v; }
static const float *rf(int n) { al(); const float *p = (const float*)P; P += 4 * (size_t)n; return p; }
static I4 r4() { al(); int N = ri(); int ng = ri(); al(); const float *sc = (const float*)P; P += 4 * (size_t)ng;
                 al(); const uint8_t *pk = P; P += (N + 1) / 2; return I4{sc, pk}; }

static void loadModel(const uint8_t *base) {
  BASE = base; P = base;
  P += 4;                                    // magic "PIG4"
  ri(); NL = ri(); NH = ri(); NE = ri(); BK = ri(); VC = ri(); GR = ri(); HD = NE / NH;
  VOCABC = (const char*)P; P += VC; al();
  tok_emb = rf(VC * NE); pos_emb = rf(BK * NE); ln_f_w = rf(NE); ln_f_b = rf(NE); head_w = rf(VC * NE);
  for (int l = 0; l < NL; l++) {
    LY[l].ln1_w = rf(NE); LY[l].ln1_b = rf(NE);
    LY[l].cattn = r4(); LY[l].cattn_b = rf(3 * NE);
    LY[l].cproj = r4(); LY[l].cproj_b = rf(NE);
    LY[l].fc = r4();    LY[l].fc_b = rf(4 * NE);
    LY[l].proj = r4();  LY[l].proj_b = rf(NE);
    LY[l].ln2_w = rf(NE); LY[l].ln2_b = rf(NE);
  }
}

// buffers (PSRAM)
static float *x, *h, *qkv, *att_y, *scores, *m1, *logits;
static float *kc[MAXL], *vc[MAXL];
static int cache_pos = 0;

static int stoi(char c) { for (int i = 0; i < VC; i++) if (VOCABC[i] == c) return i; return 0; }
static void layernorm(const float *in, const float *w, const float *b, float *out) {
  float m = 0; for (int i = 0; i < NE; i++) m += in[i]; m /= NE;
  float v = 0; for (int i = 0; i < NE; i++) { float d = in[i] - m; v += d * d; } v /= NE;
  float inv = 1.0f / sqrtf(v + 1e-5f);
  for (int i = 0; i < NE; i++) out[i] = (in[i] - m) * inv * w[i] + b[i];
}
static void linear_f(const float *in, const float *W, const float *b, int inD, int outD, float *out) {
  for (int o = 0; o < outD; o++) { float s = b ? b[o] : 0; const float *wr = W + (size_t)o * inD;
    for (int i = 0; i < inD; i++) s += in[i] * wr[i]; out[o] = s; }
}
static inline float deq4(const I4 &w, int idx) {
  float s = w.scales[idx / GR]; uint8_t by = w.packed[idx >> 1];
  int nib = (idx & 1) ? (by >> 4) : (by & 0x0F); if (nib & 0x08) nib -= 16; return nib * s;
}
static void linear_i4(const float *in, const I4 &w, const float *b, int inD, int outD, float *out) {
  for (int o = 0; o < outD; o++) { float s = 0; int base = o * inD;
    for (int i = 0; i < inD; i++) s += in[i] * deq4(w, base + i); out[o] = s + (b ? b[o] : 0); }
}
static inline float geluf(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678f)); }

static void step(int tok) {
  int t = cache_pos;
  for (int i = 0; i < NE; i++) x[i] = tok_emb[tok * NE + i] + pos_emb[t * NE + i];
  float sc = 1.0f / sqrtf((float)HD);
  for (int l = 0; l < NL; l++) {
    layernorm(x, LY[l].ln1_w, LY[l].ln1_b, h);
    linear_i4(h, LY[l].cattn, LY[l].cattn_b, NE, 3 * NE, qkv);
    float *q = qkv, *k = qkv + NE, *v = qkv + 2 * NE;
    for (int i = 0; i < NE; i++) { kc[l][t * NE + i] = k[i]; vc[l][t * NE + i] = v[i]; }
    for (int i = 0; i < NE; i++) att_y[i] = 0;
    for (int hh = 0; hh < NH; hh++) {
      int off = hh * HD; float mx = -1e30f;
      for (int j = 0; j <= t; j++) { float d = 0; const float *kj = kc[l] + j * NE + off;
        for (int dd = 0; dd < HD; dd++) d += q[off + dd] * kj[dd]; d *= sc; scores[j] = d; if (d > mx) mx = d; }
      float sum = 0; for (int j = 0; j <= t; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
      for (int j = 0; j <= t; j++) { float wg = scores[j] / sum; const float *vj = vc[l] + j * NE + off;
        for (int dd = 0; dd < HD; dd++) att_y[off + dd] += wg * vj[dd]; }
    }
    linear_i4(att_y, LY[l].cproj, LY[l].cproj_b, NE, NE, h);
    for (int i = 0; i < NE; i++) x[i] += h[i];
    layernorm(x, LY[l].ln2_w, LY[l].ln2_b, h);
    linear_i4(h, LY[l].fc, LY[l].fc_b, NE, 4 * NE, m1);
    for (int i = 0; i < 4 * NE; i++) m1[i] = geluf(m1[i]);
    linear_i4(m1, LY[l].proj, LY[l].proj_b, 4 * NE, NE, h);
    for (int i = 0; i < NE; i++) x[i] += h[i];
  }
  layernorm(x, ln_f_w, ln_f_b, h);
  linear_f(h, head_w, nullptr, NE, VC, logits);
  cache_pos = t + 1;
}
static int sample(float temp) {
  float mx = -1e30f; for (int i = 0; i < VC; i++) if (logits[i] > mx) mx = logits[i];
  float sum = 0; for (int i = 0; i < VC; i++) { logits[i] = expf((logits[i] - mx) / temp); sum += logits[i]; }
  float r = (float)esp_random() / (float)UINT32_MAX * sum;
  for (int i = 0; i < VC; i++) { r -= logits[i]; if (r <= 0) return i; }
  return VC - 1;
}
static void generate(const String &prompt, int maxNew, float temp) {
  cache_pos = 0; unsigned long t0 = millis(); int n = 0; Serial.print("> ");
  for (int i = 0; i < (int)prompt.length() && cache_pos < BK; i++) { Serial.write(prompt[i]); step(stoi(prompt[i])); }
  for (; n < maxNew && cache_pos < BK; n++) { int nx = sample(temp); Serial.write((char)VOCABC[nx]); step(nx); }
  float s = (millis() - t0) / 1000.0f;
  Serial.printf("\n(%d chars in %.1fs = %.2f char/s)\ntype another prompt:\n", n, s, n / (s + 1e-6f));
}

void setup() {
  Serial.begin(115200); delay(1500);
  Serial.println("\n=== Pi-of-AI: a bigger model, streamed from flash ===");
  const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("ERROR: 'model' partition not found — is Partition Scheme set to Custom?"); return; }
  const void *mapped; esp_partition_mmap_handle_t hmap;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped, &hmap) != ESP_OK) {
    Serial.println("ERROR: mmap failed"); return; }
  if (memcmp(mapped, "PIG4", 4) != 0) { Serial.println("ERROR: no model in partition — flash model.bin to 0x310000 (see header)."); return; }
  loadModel((const uint8_t*)mapped);
  Serial.printf("model: %dL %dH %dd, block %d, vocab %d, int4 group %d\n", NL, NH, NE, BK, VC, GR);
  Serial.printf("PSRAM free: %.2f MB\n", ESP.getFreePsram() / 1048576.0);
  x = (float*)ps_malloc(NE * 4); h = (float*)ps_malloc(4 * NE * 4); qkv = (float*)ps_malloc(3 * NE * 4);
  att_y = (float*)ps_malloc(NE * 4); scores = (float*)ps_malloc(BK * 4); m1 = (float*)ps_malloc(4 * NE * 4);
  logits = (float*)ps_malloc(VC * 4);
  for (int l = 0; l < NL; l++) { kc[l] = (float*)ps_malloc((size_t)BK * NE * 4); vc[l] = (float*)ps_malloc((size_t)BK * NE * 4); }
  if (!logits || !vc[NL - 1]) { Serial.println("ERROR: PSRAM alloc failed"); return; }
  Serial.println("ready. type a prompt (e.g.  def   ) and press Enter:");
}
void loop() {
  if (Serial.available()) { String p = Serial.readStringUntil('\n'); p.trim(); if (p.length()) generate(p, 200, 0.7f); }
}
