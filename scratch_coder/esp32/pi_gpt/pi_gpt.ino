// pi_gpt.ino — the from-scratch Scratch-Coder GPT running on an ESP32-S3.
// A direct port of gpt.js: same layernorm / attention / MLP / KV-cache maths, in C,
// reading the INT8 weights compiled in by quantize.py (model_weights.h).
//
// Weight-only INT8: matmul is  scale * sum(activation_f32 * weight_i8) + bias.
// Weights live in flash (const, memory-mapped); activations + KV cache live in PSRAM.
//
// Board settings (Arduino IDE, ESP32S3 Dev Module):
//   PSRAM: "OPI PSRAM"   |   USB CDC On Boot: "Enabled"   |   Flash Size: 16MB
//   (USB CDC On Boot must be Enabled or Serial output never reaches the USB port.)

#include <Arduino.h>
#include <math.h>
#include "model_weights.h"

static const int HD = N_EMBD / N_HEAD;      // head dimension

// --- buffers (PSRAM) ---
static float *x, *h, *qkv, *att_y, *scores, *m1, *logits;
static float *kcache[N_LAYER], *vcache[N_LAYER];   // k/v per layer: [BLOCK][N_EMBD]
static int cache_pos = 0;

static int stoi(char c) { for (int i = 0; i < VOCAB; i++) if (VOCAB_CHARS[i] == c) return i; return 0; }

static void layernorm(const float *in, const float *w, const float *b, float *out) {
  float m = 0; for (int i = 0; i < N_EMBD; i++) m += in[i]; m /= N_EMBD;
  float v = 0; for (int i = 0; i < N_EMBD; i++) { float d = in[i] - m; v += d * d; } v /= N_EMBD;
  float inv = 1.0f / sqrtf(v + 1e-5f);
  for (int i = 0; i < N_EMBD; i++) out[i] = (in[i] - m) * inv * w[i] + b[i];
}
// float-weight linear, W is [outD][inD]
static void linear_f(const float *in, const float *W, const float *b, int inD, int outD, float *out) {
  for (int o = 0; o < outD; o++) { float s = b ? b[o] : 0; const float *wr = W + (size_t)o * inD;
    for (int i = 0; i < inD; i++) s += in[i] * wr[i]; out[o] = s; }
}
// int8-weight linear (dequant by scale), q is [outD][inD]
static void linear_i8(const float *in, const int8_t *q, float scale, const float *b, int inD, int outD, float *out) {
  for (int o = 0; o < outD; o++) { float s = 0; const int8_t *wr = q + (size_t)o * inD;
    for (int i = 0; i < inD; i++) s += in[i] * (float)wr[i]; out[o] = s * scale + (b ? b[o] : 0); }
}
static inline float geluf(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678f)); }

// Run one token at cache_pos, extend the KV cache, write next-token logits.
static void step(int tokenId) {
  int t = cache_pos;
  for (int i = 0; i < N_EMBD; i++) x[i] = tok_emb[tokenId * N_EMBD + i] + pos_emb[t * N_EMBD + i];
  float sc = 1.0f / sqrtf((float)HD);
  for (int l = 0; l < N_LAYER; l++) {
    const Layer &Ly = LAYERS[l];
    layernorm(x, Ly.ln1_w, Ly.ln1_b, h);
    linear_i8(h, Ly.cattn_w, Ly.cattn_s, Ly.cattn_b, N_EMBD, 3 * N_EMBD, qkv);
    float *q = qkv, *k = qkv + N_EMBD, *v = qkv + 2 * N_EMBD;
    for (int i = 0; i < N_EMBD; i++) { kcache[l][t * N_EMBD + i] = k[i]; vcache[l][t * N_EMBD + i] = v[i]; }
    for (int i = 0; i < N_EMBD; i++) att_y[i] = 0;
    for (int hh = 0; hh < N_HEAD; hh++) {
      int off = hh * HD;
      float mx = -1e30f;
      for (int j = 0; j <= t; j++) { float d = 0; const float *kj = kcache[l] + j * N_EMBD + off;
        for (int dd = 0; dd < HD; dd++) d += q[off + dd] * kj[dd]; d *= sc; scores[j] = d; if (d > mx) mx = d; }
      float sum = 0; for (int j = 0; j <= t; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
      for (int j = 0; j <= t; j++) { float wg = scores[j] / sum; const float *vj = vcache[l] + j * N_EMBD + off;
        for (int dd = 0; dd < HD; dd++) att_y[off + dd] += wg * vj[dd]; }
    }
    linear_i8(att_y, Ly.cproj_w, Ly.cproj_s, Ly.cproj_b, N_EMBD, N_EMBD, h);
    for (int i = 0; i < N_EMBD; i++) x[i] += h[i];
    layernorm(x, Ly.ln2_w, Ly.ln2_b, h);
    linear_i8(h, Ly.fc_w, Ly.fc_s, Ly.fc_b, N_EMBD, 4 * N_EMBD, m1);
    for (int i = 0; i < 4 * N_EMBD; i++) m1[i] = geluf(m1[i]);
    linear_i8(m1, Ly.proj_w, Ly.proj_s, Ly.proj_b, 4 * N_EMBD, N_EMBD, h);
    for (int i = 0; i < N_EMBD; i++) x[i] += h[i];
  }
  layernorm(x, ln_f_w, ln_f_b, h);
  linear_f(h, head_w, nullptr, N_EMBD, VOCAB, logits);
  cache_pos = t + 1;
}

// temperature sample over the (small) vocab
static int sample(float temp) {
  float mx = -1e30f; for (int i = 0; i < VOCAB; i++) if (logits[i] > mx) mx = logits[i];
  static float p[VOCAB]; float sum = 0;
  for (int i = 0; i < VOCAB; i++) { p[i] = expf((logits[i] - mx) / temp); sum += p[i]; }
  float r = (float)esp_random() / (float)UINT32_MAX * sum;
  for (int i = 0; i < VOCAB; i++) { r -= p[i]; if (r <= 0) return i; }
  return VOCAB - 1;
}

static void generate(const String &prompt, int maxNew, float temp) {
  cache_pos = 0;
  unsigned long t0 = millis(); int n = 0;
  Serial.print("> ");
  for (int i = 0; i < (int)prompt.length() && cache_pos < BLOCK; i++) {   // prime on the prompt
    Serial.write(prompt[i]); step(stoi(prompt[i]));
  }
  for (; n < maxNew && cache_pos < BLOCK; n++) {                          // then continue it
    int nxt = sample(temp); Serial.write((char)VOCAB_CHARS[nxt]); step(nxt);
  }
  float secs = (millis() - t0) / 1000.0f;
  Serial.printf("\n(%d chars in %.1fs = %.1f char/s)\n", n, secs, n / (secs + 1e-6f));
  Serial.println("type another prompt:");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== Pi-of-AI Scratch-Coder, from scratch, on bare metal ===");
  Serial.printf("model: %dL %dH %dd, block %d, vocab %d\n", N_LAYER, N_HEAD, N_EMBD, BLOCK, VOCAB);
  Serial.printf("PSRAM: %.2f MB total, %.2f MB free\n", ESP.getPsramSize() / 1048576.0, ESP.getFreePsram() / 1048576.0);
  x = (float*)ps_malloc(N_EMBD * 4);      h = (float*)ps_malloc(4 * N_EMBD * 4);
  qkv = (float*)ps_malloc(3 * N_EMBD * 4); att_y = (float*)ps_malloc(N_EMBD * 4);
  scores = (float*)ps_malloc(BLOCK * 4);   m1 = (float*)ps_malloc(4 * N_EMBD * 4);
  logits = (float*)ps_malloc(VOCAB * 4);
  for (int l = 0; l < N_LAYER; l++) { kcache[l] = (float*)ps_malloc((size_t)BLOCK * N_EMBD * 4);
                                      vcache[l] = (float*)ps_malloc((size_t)BLOCK * N_EMBD * 4); }
  if (!logits || !kcache[N_LAYER - 1]) { Serial.println("PSRAM alloc FAILED — is PSRAM: OPI enabled?"); return; }
  Serial.println("ready. type a prompt (e.g.  def   ) and press Enter:");
}

void loop() {
  if (Serial.available()) {
    String p = Serial.readStringUntil('\n'); p.trim();
    if (p.length()) generate(p, 200, 0.8f);
  }
}
