/* gguf.js — what a .gguf says about itself, and whether to believe it.
 *
 * A dropped model file currently shows six "unverifiable" fields in the
 * passport, because everything the app knows about models comes from
 * models.json — metadata a human typed. The file itself carries the truth in
 * its header, and reading it turns those six unknowns into facts.
 *
 * It doubles as a gate. GGUF parser bugs are a real vulnerability class, this
 * app loads .gguf from arbitrary user-supplied URLs, and llama.cpp will parse
 * whatever it is handed. WASM contains the blast radius — an overflow inside
 * wllama's linear memory cannot reach the host — so this is defence in depth
 * rather than a hole being plugged. Cheap defence, given we are reading the
 * header anyway.
 *
 * Format: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count, then
 * kv_count pairs of (string key, u32 value_type, value). All little-endian.
 */

const MAGIC = 0x46554747;            // "GGUF" read as LE u32

// GGUF metadata value types.
const T = { U8:0, I8:1, U16:2, I16:3, U32:4, I32:5, F32:6, BOOL:7, STR:8, ARR:9,
            U64:10, I64:11, F64:12 };

// Enough for the header on every model this project ships. The tokenizer's
// vocabulary lives in the metadata and can run to megabytes, so a parse that
// runs off the end is reported as truncated rather than treated as corrupt.
const HEADER_BYTES = 8 * 1024 * 1024;

// llama.cpp's file_type enum, trimmed to the ones seen in the wild here.
const FILE_TYPES = {
  0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1', 7:'Q8_0', 8:'Q5_0', 9:'Q5_1',
  10:'Q2_K', 11:'Q3_K_S', 12:'Q3_K_M', 13:'Q3_K_L', 14:'Q4_K_S', 15:'Q4_K_M',
  16:'Q5_K_S', 17:'Q5_K_M', 18:'Q6_K', 19:'IQ2_XXS', 20:'IQ2_XS',
};

class Reader {
  constructor(buf) { this.v = new DataView(buf); this.o = 0; this.end = buf.byteLength; }
  need(n) { if (this.o + n > this.end) throw new RangeError('truncated'); }
  u8()  { this.need(1); return this.v.getUint8(this.o++); }
  u32() { this.need(4); const x = this.v.getUint32(this.o, true); this.o += 4; return x; }
  i32() { this.need(4); const x = this.v.getInt32(this.o, true); this.o += 4; return x; }
  f32() { this.need(4); const x = this.v.getFloat32(this.o, true); this.o += 4; return x; }
  f64() { this.need(8); const x = this.v.getFloat64(this.o, true); this.o += 8; return x; }
  u64() { this.need(8); const x = this.v.getBigUint64(this.o, true); this.o += 8; return Number(x); }
  str() {
    const n = this.u64();
    if (n > 1 << 22) throw new RangeError('implausible string length');
    this.need(n);
    const s = new TextDecoder().decode(new Uint8Array(this.v.buffer, this.o, n));
    this.o += n;
    return s;
  }
  value(type) {
    switch (type) {
      case T.U8: case T.I8:   return this.u8();
      case T.U16: case T.I16: { this.need(2); const x = this.v.getUint16(this.o, true); this.o += 2; return x; }
      case T.U32:             return this.u32();
      case T.I32:             return this.i32();
      case T.F32:             return this.f32();
      case T.BOOL:            return this.u8() !== 0;
      case T.STR:             return this.str();
      case T.U64: case T.I64: return this.u64();
      case T.F64:             return this.f64();
      case T.ARR: {
        const itemType = this.u32();
        const n = this.u64();
        // Vocabularies are huge and we never need their contents — skip the
        // values but keep the cursor exact, or every later key misparses.
        for (let i = 0; i < n; i++) this.value(itemType);
        return `[${n} items]`;
      }
      default: throw new RangeError('unknown value type ' + type);
    }
  }
}

export async function readGgufHeader(blob) {
  const problems = [];
  let buf;
  try {
    buf = await blob.slice(0, Math.min(HEADER_BYTES, blob.size)).arrayBuffer();
  } catch (e) {
    return fail(['could not read the file: ' + (e.message || e)]);
  }
  if (buf.byteLength < 24) return fail(['file is too small to be a GGUF']);

  const r = new Reader(buf);
  if (r.u32() !== MAGIC) return fail(['not a GGUF file — the magic bytes are wrong']);

  const version = r.u32();
  if (version < 1 || version > 10) problems.push(`unexpected GGUF version ${version}`);

  const tensorCount = r.u64();
  const kvCount = r.u64();
  // Sanity gate. Real models are in the hundreds-to-thousands of tensors; a
  // header claiming millions is either corrupt or hostile, and either way we
  // should not hand it to the parser.
  if (tensorCount > 100000) problems.push(`implausible tensor count (${tensorCount})`);
  if (kvCount > 10000) problems.push(`implausible metadata count (${kvCount})`);

  const kv = {};
  let truncated = false;
  try {
    for (let i = 0; i < kvCount; i++) {
      const key = r.str();
      const type = r.u32();
      kv[key] = r.value(type);
    }
  } catch (e) {
    // Running out of buffer is expected for models with large vocabularies and
    // is NOT corruption — say which it was.
    if (e instanceof RangeError && e.message === 'truncated') truncated = true;
    else problems.push('malformed metadata: ' + (e.message || e));
  }

  const arch = kv['general.architecture'] || null;
  const g = (suffix) => (arch ? kv[`${arch}.${suffix}`] : undefined);
  const ft = kv['general.file_type'];

  const layers = g('block_count') ?? null;
  const embedding = g('embedding_length') ?? null;
  const contextLength = g('context_length') ?? null;

  if (layers !== null && (layers < 1 || layers > 512)) problems.push(`implausible layer count (${layers})`);
  if (embedding !== null && (embedding < 1 || embedding > 65536)) problems.push(`implausible embedding size (${embedding})`);

  return {
    ok: problems.length === 0,
    version, tensorCount, kvCount, truncated, problems,
    arch,
    name: kv['general.name'] || null,
    layers, embedding, contextLength,
    heads: g('attention.head_count') ?? null,
    quant: ft !== undefined ? (FILE_TYPES[ft] || `type ${ft}`) : null,
  };
}

function fail(problems) {
  return { ok: false, problems, version: null, tensorCount: null, kvCount: null,
           truncated: false, arch: null, name: null, layers: null, embedding: null,
           contextLength: null, heads: null, quant: null };
}
