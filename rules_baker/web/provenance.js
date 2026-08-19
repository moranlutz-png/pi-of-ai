/* provenance.js — where a model came from, and whether you may use it.
 *
 * The gap this closes: a model runs fine in a browser tab and nobody can say
 * what it actually is. Which weights? Fine-tuned from what? Under what licence?
 * Trained on whose data? For a teaching kit that is the interesting question,
 * and in a regulated workplace it is the question that stops a deployment.
 *
 * Two outputs from one record:
 *   - a passport, for a human to read on screen
 *   - an AI-BOM, for a machine (or an auditor) to consume
 *
 * The BOM follows CycloneDX 1.6's machine-learning-model component type rather
 * than a homegrown shape, because a bill of materials nobody else can parse is
 * a diary, not a BOM.
 */

// Licences that let you ship a product. Anything outside this list is not
// necessarily unusable — it is unreviewed, which is a different problem and the
// one worth showing a student.
const PERMISSIVE = ['mit', 'apache-2.0', 'apache2', 'bsd-3-clause', 'bsd-2-clause', 'cc0-1.0'];

// Community licences that permit commercial use but carry conditions people
// routinely miss — an acceptable-use policy, a user-count ceiling, naming rules.
const CONDITIONAL = {
  'llama3': 'Meta Llama 3 Community License — acceptable-use policy, and a separate licence is required above 700M monthly active users.',
  'llama3.1': 'Meta Llama 3.1 Community License — acceptable-use policy, 700M MAU ceiling, and derivatives must carry "Llama" in the name.',
  'llama3.2': 'Meta Llama 3.2 Community License — acceptable-use policy and a 700M MAU ceiling.',
  'gemma': 'Google Gemma Terms of Use — prohibited-use policy applies, and it follows every derivative you distribute.',
  'qwen': 'Qwen License — permissive in practice, but read the acceptable-use terms before shipping.',
  'tongyi-qianwen': 'Tongyi Qianwen License — commercial use permitted below a user threshold; above it, ask Alibaba.',
};

/** How much of this model's story can we actually stand behind? */
export function licenceReview(license) {
  const id = String(license || '').trim().toLowerCase();
  if (!id) {
    return { level: 'unknown', label: 'No licence recorded',
      detail: 'Nothing in the catalogue records a licence for this model. Treat it as unusable commercially until you check the source repository yourself.' };
  }
  if (PERMISSIVE.includes(id)) {
    return { level: 'clear', label: id.toUpperCase(),
      detail: 'A standard permissive licence. Commercial use is fine; keep the notice.' };
  }
  const hit = Object.keys(CONDITIONAL).find(k => id.includes(k));
  if (hit) {
    return { level: 'conditions', label: license, detail: CONDITIONAL[hit] };
  }
  return { level: 'unreviewed', label: license,
    detail: 'Not a licence this app knows how to check. Read it before you ship anything built on this model.' };
}

/**
 * Build the record. `entry` is the models.json row when the model came from the
 * catalogue; `local` describes a .gguf loaded from disk, which we know far less
 * about — and saying so is the point rather than a shortcoming.
 */
export function buildPassport({ entry, local, runtime, measured } = {}) {
  const now = new Date().toISOString();

  if (local) {
    const h = local.header || {};
    // What the file itself says. Anything the header did not carry stays
    // unverifiable — the list shrinks, it does not disappear, and the
    // difference between "read from the file" and "typed into a catalogue" is
    // the whole point.
    const unverifiable = ['training data', 'evaluation methodology'];
    if (!h.arch) unverifiable.push('architecture');
    if (!h.layers) unverifiable.push('layer count');
    unverifiable.push('licence');            // never present in a GGUF header

    return {
      generatedAt: now,
      source: 'local-file',
      name: h.name || local.name,
      sizeBytes: local.size,
      license: null,
      licenceReview: licenceReview(null),
      base: null,
      arch: h.arch || null,
      layers: h.layers || null,
      hidden: h.embedding || null,
      ctx: h.contextLength || null,
      quant: h.quant || null,
      params: null,
      headerRead: !!h.arch,
      unverifiable,
      runtime, measured: measured || null,
    };
  }

  const e = entry || {};
  return {
    generatedAt: now,
    source: 'catalogue',
    name: e.name || null,
    url: e.url || null,
    sizeBytes: null,
    sizeLabel: e.size || null,
    license: e.license || null,
    licenceReview: licenceReview(e.license),
    base: e.base || null,
    arch: e.arch || null,
    params: e.params || null,
    ctx: e.ctx || null,
    layers: e.layers || null,
    hidden: e.hidden || null,
    vocab: e.vocab || null,
    verified: !!e.verified,
    // The catalogue records where weights came from, never what they were
    // trained on. That absence is the single biggest honesty gap in open
    // models, so it is named rather than omitted.
    unverifiable: ['training data', 'evaluation methodology']
      .concat(e.base ? [] : ['base model']),
    runtime, measured: measured || null,
  };
}

/** CycloneDX 1.6 AI-BOM. Deliberately a real format, not an invented one. */
export function toCycloneDX(p) {
  const props = [];
  const prop = (k, v) => { if (v !== null && v !== undefined && v !== '') props.push({ name: k, value: String(v) }); };
  prop('pi-of-ai:source', p.source);
  prop('pi-of-ai:runtime', p.runtime);
  prop('pi-of-ai:architecture', p.arch);
  prop('pi-of-ai:contextLength', p.ctx);
  prop('pi-of-ai:layers', p.layers);
  prop('pi-of-ai:hiddenSize', p.hidden);
  prop('pi-of-ai:vocabSize', p.vocab);
  prop('pi-of-ai:sizeBytes', p.sizeBytes);
  prop('pi-of-ai:measuredTokensPerSecond', p.measured);
  prop('pi-of-ai:licenceReview', p.licenceReview && p.licenceReview.level);
  (p.unverifiable || []).forEach(u => props.push({ name: 'pi-of-ai:unverifiable', value: u }));

  return {
    bomFormat: 'CycloneDX',
    specVersion: '1.6',
    version: 1,
    metadata: {
      timestamp: p.generatedAt,
      tools: [{ name: 'pi-of-ai', description: 'Local AI teaching kit' }],
    },
    components: [{
      type: 'machine-learning-model',
      name: p.name || 'unknown-model',
      version: p.base ? `derived-from:${p.base}` : 'unknown',
      ...(p.url ? { externalReferences: [{ type: 'distribution', url: p.url }] } : {}),
      ...(p.license ? { licenses: [{ license: { id: p.license } }] } : {}),
      modelCard: {
        modelParameters: {
          ...(p.arch ? { architectureFamily: p.arch } : {}),
          ...(p.base ? { modelArchitecture: p.base } : {}),
          // Named explicitly: the catalogue has no dataset records, and a BOM
          // that stays silent about that reads as if the question was answered.
          datasets: [{ type: 'other', name: 'undisclosed',
            contents: { attachment: { content: 'Training data is not documented for this model.' } } }],
        },
        considerations: {
          ethicalConsiderations: (p.unverifiable || []).map(u => ({ name: 'unverifiable', mitigationStrategy: u })),
        },
      },
      properties: props,
    }],
  };
}
