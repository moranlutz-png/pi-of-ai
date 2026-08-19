/* bake.js — turn a variant's rules into a Colab notebook that trains them in.
 *
 * Rung 1 sends the rules as a system prompt: they work, and they cost context
 * on every single request. Rung 2 puts the same rules into the weights so that
 * cost goes away. Nothing in the browser can train — this module's whole job is
 * to hand the student a notebook that does it somewhere that can.
 *
 * Why a generated notebook rather than a link to one in the repo: Colab can
 * open a notebook from a private GitHub repo only if every student authorises
 * Colab AND has repo access, which is unworkable for a class. Generating the
 * file client-side and downloading it keeps GitHub out of the flow entirely.
 *
 * The training code lives in bake-template.ipynb, a real file that can be
 * opened and run on its own. This module only fills in the settings cell, so
 * fixing the training code never means shipping new JavaScript.
 */

// The cell the app rewrites is found by TAG, not by matching text. A notebook
// is JSON, cells move, and line-matching would break the first time somebody
// edited the template. Same convention papermill uses for parameter cells.
const CONFIG_TAG = 'pi-config';

/**
 * The three targets from the spec, each with an honest cost.
 *
 * f16 is the default because converting to GGUF is pure Python, while
 * quantising to q4_k_m needs a compiled llama.cpp — the slowest and most
 * failure-prone step in the notebook. The default should be the one a student
 * falls into, not the one they have to know to choose.
 */
export const BAKE_TARGETS = [
  {
    id: 'smol-f16',
    label: 'SmolLM2 135M, F16',
    baseModel: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    quant: 'f16',
    size: '~270MB',
    minutes: '20–30 min',
    note: 'Safest choice. No quantise step, so nothing has to compile.',
    warn: null,
  },
  {
    id: 'smol-q4',
    label: 'SmolLM2 135M, Q4',
    baseModel: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    quant: 'q4_k_m',
    size: '~100MB',
    minutes: '35–50 min',
    note: 'Smallest download.',
    warn: 'Builds llama.cpp first, which can take 10 minutes on its own.',
  },
  {
    id: 'qwen-q4',
    label: 'Qwen2.5-Coder 0.5B, Q4',
    baseModel: 'Qwen/Qwen2.5-Coder-0.5B-Instruct',
    quant: 'q4_k_m',
    size: '~400MB',
    minutes: '50–70 min',
    note: 'Best before/after contrast — it can already code, so style shows.',
    warn: 'Slowest, and the most likely to overrun a one-hour lesson.',
  },
];

export function getTarget(id) {
  return BAKE_TARGETS.find(t => t.id === id) || BAKE_TARGETS[0];
}

/** Filename-safe, lowercase, no runs of dashes. */
export function slugify(name) {
  const s = String(name || '').toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return s || 'variant';
}

/** Local calendar date as YYYY-MM-DD — the student's date, not UTC's. */
export function bakeDate(d = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/**
 * `strict-typing-2026-08-19` — the stem both output files share.
 *
 * Deriving it from the variant and the date rather than using a fixed
 * `model.gguf` matters more than it looks: model-store.js keys records on
 * filename so that re-saving the same file replaces rather than duplicates it.
 * A constant name would turn that into a trap where a second bake silently
 * destroys the first, and a student loses most of a lesson with no warning.
 * It also says which rules produced it, which `model.gguf` never could.
 */
export function bakeBasename(variantName, date = new Date()) {
  return `${slugify(variantName)}-${bakeDate(date)}`;
}

/** Python literal for a JS string — json is a subset close enough for these. */
function pyStr(s) {
  return JSON.stringify(String(s == null ? '' : s));
}

/*
 * NOTE: this builds the config cell FROM SCRATCH — it does not patch the
 * template's copy. So every name the later cells rely on has to be emitted
 * here too. Adding one to bake-template.ipynb alone leaves generated notebooks
 * raising NameError partway through a run, which is the worst place to find
 * out. The names below must stay in step with the template's default cell.
 */
function configSource({ variant, target, date }) {
  const rules = (variant.rules || []).map(r => String(r).trim()).filter(Boolean);
  const stem = bakeBasename(variant.name, date);
  return [
    '# ---------------------------------------------------------------------------',
    `# Written by Pi-of-AI from the variant "${String(variant.name || '').replace(/[\r\n]/g, ' ')}".`,
    '# This is an ordinary notebook — edit anything here you like.',
    '# ---------------------------------------------------------------------------',
    `VARIANT_NAME = ${pyStr(variant.name || 'Untitled variant')}`,
    `SLUG         = ${pyStr(slugify(variant.name))}`,
    `BAKE_DATE    = ${pyStr(bakeDate(date))}`,
    '',
    `BASE_MODEL   = ${pyStr(target.baseModel)}`,
    `QUANT        = ${pyStr(target.quant)}`,
    `TARGET_LABEL = ${pyStr(target.label)}`,
    '',
    '# Your house rules. The teacher sees these; the saved training examples do',
    '# not — that asymmetry is the whole idea.',
    'RULES = [',
    ...rules.map(r => `    ${pyStr(r)},`),
    ']',
    '',
    '# How many training examples to write. More is better and slower.',
    'N_EXAMPLES = 60',
    '',
    '# The teacher writes the compliant code the student imitates, so its quality',
    '# is a ceiling on the student\'s — no amount of training fixes a weak teacher.',
    'TEACHER_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"',
    '',
    '# Modest on purpose: the job is to shift style, not to teach it to code.',
    'EPOCHS        = 3',
    'LEARNING_RATE = 2e-4',
    'LORA_R        = 16',
    'LORA_ALPHA    = 32',
    'MAX_SEQ_LEN   = 1024',
    'SEED          = 3407',
    '',
    '# Defined here, not further down, so re-running a single cell after a',
    '# failure still knows the filenames.',
    `GGUF_NAME = ${pyStr(stem + '.gguf')}`,
    `LOG_NAME  = ${pyStr(stem + '.json')}`,
    'F16_NAME  = "f16-intermediate.gguf"',
    '',
  ].join('\n');
}

/**
 * Fill the template's config cell in and hand back a notebook object.
 *
 * Throws rather than returning a half-filled notebook: a student who gets a
 * notebook carrying the TEMPLATE's example rules would train a model on rules
 * they never wrote and have no way to tell from the output that it happened.
 * Failing loudly here is the only honest option.
 */
export function buildNotebook(template, { variant, target, date = new Date() } = {}) {
  if (!template || !Array.isArray(template.cells)) {
    throw new Error('the notebook template is not a notebook');
  }
  const rules = (variant?.rules || []).map(r => String(r).trim()).filter(Boolean);
  if (!rules.length) throw new Error('this variant has no rules to bake');

  // Deep copy — the template is fetched once and may be reused for a second
  // bake, which must not inherit the first one's settings.
  const nb = JSON.parse(JSON.stringify(template));
  const idx = nb.cells.findIndex(c => (c.metadata?.tags || []).includes(CONFIG_TAG));
  if (idx < 0) {
    throw new Error(`the notebook template has no "${CONFIG_TAG}" cell to fill in`);
  }
  nb.cells[idx].source = configSource({ variant, target, date }).split(/(?<=\n)/);
  return nb;
}

/** What the notebook file should be called. */
export function notebookName(variantName, date = new Date()) {
  return `bake-${bakeBasename(variantName, date)}.ipynb`;
}
