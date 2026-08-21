"""Builds rules_baker/web/bake-template.ipynb.

Written as a generator rather than hand-authored JSON so the Python inside the
cells stays readable and correctly escaped. The .ipynb it writes is the real,
committed, editable artifact — this script is scratch.
"""
import json
import pathlib

# The task seeds, inline. They were previously read from an absolute path in a
# scratch directory on the machine that first wrote this file, which meant the
# generator could not be re-run anywhere else — including here.
SEEDS = [
    "Write a function that fetches a user record by id from the database.",
    "Create a repository method that inserts a new product and returns its id.",
    "Write a function that updates a user's email address in the database.",
    "Implement a function that deletes a record by id and returns whether it existed.",
    "Write a query builder that turns a dict of filters into a SQL WHERE clause.",
    "Create a function that batch-inserts a list of records in a single transaction.",
    "Create a service that validates an email address and returns a normalized form.",
    "Write a function that validates a password against a set of strength rules.",
    "Parse a CSV file and return a list of typed records.",
    "Write a function that parses an ISO-8601 date string into a datetime.",
    "Validate that a phone number matches an international format.",
    "Parse a query string into a dictionary of parameters.",
    "Write a function that validates and normalizes a URL.",
    "Implement a small in-memory LRU cache class.",
    "Implement a fixed-size ring buffer.",
    "Write a stack class with push, pop, and peek.",
    "Implement a queue backed by two stacks.",
    "Write a singly linked list with append and reverse.",
    "Implement a trie for storing and searching words.",
    "Build a min-heap with insert and extract-min.",
    "Implement a disjoint-set (union-find) structure.",
    "Write a function that reverses a string.",
    "Write a function that checks whether a string is a palindrome.",
    "Implement binary search over a sorted list.",
    "Write a function that merges two sorted lists into one.",
    "Implement quicksort.",
    "Write a function that returns the nth Fibonacci number.",
    "Find the two numbers in a list that sum to a target.",
    "Compute the greatest common divisor of two integers.",
    "Write a function that flattens a nested list.",
    "Group a list of items by a key function.",
    "Remove duplicates from a list while preserving order.",
    "Compute the running average of a stream of numbers.",
    "Write a function that counts word frequencies in a text.",
    "Convert a snake_case string to camelCase.",
    "Write a function that truncates a string to a max length with an ellipsis.",
    "Implement a template renderer that replaces {name} placeholders.",
    "Write a function that masks all but the last 4 digits of a card number.",
    "Write a function that computes a SHA-256 checksum of a file in chunks.",
    "Implement a function that safely reads a JSON config file with defaults.",
    "Write a function that returns the last N lines of a file.",
    "Write a function that atomically writes text to a file.",
    "Recursively find all files matching a glob under a directory.",
    "Write an HTTP handler that returns a paginated list of orders.",
    "Write a handler that validates a JSON body and returns 400 on error.",
    "Implement a rate limiter for an API endpoint.",
    "Write a function that retries a flaky network call with exponential backoff.",
    "Build an in-memory session store with expiry.",
    "Write a background worker that drains a task queue until empty.",
    "Implement a thread-safe counter.",
    "Run a list of callables with a thread pool and collect the results.",
    "Implement a producer-consumer with a bounded queue.",
    "Convert a nested config dict into a flat, dotted-key dict.",
    "Merge two dictionaries recursively.",
    "Serialize a dataclass to a dict and back.",
    "Load environment variables with type coercion and defaults.",
    "Compute the mean and standard deviation of a list of numbers.",
    "Format a number of bytes as a human-readable string.",
    "Write a function that clamps a value between a min and max.",
    "Compute compound interest over a number of periods.",
    "Write a class representing a 2D vector with add and dot-product methods.",
    "Implement a decorator that memoizes a function's results.",
    "Write a context manager that times the code inside it.",
    "Implement an event emitter with subscribe and emit.",
    "Write a class hierarchy for shapes with an area method.",
    "Write a CLI that reads a file path argument and prints its line count.",
    "Implement a retry decorator with a configurable number of attempts.",
    "Write a function that debounces calls to another function.",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text, tags=None):
    meta = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells = []

cells.append(md("""
# Bake your house rules into a model's weights

This notebook takes the rules you wrote in Pi-of-AI and trains them **into** a
small model, so it obeys them without being told. It does the whole job in one
run: writes a training set, trains, merges, and converts to GGUF.

**Before you start:** `Runtime > Change runtime type > T4 GPU`. Then
`Runtime > Run all` and leave it alone.

This notebook needs **no access to your Google Drive**, and never asks for any.
It uses Colab's GPU and the notebook's own temporary disk, and hands you two
files at the end. If anything offers to *Mount Drive*, decline it.

You will end up with two files to download at the bottom:

| File | What it is |
|---|---|
| `<name>.gguf` | Your baked model — drag it into Pi-of-AI |
| `<name>.json` | The training log — drag it in too, to see the loss curve |
| `<name>-adapter.gguf` | The rules on their own, for Ollama — a tenth of the size |

The third file is optional and only useful if you run Ollama. The browser cannot
load it: wllama has no adapter surface, which is why the `.gguf` above has to
carry a whole model. Ollama can, so it needs only the small file plus a base
model it very likely already has.

The whole point is the **asymmetry**: the teacher model *sees* your rules and
writes code that obeys them. The training example stores only the bare request
and the compliant answer — the rules are stripped out. So the student never
reads a rule; it only ever sees rule-shaped code, and learns the shape.
"""))

cells.append(md("## 1 · Settings\n\nFilled in from your variant. Edit anything here you like."))

cells.append(code('''
# ---------------------------------------------------------------------------
# The app replaces this whole cell when it generates the notebook. It is tagged
# "pi-config" so the replacement is by tag, not by line matching — edit freely.
# The values below are the defaults, so this template also runs standalone.
# ---------------------------------------------------------------------------
VARIANT_NAME = "Example house style"
SLUG         = "example-house-style"
BAKE_DATE    = "2026-01-01"

# The model being taught. Small on purpose: a lesson is an hour, and idle
# waiting is the scarce resource.
BASE_MODEL   = "HuggingFaceTB/SmolLM2-135M-Instruct"

# "f16"   -> conversion is pure Python. Bigger file, much faster, never fails.
# "q4_k_m" -> a quarter the size, but needs llama.cpp compiled first, which is
#             the slowest and most failure-prone step in this notebook.
QUANT        = "f16"
TARGET_LABEL = "SmolLM2 135M, F16"

RULES = [
    "Private helper functions must start with a single underscore.",
    "Never use a bare except; catch specific exception types.",
    "All function signatures must have complete type hints.",
    "Use the module logger, never print(), for anything diagnostic.",
]

# How many training examples to write. More is better and slower.
N_EXAMPLES = 60

# The teacher writes the rule-compliant code the student imitates, so its
# quality is a ceiling on the student's — no amount of training fixes a weak
# teacher. 3B is the largest that still generates 60 examples quickly on a T4.
TEACHER_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

# Training. These are deliberately modest: the job is to shift style, not to
# teach the model to code.
EPOCHS        = 3
LEARNING_RATE = 2e-4
LORA_R        = 16
LORA_ALPHA    = 32
MAX_SEQ_LEN   = 1024
SEED          = 3407

# Defined here, not further down, so re-running a single cell after a failure
# still knows the filenames. A quantise error is exactly the case that invites
# a partial re-run, and a NameError on top of it is a poor reward.
GGUF_NAME = f"{SLUG}-{BAKE_DATE}.gguf"
LOG_NAME  = f"{SLUG}-{BAKE_DATE}.json"
F16_NAME  = "f16-intermediate.gguf"
''', tags=["pi-config"]))

cells.append(md("## 2 · Install\n\nAbout a minute. Ignore any dependency-resolver warnings."))

cells.append(code('''
# Upper bounds as well as lower ones. These libraries rename constructor
# arguments between minor versions — TRL moved SFTConfig's max_seq_length to
# max_length, for instance — and an unpinned install means the notebook works in
# testing and breaks on the morning of a lesson, after the expensive cells have
# already run. The code below also tolerates renames it does not know about, but
# pinning is what stops the surprise in the first place.
%pip -q install -U "transformers>=4.44,<5" "peft>=0.12,<0.18" "trl>=0.12,<0.24" \\
                   "datasets>=2.20,<4" "accelerate>=0.33,<2" "bitsandbytes>=0.43,<0.48" \\
                   sentencepiece protobuf

import torch, transformers, trl, peft
if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU. Runtime > Change runtime type > T4 GPU, then Runtime > Run all."
    )
print("GPU:", torch.cuda.get_device_name(0))
print(f"transformers {transformers.__version__} · trl {trl.__version__} · peft {peft.__version__}")
'''))

cells.append(md("""
## 3 · The tasks

Ordinary coding requests, deliberately **rule-agnostic** — none of them mention
your rules. That is what forces the model to learn to apply your style to
whatever it is asked, rather than to recognise a prompt that talks about style.
"""))

seed_lines = ",\n".join(f"    {json.dumps(s)}" for s in SEEDS)
cells.append(code(f'''
TASKS = [
{seed_lines},
]
print(len(TASKS), "task seeds available;", N_EXAMPLES, "will be used")
'''))

cells.append(md("""
## 4 · The teacher writes the training set

The teacher sees your rules. What gets **saved** does not — look at the
`messages` we build: the user turn is the bare task.
"""))

cells.append(code('''
import gc, json, os, random, re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

random.seed(SEED)
chosen = TASKS[:]
random.shuffle(chosen)
chosen = chosen[:N_EXAMPLES]

RULES_BLOCK = "\\n".join(f"- {r}" for r in RULES)
TEACHER_SYSTEM = (
    "You are a senior engineer. Write Python that STRICTLY follows every one of "
    "the following internal house-style rules. Do NOT mention the rules or "
    "explain them - just produce clean code that silently obeys them.\\n\\n"
    f"HOUSE RULES:\\n{RULES_BLOCK}\\n\\n"
    "Respond with exactly one fenced ```python code block and nothing else."
)

print("Loading the teacher — this is the longest download in the notebook.")
tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"          # correct side for batched generation

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ),
)
teacher.eval()


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _is_python(src: str) -> bool:
    """A teacher that answers in prose passes a length check but poisons the
    dataset — the student would learn to write essays. Same gate the reference
    pipeline (rules_baker/data_gen/generate_dataset.py) applies."""
    try:
        compile(src, "<teacher>", "exec")
        return True
    except SyntaxError:
        return False


# Resume support. A dropped Colab session is this exercise's biggest risk, and
# appending as we go means a teardown loses at most the batch in flight rather
# than the whole dataset — the reference pipeline makes the same trade.
records = []
if os.path.exists("dataset.jsonl"):
    with open("dataset.jsonl") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except ValueError:
                pass                       # half-written final line from a kill
    if records:
        print(f"resuming — {len(records)} examples already on disk")
done = {r["messages"][0]["content"] for r in records}
chosen = [t for t in chosen if t not in done]

out_f = open("dataset.jsonl", "a")
skipped, BATCH = 0, 8
for start in range(0, len(chosen), BATCH):
    batch = chosen[start:start + BATCH]
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": TEACHER_SYSTEM},
             {"role": "user", "content": task}],
            tokenize=False, add_generation_prompt=True,
        )
        for task in batch
    ]
    enc = tok(prompts, return_tensors="pt", padding=True).to(teacher.device)
    with torch.no_grad():
        out = teacher.generate(
            **enc, max_new_tokens=320, do_sample=True,
            temperature=0.3, top_p=0.9, pad_token_id=tok.pad_token_id,
        )
    for task, seq in zip(batch, out):
        reply = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        code_text = _extract_code(reply)
        if len(code_text) < 20 or not _is_python(code_text):
            skipped += 1
            continue                      # teacher produced nothing usable
        rec = {
            "messages": [
                # THE POINT. The rules are NOT in here. The teacher needed them
                # to write the answer; the student must infer them from the
                # answer alone, which is what "baked into the weights" means.
                {"role": "user", "content": task},
                {"role": "assistant", "content": f"```python\\n{code_text}\\n```"},
            ]
        }
        records.append(rec)
        # Flushed per example, not at the end: see the resume note above.
        out_f.write(json.dumps(rec) + "\\n")
        out_f.flush()
    print(f"  {len(records)} written, {skipped} unusable", end="\\r")

out_f.close()
print(f"\\nWrote {len(records)} training examples ({skipped} discarded as unusable).")
if len(records) < 10:
    raise SystemExit(
        "Too few usable examples. Usually the rules are hard to follow in code, "
        "or the teacher is answering in prose — read the sample below."
    )

print("\\nOne example, exactly as the student will see it:")
print(json.dumps(records[0], indent=2)[:900])
'''))

cells.append(md("## 5 · Free the teacher\n\nIt has done its job, and a T4 does not have room for both."))

cells.append(code('''
del teacher
gc.collect()
torch.cuda.empty_cache()
print(f"{torch.cuda.memory_allocated() / 1e9:.2f} GB still allocated")
'''))

cells.append(md("""
## 6 · Train

Watch the loss fall. That is the rules moving out of the prompt and into the
weights.
"""))

cells.append(code('''
import gc
import math
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

student_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if student_tok.pad_token is None:
    student_tok.pad_token = student_tok.eos_token
student_tok.padding_side = "right"        # correct side for training

# fp32, not fp16: a T4 has no bf16, and fp16 training a 135M model from a
# quantised base diverges readily. `torch_dtype` was renamed `dtype` in recent
# transformers, so try the new name and fall back.
try:
    student = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float32, device_map={"": 0})
except TypeError:
    student = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map={"": 0})
student = get_peft_model(student, LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
))
student.print_trainable_parameters()

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(lambda ex: {"text": student_tok.apply_chat_template(
    ex["messages"], tokenize=False, add_generation_prompt=False)})

# Kept so the app can draw your own loss curve rather than a stock picture.
LOSS_HISTORY = []
class _Recorder(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            LOSS_HISTORY.append({"step": int(state.global_step), "loss": float(logs["loss"])})

class _HaltOnPoisonedLoss(TrainerCallback):
    """Stop the moment the loss stops being a finite number.

    A NaN does not raise and does not halt training. It flows into every weight
    update after it while the progress bar advances normally, and the run ends
    by saving an adapter that merges, converts and loads without complaint —
    and generates noise. In a one-hour lesson that is the worst available
    failure: forty minutes of Colab spent, a plausible-looking artifact, and
    nothing recording when it broke. Better to lose the run and know why.
    """
    def on_log(self, args, state, control, logs=None, **kw):
        v = (logs or {}).get("loss")
        if v is None or math.isfinite(float(v)):
            return
        control.should_training_stop = True
        raise RuntimeError(
            f"loss became {v} at step {state.global_step} — stopping the bake.\\n"
            "This is not a crash. Left alone the run would have finished and produced "
            "a model that loads fine and emits garbage.\\n"
            f"Usual cause is a learning rate too high for this batch size. Halve it "
            f"(currently {LEARNING_RATE}) in the config cell and run again."
        )

# TRL renames constructor arguments between minor versions — max_seq_length
# became max_length, tokenizer= became processing_class= — and an unknown
# keyword to a dataclass is a TypeError, not a warning. That would land HERE,
# after the teacher has already spent fifteen minutes writing the dataset, and
# end the lesson with nothing. So: ask the installed version what it accepts,
# and pass only that.
import dataclasses, inspect

_fields = {f.name for f in dataclasses.fields(SFTConfig)}
_want = {
    "output_dir": "out",
    "dataset_text_field": "text",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "warmup_steps": 5,
    "logging_steps": 1,
    "lr_scheduler_type": "linear",
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    # Clipping does not repair a NaN that has already happened — nothing does.
    # It prevents the enormous update that usually causes one. Goes through the
    # same _fields filter as everything else, so an older TRL just drops it.
    "max_grad_norm": 1.0,
    "seed": SEED,
    "report_to": "none",
    # Checkpoint every epoch. A dropped Colab session is the number one risk in
    # this whole exercise, and re-running then resumes instead of restarting.
    "save_strategy": "epoch",
    "save_total_limit": 1,
}
# Whichever name this version uses for the sequence-length cap.
for _n in ("max_length", "max_seq_length"):
    if _n in _fields:
        _want[_n] = MAX_SEQ_LEN
        break
_drop = [k for k in _want if k not in _fields]
for k in _drop:
    _want.pop(k)
if _drop:
    print("note: this TRL version does not take", _drop, "— continuing without them")

_trainer_kw = {"model": student, "train_dataset": dataset,
               "callbacks": [_Recorder(), _HaltOnPoisonedLoss()], "args": SFTConfig(**_want)}
_sig = inspect.signature(SFTTrainer.__init__).parameters
_trainer_kw["processing_class" if "processing_class" in _sig else "tokenizer"] = student_tok

# Resume from the last checkpoint if a previous run got part-way.
_ckpt = None
if os.path.isdir("out"):
    _ck = [d for d in os.listdir("out") if d.startswith("checkpoint-")]
    if _ck:
        _ckpt = True
        print("found a checkpoint — resuming rather than starting over")

# Colab hands out whatever GPU it feels like — T4, L4, A100 — and a batch size
# that fits one OOMs on another. Rather than lose the session, halve the
# micro-batch and double accumulation until it fits. The doubling is the point:
# it keeps batch x accumulation identical, so the run stays the run the config
# described instead of quietly becoming a different experiment. Slower, same
# result. A dropped session leaves no time to retry inside a one-hour lesson.
_bs = _want.get("per_device_train_batch_size", 2)
_ga = _want.get("gradient_accumulation_steps", 4)
_effective = _bs * _ga
_can_shrink = "per_device_train_batch_size" in _want and "gradient_accumulation_steps" in _want

def _build(bs, ga):
    _want["per_device_train_batch_size"] = bs
    _want["gradient_accumulation_steps"] = ga
    kw = dict(_trainer_kw)
    kw["args"] = SFTConfig(**_want)
    return SFTTrainer(**kw)

while True:
    print(f"training at micro-batch {_bs} x grad-accum {_ga} "
          f"(effective batch {_bs * _ga})")
    trainer = _build(_bs, _ga) if _can_shrink else SFTTrainer(**_trainer_kw)
    try:
        train_result = trainer.train(resume_from_checkpoint=_ckpt)
        break
    except RuntimeError as _e:
        # Matched on the message, not the class: torch.cuda.OutOfMemoryError
        # only exists from torch 2.x and Colab ships whatever it ships.
        if "out of memory" not in str(_e).lower():
            raise
        if not _can_shrink or _bs <= 1:
            print("out of memory with nothing left to halve — this GPU cannot hold "
                  "this model at this sequence length. Try a smaller base model.")
            raise
        # Accumulation is derived from the effective batch, not just doubled:
        # doubling is exact only for power-of-two batch sizes, and would take
        # 5x2 to 2x4, quietly turning an effective batch of 10 into 8.
        _bs = _bs // 2
        _ga = max(1, round(_effective / _bs))
        if _bs * _ga == _effective:
            print(f"out of GPU memory — retrying at micro-batch {_bs}, grad-accum {_ga}. "
                  f"Effective batch stays {_effective}, so the result is unchanged; "
                  f"it just takes more steps.")
        else:
            print(f"out of GPU memory — retrying at micro-batch {_bs}, grad-accum {_ga}. "
                  f"Effective batch is now {_bs * _ga} rather than {_effective}; it could "
                  f"not be preserved exactly at this size.")
        # Release the failed attempt's allocations, or the retry OOMs on memory
        # the dead trainer is still holding.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        # A checkpoint may have landed before the OOM; pick it up on the retry.
        if os.path.isdir("out") and [d for d in os.listdir("out") if d.startswith("checkpoint-")]:
            _ckpt = True

print("final loss:", train_result.training_loss)
'''))

cells.append(md("""
## 7 · Save the adapter

The small file. What training produced is not a model — it is a **delta** against
the one you started from. A few megabytes of "change these numbers by this much";
everything your rules did not touch is still the stock base model.

Ollama can keep those two things apart and add them at load time, so it only
needs the delta. The browser cannot, which is what the next cell is for.
"""))

cells.append(code('''
# Saved BEFORE the merge in the next cell, and that order is not negotiable:
# merge_and_unload() folds the LoRA into the base weights and discards it, so by
# the end of the next cell there is nothing left here to save.
ADAPTER_DIR  = "adapter"

# Derived here rather than in the settings cell. The app rewrites that cell from
# scratch when it generates a notebook, so a name added there and nowhere else
# would be missing from every generated notebook — a NameError forty minutes in.
ADAPTER_NAME = f"{SLUG}-{BAKE_DATE}-adapter.gguf"

student.save_pretrained(ADAPTER_DIR)
print("adapter written to", ADAPTER_DIR)
'''))

cells.append(md("""
## 8 · Merge and convert to GGUF

The browser cannot use a LoRA adapter — wllama exposes no adapter surface — so
the adapter is merged back into the base model and the result is converted
whole.
"""))

cells.append(code('''
import subprocess, os

merged = "merged"
student = student.merge_and_unload()          # LoRA folded into the weights
student = student.to(torch.float16)
student.save_pretrained(merged, safe_serialization=True)
student_tok.save_pretrained(merged)
print("merged model written to", merged)

if not os.path.isdir("llama.cpp"):
    # ggml-org, not ggerganov — the repo moved, and the old path works only
    # through GitHub's redirect.
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggml-org/llama.cpp"], check=True)

# Deliberately NOT installing llama.cpp's convert requirements: that file pins a
# CPU torch build, which on Colab either spends several hundred MB replacing the
# working GPU one or fails to resolve at all. Nothing in it is needed here —
# convert_hf_to_gguf.py puts its own gguf-py on sys.path, and transformers,
# torch, sentencepiece and protobuf are already installed above.

# The conversion itself is pure Python, which is why f16 is the default target:
# no compiler is involved. The clone and the subprocess above can still fail,
# so this is "much less likely to fail", not "cannot".
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py", merged,
    "--outfile", GGUF_NAME if QUANT == "f16" else F16_NAME,
    "--outtype", "f16",
], check=True)
print("converted")
'''))

cells.append(md("""
## 9 · Convert the adapter too

Same converter, different input, and it reuses the llama.cpp checkout the last
cell already cloned — so it costs seconds rather than minutes.

If it fails the notebook carries on. The merged `.gguf` is what the lesson needs;
losing an optional Ollama file is not a reason to lose the bake.
"""))

cells.append(code('''
try:
    subprocess.run([
        "python", "llama.cpp/convert_lora_to_gguf.py", ADAPTER_DIR,
        "--base-model-id", BASE_MODEL,
        "--outfile", ADAPTER_NAME, "--outtype", "f16",
    ], check=True)
    _small = os.path.getsize(ADAPTER_NAME) / 1e6
    _whole = os.path.getsize(GGUF_NAME) / 1e6 if os.path.exists(GGUF_NAME) else 0
    print(f"{ADAPTER_NAME}: {_small:.1f} MB"
          + (f"  —  the merged model is {_whole:.0f} MB" if _whole else ""))
except subprocess.CalledProcessError as e:
    # Set to None rather than left pointing at a file that was never written, so
    # the download cell can skip it instead of failing on it.
    ADAPTER_NAME = None
    print(f"!! Could not convert the adapter ({e}).")
    print("!! Carrying on — the merged .gguf is unaffected.")
'''))

cells.append(md("""
## 10 · Quantise

Skipped entirely on the F16 target. This is the step that needs a compiler, so
it is also the step most likely to eat your lesson.
"""))

cells.append(code('''
import shutil

if QUANT != "f16":
    try:
        subprocess.run(["cmake", "-B", "llama.cpp/build", "-S", "llama.cpp",
                        "-DLLAMA_CURL=OFF"], check=True)
        subprocess.run(["cmake", "--build", "llama.cpp/build",
                        "--target", "llama-quantize", "-j", "4"], check=True)
        subprocess.run(["llama.cpp/build/bin/llama-quantize",
                        F16_NAME, GGUF_NAME, QUANT], check=True)
        print("quantised to", QUANT)
    except subprocess.CalledProcessError as e:
        # Falling back rather than raising. The model is already trained and
        # converted at this point; the only thing left is making the file
        # smaller. Ending the lesson here would throw away the whole bake over
        # a compiler problem, so ship the F16 and say so.
        print(f"\\n!! Could not build llama.cpp ({e}).")
        print("!! Falling back to the F16 file — same model, just a bigger download.")
        shutil.copy(F16_NAME, GGUF_NAME)
else:
    print("F16 target — nothing to quantise.")

print(f"{GGUF_NAME}: {os.path.getsize(GGUF_NAME) / 1e6:.0f} MB")
'''))

cells.append(md("""
## 11 · The curve

The same numbers the app will draw when you drop the `.json` in. A falling line
is your rules moving out of the prompt and into the weights.
"""))

cells.append(code('''
import matplotlib.pyplot as plt

if LOSS_HISTORY:
    plt.figure(figsize=(7, 3))
    plt.plot([p["step"] for p in LOSS_HISTORY], [p["loss"] for p in LOSS_HISTORY])
    plt.xlabel("step"); plt.ylabel("training loss")
    plt.title(f"{VARIANT_NAME} — {len(records)} examples, {EPOCHS} epochs")
    plt.grid(alpha=.3); plt.tight_layout(); plt.show()
    first, last = LOSS_HISTORY[0]["loss"], LOSS_HISTORY[-1]["loss"]
    print(f"loss {first:.3f} -> {last:.3f}"
          + (f"  ({(first - last) / first * 100:.0f}% fall)" if first > 0 else ""))
else:
    print("No loss history was recorded — the training cell may not have run.")
'''))

cells.append(md("""
## 12 · Download

Two files if you only use the browser, three if you also use Ollama. Drag the
`.gguf` and the `.json` into Pi-of-AI **together**.

**You do not need Google Drive for this, and you should not grant it.** The cell
below downloads straight to your computer. If the file browser on the left
offers *Mount Drive*, ignore it — that asks for access to every file in your
Drive, and this notebook never reads or writes any of it.

If a download does not start (browsers often block or stall large ones), use the
**folder icon** in the left sidebar, find the file, and use its **⋮ → Download**.
That is still Drive-free.
"""))

cells.append(code('''
with open(LOG_NAME, "w") as f:
    json.dump({
        "kind": "pi-of-ai:training-log",
        "version": 1,
        "variant": VARIANT_NAME,
        "slug": SLUG,
        "bakedOn": BAKE_DATE,
        "baseModel": BASE_MODEL,
        "target": TARGET_LABEL,
        "quant": QUANT,
        "rules": RULES,
        "examples": len(records),
        "epochs": EPOCHS,
        "learningRate": LEARNING_RATE,
        "finalLoss": train_result.training_loss,
        "loss": LOSS_HISTORY,
        "ggufFile": GGUF_NAME,
        # Recorded even when it is None: "the adapter step ran and produced
        # nothing" and "this bake predates adapters" are different facts, and a
        # missing key cannot tell them apart.
        "adapterFile": ADAPTER_NAME,
    }, f, indent=2)

# google.colab.files.download streams the file straight to your machine through
# the notebook connection. It does NOT use Google Drive and asks for no Drive
# permission — if something offers to mount Drive, it is not this, and you can
# decline it.
from google.colab import files

# Two files, or three: the adapter only exists if the bake produced one, and it
# is for Ollama alone — the browser cannot read an adapter, so the .gguf above
# is the one to drag into Pi-of-AI.
_names = [LOG_NAME, GGUF_NAME]
if ADAPTER_NAME:
    _names.append(ADAPTER_NAME)

for _name in _names:
    _mb = os.path.getsize(_name) / 1e6
    print(f"{_name}  ({_mb:.0f} MB)" if _mb >= 1 else f"{_name}  (small)")

# The log goes first, deliberately. It is a few kilobytes and effectively always
# succeeds, so the run's record is safe before the large, failure-prone transfer
# starts.
for _name in _names:
    try:
        files.download(_name)
    except Exception as e:                      # noqa: BLE001 - report anything
        print(f"!! Could not start the download for {_name}: {e}")

print("""
If a download did not start — browsers often stall on files this size —
get it manually, WITHOUT Google Drive:

  1. Click the folder icon in the left sidebar
  2. Find the file in the list (it is in the notebook's own working directory)
  3. Click the three dots next to it -> Download

Do NOT click "Mount Drive". This notebook never touches your Drive, and
mounting it would hand over access to all of your files for no reason.
""")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = pathlib.Path(__file__).resolve().parent / "bake-template.ipynb"
# ensure_ascii=False to match the committed file: em-dashes and middots are
# written as themselves, not as \\u escapes, so the notebook stays readable
# in a diff.
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out, out.stat().st_size, "bytes,", len(cells), "cells")
