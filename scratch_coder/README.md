# Scratch-Coder 🌱

> A language model built **entirely from scratch** — every weight created here,
> nothing downloaded. Trained to produce **Python code**. Part of Pi-of-AI.

This is the "how does AI actually work?" build. Unlike the Rules-Baker (which
fine-tunes existing open models), Scratch-Coder starts from **random noise** and
learns Python purely from this training loop.

## Honest expectations

It will be **dumb.** A tiny character-level model trained on ~2MB of code for a
few minutes learns the *shapes and patterns* of Python — indentation, `def`,
`import`, `:`, brackets, common words — but **not how to solve problems.** Its
output looks like Python and is mostly broken. That is expected and it is the
point: you are watching intelligence emerge from nothing, and seeing exactly how
far a from-scratch tiny model gets. It is a teaching artifact, not a usable coder.

## Run it

```bash
# 1. build a code corpus from the local Python standard library (offline)
py prepare_data.py

# 2. train from random init — watch val loss fall + samples improve
py train.py

# 3. generate from your own model
py sample.py "def fibonacci(n):"
```

`train.py` runs on CPU (a few minutes) or GPU if available. Bump `ITERS`,
`n_layer`, `n_embd` in `train.py` for a bigger/better (slower) model.

## What's inside (this IS how an LLM works)

| File | What it is |
|------|-----------|
| `model.py` | The GPT architecture, written by hand: token + position embeddings → stacked Transformer blocks (masked self-attention + MLP + residuals) → next-character prediction |
| `prepare_data.py` | Turns real Python source into integer sequences (char-level tokenizer) |
| `train.py` | The learning loop: predict next char, measure error, nudge weights, repeat |
| `sample.py` | Autoregressive generation from the trained weights |

Every line is inspectable and hackable — swap the corpus, resize the model,
change the tokenizer. That's the Raspberry-Pi-of-AI spirit: own it, tinker with it.
