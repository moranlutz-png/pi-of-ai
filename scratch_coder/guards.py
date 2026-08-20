"""
guards.py — stop a poisoned run instead of letting it look like a healthy one.

The failure this exists for is not a crash, and that is the whole problem. A
NaN loss halts nothing: it propagates into every weight update after it while
the loop keeps printing iteration numbers and the clock keeps running. What you
get is forty minutes of training, a checkpoint that loads without complaint,
and a model that emits noise — with nothing, anywhere, recording when it broke.

So the rule is: the moment the loss stops being a finite number, stop and say
which step it happened on. A run that dies loudly at iteration 300 costs five
minutes. One that finishes quietly costs the lesson.

Gradient clipping is the other half, and it is worth being precise about which
job it does. Clipping does not repair a NaN that has already happened — nothing
does. It prevents the enormous update that usually produces one.

The functions that make decisions take plain floats rather than tensors, so
they can be exercised without torch or a GPU present. That matters here: the
machine that reads this code is usually not the machine that trains on it.
"""
from __future__ import annotations

import math

# nanoGPT, HuggingFace and most reference implementations all default to 1.0.
# There is nothing magic about it — small enough to flatten a spike, large
# enough to leave ordinary training alone.
DEFAULT_CLIP = 1.0


def is_poisoned(value: float) -> bool:
    """True if training can no longer learn anything from this loss.

    NaN and both infinities collapse to the same verdict: every weight update
    computed from here on is meaningless, so there is nothing to be gained by
    continuing.
    """
    return not math.isfinite(value)


def poisoned_report(step: int, value: float, grad_norm: float | None = None) -> str:
    """The message a student should see when a run is stopped.

    Names the step, because "it broke somewhere" is not actionable, and names
    the likely cause, because the fix is nearly always the learning rate.
    """
    what = "NaN" if math.isnan(value) else f"{value}"
    lines = [
        f"loss became {what} at iteration {step} — stopping.",
        "",
        "This is not a crash. Left alone, the run would have continued to the",
        "end and saved a checkpoint that loads fine and generates noise, with",
        "no record of when it went wrong. That is why it stops here instead.",
    ]
    if grad_norm is not None and math.isfinite(grad_norm):
        lines.append("")
        lines.append(f"Gradient norm on the last good step was {grad_norm:.3e}.")
        if grad_norm > 10 * DEFAULT_CLIP:
            lines.append("That is large — the gradients were already exploding before this.")
    lines += [
        "",
        "Usual cause is a learning rate too high for this batch size. Halve LR",
        "and run again.",
    ]
    return "\n".join(lines)


def format_layer_norms(norms: list[float]) -> str:
    """Per-layer gradient norms as one short line.

    A student is told "gradients vanish in deep networks" and has to take it on
    faith. These are the actual numbers. If the early layers read orders of
    magnitude smaller than the late ones, that is vanishing gradients happening
    in front of them rather than a claim in a textbook — and at four layers the
    whole picture fits on one line.
    """
    return " ".join(f"L{i}:{n:.1e}" for i, n in enumerate(norms))


# --- the two functions that need torch -------------------------------------
# Kept separate and thin, so everything above stays checkable on a laptop with
# no CUDA and no torch installed.

def clip_and_measure(model, max_norm: float = DEFAULT_CLIP) -> float:
    """Clip gradients in place; return the norm they had BEFORE clipping.

    Before, not after: the pre-clip norm is the diagnostic. After clipping it
    reads max_norm on every step that mattered, which tells you nothing.
    """
    import torch  # deferred, so importing this module never requires torch

    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total)


def layer_grad_norms(model) -> list[float]:
    """L2 norm of the gradients in each transformer block, early to late."""
    import torch  # noqa: PLC0415

    out = []
    for block in model.blocks:
        sq = [(p.grad.detach() ** 2).sum() for p in block.parameters() if p.grad is not None]
        out.append(float(torch.sqrt(sum(sq))) if sq else 0.0)
    return out
