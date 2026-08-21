"""
example_layer.py — a template for a custom block MLP.

model.py imports every *.py in this folder on startup, so a layer that calls
@register_mlp("name") becomes available by that name. Select it with the config's
mlp field — `python train.py --mlp example` — and the Knob Matrix in the web
inspector redraws with this layer's tensors.

This example is a gated MLP (SwiGLU-style): TWO input projections instead of one,
where c_gate gates c_fc. It is deliberately a different shape from the default —
three weight tensors, not two — so selecting it visibly changes the stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import register_mlp


@register_mlp("example")
class GatedMLP(nn.Module):
    """A gated feed-forward block. Same width in and out as the default MLP, so it
    drops into the residual stream unchanged; different internals, so the map shows
    a c_gate tensor the default does not have."""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_gate = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))
