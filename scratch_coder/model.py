"""
model.py — a GPT written from scratch (our own AI architecture).

This is a small decoder-only Transformer (the same family as GPT/Llama), built
here line by line — no pretrained weights, nothing downloaded. Character-level.
Educational: this is *how an LLM actually works* under the hood.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """Hyperparameters that define the model's size and shape."""

    def __init__(self, vocab_size: int, block_size: int = 128, n_layer: int = 4,
                 n_head: int = 4, n_embd: int = 128, dropout: float = 0.0,
                 mlp: str = "default"):
        self.vocab_size = vocab_size    # number of distinct characters
        self.block_size = block_size    # context length (chars the model can see)
        self.n_layer = n_layer          # transformer blocks stacked
        self.n_head = n_head            # attention heads
        self.n_embd = n_embd            # embedding width
        self.dropout = dropout
        self.mlp = mlp                  # which block MLP to build (see MLP_REGISTRY)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention — each position looks only at the past."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head = C // self.n_head
        q = q.view(B, T, self.n_head, head).transpose(1, 2)
        k = k.view(B, T, self.n_head, head).transpose(1, 2)
        v = v.view(B, T, self.n_head, head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """Per-position feed-forward network."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


# The one extension seam. The Block asks this registry for its MLP by name. Not a
# plugin framework — a dict and a name in the config. Drop a file in layers/ that
# calls @register_mlp("yourname"), set config.mlp to it, and the Knob Matrix redraws
# with your tensors. Kept to the MLP alone so the architecture has exactly one seam.
MLP_REGISTRY: dict[str, type] = {"default": MLP}


def register_mlp(name: str):
    """Decorator: register a block-MLP class under `name` (used from layers/)."""
    def deco(cls):
        MLP_REGISTRY[name] = cls
        return cls
    return deco


class Block(nn.Module):
    """One transformer block: attention + MLP, each with a residual connection."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP_REGISTRY[getattr(cfg, "mlp", "default")](cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The whole model: embeddings -> N transformer blocks -> next-char prediction."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int | None = 40) -> torch.Tensor:
        """Autoregressively sample new characters, one at a time."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def _load_custom_layers() -> None:
    """Import every module in ./layers so its @register_mlp calls take effect,
    making a dropped-in layer available by name. Defensive on purpose: a broken
    layer file prints a note and is skipped — it must never take down the whole
    model import, which train.py, the exporter and everything else depend on."""
    import importlib.util

    d = Path(__file__).resolve().parent / "layers"
    if not d.is_dir():
        return
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"scratch_layers.{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:   # noqa: BLE001 — one bad layer must not break the model
            print(f"[model] skipped custom layer {f.name}: {e}")


_load_custom_layers()
