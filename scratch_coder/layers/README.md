# Custom block layers

`model.py` imports every `*.py` in this folder on startup, so a layer that calls
`@register_mlp("name")` becomes available by that name. Select it with the config's
`mlp` field:

```bash
python train.py --mlp example        # train with the example gated MLP
python export_inspect.py --ckpt data/ckpt.pt   # then look at it in the inspector
```

The Knob Matrix in the web inspector (`web/`) redraws with your layer's tensors.

## One seam, on purpose

The only swappable part is the block's **MLP**. Attention, the embeddings and the
residual stream are fixed. That is enough for a student to change the architecture
and watch the parameter map change shape — without a plugin framework, an entry
point system, or a config schema. A dict and a name.

## Template

See `example_layer.py`: a gated (SwiGLU-style) MLP with three weight tensors instead
of the default two.

```python
from model import register_mlp

@register_mlp("example")
class GatedMLP(nn.Module):
    def __init__(self, cfg): ...      # build from cfg.n_embd
    def forward(self, x): ...          # return a tensor the same width as x
```

The one rule: keep the width the same in and out (`cfg.n_embd` → `cfg.n_embd`), so
it drops into the residual stream unchanged.

## Honesty

`arch_map.py` computes the tensor map from the architecture alone, with no
checkpoint — so it cannot know a custom layer's tensors. When you select one it
**says so** rather than drawing the default's tensors and lying about the shape; the
inspector then takes the real shapes from the exported checkpoint instead. A layer
file that fails to import is skipped with a printed note, never silently, and never
by taking down the whole model import.
