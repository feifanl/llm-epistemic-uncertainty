"""
Architecture adapter: one place that knows where each model family keeps its
decoder layers and the attn/mlp submodules, so cache_activations.py and steer.py
don't hardcode `model.model.layers` / `layer.self_attn`.

Two families covered:
  - llama-like  (Gemma, Qwen2, Llama):  model.model.layers[i] -> .self_attn / .mlp
  - gpt2-like   (GPT-2):                model.transformer.h[i] -> .attn / .mlp

The hook points are the same conceptually in both:
  - resid: the decoder BLOCK's forward output (post-residual hidden state)
  - attn:  the self-attention submodule's output
  - mlp:   the MLP submodule's output
Block + attn outputs are tuples (hidden, ...); mlp output is a bare tensor. Both
cache and steer already handle `out[0] if isinstance(out, tuple) else out`, so the
adapter only needs to hand back the right *module* to hook.

Detection is by attribute presence, not a model_type whitelist, so any future
llama-shaped model works without edits.
"""
from __future__ import annotations


def is_gpt2_like(model) -> bool:
    """True for GPT-2 (decoder stack at model.transformer.h)."""
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")


def is_llama_like(model) -> bool:
    """True for Gemma / Qwen2 / Llama (stack at model.model.layers)."""
    return hasattr(model, "model") and hasattr(model.model, "layers")


def get_layers(model):
    """Return the indexable list of decoder blocks."""
    if is_llama_like(model):
        return model.model.layers
    if is_gpt2_like(model):
        return model.transformer.h
    raise ValueError(
        f"Unsupported architecture {type(model).__name__}: no model.model.layers "
        "or model.transformer.h"
    )


def block_module(layer):
    """Module whose forward output is the post-residual hidden state ('resid').

    The decoder block itself is that module in both families.
    """
    return layer


def attn_module(layer):
    """Self-attention submodule ('attn' hook point)."""
    if hasattr(layer, "self_attn"):      # llama-like
        return layer.self_attn
    if hasattr(layer, "attn"):           # gpt2-like
        return layer.attn
    raise ValueError(f"No attn submodule on {type(layer).__name__}")


def mlp_module(layer):
    """MLP submodule ('mlp' hook point). Both families name it `.mlp`."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise ValueError(f"No mlp submodule on {type(layer).__name__}")


def d_model(model) -> int:
    """Residual-stream width, from config (hidden_size / n_embd)."""
    cfg = model.config
    for attr in ("hidden_size", "n_embd"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise ValueError(f"Can't find hidden size on config {type(cfg).__name__}")


def has_chat_template(tok) -> bool:
    """Whether the tokenizer carries a chat template (GPT-2 does not)."""
    return getattr(tok, "chat_template", None) is not None
