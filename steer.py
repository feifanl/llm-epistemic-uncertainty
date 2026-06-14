"""
Steer the model along the known<->unknown direction and watch generation shift.

This is the causal test the probe alone can't give. The probe *decodes* an
uncertainty direction from activations; steering *injects* it and asks whether
the model's behavior moves the way the direction claims it should. If adding the
"unknown" direction makes the model hedge on an unrelated history/cooking
question, the direction is causal for expressed uncertainty — not a decoded
correlate of the dataset.

Direction (default: diff-of-means, the robust steering choice):
    v = mean(act | known) - mean(act | unknown)     # raw activation space
at one (layer, point, pos), from cached activations. alpha=+1 adds one full
known-minus-unknown gap (toward known/confident); alpha<0 steers toward unknown.

Injection: a forward hook on decoder layer `layer` adds alpha * v to the residual
stream (the layer's output hidden state) at every position, every decode step --
the same point the probe reads.

Note: greedy decoding (do_sample=False) so any output change is attributable to
the steering vector, not sampling noise.

Usage:
    python steer.py --acts activations/gemma-2-9b-it/ --model google/gemma-2-9b-it \
        --layer 20 --point resid --pos -1 --alphas -8 -4 0 4 8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from linear_probes import sigmoid, train_logreg, load_tensor
from cache_activations import load_model

# Neutral prompts: topics unrelated to the AI/education dataset. If steering
# toward "unknown" injects hedging / AI-in-education / year mentions here, the
# direction carries dataset-specific baggage. If it cleanly modulates
# confidence vs hedging, it's a general uncertainty knob.
DEFAULT_PROMPTS = [
    "What year did the French Revolution begin?",
    "How do I make a basic tomato sauce?",
    "What causes the seasons on Earth?",
    "Who wrote the play Hamlet?",
    "Will it rain in Chicago next Tuesday?",
]

HEDGE_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "likely", "unlikely",
    "uncertain", "unclear", "depends", "hard to say", "not sure", "i'm not",
    "cannot be certain", "no way to know", "difficult to predict", "speculative",
]


# ---------- steering vector ----------

def steering_vector(X, y, method):
    """Return (v [D] float32, info str). X raw acts, y in {0,1} (1=known)."""
    if method == "diffmean":
        v = X[y == 1].mean(0) - X[y == 0].mean(0)
        return v.astype(np.float32), f"diffmean |v|={np.linalg.norm(v):.2f}"
    # probe: logit gradient wrt raw x is w/sd; that's the "more known" direction
    mu, sd = X.mean(0), X.std(0) + 1e-8
    w, b = train_logreg((X - mu) / sd, y, lr=0.1, epochs=300, l2=1e-3)
    v = (w / sd).astype(np.float32)
    # scale probe dir to the diffmean magnitude so alpha is comparable
    dm = X[y == 1].mean(0) - X[y == 0].mean(0)
    v = v / np.linalg.norm(v) * np.linalg.norm(dm)
    return v, f"probe(scaled to diffmean) |v|={np.linalg.norm(v):.2f}"


# ---------- steering hook ----------

class Steerer:
    """Adds alpha * v to one decoder layer's output residual, all positions."""

    def __init__(self, model, layer_idx, v, device, dtype):
        self.layer = model.model.layers[layer_idx]
        self.v = torch.tensor(v, device=device, dtype=dtype)
        self.alpha = 0.0
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.alpha == 0.0:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + self.alpha * self.v
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.handle.remove()


# ---------- generation ----------

def build_input(tok, question, device):
    msgs = [{"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    return ids.to(device)


@torch.no_grad()
def generate(model, tok, ids, max_new):
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def hedge_count(text):
    t = text.lower()
    return sum(t.count(h) for h in HEDGE_WORDS)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--method", default="diffmean", choices=["diffmean", "probe"])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-8, -4, 0, 4, 8])
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--prompts", type=Path, default=None,
                   help="Optional text file, one neutral question per line")
    args = p.parse_args()

    cfg = json.loads((args.acts / "config.json").read_text())
    n_pos = cfg.get("n_pos", 5)
    idx = args.pos if args.pos >= 0 else n_pos + args.pos

    # --- build steering vector from cached acts ---
    meta = pd.read_parquet(args.acts / "meta.parquet")
    y = meta["label"].values.astype(int)
    X = load_tensor(args.acts, args.layer, args.point)[:, idx, :]
    v, info = steering_vector(X, y, args.method)
    print(f"Steering vector: {info}  (layer={args.layer} {args.point} pos={idx})")

    # --- model ---
    print(f"Loading {args.model} ({args.dtype})...")
    model, tok = load_model(args.model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    steerer = Steerer(model, args.layer, v, device, dtype)

    prompts = (args.prompts.read_text().splitlines() if args.prompts
               else DEFAULT_PROMPTS)
    prompts = [q.strip() for q in prompts if q.strip()]

    rows = []
    for q in prompts:
        ids = build_input(tok, q, device)
        print(f"\n{'='*70}\nQ: {q}")
        for a in args.alphas:
            steerer.alpha = float(a)
            text = generate(model, tok, ids, args.max_new_tokens)
            hc = hedge_count(text)
            tag = "(baseline)" if a == 0 else ("(->known)" if a > 0 else "(->unknown)")
            print(f"\n  alpha={a:+.0f} {tag}  hedges={hc}\n  {text}")
            rows.append({"question": q, "alpha": a, "hedges": hc, "text": text})
    steerer.alpha = 0.0
    steerer.remove()

    out_dir = args.acts.parent.parent / "probe_results" / args.acts.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"steer_{args.method}_{args.point}_L{args.layer}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
