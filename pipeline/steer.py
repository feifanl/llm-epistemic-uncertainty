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

Ablation (the necessity test, complementary to steering's sufficiency): instead
of adding the direction, remove the residual's component along it and watch
whether the model loses its known/unknown stance. Run on the dataset's own
prompts (--prompts), where the model has a natural confidence level to lose.

Usage:
    # steering (sufficiency)
    python steer.py --acts activations/gemma-2-9b-it/ --model google/gemma-2-9b-it \
        --layer 20 --point resid --pos -1 --alphas -8 -4 0 4 8

    # ablation (necessity); 0=baseline, 1=full mean-ablation
    python steer.py --acts activations/gemma-2-9b-it/ --model google/gemma-2-9b-it \
        --layer 20 --point resid --pos -1 --mode ablate --prompts unknown_qs.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from linear_probes import sigmoid, train_logreg, load_tensor
from cache_activations import load_model
from model_adapter import get_layers, has_chat_template

# Neutral prompts: topics unrelated to the AI/education dataset. If steering
# toward "unknown" injects hedging / AI-in-education / year mentions here, the
# direction carries dataset-specific baggage. If it cleanly modulates
# confidence vs hedging, it's a general uncertainty knob.
DEFAULT_PROMPTS = [
    # hard facts: single correct answer. Steering toward "unknown" should
    # manufacture doubt where none is warranted -> strongest signal if it does.
    "What year did the French Revolution begin?",
    "Who wrote the play Hamlet?",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
    "What is the boiling point of water at sea level in Celsius?",
    "What is the capital of Australia?",
    # explanatory: correct but room to caveat. Mid sensitivity expected.
    "What causes the seasons on Earth?",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What causes inflation in an economy?",
    # procedural: how-to, little to hedge. Control group.
    "How do I make a basic tomato sauce?",
    "How do I tie a shoelace?",
    # genuinely open / predictive: model SHOULD hedge at baseline already.
    # Steering toward "known" is the interesting direction here.
    "Will it rain in Chicago next Tuesday?",
    "Who will win the next World Cup?",
    "Is there life on other planets?",
    "What is the best programming language?",
]

HEDGE_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "likely", "unlikely",
    "uncertain", "unclear", "depends", "hard to say", "not sure", "i'm not",
    "cannot be certain", "no way to know", "difficult to predict", "speculative",
    # denial / epistemic-doubt register the steered model actually emits:
    "no single", "no definitive", "no consensus", "no clear", "no evidence",
    "myth", "impossible to predict", "can't predict", "cannot predict",
    "no straightforward", "complex", "controversial", "not a singular",
    "oversimplification", "no scientific consensus",
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
    """Hook on one decoder layer's output residual, all positions.

    mode="add":    hs += alpha * v                      (steering / injection)
    mode="ablate": removes the hs component along v_hat  (causal necessity test)
        zero-ablation: hs -= alpha * (hs.v_hat) v_hat
        mean-ablation: hs -= alpha * (hs.v_hat - mbar) v_hat   (on-distribution;
            replaces the projection with its dataset mean instead of zero, so the
            residual stays on the manifold the model expects)
    alpha=0 is baseline (no-op) in both modes; alpha=1 = full ablation.
    """

    def __init__(self, model, layer_idx, v, device, dtype,
                 mode="add", ablate_kind="mean", mbar=0.0):
        self.layer = get_layers(model)[layer_idx]
        self.v = torch.tensor(v, device=device, dtype=dtype)
        self.vhat = self.v / self.v.norm()
        self.mbar = float(mbar)
        self.mode = mode
        self.ablate_kind = ablate_kind
        self.alpha = 0.0
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        if self.alpha == 0.0:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        if self.mode == "add":
            hs = hs + self.alpha * self.v
        else:  # ablate
            proj = (hs * self.vhat).sum(-1, keepdim=True)        # hs . v_hat
            target = self.mbar if self.ablate_kind == "mean" else 0.0
            hs = hs - self.alpha * (proj - target) * self.vhat
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    def remove(self):
        self.handle.remove()


# ---------- generation ----------

def build_input(tok, question, device):
    # GPT-2 (no chat template) -> plain QA framing so the base model answers
    # instead of free-associating; chat models use their template.
    if not has_chat_template(tok):
        enc = tok(f"Q: {question}\nA:", return_tensors="pt").input_ids
        return enc.to(device)
    msgs = [{"role": "user", "content": question}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    if not torch.is_tensor(enc):          # newer transformers returns a dict
        enc = enc["input_ids"]
    return enc.to(device)


@torch.no_grad()
def generate(model, tok, ids, max_new):
    out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def hedge_count(text):
    t = text.lower()
    return sum(t.count(h) for h in HEDGE_WORDS)


def alpha_tag(a, mode):
    """Human label for a steering scalar, shared by console + markdown."""
    if a == 0:
        return "(baseline)"
    if mode == "ablate":
        return "(ablated)" if a == 1 else f"(ablate x{a:g})"
    return "(->known)" if a > 0 else "(->unknown)"


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="ablate mode only: ablate a PER-LAYER diffmean at each of "
                        "these layers (e.g. --layers 12 13 .. 26). Single-site "
                        "ablation is rewritten downstream; a band is the real "
                        "necessity test. Defaults to [--layer].")
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--method", default="diffmean", choices=["diffmean", "probe"])
    p.add_argument("--mode", default="add", choices=["add", "ablate"],
                   help="add=inject direction (sufficiency); ablate=remove it "
                        "(necessity)")
    p.add_argument("--ablate-kind", default="mean", choices=["mean", "zero"],
                   help="mean=replace v-projection with dataset mean (on-manifold);"
                        " zero=project the direction out entirely")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="add mode: steering scalars (default -8..8). ablate mode: "
                        "ablation strength, 1=full (default 0 1)")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--prompts", type=Path, default=None,
                   help="Optional text file, one neutral question per line")
    p.add_argument("--out", type=Path, default=None,
                   help="Markdown output path (default: repo-root steering_results.md)")
    args = p.parse_args()
    if args.alphas is None:
        args.alphas = [0.0, 1.0] if args.mode == "ablate" else [-8, -4, 0, 4, 8]

    cfg = json.loads((args.acts / "config.json").read_text())
    n_pos = cfg.get("n_pos", 5)
    idx = args.pos if args.pos >= 0 else n_pos + args.pos

    # --- build steering vector from cached acts ---
    meta = pd.read_parquet(args.acts / "meta.parquet")
    y = meta["label"].values.astype(int)
    X = load_tensor(args.acts, args.layer, args.point)[:, idx, :]
    v, info = steering_vector(X, y, args.method)
    # mean projection onto v_hat across the dataset -> on-manifold ablation target
    vhat = v / np.linalg.norm(v)
    mbar = float((X @ vhat).mean())
    print(f"Steering vector: {info}  (layer={args.layer} {args.point} pos={idx})")
    print(f"Mode: {args.mode}"
          + (f" ({args.ablate_kind}-ablation, mbar={mbar:.2f})"
             if args.mode == "ablate" else ""))

    # --- model ---
    print(f"Loading {args.model} ({args.dtype})...")
    model, tok = load_model(args.model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Steering = single site at --layer. Ablation can span a band of layers
    # (--layers): each gets its OWN per-layer diffmean removed, because a single
    # site's ablation is re-introduced by downstream layers' residual writes.
    steer_layers = (args.layers if (args.mode == "ablate" and args.layers)
                    else [args.layer])
    steerers = []
    for L in steer_layers:
        if L == args.layer:
            vL, mbarL = v, mbar
        else:
            XL = load_tensor(args.acts, L, args.point)[:, idx, :]
            vL, _ = steering_vector(XL, y, args.method)
            mbarL = float((XL @ (vL / np.linalg.norm(vL))).mean())
        steerers.append(Steerer(model, L, vL, device, dtype, mode=args.mode,
                                ablate_kind=args.ablate_kind, mbar=mbarL))
    if len(steerers) > 1:
        print(f"Multi-layer ablation across {len(steerers)} layers: {steer_layers}")

    def set_alpha(a):
        for s in steerers:
            s.alpha = float(a)

    prompts = (args.prompts.read_text().splitlines() if args.prompts
               else DEFAULT_PROMPTS)
    prompts = [q.strip() for q in prompts if q.strip()]

    rows = []
    for q in prompts:
        ids = build_input(tok, q, device)
        print(f"\n{'='*70}\nQ: {q}")
        for a in args.alphas:
            set_alpha(a)
            text = generate(model, tok, ids, args.max_new_tokens)
            hc = hedge_count(text)
            tag = alpha_tag(a, args.mode)
            print(f"\n  alpha={a:+.0f} {tag}  hedges={hc}\n  {text}")
            rows.append({"question": q, "mode": args.mode, "alpha": a,
                         "hedges": hc, "text": text})
    set_alpha(0.0)
    for s in steerers:
        s.remove()

    # --- markdown writeup ---
    ltag = (f"{min(steer_layers)}-{max(steer_layers)}" if len(steer_layers) > 1
            else str(args.layer))
    lines = [
        "# Steering results", "",
        f"Model: {args.model}",
        f"Layer: {ltag}, {args.point}",
        f"Pos: {idx}",
        f"Mode: {args.mode}"
        + (f" ({args.ablate_kind}-ablation)" if args.mode == "ablate" else ""),
        f"Method: {args.method}",
        f"Vector: {info}",
        f"Scalar values: {', '.join(f'{a:g}' for a in args.alphas)}",
        "",
    ]
    for q in prompts:
        lines += ["=" * 70, f"Q: {q}", ""]
        for r in (row for row in rows if row["question"] == q):
            tag = alpha_tag(r["alpha"], r["mode"])
            lines.append(f"  alpha={r['alpha']:+.0f} {tag}  hedges={r['hedges']}")
            lines.append(f"  {r['text']}")
            lines.append("")

    out = args.out or (args.acts.parent.parent / "steering_results.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
