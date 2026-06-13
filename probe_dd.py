"""
Delta-debug a claim against a trained uncertainty probe.

Question this answers: *which words does the probe actually rely on* to call a
claim known/unknown? We remove words from the claim and keep the ones the probe
needs to hold its verdict. If the minimal set is "2035", "research shows", etc.,
the probe is latching a lexical confound, not epistemic uncertainty.

Pipeline reuse:
  - probe          : trained inline on cached activations (linear_probes.py),
                     one (layer, point, pos) cell. Keep w, b, and the train
                     mean/std so we can score *new* reduced prompts.
  - scoring a claim: forward pass through the HF model (cache_activations.py
                     machinery) -> activation at pos -> standardize -> sigmoid.
  - minimization   : word-level ddmin (delta_debug from ddmin.py).

Interestingness (confirmed design): keep removing words while the probe's argmax
prediction stays == its prediction on the FULL claim. The continuous P(known)
score is logged at every test, so you also see the score drift even when the
label is unchanged.

Targeted ablations (cheaper than DD, run on the full claim):
  --strip-years : delete 4-digit years, report Delta P(known).
  --negate      : prepend a negation, report Delta P(known).

Note on temperature: irrelevant here. The probe reads a deterministic forward
pass, no sampling. Temperature only matters for the generation/steering phase.

Usage:
    python probe_dd.py --acts activations/gemma-2-9b-it/ \
        --model google/gemma-2-9b-it --layer 20 --point resid --pos -1 \
        --cell past_known --limit 3
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ddmin import delta_debug
from linear_probes import sigmoid, train_logreg, load_tensor
from cache_activations import (
    PROMPT_TEMPLATE, load_model, register_hooks, run_batch,
)


# ---------- probe (trained once, then frozen) ----------

class FrozenProbe:
    """Logreg probe + the train standardization stats, so it can score any
    new activation vector. mu/sd are kept explicitly (linear_probes.standardize
    folds them away, but here we need them for out-of-sample prompts)."""

    def __init__(self, X, y, **kw):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sd
        self.w, self.b = train_logreg(Xs, y, **kw)
        train_acc = ((sigmoid(Xs @ self.w + self.b) > 0.5).astype(int) == y).mean()
        print(f"Probe trained on {len(y)} rows  train_acc={train_acc:.3f}")

    def score(self, x):
        """x: [D] raw activation -> P(known) scalar."""
        xs = (x - self.mu) / self.sd
        return float(sigmoid(xs @ self.w + self.b))


# ---------- claim -> activation -> probe score ----------

class Scorer:
    """Holds the model + hooks; turns a claim string into P(known)."""

    def __init__(self, model, tok, buf, n_layers, n_pos, device,
                 layer, point, idx, probe):
        self.model, self.tok, self.buf = model, tok, buf
        self.n_layers, self.n_pos, self.device = n_layers, n_pos, device
        self.layer, self.point, self.idx, self.probe = layer, point, idx, probe
        self.trace = []   # (claim, p_known, pred) for every call

    def score(self, claim):
        prompt = PROMPT_TEMPLATE.format(claim=claim)
        out = run_batch(self.model, self.tok, [prompt], self.buf,
                        self.n_layers, self.n_pos, self.device)
        x = out[(self.layer, self.point)][0, self.idx, :].numpy()
        p = self.probe.score(x)
        self.trace.append((claim, p, int(p > 0.5)))
        return p


# ---------- targeted ablations ----------

def strip_years(text):
    cleaned = re.sub(r"\b\d{4}\b", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def negate(text):
    return f"It is not the case that {text[0].lower() + text[1:]}"


def diagnose(scorer):
    """Score content-free / degenerate inputs. If these come back 'known' with
    high confidence, the probe has a default bias and label-preserving ddmin is
    uninformative (it will trivially collapse any known claim to one word)."""
    probes = ["", "the", "a thing happened", "xyzzy qwerty",
              "asdf asdf asdf", "Something is true.", "Nothing is known."]
    print("\n--- degenerate-input diagnostic ---")
    for t in probes:
        p = scorer.score(t)
        print(f"  p(known)={p:.3f} pred={int(p > 0.5)}  \"{t}\"")
    print("--- (high p on empty/gibberish => default-known bias) ---")


# ---------- per-claim delta debug ----------

def dd_claim(claim, label, scorer):
    """Minimize claim to the words the probe needs to hold its full-claim verdict.
    Returns a result dict, or None if the full claim isn't a clean DD target."""
    words = claim.split()
    p_full = scorer.score(claim)
    ref_pred = int(p_full > 0.5)

    if ref_pred != label:
        print(f"  SKIP: probe mispredicts full claim "
              f"(p={p_full:.3f}, pred={ref_pred}, label={label})")
        return None

    def interesting(subset):
        if not subset:
            return False
        return int(scorer.score(" ".join(subset)) > 0.5) == ref_pred  # pred match

    # delta_debug asserts the full input is interesting; guaranteed above.
    minimal = delta_debug(interesting, words)
    p_min = scorer.score(" ".join(minimal))

    return {
        "label": label,
        "n_words": len(words),
        "n_minimal": len(minimal),
        "p_full": round(p_full, 4),
        "p_minimal": round(p_min, 4),
        "minimal_text": " ".join(minimal),
        "full_text": claim,
    }


# ---------- main ----------

def pick_targets(df, cells, limit, items):
    if items:
        return df[df["idx"].isin(items)]
    sub = df[df["cell"].isin(cells)]
    return sub.groupby("cell", group_keys=False).head(limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True,
                   help="Cached activations dir (to train the probe)")
    p.add_argument("--model", required=True, help="HF model id for forward passes")
    p.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    # target selection
    p.add_argument("--cells", nargs="+", default=["past_known"])
    p.add_argument("--limit", type=int, default=3, help="claims per cell")
    p.add_argument("--items", type=int, nargs="+", default=None,
                   help="explicit dataset idx values (overrides --cells/--limit)")
    # ablation modes
    p.add_argument("--strip-years", action="store_true")
    p.add_argument("--negate", action="store_true")
    p.add_argument("--diagnose", action="store_true",
                   help="Score degenerate inputs to check for default-known bias, then exit")
    # probe hyperparams (match linear_probes defaults)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--l2", type=float, default=1e-3)
    args = p.parse_args()

    # n_pos used when caching (so our forward-pass gather matches the probe's pos)
    cfg = json.loads((args.acts / "config.json").read_text())
    n_pos = cfg.get("n_pos", 5)
    idx = args.pos if args.pos >= 0 else n_pos + args.pos

    # --- train the frozen probe on cached activations ---
    meta = pd.read_parquet(args.acts / "meta.parquet")
    y = meta["label"].values.astype(int)
    T = load_tensor(args.acts, args.layer, args.point)   # [N, n_pos, D]
    X = T[:, idx, :]
    probe = FrozenProbe(X, y, lr=args.lr, epochs=args.epochs, l2=args.l2)

    # --- load model for scoring reduced prompts ---
    print(f"Loading {args.model} ({args.dtype}) for forward-pass scoring...")
    model, tok = load_model(args.model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # match cache_activations.py
    device = next(model.parameters()).device
    buf, handles, n_layers = register_hooks(model)

    scorer = Scorer(model, tok, buf, n_layers, n_pos, device,
                    args.layer, args.point, idx, probe)

    if args.diagnose:
        diagnose(scorer)
        for h in handles:
            h.remove()
        return

    # --- targets ---
    df = pd.read_csv(args.dataset)
    targets = pick_targets(df, args.cells, args.limit, args.items)
    print(f"{len(targets)} target claims  "
          f"(layer={args.layer} {args.point} pos={idx})")

    rows = []
    for r in targets.itertuples():
        print(f"\n[idx {r.idx} | {r.cell}] {r.claim}")

        if args.strip_years or args.negate:
            p_full = scorer.score(r.claim)
            print(f"  full         p(known)={p_full:.3f}")
            if args.strip_years:
                stripped = strip_years(r.claim)
                if stripped != r.claim:
                    p = scorer.score(stripped)
                    print(f"  -years       p(known)={p:.3f}  (d={p - p_full:+.3f})  \"{stripped}\"")
                    rows.append({"idx": r.idx, "cell": r.cell, "mode": "strip_years",
                                 "p_full": round(p_full, 4), "p_ablated": round(p, 4),
                                 "text": stripped})
                else:
                    print("  -years       (no year in claim)")
            if args.negate:
                neg = negate(r.claim)
                p = scorer.score(neg)
                print(f"  negated      p(known)={p:.3f}  (d={p - p_full:+.3f})  \"{neg}\"")
                rows.append({"idx": r.idx, "cell": r.cell, "mode": "negate",
                             "p_full": round(p_full, 4), "p_ablated": round(p, 4),
                             "text": neg})
        else:
            res = dd_claim(r.claim, int(r.label), scorer)
            if res is None:
                continue
            res.update({"idx": r.idx, "cell": r.cell, "mode": "ddmin"})
            print(f"  {res['n_words']} -> {res['n_minimal']} words   "
                  f"p {res['p_full']:.3f} -> {res['p_minimal']:.3f}")
            print(f"  minimal: \"{res['minimal_text']}\"")
            rows.append(res)

    for h in handles:
        h.remove()

    # --- write results next to the probe CSVs ---
    out_dir = args.acts.parent.parent / "probe_results" / args.acts.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "ablation" if (args.strip_years or args.negate) else "ddmin"
    out = out_dir / f"dd_{tag}_{args.point}_L{args.layer}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    # full score trajectory (every probe call), for the "score drifts as label holds" view
    traj = pd.DataFrame(scorer.trace, columns=["claim", "p_known", "pred"])
    traj_out = out_dir / f"dd_{tag}_{args.point}_L{args.layer}_trace.csv"
    traj.to_csv(traj_out, index=False)
    print(f"\nSaved -> {out}")
    print(f"Saved trace -> {traj_out}")


if __name__ == "__main__":
    main()
