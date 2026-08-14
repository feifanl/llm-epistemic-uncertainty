"""
Representational necessity test for the known<->unknown direction.

Steering proved SUFFICIENCY (inject the direction -> behavior moves). The
generation-side necessity test is confounded: on future-dated prompts the model
hedges because the *content* is unknowable, not because of the internal
direction, so ablating the direction at decode time changes little. And the
lexical hedge counter is anti-correlated with truth on those outputs.

So test necessity where the confounds don't live: in the representation itself.
Take the cached activations, project out the diff-of-means direction v_hat, and
ask whether a linear probe can still decode known vs unknown. If AUROC collapses
toward 0.5, the direction is NECESSARY to linearly read uncertainty off the
residual. If AUROC holds, the signal is redundant / distributed across other
directions.

Specificity control (the load-bearing part): ablate a RANDOM unit direction of
the same dimension. Removing one arbitrary axis from a high-dim space should NOT
hurt the probe. If the random ablation drops AUROC as much as v_hat does, the
collapse is just dimensionality loss, not evidence about v_hat. The claim is
real only if  drop(v_hat) >> drop(random).

v_hat is fit on TRAIN rows only (no leakage), then projected out of train and
test alike. Zero- and mean-ablation are identical here: both kill the variance
along v_hat, and standardize() absorbs the constant offset.

Usage:
    python ablate_probe.py --acts activations/gemma-2-9b-it/ --point resid --pos -1
    python ablate_probe.py --acts activations/gemma-2-9b-it/ --point resid --transfer
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from linear_probes import (
    sigmoid, standardize, train_logreg, load_tensor, discover_layers, make_masks,
    roc_auc,
)


def diffmean_unit(X, y, tr):
    """Unit diff-of-means direction from TRAIN rows. v_hat in raw act space."""
    m1 = X[tr & (y == 1)].mean(0)
    m0 = X[tr & (y == 0)].mean(0)
    v = m1 - m0
    return v / (np.linalg.norm(v) + 1e-12)


def ablate(X, vhat):
    """Remove the component of every row along vhat (zero the axis)."""
    return X - np.outer(X @ vhat, vhat)


def probe_auroc(X, y, tr, te, **kw):
    Xtr, Xte = standardize(X[tr], X[te])
    w, b = train_logreg(Xtr, y[tr], **kw)
    return roc_auc(y[te], sigmoid(Xte @ w + b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to test (default: all cached).")
    p.add_argument("--transfer", action="store_true",
                   help="Cross-time transfer (train past/test future + reverse) "
                        "instead of random 80/20 split.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random-direction control.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--l2", type=float, default=1e-3)
    args = p.parse_args()

    meta = pd.read_parquet(args.acts / "meta.parquet")
    directions, y = make_masks(meta, "transfer" if args.transfer else "split")
    layers = args.layers or discover_layers(args.acts, args.point)
    kw = dict(lr=args.lr, epochs=args.epochs, l2=args.l2)
    rng = np.random.default_rng(args.seed)
    mode = "transfer" if args.transfer else "split"

    cfg_npos = load_tensor(args.acts, layers[0], args.point).shape[1]
    idx = args.pos if args.pos >= 0 else cfg_npos + args.pos
    print(f"Necessity (ablate v_hat) [{mode}] {args.point} pos{idx}, "
          f"{len(layers)} layers\n"
          f"{'L':>3} {'dir':>4} {'full':>6} {'ablated':>8} {'random':>7} "
          f"{'drop':>6} {'specific?':>9}")

    rows = []
    for layer in layers:
        X = load_tensor(args.acts, layer, args.point)[:, idx, :]
        rand = rng.standard_normal(X.shape[1])
        rand = rand / np.linalg.norm(rand)
        for name, tr, te in directions:
            vhat = diffmean_unit(X, y, tr)
            a_full = probe_auroc(X, y, tr, te, **kw)
            a_abl = probe_auroc(ablate(X, vhat), y, tr, te, **kw)
            a_rnd = probe_auroc(ablate(X, rand), y, tr, te, **kw)
            drop = a_full - a_abl
            specific = (a_full - a_abl) - (a_full - a_rnd)   # excess over control
            flag = "YES" if drop > 0.1 and specific > 0.05 else "no"
            print(f"{layer:>3} {name:>4} {a_full:>6.3f} {a_abl:>8.3f} "
                  f"{a_rnd:>7.3f} {drop:>6.3f} {flag:>9}")
            rows.append({"layer": layer, "direction": name, "auroc_full": a_full,
                         "auroc_ablated": a_abl, "auroc_random": a_rnd,
                         "drop": drop, "drop_specific": specific})

    out_dir = args.acts.parent.parent / "probe_results" / args.acts.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ablate_necessity_{args.point}_{mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print("Read: necessity holds where 'ablated' falls toward 0.5 AND 'random' "
          "stays near 'full' (drop is specific to v_hat, not dimensionality).")


if __name__ == "__main__":
    main()
