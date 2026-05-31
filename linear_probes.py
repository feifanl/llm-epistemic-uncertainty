"""
Logistic-regression probes on cached activations.

Goal: for each layer, fit a linear probe that classifies known (1) vs
unknown (0) from the residual/MLP/attn activation at one token position.
The interesting result is the *layer-vs-accuracy curve* and *cross-time
transfer* (train on past, test on future), not in-distribution accuracy.

Usage:
    python linear_probes.py --acts activations/gemma-2-9b-it/ --point resid --pos -1
    python linear_probes.py --acts activations/gemma-2-9b-it/ --point resid --transfer
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------- the probe ----------

def sigmoid(z):
    # clip to avoid overflow in exp for large |z|
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg(X, y, lr=0.1, epochs=300, l2=1e-3, verbose=False):
    """
    Hand-rolled binary logistic regression via full-batch gradient descent.

    X: [N, D] activations (ASSUMED already standardized — see standardize()).
    y: [N]   labels in {0, 1}.

    Model:   p = sigmoid(X @ w + b)              # predicted P(y=1)
    Loss:    mean cross-entropy + l2 * ||w||^2   # continuous, differentiable
    Grad:    dL/dw = X.T @ (p - y) / N + 2*l2*w  # the (p - y) error term is the key
             dL/db = mean(p - y)
    Step:    w -= lr * dL/dw                      # walk downhill on the loss
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0

    for epoch in range(epochs):
        # --- forward: activations -> probability ---
        z = X @ w + b               # [N]  linear score (logit)
        p = sigmoid(z)              # [N]  squashed to (0,1)

        # --- loss (only for logging; the grad below doesn't need it) ---
        eps = 1e-9
        ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss = ce + l2 * np.sum(w * w)

        # --- backward: gradient of loss wrt params ---
        error = p - y               # [N]  how wrong + which direction
        grad_w = X.T @ error / N + 2 * l2 * w
        grad_b = np.mean(error)

        # --- step: nudge params opposite the gradient ---
        w -= lr * grad_w
        b -= lr * grad_b

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            acc = ((p > 0.5).astype(int) == y).mean()
            print(f"    epoch {epoch:4d}  loss {loss:.4f}  train_acc {acc:.3f}")

    return w, b


def predict(X, w, b):
    return (sigmoid(X @ w + b) > 0.5).astype(int)


def accuracy(X, y, w, b):
    return (predict(X, w, b) == y).mean()


# ---------- data ----------

def standardize(X_train, X_test=None):
    """
    Zero-mean unit-variance per feature, using TRAIN stats only.
    Critical: raw activations have large/uneven magnitude; without this,
    gradient descent crawls or diverges. (sklearn hides this behind solvers.)
    """
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xtr = (X_train - mu) / sd
    if X_test is None:
        return Xtr
    return Xtr, (X_test - mu) / sd


def load_tensor(acts_dir, layer, point):
    """Return full cached tensor [N, n_pos, D] for one layer (load once, slice per pos)."""
    t = torch.load(acts_dir / f"L{layer}_{point}.pt", map_location="cpu")
    return t.float().numpy()


def discover_layers(acts_dir, point):
    files = sorted(acts_dir.glob(f"L*_{point}.pt"))
    return sorted(int(f.stem.split("_")[0][1:]) for f in files)


def make_masks(meta, mode):
    """
    Return list of (name, train_mask, test_mask) directions to evaluate.
    'split'    -> one random 80/20 direction.
    'transfer' -> both cross-time directions (past->future, future->past).
    Plus the shared label vector y.
    """
    y = meta["label"].values.astype(int)   # 1=known, 0=unknown
    if mode == "transfer":
        time = meta["cell"].str.split("_").str[0].values   # 'past' / 'future'
        past, future = (time == "past"), (time == "future")
        return [("p2f", past, future), ("f2p", future, past)], y
    # in-distribution: random 80/20
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr = np.zeros(len(y), bool); tr[perm[:cut]] = True
    return [("split", tr, ~tr)], y


def fit_eval(X, y, tr, te, **kw):
    """Standardize on train, fit probe, return (train_acc, test_acc)."""
    Xtr, Xte = standardize(X[tr], X[te])
    w, b = train_logreg(Xtr, y[tr], **kw)
    return accuracy(Xtr, y[tr], w, b), accuracy(Xte, y[te], w, b)


# ---------- experiments ----------

def run(acts_dir, point, positions, meta, mode, **kw):
    """
    Train a probe for every (layer, pos) cell. Return DataFrame of results.
    Transfer mode evaluates both directions (p2f, f2p) per cell, side by side.
    """
    directions, y = make_masks(meta, mode)
    layers = discover_layers(acts_dir, point)
    print(f"Probe [{mode}] {point} — {len(layers)} layers × {len(positions)} pos "
          f"× {len(directions)} dir")

    rows = []
    for layer in layers:
        T = load_tensor(acts_dir, layer, point)   # [N, n_pos, D]
        for pos in positions:
            idx = pos if pos >= 0 else T.shape[1] + pos
            X = T[:, idx, :]
            row = {"layer": layer, "pos": idx}
            for name, tr, te in directions:
                tr_acc, te_acc = fit_eval(X, y, tr, te, **kw)
                row[f"{name}_train"] = tr_acc
                row[f"{name}_test"] = te_acc
            rows.append(row)
            summary = "  ".join(f"{n}_test {row[f'{n}_test']:.3f}"
                                for n, _, _ in directions)
            print(f"  L{layer:<2d} pos{idx}  {summary}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1)
    p.add_argument("--all-pos", action="store_true",
                   help="Sweep every cached position instead of just --pos")
    p.add_argument("--transfer", action="store_true",
                   help="Cross-time transfer instead of in-distribution split")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--l2", type=float, default=1e-3)
    args = p.parse_args()

    meta = pd.read_parquet(args.acts / "meta.parquet")
    kw = dict(lr=args.lr, epochs=args.epochs, l2=args.l2)
    mode = "transfer" if args.transfer else "split"

    if args.all_pos:
        layers = discover_layers(args.acts, args.point)
        n_pos = load_tensor(args.acts, layers[0], args.point).shape[1]
        positions = list(range(n_pos))
    else:
        positions = [args.pos]

    df = run(args.acts, args.point, positions, meta, mode, **kw)

    out_dir = args.acts / "probes"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"probe_{args.point}_{mode}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()