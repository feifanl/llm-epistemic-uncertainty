"""
PCA 2D scatter per layer, colored by cell. Pilot viz for activations.

Usage:
    python viz/visualize_activations.py --acts activations/gemma-2-2b-it/ --point resid --pos -1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

CELL_COLORS = {
    "past_known":     "#1f77b4",  # blue
    "past_unknown":   "#ff7f0e",  # orange
    "future_known":   "#2ca02c",  # green
    "future_unknown": "#d62728",  # red
}


def pick_layers(acts, point, layers):
    """Discover layers; default to 12 evenly spaced."""
    files = sorted(acts.glob(f"L*_{point}.pt"))
    all_layers = sorted(int(f.stem.split("_")[0][1:]) for f in files)
    if layers is not None:
        return layers
    n = len(all_layers)
    idxs = np.linspace(0, n - 1, min(12, n)).astype(int)
    return [all_layers[i] for i in idxs]


def plot_pos(acts, point, pos, layers, meta, out=None):
    """One PCA-grid PNG for a single token position.

    Writes to {repo}/plots/pca/{model}_pca_{point}_pos{pos}.png by default.
    """
    model = acts.name                       # activations/{model}/
    root = acts.parent.parent               # repo root
    n = len(layers)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)

    for ax, layer in zip(axes.flat, layers):
        t = torch.load(acts / f"L{layer}_{point}.pt",
                       map_location="cpu")  # [N, n_pos, d]
        idx = pos if pos >= 0 else t.shape[1] + pos
        X = t[:, idx, :].numpy()              # [N, d]

        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_

        for cell, color in CELL_COLORS.items():
            mask = (meta["cell"] == cell).values
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, label=cell,
                       s=18, alpha=0.7, edgecolors="none")

        ax.set_title(f"L{layer}  (PC1 {ev[0]:.2f}, PC2 {ev[1]:.2f})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.flat[n:]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8, label=k)
               for k, c in CELL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)

    fig.suptitle(f"PCA per layer — {model} | {point} | pos={pos}",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    if out is None:
        pca_dir = root / "plots" / "pca" / model
        pca_dir.mkdir(parents=True, exist_ok=True)
        out = pca_dir / f"pca_{point}_pos{pos}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True,
                   help="activations/{model}/ dir")
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1,
                   help="Which cached position (-1 = last token)")
    p.add_argument("--all-pos", action="store_true",
                   help="Loop every cached position, one PNG each")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Subset of layers to plot. Default: evenly-spaced 12")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    meta = pd.read_parquet(args.acts / "meta.parquet")
    print(f"{len(meta)} prompts, cells: {meta['cell'].value_counts().to_dict()}")

    layers = pick_layers(args.acts, args.point, args.layers)
    print(f"Plotting layers: {layers}")

    if args.all_pos:
        # discover n_pos from any layer's tensor
        probe = torch.load(args.acts / f"L{layers[0]}_{args.point}.pt",
                           map_location="cpu")
        n_pos = probe.shape[1]
        for pos in range(n_pos):
            plot_pos(args.acts, args.point, pos, layers, meta)
    else:
        plot_pos(args.acts, args.point, args.pos, layers, meta, args.out)


if __name__ == "__main__":
    main()