"""
PCA 2D scatter per layer, colored by cell. Pilot viz for activations.

Usage:
    python scripts/viz_pca.py --acts activations/gemma-2-2b-it/ --point resid --pos -1
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, required=True,
                   help="activations/{model}/ dir")
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1,
                   help="Which cached position (-1 = last token)")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Subset of layers to plot. Default: evenly-spaced 12")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    meta = pd.read_parquet(args.acts / "meta.parquet")
    print(f"{len(meta)} prompts, cells: {meta['cell'].value_counts().to_dict()}")

    # discover layers
    files = sorted(args.acts.glob(f"L*_{args.point}.pt"))
    all_layers = sorted(int(f.stem.split("_")[0][1:]) for f in files)
    if args.layers is None:
        # 12 evenly spaced including last
        n = len(all_layers)
        idxs = np.linspace(0, n - 1, min(12, n)).astype(int)
        layers = [all_layers[i] for i in idxs]
    else:
        layers = args.layers
    print(f"Plotting layers: {layers}")

    # grid
    n = len(layers)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)

    pos_idx = args.pos if args.pos >= 0 else None  # handle -1 below

    for ax, layer in zip(axes.flat, layers):
        t = torch.load(args.acts / f"L{layer}_{args.point}.pt",
                       map_location="cpu")  # [N, n_pos, d]
        idx = args.pos if args.pos >= 0 else t.shape[1] + args.pos
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

    # hide unused axes
    for ax in axes.flat[n:]:
        ax.axis("off")

    # one legend for whole fig
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8, label=k)
               for k, c in CELL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)

    model_name = args.acts.name
    fig.suptitle(f"PCA per layer — {model_name} | {args.point} | pos={args.pos}",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    out = args.out or args.acts / f"pca_{args.point}_pos{args.pos}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()