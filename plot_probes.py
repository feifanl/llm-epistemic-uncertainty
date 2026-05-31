"""
Plot probe results from linear_probes.py CSVs.

CSV columns:
    split mode:    layer, pos, split_train, split_test
    transfer mode: layer, pos, p2f_train, p2f_test, f2p_train, f2p_test

Two uses:

  Single CSV -> layer-accuracy curve (per direction, train dashed / test
  solid, chance line) + layer×pos heatmap. The core analysis figures.

  Multiple CSVs -> overlay test accuracy vs layer at one position, labeled
  by file. Use for point (resid/attn/mlp) or model (2b/9b/9b-it) comparison.

Usage:
    python plot_probes.py --csv activations/gemma-2-9b-it/probes/probe_resid_transfer.csv
    python plot_probes.py --csv .../probe_resid_transfer.csv .../probe_attn_transfer.csv \
                          --metric p2f_test --pos 4
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def directions(df):
    """Direction prefixes present, e.g. ['split'] or ['p2f', 'f2p']."""
    return [c[:-5] for c in df.columns if c.endswith("_test")]


def curve_and_heatmap(csv, out_dir):
    """Single-CSV analysis: layer curve (best pos) + per-direction heatmap."""
    df = pd.read_csv(csv)
    dirs = directions(df)
    positions = sorted(df["pos"].unique())
    last_pos = positions[-1]   # readout token
    model = csv.parent.parent.name   # activations/{model}/probes/probe_*.csv

    # --- layer curve at the readout position ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[df["pos"] == last_pos].sort_values("layer")
    for d in dirs:
        ax.plot(sub["layer"], sub[f"{d}_test"], "-o", ms=4, label=f"{d} test")
        ax.plot(sub["layer"], sub[f"{d}_train"], "--", alpha=0.5,
                label=f"{d} train")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{model} | {csv.stem} — pos={last_pos}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"{csv.stem}_curve.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved → {out}")

    # --- heatmap layer × pos, one panel per direction (test acc) ---
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(5 * len(dirs), 5),
                             squeeze=False)
    for ax, d in zip(axes.flat, dirs):
        grid = df.pivot(index="pos", columns="layer", values=f"{d}_test")
        grid = grid.reindex(index=positions, columns=layers)
        im = ax.imshow(grid.values, aspect="auto", origin="lower",
                       vmin=0.5, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks(range(len(positions))); ax.set_yticklabels(positions)
        ax.set_xlabel("layer"); ax.set_ylabel("pos")
        ax.set_title(f"{d}_test")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{model} | {csv.stem} — accuracy heatmap")
    fig.tight_layout()
    out = out_dir / f"{csv.stem}_heatmap.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved → {out}")


def compare(csvs, metric, pos, out):
    """Multi-CSV: overlay one metric vs layer at one pos, labeled by file."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for csv in csvs:
        df = pd.read_csv(csv)
        if metric not in df.columns:
            print(f"  skip {csv.name}: no column '{metric}'")
            continue
        sub = df[df["pos"] == pos].sort_values("layer")
        ax.plot(sub["layer"], sub[metric], "-o", ms=4, label=csv.stem)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    ax.set_xlabel("layer"); ax.set_ylabel(metric)
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{metric} vs layer  (pos={pos})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, nargs="+", required=True)
    p.add_argument("--metric", default="p2f_test",
                   help="Column to overlay when comparing multiple CSVs")
    p.add_argument("--pos", type=int, default=None,
                   help="Position for compare mode (default: last)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path for compare mode")
    args = p.parse_args()

    if len(args.csv) == 1:
        csv = args.csv[0]
        curve_and_heatmap(csv, csv.parent)
    else:
        pos = args.pos
        if pos is None:
            pos = int(sorted(pd.read_csv(args.csv[0])["pos"].unique())[-1])
        out = args.out or Path(f"compare_{args.metric}_pos{pos}.png")
        compare(args.csv, args.metric, pos, out)


if __name__ == "__main__":
    main()
