"""
Cosine between the base and instruct uncertainty directions, per layer.

Equal probe accuracy in two checkpoints does not mean they encode uncertainty
along the same axis -- "the direction is a pretraining feature" is really "a
direction of equal decodability exists in both" until you check the angle. Same
family means same width and same layer count, so the comparison is well posed.

Reports cosine(v_base, v_instruct) per layer plus, as the scale reference, the
cosine you'd expect from two random vectors of that width (~1/sqrt(D)).

Usage:
    python pipeline/direction_cosine.py \
        --base activations/google_gemma-2-9b/ \
        --instruct activations/google_gemma-2-9b-it/ --pos -1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from linear_probes import discover_layers, load_tensor


def diffmean(acts_dir, layer, point, pos, y):
    T = load_tensor(acts_dir, layer, point)      # [N, n_pos, D]
    idx = pos if pos >= 0 else T.shape[1] + pos
    X = T[:, idx, :]
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--instruct", type=Path, required=True)
    p.add_argument("--point", default="resid", choices=["resid", "attn", "mlp"])
    p.add_argument("--pos", type=int, default=-1,
                   help="cached position index; -1 = final token in both caches")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    yb = pd.read_parquet(args.base / "meta.parquet")["label"].values.astype(int)
    yi = pd.read_parquet(args.instruct / "meta.parquet")["label"].values.astype(int)
    if not np.array_equal(yb, yi):
        raise SystemExit("label vectors differ -- the two caches aren't the same "
                         "dataset in the same row order, cosine is meaningless")

    lb = discover_layers(args.base, args.point)
    li = discover_layers(args.instruct, args.point)
    if lb != li:
        raise SystemExit(f"layer counts differ ({len(lb)} vs {len(li)}); "
                         "only same-family checkpoints are comparable")

    rows = []
    for layer in lb:
        vb = diffmean(args.base, layer, args.point, args.pos, yb)
        vi = diffmean(args.instruct, layer, args.point, args.pos, yi)
        cos = float(vb @ vi / (np.linalg.norm(vb) * np.linalg.norm(vi)))
        rows.append({"layer": layer, "cosine": cos,
                     "norm_base": float(np.linalg.norm(vb)),
                     "norm_instruct": float(np.linalg.norm(vi))})
        print(f"  L{layer:<2d}  cos {cos:+.3f}")

    df = pd.DataFrame(rows)
    D = load_tensor(args.base, lb[0], args.point).shape[-1]
    print(f"\nrandom-vector reference at D={D}: |cos| ~ {1 / np.sqrt(D):.3f}")
    print(f"mean cosine {df['cosine'].mean():+.3f}, "
          f"max {df['cosine'].max():+.3f} at L{int(df.loc[df.cosine.idxmax(), 'layer'])}")

    out = args.out or (args.base.parent.parent / "probe_results" /
                       f"_cosine_{args.base.name}_vs_{args.instruct.name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
