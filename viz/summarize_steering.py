"""
Aggregate per-run steering CSVs into one cross-model comparison.

steer.py writes steering_results/{model_slug}__{experiment}.csv per run, one row
per (question, alpha). This rolls them all up into:

  - a master table: rows = (model, experiment, mode), cols = alpha,
    cells = mean hedge count across prompts. One look tells you whether hedging
    moves monotonically with alpha (toward "known" -> fewer hedges) and lets you
    compare Qwen's transfer (Gemma's direction) vs own direction side by side.
  - a "trend" column: Pearson corr(alpha, hedges) per row. Negative = expected
    (more known => fewer hedges); near-zero = the direction does nothing here.
  - a heatmap PNG of the same table.

Only `add`-mode (steering) rows feed the table; ablate rows are reported
separately if present.

Usage:
    python viz/summarize_steering.py
    python viz/summarize_steering.py --results steering_results/ --mode add
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_runs(results_dir: Path) -> pd.DataFrame:
    """Concat every *.csv in results_dir. Errors if none found."""
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs in {results_dir} -- run steer.py first.")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f"Loaded {len(csvs)} runs, {len(df)} rows from {results_dir}")
    return df


def pivot_mean_hedges(df: pd.DataFrame) -> pd.DataFrame:
    """(model, experiment, mode) x alpha -> mean hedges across questions."""
    return df.pivot_table(index=["model", "experiment", "mode"],
                          columns="alpha", values="hedges", aggfunc="mean")


def trend(df: pd.DataFrame) -> pd.Series:
    """Per (model, experiment, mode): corr(alpha, hedges) over all rows.

    Negative = hedging falls as alpha rises (the expected known-direction
    effect). NaN if alpha has no variance (e.g. a baseline-only run).
    """
    def corr(g):
        if g["alpha"].nunique() < 2:
            return np.nan
        return np.corrcoef(g["alpha"], g["hedges"])[0, 1]
    return df.groupby(["model", "experiment", "mode"]).apply(corr)


def write_markdown(piv: pd.DataFrame, tr: pd.Series, out: Path):
    alphas = list(piv.columns)
    header = ["model", "experiment", "mode"] + [f"a={a:g}" for a in alphas] + ["corr(a,hedges)"]
    lines = ["# Steering summary", "",
             "Mean hedge count across prompts, per run, by steering scalar. "
             "`corr(a,hedges)` < 0 = hedging falls toward 'known' (expected).", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for key, row in piv.iterrows():
        model, exp, mode = key
        cells = [f"{row[a]:.2f}" if pd.notna(row[a]) else "" for a in alphas]
        c = tr.get(key, np.nan)
        cells.append(f"{c:+.2f}" if pd.notna(c) else "")
        lines.append("| " + " | ".join([model, exp, mode] + cells) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out}")


def heatmap(piv: pd.DataFrame, out: Path):
    labels = [f"{m.split('/')[-1]}\n{e} ({mo})" for m, e, mo in piv.index]
    data = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 3, 0.6 * piv.shape[0] + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{a:g}" for a in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("alpha (steering scalar)")
    ax.set_title("Mean hedge count")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="hedges")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("steering_results"))
    p.add_argument("--mode", default="add", choices=["add", "ablate", "all"],
                   help="Which steering mode to summarize (default add=steering).")
    p.add_argument("--plot", type=Path, default=Path("plots/steering_summary.png"))
    args = p.parse_args()

    df = load_runs(args.results)
    if args.mode != "all":
        df = df[df["mode"] == args.mode]
        if df.empty:
            raise SystemExit(f"No rows with mode={args.mode}.")

    piv = pivot_mean_hedges(df)
    tr = trend(df)
    print("\n" + piv.round(2).to_string())

    write_markdown(piv, tr, args.results / "summary.md")
    heatmap(piv, args.plot)


if __name__ == "__main__":
    main()
