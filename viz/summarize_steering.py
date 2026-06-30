"""
Aggregate per-run steering CSVs into one cross-model comparison.

steer.py writes steering_results/{model_slug}__{experiment}.csv per run, one row
per (question, alpha). judge_confidence.py optionally adds a `confidence` column.
This rolls every run up into, for each metric present:

  - a master table: rows = (model, experiment, mode), cols = alpha,
    cells = mean metric across prompts. Lets own / random / transfer runs sit
    side by side so the control (random) is visible next to the real direction.
  - a `corr(alpha, metric)` trend column. Expected sign differs by metric:
      hedges      -> NEGATIVE (toward "known" => fewer hedges)
      confidence  -> POSITIVE (toward "known" => more confident)
    The key read is own-dir trend vs random-dir trend: a real knob beats random.
  - a heatmap PNG.

Only `add`-mode rows feed the tables (steering); ablate rows skipped here.

Usage:
    python viz/summarize_steering.py
    python viz/summarize_steering.py --results steering_results/ --mode add
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# metric -> (human direction of the expected effect toward "known")
METRICS = {
    "hedges": "lower = more confident; expect corr(alpha,hedges) < 0",
    "confidence": "higher = more confident; expect corr(alpha,confidence) > 0",
}


def load_runs(results_dir: Path) -> pd.DataFrame:
    csvs = [c for c in sorted(results_dir.glob("*.csv")) if c.stem != "summary"]
    if not csvs:
        raise SystemExit(f"No CSVs in {results_dir} -- run steer.py first.")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f"Loaded {len(csvs)} runs, {len(df)} rows from {results_dir}")
    return df


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(index=["model", "experiment", "mode"],
                          columns="alpha", values=metric, aggfunc="mean")


def trend(df: pd.DataFrame, metric: str) -> pd.Series:
    """Per (model, experiment, mode): corr(alpha, metric). NaN if alpha or the
    metric has no variance (e.g. baseline-only run, or all-unscored)."""
    def corr(g):
        a, m = g["alpha"], g[metric]
        ok = m.notna()
        if a[ok].nunique() < 2 or m[ok].nunique() < 2:
            return np.nan
        return np.corrcoef(a[ok], m[ok])[0, 1]
    return df.groupby(["model", "experiment", "mode"]).apply(corr)


def write_markdown(tables: dict, out: Path):
    lines = ["# Steering summary", ""]
    for metric, (piv, tr) in tables.items():
        alphas = list(piv.columns)
        header = (["model", "experiment", "mode"]
                  + [f"a={a:g}" for a in alphas] + [f"corr(a,{metric})"])
        lines += [f"## {metric}", "", METRICS[metric], "",
                  "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        for key, row in piv.iterrows():
            model, exp, mode = key
            cells = [f"{row[a]:.1f}" if pd.notna(row[a]) else "" for a in alphas]
            c = tr.get(key, np.nan)
            cells.append(f"{c:+.2f}" if pd.notna(c) else "")
            lines.append("| " + " | ".join([model, exp, mode] + cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out}")


def heatmap(piv: pd.DataFrame, metric: str, out: Path):
    labels = [f"{m.split('/')[-1]}\n{e} ({mo})" for m, e, mo in piv.index]
    data = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 3, 0.6 * piv.shape[0] + 2))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{a:g}" for a in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("alpha (steering scalar)")
    ax.set_title(f"Mean {metric}")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("steering_results"))
    p.add_argument("--mode", default="add", choices=["add", "ablate", "all"])
    p.add_argument("--plot-dir", type=Path, default=Path("plots"))
    args = p.parse_args()

    df = load_runs(args.results)
    if args.mode != "all":
        df = df[df["mode"] == args.mode]
        if df.empty:
            raise SystemExit(f"No rows with mode={args.mode}.")

    tables = {}
    for metric in METRICS:
        if metric not in df.columns or df[metric].notna().sum() == 0:
            continue
        piv = pivot_metric(df, metric)
        tables[metric] = (piv, trend(df, metric))
        print(f"\n[{metric}]\n" + piv.round(1).to_string())
        heatmap(piv, metric, args.plot_dir / f"steering_summary_{metric}.png")

    if not tables:
        raise SystemExit("No usable metric columns (hedges/confidence).")
    write_markdown(tables, args.results / "summary.md")


if __name__ == "__main__":
    main()
