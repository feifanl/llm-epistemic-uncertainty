"""
Aggregate per-run steering CSVs into one cross-model comparison.

steer.py writes steering_results/judge_qwen/{model_slug}__{experiment}.csv per run, one row
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
    python viz/summarize_steering.py --results steering_results/judge_qwen/ --mode add
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
    frames = []
    for c in csvs:
        try:
            frames.append(pd.read_csv(c))
        except Exception as e:  # truncated/corrupt (e.g. interrupted run)
            print(f"  WARN skipping unreadable {c.name}: {type(e).__name__} "
                  f"({e}); rerun that steer command to regenerate it")
    if not frames:
        raise SystemExit("No readable CSVs.")
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(frames)}/{len(csvs)} runs, {len(df)} rows from {results_dir}")
    return df


def pivot_metric(df: pd.DataFrame, metric: str, agg="mean") -> pd.DataFrame:
    return df.pivot_table(index=["model", "experiment", "mode"],
                          columns="alpha", values=metric, aggfunc=agg)


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
    lines = ["# Steering summary", "",
             "Cells are mean±SEM across prompts.", ""]
    for metric, (piv, sem, tr) in tables.items():
        alphas = list(piv.columns)
        header = (["model", "experiment", "mode"]
                  + [f"a={a:g}" for a in alphas] + [f"corr(a,{metric})"])
        lines += [f"## {metric}", "", METRICS[metric], "",
                  "| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
        for key, row in piv.iterrows():
            model, exp, mode = key
            cells = []
            for a in alphas:
                if pd.isna(row[a]):
                    cells.append("")
                else:
                    s = sem.loc[key, a]
                    cells.append(f"{row[a]:.1f}±{s:.0f}" if pd.notna(s)
                                 else f"{row[a]:.1f}")
            c = tr.get(key, np.nan)
            cells.append(f"{c:+.2f}" if pd.notna(c) else "")
            lines.append("| " + " | ".join([model, exp, mode] + cells) + " |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out}")


def lineplot(df: pd.DataFrame, metric: str, out: Path):
    """Per-model curve: mean metric vs alpha, one line per experiment, SEM band.

    This is the headline figure: own-dir should climb (confidence) / fall
    (hedges) with alpha while the random control stays flat -- the separation
    is the causal claim.
    """
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), squeeze=False,
                             figsize=(4 * len(models), 3.8), sharey=True)
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        for exp in sorted(sub["experiment"].unique()):
            g = sub[sub["experiment"] == exp].groupby("alpha")[metric]
            a = sorted(g.groups)
            mean = g.mean().reindex(a)
            sem = g.sem().reindex(a).fillna(0)
            ax.plot(a, mean, marker="o", label=exp)
            ax.fill_between(a, mean - sem, mean + sem, alpha=0.2)
        ax.set_title(model.split("/")[-1], fontsize=9)
        ax.set_xlabel("alpha")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(f"mean {metric}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
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
    p.add_argument("--results", type=Path, default=Path("steering_results/judge_qwen"))
    p.add_argument("--mode", default="add", choices=["add", "ablate", "all"])
    p.add_argument("--alpha-max", type=float, default=None,
                   help="Drop |alpha| > this before computing trends/tables. Use to "
                        "exclude off-manifold breakdown (e.g. --alpha-max 0.5) so "
                        "corr reflects the coherent steering band, not token salad.")
    p.add_argument("--plot-dir", type=Path, default=Path("plots"))
    args = p.parse_args()

    df = load_runs(args.results)
    if args.mode != "all":
        df = df[df["mode"] == args.mode]
        if df.empty:
            raise SystemExit(f"No rows with mode={args.mode}.")
    if args.alpha_max is not None:
        df = df[df["alpha"].abs() <= args.alpha_max]
        print(f"Filtered to |alpha| <= {args.alpha_max}: {len(df)} rows")

    tables = {}
    for metric in METRICS:
        if metric not in df.columns or df[metric].notna().sum() == 0:
            continue
        piv = pivot_metric(df, metric)
        sem = pivot_metric(df, metric, agg="sem")
        tables[metric] = (piv, sem, trend(df, metric))
        print(f"\n[{metric}]\n" + piv.round(1).to_string())
        heatmap(piv, metric, args.plot_dir / f"steering_summary_{metric}.png")
        lineplot(df, metric, args.plot_dir / f"steering_curve_{metric}.png")

    if not tables:
        raise SystemExit("No usable metric columns (hedges/confidence).")
    write_markdown(tables, args.results / "summary.md")


if __name__ == "__main__":
    main()
