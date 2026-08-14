"""
Lexical baseline: how well does a dumb TEXT classifier separate known/unknown
from the claim string alone, no model activations?

This is the control floor for linear_probes.py. A probe layer is only
"finding uncertainty" to the extent it BEATS this number. In particular:

    split    : TF-IDF can exploit topic/vocab/year tokens -> usually high.
    transfer : TF-IDF cannot cross the past<->future axis (different vocab)
               -> collapses to chance. Any probe transfer above this is the
               genuine, time-invariant signal.

Mirrors linear_probes.py conventions:
  - same paraphrase expansion (claim + para_1..5)  -> N rows
  - split mode    : random 80/20, seed 0  (+ 5-fold CV for an error bar)
  - transfer mode : cell prefix 'past'/'future' -> p2f and f2p

Usage:
    python lexical_baseline.py --dataset dataset.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

PARAPHRASE_COLS = ["claim", "para_1", "para_2", "para_3", "para_4", "para_5"]


def expand(df):
    """Paraphrase-expand to one row per (item, paraphrase), matches the
    rows the probe is trained on (see cache_activations.build_prompts)."""
    rows = []
    for r in df.itertuples():
        for col in PARAPHRASE_COLS:
            text = getattr(r, col)
            if pd.isna(text):
                continue
            rows.append({"text": str(text), "label": int(r.label), "cell": r.cell})
    return pd.DataFrame(rows)


def make_model():
    # 1-2gram TF-IDF + logreg. min_df=2 drops singleton tokens (noise).
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True),
        LogisticRegression(max_iter=2000),
    )


def split_eval(t):
    """Random 80/20 (seed 0, matches linear_probes) + 5-fold CV error bar."""
    X = np.asarray(t["text"].tolist(), dtype=object)   # pyarrow-safe
    y = t["label"].to_numpy()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    m = make_model().fit(X[tr], y[tr])
    single = (m.predict(X[te]) == y[te]).mean()
    cv = cross_val_score(make_model(), X, y, cv=5)
    return single, cv.mean(), cv.std()


def transfer_eval(t):
    """Train one time-half, test the other. p2f and f2p."""
    X = np.asarray(t["text"].tolist(), dtype=object)   # pyarrow-safe
    y = t["label"].to_numpy()
    time = t["cell"].str.split("_").str[0].values
    past, future = time == "past", time == "future"
    out = {}
    for name, tr, te in [("p2f", past, future), ("f2p", future, past)]:
        m = make_model().fit(X[tr], y[tr])
        out[name] = (m.predict(X[te]) == y[te]).mean()
    return out["p2f"], out["f2p"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    p.add_argument("--out", type=Path, default=Path("probe_results/_lexical_baseline.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.dataset)
    t = expand(df)

    single, cv_mean, cv_std = split_eval(t)
    p2f, f2p = transfer_eval(t)

    print(f"rows (paraphrase-expanded): {len(t)}   chance: 0.500")
    print(f"split    single80/20={single:.3f}   5fold={cv_mean:.3f}±{cv_std:.3f}")
    print(f"transfer p2f={p2f:.3f}   f2p={f2p:.3f}")
    print("-> probe beats text where probe_acc > these. transfer is the clean test.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": args.dataset.name,
        "n_rows": len(t),
        "split_single": round(single, 4),
        "split_cv_mean": round(cv_mean, 4),
        "split_cv_std": round(cv_std, 4),
        "transfer_p2f": round(p2f, 4),
        "transfer_f2p": round(f2p, 4),
    }]).to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
