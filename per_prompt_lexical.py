"""
Per-prompt DATASET-fault analysis using the lexical (TF-IDF) classifier.

No model / no activations needed. The text classifier is the dataset-fault
detector: if known/unknown is separable from raw text, the dataset leaks.
This drills into WHERE it leaks, per prompt.

Reports:
  1. paraphrase agreement  - do 6 paraphrases of one item get the same call?
                             scatter => label rides on wording, not meaning.
  2. transfer errors        - train past, predict future (+reverse); per-prompt.
  3. error clustering       - misclassified prompts bucketed by year / topic.
  4. confident-wrong        - high-prob predictions that are wrong (caught).

Usage:
    python per_prompt_lexical.py --dataset dataset.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline

PARAPHRASE_COLS = ["claim", "para_1", "para_2", "para_3", "para_4", "para_5"]
AI_TOPIC = re.compile(r"\b(ai|gpt|chatgpt|llm|tutor|tutors|generative|vr|ar|augmented|virtual)\b", re.I)


def expand(df):
    rows = []
    for r in df.itertuples():
        for pid, col in enumerate(PARAPHRASE_COLS):
            txt = getattr(r, col)
            if pd.isna(txt):
                continue
            yr = re.search(r"\b(20\d\d)\b", str(txt))
            rows.append({
                "item": r.idx, "pid": pid, "text": str(txt),
                "label": int(r.label), "cell": r.cell,
                "time": r.cell.split("_")[0],
                "year": int(yr.group(1)) if yr else 0,
                "ai_topic": bool(AI_TOPIC.search(str(txt))),
            })
    return pd.DataFrame(rows)


def model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True),
        LogisticRegression(max_iter=2000),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    ap.add_argument("--out", type=Path, default=Path("probe_results/_per_prompt_lexical.csv"))
    args = ap.parse_args()

    t = expand(pd.read_csv(args.dataset))
    X, y = t["text"].values, t["label"].values

    # --- split-mode per-prompt prob via 5-fold out-of-fold prediction ---
    t["prob_split"] = cross_val_predict(model(), X, y, cv=5, method="predict_proba")[:, 1]
    t["pred_split"] = (t["prob_split"] > 0.5).astype(int)
    t["correct_split"] = (t["pred_split"] == t["label"]).astype(int)

    # --- transfer-mode per-prompt: train one time-half, predict the other ---
    t["prob_tr"] = np.nan
    for tr_time in ["past", "future"]:
        tr = t["time"] == tr_time
        te = ~tr
        m = model().fit(X[tr.values], y[tr.values])
        t.loc[te, "prob_tr"] = m.predict_proba(X[te.values])[:, 1]
    t["pred_tr"] = (t["prob_tr"] > 0.5).astype(int)
    t["correct_tr"] = (t["pred_tr"] == t["label"]).astype(int)

    # ============ 1. paraphrase agreement ============
    g = t.groupby("item")
    agree = g["pred_split"].agg(lambda s: s.value_counts(normalize=True).max())  # frac in majority
    prob_std = g["prob_split"].std()
    print("=== 1. PARAPHRASE AGREEMENT (split) ===")
    print(f"items where all 6 paraphrases agree: {(agree == 1.0).sum()}/{g.ngroups}")
    print(f"mean within-item prob std: {prob_std.mean():.3f}  (0=meaning-based, high=wording-based)")
    worst = prob_std.sort_values(ascending=False).head(5)
    print("most wording-sensitive items (high prob std):")
    for it, sd in worst.items():
        c = t[t.item == it].iloc[0]
        print(f"  item {it:3d} [{c.cell}] std={sd:.2f}  '{c.text[:60]}'")

    # ============ 2/3. transfer errors + clustering ============
    print("\n=== 2. TRANSFER ACC (per-prompt lexical) ===")
    for tm in ["past", "future"]:
        sub = t[t.time == tm]
        print(f"  predict {tm:6s} (train other): acc={sub.correct_tr.mean():.3f}  n={len(sub)}")
    err = t[t.correct_tr == 0]
    print("\n=== 3. TRANSFER ERROR CLUSTERING ===")
    print(f"  total transfer errors: {len(err)}/{len(t)}")
    print(f"  errors by time:  {err['time'].value_counts().to_dict()}")
    print(f"  errors ai_topic: {err['ai_topic'].value_counts().to_dict()}")
    fe = err[err.year > 0]
    if len(fe):
        print(f"  future-error year buckets: <=2030={ (fe.year<=2030).sum() }  >=2035={ (fe.year>=2035).sum() }")

    # ============ 4. confident-wrong ============
    print("\n=== 4. CONFIDENT-WRONG (split, |prob-0.5| large but wrong) ===")
    cw = t[t.correct_split == 0].copy()
    cw["conf"] = (cw["prob_split"] - 0.5).abs()
    for _, r in cw.sort_values("conf", ascending=False).head(6).iterrows():
        call = "known" if r["pred_split"] else "unknown"
        print(f"  item {r['item']:3d} [{r['cell']}] p={r['prob_split']:.2f}->{call} WRONG  '{r['text'][:55]}'")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    print(f"\nsaved per-prompt table -> {args.out}")


if __name__ == "__main__":
    main()
