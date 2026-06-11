"""
Decisive dataset-fault test: does the model probe work on prompts that the
LEXICAL classifier CANNOT solve?

Join:
  - model probe per-prompt   (linear_probes.py --dump-preds -> perprompt_*.csv)
  - lexical difficulty        (per_prompt_lexical.py        -> _per_prompt_lexical.csv)

Difficulty axis = lexical SPLIT solvability (text-solvable = "easy/leaky").
  easy item : text classifies it correctly in-distribution  -> probe acc here proves little
  hard item : text fails it                                 -> only real knowledge separates

Verdict per layer:
  probe acc on HARD ~ acc on EASY, both >> chance  -> GENUINE uncertainty, confound doesn't bite
  probe acc on HARD ~ chance (works only on EASY)  -> CONFOUND, probe re-reads text

Usage:
    python easy_hard_compare.py \
        --perprompt probe_results/gemma-2-2b-it/perprompt_resid_transfer.csv \
        --lexical   probe_results/_per_prompt_lexical.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perprompt", type=Path, required=True,
                    help="model probe per-prompt CSV from linear_probes --dump-preds")
    ap.add_argument("--lexical", type=Path,
                    default=Path("probe_results/_per_prompt_lexical.csv"))
    ap.add_argument("--conf", type=float, default=0.5,
                    help="lexical prob margin to count an item 'easy' (default any correct)")
    args = ap.parse_args()

    probe = pd.read_csv(args.perprompt)
    lex = pd.read_csv(args.lexical)

    # item-level difficulty from lexical SPLIT: mean correctness across 6 paraphrases
    diff = (lex.groupby("item")
               .agg(lex_correct=("correct_split", "mean"),
                    lex_conf=("prob_split", lambda s: (s - 0.5).abs().mean()))
               .reset_index())
    # easy = text solves it (majority paraphrases correct AND confident enough)
    diff["bucket"] = (
        (diff["lex_correct"] >= 0.5) & (diff["lex_conf"] >= (args.conf - 0.5))
    ).map({True: "easy", False: "hard"})
    probe = probe.merge(diff[["item", "bucket"]],
                        left_on="item_idx", right_on="item", how="left")

    n_easy = (diff.bucket == "easy").sum()
    n_hard = (diff.bucket == "hard").sum()
    print(f"items: {n_easy} easy (text-solvable) / {n_hard} hard (text fails)\n")

    chance = 0.5
    # per layer × direction, accuracy split by bucket
    g = (probe.groupby(["layer", "direction", "bucket"])["correct"]
              .mean().unstack("bucket"))
    g["gap"] = g.get("easy", 0) - g.get("hard", 0)
    print("=== probe accuracy: easy vs hard, per layer ===")
    print(g.round(3).to_string())

    # headline: best transfer layer (by hard acc) and the verdict
    if "hard" in g.columns:
        best = g["hard"].idxmax()
        ha, ea = g.loc[best, "hard"], g.loc[best, "easy"]
        print(f"\nbest HARD layer {best}: hard={ha:.3f} easy={ea:.3f} chance={chance}")
        if ha - chance > 0.15 and (ea - ha) < 0.15:
            print("VERDICT: GENUINE — probe solves text-unsolvable items. Confound does not bite.")
        elif ha - chance < 0.08:
            print("VERDICT: CONFOUND — probe ~chance on hard items, only works where text works.")
        else:
            print("VERDICT: PARTIAL — probe adds signal on hard items but weaker than easy.")

    out = args.perprompt.with_name(args.perprompt.stem + "_easyhard.csv")
    g.to_csv(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
