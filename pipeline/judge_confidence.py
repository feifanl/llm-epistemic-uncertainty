"""
Model-agnostic confidence scoring for steering CSVs.

hedge_count (in steer.py) was hand-tuned to the words Gemma emits, so it
under-reads other models' uncertainty register ("No claim can state...") and
scores ~0 on degenerate text. This judge replaces it with one consistent ruler.

We do NOT ask the judge for a number -- a 7B judge ignores a 0-100 rubric and
buckets to lazy round digits (5, 1, 10), giving no resolution. Instead we pose a
forced Yes/No ("is this answer confident and assertive?") and read the
next-token logits: confidence = P(Yes) / (P(Yes)+P(No)) in [0,1], scaled to
0-100. Continuous, parse-free, and comparable across judges.

Writes a `confidence` column in place into every steering_results/*.csv
(idempotent -- skips already-scored rows unless --force) so
summarize_steering.py can pivot it like hedges.

Gotcha: it writes back into the raw steering csvs in place. re-judging with a
different model mixes scales silently -- use --force and re-score everything if
you change --judge-model.

Usage:
    python pipeline/judge_confidence.py
    python pipeline/judge_confidence.py --judge-model google/gemma-2-9b-it --force
    python pipeline/judge_confidence.py --debug 10        # print P(Yes) for a few
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from cache_activations import load_model
from model_adapter import has_chat_template

RUBRIC = (
    "Judge whether the following answer is CONFIDENT and ASSERTIVE (states things "
    "plainly, commits to a claim) or HEDGED and UNCERTAIN (caveats, refuses, 'no "
    "one knows', 'it depends', 'no consensus'). Ignore whether it is correct.\n"
    "Question: {q}\n"
    "Answer: {a}\n"
    "Is the answer confident and assertive? Answer with one word: Yes or No."
)


def cand_ids(tok, words):
    """First-token ids for each word form (with/without leading space)."""
    ids = set()
    for w in words:
        for form in (w, " " + w):
            t = tok(form, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
    return sorted(ids)


@torch.no_grad()
def score_one(model, tok, device, question, answer, yes_ids, no_ids,
              max_chars=1200):
    prompt = RUBRIC.format(q=question, a=str(answer)[:max_chars])
    if has_chat_template(tok):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt")
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits[0, -1].float()          # [vocab]
    yes = torch.logsumexp(logits[yes_ids], 0)
    no = torch.logsumexp(logits[no_ids], 0)
    p_yes = torch.softmax(torch.stack([yes, no]), 0)[0].item()
    return round(p_yes * 100, 1), f"P(Yes)={p_yes:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("steering_results"))
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HF instruct model used as the rater.")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--force", action="store_true",
                   help="Re-score rows that already have a confidence value.")
    p.add_argument("--debug", type=int, default=0,
                   help="Print P(Yes) for the first N rows (sanity check).")
    args = p.parse_args()

    csvs = [c for c in sorted(args.results.glob("*.csv")) if c.stem != "summary"]
    if not csvs:
        raise SystemExit(f"No CSVs in {args.results}.")

    print(f"Loading judge {args.judge_model} ({args.dtype})...")
    model, tok = load_model(args.judge_model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    yes_ids = cand_ids(tok, ["Yes", "yes", "YES"])
    no_ids = cand_ids(tok, ["No", "no", "NO"])
    print(f"Yes token ids {yes_ids}, No token ids {no_ids}")

    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as e:  # truncated/corrupt (e.g. interrupted run)
            print(f"{csv.name}: unreadable ({type(e).__name__}), skip + regenerate")
            continue
        # Force float dtype: an old run may have written int64 confidence, which
        # rejects the new fractional P(Yes)*100 scores.
        if "confidence" not in df.columns:
            df["confidence"] = float("nan")
        else:
            df["confidence"] = pd.to_numeric(
                df["confidence"], errors="coerce").astype("float64")
        todo = df["confidence"].isna() if not args.force else pd.Series(True, df.index)
        n = int(todo.sum())
        if n == 0:
            print(f"{csv.name}: already scored, skip")
            continue
        print(f"{csv.name}: scoring {n} rows...")
        dbg = args.debug
        for i in df.index[todo]:
            score, raw = score_one(model, tok, device, df.at[i, "question"],
                                   df.at[i, "text"], yes_ids, no_ids)
            if dbg > 0:
                print(f"  [debug] alpha={df.at[i,'alpha']} {raw} -> {score}")
                dbg -= 1
            df.at[i, "confidence"] = score
        df.to_csv(csv, index=False)
        print(f"  mean confidence by alpha:\n"
              + df.groupby('alpha')['confidence'].mean().round(1).to_string())

    print("\nDone. Re-run viz/summarize_steering.py to fold confidence in.")


if __name__ == "__main__":
    main()
