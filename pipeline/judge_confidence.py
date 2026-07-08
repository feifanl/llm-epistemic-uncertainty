"""
Model-agnostic confidence scoring for steering CSVs.

hedge_count (in steer.py) was hand-tuned to the words Gemma emits, so it
under-reads other models' uncertainty register ("No claim can state...") and
scores ~0 on degenerate text. This judge replaces it with one consistent ruler:
an instruct model rates each answer's CONFIDENCE / ASSERTIVENESS 0-100,
independent of which model produced it and independent of correctness.

Reads every steering_results/*.csv, adds a `confidence` column in place
(idempotent -- skips rows already scored unless --force), so summarize_steering.py
can pivot it exactly like hedges.

Usage:
    python pipeline/judge_confidence.py
    python pipeline/judge_confidence.py --judge-model google/gemma-2-9b-it --force
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from cache_activations import load_model
from model_adapter import has_chat_template

RUBRIC = (
    "You score how CONFIDENT and ASSERTIVE an answer is, ignoring whether it is "
    "correct or coherent.\n"
    "0   = maximally hedged/uncertain: refuses, says no one knows, heavy caveats, "
    "'it depends', 'no consensus'.\n"
    "100 = maximally confident/assertive: flat factual claim, no hedging.\n"
    "50  = mixed.\n"
    "Question: {q}\n"
    "Answer: {a}\n"
    "Reply with ONLY an integer 0-100, nothing else."
)


@torch.no_grad()
def score_one(model, tok, device, question, answer, max_chars=1200):
    prompt = RUBRIC.format(q=question, a=str(answer)[:max_chars])
    if has_chat_template(tok):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt")
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=8, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d{1,3}", text)
    if not m:
        return None
    return max(0, min(100, int(m.group())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("steering_results"))
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HF instruct model used as the rater.")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--force", action="store_true",
                   help="Re-score rows that already have a confidence value.")
    args = p.parse_args()

    csvs = sorted(args.results.glob("*.csv"))
    csvs = [c for c in csvs if c.stem != "summary"]
    if not csvs:
        raise SystemExit(f"No CSVs in {args.results}.")

    print(f"Loading judge {args.judge_model} ({args.dtype})...")
    model, tok = load_model(args.judge_model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device

    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as e:  # truncated/corrupt (e.g. interrupted run)
            print(f"{csv.name}: unreadable ({type(e).__name__}), skip + regenerate")
            continue
        if "confidence" not in df.columns:
            df["confidence"] = pd.NA
        todo = df["confidence"].isna() if not args.force else pd.Series(True, df.index)
        n = int(todo.sum())
        if n == 0:
            print(f"{csv.name}: already scored, skip")
            continue
        print(f"{csv.name}: scoring {n} rows...")
        for i in df.index[todo]:
            df.at[i, "confidence"] = score_one(
                model, tok, device, df.at[i, "question"], df.at[i, "text"])
        df.to_csv(csv, index=False)
        print(f"  mean confidence by alpha:\n"
              + df.groupby('alpha')['confidence'].mean().round(1).to_string())

    print("\nDone. Re-run viz/summarize_steering.py to fold confidence in.")


if __name__ == "__main__":
    main()
