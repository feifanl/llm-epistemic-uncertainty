#!/usr/bin/env bash
# One unattended VM queue. Run under tmux:
#     tmux new -s uq
#     ./vm_queue.sh 2>&1 | tee vm_queue.log
#     # detach: ctrl-b d      reattach: tmux attach -t uq
#
# Deliberately does NOT call `reproduce.sh all`: that stage-reruns every
# steering run and overwrites the committed CSVs behind Table 2. Only the two
# new Llama layers are steered here.
set -euo pipefail
cd "$(dirname "$0")"

step() { echo; echo "=== $* ==="; date; }

# ---- 0. snapshot anything the judge writes into, before it writes ----
step "snapshot steering_results"
cp -r steering_results "steering_results.bak.$(date +%Y%m%d-%H%M%S)"

# ---- 1. matched re-cache, all seven checkpoints at n_pos 5 ----
# The three surviving caches are already n_pos=5, but they predate the current
# transformers; re-caching all of them keeps one version across the comparison
# instead of trading a position confound for a library-version one.
step "cache (all checkpoints, --n-pos 5)"
./reproduce.sh cache

# ---- 2. re-probe: every position, real AUROC ----
# This is the experiment that decides finding #1. Compare base and instruct at
# pos 4 (final token) once it lands -- the committed instruct CSVs were probed
# at pos 0 of a 5-position cache, i.e. the 5th-from-last token.
step "probes (--all-pos --auroc --dump-preds)"
./reproduce.sh probes

# ---- 3. base-vs-instruct direction cosine (VM Exp A) ----
step "direction cosine, base vs instruct"
for PAIR in "google_gemma-2-9b google_gemma-2-9b-it" \
            "Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct" \
            "meta-llama_Llama-3.1-8B meta-llama_Llama-3.1-8B-Instruct"; do
  set -- $PAIR
  python pipeline/direction_cosine.py \
    --base "activations/$1/" --instruct "activations/$2/" --pos -1
done

# ---- 4. Llama at its actual probe peak (the paper's null was measured at L18) ----
step "llama single-site L22 and L25"
LL=activations/meta-llama_Llama-3.1-8B-Instruct/
for L in 22 25; do
  python pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
    --layer "$L" --alpha-mode relative --alphas -1 -0.5 -0.25 0 0.25 0.5 1 \
    --prompts prompts_eval.txt --experiment "own_L$L"
  for S in 0 1 2; do
    python pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
      --layer "$L" --alpha-mode relative --alphas -1 -0.5 -0.25 0 0.25 0.5 1 \
      --prompts prompts_eval.txt --random-dir --seed "$S" \
      --experiment "random_L$L"
  done
done

step "llama band re-centered on the peak (L20-28)"
python pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
  --layers 20 21 22 23 24 25 26 27 28 --alpha-mode relative \
  --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
  --experiment band_own_L20-28
for S in 0 1 2; do
  python pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
    --layers 20 21 22 23 24 25 26 27 28 --alpha-mode relative \
    --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
    --random-dir --seed "$S" --experiment band_random_L20-28
done

# ---- 5. judge the new runs (no --force: existing rows keep their scores) ----
step "judge new rows, Qwen judge"
python pipeline/judge_confidence.py --judge-model Qwen/Qwen2.5-7B-Instruct

# ---- 6. second judge, on a COPY. --force here would clobber the paper's numbers.
step "second judge (llama) on a copy"
rm -rf steering_results_llamajudge
cp -r steering_results steering_results_llamajudge
python pipeline/judge_confidence.py --results steering_results_llamajudge \
  --judge-model meta-llama/Llama-3.1-8B-Instruct --force

# ---- 7. regenerate everything derived ----
step "figures + tables"
./reproduce.sh figures
./reproduce.sh tables

step "done"
