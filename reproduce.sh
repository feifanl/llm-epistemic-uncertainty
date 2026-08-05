#!/usr/bin/env bash
# usage: ./reproduce.sh [cache|probes|steer|judge|figures|tables|test|all]
#
# bash rather than make or a python driver: the gpu stages only ever run on the
# linux vm, and a python driver would just shell out to these same commands.
# stages are independent so a partial rerun is cheap -- `steer` alone is an
# overnight job, `tables` is seconds.
#
# assumes cwd = repo root and the venv already active (or PY set below).
set -euo pipefail
stage=${1:-all}
PY=${PY:-python}

MODELS_ALL="google/gemma-2-9b google/gemma-2-9b-it Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct gpt2-large"
slug() { echo "$1" | tr / _; }

# cache every checkpoint at the SAME --n-pos. the committed instruct csvs have
# only pos 0 and the base ones pos 4; whether those are the same token is
# recorded only in the gitignored config.json. caching everything at 5 makes
# the base-vs-instruct comparison unambiguous by construction.
if [[ $stage == cache || $stage == all ]]; then
  for M in $MODELS_ALL; do
    $PY pipeline/cache_activations.py --model "$M" --n-pos 5 \
      --out "activations/$(slug "$M")/"
  done
fi

if [[ $stage == probes || $stage == all ]]; then
  for M in $MODELS_ALL; do
    $PY pipeline/linear_probes.py --acts "activations/$(slug "$M")/" \
      --point resid --transfer --all-pos --dump-preds --auroc
  done
  $PY baselines/lexical_baseline.py --dataset dataset.csv
fi

if [[ $stage == steer || $stage == all ]]; then
  # single-site own direction + 3-seed matched-norm random control, per model.
  # layers: gemma L20, qwen L19, llama L18, gpt2 L22 (as in the paper). these
  # were fixed a priori mid-stack; llama's is NOT its probe peak (L22/L23) --
  # see paper limitations, and VM_EXPERIMENTS for the rerun.
  declare -A LAYER=( [google/gemma-2-9b-it]=20 [Qwen/Qwen2.5-7B-Instruct]=19
                     [meta-llama/Llama-3.1-8B-Instruct]=18 [gpt2-large]=22 )
  for M in "${!LAYER[@]}"; do
    A="activations/$(slug "$M")/"
    $PY pipeline/steer.py --acts "$A" --model "$M" --layer "${LAYER[$M]}" \
      --alpha-mode relative --alphas -1 -0.5 -0.25 0 0.25 0.5 1 \
      --prompts prompts_eval.txt --experiment own
    for S in 0 1 2; do
      $PY pipeline/steer.py --acts "$A" --model "$M" --layer "${LAYER[$M]}" \
        --alpha-mode relative --alphas -1 -0.5 -0.25 0 0.25 0.5 1 \
        --prompts prompts_eval.txt --random-dir --seed "$S" --experiment random
    done
  done

  # band injection: nine layers, each with its own per-layer vector, total push
  # divided by band width so it's comparable to a single site.
  QWEN=activations/Qwen_Qwen2.5-7B-Instruct/
  LLAMA=activations/meta-llama_Llama-3.1-8B-Instruct/
  $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
    --layers 15 16 17 18 19 20 21 22 23 --alpha-mode relative \
    --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
    --experiment band_own
  $PY pipeline/steer.py --acts "$LLAMA" --model meta-llama/Llama-3.1-8B-Instruct \
    --layers 18 19 20 21 22 23 24 25 26 --alpha-mode relative \
    --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
    --experiment band_own
  for S in 0 1 2; do
    $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
      --layers 15 16 17 18 19 20 21 22 23 --alpha-mode relative \
      --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
      --random-dir --seed "$S" --experiment band_random
    $PY pipeline/steer.py --acts "$LLAMA" --model meta-llama/Llama-3.1-8B-Instruct \
      --layers 18 19 20 21 22 23 24 25 26 --alpha-mode relative \
      --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
      --random-dir --seed "$S" --experiment band_random
  done

  # cross-family transfer: build gemma's L20 direction, inject it into qwen.
  # the save pass still runs one no-op generation; cheap enough to leave alone.
  mkdir -p directions
  $PY pipeline/steer.py --acts activations/google_gemma-2-9b-it/ \
    --model google/gemma-2-9b-it --layer 20 \
    --save-direction directions/gemma_L20.npy --alphas 0
  $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
    --layer 19 --direction directions/gemma_L20.npy --alpha-mode relative \
    --alphas -1 -0.5 -0.25 0 0.25 0.5 1 --prompts prompts_eval.txt \
    --experiment transfer
fi

# writes the confidence column back into steering_results/*.csv in place.
# changing --judge-model without --force mixes two scales silently.
if [[ $stage == judge || $stage == all ]]; then
  $PY pipeline/judge_confidence.py --judge-model Qwen/Qwen2.5-7B-Instruct
fi

if [[ $stage == figures || $stage == all ]]; then
  $PY viz/summarize_steering.py --alpha-max 0.5
  $PY viz/plot_paper_figures.py
fi

if [[ $stage == tables || $stage == all ]]; then
  $PY viz/make_paper_tables.py
fi

if [[ $stage == test || $stage == all ]]; then
  $PY -m pytest tests/ -q
fi
