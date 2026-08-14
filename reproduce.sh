#!/usr/bin/env bash
# usage: ./reproduce.sh [cache|probes|steer|steer-peak|judge|judge2|figures|tables|test|all]
#
# bash rather than make or a python driver: the gpu stages only ever run on the
# linux vm, and a python driver would just shell out to these same commands.
# stages are independent so a partial rerun is cheap: `steer` alone is an
# overnight job, `tables` is seconds.
#
# `all` skips steer/steer-peak/judge/judge2. those four regenerate the committed
# CSVs behind every paper number and take most of a day on one gpu, so they are
# opt-in by name. run them under tmux.
#
# assumes cwd = repo root and the venv already active (or PY set below).
set -euo pipefail
stage=${1:-all}
PY=${PY:-python}

MODELS_ALL="google/gemma-2-9b google/gemma-2-9b-it Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct gpt2-large"
ALPHAS="-1 -0.5 -0.25 0 0.25 0.5 1"
slug() { echo "$1" | tr / _; }

# resid only: it's the sole hook point behind a paper number, and caching all
# three costs ~5 GB per 9B checkpoint instead of ~2. POINTS="resid attn mlp"
# to reproduce the early gemma attn/mlp probe csvs.
POINTS=${POINTS:-resid}

# every checkpoint is cached at the same --n-pos so the base-vs-instruct
# comparison is position-matched by construction. pos 0 is the 5th-from-last
# prompt token, pos 4 the final one; the paper's matched comparison uses pos 4.
if [[ $stage == cache || $stage == all ]]; then
  for M in $MODELS_ALL; do
    $PY pipeline/cache_activations.py --model "$M" --n-pos 5 \
      --points $POINTS --out "activations/$(slug "$M")/"
  done
fi

if [[ $stage == probes || $stage == all ]]; then
  for M in $MODELS_ALL; do
    $PY pipeline/linear_probes.py --acts "activations/$(slug "$M")/" \
      --point resid --transfer --all-pos --dump-preds --auroc
  done
  $PY baselines/lexical_baseline.py --dataset dataset.csv
  # base-vs-instruct direction alignment (the cosine numbers in section 4.1)
  for PAIR in "google_gemma-2-9b google_gemma-2-9b-it" \
              "Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct" \
              "meta-llama_Llama-3.1-8B meta-llama_Llama-3.1-8B-Instruct"; do
    set -- $PAIR
    $PY pipeline/direction_cosine.py \
      --base "activations/$1/" --instruct "activations/$2/" --pos -1
  done
fi

# ---- steering: the a priori sites (table 2, rows 1-3 and 5) -----------------
# layers were fixed a priori mid-stack: gemma L20, qwen L19, llama L18, gpt2 L22.
# llama's is NOT its probe peak (L22); `steer-peak` below covers that.
if [[ $stage == steer ]]; then
  declare -A LAYER=( [google/gemma-2-9b-it]=20 [Qwen/Qwen2.5-7B-Instruct]=19
                     [meta-llama/Llama-3.1-8B-Instruct]=18 [gpt2-large]=22 )
  for M in "${!LAYER[@]}"; do
    A="activations/$(slug "$M")/"
    $PY pipeline/steer.py --acts "$A" --model "$M" --layer "${LAYER[$M]}" \
      --alpha-mode relative --alphas $ALPHAS \
      --prompts prompts_eval.txt --experiment own
    for S in 0 1 2; do
      $PY pipeline/steer.py --acts "$A" --model "$M" --layer "${LAYER[$M]}" \
        --alpha-mode relative --alphas $ALPHAS \
        --prompts prompts_eval.txt --random-dir --seed "$S" --experiment random
    done
  done

  # band injection: nine layers, each with its own per-layer vector, total push
  # divided by band width so it's comparable to a single site. the experiment
  # name carries the band so the csv is self-describing.
  QWEN=activations/Qwen_Qwen2.5-7B-Instruct/
  LLAMA=activations/meta-llama_Llama-3.1-8B-Instruct/
  $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
    --layers 15 16 17 18 19 20 21 22 23 --alpha-mode relative \
    --alphas $ALPHAS --prompts prompts_eval.txt --experiment band_own_L15-23
  $PY pipeline/steer.py --acts "$LLAMA" --model meta-llama/Llama-3.1-8B-Instruct \
    --layers 18 19 20 21 22 23 24 25 26 --alpha-mode relative \
    --alphas $ALPHAS --prompts prompts_eval.txt --experiment band_own_L18-26
  for S in 0 1 2; do
    $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
      --layers 15 16 17 18 19 20 21 22 23 --alpha-mode relative \
      --alphas $ALPHAS --prompts prompts_eval.txt \
      --random-dir --seed "$S" --experiment band_random_L15-23
    $PY pipeline/steer.py --acts "$LLAMA" --model meta-llama/Llama-3.1-8B-Instruct \
      --layers 18 19 20 21 22 23 24 25 26 --alpha-mode relative \
      --alphas $ALPHAS --prompts prompts_eval.txt \
      --random-dir --seed "$S" --experiment band_random_L18-26
  done

  # cross-family transfer: build gemma's L20 direction, inject it into qwen.
  # the save pass still runs one no-op generation; cheap enough to leave alone.
  mkdir -p directions
  $PY pipeline/steer.py --acts activations/google_gemma-2-9b-it/ \
    --model google/gemma-2-9b-it --layer 20 \
    --save-direction directions/gemma-2-9b-it_L20_resid.npy --alphas 0
  $PY pipeline/steer.py --acts "$QWEN" --model Qwen/Qwen2.5-7B-Instruct \
    --layer 19 --direction directions/gemma-2-9b-it_L20_resid.npy \
    --alpha-mode relative --alphas $ALPHAS --prompts prompts_eval.txt \
    --experiment transfer
fi

# ---- steering: llama at its own probe peak (table 2, row 4) -----------------
# the a priori site missed llama's peak by four layers, so the null measured
# there was confounded with site choice. this rules that out: same protocol,
# single site at L22 and band re-centred on L20-28.
if [[ $stage == steer-peak ]]; then
  LL=activations/meta-llama_Llama-3.1-8B-Instruct/
  $PY pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 22 --alpha-mode relative --alphas $ALPHAS \
    --prompts prompts_eval.txt --experiment own_L22
  for S in 0 1 2; do
    $PY pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
      --layer 22 --alpha-mode relative --alphas $ALPHAS \
      --prompts prompts_eval.txt --random-dir --seed "$S" --experiment random_L22
  done
  $PY pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
    --layers 20 21 22 23 24 25 26 27 28 --alpha-mode relative \
    --alphas $ALPHAS --prompts prompts_eval.txt --experiment band_own_L20-28
  for S in 0 1 2; do
    $PY pipeline/steer.py --acts "$LL" --model meta-llama/Llama-3.1-8B-Instruct \
      --layers 20 21 22 23 24 25 26 27 28 --alpha-mode relative \
      --alphas $ALPHAS --prompts prompts_eval.txt \
      --random-dir --seed "$S" --experiment band_random_L20-28
  done
fi

# ---- judging ----------------------------------------------------------------
# the judge writes a `confidence` column in place. without --force it only fills
# rows that don't have one, so a new steering run can be judged without
# rescoring (and perturbing) the rows already behind a paper number.
if [[ $stage == judge ]]; then
  $PY pipeline/judge_confidence.py --results steering_results/judge_qwen \
    --judge-model Qwen/Qwen2.5-7B-Instruct
fi

# second judge over the same generations, into a parallel tree, so a result can
# be checked against a judge that isn't same-family with any model under test.
if [[ $stage == judge2 ]]; then
  mkdir -p steering_results/judge_llama
  cp steering_results/judge_qwen/*.csv steering_results/judge_qwen/*.md \
     steering_results/judge_llama/
  $PY pipeline/judge_confidence.py --results steering_results/judge_llama \
    --judge-model meta-llama/Llama-3.1-8B-Instruct --force
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
