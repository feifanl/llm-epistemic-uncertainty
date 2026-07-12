# Steering summary

Cells are mean±SEM across prompts.

## hedges

lower = more confident; expect corr(alpha,hedges) < 0

| model | experiment | mode | a=-1 | a=-0.5 | a=-0.25 | a=0 | a=0.25 | a=0.5 | a=1 | corr(a,hedges) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | own | add | 0.6±0 | 0.4±0 | 0.5±0 | 0.3±0 | 0.2±0 | 0.3±0 | 0.1±0 | -0.21 |
| Qwen/Qwen2.5-7B-Instruct | random | add | 0.4±0 | 0.3±0 | 0.3±0 | 0.3±0 | 0.3±0 | 0.3±0 | 0.3±0 | -0.04 |
| Qwen/Qwen2.5-7B-Instruct | transfer | add | 0.2±0 | 0.4±0 | 0.3±0 | 0.3±0 | 0.3±0 | 0.2±0 | 0.3±0 | +0.01 |
| google/gemma-2-9b-it | own | add | 2.0±0 | 0.9±0 | 0.7±0 | 0.4±0 | 0.4±0 | 0.2±0 | 0.1±0 | -0.56 |
| google/gemma-2-9b-it | random | add | 0.6±0 | 0.5±0 | 0.5±0 | 0.4±0 | 0.5±0 | 0.6±0 | 0.5±0 | -0.02 |
| gpt2-large | own | add | 1.7±0 | 0.6±0 | 0.2±0 | 0.1±0 | 0.1±0 | 0.0±0 | 0.0±0 | -0.31 |
| gpt2-large | random | add | 0.1±0 | 0.2±0 | 0.0±0 | 0.1±0 | 0.1±0 | 0.0±0 | 0.0±0 | -0.05 |
| meta-llama/Llama-3.1-8B-Instruct | own | add | 0.8±0 | 0.6±0 | 0.4±0 | 0.5±0 | 0.3±0 | 0.3±0 | 0.1±0 | -0.29 |
| meta-llama/Llama-3.1-8B-Instruct | random | add | 0.4±0 | 0.3±0 | 0.4±0 | 0.5±0 | 0.5±0 | 0.5±0 | 0.5±0 | +0.06 |

## confidence

higher = more confident; expect corr(alpha,confidence) > 0

| model | experiment | mode | a=-1 | a=-0.5 | a=-0.25 | a=0 | a=0.25 | a=0.5 | a=1 | corr(a,confidence) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | own | add | 34.5±7 | 47.3±8 | 43.2±8 | 44.8±8 | 52.5±8 | 56.5±7 | 7.0±3 | -0.10 |
| Qwen/Qwen2.5-7B-Instruct | random | add | 40.8±4 | 48.5±4 | 44.9±4 | 44.8±4 | 45.7±4 | 47.6±4 | 35.0±4 | -0.03 |
| Qwen/Qwen2.5-7B-Instruct | transfer | add | 41.2±7 | 46.2±8 | 44.9±8 | 44.8±8 | 47.2±8 | 49.1±8 | 37.6±7 | -0.01 |
| google/gemma-2-9b-it | own | add | 0.1±0 | 30.3±7 | 37.8±7 | 49.5±8 | 53.6±8 | 65.2±7 | 48.0±8 | +0.34 |
| google/gemma-2-9b-it | random | add | 36.5±4 | 43.9±4 | 48.7±4 | 49.5±4 | 50.9±4 | 46.8±4 | 41.3±4 | +0.03 |
| gpt2-large | own | add | 5.0±3 | 12.5±5 | 12.6±5 | 18.5±6 | 26.6±6 | 18.0±6 | 0.0±0 | +0.01 |
| gpt2-large | random | add | 7.1±2 | 25.1±4 | 16.1±3 | 18.5±3 | 17.3±3 | 17.7±3 | 2.0±1 | -0.06 |
| meta-llama/Llama-3.1-8B-Instruct | own | add | 10.5±4 | 41.9±7 | 41.7±7 | 52.0±8 | 50.2±7 | 48.0±8 | 29.8±7 | +0.13 |
| meta-llama/Llama-3.1-8B-Instruct | random | add | 34.2±4 | 47.5±4 | 49.9±4 | 52.0±4 | 47.8±4 | 45.3±4 | 19.2±3 | -0.08 |

