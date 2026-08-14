# Decodability and Causal Control of Epistemic Uncertainty in LLMs

We would like for LLMs to be well-calibrated and express their epistemic
uncertainty in model responses, especially when considering their sycophantic
tendencies. We use linear probes in three model families to decode a direction
from the residual stream activations that corresponds to epistemic uncertainty.
Such a direction exists in every model we tested above 7B parameters.

Interestingly, steering a model via its decoded direction affects its expressed
confidence in only some model families. This repository contains our results and
the code to replicate our experiments.

Pre-print: [`paper/main.tex`](paper/main.tex).
Authors: Feifan Liu (UT Austin), advised by Dr. Amin Alipour (University of Houston).

## Findings

1. **The direction is present after pre-training.** Every model checkpoint at 7B
   parameters or above decodes known/unknown from the residual stream at 0.78–0.92
   transfer accuracy. Base and instruct variants decode comparably, and their
   difference-of-means directions are closely aligned (cosine 0.57–0.75, compared to
   ~0.02 for random same-width vectors). Thus, instruction tuning improves an
   existing direction rather than creating a new one.
2. **Steerability does not track decodability.** Gemma-2-9b-it steers from
   single site injection. Despite decoding just as well (transfer accuracy ~0.8),
   Llama-3.1-8B-Instruct is inert when steered via both single-site and multi-site
   (banded across layers) injection.
3. **Single-site injection can be too weak.** Qwen2.5-7B-Instruct is inert at
   one layer but causal across a nine-layer band (33 -> 57), separating cleanly
   from a matched-norm random band.
4. **No cross-family transfer.** The direction we decoded from Gemma unsurprisingly
   had no coherent effect in Qwen (same residual width, 3584) and steering via Gemma's
   direction had the same effect as the random control.

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | Activation caching, probes, steering, judging, ablation, delta-debugging |
| `viz/` | Probe curves, steering summaries, paper figures, table regeneration |
| `baselines/` | TF-IDF floor, per-prompt lexical, easy/hard comparison |
| `probe_results/` | One directory per checkpoint, plus cosine and lexical baselines |
| `steering_results/judge_qwen/` | Primary judge. Every paper number cites these CSVs |
| `steering_results/judge_llama/` | Same generations, second judge, as a robustness check |
| `paper/` | `main.tex`, references, figures, `tables_generated.md` |
| `tests/` | One CPU smoke test over the pipeline |
| `NOTES.md` | Lab notes this README is distilled from |

Directory names under `steering_results/` identify the judge, not the model
being steered. Steering CSVs are named `{model_slug}__{experiment}`, where the
experiment carries its injection site (`own`, `own_L22`, `band_own_L15-23`,
`band_random_L18-26_s0`), so a file is readable without the script that produced
it.

## Reproducing

```bash
./reproduce.sh
```

Runs caching, probes, figures, tables and the smoke test. The four GPU-heavy
stages are opt-in by name, because they overwrite the committed CSVs behind
every number in the paper:

```bash
./reproduce.sh steer && ./reproduce.sh steer-peak && ./reproduce.sh judge && ./reproduce.sh judge2
```

`viz/make_paper_tables.py` regenerates every number the paper reports from the
committed CSVs, including seeded bootstrap intervals and the degeneration counts
behind Section 4.2. `activations/` is gitignored, so anything downstream of it
needs a re-cache first.

Hardware: one 24 GB GPU for the 9B models in bf16. Caching is 10–20 minutes per
model, steering 30–60 minutes per run, each judge pass a few hours.
`google/gemma-2-9b(-it)` and `meta-llama/Llama-3.1-8B(-Instruct)` are gated on
Hugging Face; Qwen and GPT-2 are open. Bit-exact reproduction is not guaranteed
across GPUs and library versions, and no conclusion here depends on it.

## Citation

```bibtex
@misc{liu2026uncertainty,
  title  = {Decodability and Causal Control of Epistemic Uncertainty in LLMs},
  author = {Feifan Liu and Amin Alipour},
  year   = {2026}
}
```

MIT licensed. See [`LICENSE`](LICENSE).
