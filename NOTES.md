lab notebook, kept as-is. chronological notes from each phase of the
project -- the readme is the distilled version.

# Quantifying Uncertainty in LLMs

A research project attempting to locate directions in latent space that encode model uncertainty and prove causality through ablation and steering. Previous research has shown promise, but has noted the difficulty in cross-domain transfer (the concept of uncertainty changes across domains!). Hopefully, we can identify methods of uncertainty quantification that work across domains and can be utilized by model providers to alleviate sycophantic tendencies of models.

Research done by Feifan Liu with advising from Dr. Amin Alipour from the University of Houston.

## Notes on each Phase

### Dataset generation

- length of prompts might've been important, had to verify
- some prompts included "research shows" "empirically" which is a huge confounder that probe will latch onto
- wanted to verify their effects by training a linear probe without activations and checking AUROC -- AUROC was higher than .5 (random chance) w/ evidentials, and very slightly higher than .5 b/c of prompt length. fixed by removing all evidentials + changing prompt length

### Caching and Visualizing Activations

- positions 1, 3, and 4 seem most usable -- an interesting pattern worth exploring, but i think it's likely because 1 corresponds to ", 3 to A, 4 to :, whereas 2 is \n
- mid layers (18-26) seem to be have the strongest separation between past_known, past_unknown, future_known, future_unknown prompts
- every model and activations seem to show distinction along the time axis as well
- gemma-2-2b-it, gemma-2-9b seem to show similar patterns
  - gemma-2-9b-it however seems to have strong geometric shape at MLP and resid (four corners) -- why might this be? RLHF may make the model more apt to express its level of uncertainty
- considering incorporating conformal prediction into this research

### Linear Probes

- i believe the probe along will easily find a relationship between the activations and what category the prompt was; the real test is whether steering and ablation cause the model's output to reflect a different level of uncertainty
- cross-validation on held out sets split by time will also help. if a probe that is trained on past_unknown and past_known also scores high on their future counterparts, it is more likely that the model is encoding epistemic uncertainty rather than some temporal markers
- results are interesting: probe easily separates known/unknown in distribution since it memorizes the training data, so cross-validation is important
- probe scored excellently on split test, which randomly split all data 80/20 (80 train, 20 test) with both past and future data on both sides
  - when exposed to both sets of time directions, the probe cannot simply learn the temporal signals from just path or just future, so it scores high
- probe scored well above change (.7-.9 as opposed to .5) in both directions (f2p and p2f) across the time axis in the transfer test, peaking in the mid layers (L12-25)
  - across models, f2p seems to perform worse than p2f
    - why?
  - across models, gemma-2-2b-it peaks around layer 12
    - mlp, attention, and resid are all somewhat similar
      - mlp and resid are more similar generally speaking
      - in gemma-2-9b-it, mlp and attn curves show rapid swings from ~.9 to .5 around later layers (L25+). only resid is stable, but it crashes in later layers too
        - possibly because RLHF leads to temporal signals being very differently expressed by model in later layers which interferes with transfer
      - in gemma-2-9b, the mlp and resid transfer tests generally show convergence in the later layers
- its possible the dataset has some contamination -- known labels generally have more hedging and also future_known has shorter timeframes (mentions 2027, 2030) than future_unknown (2035, 2050)
  - i considered rewriting parts of the dataset, but the ablation + activation steering phase coming up should help indicate whether it is necessary. if, when steering a model on answering a history or cooking question, we get mentions related to AI in education, or hedging verbs, or years, then we know we need to change the dataset
  - also added TF-IDF which is a naive text classifier heuristic as a baseline -- if probe can't score higher then there's no point
    - the probe scores about .07 higher than the baseline on the split test (.83 compared to ~.9), but scores much higher on the transfer test (.7-.9 vs. .5 (about random chance!))
- is there a way to look at each individual prompt and the probe results to isolate whether the dataset is truly faulty?
- systematically remove words from the input using delta debugging -- maybe also use the probe to score uncertainty even as the label stays the same
  - you can remove years for example to check how much the probe relies on it
- maybe vary temperature of model
- add negations, the model should flag future_unknown to be the same anyway. but for past_known, negating it would cause model behavior to become funky. is it "known" if the fact is false, or "unknown"

### Activation Steering

- we use steering to inject the direction that the probe identifies in the residual stream and amplify it in positive and negative directions to test its effect on model outputs
- this is a much stronger causal indication that the direction identified is close to uncertainty, since the probe could be picking up some other signal
- the direction is diff-of-means: v = mean(act | known) - mean(act | unknown) at a *mid-layer*, layer 20 at the residual stream at position -1 (":")
- we hook the output residual of layer 20 at **every position** and decoding step, and add the direction v and scale it by alpha
- we also set do_sample=False for greedy decoding (pick the top logit) so changes in outputs are attributable to the steering rather than noise
- with alpha [-4 ,4] on neutral OOD prompts (French rev., tomato sauce recipe, Hamlet, weather in Chicago)
  - negative alpha (toward unknown): the model manufactures doubt -- "no single answer", "no consensus", "complex", calls plain facts "a myth"
  - positive alpha (toward known): assertive, at +8 it tips into overconfidence and sycophancy ("next tuesday is today") and gets facts wrong
- the effect is monotonic across all 5 unrelated topics, confirming the findings

### Ablation

Ablating at pos=-1 on L20 of the residual stream with alpha = 1 and even alpha = 2, 4, 8 didn't yield the expected results. Model outputs only led to over-confidence at alpha=8, and even at alpha=4 the model often claimed the "unknown" prompt was debatable (we would expect the model to exhibit confidence at this alpha when ablating the vector we believe to correspond to uncertainty).

So, I decided to ablate per-layer from L12-26.

### Multi-model steering (gpt2-large, qwen2.5-7b, llama-3.1-8b)

- wanted to see if the gemma finding generalizes -- does the SAME direction transfer to another model, and does each model have its OWN steerable direction?
- same direction (gemma's L20 diffmean) only makes sense to add if d_model matches -- only qwen2.5-7b (3584) lines up. tried it: flat. at coherent alpha nothing moves, push harder and it just breaks into salad. gemma's direction is noise in qwen's basis. no cross-family transfer
- own direction per model (diffmean at each model's probe-peak layer) is the real test. two things had to get fixed first:
    - alpha scale: |v| and residual norm differ a lot across models (gemma ||hs||~276 vs qwen ~82) so a fixed alpha overshoots one and undershoots the other. switched to relative alpha = fraction of mean||hs||, so a given alpha means the same push everywhere and the summary table is actually comparable
    - metric: the gemma-tuned hedge-word counter reads ~0 on qwen's register ("no claim can state..."), so swapped in an llm-judge confidence score -- P(yes) from yes/no logits, 0-100. and added a random-direction control (same norm) since "steering breaks output" could just be "any big vector breaks output" -- a real knob has to beat random
- results (coherent band |alpha| <= 0.5, 40 prompts, 3 random seeds):
    - gemma-2-9b-it: single-layer injection is already causal, confidence climbs monotonically, way above random. clean
    - qwen2.5-7b: single layer does basically nothing (~random). BUT injecting across a band (L15-23, each layer its own per-layer vector, total push normalized by 1/n) wakes it up -- confidence 33->57 monotonic, beats random_band. so the direction IS causal, single-site just dilutes it (downstream layers rewrite the residual)
    - llama-3.1-8b: nothing. single layer flat, band flat, both ~random. decodable but not steerable
    - gpt2-large: nothing, but its probe is basically chance (.55) so there's no real direction to inject anyway
- so steerability is a gradient and it does NOT track how decodable the direction is: gemma (single-site) > qwen (band only) > llama (inert) > gpt2 (no direction). llama's probe reads .80 but you can't steer it at all
- takeaway: the probe finding a direction != that direction being a causal knob. decode and control are different things

### base vs instruct (does RLHF create the direction?)

- gpt2 being near-chance made me wonder if base models just don't encode uncertainty. but gpt2 is tiny + old, so base-ness is confounded with scale/era -- can't conclude anything from one model. so i ran the base version of each pair (probe only, transfer test)
- peak resid transfer-AUROC:
    - gemma-2-9b base .903 vs -it .890
    - qwen2.5-7b base .783 vs instruct .823
    - llama-3.1-8b base .797 vs instruct .797
- base ~= instruct in all three, deltas < .04 (noise). so the decodable direction is a PRETRAINING feature -- RLHF doesn't create it
- gpt2's .553 is a capability/scale floor, not base-ness -- the three bigger bases all decode fine
- so RLHF's role (if any) is in CONTROL, not decode. the direction is there from pretraining; whether you can causally push on it is the separate, model-dependent question
