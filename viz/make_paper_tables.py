"""
Regenerate every number in paper/main.tex from the committed CSVs.

Exists because Table 2's bootstrap CIs were computed once in a scratch session
and never committed: the method survived only in a changelog. If a number is in
the paper, it should come out of this script.

Covers Table 1 (peak + selection-free transfer accuracy per checkpoint), the
position-matched base-vs-instruct figures quoted in section 4.1 (the same two
estimators restricted to the final prompt token), the abstract's accuracy range,
the TF-IDF baseline row, and Table 2 (mean confidence at alpha=-0.5/+0.5,
Pearson corr(alpha, confidence) over the coherent band, 95% bootstrap CIs by
resampling the 40 prompts, and the degeneration counts behind the Llama L22
discussion). Not covered: mean residual norms, which need activations/.

Note the probe csvs' *_test columns are accuracy at a 0.5 threshold, not auroc
(see pipeline/linear_probes.py). The paper says accuracy for that reason.

Usage:
    venv/Scripts/python.exe viz/make_paper_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "probe_results"
STEER = ROOT / "steering_results" / "judge_qwen"
STEER2 = ROOT / "steering_results" / "judge_llama"
OUT = ROOT / "paper" / "tables_generated.md"

BAND = 0.5      # coherent steering band |alpha| <= BAND
DRAWS = 3000    # bootstrap resamples of the prompt set
SEED = 0        # fixed so the CIs are reproducible, not just plausible
FINAL_POS = 4   # final prompt token in a 5-position cache

# Table 1: family -> (base slug, instruct slug). None where the release doesn't exist.
FAMILIES = [
    ("Gemma-2-9b", "gemma-2-9b", "gemma-2-9b-it"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"),
    ("Llama-3.1-8B", "meta-llama_Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("GPT-2-large", "gpt2-large", None),
]

# Table 2: one row per (model, injection site). `own`/`random` are the a priori
# single sites; the banded runs carry their layer range in the experiment name.
# Llama appears twice because the a priori site (L18) missed its probe peak.
STEER_ROWS = [
    ("Gemma-2-9b-it", "google/gemma-2-9b-it", "gemma-2-9b-it",
     "L20", "own", "random"),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-7B-Instruct",
     "L19 / L15-23", "band_own_L15-23", "band_random_L15-23"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "meta-llama_Llama-3.1-8B-Instruct",
     "L18 / L18-26", "band_own_L18-26", "band_random_L18-26"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "meta-llama_Llama-3.1-8B-Instruct",
     "L22 single site", "own_L22", "random_L22"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "meta-llama_Llama-3.1-8B-Instruct",
     "L22 / L20-28", "band_own_L20-28", "band_random_L20-28"),
    ("GPT-2-large", "gpt2-large", "gpt2-large", "L22", "own", "random"),
]

# The asymmetric Llama effect the paper reports and then discounts. Both are
# measured at alpha=-1, outside the coherent band, hence the degeneration check.
LLAMA_OOB = [
    ("single site L22", "own_L22", "random_L22"),
    ("band L20-28", "band_own_L20-28", "band_random_L20-28"),
]


def _probe(slug):
    f = PROBE / slug / "probe_resid_transfer.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["best"] = d[["p2f_test", "f2p_test"]].max(axis=1)
    return d


def peak_transfer(d):
    """Peak accuracy over the grid, plus every layer that ties it.

    max over rows of max(p2f_test, f2p_test): best-of-direction stacked on
    best-of-layer (and, when d spans positions, best-of-position). That triple
    selection is why the paper calls this an upper bound rather than a point
    estimate.
    """
    peak = d["best"].max()
    row = d.loc[d["best"].idxmax()]
    # ties at 4 s.f. -- the paper prints 3 decimals, so anything closer than
    # that is an argmax-first-row artifact, not a real peak layer
    tied = sorted(d.loc[np.isclose(d["best"], peak, atol=5e-5), "layer"].unique())
    return peak, int(row["layer"]), int(row["pos"]), tied


def cross_selected(d):
    """Selection-free companion to the peak: pick the cell on ONE transfer
    direction and report the OTHER, then average the two orders.

    The peak reuses the test direction for selection, so it can only be read as
    an upper bound. This never scores the direction it selected on, so it is a
    point estimate.
    """
    return (d.loc[d["p2f_test"].idxmax(), "f2p_test"]
            + d.loc[d["f2p_test"].idxmax(), "p2f_test"]) / 2


def peak_over_layers_at(slug, pos):
    """Peak accuracy at one token position: best-of-layer, best-of-direction.

    Falls back to the last cached position when `pos` wasn't cached, which is
    the gpt-2 case (one position, not five).
    """
    d = _probe(slug)
    if d is None:
        return None
    pos = pos if pos in set(d["pos"]) else d["pos"].max()
    return d[d["pos"] == pos]["best"].max()


def band_rows(df, model, experiment):
    """Coherent-band add rows for one (model, experiment)."""
    m = ((df["model"] == model) & (df["experiment"] == experiment)
         & (df["mode"] == "add") & (df["alpha"].abs() <= BAND))
    return df[m]


def corr_ci(sub, seed=SEED):
    """Pearson corr(alpha, confidence) + 95% CI, resampling PROMPTS.

    The prompt is the unit of replication (each contributes one row per alpha),
    so resampling rows would understate the interval. The rng is seeded per
    call, not shared across rows, so adding a row can't shift another row's CI.
    """
    rng = np.random.default_rng(seed)
    qs = sorted(sub["question"].unique())
    by_q = {q: sub[sub["question"] == q] for q in qs}
    point = np.corrcoef(sub["alpha"], sub["confidence"])[0, 1]
    boots = np.empty(DRAWS)
    for i in range(DRAWS):
        pick = rng.choice(len(qs), len(qs), replace=True)
        d = pd.concat([by_q[qs[j]] for j in pick], ignore_index=True)
        boots[i] = np.corrcoef(d["alpha"], d["confidence"])[0, 1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def load_steer(root):
    frames = []
    for c in sorted(root.glob("*.csv")):
        if c.stem == "summary":
            continue
        d = pd.read_csv(c)
        if "confidence" in d.columns and d["confidence"].notna().any():
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def is_degenerate(text, ngram=4, thresh=0.5):
    """Repetition collapse: >50% of 4-grams are repeats.

    Only used to check whether the large out-of-band Llama effect survives
    dropping broken generations. It doesn't survive as a clean comparison,
    which is the point the paper makes.
    """
    w = str(text).split()
    if len(w) < 3 * ngram:
        return False
    grams = [" ".join(w[i:i + ngram]) for i in range(len(w) - ngram + 1)]
    return (1 - len(set(grams)) / len(grams)) > thresh


def table1(lines):
    lines.append("## Table 1 -- transfer accuracy over the (layer, position) grid\n")
    lines.append("| family | base (peak) | instruct (peak) | delta | "
                 "base (sel-free) | instruct (sel-free) |")
    lines.append("|---|---|---|---|---|---|")
    peaks, notes = [], []
    for family, base, instruct in FAMILIES:
        cells, free, raw = {}, {}, {}
        for role, slug in (("base", base), ("instruct", instruct)):
            d = _probe(slug) if slug else None
            if d is None:
                cells[role] = free[role] = "--"
                continue
            peak, layer, pos, tied = peak_transfer(d)
            peaks.append(peak)
            raw[role] = peak
            cells[role] = f"{peak:.3f} (L{tied[0]}, p{pos})"
            free[role] = f"{cross_selected(d):.3f}"
            if len(tied) > 1:
                notes.append(f"{family} {role} ties at layers "
                             f"{', '.join('L' + str(t) for t in tied)} "
                             f"({peak:.4f}) -- a single-layer label there is "
                             f"an argmax-first-row artifact")
        delta = f"{raw['instruct'] - raw['base']:+.3f}" if len(raw) == 2 else "--"
        lines.append(f"| {family} | {cells['base']} | {cells['instruct']} | "
                     f"{delta} | {free['base']} | {free['instruct']} |")
    lines.append("")
    for n in notes:
        lines.append(f"- tie warning: {n}")
    lines.append(f"- abstract range (>=7B checkpoints): "
                 f"{min(p for p in peaks if p > 0.6):.2f}--{max(peaks):.2f}")
    lines.append("- the peak selects a token position as well as a layer, and it"
                 " lands on a different position for base and instruct in every"
                 " family, so the delta column compares two differently-selected"
                 " cells. the matched-position block below is the comparison the paper quotes.")

    lex = PROBE / "_lexical_baseline.csv"
    if lex.exists():
        r = pd.read_csv(lex).iloc[0]
        lines.append(f"- tf-idf baseline on the same split: in-distribution "
                     f"{r['split_single']:.2f}, transfer p2f {r['transfer_p2f']:.3f} "
                     f"/ f2p {r['transfer_f2p']:.3f} (chance)")
    lines.append("")


def table3(lines):
    lines.append(f"## Position-matched base vs instruct, final prompt token (p{FINAL_POS})\n")
    lines.append("| family | base (peak/layer) | instruct (peak/layer) | delta | "
                 "base (sel-free) | instruct (sel-free) | delta |")
    lines.append("|---|---|---|---|---|---|---|")
    per_pos = []
    for family, base, instruct in FAMILIES:
        if instruct is None:
            continue
        db, di = _probe(base), _probe(instruct)
        if db is None or di is None or FINAL_POS not in set(db["pos"]):
            continue
        b4, i4 = db[db["pos"] == FINAL_POS], di[di["pos"] == FINAL_POS]
        pb, pi = b4["best"].max(), i4["best"].max()
        sb, si = cross_selected(b4), cross_selected(i4)
        lines.append(f"| {family} | {pb:.3f} | {pi:.3f} | {pi - pb:+.3f} | "
                     f"{sb:.3f} | {si:.3f} | {si - sb:+.3f} |")
        for p in sorted(set(db["pos"]) & set(di["pos"])):
            per_pos.append(di[di["pos"] == p]["best"].max()
                           - db[db["pos"] == p]["best"].max())
    lines.append("")
    lines.append(f"- peak-over-layers delta computed separately at each cached"
                 f" position ranges {min(per_pos):+.3f} to {max(per_pos):+.3f}"
                 f" and changes sign within every family")
    lines.append("- the selection-free estimator is positive in all three"
                 " families here; the peak estimator is not")
    lines.append("")


def table2(lines):
    df = load_steer(STEER)
    lines.append("## Table 2 -- steering outcome (primary judge)\n")
    lines.append("| model | site | accuracy @ p4 | experiment | conf @ -0.5 | "
                 "conf @ +0.5 | corr (95% CI) |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, model, slug, site, own, control in STEER_ROWS:
        acc = peak_over_layers_at(slug, FINAL_POS)
        acc_s = "--" if acc is None else f"{acc:.2f}"
        for exp in (own, control):
            sub = band_rows(df, model, exp)
            if sub.empty:
                lines.append(f"| {label} | {site} | {acc_s} | {exp} | -- | -- | MISSING |")
                continue
            means = sub.groupby("alpha")["confidence"].mean()
            p, lo, hi = corr_ci(sub)
            lines.append(f"| {label} | {site} | {acc_s} | {exp} | "
                         f"{means.get(-BAND, np.nan):.1f} | "
                         f"{means.get(BAND, np.nan):.1f} | "
                         f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines.append("")
    lines.append(f"- bootstrap: {DRAWS} draws, resampling the "
                 f"{df['question'].nunique()} prompts with replacement, "
                 f"numpy default_rng({SEED})")
    lines.append("- random controls pool the three seeds")
    lines.append("- accuracy @ p4 is peak-over-layers at the final prompt token,"
                 " best-of-direction: stricter than the table 1 entry because it"
                 " does not also select over position")
    lines.append("")


def llama_out_of_band(lines):
    """The alpha=-1 Llama asymmetry, and why the paper doesn't lean on it."""
    lines.append("## Llama at alpha=-1 (outside the coherent band)\n")
    lines.append("| judge | experiment | own | random | gap | own degenerate | "
                 "random degenerate | own after filter |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for judge, root in (("qwen (primary)", STEER), ("llama (second)", STEER2)):
        if not root.exists():
            continue
        df = load_steer(root)
        df = df[(df["mode"] == "add") & (df["alpha"] == -1.0)]
        for site, own, control in LLAMA_OOB:
            o = df[df["experiment"] == own]
            r = df[df["experiment"] == control]
            if o.empty or r.empty:
                continue
            od = o["text"].map(is_degenerate)
            rd = r["text"].map(is_degenerate)
            lines.append(
                f"| {judge} | {site} | {o['confidence'].mean():.1f} | "
                f"{r['confidence'].mean():.1f} | "
                f"{o['confidence'].mean() - r['confidence'].mean():+.1f} | "
                f"{od.sum()}/{len(o)} | {rd.sum()}/{len(r)} | "
                f"{o[~od]['confidence'].mean():.1f} |")
    lines.append("")
    lines.append("- the gap is large and both judges agree on it, but it sits"
                 " outside the |alpha| <= 0.5 range where generation stays"
                 " coherent, and the own-direction and control means are taken"
                 " over very different numbers of intact generations. section"
                 " 4.2 reports it and declines to read it as causal.")
    lines.append("")


def main():
    lines = ["# paper numbers, regenerated",
             "",
             "Generated by viz/make_paper_tables.py from the committed CSVs.",
             "*_test columns are accuracy at a 0.5 threshold, not auroc.",
             ""]
    table1(lines)
    table3(lines)
    table2(lines)
    llama_out_of_band(lines)
    lines += [
        "## not derivable from committed csvs",
        "",
        "- mean residual norms (~276 gemma / ~82 qwen): needs activations/,"
        " which is gitignored. recompute during a re-cache.",
        "- base-vs-instruct direction cosines: probe_results/_cosine_*.csv,"
        " written by pipeline/direction_cosine.py during the probes stage.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
