"""
Regenerate every number in paper/main.tex from the committed CSVs.

Exists because Table 2's bootstrap CIs were computed once in a scratch session
and never committed -- the method survived only in a changelog. If a number is
in the paper, it should come out of this script.

Covers: Table 1 (peak transfer accuracy per checkpoint + delta), the abstract's
0.78-0.90 range, the TF-IDF baseline row, and Table 2 (mean confidence at
alpha=-0.5/+0.5, Pearson corr(alpha, confidence) over the coherent band, and
95% bootstrap CIs by resampling the 40 prompts). Not covered: mean residual
norms (needs activations/), flagged in the output.

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
STEER = ROOT / "steering_results"
OUT = ROOT / "paper" / "tables_generated.md"

BAND = 0.5      # coherent steering band |alpha| <= BAND
DRAWS = 3000    # bootstrap resamples of the prompt set
SEED = 0        # fixed so the CIs are reproducible, not just plausible

# Table 1: family -> (base slug, instruct slug). None where the release doesn't exist.
FAMILIES = [
    ("Gemma-2-9b", "gemma-2-9b", "gemma-2-9b-it"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct"),
    ("Llama-3.1-8B", "meta-llama_Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct"),
    ("GPT-2-large", "gpt2-large", None),
]

# Table 2: model -> (own experiment that the paper reports, its matched control).
# The reported method is the one that works for that model: gemma steers at a
# single site, qwen only across the band, llama neither (band reported since
# it's the stronger test of the null).
STEER_ROWS = [
    ("Gemma-2-9b-it", "google/gemma-2-9b-it", "own", "random"),
    ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "band_own", "band_random"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "own_band", "random_band"),
    ("GPT-2-large", "gpt2-large", "own", "random"),
]


def peak_transfer(slug):
    """Peak accuracy over the (layer, pos) grid + every layer that ties it.

    max over rows of max(p2f_test, f2p_test) -- best-of-direction stacked on
    top of best-of-layer, which is why the paper calls it an upper bound.
    """
    f = PROBE / slug / "probe_resid_transfer.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["best"] = d[["p2f_test", "f2p_test"]].max(axis=1)
    peak = d["best"].max()
    # ties at 4 s.f. -- the paper prints 3 decimals, so anything closer than
    # that is an argmax-first-row artifact, not a real peak layer
    tied = sorted(d.loc[np.isclose(d["best"], peak, atol=5e-5), "layer"].unique())
    return peak, tied


def cross_selected(slug):
    """Selection-free companion to the peak: pick the (layer, pos) cell on ONE
    transfer direction and report the OTHER, then average the two.

    The peak stacks best-of-layer on best-of-direction, so it can only be read
    as an upper bound. This picks the cell on held-out-ish evidence, so it's a
    point estimate. Base checkpoints drop much further than instruct ones --
    but they're also cached at a different token position, so the gap is not
    yet interpretable (see the matched re-cache in reproduce.sh).
    """
    f = PROBE / slug / "probe_resid_transfer.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return (d.loc[d["p2f_test"].idxmax(), "f2p_test"]
            + d.loc[d["f2p_test"].idxmax(), "p2f_test"]) / 2


def band(df, model, experiment):
    """Coherent-band add rows for one (model, experiment)."""
    m = ((df["model"] == model) & (df["experiment"] == experiment)
         & (df["mode"] == "add") & (df["alpha"].abs() <= BAND))
    return df[m]


def corr_ci(sub, rng):
    """Pearson corr(alpha, confidence) + 95% CI, resampling PROMPTS.

    The prompt is the unit of replication (each contributes one row per alpha),
    so resampling rows would understate the interval.
    """
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


def load_steer():
    frames = []
    for c in sorted(STEER.glob("*.csv")):
        if c.stem == "summary":
            continue
        d = pd.read_csv(c)
        if "confidence" in d.columns and d["confidence"].notna().any():
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def table1(lines):
    lines.append("## Table 1 -- peak residual transfer accuracy\n")
    lines.append("| family | base (peak) | instruct (peak) | delta | "
                 "base (sel-free) | instruct (sel-free) |")
    lines.append("|---|---|---|---|---|---|")
    peaks = []
    notes = []
    for family, base, instruct in FAMILIES:
        cells, free = {}, {}
        for role, slug in (("base", base), ("instruct", instruct)):
            r = peak_transfer(slug) if slug else None
            if r is None:
                cells[role] = free[role] = "--"
                continue
            peak, tied = r
            peaks.append(peak)
            cells[role] = f"{peak:.3f} (L{tied[0]})"
            free[role] = f"{cross_selected(slug):.3f}"
            if len(tied) > 1:
                notes.append(f"{family} {role} ties at layers "
                             f"{', '.join('L' + str(t) for t in tied)} "
                             f"({peak:.4f}) -- a single-layer label there is "
                             f"an argmax-first-row artifact")
        if cells["base"] != "--" and cells["instruct"] != "--":
            d = peak_transfer(instruct)[0] - peak_transfer(base)[0]   # as in Table 1
            delta = f"{d:+.3f}"
        else:
            delta = "--"
        lines.append(f"| {family} | {cells['base']} | {cells['instruct']} | "
                     f"{delta} | {free['base']} | {free['instruct']} |")
    lines.append("")
    for n in notes:
        lines.append(f"- tie warning: {n}")
    lines.append(f"- abstract range (>=7B checkpoints): "
                 f"{min(p for p in peaks if p > 0.6):.2f}--{max(peaks):.2f}")

    lex = PROBE / "_lexical_baseline.csv"
    if lex.exists():
        r = pd.read_csv(lex).iloc[0]
        lines.append(f"- tf-idf baseline on the same split: in-distribution "
                     f"{r['split_single']:.2f}, transfer p2f {r['transfer_p2f']:.3f} "
                     f"/ f2p {r['transfer_f2p']:.3f} (chance)")
    lines.append("")


def table2(lines):
    df = load_steer()
    rng = np.random.default_rng(SEED)
    lines.append("## Table 2 -- steering outcome\n")
    lines.append("| model | experiment | conf @ -0.5 | conf @ +0.5 | corr (95% CI) |")
    lines.append("|---|---|---|---|---|")
    for label, model, own, control in STEER_ROWS:
        for exp in (own, control):
            sub = band(df, model, exp)
            if sub.empty:
                lines.append(f"| {label} | {exp} | -- | -- | MISSING |")
                continue
            means = sub.groupby("alpha")["confidence"].mean()
            p, lo, hi = corr_ci(sub, rng)
            lines.append(f"| {label} | {exp} | {means.get(-BAND, np.nan):.1f} | "
                         f"{means.get(BAND, np.nan):.1f} | "
                         f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines.append("")
    lines.append(f"- bootstrap: {DRAWS} draws, resampling the "
                 f"{df['question'].nunique()} prompts with replacement, "
                 f"numpy default_rng({SEED})")
    lines.append("- random controls pool the three seeds")
    lines.append("")


def main():
    lines = ["# paper numbers, regenerated",
             "",
             "Generated by viz/make_paper_tables.py from the committed CSVs.",
             "*_test columns are accuracy at a 0.5 threshold, not auroc.",
             ""]
    table1(lines)
    table2(lines)
    lines += [
        "## not derivable from committed csvs",
        "",
        "- mean residual norms (~276 gemma / ~82 qwen): needs activations/,"
        " which is gitignored. recompute during a re-cache.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text, encoding="utf-8")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
