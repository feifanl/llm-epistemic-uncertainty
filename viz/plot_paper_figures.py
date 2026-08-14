"""
Paper figures for paper/main.tex. Vector PDF, plain style (no seaborn, no in-axes
titles beyond a small panel tag -- the LaTeX caption carries the description).

Fig 1  steering dose-response: judge confidence vs alpha, 2x2 instruct models,
       own-direction vs matched-norm random control (SEM bands), coherent band
       |alpha|<=0.5 shaded. Qwen/Llama also show the multi-layer band injection.
Fig 2  transfer accuracy vs normalized layer depth, four instruct models with
       their base checkpoints dashed (envelope: max over cached positions per
       layer -- only gemma-2-9b base has >1).

The probe csvs' *_test columns are accuracy at a 0.5 threshold, not auroc; see
pipeline/linear_probes.py.

Data: steering_results/judge_qwen/*.csv (confidence col from judge_confidence.py),
      probe_results/<slug>/probe_resid_transfer.csv.

Usage:
    venv/Scripts/python.exe viz/plot_paper_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STEER = ROOT / "steering_results/judge_qwen"
PROBE = ROOT / "probe_results"
OUT = ROOT / "paper" / "figures"
BAND = 0.5  # coherent steering band |alpha| <= BAND

# Fig 1: panel -> (model id, [(experiment, label, style_kwargs)])
PANELS = [
    ("google/gemma-2-9b-it", "Gemma-2-9b-it", [
        ("own", "single-site", dict(color="C0", marker="o")),
        ("random", "random", dict(color="0.6", marker="x", ls="--")),
    ]),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", [
        ("own", "single-site", dict(color="C0", marker="o")),
        ("band_own_L15-23", "band (L15-23)", dict(color="C3", marker="s")),
        ("random", "random", dict(color="0.6", marker="x", ls="--")),
    ]),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct", [
        ("own", "single-site", dict(color="C0", marker="o")),
        ("band_own_L18-26", "band (L18-26)", dict(color="C3", marker="s")),
        ("random", "random", dict(color="0.6", marker="x", ls="--")),
    ]),
    ("gpt2-large", "GPT-2-large", [
        ("own", "single-site", dict(color="C0", marker="o")),
        ("random", "random", dict(color="0.6", marker="x", ls="--")),
    ]),
]

# Fig 2: family -> (probe_results slug for instruct, for base). One color per
# family, base dashed, so the base-vs-instruct claim is visible without argmax.
TRANSFER = [
    ("Gemma-2-9b", "gemma-2-9b-it", "gemma-2-9b"),
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-7B"),
    ("Llama-3.1-8B", "meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-8B"),
    ("GPT-2-large", "gpt2-large", None),   # base only, no instruct release
]


def load_steer():
    frames = []
    for c in sorted(STEER.glob("*.csv")):
        if c.stem == "summary":
            continue
        try:
            d = pd.read_csv(c)
        except Exception:
            continue
        if "confidence" in d.columns and d["confidence"].notna().any():
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fig1():
    df = load_steer()
    df = df[df["mode"] == "add"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4), sharex=True, sharey=True)
    for ax, (model, title, series) in zip(axes.ravel(), PANELS):
        ax.axvspan(-BAND, BAND, color="0.9", zorder=0)
        sub = df[df["model"] == model]
        for exp, label, style in series:
            g = sub[sub["experiment"] == exp].groupby("alpha")["confidence"]
            if g.ngroups == 0:
                continue
            a = np.array(sorted(g.groups))
            mean = g.mean().reindex(a).values
            sem = g.sem().reindex(a).fillna(0).values
            ax.plot(a, mean, label=label, markersize=4, **style)
            ax.fill_between(a, mean - sem, mean + sem, color=style["color"],
                            alpha=0.15)
        ax.axhline(50, color="0.8", lw=0.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel(r"steering scale $\alpha$ (fraction of mean $\|h\|$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("judge confidence (0-100)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig1_steering.pdf")
    print(f"Saved -> {OUT/'fig1_steering.pdf'}")


def depth_curve(slug):
    """(depth, accuracy) envelope for one checkpoint, or None if not probed."""
    f = PROBE / slug / "probe_resid_transfer.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["acc"] = d[["p2f_test", "f2p_test"]].mean(axis=1)
    by_layer = d.groupby("layer")["acc"].max()   # envelope over cached positions
    layers = by_layer.index.values
    return layers / layers.max(), by_layer.values


def fig2():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (family, instruct, base) in enumerate(TRANSFER):
        color = f"C{i}"
        for slug, style in ((instruct, dict(ls="-", marker="o")),
                            (base, dict(ls="--", marker="^", alpha=0.7))):
            if slug is None:
                continue
            curve = depth_curve(slug)
            if curve is None:
                print(f"  skip {slug}: probe_resid_transfer.csv missing")
                continue
            suffix = " (base)" if slug == base and instruct != base else ""
            ax.plot(*curve, color=color, markersize=3,
                    label=family + suffix, **style)
    ax.axhline(0.5, color="0.7", lw=0.8, ls=":", label="chance")
    ax.set_xlabel("normalized layer depth")
    ax.set_ylabel("transfer accuracy (mean of p2f, f2p)")
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig2_transfer_accuracy.pdf")
    print(f"Saved -> {OUT/'fig2_transfer_accuracy.pdf'}")


if __name__ == "__main__":
    fig1()
    fig2()
