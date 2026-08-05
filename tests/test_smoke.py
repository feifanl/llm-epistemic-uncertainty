"""
One CPU smoke test over the whole pipeline, on gpt2 (124M).

Not a correctness test of the science -- it guards the three places where a
silent wrong answer is easy: the last-token gather in run_batch (right-padding
+ torch.gather, the most fragile indexing in the repo), the meta/tensor row
alignment that every downstream join assumes, and the Steerer's add/ablate
arithmetic including the tuple-vs-tensor hook output.

Downloads gpt2 (~550MB) on first run. ~1-2 min on CPU.

Usage:
    venv/Scripts/python.exe -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import cache_activations  # noqa: E402
import linear_probes  # noqa: E402
from model_adapter import get_layers  # noqa: E402
from steer import Steerer  # noqa: E402

MODEL = "gpt2"
N_PROMPTS = 8
N_POS = 3
D_MODEL = 768


@pytest.fixture(scope="module")
def cached(tmp_path_factory):
    """Cache 8 prompts x 3 positions, resid only, through the real CLI."""
    out = tmp_path_factory.mktemp("acts") / "gpt2"
    argv = [
        "cache_activations.py",
        "--model", MODEL,
        "--dataset", str(ROOT / "dataset.csv"),
        "--out", str(out),
        "--limit", str(N_PROMPTS),
        "--n-pos", str(N_POS),
        "--points", "resid",
        "--dtype", "bf16",
        "--batch-size", "4",
    ]
    old, sys.argv = sys.argv, argv
    try:
        cache_activations.main()
    finally:
        sys.argv = old
    return out


def test_cache_shapes(cached):
    t = torch.load(cached / "L0_resid.pt", map_location="cpu")
    assert tuple(t.shape) == (N_PROMPTS, N_POS, D_MODEL)


def test_meta_aligns_with_tensor_rows(cached):
    """Row i of meta must be row i of every tensor -- every downstream join
    (probe labels, per-prompt dumps, dd traces) assumes it and none check it."""
    meta = pd.read_parquet(cached / "meta.parquet")
    assert len(meta) == N_PROMPTS
    assert list(meta["row_idx"]) == list(range(N_PROMPTS))
    assert set(meta["label"]) <= {0, 1}


def test_gather_picks_the_final_tokens(cached):
    """The gather must land on the last n_pos real tokens, not on padding.

    Checked against an unpadded single-prompt forward: the cached vectors for
    prompt 0 have to match a batch of one, where padding can't interfere.
    """
    meta = pd.read_parquet(cached / "meta.parquet")
    prompt = cache_activations.PROMPT_TEMPLATE.format(claim=meta.loc[0, "text"])

    model, tok = cache_activations.load_model(MODEL, "bf16")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    buf, handles, _ = cache_activations.register_hooks(model)
    device = next(model.parameters()).device
    single = cache_activations.run_batch(model, tok, [prompt], buf, N_POS, device)
    for h in handles:
        h.remove()

    batched = torch.load(cached / "L0_resid.pt", map_location="cpu")[0]
    torch.testing.assert_close(single[(0, "resid")][0], batched,
                               rtol=1e-2, atol=1e-2)


def test_transfer_masks_partition():
    """p2f/f2p train and test masks must be complementary and cover everything."""
    df = pd.read_csv(ROOT / "dataset.csv")
    _, meta = cache_activations.build_prompts(df)
    meta = pd.DataFrame(meta)
    directions, y = linear_probes.make_masks(meta, "transfer")

    assert {name for name, _, _ in directions} == {"p2f", "f2p"}
    assert set(y) == {0, 1}
    for _, tr, te in directions:
        assert (tr & te).sum() == 0
        assert (tr | te).all()
    (_, p_tr, _), (_, f_tr, _) = directions
    assert (p_tr == ~f_tr).all()      # the two directions are each other's flip


def test_probe_returns_valid_accuracies(cached):
    """fit_eval on real cached acts: accuracies in [0,1], score is a probability."""
    X = torch.load(cached / "L0_resid.pt", map_location="cpu")[:, -1, :].float().numpy()
    y = np.array([0, 1] * (N_PROMPTS // 2))
    tr = np.zeros(N_PROMPTS, bool)
    tr[: N_PROMPTS // 2] = True

    tr_acc, te_acc, score = linear_probes.fit_eval(X, y, tr, ~tr, epochs=20)
    assert 0.0 <= tr_acc <= 1.0 and 0.0 <= te_acc <= 1.0
    assert score.shape == (N_PROMPTS // 2,)
    assert ((score >= 0) & (score <= 1)).all()
    assert 0.0 <= linear_probes.roc_auc(y[~tr], score) <= 1.0


@pytest.fixture(scope="module")
def fp32_model():
    """fp32, not bf16: the hook assertions below are exact arithmetic and bf16's
    ~3 significant digits can't resolve a +2.0 shift on a residual of ~100."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, AutoTokenizer.from_pretrained(MODEL)


def _observe(model, layer_idx, ids):
    """Read a layer's output residual with a hook registered AFTER the Steerer.

    Do NOT use output_hidden_states here: transformers captures those with its
    own hooks, registered before ours, so hidden_states[layer+1] is the
    PRE-steer value and the intervention only shows up one layer downstream.
    Hook order is registration order, so an observer added after the Steerer
    sees what the Steerer actually returned.
    """
    seen = {}

    def hook(_m, _i, out):
        seen["hs"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    h = get_layers(model)[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return seen["hs"]


def test_steerer_add_lands_alpha_v_on_the_residual(fp32_model):
    model, tok = fp32_model
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    v = np.zeros(D_MODEL, dtype=np.float32)
    v[0] = 1.0

    s = Steerer(model, 6, v, ids.device, torch.float32, mode="add")
    try:
        s.alpha = 0.0                       # alpha=0 must be an exact no-op
        base = _observe(model, 6, ids)
        s.alpha = 2.0
        steered = _observe(model, 6, ids)
    finally:
        s.remove()

    delta = (steered - base)[0]
    assert torch.allclose(delta[:, 0], torch.full((ids.shape[1],), 2.0), atol=1e-3)
    assert torch.allclose(delta[:, 1:], torch.zeros_like(delta[:, 1:]), atol=1e-4)


def test_steerer_mean_ablate_pins_projection_to_mbar(fp32_model):
    model, tok = fp32_model
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    v = np.zeros(D_MODEL, dtype=np.float32)
    v[0] = 1.0
    mbar = 3.0

    s = Steerer(model, 6, v, ids.device, torch.float32,
                mode="ablate", ablate_kind="mean", mbar=mbar)
    try:
        s.alpha = 1.0
        steered = _observe(model, 6, ids)
    finally:
        s.remove()

    proj = steered[0] @ torch.tensor(v)       # v is already unit here
    assert torch.allclose(proj, torch.full_like(proj, mbar), atol=1e-3)
