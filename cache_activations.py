"""
Cache residual stream + attn + MLP activations per layer, last K token positions,
for each (claim, paraphrase) in dataset.csv.

Output layout:
    activations/{model_slug}/
        L{layer}_resid.pt      # tensor [N, K, d_model]
        L{layer}_attn.pt
        L{layer}_mlp.pt
        meta.parquet           # N rows, aligned with tensors' row dim
        config.json            # run config snapshot
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = 'Q: Is the following claim true? "{claim}"\nA:'
PARAPHRASE_COLS = ["claim", "para_1", "para_2", "para_3", "para_4", "para_5"]


# ---------- helpers ----------

def slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_model(name: str, dtype: str):
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",  # hooks on attn output need eager
    )
    model.eval()
    return model, tok


def register_hooks(model):
    """
    Hooks on each decoder layer:
      - resid: layer output (post-residual)
      - attn:  self_attn output
      - mlp:   mlp output
    Returns (buffer dict, handle list).
    Buffer shape per entry: [batch, seq, d_model]  (overwritten every forward).
    """
    layers = model.model.layers
    n = len(layers)
    buf: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx, point):
        def hook(_module, _inp, out):
            # layer output is tuple; attn output also tuple; mlp is tensor
            t = out[0] if isinstance(out, tuple) else out
            buf[(layer_idx, point)] = t.detach()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i, "resid")))
        handles.append(layer.self_attn.register_forward_hook(make_hook(i, "attn")))
        handles.append(layer.mlp.register_forward_hook(make_hook(i, "mlp")))

    return buf, handles, n


def build_prompts(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    prompts, meta = [], []
    for row in df.itertuples():
        for p_idx, col in enumerate(PARAPHRASE_COLS):
            text = getattr(row, col)
            if pd.isna(text):
                continue
            prompts.append(PROMPT_TEMPLATE.format(claim=text))
            meta.append({
                "row_idx": len(prompts) - 1,
                "item_idx": row.idx,
                "cell": row.cell,
                "label": row.label,
                "paraphrase_id": p_idx,
                "paraphrase_col": col,
                "text": text,
            })
    return prompts, meta


@torch.no_grad()
def run_batch(model, tok, prompts: list[str], buf, n_layers, n_pos, device):
    """
    Forward pass on batch. Return dict[(layer, point)] -> tensor[B, n_pos, d_model].
    Slices last n_pos tokens of each sequence (right-padded → use attention mask).
    """
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    model(**enc)

    # actual length per sample = sum of attention mask
    lengths = enc.attention_mask.sum(dim=1)  # [B]

    out = {}
    for (layer, point), t in buf.items():
        # t: [B, S, d]. Gather last n_pos positions per sample.
        B, S, D = t.shape
        # build index: for sample b, positions [len_b - n_pos, ..., len_b - 1]
        idx = torch.stack([
            torch.arange(L - n_pos, L, device=t.device) for L in lengths.tolist()
        ])  # [B, n_pos]
        gathered = torch.gather(
            t, 1, idx.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, n_pos, D]
        out[(layer, point)] = gathered.cpu().float()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    p.add_argument("--out", type=Path, required=True,
                   help="Output dir (e.g. activations/gemma-2-9b-it/)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--n-pos", type=int, default=5,
                   help="Cache last N token positions")
    p.add_argument("--points", nargs="+",
                   default=["resid", "attn", "mlp"],
                   choices=["resid", "attn", "mlp"])
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: cap number of prompts")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # log config
    (args.out / "config.json").write_text(json.dumps({
        **vars(args),
        "out": str(args.out),
        "dataset": str(args.dataset),
        "prompt_template": PROMPT_TEMPLATE,
    }, indent=2))

    print(f"Loading {args.model} ({args.dtype})...")
    model, tok = load_model(args.model, args.dtype)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = next(model.parameters()).device

    buf, handles, n_layers = register_hooks(model)
    print(f"Hooked {n_layers} layers × {len(args.points)} points")

    df = pd.read_csv(args.dataset)
    prompts, meta = build_prompts(df)
    if args.limit:
        prompts, meta = prompts[:args.limit], meta[:args.limit]
    print(f"{len(prompts)} prompts to process")

    # preallocate accumulators per (layer, point)
    accum: dict[tuple[int, str], list[torch.Tensor]] = {
        (l, pt): [] for l in range(n_layers) for pt in args.points
    }

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i:i + args.batch_size]
        out = run_batch(model, tok, batch, buf, n_layers, args.n_pos, device)
        for (layer, point), tensor in out.items():
            if point in args.points:
                accum[(layer, point)].append(tensor)

    # stack + save per (layer, point)
    print("Saving tensors...")
    for (layer, point), chunks in accum.items():
        stacked = torch.cat(chunks, dim=0)  # [N, n_pos, d_model]
        torch.save(stacked, args.out / f"L{layer}_{point}.pt")

    # save meta
    pd.DataFrame(meta).to_parquet(args.out / "meta.parquet")

    # cleanup hooks
    for h in handles:
        h.remove()

    print(f"Done. Wrote {len(accum)} tensor files + meta.parquet to {args.out}")


if __name__ == "__main__":
    main()