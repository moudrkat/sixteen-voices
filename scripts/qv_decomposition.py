#!/usr/bin/env python3
"""Q vs V decomposition: which LoRA component drives vocabulary redirection?

Three analyses:
1. Distance-redirection correlation (base PPL vs vocabulary overlap)
2. Q-only vs V-only functional decomposition
3. Attention pattern similarity before/after adaptation

Usage:
    python scripts/qv_decomposition.py
    python scripts/qv_decomposition.py --analysis 2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy import stats

from sixteen_voices import (
    HEAD_DIM,
    MODEL_NAME,
    NUM_HEADS,
    get_attn_out,
    load_adapted_model,
    load_base_model,
    load_tokenizer,
)

ADAPTERS_DIR = Path("outputs/authors")
KNOCKOUT_FILE = Path("outputs/knockout_all_heads.json")
TOP_K = 50

AUTHORS_6 = ["shelley", "poe", "homer", "grimm", "carroll", "wilde"]

TEXTS = [
    "Once upon a time there was a little",
    "The dark forest was full of",
    "The princess smiled and said",
]


def make_knockout_hook(head_idx):
    def hook_fn(module, args):
        h = args[0].clone()
        s = head_idx * HEAD_DIM
        h[:, :, s : s + HEAD_DIM] = 0
        return (h,) + args[1:]
    return hook_fn


def get_promoted(model, tokenizer, text, head_idx):
    """Words promoted by a head (in full but not in knockout)."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    full = {tokenizer.decode(i.item()).strip()
            for i in torch.topk(torch.softmax(logits, -1), TOP_K).indices}

    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_knockout_hook(head_idx))
    with torch.no_grad():
        logits_ko = model(**inputs).logits[0, -1]
    hook.remove()
    ko = {tokenizer.decode(i.item()).strip()
          for i in torch.topk(torch.softmax(logits_ko, -1), TOP_K).indices}

    return full - ko


def analysis_qv_weight_and_function():
    """Weight-space and functional Q vs V decomposition."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = load_tokenizer()

    # Weight-space norms
    print("\n  Weight-space decomposition (Frobenius norm)...")
    q_norms = np.zeros((len(AUTHORS_6), NUM_HEADS))
    v_norms = np.zeros((len(AUTHORS_6), NUM_HEADS))

    for i, author in enumerate(AUTHORS_6):
        path = ADAPTERS_DIR / author / "adapter" / "adapter_model.safetensors"
        weights = load_file(str(path))
        for proj_idx, proj in enumerate(["q_proj", "v_proj"]):
            A = weights[f"base_model.model.transformer.h.0.attn.attention.{proj}.lora_A.weight"]
            B = weights[f"base_model.model.transformer.h.0.attn.attention.{proj}.lora_B.weight"]
            delta = (B @ A).numpy()
            for h in range(NUM_HEADS):
                s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
                norm = np.linalg.norm(delta[s:e])
                if proj_idx == 0:
                    q_norms[i, h] = norm
                else:
                    v_norms[i, h] = norm

    q_fraction = q_norms / (q_norms + v_norms + 1e-10)

    print("  Q fraction per head (mean across 6 authors):")
    for h in range(NUM_HEADS):
        print(f"    H{h:2d}: Q={q_fraction[:, h].mean():.1%}  V={1 - q_fraction[:, h].mean():.1%}")

    # Functional decomposition: Q-only vs V-only
    print("\n  Functional decomposition (vocabulary overlap)...")
    q_overlap = np.zeros((len(AUTHORS_6), NUM_HEADS))
    v_overlap = np.zeros((len(AUTHORS_6), NUM_HEADS))

    for i, author in enumerate(AUTHORS_6):
        print(f"    {author}...")
        for mode in ["full", "q_only", "v_only"]:
            base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model = PeftModel.from_pretrained(base, str(ADAPTERS_DIR / author / "adapter"))
            model.eval()

            if mode == "q_only":
                model.base_model.model.transformer.h[0].attn.attention.v_proj.lora_A[
                    "default"
                ].weight.data.zero_()
            elif mode == "v_only":
                model.base_model.model.transformer.h[0].attn.attention.q_proj.lora_A[
                    "default"
                ].weight.data.zero_()

            text = TEXTS[0]
            for h in range(NUM_HEADS):
                promoted = get_promoted(model, tokenizer, text, h)
                if mode == "full":
                    if not hasattr(analysis_qv_weight_and_function, '_full_cache'):
                        analysis_qv_weight_and_function._full_cache = {}
                    analysis_qv_weight_and_function._full_cache[(author, h)] = promoted
                elif mode == "q_only":
                    full_set = analysis_qv_weight_and_function._full_cache.get((author, h), set())
                    if full_set:
                        q_overlap[i, h] = len(full_set & promoted) / len(full_set)
                elif mode == "v_only":
                    full_set = analysis_qv_weight_and_function._full_cache.get((author, h), set())
                    if full_set:
                        v_overlap[i, h] = len(full_set & promoted) / len(full_set)

            del model, base

    # Summary
    mean_q = q_overlap.mean(axis=0)
    mean_v = v_overlap.mean(axis=0)
    q_wins = (mean_q > mean_v).sum()
    v_wins = NUM_HEADS - q_wins
    winner = "Q (attention patterns)" if q_wins > v_wins else "V (output vocabulary)"

    print(f"\n  Functional overlap with full LoRA per head:")
    for h in range(NUM_HEADS):
        marker = "Q" if mean_q[h] > mean_v[h] else "V"
        print(f"    H{h:2d}: Q-only={mean_q[h]:.1%}  V-only={mean_v[h]:.1%}  [{marker} wins]")

    print(f"\n  Q wins {q_wins}/16 heads, V wins {v_wins}/16")
    print(f"  Redirection primarily driven by: {winner}")

    # Save
    result = {
        "authors": AUTHORS_6,
        "q_fraction_weights": q_fraction.tolist(),
        "q_overlap_functional": q_overlap.tolist(),
        "v_overlap_functional": v_overlap.tolist(),
        "q_wins": int(q_wins),
        "v_wins": int(v_wins),
    }
    out = Path("outputs/qv_decomposition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Q vs V decomposition analysis")
    args = parser.parse_args()
    analysis_qv_weight_and_function()


if __name__ == "__main__":
    main()
