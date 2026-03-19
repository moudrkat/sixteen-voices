#!/usr/bin/env python3
"""Steer Carroll's H11 and H14 separately, generate text at 0x-2x."""

import json
from pathlib import Path

import torch

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_out
from sixteen_voices.steering import generate, steered_perplexity

ADAPTERS_DIR = Path("outputs/authors")
PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 80
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
HEADS = [11, 14]

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)


def main():
    tokenizer = load_tokenizer()
    model = load_adapted_model(str(ADAPTERS_DIR / "carroll" / "adapter"))
    attn_out = get_attn_out(model)

    results = {}
    for head in HEADS:
        print(f"\n{'='*60}")
        print(f"  Carroll — H{head}")
        print(f"{'='*60}")
        head_results = {}
        for scale in SCALES:
            scales = {head: scale}
            text = generate(model, tokenizer, PROMPT, head_scales=scales,
                           attn_out=attn_out, max_new_tokens=MAX_NEW, seed=SEED)
            ppl = steered_perplexity(model, tokenizer, EVAL_TEXT,
                                     head_scales=scales, attn_out=attn_out)
            label = f"{scale:.1f}"
            head_results[label] = {"text": text, "ppl": ppl}
            tag = ""
            if scale == 0.0:
                tag = " (killed)"
            elif scale == 1.0:
                tag = " (normal)"
            elif scale == 2.0:
                tag = " (amplified)"
            print(f"  {scale:.1f}×{tag}  PPL={ppl:.1f}")
            print(f"    {text[:120]}...")

        results[f"H{head}"] = head_results

    out = {
        "author": "carroll",
        "prompt": PROMPT,
        "seed": SEED,
        "heads": results,
    }
    with open("outputs/carroll_steering.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved outputs/carroll_steering.json")


if __name__ == "__main__":
    main()