#!/usr/bin/env python3
"""Generate Poe steering text samples for both H11 and H14.

Saves to outputs/poe_steering_both.json in the same format as
carroll_steering.json.
"""

import json
from pathlib import Path

import torch

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_out
from sixteen_voices.steering import generate, steered_perplexity

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/poe_steering_both.json")

PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 80
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)


def main():
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_adapted_model(str(ADAPTERS_DIR / "poe" / "adapter"))
    attn_out = get_attn_out(model)

    results = {
        "author": "poe",
        "prompt": PROMPT,
        "seed": SEED,
        "heads": {},
    }

    for head_idx in [11, 14]:
        head_key = f"H{head_idx}"
        print(f"\nSteering {head_key}...")
        results["heads"][head_key] = {}

        for scale in SCALES:
            print(f"  scale={scale}...")
            head_scales = {head_idx: scale}
            text = generate(model, tokenizer, PROMPT,
                            head_scales=head_scales,
                            attn_out=attn_out,
                            seed=SEED, max_new_tokens=MAX_NEW)
            ppl = steered_perplexity(model, tokenizer, EVAL_TEXT,
                                     head_scales=head_scales,
                                     attn_out=attn_out)

            results["heads"][head_key][str(scale)] = {
                "text": text,
                "ppl": round(ppl, 1),
            }
            print(f"    PPL={ppl:.1f}: {text[:100]}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()