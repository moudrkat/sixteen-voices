#!/usr/bin/env python3
"""Generate steering samples using the canonical sampler from generate_samples.py.

Uses sixteen_voices.generate() with the SAME settings as scripts/generate_samples.py
(prompt="It was a dark and stormy", seed=42, temperature=0.8, top_k=50,
max_new_tokens=60). At scale=1.0 this produces the identical text that appears
in outputs/samples.json for the given author — so the "Shelley 1×" in Q2 slide
matches the "Shelley" sample shown elsewhere byte-for-byte.

Output: outputs/steering_canonical_samples.json

Usage:
    uv run python scripts/gen_steering_canonical.py
"""

import json
from pathlib import Path

from sixteen_voices import load_adapted_model, load_tokenizer, generate
from sixteen_voices.model import get_attn_out

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/steering_canonical_samples.json")

# Match generate_samples.py defaults — DO NOT CHANGE without also updating that script.
PROMPT = "It was a dark and stormy"
SEED = 42
TEMPERATURE = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 60
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]

# Authors and the head to steer. Explicit — we want H11 for Shelley (style
# carrier) even though H14 has larger |recovery|.
AUTHORS = [
    ("shelley", 11),
]


def main():
    tokenizer = load_tokenizer()

    results = {
        "prompt": PROMPT,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "scales": SCALES,
        "authors": {},
    }

    for author, head in AUTHORS:
        print(f"Generating for {author}, H{head}...")
        adapter_path = ADAPTERS_DIR / author / "adapter"
        model = load_adapted_model(adapter_path)
        attn_out = get_attn_out(model)

        samples = {}
        for scale in SCALES:
            text = generate(
                model, tokenizer, PROMPT,
                head_scales={head: scale},
                attn_out=attn_out,
                seed=SEED,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )
            samples[str(scale)] = text
            print(f"  {scale}x: {text[:90]}...")

        results["authors"][author] = {
            "head": head,
            "samples": samples,
        }
        del model

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
