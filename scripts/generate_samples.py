#!/usr/bin/env python3
"""Generate text samples for all authors with a fixed prompt and seed.

Saves exact outputs to JSON for reproducibility and article use.

Usage:
    python scripts/generate_samples.py
    python scripts/generate_samples.py --prompt "The old house on the hill" --seed 123
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoModelForCausalLM

from sixteen_voices import load_adapted_model, load_tokenizer, generate

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/samples.json")

DEFAULT_PROMPT = "It was a dark and stormy"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 60


def main():
    parser = argparse.ArgumentParser(description="Generate samples for all authors")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    results = {
        "prompt": args.prompt,
        "seed": args.seed,
        "max_new_tokens": args.max_tokens,
        "samples": {},
    }

    # Base model
    print("Generating: base")
    base = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1Layer-21M")
    text = generate(base, tokenizer, args.prompt, seed=args.seed,
                    max_new_tokens=args.max_tokens)
    results["samples"]["base"] = text
    print(f"  {text[:100]}")
    del base

    # All authors
    for author in sorted(os.listdir(ADAPTERS_DIR)):
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if not adapter_path.exists():
            continue
        print(f"Generating: {author}")
        model = load_adapted_model(str(adapter_path))
        text = generate(model, tokenizer, args.prompt, seed=args.seed,
                        max_new_tokens=args.max_tokens)
        results["samples"][author] = text
        print(f"  {text[:100]}")
        del model

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} ({len(results['samples'])} samples)")


if __name__ == "__main__":
    main()