#!/usr/bin/env python3
"""Poe steering with multiple prompts — find the most dramatic effect.

Generates text at 0×, 0.5×, 1×, 1.5×, 2× for Poe's H14
across many prompts including Raven-inspired ones.

All outputs are reproducible (seed=42).

Usage:
    uv run python scripts/poe_steering_prompts.py
"""

import json
from pathlib import Path

import torch

from sixteen_voices import NUM_HEADS, load_adapted_model, load_tokenizer
from sixteen_voices.model import get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/poe_steering_prompts.json")

PROMPTS = [
    "It was a dark and stormy",
    "Once upon a time",
    "In the dark of night",
    "The little girl walked into the forest",
    "There was a king who had",
    "While I nodded nearly napping",
    "Once upon a midnight dreary",
    "Deep into that darkness peering",
    "The raven sat upon the",
    "In the chamber of the",
    "The shadows crept along the",
    "There came a tapping at",
    "The wind howled through the",
    "Beneath the pale moonlight",
    "The old house stood alone",
]

HEAD = 14
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
MAX_NEW = 100
TEMP = 0.7
SEED = 42


def generate_steered(model, tokenizer, prompt, head, scale, seed=SEED):
    torch.manual_seed(seed)
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_hook({head: scale}))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, temperature=TEMP,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    hook.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def main():
    tokenizer = load_tokenizer()
    model = load_adapted_model(str(ADAPTERS_DIR / "poe" / "adapter"))

    results = {
        "author": "poe",
        "head": HEAD,
        "seed": SEED,
        "max_new_tokens": MAX_NEW,
        "temperature": TEMP,
        "prompts": {},
    }

    for prompt in PROMPTS:
        print(f'\n=== "{prompt}" ===')
        prompt_data = {}
        for scale in SCALES:
            text = generate_steered(model, tokenizer, prompt, HEAD, scale)
            prompt_data[str(scale)] = text
            tag = {0.0: "KILLED", 0.5: "half", 1.0: "NORMAL", 1.5: "boost", 2.0: "AMPLIFIED"}[scale]
            print(f"  {scale}× [{tag:>8s}]: {text[:100]}...")
        results["prompts"][prompt] = prompt_data

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()