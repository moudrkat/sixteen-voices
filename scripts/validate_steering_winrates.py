#!/usr/bin/env python3
"""Quantitative steering validation: measure text properties across seeds.

For each feature group, steer the model and count how often the text
property moves in the expected direction.

Usage:
    uv run python scripts/validate_steering_winrates.py
    uv run python scripts/validate_steering_winrates.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
import re
from pathlib import Path

import torch

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


TESTS = [
    {
        "name": "simplicity (f665)",
        "features": {665: 10.0},
        "author": None,
        "metric": "avg_sentence_len",
        "direction": "lower",
    },
    {
        "name": "dialogue (f1777+f689)",
        "features": {1777: 10.0, 689: 10.0},
        "author": None,
        "metric": "quote_count",
        "direction": "higher",
    },
    {
        "name": "poe + simplicity",
        "features": {665: 15.0},
        "author": "poe",
        "metric": "avg_sentence_len",
        "direction": "lower",
    },
    {
        "name": "grimm + dialogue",
        "features": {1777: 5.0, 689: 5.0},
        "author": "grimm",
        "metric": "quote_count",
        "direction": "higher",
    },
    {
        "name": "complexity (f883+f993+f60)",
        "features": {883: 12.0, 993: 12.0, 60: 12.0},
        "author": "minimalist",
        "metric": "avg_sentence_len",
        "direction": "higher",
    },
]


def avg_sentence_len(text):
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sents:
        return 0
    return sum(len(s.split()) for s in sents) / len(sents)


def quote_count(text):
    return text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d")


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative steering validation")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "steering_winrates.json")

    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)
    w = sae.decoder.weight.detach()

    tokenizer = load_tokenizer()
    model_cache = {}
    results = []

    for test in TESTS:
        author = test["author"]
        cache_key = author or "__base__"
        if cache_key not in model_cache:
            if author:
                model_cache[cache_key] = load_adapted_model(
                    f"outputs/authors/{author}/adapter")
            else:
                model_cache[cache_key] = load_base_model()
        model = model_cache[cache_key]

        vec = sum(scale * w[:, fi] for fi, scale in test["features"].items())
        measure = avg_sentence_len if test["metric"] == "avg_sentence_len" else quote_count

        baseline_vals = []
        steered_vals = []

        for seed in range(args.n_seeds):
            text_b = generate(model, tokenizer, args.prompt,
                              max_new_tokens=80, seed=seed)
            baseline_vals.append(measure(text_b))

            hook = model.transformer.ln_f.register_forward_hook(
                lambda mod, inp, out, v=vec: (
                    (out[0] + v,) + out[1:] if isinstance(out, tuple)
                    else out + v))
            text_s = generate(model, tokenizer, args.prompt,
                              max_new_tokens=80, seed=seed)
            hook.remove()
            steered_vals.append(measure(text_s))

        b_mean = sum(baseline_vals) / len(baseline_vals)
        s_mean = sum(steered_vals) / len(steered_vals)

        if test["direction"] == "lower":
            wins = sum(1 for b, s in zip(baseline_vals, steered_vals) if s < b)
        else:
            wins = sum(1 for b, s in zip(baseline_vals, steered_vals) if s > b)

        print(f"{test['name']}:")
        print(f"  {test['metric']}: baseline={b_mean:.1f}  "
              f"steered={s_mean:.1f}  ({test['direction']} is better)")
        print(f"  wins: {wins}/{args.n_seeds} ({100*wins/args.n_seeds:.0f}%)")
        print()

        results.append({
            "name": test["name"],
            "metric": test["metric"],
            "direction": test["direction"],
            "baseline_mean": b_mean,
            "steered_mean": s_mean,
            "wins": wins,
            "n_seeds": args.n_seeds,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
