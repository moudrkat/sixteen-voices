#!/usr/bin/env python3
"""Validate SAE feature steering against random baselines.

Tests whether steering with SAE feature directions produces more
consistent, interpretable effects than random directions of the same
magnitude. This is the null hypothesis test for feature steering.

Usage:
    uv run python scripts/validate_sae_steering.py
"""

import json
from pathlib import Path

import torch
import numpy as np

from sixteen_voices.model import load_adapted_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.text import compute_perplexity


FEATURE_GROUPS = {
    "folk_voice": {"pos": [198, 33, 140], "neg": []},
    "event_narration": {"pos": [160, 144, 205], "neg": []},
    "speech_patterns": {"pos": [68, 113, 122], "neg": []},
}

TEST_AUTHORS = ["poet", "grimm", "minimalist", "carroll", "dark", "wilde", "poe"]
N_RANDOM = 20
SCALE = 8


def make_hook(vec):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + output[1:]
        return output + vec
    return hook_fn


def ppl_with_vec(model, tokenizer, text, vec):
    hook = model.transformer.ln_f.register_forward_hook(make_hook(vec))
    ppl = compute_perplexity(model, tokenizer, text, max_length=256)
    hook.remove()
    return ppl


def generate_with_vec(model, tokenizer, prompt, vec, seed=42):
    hook = model.transformer.ln_f.register_forward_hook(make_hook(vec))
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=60, temperature=0.7,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    hook.remove()
    return text


def main():
    print("Loading SAE...")
    sae_dir = Path("outputs/sae")
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)
    w = sae.decoder.weight.detach()

    tokenizer = load_tokenizer()

    results = {}

    for author in TEST_AUTHORS:
        print(f"\n=== {author} ===")
        model = load_adapted_model(f"outputs/authors/{author}/adapter")
        eval_text = open(f"data/eval/{author}.txt").read()[:2000]

        base_ppl = compute_perplexity(model, tokenizer, eval_text, max_length=256)
        print(f"  Baseline PPL: {base_ppl:.1f}")

        author_results = {"base_ppl": base_ppl, "groups": {}}

        for group_name, group in FEATURE_GROUPS.items():
            # Build feature vector
            feat_vec = torch.zeros(1024)
            for f in group["pos"]:
                feat_vec += SCALE * w[:, f]
            for f in group["neg"]:
                feat_vec -= SCALE * w[:, f]

            norm = feat_vec.norm().item()

            # Feature PPL
            feat_ppl = ppl_with_vec(model, tokenizer, eval_text, feat_vec)

            # Random PPLs (same norm)
            rand_ppls = []
            for i in range(N_RANDOM):
                torch.manual_seed(i * 13 + 7)
                rv = torch.randn(1024)
                rv = rv / rv.norm() * norm
                rand_ppls.append(ppl_with_vec(model, tokenizer, eval_text, rv))

            rand_mean = np.mean(rand_ppls)
            rand_std = np.std(rand_ppls)

            # Is feature PPL outside the random distribution?
            z_score = (feat_ppl - rand_mean) / (rand_std + 1e-8)

            print(f"  {group_name:>15s}: feature={feat_ppl:.0f}  "
                  f"random={rand_mean:.0f}±{rand_std:.0f}  "
                  f"z={z_score:+.1f}  norm={norm:.1f}")

            author_results["groups"][group_name] = {
                "feature_ppl": feat_ppl,
                "random_mean": rand_mean,
                "random_std": rand_std,
                "z_score": z_score,
                "norm": norm,
            }

        # Generation comparison (feature vs random)
        prompt = "Once upon a time"
        print(f"\n  Generation comparison (prompt: '{prompt}'):")

        # Baseline
        torch.manual_seed(42)
        ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=60, temperature=0.7,
                do_sample=True, top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"  BASELINE:    {baseline[:120]}")

        # Complexity feature
        feat_vec = torch.zeros(1024)
        for f in FEATURE_GROUPS["complexity"]["pos"]:
            feat_vec += SCALE * w[:, f]
        print(f"  COMPLEXITY:  {generate_with_vec(model, tokenizer, prompt, feat_vec)[:120]}")

        # 3 random directions of same norm
        for i in range(3):
            torch.manual_seed(i * 7 + 1)
            rv = torch.randn(1024)
            rv = rv / rv.norm() * feat_vec.norm().item()
            print(f"  RANDOM_{i}:    {generate_with_vec(model, tokenizer, prompt, rv)[:120]}")

        results[author] = author_results

    # Save
    out_path = Path("outputs/sae/steering_validation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()