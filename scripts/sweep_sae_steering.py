#!/usr/bin/env python3
"""Systematic sweep of SAE feature steering across authors and prompts.

Generates baseline + steered text for all combinations, saves everything,
and computes summary statistics. No cherry-picking — report the hit rate.

Usage:
    uv run python scripts/sweep_sae_steering.py
    uv run python scripts/sweep_sae_steering.py --output outputs/sae/steering_sweep.json
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices.model import load_adapted_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder


AUTHORS = [
    "poet", "minimalist", "dark", "grimm", "carroll",
    "wilde", "dialogue", "poe", "baum", "homer",
    "cozy", "barrie", "andersen",
]

PROMPTS = [
    "Once upon a time",
    "The door opened slowly",
    "It was a dark and stormy",
]

SEEDS = [42, 123, 456]

SCALE = 8


def build_vecs(w):
    return {
        "folk_voice": SCALE * (w[:, 198] + w[:, 33] + w[:, 140]),
        "event_narration": SCALE * (w[:, 160] + w[:, 144] + w[:, 205]),
        "speech_patterns": SCALE * (w[:, 68] + w[:, 113] + w[:, 122]),
    }


def gen(model, tokenizer, prompt, vec=None, seed=42, max_new=70):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + output[1:]
        return output + vec
    hook = model.transformer.ln_f.register_forward_hook(hook_fn) if vec is not None else None
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=0.7,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    if hook:
        hook.remove()
    return text


def texts_differ(a, b, threshold=0.3):
    """Check if two texts differ meaningfully (not just a word or two)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap < (1 - threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/sae/steering_sweep.json")
    args = parser.parse_args()

    # Load SAE
    sae_dir = Path("outputs/sae")
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)
    w = sae.decoder.weight.detach()

    tokenizer = load_tokenizer()
    vecs = build_vecs(w)

    results = []
    total = 0
    changed = 0
    coherent_and_changed = 0

    for author in AUTHORS:
        print(f"  {author}...", flush=True)
        model = load_adapted_model(f"outputs/authors/{author}/adapter")

        for prompt in PROMPTS:
            for seed in SEEDS:
                baseline = gen(model, tokenizer, prompt, seed=seed)

                for knob_name, vec in vecs.items():
                    steered = gen(model, tokenizer, prompt, vec, seed=seed)
                    total += 1

                    differs = texts_differ(baseline, steered)
                    if differs:
                        changed += 1

                    # Rough coherence: no single token repeated 5+ times
                    words = steered.split()
                    degenerate = any(
                        words[i] == words[i+1] == words[i+2] == words[i+3]
                        for i in range(len(words) - 3)
                    ) if len(words) > 3 else False

                    if differs and not degenerate:
                        coherent_and_changed += 1

                    entry = {
                        "author": author,
                        "prompt": prompt,
                        "seed": seed,
                        "knob": knob_name,
                        "baseline": baseline,
                        "steered": steered,
                        "changed": differs,
                        "degenerate": degenerate,
                    }
                    results.append(entry)

    # Summary
    print(f"\n=== Steering Sweep Summary ===")
    print(f"Total combos: {total}")
    print(f"Changed (>30% word diff): {changed} ({changed/total:.0%})")
    print(f"Changed + coherent: {coherent_and_changed} ({coherent_and_changed/total:.0%})")
    print(f"Degenerate: {changed - coherent_and_changed}")

    # Per-knob stats
    print(f"\nPer knob:")
    for knob_name in vecs:
        knob_results = [r for r in results if r["knob"] == knob_name]
        n = len(knob_results)
        n_changed = sum(1 for r in knob_results if r["changed"])
        n_good = sum(1 for r in knob_results if r["changed"] and not r["degenerate"])
        n_degen = sum(1 for r in knob_results if r["degenerate"])
        print(f"  {knob_name:>15s}: {n_good}/{n} good ({n_good/n:.0%}), "
              f"{n_degen} degenerate")

    # Per-author stats
    print(f"\nPer author:")
    for author in AUTHORS:
        auth_results = [r for r in results if r["author"] == author]
        n = len(auth_results)
        n_good = sum(1 for r in auth_results if r["changed"] and not r["degenerate"])
        n_degen = sum(1 for r in auth_results if r["degenerate"])
        print(f"  {author:>12s}: {n_good}/{n} good ({n_good/n:.0%}), "
              f"{n_degen} degenerate")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": total,
        "changed": changed,
        "coherent_and_changed": coherent_and_changed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
