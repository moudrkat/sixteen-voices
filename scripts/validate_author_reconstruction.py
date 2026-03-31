#!/usr/bin/env python3
"""Validate SAE feature steering by checking if steered text looks like an author.

For each author:
1. Generate text with the base model (baseline)
2. Generate text with the base model + feature steering (steered)
3. Generate text with the base model + random direction (control)
4. Measure perplexity of all three under the ADAPTED model
5. If adapted-model PPL is lower for steered text → features push toward that author

Usage:
    uv run python scripts/validate_author_reconstruction.py
    uv run python scripts/validate_author_reconstruction.py --authors poe grimm carroll minimalist
    uv run python scripts/validate_author_reconstruction.py --scale 8.0
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


# Structural features that steer well
STRUCTURAL_FEATURES = {
    "simplicity": [665],
    "complexity": [883, 993, 60],
    "dialogue": [1777, 689],
    "questions": [329, 1385],
    "verse": [344],
    "first_person": [1779, 627],
}

PROMPTS = [
    "Once upon a time",
    "It was a dark and stormy",
    "The door opened slowly",
    "The little girl walked into the forest",
    "The old man sat down and",
]


def measure_ppl(model, tokenizer, texts):
    """Measure mean perplexity of a list of texts under a model."""
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt")
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        total_loss += out.loss.item() * ids.numel()
        total_tokens += ids.numel()

    if total_tokens == 0:
        return float("inf")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def build_author_steering_vec(sae, author_features, global_mean, global_std,
                               scale=5.0):
    """Build steering vector from author's elevated structural features."""
    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])
    used_features = []

    for group_name, feat_indices in STRUCTURAL_FEATURES.items():
        for fi in feat_indices:
            z = (author_features[fi] - global_mean[fi]) / (global_std[fi] + 1e-8)
            if z > 1.0:
                vec += scale * z * w[:, fi]
                used_features.append((group_name, fi, z))

    return vec, used_features


def generate_texts(model, tokenizer, prompts, seeds, max_tokens=80,
                   steering_vec=None):
    """Generate texts for all prompt×seed combinations, optionally steered."""
    hook = None
    if steering_vec is not None and steering_vec.abs().max() > 0.01:
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (output[0] + steering_vec,) + output[1:]
            return output + steering_vec
        hook = model.transformer.ln_f.register_forward_hook(hook_fn)

    texts = []
    for prompt in prompts:
        for seed in seeds:
            text = generate(model, tokenizer, prompt,
                            max_new_tokens=max_tokens, seed=seed)
            texts.append(text)

    if hook is not None:
        hook.remove()

    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Validate author reconstruction: does steered text look "
                    "more like the author to the adapted model?")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--authors", nargs="*", default=None)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 123, 456])
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "author_reconstruction.json")

    # Load SAE and author-feature matrix
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    afm = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    all_authors = afm["authors"]
    matrix = afm["matrix"].numpy()
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0)

    # Load base model
    print("Loading base model and tokenizer...")
    tokenizer = load_tokenizer()
    base_model = load_base_model()

    adapters_dir = Path("outputs/authors")

    # Select authors
    if args.authors:
        authors = args.authors
    else:
        authors = sorted(
            d.name for d in adapters_dir.iterdir()
            if (d / "adapter" / "adapter_model.safetensors").exists()
            and d.name in all_authors
        )

    print(f"\nTesting {len(authors)} authors, scale={args.scale}, "
          f"{len(PROMPTS)} prompts × {len(args.seeds)} seeds\n")

    results = []
    improved = 0
    beat_random = 0
    total = 0

    for i, author in enumerate(authors):
        adapter_path = adapters_dir / author / "adapter"
        if not adapter_path.exists():
            continue

        author_idx = all_authors.index(author)
        author_features = matrix[author_idx]

        # Build steering vector
        vec, used = build_author_steering_vec(
            sae, author_features, global_mean, global_std,
            scale=args.scale)

        if not used:
            print(f"  [{i+1}/{len(authors)}] {author:>20s}: "
                  f"no elevated structural features, skipping")
            continue

        # Generate baseline, steered, and random texts with base model
        baseline_texts = generate_texts(
            base_model, tokenizer, PROMPTS, args.seeds,
            max_tokens=args.max_tokens)

        steered_texts = generate_texts(
            base_model, tokenizer, PROMPTS, args.seeds,
            max_tokens=args.max_tokens, steering_vec=vec)

        random_vec = torch.randn_like(vec)
        random_vec = random_vec / random_vec.norm() * vec.norm()
        random_texts = generate_texts(
            base_model, tokenizer, PROMPTS, args.seeds,
            max_tokens=args.max_tokens, steering_vec=random_vec)

        # Measure perplexity under the ADAPTED model
        adapted_model = load_adapted_model(str(adapter_path))

        ppl_baseline = measure_ppl(adapted_model, tokenizer, baseline_texts)
        ppl_steered = measure_ppl(adapted_model, tokenizer, steered_texts)
        ppl_random = measure_ppl(adapted_model, tokenizer, random_texts)

        del adapted_model

        improved_flag = ppl_steered < ppl_baseline
        beats_random = ppl_steered < ppl_random

        feat_str = ", ".join(f"{g}(f{fi},z={z:.1f})"
                             for g, fi, z in used[:4])

        print(f"  [{i+1}/{len(authors)}] {author:>20s}: "
              f"baseline={ppl_baseline:.1f}  steered={ppl_steered:.1f}  "
              f"random={ppl_random:.1f}  "
              f"{'IMPROVED' if improved_flag else 'worse':>8s}  "
              f"{'beats_random' if beats_random else '':>12s}  "
              f"{feat_str}")

        results.append({
            "author": author,
            "ppl_baseline": ppl_baseline,
            "ppl_steered": ppl_steered,
            "ppl_random": ppl_random,
            "improved": improved_flag,
            "beats_random": beats_random,
            "features_used": [(g, fi, float(z)) for g, fi, z in used],
        })

        if improved_flag:
            improved += 1
        if beats_random:
            beat_random += 1
        total += 1

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Authors tested: {total}")
    if total > 0:
        print(f"  Steered < baseline (adapted PPL drops): "
              f"{improved}/{total} ({100*improved/total:.0f}%)")
        print(f"  Steered < random (beats control): "
              f"{beat_random}/{total} ({100*beat_random/total:.0f}%)")

        deltas = [r["ppl_steered"] - r["ppl_baseline"] for r in results]
        print(f"  Mean PPL change: {np.mean(deltas):+.1f}")

    output = {
        "sae_dir": str(sae_dir),
        "scale": args.scale,
        "prompts": PROMPTS,
        "seeds": args.seeds,
        "structural_features": {k: v for k, v in STRUCTURAL_FEATURES.items()},
        "results": results,
        "summary": {
            "n_tested": total,
            "n_improved": improved,
            "n_beat_random": beat_random,
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
