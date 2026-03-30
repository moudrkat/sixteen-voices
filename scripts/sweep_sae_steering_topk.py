#!/usr/bin/env python3
"""Sweep SAE feature steering experiments with the overcomplete TopK SAE.

Runs a set of steering experiments, generates baseline + steered text,
and saves all results to JSON for reproducibility.

Usage:
    uv run python scripts/sweep_sae_steering_topk.py
    uv run python scripts/sweep_sae_steering_topk.py --sae-dir outputs/sae_topk16_2048
    uv run python scripts/sweep_sae_steering_topk.py --experiments poe_minimalist grimm_dialogue
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


# Feature definitions for the overcomplete TopK SAE (2048 features, k=16)
FEATURES = {
    "minimalist": {665: 1.0},       # simplicity direction, head-independent
    "complexity": {883: 1.0, 993: 1.0, 60: 1.0},  # Lear/Baker/Poe ornate prose
    "dialogue": {1777: 1.0, 689: 1.0},             # dialogue style
    "rhetorical_q": {1385: 1.0},    # rhetorical questions (questioner)
    "marilla": {1518: 1.0},         # character name detector (Montgomery)
    "dialect": {111: 1.0},          # dialect spelling (repeater/Harris)
    "reporter": {342: 1.0},         # reporter style
    "childrens_lit": {1250: 1.0, 811: 1.0},  # Carroll/Barrie/Collodi voice
}

EXPERIMENTS = {
    "poe_minimalist": {
        "author": "poe",
        "features": "minimalist",
        "scale": 15.0,
        "prompt": "It was a dark and stormy",
        "description": "Strip gothic prose to bare-bones minimalism",
    },
    "minimalist_complexity": {
        "author": "minimalist",
        "features": "complexity",
        "scale": 12.0,
        "prompt": "The cat sat on",
        "description": "Push simple voice toward ornate prose",
    },
    "grimm_dialogue": {
        "author": "grimm",
        "features": "dialogue",
        "scale": 5.0,
        "prompt": "Once upon a time there lived",
        "description": "Make fairy tale characters talk",
    },
    "grimm_dialogue_strong": {
        "author": "grimm",
        "features": "dialogue",
        "scale": 12.0,
        "prompt": "Once upon a time there lived",
        "description": "Dialogue too strong — expect degeneration",
    },
    "montgomery_marilla": {
        "author": "montgomery",
        "features": "marilla",
        "scale": 20.0,
        "prompt": "She looked around the room and",
        "description": "Inject character-specific feature into author",
    },
    "base_minimalist": {
        "author": None,
        "features": "minimalist",
        "scale": 10.0,
        "prompt": "Once upon a time",
        "description": "Base model + minimalist simplification",
    },
    "base_dialogue": {
        "author": None,
        "features": "dialogue",
        "scale": 10.0,
        "prompt": "Once upon a time",
        "description": "Base model + dialogue injection",
    },
    "poe_dialogue": {
        "author": "poe",
        "features": "dialogue",
        "scale": 10.0,
        "prompt": "It was a dark and stormy",
        "description": "Gothic + dialogue — expect degeneration",
    },
    "wilde_minimalist": {
        "author": "wilde",
        "features": "minimalist",
        "scale": 12.0,
        "prompt": "The garden was beautiful and",
        "description": "Strip Wilde's ornate style down",
    },
    "homer_dialogue": {
        "author": "homer",
        "features": "dialogue",
        "scale": 5.0,
        "prompt": "The warrior stood before the gates",
        "description": "Epic narration gets dialogue",
    },
    "base_childrens_lit": {
        "author": None,
        "features": "childrens_lit",
        "scale": 10.0,
        "prompt": "Once upon a time",
        "description": "Inject Carroll/Barrie/Collodi voice",
    },
    "carroll_minimalist": {
        "author": "carroll",
        "features": "minimalist",
        "scale": 12.0,
        "prompt": "Alice was beginning to get very",
        "description": "Strip Wonderland to simple sentences",
    },
}

SEEDS = [42, 123, 456]


def make_steering_hook(sae, feature_scales):
    w = sae.decoder.weight.detach()
    steering_vec = torch.zeros(w.shape[0])
    for feat_idx, scale in feature_scales.items():
        steering_vec += scale * w[:, feat_idx]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            modified = output[0] + steering_vec.to(output[0].device)
            return (modified,) + output[1:]
        return output + steering_vec.to(output.device)
    return hook_fn


def run_experiment(exp_name, exp_config, sae, tokenizer, model_cache,
                   max_tokens=80):
    author = exp_config["author"]
    feat_name = exp_config["features"]
    scale = exp_config["scale"]
    prompt = exp_config["prompt"]

    # Build scaled feature dict
    feature_scales = {k: v * scale for k, v in FEATURES[feat_name].items()}

    # Get or load model
    cache_key = author or "__base__"
    if cache_key not in model_cache:
        if author:
            model_cache[cache_key] = load_adapted_model(
                f"outputs/authors/{author}/adapter")
        else:
            model_cache[cache_key] = load_base_model()
    model = model_cache[cache_key]

    result = {
        "name": exp_name,
        "description": exp_config["description"],
        "author": author,
        "feature_group": feat_name,
        "feature_indices": list(feature_scales.keys()),
        "scale": scale,
        "prompt": prompt,
        "baseline": {},
        "steered": {},
    }

    # Baseline
    for seed in SEEDS:
        text = generate(model, tokenizer, prompt,
                        max_new_tokens=max_tokens, seed=seed)
        result["baseline"][str(seed)] = text

    # Steered
    hook_target = model.transformer.ln_f
    hook = hook_target.register_forward_hook(
        make_steering_hook(sae, feature_scales))
    try:
        for seed in SEEDS:
            text = generate(model, tokenizer, prompt,
                            max_new_tokens=max_tokens, seed=seed)
            result["steered"][str(seed)] = text
    finally:
        hook.remove()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sweep SAE steering experiments")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Run specific experiments (default: all)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <sae-dir>/steering_sweep.json)")
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = Path(args.output) if args.output else sae_dir / "steering_sweep.json"

    # Load SAE
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Select experiments
    exp_names = args.experiments or list(EXPERIMENTS.keys())
    print(f"\nRunning {len(exp_names)} experiments: {', '.join(exp_names)}\n")

    model_cache = {}
    results = []

    for i, name in enumerate(exp_names):
        if name not in EXPERIMENTS:
            print(f"  Unknown experiment: {name}, skipping")
            continue
        exp = EXPERIMENTS[name]
        author_str = exp['author'] or 'base'
        print(f"  [{i+1}/{len(exp_names)}] {name} "
              f"({author_str} + {exp['features']} @ {exp['scale']})")

        result = run_experiment(name, exp, sae, tokenizer, model_cache,
                                max_tokens=args.max_tokens)
        results.append(result)

        # Print a quick preview
        seed0 = str(SEEDS[0])
        baseline_preview = result["baseline"][seed0][:60]
        steered_preview = result["steered"][seed0][:60]
        print(f"         baseline: {baseline_preview}...")
        print(f"         steered:  {steered_preview}...")
        print()

    # Save
    output = {
        "sae_dir": str(sae_dir),
        "sae_config": config,
        "feature_definitions": {k: {str(fi): s for fi, s in v.items()}
                                for k, v in FEATURES.items()},
        "seeds": SEEDS,
        "experiments": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(results)} experiments to {output_path}")


if __name__ == "__main__":
    main()
