#!/usr/bin/env python3
"""Steer generation by injecting SAE feature directions into the residual stream.

Instead of scaling whole heads, this adds/subtracts specific SAE decoder
columns to the residual stream during generation. This lets you "turn up"
or "turn down" individual features like complexity, dialogue, or narrative.

Usage:
    uv run python scripts/steer_sae_features.py
    uv run python scripts/steer_sae_features.py --author poe --features 218:+5.0 92:-3.0
    uv run python scripts/steer_sae_features.py --author minimalist --features 218:+10.0
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


def make_feature_hook(sae, feature_scales: dict[int, float]):
    """Hook that adds scaled SAE decoder columns to the residual stream.

    For each feature f with scale s, adds s * decoder.weight[:, f] to the
    residual stream output. Positive scale = more of that feature,
    negative = less.
    """
    # Pre-compute the steering vector (sum of scaled decoder columns)
    # decoder = nn.Linear(n_features, input_dim) → weight is (1024, 256)
    # column [:, feat_idx] gives the 1024-dim direction for that feature
    w = sae.decoder.weight.detach()  # (1024, 256)
    steering_vec = torch.zeros(w.shape[0])  # (1024,)
    for feat_idx, scale in feature_scales.items():
        steering_vec += scale * w[:, feat_idx]

    def hook_fn(module, input, output):
        # output is the residual stream after attention+MLP
        if isinstance(output, tuple):
            modified = output[0] + steering_vec.to(output[0].device)
            return (modified,) + output[1:]
        return output + steering_vec.to(output.device)

    return hook_fn


def main():
    parser = argparse.ArgumentParser(description="Steer with SAE features")
    parser.add_argument("--author", type=str, default=None,
                        help="Load this author's adapter (or base model if omitted)")
    parser.add_argument("--features", type=str, nargs="+", default=[],
                        help="Feature steering: FEAT_IDX:SCALE (e.g. 218:+5.0 92:-3.0)")
    parser.add_argument("--prompt", type=str,
                        default="It was a dark and stormy",
                        help="Generation prompt")
    parser.add_argument("--sae-dir", type=str, default="outputs/sae")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for multiple samples")
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    # Parse feature scales
    feature_scales = {}
    for spec in args.features:
        feat_str, scale_str = spec.split(":")
        feature_scales[int(feat_str)] = float(scale_str)

    # Load SAE
    sae_dir = Path(args.sae_dir)
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    # Load model
    print("Loading model...")
    tokenizer = load_tokenizer()

    if args.author:
        from sixteen_voices.model import load_adapted_model
        model = load_adapted_model(f"outputs/authors/{args.author}/adapter")
        print(f"Loaded adapter: {args.author}")
    else:
        model = load_base_model()
        print("Using base model")

    # Baseline (no steering)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"\n--- Baseline ---")
    for seed in args.seeds:
        text = generate(model, tokenizer, args.prompt,
                        max_new_tokens=args.max_tokens, seed=seed)
        print(f"  [{seed}] {text}")

    if not feature_scales:
        print("\nNo features specified. Try: --features 198:+5.0 160:+5.0")
        print("\nValidated features (closed-loop tested):")
        print("  f198 = folk voice (↑Harris/Grimm dialect, head-independent)")
        print("  f33  = folk voice (↑traditional narration, most head-independent)")
        print("  f160 = event-chain narration (↑sequential storytelling)")
        print("  f144 = clause embedding (↑complex sentences)")
        print("  f68  = direct address (↑'now','you','then')")
        print("  f122 = speech attribution (↑'said the X')")
        return

    # Steered
    print(f"\n--- Steered: {feature_scales} ---")
    hook_target = model.transformer.ln_f  # residual stream before final LN
    hook = hook_target.register_forward_hook(
        make_feature_hook(sae, feature_scales)
    )

    try:
        for seed in args.seeds:
            text = generate(model, tokenizer, args.prompt,
                            max_new_tokens=args.max_tokens, seed=seed)
            print(f"  [{seed}] {text}")
    finally:
        hook.remove()


if __name__ == "__main__":
    main()