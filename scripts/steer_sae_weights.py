#!/usr/bin/env python3
"""Feature-guided LoRA weight steering.

Instead of injecting vectors at runtime, modify the adapter weights
along SAE feature directions in weight space. This is a permanent,
clean modification.

For each feature, we find its direction in weight space by comparing
high-activation vs low-activation authors' LoRA deltas.

Usage:
    uv run python scripts/steer_sae_weights.py --author minimalist --feature complexity --scale 1.0
    uv run python scripts/steer_sae_weights.py --author poet --feature complexity --scale 0.5
    uv run python scripts/steer_sae_weights.py --author grimm --feature dialogue --scale 1.0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from sixteen_voices.model import load_base_model, load_tokenizer, load_adapted_model
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.constants import RANK

SAE_DIR = Path("outputs/sae")
ADAPTERS_DIR = Path("outputs/authors")

FEATURE_GROUPS = {
    "folk_voice": [198, 33, 140],
    "event_narration": [160, 144, 205],
    "speech_patterns": [68, 113, 122],
}


def compute_feature_directions():
    """For each feature group, compute the direction in weight space.

    Direction = mean delta(high authors) - mean delta(low authors)
    for both q_proj and v_proj.
    """
    # Load feature matrix
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()  # (77, 256)

    # Load all deltas
    print("Loading all adapter deltas...")
    all_deltas = {}
    for author in authors:
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if adapter_path.exists():
            all_deltas[author] = load_adapter_deltas(adapter_path)

    directions = {}
    for group_name, feature_indices in FEATURE_GROUPS.items():
        # Average activation across features in group
        group_act = matrix[:, feature_indices].mean(axis=1)  # (77,)

        # Top and bottom quartile
        n = len(authors)
        k = max(n // 4, 5)
        high_idx = np.argsort(group_act)[-k:]
        low_idx = np.argsort(group_act)[:k]

        high_authors = [authors[i] for i in high_idx if authors[i] in all_deltas]
        low_authors = [authors[i] for i in low_idx if authors[i] in all_deltas]

        print(f"  {group_name}: high={high_authors[:5]}, low={low_authors[:5]}")

        direction = {}
        for proj in ["q_proj", "v_proj"]:
            high_mean = torch.stack([all_deltas[a][proj] for a in high_authors]).mean(dim=0)
            low_mean = torch.stack([all_deltas[a][proj] for a in low_authors]).mean(dim=0)
            diff = high_mean - low_mean
            # Normalize to unit norm so scale is interpretable
            direction[proj] = diff / diff.norm()

        directions[group_name] = direction

    return directions


def apply_weight_steering(model, author, direction, scale):
    """Modify model's LoRA weights by adding scale * direction."""
    from sixteen_voices.model import get_attn_module

    # Load original deltas
    original = load_adapter_deltas(ADAPTERS_DIR / author / "adapter")

    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        # New delta = original + scale * direction * original_norm
        # (scale direction relative to the adapter's own magnitude)
        orig_norm = original[proj].norm()
        new_delta = original[proj] + scale * direction[proj] * orig_norm

        # Re-factorize into A, B
        A, B = delta_to_AB(new_delta, rank=RANK)

        # Inject
        lora = getattr(attn, proj)
        lora.lora_A["default"].weight.data.copy_(A)
        lora.lora_B["default"].weight.data.copy_(B)


def save_steered_adapter(author, direction, scale, save_dir):
    """Save the steered adapter as a new adapter directory."""
    import shutil
    from safetensors.torch import save_file

    src = ADAPTERS_DIR / author / "adapter"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy(src / "adapter_config.json", save_dir / "adapter_config.json")

    # Load original weights, compute steered deltas, re-factorize
    original = load_adapter_deltas(src)
    weights = {}
    prefix = "base_model.model.transformer.h.0.attn.attention"

    for proj in ["q_proj", "v_proj"]:
        orig_norm = original[proj].norm()
        new_delta = original[proj] + scale * direction[proj] * orig_norm
        A, B = delta_to_AB(new_delta, rank=RANK)
        weights[f"{prefix}.{proj}.lora_A.weight"] = A
        weights[f"{prefix}.{proj}.lora_B.weight"] = B

    save_file(weights, str(save_dir / "adapter_model.safetensors"))

    # Save metadata
    import json
    meta = {
        "base_author": author,
        "feature": "weight_steered",
        "scale": scale,
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def generate(model, tokenizer, prompt, max_new=80, temp=0.7, seed=42):
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=temp,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--feature", required=True, choices=list(FEATURE_GROUPS.keys()))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="How much to steer (0=none, 1=full direction, -1=reverse)")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--save", action="store_true",
                        help="Save the steered adapter to outputs/steered/")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    directions = compute_feature_directions()
    direction = directions[args.feature]

    # Baseline
    print(f"\nLoading {args.author}...")
    model = load_adapted_model(str(ADAPTERS_DIR / args.author / "adapter"))

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"\n--- Baseline ({args.author}) ---")
    for seed in args.seeds:
        text = generate(model, tokenizer, args.prompt, seed=seed)
        print(f"  [{seed}] {text}")

    # Steered
    print(f"\n--- Weight-steered: {args.feature} × {args.scale:+.1f} ---")
    apply_weight_steering(model, args.author, direction, args.scale)
    for seed in args.seeds:
        text = generate(model, tokenizer, args.prompt, seed=seed)
        print(f"  [{seed}] {text}")

    # Save steered adapter
    if args.save:
        save_dir = Path("outputs/steered") / f"{args.author}_{args.feature}_{args.scale:+.1f}"
        save_steered_adapter(args.author, direction, args.scale, save_dir)
        print(f"\nSaved steered adapter to {save_dir}")

    # Also try reverse
    if args.scale > 0:
        print(f"\n--- Weight-steered: {args.feature} × {-args.scale:+.1f} (reverse) ---")
        model = load_adapted_model(str(ADAPTERS_DIR / args.author / "adapter"))
        apply_weight_steering(model, args.author, direction, -args.scale)
        for seed in args.seeds:
            text = generate(model, tokenizer, args.prompt, seed=seed)
            print(f"  [{seed}] {text}")


if __name__ == "__main__":
    main()