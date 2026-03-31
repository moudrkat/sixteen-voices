#!/usr/bin/env python3
"""Do LoRAs create new features or amplify existing ones?

Runs the same text through the base model and each adapted model,
compares SAE feature activations. If LoRAs mostly amplify, the same
features fire but stronger. If they create, new features appear.

Usage:
    uv run python scripts/analyze_lora_amplification.py
    uv run python scripts/analyze_lora_amplification.py --authors poe grimm carroll minimalist
    uv run python scripts/analyze_lora_amplification.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from sixteen_voices import load_base_model, load_tokenizer, TextChunkDataset
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder
from torch.utils.data import DataLoader, Subset


def collect_features(model, sae, tokenizer, text, max_chunks=100,
                     seq_len=128, seed=42):
    """Run text through model, get mean SAE feature activations."""
    dataset = TextChunkDataset(text, tokenizer, max_length=seq_len)
    if len(dataset) > max_chunks:
        torch.manual_seed(seed)
        indices = torch.randperm(len(dataset))[:max_chunks].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    activations = []
    hook_handle = model.transformer.ln_f.register_forward_hook(
        lambda mod, inp, out: activations.append(
            inp[0].detach() if isinstance(inp, tuple) else out.detach()))

    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(input_ids=batch["input_ids"])
    hook_handle.remove()

    acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)

    with torch.no_grad():
        _, hidden = sae(acts)

    # Mean activation per feature
    mean_act = hidden.mean(dim=0)
    # Firing fraction per feature
    firing = (hidden > 0.01).float().mean(dim=0)

    return mean_act.numpy(), firing.numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether LoRAs amplify or create SAE features")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--authors", nargs="*", default=None)
    parser.add_argument("--max-chunks", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "lora_amplification.json")

    # Load SAE
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    tokenizer = load_tokenizer()
    text_dir = Path("data/authors")
    adapters_dir = Path("outputs/authors")

    # Select authors
    if args.authors:
        authors = args.authors
    else:
        authors = sorted(
            d.name for d in adapters_dir.iterdir()
            if (d / "adapter" / "adapter_model.safetensors").exists()
            and (text_dir / f"{d.name}.txt").exists()
        )

    print(f"Analyzing {len(authors)} authors\n")

    # Load base model once
    print("Loading base model...")
    base_model = load_base_model()

    results = []
    all_amplified_pct = []
    all_created_pct = []
    all_correlation = []

    for i, author in enumerate(authors):
        text_path = text_dir / f"{author}.txt"
        adapter_path = adapters_dir / author / "adapter"
        if not text_path.exists() or not adapter_path.exists():
            continue

        text = text_path.read_text()

        # Base model features on this author's text
        base_mean, base_firing = collect_features(
            base_model, sae, tokenizer, text, max_chunks=args.max_chunks)

        # Adapted model features on same text
        adapted_model = load_adapted_model(str(adapter_path))
        adapted_mean, adapted_firing = collect_features(
            adapted_model, sae, tokenizer, text, max_chunks=args.max_chunks)
        del adapted_model

        # Analysis
        alive = base_mean > 0.001  # features that fire at all in base
        alive_adapted = adapted_mean > 0.001

        # Features active in both (amplified or suppressed)
        both_active = alive & alive_adapted
        # Features active only in adapted (truly new)
        created = (~alive) & alive_adapted
        # Features active only in base (suppressed by adapter)
        suppressed = alive & (~alive_adapted)

        n_both = both_active.sum()
        n_created = created.sum()
        n_suppressed = suppressed.sum()
        n_adapted_total = alive_adapted.sum()

        # Among shared features: correlation of magnitudes
        if n_both > 2:
            corr = np.corrcoef(base_mean[both_active],
                               adapted_mean[both_active])[0, 1]
        else:
            corr = 0.0

        # Among shared features: how many got amplified vs suppressed?
        if n_both > 0:
            amplified = (adapted_mean[both_active] >
                         base_mean[both_active] * 1.2).sum()
            dampened = (adapted_mean[both_active] <
                        base_mean[both_active] * 0.8).sum()
            unchanged = n_both - amplified - dampened
        else:
            amplified = dampened = unchanged = 0

        pct_created = 100 * n_created / max(n_adapted_total, 1)
        pct_amplified = 100 * n_both / max(n_adapted_total, 1)

        all_amplified_pct.append(pct_amplified)
        all_created_pct.append(pct_created)
        all_correlation.append(corr)

        print(f"  [{i+1}/{len(authors)}] {author:>20s}: "
              f"adapted uses {n_adapted_total:3d} features | "
              f"{n_both:3d} shared, {n_created:3d} new, "
              f"{n_suppressed:3d} suppressed | "
              f"r={corr:.2f} | "
              f"amplified {amplified}, dampened {dampened}")

        results.append({
            "author": author,
            "n_base_active": int(alive.sum()),
            "n_adapted_active": int(n_adapted_total),
            "n_shared": int(n_both),
            "n_created": int(n_created),
            "n_suppressed": int(n_suppressed),
            "pct_shared": float(pct_amplified),
            "pct_created": float(pct_created),
            "correlation": float(corr),
            "n_amplified": int(amplified),
            "n_dampened": int(dampened),
            "n_unchanged": int(unchanged),
        })

    # Summary
    print(f"\n=== Summary ({len(results)} authors) ===")
    print(f"  Mean features shared (base→adapted): "
          f"{np.mean(all_amplified_pct):.1f}%")
    print(f"  Mean features created (new in adapted): "
          f"{np.mean(all_created_pct):.1f}%")
    print(f"  Mean correlation of shared features: "
          f"{np.mean(all_correlation):.3f}")
    print()

    if np.mean(all_created_pct) < 10:
        print("  → LoRAs mostly AMPLIFY existing base-model features")
    elif np.mean(all_created_pct) > 40:
        print("  → LoRAs mostly CREATE new features")
    else:
        print("  → LoRAs both amplify and create features")

    output = {
        "sae_dir": str(sae_dir),
        "n_authors": len(results),
        "results": results,
        "summary": {
            "mean_pct_shared": float(np.mean(all_amplified_pct)),
            "mean_pct_created": float(np.mean(all_created_pct)),
            "mean_correlation": float(np.mean(all_correlation)),
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
