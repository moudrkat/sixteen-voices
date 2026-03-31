#!/usr/bin/env python3
"""Test whether weight steering works for SAE features.

Computes a weight-space direction (mean delta of high-scoring authors
minus low-scoring authors) and applies it to test authors. Measures
whether the targeted text property changes.

Usage:
    uv run python scripts/validate_weight_steering.py
    uv run python scripts/validate_weight_steering.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
import re
from pathlib import Path

import torch
import numpy as np

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


def avg_sentence_len(texts):
    lengths = []
    for t in texts:
        sents = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
        if sents:
            lengths.extend(len(s.split()) for s in sents)
    return np.mean(lengths) if lengths else 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate weight steering for SAE features")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--feature", type=int, default=665,
                        help="Feature to test (default: 665 = simplicity)")
    parser.add_argument("--n-high", type=int, default=5)
    parser.add_argument("--n-low", type=int, default=5)
    parser.add_argument("--test-authors", nargs="*",
                        default=["poe", "grimm", "carroll", "wilde", "homer"])
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "weight_steering_validation.json")

    afm = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    authors = afm["authors"]
    matrix = afm["matrix"].numpy()

    # Find high and low authors for this feature
    feat_col = matrix[:, args.feature]
    high_idx = feat_col.argsort()[-args.n_high:]
    low_idx = feat_col.argsort()[:args.n_low]
    high_authors = [authors[i] for i in high_idx]
    low_authors = [authors[i] for i in low_idx]

    print(f"Feature {args.feature}:")
    print(f"  High: {high_authors}")
    print(f"  Low: {low_authors}")

    # Compute weight direction
    tokenizer = load_tokenizer()
    base_model = load_base_model()
    base_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    def get_deltas(author_list):
        deltas = []
        for author in author_list:
            adapter_path = f"outputs/authors/{author}/adapter"
            if not Path(adapter_path).exists():
                continue
            model = load_adapted_model(adapter_path)
            delta = {}
            for k, v in model.state_dict().items():
                if k in base_state:
                    d = v - base_state[k]
                    if d.abs().max() > 1e-8:
                        delta[k] = d
            deltas.append(delta)
            del model
        return deltas

    print("\nComputing weight direction...")
    high_deltas = get_deltas(high_authors)
    low_deltas = get_deltas(low_authors)

    keys = set(high_deltas[0].keys()) & set(low_deltas[0].keys())
    direction = {}
    for k in keys:
        high_mean = torch.stack([d[k] for d in high_deltas]).mean(0)
        low_mean = torch.stack([d[k] for d in low_deltas]).mean(0)
        direction[k] = high_mean - low_mean

    # Test on authors
    print(f"\nWeight steering test (scale={args.scale}):\n")
    results = []
    improved = 0

    for author in args.test_authors:
        adapter_path = f"outputs/authors/{author}/adapter"
        if not Path(adapter_path).exists():
            continue

        model = load_adapted_model(adapter_path)

        # Baseline
        baseline_texts = [
            generate(model, tokenizer, "Once upon a time",
                     max_new_tokens=60, seed=s)
            for s in range(args.n_seeds)]

        # Apply direction
        with torch.no_grad():
            params = dict(model.named_parameters())
            for k, d in direction.items():
                if k in params:
                    params[k].data += d * args.scale

        # Steered
        steered_texts = [
            generate(model, tokenizer, "Once upon a time",
                     max_new_tokens=60, seed=s)
            for s in range(args.n_seeds)]

        del model

        base_sl = avg_sentence_len(baseline_texts)
        steer_sl = avg_sentence_len(steered_texts)
        did_change = steer_sl < base_sl  # for simplicity, shorter is success

        print(f"  {author:>10s}: baseline={base_sl:.1f}  "
              f"steered={steer_sl:.1f}  "
              f"{'SHORTER' if did_change else 'same/longer'}")

        if did_change:
            improved += 1

        results.append({
            "author": author,
            "baseline_sent_len": float(base_sl),
            "steered_sent_len": float(steer_sl),
            "improved": did_change,
        })

    total = len(results)
    print(f"\n  Result: {improved}/{total} improved "
          f"({100*improved/total:.0f}%)")

    output = {
        "feature": args.feature,
        "high_authors": high_authors,
        "low_authors": low_authors,
        "scale": args.scale,
        "results": results,
        "summary": {"improved": improved, "total": total},
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
