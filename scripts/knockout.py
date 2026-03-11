#!/usr/bin/env python3
"""Head knockout: keep only one head's LoRA contribution, measure perplexity recovery.

For each of 16 heads and each author: zero all other heads' rows in the LoRA delta,
reconstruct A/B via SVD, measure PPL on actual prose. Outputs a JSON with recovery
scores (1.0 = head alone recovers all of full adapter's improvement).

Usage:
    python scripts/knockout.py
    python scripts/knockout.py --authors shelley poe grimm
    python scripts/knockout.py --output results.json
"""

import argparse
import json
from pathlib import Path

from sixteen_voices import (
    NUM_HEADS,
    compute_perplexity,
    extract_prose,
    inject_knockout,
    load_adapted_model,
    load_adapter_deltas,
    load_base_model,
    load_tokenizer,
)

ADAPTERS_DIR = Path("outputs/authors")
DATA_DIR = Path("data/authors")


def main():
    parser = argparse.ArgumentParser(description="Head knockout experiment")
    parser.add_argument("--authors", nargs="+", help="Only these authors")
    parser.add_argument("--output", type=str, default="outputs/knockout_all_heads.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # Discover authors
    all_authors = sorted([
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_model.safetensors").exists()
    ])
    authors = args.authors or all_authors
    print(f"Testing {len(authors)} authors, all {NUM_HEADS} heads")

    # Base model perplexities
    print("Loading base model...")
    base_model = load_base_model()

    results = {}
    for ai, author in enumerate(authors):
        txt_path = DATA_DIR / f"{author}.txt"
        if not txt_path.exists():
            print(f"  {author}: no text file, skipping")
            continue
        text = extract_prose(txt_path.read_text())
        if len(text) < 100:
            print(f"  {author}: text too short, skipping")
            continue

        base_ppl = compute_perplexity(base_model, tokenizer, text)

        # Full adapter
        adapter_path = str(ADAPTERS_DIR / author / "adapter")
        model = load_adapted_model(adapter_path)
        full_ppl = compute_perplexity(model, tokenizer, text)
        deltas = load_adapter_deltas(adapter_path)
        improvement = base_ppl - full_ppl

        head_recovery = {}
        for h in range(NUM_HEADS):
            inject_knockout(model, deltas, h)
            h_ppl = compute_perplexity(model, tokenizer, text)
            if abs(improvement) < 0.01:
                rec = 0.0
            else:
                rec = (base_ppl - h_ppl) / improvement
            head_recovery[f"H{h}"] = round(rec, 3)

        del model

        results[author] = {
            "base_ppl": round(base_ppl, 2),
            "full_ppl": round(full_ppl, 2),
            "head_recovery": head_recovery,
        }

        best_h = max(head_recovery, key=head_recovery.get)
        best_v = head_recovery[best_h]
        rec_str = " ".join(f"{head_recovery[f'H{h}']:+.2f}" for h in range(NUM_HEADS))
        print(f"  [{ai+1:2d}/{len(authors)}] {author:20s}  best={best_h}({best_v:+.2f})  {rec_str}")

    del base_model

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
