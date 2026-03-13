#!/usr/bin/env python3
"""Evaluate each author's LoRA adapter vs the base model.

For each adapter, computes perplexity on a held-out chunk (last ~2000 words
of the cleaned author text — roughly matching the val split used during
training) with the base model and the adapted model.

Usage:
    python scripts/eval_adapters.py
    python scripts/eval_adapters.py --only minimalist browne
    python scripts/eval_adapters.py --eval-words 3000
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices import (
    load_base_model,
    load_adapted_model,
    load_tokenizer,
    compute_perplexity,
    extract_prose,
)

DATA_DIR = Path("data/authors")
ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/eval_adapters.json")

EVAL_WORDS = 2000
MAX_LENGTH = 512


def get_eval_text(raw_text: str, eval_words: int = EVAL_WORDS) -> str:
    """Return eval_words words of extracted prose (consistent with knockout)."""
    prose = extract_prose(raw_text, length=eval_words * 7)  # ~7 chars/word
    words = prose.split()
    if len(words) > eval_words:
        prose = " ".join(words[:eval_words])
    return prose


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapters vs base model")
    parser.add_argument("--only", nargs="+", help="Evaluate only these authors")
    parser.add_argument("--eval-words", type=int, default=EVAL_WORDS,
                        help="Number of words from end of text to use for eval")
    args = parser.parse_args()

    # Discover adapters
    adapter_names = sorted(
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_config.json").exists()
    )
    if args.only:
        adapter_names = [n for n in adapter_names if n in args.only]

    print(f"Evaluating {len(adapter_names)} adapters\n")

    tokenizer = load_tokenizer()
    base_model = load_base_model()

    results = []

    for i, name in enumerate(adapter_names):
        text_path = DATA_DIR / f"{name}.txt"
        if not text_path.exists():
            print(f"  [{name}] no text file, skipping")
            continue

        raw_text = text_path.read_text(encoding="utf-8")
        eval_text = get_eval_text(raw_text, args.eval_words)
        eval_word_count = len(eval_text.split())

        # Base perplexity
        base_ppl = compute_perplexity(base_model, tokenizer, eval_text, max_length=MAX_LENGTH)

        # Adapted perplexity
        adapter_path = ADAPTERS_DIR / name / "adapter"
        adapted_model = load_adapted_model(adapter_path)
        adapted_ppl = compute_perplexity(adapted_model, tokenizer, eval_text, max_length=MAX_LENGTH)
        del adapted_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        delta = adapted_ppl - base_ppl
        ratio = adapted_ppl / base_ppl if base_ppl > 0 else float("inf")
        learned = ratio < 0.85

        results.append({
            "author": name,
            "base_ppl": round(base_ppl, 1),
            "adapted_ppl": round(adapted_ppl, 1),
            "delta_ppl": round(delta, 1),
            "ratio": round(ratio, 3),
            "learned": learned,
            "eval_words": eval_word_count,
        })

        status = "\u2713 learned" if learned else "\u2717 weak"
        print(f"  [{i+1}/{len(adapter_names)}] {name:20s}  "
              f"base={base_ppl:8.1f}  adapted={adapted_ppl:8.1f}  "
              f"ratio={ratio:.3f}  {status}")

    # Sort by ratio (best improvement first)
    results.sort(key=lambda r: r["ratio"])

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Author':<20s} {'Base PPL':>10s} {'Adapted PPL':>12s} "
          f"{chr(916)+' PPL':>8s} {'Ratio':>7s}  Status")
    print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*8} {'-'*7}  {'-'*10}")
    for r in results:
        status = "\u2713 learned" if r["learned"] else "\u2717 weak"
        print(f"{r['author']:<20s} {r['base_ppl']:10.1f} {r['adapted_ppl']:12.1f} "
              f"{r['delta_ppl']:8.1f} {r['ratio']:7.3f}  {status}")

    n_learned = sum(1 for r in results if r["learned"])
    print(f"\n{n_learned}/{len(results)} adapters showed meaningful learning (ratio < 0.85)")

    # Save JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
