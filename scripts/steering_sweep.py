#!/usr/bin/env python3
"""Steering sweep: for each author's top head, scale from 0× to 2× and measure PPL.

Produces a JSON with per-author steering curves that show how smoothly
each author responds to continuous head scaling.

Usage:
    uv run --extra viz python scripts/steering_sweep.py
    uv run --extra viz python scripts/steering_sweep.py --authors poe carroll grimm
    uv run --extra viz python scripts/steering_sweep.py --heads 11 14
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices import (
    NUM_HEADS,
    compute_perplexity,
    load_adapted_model,
    load_base_model,
    load_eval_text,
    load_tokenizer,
)
from sixteen_voices.model import get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
KNOCKOUT_JSON = Path("outputs/knockout_all_heads.json")
OUTPUT_PATH = Path("outputs/steering_sweep.json")

EVAL_WORDS = 2000
MAX_LENGTH = 512
SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def get_eval_text(author: str) -> str:
    prose = load_eval_text(author, length=EVAL_WORDS * 7)
    words = prose.split()
    if len(words) > EVAL_WORDS:
        prose = " ".join(words[:EVAL_WORDS])
    return prose


def steered_ppl(model, tokenizer, text, head_idx, scale):
    """Compute PPL with one head scaled, rest at 1×."""
    if scale == 1.0:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(out.loss).item()

    head_scales = {head_idx: scale}
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_hook(head_scales))
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    hook.remove()
    return torch.exp(out.loss).item()


def main():
    parser = argparse.ArgumentParser(description="Steering sweep experiment")
    parser.add_argument("--authors", nargs="+", help="Only these authors")
    parser.add_argument("--heads", nargs="+", type=int,
                        help="Only sweep these heads (default: top 2 per author from knockout)")
    parser.add_argument("--all-heads", action="store_true",
                        help="Sweep all 16 heads (slow)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # Load knockout data for head ranking
    with open(KNOCKOUT_JSON) as f:
        knockout = json.load(f)

    # Discover authors
    all_authors = sorted([
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_model.safetensors").exists()
    ])
    authors = args.authors or all_authors
    authors = [a for a in authors if a in knockout]
    print(f"Sweeping {len(authors)} authors")

    base_model = load_base_model()
    results = {}

    for i, author in enumerate(authors):
        print(f"\n[{i+1}/{len(authors)}] {author}")
        eval_text = get_eval_text(author)
        recovery = knockout[author]["head_recovery"]

        # Which heads to sweep
        if args.all_heads:
            heads_to_sweep = list(range(NUM_HEADS))
        elif args.heads:
            heads_to_sweep = args.heads
        else:
            # Top 2 by absolute recovery + H14 + H11 (always interesting)
            ranked = sorted(range(NUM_HEADS),
                            key=lambda h: abs(recovery[f"H{h}"]), reverse=True)
            heads_to_sweep = sorted(set(ranked[:2]) | {11, 14})

        # Base PPL (no adapter)
        base_ppl = compute_perplexity(base_model, tokenizer, eval_text, max_length=MAX_LENGTH)

        # Load adapted model
        adapter_path = ADAPTERS_DIR / author / "adapter"
        model = load_adapted_model(adapter_path)

        # Full adapter PPL (all heads at 1×)
        full_ppl = compute_perplexity(model, tokenizer, eval_text, max_length=MAX_LENGTH)

        author_data = {
            "base_ppl": round(base_ppl, 2),
            "full_ppl": round(full_ppl, 2),
            "head_recovery": recovery,
            "curves": {},
        }

        for h in heads_to_sweep:
            rec = recovery[f"H{h}"]
            curve = {}
            for scale in SCALES:
                ppl = steered_ppl(model, tokenizer, eval_text, h, scale)
                curve[str(scale)] = round(ppl, 2)
            author_data["curves"][f"H{h}"] = curve
            # Summarize
            ppl_0 = curve["0.0"]
            ppl_2 = curve["2.0"]
            print(f"  H{h} (rec={rec:+.2f}): "
                  f"0×={ppl_0:.0f}  1×={full_ppl:.0f}  2×={ppl_2:.0f}  "
                  f"range={ppl_0-ppl_2:.0f}")

        results[author] = author_data
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output}")
    print(f"Total: {len(results)} authors, "
          f"{sum(len(d['curves']) for d in results.values())} steering curves")


if __name__ == "__main__":
    main()