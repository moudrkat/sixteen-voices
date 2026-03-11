#!/usr/bin/env python3
"""Per-head vocabulary attribution: what words does each head promote/suppress?

Compares base model vs LoRA-adapted vocabulary changes per head by knocking out
individual heads and observing which top-k words appear/disappear.

Usage:
    python scripts/vocab_knockout.py
    python scripts/vocab_knockout.py --authors shelley poe grimm
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from sixteen_voices import (
    HEAD_DIM,
    MODEL_NAME,
    NUM_HEADS,
    get_attn_out,
    load_adapted_model,
    load_base_model,
    load_tokenizer,
)
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
TOP_K = 50

TEXTS = [
    "Once upon a time there was a little",
    "The dark forest was full of",
    "The princess smiled and said",
]

DEFAULT_AUTHORS = ["shelley", "poe", "homer", "grimm", "carroll", "wilde"]


def make_knockout_hook(head_idx):
    """Zero out a single head's output."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        start = head_idx * HEAD_DIM
        h[:, :, start : start + HEAD_DIM] = 0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def get_top_tokens(model, tokenizer, text, top_k=TOP_K):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k)
    return {
        tokenizer.decode(idx.item()).strip(): prob.item()
        for idx, prob in zip(topk.indices, topk.values)
    }


def get_vocab_diff(model, tokenizer, text, head_idx, top_k=TOP_K):
    full = get_top_tokens(model, tokenizer, text, top_k)
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_hook(make_knockout_hook(head_idx))
    ko = get_top_tokens(model, tokenizer, text, top_k)
    hook.remove()

    promoted = sorted(
        [(w, p) for w, p in full.items() if w not in ko], key=lambda x: x[1], reverse=True
    )
    suppressed = sorted(
        [(w, p) for w, p in ko.items() if w not in full], key=lambda x: x[1], reverse=True
    )
    return promoted, suppressed


def main():
    parser = argparse.ArgumentParser(description="Per-head vocabulary attribution")
    parser.add_argument("--authors", nargs="+", default=DEFAULT_AUTHORS)
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # Base model
    print("Base model — all 16 heads...")
    base_model = load_base_model()
    base_data = {}
    for text in TEXTS:
        for h in range(NUM_HEADS):
            base_data[(text, h)] = get_vocab_diff(base_model, tokenizer, text, h)
        print(f"  done: '{text[:35]}'")
    del base_model

    # Adapted models
    adapted_data = {}
    for author in args.authors:
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if not adapter_path.exists():
            print(f"  Skipping {author}")
            continue

        print(f"\n{author} — all 16 heads...")
        model = load_adapted_model(str(adapter_path))
        for text in TEXTS:
            for h in range(NUM_HEADS):
                adapted_data[(author, text, h)] = get_vocab_diff(model, tokenizer, text, h)
            print(f"  done: '{text[:35]}'")
        del model

    # Compute overlap
    overlap = np.zeros((len(args.authors), NUM_HEADS))
    for i, author in enumerate(args.authors):
        for h in range(NUM_HEADS):
            fracs = []
            for text in TEXTS:
                if (text, h) not in base_data or (author, text, h) not in adapted_data:
                    continue
                base_words = set(w for w, _ in base_data[(text, h)][0][:10])
                adapted_words = set(w for w, _ in adapted_data[(author, text, h)][0][:10])
                if base_words:
                    fracs.append(len(base_words & adapted_words) / len(base_words))
            if fracs:
                overlap[i, h] = np.mean(fracs)

    # Print results
    print(f"\n{'='*70}")
    print("OVERLAP: fraction of base promoted words that survive adaptation")
    print(f"{'='*70}")
    print(f"{'':>10s}" + "".join(f"  H{h:>2d}" for h in range(NUM_HEADS)))
    for i, author in enumerate(args.authors):
        row = "".join(f" {overlap[i, h]:4.0%}" for h in range(NUM_HEADS))
        print(f"{author:>10s}{row}")

    mean_overlap = overlap.mean(axis=0)
    print(f"\nMean overlap per head:")
    for h in range(NUM_HEADS):
        print(f"  H{h:>2d}: {mean_overlap[h]:.0%}")

    redirected = [h for h in range(NUM_HEADS) if mean_overlap[h] < 0.15]
    amplified = [h for h in range(NUM_HEADS) if mean_overlap[h] >= 0.25]
    print(f"\nRedirected (< 15%): {', '.join(f'H{h}' for h in redirected)}")
    print(f"Amplified (>= 25%): {', '.join(f'H{h}' for h in amplified)}")


if __name__ == "__main__":
    main()
