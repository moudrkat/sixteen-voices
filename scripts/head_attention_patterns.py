#!/usr/bin/env python3
"""Analyze attention patterns: what does each head attend to?

For each head, computes attention weight statistics:
- Position bias: does it attend to the previous token? first token? nearby tokens?
- Token type: does it attend more to punctuation, function words, content words?
- Entropy: is attention focused (low entropy) or spread out (high entropy)?

This is the classic Voita et al. analysis, trivially clean on a 1-layer model.

Usage:
    uv run python scripts/head_attention_patterns.py
    uv run python scripts/head_attention_patterns.py --with-adapters poe grimm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sixteen_voices import NUM_HEADS, load_base_model, load_adapted_model, load_tokenizer
from sixteen_voices.model import get_attn_module

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/head_attention_patterns.json")

TEXTS = [
    "Once upon a time there was a little girl who lived in a small house near the forest.",
    "The dark sky above the clouds seemed to go away and the night was very cold.",
    "She walked into the garden and found a beautiful flower that was red and blue.",
    "The king said to the princess you must go on a journey to find the golden key.",
    "It was raining and the wind was blowing hard but the little boy did not care.",
    "The cat sat on the mat and watched the birds flying in the sky above the trees.",
    "Once there was a brave knight who fought a dragon and saved the village.",
    "The old woman told the children a story about a magic lamp that could grant wishes.",
    "In the deep dark forest there lived a bear who loved to eat honey and berries.",
    "The little fish swam in the pond and played with his friends all day long.",
]

FUNCTION_WORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "and", "or", "but", "if", "then", "that", "which", "who", "whom",
    "this", "these", "those", "it", "its", "he", "she", "they", "we",
    "his", "her", "their", "our", "my", "your", "him", "them", "us",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "not", "no", "nor", "so", "as", "very", "too", "also", "just",
}

PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"}


def get_attention_weights(model, tokenizer, text):
    """Get attention weights [num_heads, seq_len, seq_len] for a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # attentions is tuple of (batch, heads, seq, seq) per layer — we have 1 layer
    attn = outputs.attentions[0][0]  # (num_heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return attn.numpy(), tokens


def analyze_head(attn_weights_list, tokens_list):
    """Analyze one head's attention patterns across multiple texts.

    Returns dict with pattern statistics.
    """
    prev_token_fracs = []
    first_token_fracs = []
    local_fracs = []  # within 3 tokens
    entropies = []
    function_word_fracs = []
    punct_fracs = []

    for attn, tokens in zip(attn_weights_list, tokens_list):
        seq_len = len(tokens)
        if seq_len < 4:
            continue

        for pos in range(1, seq_len):  # skip position 0 (no previous)
            weights = attn[pos, :pos + 1]  # attention from this position

            # Previous token fraction
            prev_token_fracs.append(float(weights[pos - 1]))

            # First token (BOS/start) fraction
            first_token_fracs.append(float(weights[0]))

            # Local (within 3 tokens) fraction
            local_start = max(0, pos - 3)
            local_fracs.append(float(weights[local_start:pos + 1].sum()))

            # Entropy
            w = weights[weights > 1e-8]
            if len(w) > 0:
                entropy = -float((w * np.log2(w)).sum())
                entropies.append(entropy)

            # Function word attention
            func_mask = np.array([
                1.0 if tokens[i].lower().strip("Ġ") in FUNCTION_WORDS else 0.0
                for i in range(pos + 1)
            ])
            func_attn = float((weights * func_mask).sum())
            function_word_fracs.append(func_attn)

            # Punctuation attention
            punct_mask = np.array([
                1.0 if any(c in tokens[i] for c in PUNCTUATION_TOKENS) else 0.0
                for i in range(pos + 1)
            ])
            punct_attn = float((weights * punct_mask).sum())
            punct_fracs.append(punct_attn)

    return {
        "prev_token_frac": round(float(np.mean(prev_token_fracs)), 4),
        "first_token_frac": round(float(np.mean(first_token_fracs)), 4),
        "local_frac": round(float(np.mean(local_fracs)), 4),
        "entropy": round(float(np.mean(entropies)), 4),
        "function_word_frac": round(float(np.mean(function_word_fracs)), 4),
        "punct_frac": round(float(np.mean(punct_fracs)), 4),
    }


def classify_head(stats):
    """Give a human-readable label based on attention pattern."""
    labels = []
    if stats["prev_token_frac"] > 0.3:
        labels.append("previous-token")
    if stats["first_token_frac"] > 0.3:
        labels.append("BOS/positional")
    if stats["local_frac"] > 0.7:
        labels.append("local-window")
    if stats["entropy"] < 1.5:
        labels.append("focused")
    if stats["entropy"] > 3.0:
        labels.append("diffuse")
    if stats["function_word_frac"] > 0.5:
        labels.append("function-words")
    if stats["punct_frac"] > 0.15:
        labels.append("punctuation")
    return ", ".join(labels) if labels else "mixed"


def probe_model_attention(model, tokenizer, label):
    """Analyze all heads' attention patterns for a model."""
    print(f"\n{'='*70}")
    print(f"Attention patterns: {label}")
    print(f"{'='*70}")

    # Collect attention weights across all texts
    all_attn = []
    all_tokens = []
    for text in TEXTS:
        attn, tokens = get_attention_weights(model, tokenizer, text)
        all_attn.append(attn)
        all_tokens.append(tokens)

    result = {"label": label, "heads": {}}

    print(f"{'Head':>6s} {'prev':>6s} {'first':>6s} {'local':>6s} "
          f"{'entropy':>8s} {'func':>6s} {'punct':>6s}  Pattern")
    print("-" * 70)

    for h in range(NUM_HEADS):
        # Extract this head's attention across all texts
        head_attn = [a[h] for a in all_attn]
        stats = analyze_head(head_attn, all_tokens)
        pattern = classify_head(stats)
        result["heads"][f"H{h}"] = {**stats, "pattern": pattern}

        print(f"  H{h:2d}  {stats['prev_token_frac']:6.3f} {stats['first_token_frac']:6.3f} "
              f"{stats['local_frac']:6.3f} {stats['entropy']:8.3f} "
              f"{stats['function_word_frac']:6.3f} {stats['punct_frac']:6.3f}  {pattern}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-adapters", nargs="+", help="Also analyze adapted models")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    results = {}

    # Base model
    base_model = load_base_model()
    results["base"] = probe_model_attention(base_model, tokenizer, "TinyStories base")

    # Adapted models
    if args.with_adapters:
        for author in args.with_adapters:
            adapter_path = ADAPTERS_DIR / author / "adapter"
            if not adapter_path.exists():
                print(f"Skipping {author}")
                continue
            model = load_adapted_model(adapter_path)
            results[author] = probe_model_attention(model, tokenizer, f"{author} adapter")
            del model

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()