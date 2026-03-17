#!/usr/bin/env python3
"""Figure: simple annotated attention patterns — 3 example heads.

Shows 2 heads side by side:
- One "local" head (strong diagonal, previous-token)
- One "spread" head (diffuse attention across context)

Usage:
    uv run --extra viz python scripts/fig_attention_simple.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from sixteen_voices import NUM_HEADS, load_base_model, load_tokenizer

OUTPUT_DIR = Path("figures")
TEXT = "Once upon a time there was a little girl who lived in a small house near the forest"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_base_model()

    inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean_tokens = [t.replace("Ġ", "").strip() or t for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[0][0].numpy()  # (16, seq, seq)

    seq_len = len(tokens)

    # Pick 2 representative heads: local vs spread
    examples = [
        (6, "H6: Local"),
        (10, "H10: Spread"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, (head_idx, title) in zip(axes, examples):
        im = ax.imshow(attn[head_idx, :seq_len, :seq_len],
                       cmap="Blues", vmin=0, vmax=0.5, aspect="equal")

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(clean_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(clean_tokens, fontsize=8)

        ax.set_xlabel("looking at →", fontsize=10)
        ax.set_ylabel("← I am this word", fontsize=10)

    fig.suptitle("Attention patterns per head  (base TinyStories model, no adapter)",
                 fontsize=14, y=1.02, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "attention_simple.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved figures/attention_simple.png")


if __name__ == "__main__":
    main()