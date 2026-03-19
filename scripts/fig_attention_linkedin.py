#!/usr/bin/env python3
"""LinkedIn-optimized attention patterns figure with witch quote as prompt.

Landscape 16:9 for LinkedIn image preview. 4x4 grid of attention
heatmaps — the input sentence IS the witch quote.

Usage:
    uv run --extra viz python scripts/fig_attention_linkedin.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices import NUM_HEADS, load_base_model, load_tokenizer

OUTPUT_DIR = Path("figures")
TEXT = "Every head sees something different, whispered the witch. That's the trick."


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_base_model()

    inputs = tokenizer(TEXT, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean_tokens = [t.replace("\u0120", " ").strip() for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[0][0].numpy()

    seq_len = len(tokens)

    # --- Figure: 4x4 grid, full width, landscape ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for h in range(NUM_HEADS):
        ax = axes[h // 4, h % 4]
        ax.imshow(attn[h, :seq_len, :seq_len], cmap="Blues", vmin=0, vmax=0.5)

        ax.set_title(f"H{h}", fontsize=9, fontweight="bold")

        fs = 5 if seq_len <= 20 else 3.5
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(clean_tokens, rotation=90, fontsize=fs)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(clean_tokens, fontsize=fs)

    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    out = OUTPUT_DIR / "attention_patterns_linkedin.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()