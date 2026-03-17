#!/usr/bin/env python3
"""Figure: attention pattern heatmaps for each head.

Shows what each head attends to — position bias, entropy, and
specialization. Two panels:
1. Attention heatmap for one sentence (16 heads × seq_len × seq_len)
2. Summary bar chart of head types vs knockout recovery

Usage:
    uv run --extra viz python scripts/fig_attention_patterns.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    # Clean up token display
    clean_tokens = [t.replace("Ġ", " ").strip() for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[0][0].numpy()  # (16, seq, seq)

    seq_len = len(tokens)

    # --- Figure: 4×4 grid of attention heatmaps ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    # Load attention pattern data for labels
    attn_path = Path("outputs/head_attention_patterns.json")
    patterns = {}
    if attn_path.exists():
        with open(attn_path) as f:
            pdata = json.load(f)
        for h in range(16):
            hd = pdata["base"]["heads"][f"H{h}"]
            patterns[h] = hd["pattern"]

    for h in range(NUM_HEADS):
        ax = axes[h // 4, h % 4]
        # Show attention weights as heatmap
        im = ax.imshow(attn[h, :seq_len, :seq_len], cmap="Blues", vmin=0, vmax=0.5)

        pattern = patterns.get(h, "")
        short = pattern.split(",")[0] if pattern else ""
        ax.set_title(f"H{h} ({short})", fontsize=9, fontweight="bold")

        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(clean_tokens, rotation=90, fontsize=5)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(clean_tokens, fontsize=5)
        else:
            ax.set_xticks([0, seq_len-1])
            ax.set_xticklabels([clean_tokens[0], clean_tokens[-1]], fontsize=6)
            ax.set_yticks([0, seq_len-1])

    fig.suptitle(f'Attention patterns: "{TEXT[:50]}..."', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "attention_patterns.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/attention_patterns.png")

    # --- Figure: Head specialization summary ---
    # Compact 2-panel: entropy + prev-token per head, colored by type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if attn_path.exists():
        with open(attn_path) as f:
            pdata = json.load(f)
        base = pdata["base"]["heads"]

        entropies = [base[f"H{h}"]["entropy"] for h in range(16)]
        prev_fracs = [base[f"H{h}"]["prev_token_frac"] for h in range(16)]
        local_fracs = [base[f"H{h}"]["local_frac"] for h in range(16)]

        # Color by type
        colors = []
        for h in range(16):
            p = patterns.get(h, "mixed")
            if "previous-token" in p:
                colors.append("#f97316")
            elif "focused" in p:
                colors.append("#eab308")
            elif "local-window" in p:
                colors.append("#22c55e")
            else:
                colors.append("#6366f1")

        ax1.bar(range(16), entropies, color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_xticks(range(16))
        ax1.set_xticklabels([f"H{h}" for h in range(16)], fontsize=8)
        ax1.set_ylabel("Attention entropy (bits)")
        ax1.set_title("How focused is each head?")
        ax1.axhline(y=2.0, color="gray", linewidth=0.5, linestyle="--", label="focused ↔ diffuse")

        ax2.bar(range(16), prev_fracs, color=colors, edgecolor="white", linewidth=0.5)
        ax2.set_xticks(range(16))
        ax2.set_xticklabels([f"H{h}" for h in range(16)], fontsize=8)
        ax2.set_ylabel("Previous-token attention fraction")
        ax2.set_title("Previous-token attention bias")
        ax2.axhline(y=0.2, color="gray", linewidth=0.5, linestyle="--")

        from matplotlib.patches import Patch
        ax2.legend(
            [Patch(color="#f97316"), Patch(color="#eab308"),
             Patch(color="#22c55e"), Patch(color="#6366f1")],

            ["prev-token", "focused", "local", "diffuse"],
            fontsize=8, loc="upper right",
        )

    fig.suptitle("Head specialization in TinyStories-1Layer-21M (base model)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "head_specialization.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/head_specialization.png")


if __name__ == "__main__":
    main()