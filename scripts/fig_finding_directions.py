#!/usr/bin/env python3
"""Two figures showing how directions are found:
1. Anthropic's method: probing (stories → activations → direction)
2. Ours: SAE (activations → sparse autoencoder → decoder columns)

Usage:
    python scripts/fig_finding_directions.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_TEXT = "#2C3E50"
C_ARROW = "#555555"
C_DIRECTION = "#E74C3C"


def rbox(ax, x, y, w, h, label, color, fontsize=10, text_color="white",
         alpha=1.0, lw=1.5, sublabel=None, sublabel_fs=7):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=text_color, alpha=0.85, zorder=3,
                style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def arrow_r(ax, x0, y, x1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def arrow_label(ax, x, y, label, fontsize=7):
    ax.text(x, y + 0.12, label, ha="center", va="bottom",
            fontsize=fontsize, color="#666666", style="italic")


# ═══════════════════════════════════════════════════════════════════
# ANTHROPIC: probing
# ═══════════════════════════════════════════════════════════════════
def draw_anthropic(ax):
    y = 1.0
    bh = 1.2

    # Step 1: emotion word
    rbox(ax, 0.5, y, 2.2, bh, '"desperate"', "#7E57C2", fontsize=11,
         sublabel="emotion word")
    arrow_r(ax, 2.7, y + bh / 2, 3.5)
    arrow_label(ax, 3.1, y + bh / 2, "prompt")

    # Step 2: model writes stories
    rbox(ax, 3.5, y, 3.0, bh, "Claude writes\nstories about it", "#4A90D9",
         fontsize=10)
    arrow_r(ax, 6.5, y + bh / 2, 7.3)

    # Step 3: feed back, record activations
    rbox(ax, 7.3, y, 3.2, bh, "Feed stories back\nrecord activations", "#4A90D9",
         fontsize=10, alpha=0.8)
    arrow_r(ax, 10.5, y + bh / 2, 11.3)

    # Step 4: extract direction
    rbox(ax, 11.3, y, 3.0, bh, "Average pattern\n= direction", C_DIRECTION,
         fontsize=10)

    # Result arrow
    arrow_r(ax, 14.3, y + bh / 2, 15.1, color=C_DIRECTION, lw=2.0)

    # Direction vector
    rbox(ax, 15.1, y, 2.8, bh, "direction\nvector", C_DIRECTION, fontsize=11,
         sublabel='"desperate"')

    # Example text below
    ax.text(0.5, y - 0.4,
            'Stories: "Maria clutched her phone, hands trembling. The deadline was in two hours and nothing was working..."',
            fontsize=8, color="#666666", style="italic", va="top")
    ax.text(0.5, y - 0.75,
            'Activations on these stories share a pattern  →  that pattern IS the "desperate" direction',
            fontsize=8, color="#666666", va="top")

    # 171 of these
    ax.text(15.1 + 1.4, y - 0.3,
            "repeat for\n171 emotions",
            fontsize=8, color="#999999", ha="center", va="top",
            fontweight="bold")

    # Title
    ax.text(9.5, y + bh + 0.7,
            "How Anthropic finds directions: linear probing",
            fontsize=15, fontweight="bold", color=C_TEXT, ha="center")
    ax.text(9.5, y + bh + 0.35,
            "Have the model write about an emotion, then extract the activation pattern",
            fontsize=10, color="#666666", ha="center", style="italic")

    ax.set_xlim(0, 18.5)
    ax.set_ylim(-1.2, y + bh + 1.1)
    ax.set_aspect("equal")
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════
# OURS: SAE
# ═══════════════════════════════════════════════════════════════════
def draw_ours(ax):
    y = 1.0
    bh = 1.2

    # Step 1: run model, collect activations
    rbox(ax, 0.5, y, 2.8, bh, "Run model on\nmany texts", "#F39C12", fontsize=10,
         sublabel="77 authors")
    arrow_r(ax, 3.3, y + bh / 2, 4.1)
    arrow_label(ax, 3.7, y + bh / 2, "collect")

    # Step 2: activations
    rbox(ax, 4.1, y, 2.5, bh, "Activations\n(residual stream)", "#4A90D9",
         fontsize=10)
    arrow_r(ax, 6.6, y + bh / 2, 7.4)
    arrow_label(ax, 7.0, y + bh / 2, "train")

    # Step 3: SAE
    rbox(ax, 7.4, y - 0.2, 3.5, bh + 0.4, "Sparse\nAutoencoder", "#9B59B6",
         fontsize=12)

    # SAE internals note
    ax.text(9.15, y - 0.55,
            "learns to compress → reconstruct\n"
            "forces sparsity: only ~16 features\n"
            "active at once (out of 2048)",
            fontsize=7, color="#666666", ha="center", va="top",
            style="italic")

    arrow_r(ax, 10.9, y + bh / 2, 11.7)

    # Step 4: decoder columns = directions
    rbox(ax, 11.7, y, 3.0, bh, "Decoder columns\n= directions", C_DIRECTION,
         fontsize=10)
    arrow_r(ax, 14.7, y + bh / 2, 15.5, color=C_DIRECTION, lw=2.0)

    # Direction vector
    rbox(ax, 15.5, y, 2.5, bh, "direction\nvector", C_DIRECTION, fontsize=11,
         sublabel='"simplicity"')

    # Example text below
    ax.text(0.5, y - 0.4,
            "Each decoder column is a direction in activation space. Each one fires on a specific pattern.",
            fontsize=8, color="#666666", va="top")
    ax.text(0.5, y - 0.75,
            "f665 fires on short, simple sentences  →  that column IS the \"simplicity\" direction",
            fontsize=8, color="#666666", va="top")

    # Count
    ax.text(15.5 + 1.25, y - 0.3,
            "2048 features\n314 alive\n~25 interpretable",
            fontsize=8, color="#999999", ha="center", va="top",
            fontweight="bold")

    # Title
    ax.text(9.5, y + bh + 0.9,
            "How we find directions: sparse autoencoder (SAE)",
            fontsize=15, fontweight="bold", color=C_TEXT, ha="center")
    ax.text(9.5, y + bh + 0.5,
            "Train a network that compresses activations — each learned dimension becomes an interpretable direction",
            fontsize=10, color="#666666", ha="center", style="italic")

    ax.set_xlim(0, 18.5)
    ax.set_ylim(-1.3, y + bh + 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Anthropic figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(18, 3.5))
    fig1.patch.set_facecolor("white")
    draw_anthropic(ax1)
    plt.tight_layout()
    p1 = out_dir / "finding_directions_anthropic.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {p1}")
    plt.close(fig1)

    # Ours figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 4.0))
    fig2.patch.set_facecolor("white")
    draw_ours(ax2)
    plt.tight_layout()
    p2 = out_dir / "finding_directions_ours.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {p2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()