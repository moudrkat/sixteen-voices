#!/usr/bin/env python3
"""Draw LoRA bypass flow diagram.

Usage:
    python scripts/fig_lora_flow.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_FROZEN = "#B0BEC5"
C_LORA_A = "#E8573A"
C_LORA_B = "#C0392B"
C_TEXT = "#2C3E50"
C_ARROW = "#555555"
C_BG_LORA = "#FFF0ED"


def box(ax, x, y, w, h, label, color, fontsize=10, text_color="white",
        sublabel=None, sublabel_size=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                        facecolor=color, edgecolor="#333333",
                        linewidth=1.2, zorder=2)
    ax.add_patch(b)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.15, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=text_color, zorder=3, alpha=0.9)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def flow_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="figures/lora_flow.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    # Layout
    top_y = 3.2      # frozen W path
    mid_y = 1.8      # input/output
    bot_y = 0.4      # LoRA bypass path
    bw, bh = 2.2, 0.6  # box size

    # Title
    ax.text(4.5, 4.5, "LoRA: Data Flow", fontsize=15, ha="center",
            va="center", color=C_TEXT, fontweight="bold")

    # --- Input x ---
    ax.text(0.5, mid_y, "x", fontsize=22, ha="center", va="center",
            color=C_TEXT, fontweight="bold", style="italic")
    ax.text(0.5, mid_y - 0.4, "1024-dim", fontsize=7, ha="center", color="#888")

    # --- Split point ---
    sx = 1.5
    ax.plot([0.8, sx], [mid_y, mid_y], color=C_ARROW, lw=1.5, zorder=1)
    ax.plot(sx, mid_y, "o", color=C_ARROW, markersize=6, zorder=3)

    # === TOP PATH: Frozen W ===
    flow_arrow(ax, sx, mid_y, sx, top_y - 0.1)
    flow_arrow(ax, sx, top_y, 2.2, top_y)
    box(ax, 2.2, top_y - bh / 2, bw, bh, "W  (frozen)", C_FROZEN, fontsize=11,
        text_color=C_TEXT)
    ax.text(2.2 + bw / 2, top_y + bh / 2 + 0.15, "1024 → 1024",
            fontsize=7, ha="center", color="#888")
    flow_arrow(ax, 2.2 + bw, top_y, 6.5, top_y)

    # === BOTTOM PATH: LoRA bypass ===
    flow_arrow(ax, sx, mid_y, sx, bot_y + bh / 2 + 0.1)

    # LoRA background
    lora_bg = FancyBboxPatch((1.9, bot_y - 0.35), 5.0, bh + 0.7,
                              boxstyle="round,pad=0.12",
                              facecolor=C_BG_LORA, edgecolor=C_LORA_A,
                              linewidth=1.0, alpha=0.4, zorder=0)
    ax.add_patch(lora_bg)
    ax.text(4.4, bot_y + bh + 0.15, "LoRA bypass  (trainable)",
            fontsize=8, ha="center", color=C_LORA_A, fontweight="bold")

    # A: down-project
    flow_arrow(ax, sx, bot_y + bh / 2, 2.2, bot_y + bh / 2)
    box(ax, 2.2, bot_y, bw, bh, "A", C_LORA_A, fontsize=13)
    ax.text(2.2 + bw / 2, bot_y - 0.2, "1024 → 8", fontsize=7,
            ha="center", color=C_LORA_A)

    # Bottleneck annotation
    ax.text(2.2 + bw + 0.5, bot_y + bh / 2, "r = 8",
            fontsize=9, ha="center", va="center", color="#888",
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#ddd", linewidth=0.7))

    # Arrow A -> B
    flow_arrow(ax, 2.2 + bw, bot_y + bh / 2, 4.8, bot_y + bh / 2)

    # B: up-project
    box(ax, 4.8, bot_y, bw, bh, "B", C_LORA_B, fontsize=13)
    ax.text(4.8 + bw / 2, bot_y - 0.2, "8 → 1024", fontsize=7,
            ha="center", color=C_LORA_B)

    # === SUM ===
    sum_x, sum_y = 6.8, mid_y

    # Line from B to below sum, then up to sum
    b_end = 4.8 + bw
    ax.plot([b_end, sum_x], [bot_y + bh / 2, bot_y + bh / 2],
            color=C_LORA_A, lw=1.2, zorder=1)
    ax.plot([sum_x, sum_x], [bot_y + bh / 2, sum_y - 0.2],
            color=C_LORA_A, lw=1.2, zorder=1)

    # Line from W to above sum, then down to sum
    ax.plot([6.5, sum_x], [top_y, sum_y + 0.2],
            color=C_ARROW, lw=1.2, zorder=1)

    # Plus circle
    circle = plt.Circle((sum_x, sum_y), 0.2, facecolor="white",
                         edgecolor=C_ARROW, linewidth=1.8, zorder=3)
    ax.add_patch(circle)
    ax.text(sum_x, sum_y, "+", fontsize=16, ha="center", va="center",
            color=C_TEXT, fontweight="bold", zorder=4)

    # === Output h ===
    flow_arrow(ax, sum_x + 0.2, sum_y, 7.8, sum_y)
    ax.text(8.1, sum_y, "h", fontsize=22, ha="center", va="center",
            color=C_TEXT, fontweight="bold", style="italic")
    ax.text(8.1, sum_y - 0.4, "1024-dim", fontsize=7, ha="center", color="#888")

    # Key insight annotation
    ax.text(4.5, 4.1,
            "Only A and B are trained  —  W stays frozen  —  32K vs 21M params",
            fontsize=9, ha="center", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                      edgecolor="#ddd", linewidth=0.8))

    ax.set_xlim(-0.2, 8.8)
    ax.set_ylim(-0.3, 4.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()