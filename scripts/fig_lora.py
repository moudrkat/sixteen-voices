#!/usr/bin/env python3
"""Draw LoRA weight decomposition figure.

Usage:
    python scripts/fig_lora.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_FROZEN = "#B0BEC5"
C_LORA_A = "#E8573A"
C_LORA_B = "#C0392B"
C_DELTA = "#E74C3C"
C_TEXT = "#2C3E50"
C_ARROW = "#555555"


def matrix_box(ax, x, y, w, h, label, color, fontsize=9, alpha=1.0,
               sublabel=None, sublabel_size=6.5, text_color="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor="#333333",
                         linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.15, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=text_color, zorder=3, alpha=0.9)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="figures/lora_weights.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    y0 = 0.3
    mat_s = 1.8

    # Title
    ax.text(4.5, y0 + mat_s + 0.7, "LoRA: Low-Rank Adaptation",
            fontsize=15, ha="center", va="center", color=C_TEXT, fontweight="bold")

    # Equation
    ax.text(4.5, y0 + mat_s + 0.35, "output  =  W·x  +  B·A·x",
            fontsize=12, ha="center", va="center", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", linewidth=0.8))

    # --- Frozen W ---
    wx = 0.3
    matrix_box(ax, wx, y0, mat_s, mat_s, "W", C_FROZEN, fontsize=18,
               sublabel="1024 × 1024\nfrozen")
    ax.text(wx + mat_s / 2, y0 - 0.25, "21M params",
            fontsize=7, ha="center", color="#888", style="italic")

    # Plus
    ax.text(2.5, y0 + mat_s / 2, "+", fontsize=22, ha="center", va="center",
            color=C_TEXT, fontweight="bold")

    # --- LoRA B (tall thin: 1024 × 8) ---
    bx, bw, bh = 3.0, 0.35, mat_s
    matrix_box(ax, bx, y0, bw, bh, "B", C_LORA_B, fontsize=11)
    ax.text(bx + bw / 2, y0 - 0.25, "1024×8", fontsize=6, ha="center",
            color=C_LORA_B, fontweight="bold")

    # Dot
    ax.text(3.55, y0 + mat_s / 2, "·", fontsize=24, ha="center", va="center",
            color=C_TEXT, fontweight="bold")

    # --- LoRA A (short wide: 8 × 1024) ---
    a_x, aw, ah = 3.8, mat_s, 0.35
    matrix_box(ax, a_x, y0 + (mat_s - ah) / 2, aw, ah, "A", C_LORA_A, fontsize=11)
    ax.text(a_x + aw / 2, y0 - 0.25, "8×1024", fontsize=6, ha="center",
            color=C_LORA_A, fontweight="bold")

    # Brace for trainable params
    ax.plot([3.0, 5.6], [y0 - 0.42, y0 - 0.42], color=C_LORA_A, lw=1)
    ax.text(4.3, y0 - 0.6, "32,768 trainable params (0.15%)",
            fontsize=7, ha="center", color=C_LORA_A, fontweight="bold")

    # Equals
    ax.text(6.0, y0 + mat_s / 2, "=", fontsize=20, ha="center", va="center",
            color=C_TEXT, fontweight="bold")

    # --- Result: W + ΔW ---
    rx = 6.4
    matrix_box(ax, rx, y0, mat_s, mat_s, "", C_FROZEN, fontsize=16, alpha=0.6)
    overlay = FancyBboxPatch((rx, y0), mat_s, mat_s, boxstyle="round,pad=0.03",
                              facecolor=C_DELTA, edgecolor="none",
                              linewidth=0, alpha=0.3, zorder=2)
    ax.add_patch(overlay)
    ax.text(rx + mat_s / 2, y0 + mat_s / 2 + 0.15, "W + ΔW", ha="center",
            va="center", fontsize=14, color=C_TEXT, fontweight="bold", zorder=3)
    ax.text(rx + mat_s / 2, y0 + mat_s / 2 - 0.15, "adapted", ha="center",
            va="center", fontsize=8, color=C_TEXT, zorder=3)

    ax.set_xlim(-0.2, 8.8)
    ax.set_ylim(-0.85, y0 + mat_s + 1.0)
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