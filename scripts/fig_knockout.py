#!/usr/bin/env python3
"""Draw head knockout explanation: ΔW row-sliced into 16 head blocks, then knockout.

Usage:
    python scripts/fig_knockout.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_DELTA = "#E74C3C"
C_TEXT = "#2C3E50"
C_ARROW = "#555555"
C_HEAD_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabebe", "#469990", "#e6beff",
    "#9A6324", "#800000", "#aaffc3", "#808000",
]


def matrix_box(ax, x, y, w, h, label, color, fontsize=9, alpha=1.0,
               sublabel=None, sublabel_size=6.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=color, edgecolor="#333333",
                         linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.18, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color="white", zorder=3, alpha=0.9)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=3)


def head_block(ax, x, y, w, h, idx, color, alpha=0.8, zeroed=False):
    """Draw one head block (64 rows)."""
    fc = "#DDDDDD" if zeroed else color
    a = 0.25 if zeroed else alpha
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor="#555555",
        linewidth=0.6, alpha=a, zorder=2)
    ax.add_patch(box)
    # Head label inside block
    if not zeroed:
        ax.text(x + w / 2, y + h / 2, f"H{idx}",
                ha="center", va="center", fontsize=6,
                fontweight="bold", color="white", zorder=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="figures/knockout.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    y0 = 0.5
    mat_h = 5.0
    mat_w = 3.5
    block_h = mat_h / 16       # each head block
    block_gap = block_h * 0.08  # gap between blocks

    # Title
    ax.text(6.0, y0 + mat_h + 0.8, "Per-Head Knockout Experiment",
            fontsize=15, ha="center", va="center", color=C_TEXT, fontweight="bold")

    # ============ STEP 1: Full ΔW matrix ============
    s1_x = 0.5
    matrix_box(ax, s1_x, y0, mat_w, mat_h, "ΔW", C_DELTA, fontsize=20,
               sublabel="1024 × 1024", alpha=0.85)

    # Dimension annotations on ΔW
    ax.annotate("", xy=(s1_x - 0.15, y0), xytext=(s1_x - 0.15, y0 + mat_h),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8))
    ax.text(s1_x - 0.45, y0 + mat_h / 2, "1024\nrows", fontsize=7,
            ha="center", va="center", color="#888", rotation=90)

    ax.annotate("", xy=(s1_x, y0 - 0.12), xytext=(s1_x + mat_w, y0 - 0.12),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8))
    ax.text(s1_x + mat_w / 2, y0 - 0.35, "1024 columns", fontsize=7,
            ha="center", color="#888")

    # ============ Arrow 1 ============
    arr1_x = s1_x + mat_w + 0.5
    ax.text(arr1_x, y0 + mat_h / 2, "→", fontsize=26, ha="center", va="center",
            color=C_ARROW)
    ax.text(arr1_x, y0 + mat_h / 2 - 0.5, "slice into\n16 blocks\nof 64 rows",
            fontsize=7, ha="center", color="#888", style="italic")

    # ============ STEP 2: Sliced into 16 head blocks ============
    s2_x = arr1_x + 0.8
    s2_w = mat_w

    for i in range(16):
        by = y0 + i * block_h
        head_block(ax, s2_x, by + block_gap / 2,
                   s2_w, block_h - block_gap, i, C_HEAD_COLORS[i])

        # Row range label on the right
        row_start = i * 64
        row_end = row_start + 63
        ax.text(s2_x + s2_w + 0.1, by + block_h / 2,
                f"rows {row_start}–{row_end}",
                fontsize=5, va="center", color="#888", family="monospace")

    # Bracket showing one head = 64 rows
    bracket_i = 10
    by = y0 + bracket_i * block_h
    ax.annotate("", xy=(s2_x + s2_w + 1.5, by + block_gap / 2),
                xytext=(s2_x + s2_w + 1.5, by + block_h - block_gap / 2),
                arrowprops=dict(arrowstyle="<->", color=C_TEXT, lw=1))
    ax.text(s2_x + s2_w + 1.7, by + block_h / 2, "64\nrows",
            fontsize=7, va="center", color=C_TEXT, fontweight="bold")

    # Full height annotation
    ax.annotate("", xy=(s2_x - 0.15, y0), xytext=(s2_x - 0.15, y0 + mat_h),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=0.8))
    ax.text(s2_x - 0.45, y0 + mat_h / 2, "16 × 64\n= 1024", fontsize=7,
            ha="center", va="center", color="#888", rotation=90)

    # Title for sliced view
    ax.text(s2_x + s2_w / 2, y0 + mat_h + 0.3,
            "ΔW split by head  (each block: 64 × 1024)",
            fontsize=9, ha="center", color=C_TEXT, fontweight="bold")

    # ============ Arrow 2 ============
    arr2_x = s2_x + s2_w + 2.3
    ax.text(arr2_x, y0 + mat_h / 2, "→", fontsize=26, ha="center", va="center",
            color=C_ARROW)
    ax.text(arr2_x, y0 + mat_h / 2 - 0.5, "zero one\nhead block",
            fontsize=7, ha="center", color="#888", style="italic")

    # ============ STEP 3: Knockout ============
    s3_x = arr2_x + 0.8
    ko_head = 5

    for i in range(16):
        by = y0 + i * block_h
        zeroed = (i == ko_head)
        head_block(ax, s3_x, by + block_gap / 2,
                   s2_w, block_h - block_gap, i, C_HEAD_COLORS[i],
                   zeroed=zeroed)

    # X on knocked-out head
    ky = y0 + ko_head * block_h
    ax.plot([s3_x, s3_x + s2_w],
            [ky + block_gap / 2, ky + block_h - block_gap / 2],
            color="#CC0000", lw=2.5, zorder=4)
    ax.plot([s3_x, s3_x + s2_w],
            [ky + block_h - block_gap / 2, ky + block_gap / 2],
            color="#CC0000", lw=2.5, zorder=4)

    # Knockout label
    ax.annotate(f"H{ko_head} zeroed\n(64 rows → 0)",
                xy=(s3_x + s2_w + 0.05, ky + block_h / 2),
                xytext=(s3_x + s2_w + 0.6, ky + block_h / 2),
                fontsize=8, color="#CC0000", fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1))

    # Title for knockout
    ax.text(s3_x + s2_w / 2, y0 + mat_h + 0.3, "Head knockout",
            fontsize=9, ha="center", color=C_TEXT, fontweight="bold")

    # Result box at bottom
    ax.text(s3_x + s2_w / 2, y0 - 0.6,
            "measure Δ perplexity\n→ head importance",
            fontsize=9, ha="center", color=C_TEXT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                      edgecolor="#F39C12", linewidth=1.2))

    # Axis
    ax.set_xlim(-0.7, s3_x + s2_w + 2.5)
    ax.set_ylim(-1.0, y0 + mat_h + 1.2)
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
