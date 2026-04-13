#!/usr/bin/env python3
"""Explainer figure: How the Sparse Autoencoder (SAE) is trained.

Three panels:
1. Training: collect activations → SAE learns to compress & reconstruct
2. Inside the SAE: encoder (1024→2048), sparsity (only 16 fire), decoder (2048→1024)
3. Result: each decoder column = one direction in activation space

Usage:
    python scripts/fig_sae_training.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_TEXT = "#2C3E50"
C_BG = "white"
C_MODEL = "#4A90D9"
C_SAE = "#9B59B6"
C_ENCODER = "#8E44AD"
C_SPARSE = "#E67E22"
C_DECODER = "#E74C3C"
C_STREAM = "#3498DB"
C_ARROW = "#555555"
C_FAINT = "#AAAAAA"


def rbox(ax, x, y, w, h, label, color, fontsize=14, text_color="white",
         alpha=1.0, lw=2.0, sublabel=None, sublabel_fs=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.3, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)


def arrow_r(ax, x0, y, x1, color=C_ARROW, lw=2.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16))


def arrow_d(ax, x, y0, y1, color=C_ARROW, lw=2.0):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16))


# ═══════════════════════════════════════════════════════════════════
# PANEL 1: Training data collection
# ═══════════════════════════════════════════════════════════════════
def draw_panel1(ax):
    ax.set_title("Step 1: Collect activations", fontsize=20,
                 fontweight="bold", color=C_TEXT, pad=16)

    cx = 5.0

    # Texts
    rbox(ax, cx - 2.5, 7.5, 5.0, 1.3, "Many texts", "#7F8C8D", fontsize=14,
         sublabel="77 authors, thousands of stories")

    arrow_d(ax, cx, 7.5, 6.5)

    # Model
    rbox(ax, cx - 2.5, 5.0, 5.0, 1.3, "TinyStories model", C_MODEL, fontsize=14,
         sublabel="21M params, 1 layer")

    arrow_d(ax, cx, 5.0, 4.0)

    # Residual stream
    rbox(ax, cx - 2.5, 2.5, 5.0, 1.3, "Residual stream", C_STREAM, fontsize=14,
         sublabel="1024 numbers per token")

    arrow_d(ax, cx, 2.5, 1.5)

    # Collected dataset
    rbox(ax, cx - 2.5, 0.0, 5.0, 1.3, "Dataset of vectors", "#27AE60",
         fontsize=14, sublabel="millions of 1024-dim vectors")

    # Side note
    ax.text(cx + 3.2, 3.1,
            "for each token in\neach story, save the\n1024 numbers from\nthe residual stream",
            fontsize=12, color=C_FAINT, ha="left", va="center",
            style="italic", linespacing=1.4)

    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.2, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════
# PANEL 2: Inside the SAE
# ═══════════════════════════════════════════════════════════════════
def draw_panel2(ax):
    ax.set_title("Step 2: Train the SAE", fontsize=20,
                 fontweight="bold", color=C_TEXT, pad=16)

    cx = 5.5

    # Input vector
    rbox(ax, cx - 2.2, 8.0, 4.4, 1.0, "activation vector", C_STREAM,
         fontsize=13, sublabel="1024 numbers")

    arrow_d(ax, cx, 8.0, 7.2)

    # Encoder
    enc_w = 5.0
    rbox(ax, cx - enc_w / 2, 5.8, enc_w, 1.2, "Encoder", C_ENCODER,
         fontsize=15, sublabel="1024 → 2048  (expand)")

    arrow_d(ax, cx, 5.8, 5.0)

    # Sparsity — the key part
    sparse_w = 5.5
    sparse_box = FancyBboxPatch((cx - sparse_w / 2, 3.5), sparse_w, 1.3,
                                boxstyle="round,pad=0.12",
                                facecolor=C_SPARSE, edgecolor="#D35400",
                                linewidth=2.5, zorder=3)
    ax.add_patch(sparse_box)
    ax.text(cx, 4.45, "TopK Sparsity", fontsize=15, color="white",
            ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(cx, 3.9, "keep only top 16 out of 2048", fontsize=12,
            color="white", ha="center", va="center", style="italic",
            alpha=0.9, zorder=4)

    # Illustration of sparsity: row of boxes, most grey, few colored
    box_y = 3.0
    n_boxes = 20
    box_w = 0.42
    box_h = 0.4
    start_x = cx - (n_boxes * box_w) / 2
    active = {3, 7, 12, 16}  # which are "active"
    for i in range(n_boxes):
        bx = start_x + i * box_w
        fc = C_SPARSE if i in active else "#E0E0E0"
        ec = "#D35400" if i in active else "#CCCCCC"
        b = FancyBboxPatch((bx, box_y), box_w - 0.04, box_h,
                           boxstyle="round,pad=0.03",
                           facecolor=fc, edgecolor=ec,
                           linewidth=1.0, zorder=5)
        ax.add_patch(b)
    ax.text(cx, box_y - 0.3,
            "2048 features, only 16 active (colored) at a time",
            fontsize=10, color="#888888", ha="center", va="top",
            style="italic")

    arrow_d(ax, cx, box_y - 0.1, box_y - 0.7)

    # Decoder
    dec_y = 1.3
    rbox(ax, cx - enc_w / 2, dec_y, enc_w, 1.2, "Decoder", C_DECODER,
         fontsize=15, sublabel="2048 → 1024  (reconstruct)")

    arrow_d(ax, cx, dec_y, dec_y - 0.8)

    # Output = reconstruction
    rbox(ax, cx - 2.2, -0.7, 4.4, 1.0, "reconstruction", C_STREAM,
         fontsize=13, sublabel="should match input!", alpha=0.7)

    # Loss arrow
    ax.annotate("",
                xy=(cx + 2.8, 8.5),
                xytext=(cx + 2.8, -0.2),
                arrowprops=dict(arrowstyle="<->", color="#CC0000",
                                lw=2.0, mutation_scale=14,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(cx + 4.5, 4.5,
            "loss =\nhow different\nare these?\n\n(must be\nsmall!)",
            fontsize=12, color="#CC0000", ha="center", va="center",
            fontweight="bold", linespacing=1.4)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════
# PANEL 3: What you get — decoder columns = directions
# ═══════════════════════════════════════════════════════════════════
def draw_panel3(ax):
    ax.set_title("Step 3: Decoder columns = directions", fontsize=20,
                 fontweight="bold", color=C_TEXT, pad=16)

    cx = 5.0

    # Decoder matrix
    mat_x = cx - 3.5
    mat_y = 4.5
    mat_w = 7.0
    mat_h = 3.5

    mat_bg = FancyBboxPatch((mat_x, mat_y), mat_w, mat_h,
                            boxstyle="round,pad=0.15",
                            facecolor="#F5F0FA", edgecolor=C_DECODER,
                            linewidth=2.0, zorder=2)
    ax.add_patch(mat_bg)

    ax.text(cx, mat_y + mat_h - 0.4,
            "Decoder weight matrix", fontsize=14, color=C_DECODER,
            ha="center", va="top", fontweight="bold")
    ax.text(cx, mat_y + mat_h - 0.85,
            "2048 rows  x  1024 columns", fontsize=12, color="#888888",
            ha="center", va="top", style="italic")

    # Draw grid of "rows" — highlight a few
    n_rows = 8
    row_h = 0.28
    row_w = 5.5
    start_y = mat_y + mat_h - 1.4
    start_x = cx - row_w / 2

    labels = {
        0: ("row 665", '"simplicity"', C_DECODER),
        2: ("row 1777", '"dialogue"', "#E67E22"),
        5: ("row 329", '"questions"', "#2980B9"),
    }

    for i in range(n_rows):
        ry = start_y - i * (row_h + 0.06)
        if i in labels:
            name, feat, color = labels[i]
            fc = color
            alpha = 0.85
        else:
            fc = "#D5D5D5"
            alpha = 0.4
            name, feat, color = None, None, None

        row_box = FancyBboxPatch((start_x, ry), row_w, row_h,
                                 boxstyle="round,pad=0.03",
                                 facecolor=fc, edgecolor="#999999",
                                 linewidth=0.8, alpha=alpha, zorder=3)
        ax.add_patch(row_box)

        if name:
            ax.text(start_x + row_w + 0.3, ry + row_h / 2,
                    f"{name} → {feat}",
                    fontsize=11, color=color, ha="left", va="center",
                    fontweight="bold")

    # "..." between some rows
    dots_y = start_y - 3 * (row_h + 0.06) - 0.05
    ax.text(cx, dots_y, "...", fontsize=16, color="#999999",
            ha="center", va="center")

    # Arrow from highlighted row to direction vector
    row665_y = start_y - 0 * (row_h + 0.06) + row_h / 2

    # Direction vector below
    dir_y = 1.5
    rbox(ax, cx - 2.5, dir_y, 5.0, 1.3, "direction vector", C_DECODER,
         fontsize=14, sublabel="1024 numbers → use for steering!")

    ax.annotate("",
                xy=(cx, dir_y + 1.3),
                xytext=(cx, mat_y - 0.1),
                arrowprops=dict(arrowstyle="-|>", color=C_DECODER,
                                lw=2.5, mutation_scale=18))

    ax.text(cx + 3.2, dir_y + 2.0,
            "each row is one\nfeature's direction\n\nfound automatically\nno labels needed!",
            fontsize=12, color="#888888", ha="left", va="center",
            style="italic", linespacing=1.4)

    # Bottom note
    ax.text(cx, 0.5,
            "f665 wasn't trained to be \"simplicity\" — the SAE found it on its own.\n"
            "We checked what it fires on: short, simple sentences.",
            fontsize=12, color="#666666", ha="center", va="top",
            style="italic", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5",
                      ec="#DDDDDD", lw=1.0))

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.8, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/sae_training.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(33, 11))
    fig.patch.set_facecolor(C_BG)

    fig.suptitle("How I trained a Sparse Autoencoder (SAE) to find directions",
                 fontsize=24, fontweight="bold", color=C_TEXT, y=0.99)
    fig.text(0.5, 0.94,
             "Unsupervised — no labels, no assumptions. The SAE finds interpretable directions on its own.",
             fontsize=14, color="#888888", ha="center", style="italic")

    draw_panel1(axes[0])
    draw_panel2(axes[1])
    draw_panel3(axes[2])

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()