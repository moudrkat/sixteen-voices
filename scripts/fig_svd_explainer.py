#!/usr/bin/env python3
"""Visual explainer: why transplant is lossy — ingredient analogy."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGURES_DIR = Path("figures")

C_CARROLL = "#4C72B0"
C_POE = "#C44E52"
C_SHARED = "#8B5CF6"
C_LOST = "#FFB347"
C_TEXT = "#333333"
C_GRAY = "#cccccc"


def jar(ax, x, y, color, label, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.7, 0.9, boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha))
    ax.text(x + 0.35, y + 0.45, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")


def shelf(ax, x, y, w):
    ax.plot([x, x + w], [y, y], color="#888888", lw=3, solid_capstyle="round")


def bracket(ax, x, y1, y2, label):
    ax.annotate("", xy=(x, y1), xytext=(x, y2),
                arrowprops=dict(arrowstyle="|-|", color="#888888", lw=1.5))
    ax.text(x + 0.15, (y1 + y2) / 2, label, fontsize=10, va="center", color="#888888")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax in axes:
        ax.set_xlim(-0.5, 9)
        ax.set_ylim(-1, 8.5)
        ax.axis("off")

    # ── Panel 1: Carroll's and Poe's recipes ──
    ax = axes[0]
    ax.text(4.25, 8.0, "Carroll's adapter\n8 ingredients, fits perfectly",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_CARROLL)

    shelf(ax, 0.3, 1.5, 8.0)
    ax.text(4.25, 1.0, "shelf space = rank 8", ha="center",
            fontsize=11, color="#888888", style="italic")

    carroll_labels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    for i, label in enumerate(carroll_labels):
        jar(ax, 0.5 + i * 0.95, 1.7, C_CARROLL, label)

    # Mark which ones are "H14"
    ax.add_patch(FancyBboxPatch(
        (0.5 + 6 * 0.95 - 0.1, 2.8), 2.0, 0.5, boxstyle="round,pad=0.05",
        facecolor="none", edgecolor=C_CARROLL, linewidth=2, linestyle="--"))
    ax.text(0.5 + 6.5 * 0.95, 3.55, "H14's\ningredients",
            ha="center", fontsize=10, color=C_CARROLL)

    # Poe below
    ax.text(4.25, 6.5, "Poe's adapter\n8 ingredients, fits perfectly",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_POE)

    shelf(ax, 0.3, 4.2, 8.0)

    poe_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    for i, label in enumerate(poe_labels):
        jar(ax, 0.5 + i * 0.95, 4.4, C_POE, label)

    ax.add_patch(FancyBboxPatch(
        (0.5 + 6 * 0.95 - 0.1, 5.5), 2.0, 0.5, boxstyle="round,pad=0.05",
        facecolor="none", edgecolor=C_POE, linewidth=2, linestyle="--"))
    ax.text(0.5 + 6.5 * 0.95, 6.25, "H14's\ningredients",
            ha="center", fontsize=10, color=C_POE)

    # ── Panel 2: The mix needs more than 8 ──
    ax = axes[1]
    ax.text(4.25, 8.0, "After transplant:\nup to 12 unique ingredients!",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_TEXT)

    shelf(ax, 0.1, 1.5, 8.5)
    ax.text(4.25, 1.0, "shelf space = still only 8!", ha="center",
            fontsize=12, color=C_POE, fontweight="bold")

    # Carroll's non-H14 ingredients (6)
    mix_labels = ["C1", "C2", "C3", "C4", "C5", "C6"]
    mix_colors = [C_CARROLL] * 6
    # Poe's H14 ingredients (2) — might be different from Carroll's
    mix_labels += ["P7", "P8"]
    mix_colors += [C_POE, C_POE]

    for i in range(8):
        jar(ax, 0.5 + i * 0.95, 1.7, mix_colors[i], mix_labels[i])

    # The overflow — doesn't fit!
    overflow_labels = ["C7", "C8", "P1", "P2"]
    ax.text(4.25, 5.0, "These don't fit on the shelf anymore:",
            ha="center", fontsize=13, fontweight="bold", color=C_LOST)

    for i, label in enumerate(overflow_labels):
        c = C_CARROLL if label.startswith("C") else C_POE
        jar(ax, 1.5 + i * 1.5, 3.5, c, label, alpha=0.4)
        ax.text(1.5 + i * 1.5 + 0.35, 3.2, "✗", ha="center",
                fontsize=18, color=C_POE, fontweight="bold")

    ax.text(4.25, 6.5,
            "Mixing two rank-8 sources\ncan need rank 16\nbut LoRA only holds rank 8",
            ha="center", va="center", fontsize=13, color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LOST, alpha=0.2))

    # ── Panel 3: SVD picks the best 8 ──
    ax = axes[2]
    ax.text(4.25, 8.0, "SVD picks the 8\nmost important ones",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_SHARED)

    shelf(ax, 0.1, 1.5, 8.5)
    ax.text(4.25, 1.0, "best rank-8 approximation", ha="center",
            fontsize=11, color="#888888", style="italic")

    # The chosen 8 (mix of both)
    kept = ["C1", "C2", "C4", "C5", "C6", "P7", "P8", "C3"]
    kept_colors = [C_CARROLL, C_CARROLL, C_CARROLL, C_CARROLL, C_CARROLL,
                   C_POE, C_POE, C_CARROLL]
    for i in range(8):
        jar(ax, 0.5 + i * 0.95, 1.7, kept_colors[i], kept[i])

    # What was lost
    lost = ["C7", "C8", "P1", "P2"]
    ax.text(4.25, 5.0, "Lost in compression:",
            ha="center", fontsize=13, fontweight="bold", color=C_LOST)

    for i, label in enumerate(lost):
        c = C_CARROLL if label.startswith("C") else C_POE
        jar(ax, 1.5 + i * 1.5, 3.5, c, label, alpha=0.25)
        ax.plot([1.5 + i*1.5, 1.5 + i*1.5 + 0.7],
                [3.5, 3.5 + 0.9], color=C_POE, lw=2.5)
        ax.plot([1.5 + i*1.5 + 0.7, 1.5 + i*1.5],
                [3.5, 3.5 + 0.9], color=C_POE, lw=2.5)

    ax.text(4.25, 6.5,
            "Mostly fine!\nBut some subtle detail\nis gone forever",
            ha="center", va="center", fontsize=13, color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", alpha=0.5))

    fig.suptitle("Why head transplant loses information",
                 fontsize=20, fontweight="bold", y=0.98, color=C_TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES_DIR / "svd_explainer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()