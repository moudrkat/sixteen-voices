#!/usr/bin/env python3
"""Generate the model internals summary figure.

Shows the three head clusters, MLP, and what we can/can't explain.

Usage:
    uv run python scripts/fig_sae_heads_roles.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Colors
BLUE = "#4c72b0"
GREEN = "#55a868"
RED = "#c44e52"
ORANGE = "#e8a735"
GRAY = "#999999"
DARK = "#333333"
LIGHT = "#f0f0f0"


def main():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # ── TITLE ──
    ax.text(8, 11.6, "How a 1-Layer Transformer Computes Style",
            ha="center", fontsize=18, fontweight="bold", color=DARK)
    ax.text(8, 11.15, "16 heads, 1 MLP — what we can and can't explain",
            ha="center", fontsize=12, color=GRAY)

    # ═══════════════════════════════════════════════════
    # CLUSTER 1: Register readers (H3, H14, H15, H2)
    # ═══════════════════════════════════════════════════
    y_c1 = 9.0

    # Cluster background
    box = FancyBboxPatch((0.3, y_c1 - 1.3), 15.4, 2.6,
                          boxstyle="round,pad=0.2", facecolor=GREEN,
                          alpha=0.05, edgecolor=GREEN, linewidth=1.5,
                          linestyle="--")
    ax.add_patch(box)
    ax.text(0.6, y_c1 + 1.1, "CLUSTER 1 — Register readers",
            fontsize=11, fontweight="bold", color=GREEN)
    ax.text(4.5, y_c1 + 1.1,
            "30–50% shared features, all grounded in conversational verb density",
            fontsize=9, color=GRAY)

    # H3
    box = FancyBboxPatch((0.5, y_c1 - 1.0), 3.5, 1.8,
                          boxstyle="round,pad=0.1", facecolor=GREEN,
                          alpha=0.12, edgecolor=GREEN, linewidth=2)
    ax.add_patch(box)
    ax.text(2.25, y_c1 + 0.5, "H3 — Style Reader", ha="center",
            fontsize=11, fontweight="bold", color=GREEN)
    ax.text(2.25, y_c1 + 0.05, "81 BH features (broadest)", ha="center",
            fontsize=9, color=DARK)
    ax.text(2.25, y_c1 - 0.35, "conv verbs, pronouns,\nsentence markers, questions",
            ha="center", fontsize=8, color="#555")
    ax.text(2.25, y_c1 - 0.85, "conv_pct r=−0.49***\nword_len r=+0.42***",
            ha="center", fontsize=7.5, color=GREEN, fontfamily="monospace")

    # H14
    box = FancyBboxPatch((4.3, y_c1 - 1.0), 3.8, 1.8,
                          boxstyle="round,pad=0.1", facecolor=RED,
                          alpha=0.12, edgecolor=RED, linewidth=2)
    ax.add_patch(box)
    ax.text(6.2, y_c1 + 0.5, 'H14 — Narrator Dial', ha="center",
            fontsize=11, fontweight="bold", color=RED)
    ax.text(6.2, y_c1 + 0.05, "12 BH features, dominant for 18 authors",
            ha="center", fontsize=9, color=DARK)
    ax.text(6.2, y_c1 - 0.35,
            'suppresses "I", conv verbs\namplifies rare vocabulary',
            ha="center", fontsize=8, color="#555")
    ax.text(6.2, y_c1 - 0.85, "i_pct r=−0.31**\nconv_pct r=−0.39***",
            ha="center", fontsize=7.5, color=RED, fontfamily="monospace")

    # H15
    box = FancyBboxPatch((8.4, y_c1 - 1.0), 3.3, 1.8,
                          boxstyle="round,pad=0.1", facecolor=GREEN,
                          alpha=0.08, edgecolor=GREEN, linewidth=1.5)
    ax.add_patch(box)
    ax.text(10.05, y_c1 + 0.5, "H15", ha="center",
            fontsize=11, fontweight="bold", color=GREEN)
    ax.text(10.05, y_c1 + 0.05, "35 BH, redundant", ha="center",
            fontsize=9, color=DARK)
    ax.text(10.05, y_c1 - 0.4, "conv_pct r=−0.33**", ha="center",
            fontsize=7.5, color=GREEN, fontfamily="monospace")

    # H2
    box = FancyBboxPatch((12.0, y_c1 - 1.0), 3.3, 1.8,
                          boxstyle="round,pad=0.1", facecolor=GREEN,
                          alpha=0.08, edgecolor=GREEN, linewidth=1.5)
    ax.add_patch(box)
    ax.text(13.65, y_c1 + 0.5, "H2", ha="center",
            fontsize=11, fontweight="bold", color=GREEN)
    ax.text(13.65, y_c1 + 0.05, "59 BH, redundant", ha="center",
            fontsize=9, color=DARK)
    ax.text(13.65, y_c1 - 0.4, "conv_pct r=−0.32**", ha="center",
            fontsize=7.5, color=GREEN, fontfamily="monospace")

    # ═══════════════════════════════════════════════════
    # H11: Isolated
    # ═══════════════════════════════════════════════════
    y_h11 = 5.8

    box = FancyBboxPatch((0.5, y_h11 - 1.0), 7.0, 2.0,
                          boxstyle="round,pad=0.15", facecolor=BLUE,
                          alpha=0.12, edgecolor=BLUE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(4.0, y_h11 + 0.7, "H11 — Workhorse (isolated)",
            ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(4.0, y_h11 + 0.2, "Dominant for 51/77 authors  •  1 BH feature",
            ha="center", fontsize=10, color=DARK)
    ax.text(4.0, y_h11 - 0.25,
            "Zero feature overlap with any other head",
            ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(4.0, y_h11 - 0.65,
            "No text property predicts it (all p > 0.1)\n"
            "SAE sees storytelling patterns — but we can't ground them",
            ha="center", fontsize=8, color="#555", linespacing=1.4)

    # ═══════════════════════════════════════════════════
    # CLUSTER 2: Idiosyncratic (H0, H4, H8, H9, H12)
    # ═══════════════════════════════════════════════════
    y_c2 = 5.8

    box = FancyBboxPatch((8.3, y_c2 - 1.0), 7.2, 2.0,
                          boxstyle="round,pad=0.15", facecolor=ORANGE,
                          alpha=0.06, edgecolor=ORANGE, linewidth=1.5,
                          linestyle="--")
    ax.add_patch(box)
    ax.text(11.9, y_c2 + 0.7, "CLUSTER 2 — Idiosyncratic style",
            ha="center", fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(11.9, y_c2 + 0.2, "H0, H4, H8, H9, H12  •  20–30% shared features",
            ha="center", fontsize=9, color=DARK)
    ax.text(11.9, y_c2 - 0.25,
            "Harris tops nearly all of them",
            ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    ax.text(11.9, y_c2 - 0.65,
            "Weakly grounded in word length, excl. marks\n"
            "Dialectal / eccentric patterns",
            ha="center", fontsize=8, color="#555", linespacing=1.4)

    # ═══════════════════════════════════════════════════
    # MLP
    # ═══════════════════════════════════════════════════
    y_mlp = 3.0

    box = FancyBboxPatch((0.5, y_mlp - 0.8), 7.0, 1.6,
                          boxstyle="round,pad=0.15", facecolor=ORANGE,
                          alpha=0.12, edgecolor=ORANGE, linewidth=2)
    ax.add_patch(box)
    ax.text(4.0, y_mlp + 0.5, "MLP — Emergent axes",
            ha="center", fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(4.0, y_mlp, "27 head-independent features (e.g. f665 simplicity)",
            ha="center", fontsize=9, color=DARK)
    ax.text(4.0, y_mlp - 0.45,
            "No single head controls them  •  weight steering can't reach them\n"
            "Activation steering works (100% win rate for simplicity)",
            ha="center", fontsize=8, color="#555", linespacing=1.4)

    # ═══════════════════════════════════════════════════
    # Minor heads
    # ═══════════════════════════════════════════════════
    box = FancyBboxPatch((8.3, y_mlp - 0.8), 7.2, 1.6,
                          boxstyle="round,pad=0.15", facecolor=LIGHT,
                          alpha=0.5, edgecolor=GRAY, linewidth=1)
    ax.add_patch(box)
    ax.text(11.9, y_mlp + 0.5, "Minor heads",
            ha="center", fontsize=11, fontweight="bold", color=GRAY)
    ax.text(11.9, y_mlp, "H1, H5, H6, H7, H10, H13",
            ha="center", fontsize=9, color=DARK)
    ax.text(11.9, y_mlp - 0.45,
            "0–1 BH features each, not dominant for any author group\n"
            "H13 weakly reads conv verbs (r = −0.28*)",
            ha="center", fontsize=8, color="#555", linespacing=1.4)

    # ═══════════════════════════════════════════════════
    # Bottom: what we can/can't explain
    # ═══════════════════════════════════════════════════
    y_bot = 1.0

    # Grounded
    box = FancyBboxPatch((0.5, y_bot - 0.5), 7.0, 1.0,
                          boxstyle="round,pad=0.1", facecolor="#e8f5e9",
                          edgecolor=GREEN, linewidth=1.5)
    ax.add_patch(box)
    ax.text(4.0, y_bot + 0.2, "✓  What we can explain",
            ha="center", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(4.0, y_bot - 0.2,
            "H14 (end-to-end: features → text stats → authors)  •  "
            "MLP simplicity axis  •  steering structural features",
            ha="center", fontsize=8, color=DARK)

    # Ungrounded
    box = FancyBboxPatch((8.3, y_bot - 0.5), 7.2, 1.0,
                          boxstyle="round,pad=0.1", facecolor="#fde0dc",
                          edgecolor=RED, linewidth=1.5)
    ax.add_patch(box)
    ax.text(11.9, y_bot + 0.2, "?  What we can't explain",
            ha="center", fontsize=10, fontweight="bold", color=RED)
    ax.text(11.9, y_bot - 0.2,
            "H11 (dominant but ungrounded)  •  "
            "author identity  •  semantic feature steering",
            ha="center", fontsize=8, color=DARK)

    fig.savefig(FIGURES_DIR / "sae_heads_roles.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_heads_roles.png'}")


if __name__ == "__main__":
    main()