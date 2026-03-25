#!/usr/bin/env python3
"""Generate explanatory figures for the SAE explainer document.

Produces:
1. sae_explainer_architecture.png — SAE encode/decode/sparsity diagram
2. sae_explainer_features.png — what "features" mean visually
3. sae_explainer_superposition.png — why we need SAE (superposition problem)

Usage:
    uv run python scripts/fig_sae_explainer.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Palette consistent with project
BLUE = "#4c72b0"
GREEN = "#55a868"
RED = "#c44e52"
ORANGE = "#e8a735"
GRAY = "#999999"
LIGHT_GRAY = "#eeeeee"
DARK = "#333333"


def fig_superposition():
    """Why we need SAE: the superposition problem.

    Left: raw residual stream dimensions — everything tangled.
    Right: SAE features — each captures one concept.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Superposition / tangled dimensions ──
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-0.5, 6)
    ax1.axis("off")
    ax1.set_title("Raw residual stream", fontsize=13, fontweight="bold", pad=10)

    # Draw tangled dimension bars
    n_dims = 8
    concepts = ["dialogue?", "formality?", "narrative?", "warmth?",
                "speech?", "complexity?", "???", "???"]
    for i in range(n_dims):
        y = 5 - i * 0.65
        # Each dimension is a mix — show with multiple colors
        colors = [BLUE, GREEN, RED, ORANGE]
        np.random.seed(i + 42)
        widths = np.random.dirichlet([1, 1, 1, 1]) * 8
        x = 1.0
        for w, c in zip(widths, colors):
            ax1.barh(y, w, height=0.45, left=x, color=c, alpha=0.4, edgecolor="white")
            x += w
        ax1.text(0.8, y, f"d{i}", ha="right", va="center", fontsize=9, color=GRAY)
        ax1.text(9.2, y, concepts[i], ha="left", va="center", fontsize=8,
                 color=GRAY, style="italic")

    ax1.text(5.0, -0.3, "Each dimension mixes multiple concepts",
             ha="center", fontsize=10, color=RED, style="italic")

    # ── Right: Clean SAE features ──
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-0.5, 6)
    ax2.axis("off")
    ax2.set_title("SAE features", fontsize=13, fontweight="bold", pad=10)

    features = [
        ("f68", "direct speech", BLUE, 0.9),
        ("f198", "structured narration", ORANGE, 0.7),
        ("f33", "clause complexity", GREEN, 0.0),
        ("f147", "warm / domestic", RED, 0.5),
        ("f113", "short sentences", BLUE, 0.3),
        ("f122", "speech attribution", GREEN, 0.0),
        ("f144", "embedded clauses", ORANGE, 0.0),
        ("f160", "referential continuity", RED, 0.0),
    ]

    for i, (fname, label, color, activation) in enumerate(features):
        y = 5 - i * 0.65
        # Single clean bar — most are zero (sparse!)
        if activation > 0:
            ax2.barh(y, activation * 8, height=0.45, left=1.0, color=color,
                     alpha=0.7, edgecolor="white")
        else:
            ax2.barh(y, 0.15, height=0.45, left=1.0, color=LIGHT_GRAY, edgecolor="white")

        ax2.text(0.8, y, fname, ha="right", va="center", fontsize=9, color=DARK)
        ax2.text(9.2, y, label, ha="left", va="center", fontsize=8,
                 color=color, fontweight="bold" if activation > 0 else "normal")

    ax2.text(5.0, -0.3, "Each feature = one interpretable concept, most are off (sparse)",
             ha="center", fontsize=10, color=GREEN, style="italic")

    fig.suptitle("The Superposition Problem: Why We Need SAEs",
                 fontsize=15, fontweight="bold", y=1.02)

    fig.tight_layout()
    out = FIGURES_DIR / "sae_explainer_superposition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_architecture():
    """SAE architecture: encode → ReLU → sparse features → decode → reconstruct."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Input: residual stream vector ──
    input_x, input_y = 1.5, 4.0
    box = FancyBboxPatch((input_x - 0.8, input_y - 1.5), 1.6, 3.0,
                          boxstyle="round,pad=0.15", facecolor=BLUE, alpha=0.15,
                          edgecolor=BLUE, linewidth=2)
    ax.add_patch(box)
    ax.text(input_x, input_y + 1.8, "Input\n(residual stream)", ha="center",
            fontsize=10, fontweight="bold", color=BLUE)
    ax.text(input_x, input_y, "1024\ndims", ha="center", fontsize=11, color=BLUE)
    ax.text(input_x, input_y - 1.1, "all active", ha="center", fontsize=8,
            color=GRAY, style="italic")

    # ── Arrow: encode ──
    ax.annotate("", xy=(4.0, 4.0), xytext=(2.5, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
    ax.text(3.25, 4.5, "W_enc · x + b", ha="center", fontsize=9, color=DARK,
            family="monospace")

    # ── ReLU box ──
    relu_x = 5.0
    box = FancyBboxPatch((relu_x - 0.6, 3.0), 1.2, 2.0,
                          boxstyle="round,pad=0.1", facecolor=ORANGE, alpha=0.2,
                          edgecolor=ORANGE, linewidth=2)
    ax.add_patch(box)
    ax.text(relu_x, 4.0, "ReLU", ha="center", fontsize=12, fontweight="bold",
            color=ORANGE)
    ax.text(relu_x, 3.4, "kill\nnegatives", ha="center", fontsize=8, color=GRAY)

    # ── Arrow to features ──
    ax.annotate("", xy=(7.0, 4.0), xytext=(5.8, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))

    # ── Sparse features (the key part) ──
    feat_x = 8.5
    box = FancyBboxPatch((feat_x - 1.0, 1.5), 2.0, 5.0,
                          boxstyle="round,pad=0.15", facecolor=GREEN, alpha=0.15,
                          edgecolor=GREEN, linewidth=2)
    ax.add_patch(box)
    ax.text(feat_x, 6.8, "Hidden features", ha="center",
            fontsize=10, fontweight="bold", color=GREEN)

    # Draw individual feature slots — most are zero
    feature_states = [0, 0, 0.8, 0, 0, 0.5, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0]
    for i, val in enumerate(feature_states):
        y = 6.0 - i * 0.28
        if val > 0:
            ax.barh(y, val * 0.8, height=0.22, left=feat_x - 0.4,
                    color=GREEN, alpha=0.8)
            ax.text(feat_x + 0.8, y, f"f{[33, 68, 198][min(i // 5, 2)]}",
                    fontsize=7, color=GREEN, va="center")
        else:
            ax.barh(y, 0.05, height=0.22, left=feat_x - 0.4,
                    color=LIGHT_GRAY)

    ax.text(feat_x, 1.8, "256 features\nmost = 0", ha="center", fontsize=9,
            color=GREEN, style="italic")

    # ── Arrow: decode ──
    ax.annotate("", xy=(11.5, 4.0), xytext=(9.8, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
    ax.text(10.65, 4.5, "W_dec · h + b", ha="center", fontsize=9, color=DARK,
            family="monospace")

    # ── Output: reconstructed vector ──
    out_x = 13.0
    box = FancyBboxPatch((out_x - 0.8, 2.5), 1.6, 3.0,
                          boxstyle="round,pad=0.15", facecolor=BLUE, alpha=0.15,
                          edgecolor=BLUE, linewidth=2, linestyle="--")
    ax.add_patch(box)
    ax.text(out_x, 5.8, "Reconstruction", ha="center",
            fontsize=10, fontweight="bold", color=BLUE)
    ax.text(out_x, 4.0, "1024\ndims", ha="center", fontsize=11, color=BLUE)
    ax.text(out_x, 2.8, "x̂ ≈ x", ha="center", fontsize=10, color=GRAY)

    # ── Loss annotation at bottom ──
    ax.text(8.0, 0.5,
            "Loss = MSE(x, x̂)  +  λ · L1(h)\n"
            "        ↑ reconstruct well       ↑ keep features sparse",
            ha="center", fontsize=10, color=DARK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor=GRAY, linewidth=1))

    fig.suptitle("Sparse Autoencoder Architecture",
                 fontsize=15, fontweight="bold", y=0.97)

    out = FIGURES_DIR / "sae_explainer_architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_features_meaning():
    """What SAE features mean: directions in activation space.

    Shows a 2D projection where each feature is a direction,
    and tokens/authors land at different positions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Features as directions ──
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect("equal")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Features = directions in activation space", fontsize=12,
                  fontweight="bold", pad=10)

    # Origin crosshairs
    ax1.axhline(0, color=LIGHT_GRAY, linewidth=0.5)
    ax1.axvline(0, color=LIGHT_GRAY, linewidth=0.5)

    # Feature directions as arrows from origin
    directions = [
        (2.5, 0.5, "f68\ndirect speech", BLUE),
        (-0.3, 2.5, "f198\nnarration", ORANGE),
        (-2.0, -1.5, "f113\nshort sentences", RED),
        (1.0, -2.2, "f147\nwarm/cozy", GREEN),
    ]
    for dx, dy, label, color in directions:
        ax1.annotate("", xy=(dx, dy), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
        # Place label at arrow tip
        offset_x = 0.15 if dx >= 0 else -0.15
        offset_y = 0.15 if dy >= 0 else -0.15
        ax1.text(dx + offset_x, dy + offset_y, label, fontsize=8,
                 color=color, fontweight="bold",
                 ha="left" if dx >= 0 else "right",
                 va="bottom" if dy >= 0 else "top")

    ax1.set_xlabel("(projected dimension 1)", fontsize=9, color=GRAY)
    ax1.set_ylabel("(projected dimension 2)", fontsize=9, color=GRAY)

    # ── Right: Authors projected onto feature space ──
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect("equal")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Authors land where their features fire", fontsize=12,
                  fontweight="bold", pad=10)

    ax2.axhline(0, color=LIGHT_GRAY, linewidth=0.5)
    ax2.axvline(0, color=LIGHT_GRAY, linewidth=0.5)

    # Subtle feature arrows in background
    for dx, dy, _, color in directions:
        ax2.annotate("", xy=(dx * 0.7, dy * 0.7), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1, alpha=0.2))

    # Author positions (hand-placed for illustration)
    authors = [
        (2.0, 1.8, "dialogue", BLUE),
        (1.5, 0.3, "carroll", BLUE),
        (-0.5, 2.2, "grimm", ORANGE),
        (-0.8, 1.8, "harris", ORANGE),
        (-1.8, -1.0, "minimalist", RED),
        (-1.5, -1.5, "reporter", RED),
        (0.8, -1.8, "cozy", GREEN),
        (0.3, -1.3, "indian", GREEN),
        (0.2, 0.3, "base model", GRAY),
    ]
    for x, y, name, color in authors:
        ax2.scatter(x, y, c=color, s=80, edgecolors="white", linewidth=0.8, zorder=3)
        ax2.text(x + 0.15, y + 0.15, name, fontsize=8, color=color)

    ax2.set_xlabel("(projected dimension 1)", fontsize=9, color=GRAY)
    ax2.set_ylabel("(projected dimension 2)", fontsize=9, color=GRAY)

    fig.suptitle("What SAE Features Mean",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "sae_explainer_features.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_steering():
    """Activation steering: adding feature directions during generation."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ── Normal generation flow ──
    y_normal = 4.5
    ax.text(0.5, y_normal, "Normal:", fontsize=11, fontweight="bold", color=DARK)

    steps = [
        (2.5, "token\nembedding", BLUE),
        (5.0, "attention\nheads", GREEN),
        (7.5, "residual\nstream", BLUE),
        (10.0, "MLP", ORANGE),
        (12.5, "next\ntoken", BLUE),
    ]
    for i, (x, label, color) in enumerate(steps):
        box = FancyBboxPatch((x - 0.7, y_normal - 0.5), 1.4, 1.0,
                              boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_normal, label, ha="center", va="center", fontsize=8, color=color)
        if i < len(steps) - 1:
            next_x = steps[i + 1][0]
            ax.annotate("", xy=(next_x - 0.8, y_normal), xytext=(x + 0.8, y_normal),
                         arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # ── Steered generation flow ──
    y_steer = 1.8
    ax.text(0.5, y_steer, "Steered:", fontsize=11, fontweight="bold", color=RED)

    for i, (x, label, color) in enumerate(steps):
        box = FancyBboxPatch((x - 0.7, y_steer - 0.5), 1.4, 1.0,
                              boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y_steer, label, ha="center", va="center", fontsize=8, color=color)
        if i < len(steps) - 1:
            next_x = steps[i + 1][0]
            ax.annotate("", xy=(next_x - 0.8, y_steer), xytext=(x + 0.8, y_steer),
                         arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # ── The steering injection ──
    inject_x = 7.5
    inject_y_top = y_steer + 0.5
    ax.annotate("", xy=(inject_x, inject_y_top),
                xytext=(inject_x, inject_y_top + 1.2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))

    ax.text(inject_x, inject_y_top + 1.5, "+ feature direction",
            ha="center", fontsize=10, fontweight="bold", color=RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=RED, alpha=0.1,
                      edgecolor=RED, linewidth=1.5))

    ax.text(inject_x, inject_y_top + 2.5,
            "e.g. add f68 (direct speech)\n→ model generates more \"I\", \"you\", dialogue",
            ha="center", fontsize=9, color=GRAY, style="italic")

    fig.suptitle("Activation Steering: Nudging the Model with SAE Features",
                 fontsize=14, fontweight="bold", y=0.98)

    out = FIGURES_DIR / "sae_explainer_steering.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    fig_superposition()
    fig_architecture()
    fig_features_meaning()
    fig_steering()
    print("\nDone — all explainer figures generated.")


if __name__ == "__main__":
    main()