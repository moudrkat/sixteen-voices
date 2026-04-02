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
        ("f665", "simplicity", RED, 0.9),
        ("f1779", 'first-person "I"', BLUE, 0.7),
        ("f883", "complexity", GREEN, 0.0),
        ("f1777", "dialogue", GREEN, 0.5),
        ("f329", "question marks", ORANGE, 0.3),
        ("f689", "speech attribution", BLUE, 0.0),
        ("f344", "verse line breaks", RED, 0.0),
        ("f627", "conversational verbs", ORANGE, 0.0),
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
    """SAE architecture: encode → TopK → sparse features → decode → reconstruct."""
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

    # ── TopK box ──
    topk_x = 5.0
    box = FancyBboxPatch((topk_x - 0.7, 3.0), 1.4, 2.0,
                          boxstyle="round,pad=0.1", facecolor=ORANGE, alpha=0.2,
                          edgecolor=ORANGE, linewidth=2)
    ax.add_patch(box)
    ax.text(topk_x, 4.2, "TopK", ha="center", fontsize=12, fontweight="bold",
            color=ORANGE)
    ax.text(topk_x, 3.5, "keep 16\nzero rest", ha="center", fontsize=8, color=GRAY)

    # ── Arrow to features ──
    ax.annotate("", xy=(7.0, 4.0), xytext=(5.9, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))

    # ── Sparse features (the key part) ──
    feat_x = 8.5
    box = FancyBboxPatch((feat_x - 1.0, 1.5), 2.0, 5.0,
                          boxstyle="round,pad=0.15", facecolor=GREEN, alpha=0.15,
                          edgecolor=GREEN, linewidth=2)
    ax.add_patch(box)
    ax.text(feat_x, 6.8, "Hidden features", ha="center",
            fontsize=10, fontweight="bold", color=GREEN)

    # Draw individual feature slots — most are zero, only 16 active
    feature_states = [0, 0, 0.8, 0, 0, 0.5, 0, 0, 0, 0, 0.9, 0, 0, 0, 0, 0]
    feat_labels = {2: "f665", 5: "f1779", 10: "f1777"}
    for i, val in enumerate(feature_states):
        y = 6.0 - i * 0.28
        if val > 0:
            ax.barh(y, val * 0.8, height=0.22, left=feat_x - 0.4,
                    color=GREEN, alpha=0.8)
            if i in feat_labels:
                ax.text(feat_x + 0.8, y, feat_labels[i],
                        fontsize=7, color=GREEN, va="center")
        else:
            ax.barh(y, 0.05, height=0.22, left=feat_x - 0.4,
                    color=LIGHT_GRAY)

    ax.text(feat_x, 1.8, "2048 features\nonly 16 active", ha="center", fontsize=9,
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
    ax.text(out_x, 2.8, "x\u0302 \u2248 x", ha="center", fontsize=10, color=GRAY)

    # ── Loss annotation at bottom ──
    ax.text(8.0, 0.5,
            "Loss = MSE(x, x\u0302)\n"
            "TopK enforces sparsity \u2014 no L1 penalty needed",
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
        (2.5, 0.5, "f1779\nfirst-person", BLUE),
        (-0.3, 2.5, "f1777\ndialogue", GREEN),
        (-2.0, -1.5, "f665\nsimplicity", RED),
        (1.0, -2.2, "f329\nquestions", ORANGE),
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
    """Activation steering: two approaches — addition vs clamping."""
    fig, (ax_add, ax_clamp) = plt.subplots(2, 1, figsize=(14, 9),
                                            gridspec_kw={"height_ratios": [1, 1]})

    for ax in (ax_add, ax_clamp):
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 5.5)
        ax.axis("off")

    steps = [
        (1.8, "token\nembedding", BLUE),
        (4.3, "attention\nheads", GREEN),
        (6.8, "MLP", ORANGE),
        (9.3, "layer\nnorm", BLUE),
        (11.8, "next\ntoken", BLUE),
    ]

    def draw_pipeline(ax, y):
        for i, (x, label, color) in enumerate(steps):
            box = FancyBboxPatch((x - 0.7, y - 0.45), 1.4, 0.9,
                                  boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                                  edgecolor=color, linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, y, label, ha="center", va="center", fontsize=9, color=color)
            if i < len(steps) - 1:
                next_x = steps[i + 1][0]
                ax.annotate("", xy=(next_x - 0.8, y), xytext=(x + 0.8, y),
                             arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # ═══════════════════════════════════════
    # TOP: Addition approach
    # ═══════════════════════════════════════
    ax_add.text(0.1, 5.0, "Addition", fontsize=14, fontweight="bold", color=GREEN,
                va="center")
    ax_add.text(2.8, 5.0, "add feature direction to the activation  (preserves original)",
                fontsize=10, color=GRAY, va="center")

    # Normal flow
    y_norm = 3.6
    draw_pipeline(ax_add, y_norm)

    # Steered flow
    y_steer = 1.2
    draw_pipeline(ax_add, y_steer)

    # Injection arrow — between layer norm and next token
    inject_x = 10.55
    ax_add.annotate("", xy=(inject_x, y_steer + 0.5),
                    xytext=(inject_x, y_norm - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.5))
    ax_add.text(inject_x + 0.5, (y_norm + y_steer) / 2,
                "+ scale \u00b7 W_dec[:, f665]",
                fontsize=9, color=RED, fontweight="bold", fontfamily="monospace",
                va="center")

    # Formula
    ax_add.text(0.1, 0.2,
                "x_steered  =  x  +  scale \u00b7 W_dec[:, feature]",
                fontsize=10, color=DARK, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f8f0",
                          edgecolor=GREEN, linewidth=1))

    # ═══════════════════════════════════════
    # BOTTOM: Clamping approach
    # ═══════════════════════════════════════
    ax_clamp.text(0.1, 5.0, "Clamping", fontsize=14, fontweight="bold", color=ORANGE,
                  va="center")
    ax_clamp.text(3.2, 5.0, "encode \u2192 force feature \u2192 decode  (replaces original)",
                  fontsize=10, color=GRAY, va="center")

    # Pipeline up to layer norm
    y_flow = 3.6
    for i, (x, label, color) in enumerate(steps[:4]):
        box = FancyBboxPatch((x - 0.7, y_flow - 0.45), 1.4, 0.9,
                              boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1.5)
        ax_clamp.add_patch(box)
        ax_clamp.text(x, y_flow, label, ha="center", va="center", fontsize=9, color=color)
        if i < 3:
            next_x = steps[i + 1][0]
            ax_clamp.annotate("", xy=(next_x - 0.8, y_flow), xytext=(x + 0.8, y_flow),
                               arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # SAE encode → clamp → decode loop (below the pipeline)
    y_sae = 1.5
    sae_steps = [
        (5.5, "SAE\nencode", ORANGE),
        (7.8, "clamp\nfeature", RED),
        (10.1, "SAE\ndecode", ORANGE),
    ]
    for i, (x, label, color) in enumerate(sae_steps):
        box = FancyBboxPatch((x - 0.65, y_sae - 0.45), 1.3, 0.9,
                              boxstyle="round,pad=0.1", facecolor=color, alpha=0.2,
                              edgecolor=color, linewidth=2)
        ax_clamp.add_patch(box)
        ax_clamp.text(x, y_sae, label, ha="center", va="center", fontsize=9,
                      color=color, fontweight="bold")
        if i < len(sae_steps) - 1:
            next_x = sae_steps[i + 1][0]
            ax_clamp.annotate("", xy=(next_x - 0.75, y_sae),
                               xytext=(x + 0.75, y_sae),
                               arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

    # Arrow down from layer norm to SAE encode
    ln_x = steps[3][0]  # layer norm x
    ax_clamp.annotate("", xy=(sae_steps[0][0], y_sae + 0.5),
                      xytext=(ln_x, y_flow - 0.5),
                      arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))

    # Arrow up from SAE decode to next token
    nt_x = steps[4][0]
    box = FancyBboxPatch((nt_x - 0.7, y_flow - 0.45), 1.4, 0.9,
                          boxstyle="round,pad=0.1", facecolor=BLUE, alpha=0.15,
                          edgecolor=BLUE, linewidth=1.5)
    ax_clamp.add_patch(box)
    ax_clamp.text(nt_x, y_flow, "next\ntoken", ha="center", va="center",
                  fontsize=9, color=BLUE)
    ax_clamp.annotate("", xy=(nt_x, y_flow - 0.5),
                      xytext=(sae_steps[2][0], y_sae + 0.5),
                      arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))

    # Warning
    ax_clamp.text(0.1, 0.2,
                  "\u26a0  Clamping replaces the activation with SAE reconstruction.\n"
                  "    On our 2\u00d7 SAE (0.54 explained variance), this degenerates quickly.",
                  fontsize=9, color=RED,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                            edgecolor=RED, linewidth=1))

    fig.suptitle("Two Approaches to Activation Steering",
                 fontsize=16, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
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