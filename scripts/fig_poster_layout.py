#!/usr/bin/env python3
"""Generate the full ML Prague poster as a single image.

Seven panels telling the complete story from setup to findings.

Usage:
    uv run python scripts/fig_poster_layout.py
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def add_panel_box(ax, color="#333", alpha=0.03):
    """Add a subtle background box to a panel."""
    box = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                          boxstyle="round,pad=0.02",
                          facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=0.5)
    ax.add_patch(box)


def panel_setup(ax):
    """Panel 1: The setup."""
    ax.axis("off")
    add_panel_box(ax, "#4c72b0")

    ax.text(0.05, 0.95, "1. The Setup", fontsize=14, fontweight="bold",
            color="#4c72b0", transform=ax.transAxes, va="top")

    text = (
        "TinyStories-1Layer-21M\n"
        "21M parameters · 1 layer · 16 heads\n"
        "Trained on laptop CPU\n\n"
        "77 LoRA adapters:\n"
        "  69 real authors (Poe, Carroll, Grimm,\n"
        "  Homer, Wilde, Shelley, Harris...)\n"
        "  8 synthetic controls (minimalist, dialogue,\n"
        "  questioner, cozy, dark, poet...)\n\n"
        "Synthetics isolate ONE property each.\n"
        "They are the ground truth."
    )
    ax.text(0.05, 0.82, text, fontsize=8, color="#333",
            transform=ax.transAxes, va="top", linespacing=1.4,
            family="monospace")

    # Sample outputs
    ax.text(0.05, 0.25, "Same prompt, different adapters:", fontsize=7,
            fontweight="bold", color="#666", transform=ax.transAxes, va="top")

    samples = [
        ("Poe:", "The dark and sky wept. The dark sky\nabove the clouds...", "#c44e52"),
        ("Dialogue:", '"What do you know?" asked the moon.\n"I know sky," said the storm.', "#55a868"),
        ("Lear:", ", And the Waddle!", "#8172b2"),
    ]
    y = 0.18
    for name, text, color in samples:
        ax.text(0.05, y, name, fontsize=6.5, fontweight="bold", color=color,
                transform=ax.transAxes, va="top")
        ax.text(0.18, y, text, fontsize=6, color="#555", style="italic",
                transform=ax.transAxes, va="top", linespacing=1.2)
        y -= 0.07


def panel_heads(ax):
    """Panel 2: Which heads matter."""
    ax.axis("off")
    add_panel_box(ax, "#55a868")

    ax.text(0.05, 0.95, "2. Which Heads Matter", fontsize=14,
            fontweight="bold", color="#55a868",
            transform=ax.transAxes, va="top")

    ax.text(0.05, 0.82,
            "Knocked out each head, measured\n"
            "perplexity recovery per author.\n\n"
            "3 heads out of 16 carry most style:",
            fontsize=8, color="#333", transform=ax.transAxes, va="top",
            linespacing=1.4)

    heads = [
        ("H11", "#4c72b0", "Workhorse — dominant for 66%"),
        ("H3", "#55a868", "Reader — consistent second"),
        ("H14", "#c44e52", "Polarizing — helps some, hurts others"),
    ]
    y = 0.52
    for name, color, desc in heads:
        ax.text(0.08, y, name, fontsize=11, fontweight="bold", color=color,
                transform=ax.transAxes, va="top")
        ax.text(0.22, y, desc, fontsize=8, color="#555",
                transform=ax.transAxes, va="top")
        y -= 0.08

    ax.text(0.05, 0.22,
            "But WHAT do they compute?\n"
            "→ Need to look inside the residual stream",
            fontsize=9, color="#333", transform=ax.transAxes, va="top",
            fontweight="bold", linespacing=1.5)


def panel_sae(ax):
    """Panel 3: SAE decomposition."""
    ax.axis("off")
    add_panel_box(ax, "#e8a735")

    ax.text(0.05, 0.95, "3. Sparse Autoencoder", fontsize=14,
            fontweight="bold", color="#e8a735",
            transform=ax.transAxes, va="top")

    ax.text(0.05, 0.82,
            "Decompose the 1024-dim residual stream\n"
            "into interpretable features.\n\n"
            "TopK activation (Gao et al. 2024):\n"
            "  2048 features, only 16 fire per token\n"
            "  314 alive, 4.3% mean firing rate\n"
            "  Most selective: 0.3% of tokens\n\n"
            "Label features with synthetic controls:\n"
            "  1. Which tokens fire it?\n"
            "  2. Which synthetics correlate?\n"
            "  3. Do both agree?\n\n"
            "First SAE had 99% firing rate (not sparse!).\n"
            "TopK fixed it. The sparsity matters.",
            fontsize=8, color="#333", transform=ax.transAxes, va="top",
            linespacing=1.4)


def panel_head_roles(ax):
    """Panel 4: What each head computes."""
    ax.axis("off")
    add_panel_box(ax, "#c44e52")

    ax.text(0.05, 0.95, "4. What Each Head Computes", fontsize=14,
            fontweight="bold", color="#c44e52",
            transform=ax.transAxes, va="top")

    roles = [
        ("H11", "#4c72b0", "Workhorse", "17 features — MOSTLY OPAQUE"),
        ("H3", "#55a868", "Style Reader", "107 features — reads ALL axes"),
        ("H14", "#c44e52", "Formality Enforcer",
         "anti-correlates with \"I\", \"am/was\"\nHelps Homer (+0.73), Milton (+0.68)\nHurts Shelley (−0.68), Wilde (−0.34)"),
        ("MLP", "#e8a735", "Multi-Head Mix",
         "27 head-independent features\nincl. STRONGEST style direction\n(simplicity, max |r|=0.13 with any head)"),
    ]

    y = 0.82
    for name, color, role, detail in roles:
        ax.text(0.05, y, name, fontsize=12, fontweight="bold", color=color,
                transform=ax.transAxes, va="top")
        ax.text(0.18, y, role, fontsize=9, fontweight="bold", color="#333",
                transform=ax.transAxes, va="top")
        ax.text(0.18, y - 0.04, detail, fontsize=7, color="#666",
                transform=ax.transAxes, va="top", linespacing=1.3)
        y -= 0.18 if name != "H14" else 0.22

    ax.text(0.05, 0.06,
            "H14 mystery from Article 1: SOLVED",
            fontsize=9, fontweight="bold", color="#c44e52",
            transform=ax.transAxes, va="top")


def panel_two_layers(ax):
    """Panel 5: The main finding — two layers of style."""
    ax.axis("off")
    add_panel_box(ax, "#333")

    ax.text(0.5, 0.97, "5. Two Layers of Style",
            fontsize=16, fontweight="bold", color="#333",
            transform=ax.transAxes, va="top", ha="center")

    # Structural side
    ax.text(0.25, 0.87, "STRUCTURAL", fontsize=13, fontweight="bold",
            color="#55a868", transform=ax.transAxes, va="top", ha="center")
    ax.text(0.25, 0.80, "Shared axes · Universal knobs",
            fontsize=7, color="#55a868", transform=ax.transAxes,
            va="top", ha="center")

    struct = [
        "Simplicity (f665)  9.1→6.0 words",
        "Complexity (f883+) 8.1→12.2 words",
        "Dialogue (f1777+)  quotes ↑",
        "Questions (f329)   ? marks ↑",
        "Verse (f344)       line breaks",
    ]
    y = 0.72
    for line in struct:
        ax.text(0.05, y, line, fontsize=7.5, color="#333",
                transform=ax.transAxes, va="top", family="monospace")
        y -= 0.06

    ax.text(0.25, 0.36, "75-100% win rate\nWorks on ANY model",
            fontsize=8, fontweight="bold", color="#55a868",
            transform=ax.transAxes, va="top", ha="center")

    # Semantic side
    ax.text(0.75, 0.87, "SEMANTIC", fontsize=13, fontweight="bold",
            color="#c44e52", transform=ax.transAxes, va="top", ha="center")
    ax.text(0.75, 0.80, "Unique identity · Adapter-specific",
            fontsize=7, color="#c44e52", transform=ax.transAxes,
            va="top", ha="center")

    sem = [
        ('Dark f1224', '"not quite a smile"'),
        ('Dark f562', '"looking in, looking in"'),
        ('Cozy f1988', '"steam from the meat"'),
        ('Cozy f930', '"wool soft against fingers"'),
        ('Carroll f815', '"said Alice"'),
    ]
    y = 0.72
    for name, quote in sem:
        ax.text(0.55, y, name, fontsize=7.5, fontweight="bold", color="#333",
                transform=ax.transAxes, va="top")
        ax.text(0.72, y, quote, fontsize=7, color="#c44e52", style="italic",
                transform=ax.transAxes, va="top")
        y -= 0.06

    ax.text(0.75, 0.36, "Detects everywhere\nSteers ONLY with adapter",
            fontsize=8, fontweight="bold", color="#c44e52",
            transform=ax.transAxes, va="top", ha="center")

    # Divider
    ax.plot([0.5, 0.5], [0.30, 0.88], color="#ddd", linewidth=1.5,
            transform=ax.transAxes)

    # Bottom finding
    ax.text(0.5, 0.20,
            "Every author is primarily semantic.\n"
            "Harris: 0 structural, 40 semantic features.\n"
            "What makes an author unique isn't sentence length — it's content.",
            fontsize=8, color="#666", transform=ax.transAxes,
            va="top", ha="center", style="italic", linespacing=1.4)

    # The table
    ax.text(0.5, 0.05,
            "Base model: structural steers, semantic doesn't  |  "
            "Matching adapter: BOTH steer",
            fontsize=7.5, fontweight="bold", color="#333",
            transform=ax.transAxes, va="top", ha="center")


def panel_steering(ax):
    """Panel 6: Steering examples."""
    ax.axis("off")
    add_panel_box(ax, "#8172b2")

    ax.text(0.05, 0.95, "6. Steering Works", fontsize=14,
            fontweight="bold", color="#8172b2",
            transform=ax.transAxes, va="top")

    examples = [
        {
            "title": "Poe + Simplicity",
            "metric": "23.9 → 4.9 words (20/20)",
            "baseline": "The dark sky above the clouds seemed to go away...",
            "steered": "It was dark. I woke up. It was dark.",
            "color": "#e8a735",
        },
        {
            "title": "Grimm + Questions",
            "metric": "declarative → interrogative",
            "baseline": "He loved to go to the forest...",
            "steered": "'I want to be that?' said the frog, 'I go to the mill??'",
            "color": "#c44e52",
        },
        {
            "title": "Cozy + Cozy features",
            "metric": "semantic amplification with adapter",
            "baseline": "The oven was a bit fat and slowy...",
            "steered": "She stirred and stirred, the cat smelled the cake and the pots.",
            "color": "#55a868",
        },
    ]

    y = 0.82
    for ex in examples:
        ax.text(0.05, y, ex["title"], fontsize=9, fontweight="bold",
                color=ex["color"], transform=ax.transAxes, va="top")
        ax.text(0.45, y, ex["metric"], fontsize=7, color="#999",
                transform=ax.transAxes, va="top")

        ax.text(0.05, y - 0.06, fill(ex["baseline"], width=55),
                fontsize=6.5, color="#888", style="italic",
                transform=ax.transAxes, va="top")
        ax.text(0.05, y - 0.14, "→ " + fill(ex["steered"], width=53),
                fontsize=6.5, color=ex["color"], style="italic",
                fontweight="bold", transform=ax.transAxes, va="top")
        y -= 0.28


def panel_big_picture(ax):
    """Panel 7: The bigger picture."""
    ax.axis("off")
    add_panel_box(ax, "#333", alpha=0.05)

    ax.text(0.5, 0.90, "The Bigger Picture", fontsize=14,
            fontweight="bold", color="#333",
            transform=ax.transAxes, va="top", ha="center")

    findings = [
        ("1.", "LoRAs amplify, they don't create.",
         "98.8% of features already exist in the base model. "
         "Style is latent — fine-tuning selects, it doesn't construct."),
        ("2.", "The strongest style direction is invisible to heads.",
         "Simplicity (f665): max |r|=0.13 with any head. "
         "Weight steering: 49% (coin flip). Activation steering: 100%."),
        ("3.", "Structure steers universally, semantics needs the adapter.",
         "The model has two layers of style representation. "
         "Our tools reach both, but differently."),
    ]

    y = 0.75
    for num, title, detail in findings:
        ax.text(0.05, y, num, fontsize=10, fontweight="bold", color="#333",
                transform=ax.transAxes, va="top")
        ax.text(0.08, y, title, fontsize=9, fontweight="bold", color="#333",
                transform=ax.transAxes, va="top")
        ax.text(0.08, y - 0.08, fill(detail, width=90),
                fontsize=7.5, color="#666",
                transform=ax.transAxes, va="top", linespacing=1.3)
        y -= 0.22

    ax.text(0.5, 0.10,
            "A tiny transformer represents writing style in two layers — "
            "shared structural axes that are universally steerable,\n"
            "and unique semantic fingerprints that are detectable everywhere "
            "but controllable only through author-specific adapters.",
            fontsize=8, color="#333", transform=ax.transAxes,
            va="top", ha="center", style="italic", linespacing=1.4)


def main():
    # Poster layout: 3 columns top, 1 wide center, 3 columns bottom
    fig = plt.figure(figsize=(36, 24))

    # Title
    fig.text(0.5, 0.98,
             "Sixteen Voices: How a Tiny Transformer Represents Writing Style",
             ha="center", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.965,
             "Mechanistic interpretability on TinyStories-1Layer-21M — "
             "from head knockouts to steerable style features. "
             "All experiments on a laptop CPU.",
             ha="center", fontsize=14, color="#666")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           height_ratios=[1, 1.2, 0.8],
                           hspace=0.15, wspace=0.12,
                           left=0.03, right=0.97, top=0.94, bottom=0.03)

    # Top row: Setup, Heads, SAE
    panel_setup(fig.add_subplot(gs[0, 0]))
    panel_heads(fig.add_subplot(gs[0, 1]))
    panel_sae(fig.add_subplot(gs[0, 2]))

    # Middle row: Head roles, Two layers (wide), Steering
    panel_head_roles(fig.add_subplot(gs[1, 0]))
    panel_two_layers(fig.add_subplot(gs[1, 1]))
    panel_steering(fig.add_subplot(gs[1, 2]))

    # Bottom row: Big picture (full width)
    panel_big_picture(fig.add_subplot(gs[2, :]))

    out = FIGURES_DIR / "poster_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
