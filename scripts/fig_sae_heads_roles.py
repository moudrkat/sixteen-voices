#!/usr/bin/env python3
"""Generate the heads-features-authors-MLP connection figure.

Shows how style computation flows through the model:
- Left: style axes with synthetic controls as endpoints
- Center: heads and MLP as computational units
- Right: which authors each head serves

Usage:
    uv run python scripts/fig_sae_heads_roles.py
    uv run python scripts/fig_sae_heads_roles.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", type=str, default="outputs/sae_topk16_2048")
    args = parser.parse_args()
    sae_dir = Path(args.sae_dir)

    # Load data
    d = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko = json.load(f)

    # H14 scores for helped/hurt
    h14_scores = {a: ko[a]["head_recovery"]["H14"] for a in authors}

    # ---------- FIGURE ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 10),
                             gridspec_kw={"width_ratios": [3, 2, 3]})
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    ax_left, ax_mid, ax_right = axes

    # ── LEFT PANEL: Style axes with synthetic endpoints ──
    ax_left.set_title("Style Axes (SAE features)", fontsize=14, fontweight="bold", pad=15)

    style_axes = [
        {
            "y": 8.5,
            "left_label": "unusual_vocab\negyptian\nbrowne",
            "right_label": "minimalist\ndialogue\ncozy",
            "name": "Formal ↔ Simple",
            "features": "f2032, f1242, f1262",
            "color": "#55a868",  # H3 green
            "head": "H3",
        },
        {
            "y": 6.8,
            "left_label": "dialogue\nfirstperson\nquestioner",
            "right_label": "unusual_vocab\nlovecraft\ngibbon",
            "name": "Interactive ↔ Formal prose",
            "features": "f1779, f627, f1777",
            "color": "#55a868",
            "head": "H3",
        },
        {
            "y": 5.1,
            "left_label": "lear, baker\npoe",
            "right_label": "minimalist\nquestioner",
            "name": "Complexity ↔ Simplicity",
            "features": "f883, f993, f60",
            "color": "#55a868",
            "head": "H3",
        },
        {
            "y": 3.4,
            "left_label": "homer, melville\nmilton, pater",
            "right_label": "shelley, wilde\nwells, stoker",
            "name": "H14 Formality axis",
            "features": "f1519, f1280",
            "color": "#c44e52",  # H14 red
            "head": "H14",
        },
        {
            "y": 1.5,
            "left_label": "minimalist, poet\nsimple_vocab",
            "right_label": "gibbon, carlyle\nunusual_vocab",
            "name": "Simplicity (head-independent)",
            "features": "f665",
            "color": "#e8a735",  # MLP orange
            "head": "MLP",
        },
    ]

    for sa in style_axes:
        y = sa["y"]

        # Axis line
        ax_left.plot([2.5, 7.5], [y, y], color=sa["color"], linewidth=3, solid_capstyle="round")

        # Endpoints
        ax_left.plot(2.5, y, "o", color=sa["color"], markersize=10, zorder=5)
        ax_left.plot(7.5, y, "o", color=sa["color"], markersize=10, zorder=5)

        # Labels
        ax_left.text(2.3, y, sa["left_label"], ha="right", va="center",
                     fontsize=8, color="#333333", linespacing=1.3)
        ax_left.text(7.7, y, sa["right_label"], ha="left", va="center",
                     fontsize=8, color="#333333", linespacing=1.3)

        # Axis name + features above the line
        ax_left.text(5.0, y + 0.45, sa["name"], ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=sa["color"])
        ax_left.text(5.0, y + 0.15, sa["features"], ha="center", va="bottom",
                     fontsize=7, color="#666666", style="italic")

        # Head tag
        ax_left.text(8.2, y, sa["head"], ha="left", va="center",
                     fontsize=8, fontweight="bold", color=sa["color"],
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=sa["color"],
                              alpha=0.15, edgecolor=sa["color"], linewidth=0.5))

    # ── MIDDLE PANEL: Heads and MLP as computational units ──
    ax_mid.set_title("Computation", fontsize=14, fontweight="bold", pad=15)

    heads = [
        {
            "y": 8.0, "name": "H11", "color": "#4c72b0",
            "role": "Workhorse",
            "detail": "dominant for 66%\n17 SAE features\nMOSTLY OPAQUE",
            "size": (3.5, 1.8),
        },
        {
            "y": 5.8, "name": "H3", "color": "#55a868",
            "role": "Style Reader",
            "detail": "107 SAE features\nreads all axes\nINTERPRETABLE",
            "size": (3.5, 1.8),
        },
        {
            "y": 3.6, "name": "H14", "color": "#c44e52",
            "role": "Formality Enforcer",
            "detail": "46 SAE features\nhelps formal, hurts informal\nPOLARIZING",
            "size": (3.5, 1.8),
        },
        {
            "y": 1.3, "name": "MLP", "color": "#e8a735",
            "role": "Multi-Head Interaction",
            "detail": "27 features (incl. f665)\nemerge from multi-head\ncombination, no single head",
            "size": (3.5, 1.8),
        },
    ]

    for h in heads:
        x, y = 3.2, h["y"]
        w, ht = h["size"]

        # Box
        box = FancyBboxPatch((x - w/2, y - ht/2), w, ht,
                             boxstyle="round,pad=0.15",
                             facecolor=h["color"], alpha=0.12,
                             edgecolor=h["color"], linewidth=2)
        ax_mid.add_patch(box)

        # Head name
        ax_mid.text(x - w/2 + 0.25, y + ht/2 - 0.3, h["name"],
                    fontsize=14, fontweight="bold", color=h["color"], va="top")

        # Role
        ax_mid.text(x, y + 0.15, h["role"],
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="#333333")

        # Detail
        ax_mid.text(x, y - 0.45, h["detail"],
                    ha="center", va="center", fontsize=7.5,
                    color="#555555", linespacing=1.4)

    # ── RIGHT PANEL: Authors served by each head ──
    ax_right.set_title("Authors", fontsize=14, fontweight="bold", pad=15)

    author_groups = [
        {
            "y": 8.0, "color": "#4c72b0",
            "head": "H11",
            "text": "Dominant for most authors:\narabian, barrie, baum, burnett,\ncollodi, grahame, indian, italian,\njapanese, lang, lofting, norse,\nrussian, sewell, spyri, wyss\n+ most synthetics",
        },
        {
            "y": 5.8, "color": "#55a868",
            "head": "H3",
            "text": "Reads style broadly:\ntouches all 77 authors\nthrough 107 features\n\nBridges formal ↔ simple\ninteractive ↔ narrated\ncomplexity ↔ simplicity",
        },
        {
            "y": 3.6, "color": "#c44e52",
            "head": "H14",
            "text": "Helps (formal):\nhomer, milton, pater,\nmelville, egyptian, maya\n\nHurts (informal):\nshelley, wilde, wells,\nstoker, baum, kipling",
        },
        {
            "y": 1.3, "color": "#e8a735",
            "head": "MLP",
            "text": "Simplicity axis (f665):\nminimalist, poet, simple_vocab\nvs\ngibbon, carlyle, unusual_vocab\n\n27 head-independent features\nemerge from multi-head\ncombination through MLP",
        },
    ]

    for ag in author_groups:
        x, y = 5.0, ag["y"]

        # Colored sidebar
        ax_right.plot([1.8, 1.8], [y - 0.9, y + 0.9], color=ag["color"],
                      linewidth=4, solid_capstyle="round")

        # Text
        ax_right.text(2.2, y, ag["text"], ha="left", va="center",
                      fontsize=8, color="#333333", linespacing=1.4)

    # ── Overall title ──
    fig.suptitle("How a 1-Layer Transformer Computes Style",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94,
             "SAE features connect attention heads and MLP to specific author properties",
             ha="center", fontsize=11, color="#666666")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIGURES_DIR / "sae_heads_roles.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    main()
