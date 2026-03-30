#!/usr/bin/env python3
"""Generate style space PCA with steering direction arrows.

Usage:
    uv run python scripts/fig_sae_steering.py
    uv run python scripts/fig_sae_steering.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")

# Feature groups for the overcomplete TopK SAE
FEATURE_GROUPS = {
    "simplicity": {
        "features": [665],
        "color": "#e8a735",
        "label": "SIMPLICITY\n(head-independent)",
        "label_offset": (3.0, 0.5),  # nudge right and slightly up
    },
    "complexity": {
        "features": [883, 993, 60],
        "color": "#c44e52",
        "label": "COMPLEXITY\n(H3-controlled)",
        "label_offset": (0, 0),
    },
    "dialogue": {
        "features": [1777, 689],
        "color": "#55a868",
        "label": "DIALOGUE\n(H3-controlled)",
        "label_offset": (-3.0, 1.5),  # nudge left and up
    },
}

HEAD_COLORS = {3: "#55a868", 11: "#4c72b0", 14: "#c44e52"}
DEFAULT_COLOR = "#cccccc"


def fig_style_space_with_arrows(sae_dir):
    """PCA scatter with steering arrows overlaid."""
    d = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()
    n_features = matrix.shape[1]

    with open("outputs/knockout_all_heads.json") as f:
        ko_raw = json.load(f)
    knockout = np.array([
        [ko_raw[a]["head_recovery"][f"H{h}"] for h in range(16)]
        for a in authors
    ])
    best_head = knockout.argmax(axis=1)

    # PCA on alive features only
    alive = matrix.mean(axis=0) > 0.01
    matrix_alive = matrix[:, alive]
    centered = matrix_alive - matrix_alive.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_explained = S ** 2 / (S ** 2).sum()

    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot authors
    for i, author in enumerate(authors):
        h = best_head[i]
        c = HEAD_COLORS.get(h, DEFAULT_COLOR)
        s = 60 if h in HEAD_COLORS else 30
        alpha = 0.8 if h in HEAD_COLORS else 0.4
        ax.scatter(coords[i, 0], coords[i, 1], c=c, s=s, alpha=alpha,
                   edgecolors="white", linewidth=0.5, zorder=2)

    # Label interesting authors
    label_set = {"poe", "carroll", "grimm", "minimalist", "dark", "poet",
                 "wilde", "homer", "dialogue", "harris", "cozy", "lear",
                 "montgomery", "baker", "gibbon", "reporter"}
    for i, author in enumerate(authors):
        if author in label_set:
            ax.annotate(author, (coords[i, 0], coords[i, 1]),
                        fontsize=7.5, alpha=0.75,
                        xytext=(5, 5), textcoords="offset points")

    # Project feature directions into PCA space and draw arrows
    cx, cy = coords.mean(axis=0)
    arrow_scale = 15

    # Map alive feature indices
    alive_indices = np.where(alive)[0]
    alive_set = set(alive_indices.tolist())

    for group_name, group in FEATURE_GROUPS.items():
        # Build direction in alive-feature space
        direction = np.zeros(alive.sum())
        for fi in group["features"]:
            if fi in alive_set:
                alive_pos = np.searchsorted(alive_indices, fi)
                direction[alive_pos] = 1.0

        if np.linalg.norm(direction) < 0.01:
            print(f"  Warning: {group_name} features not alive, skipping arrow")
            continue

        direction /= np.linalg.norm(direction)
        pca_dir = Vt[:2] @ direction

        arrow_tip_x = cx + pca_dir[0] * arrow_scale
        arrow_tip_y = cy + pca_dir[1] * arrow_scale
        ax.annotate("", xy=(arrow_tip_x, arrow_tip_y),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=group["color"],
                                    lw=3, mutation_scale=20))
        ox, oy = group.get("label_offset", (0, 0))
        ax.text(arrow_tip_x * 1.15 + ox,
                arrow_tip_y * 1.15 + oy,
                group["label"],
                fontsize=10, fontweight="bold", color=group["color"],
                ha="center", va="center")

    # Legend
    for h, c in HEAD_COLORS.items():
        mask = best_head == h
        ax.scatter([], [], c=c, s=60, label=f"H{h} dominant ({mask.sum()})")
    ax.scatter([], [], c=DEFAULT_COLOR, s=30, alpha=0.5, label="Other heads")
    ax.legend(loc="lower left", fontsize=9)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%} variance)", fontsize=11)
    ax.set_title("Authors in SAE feature space with steering directions",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_style_space_arrows.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_style_space_arrows.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-dir", type=str, default="outputs/sae_topk16_2048")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sae_dir = Path(args.sae_dir)

    print("Generating style space with arrows...")
    fig_style_space_with_arrows(sae_dir)
    print("Done.")


if __name__ == "__main__":
    main()