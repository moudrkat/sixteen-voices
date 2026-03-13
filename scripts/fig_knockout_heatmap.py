#!/usr/bin/env python3
"""Knockout results: 82 authors x 16 heads recovery heatmap + strip plot.

Reads knockout_all_heads.json and produces:
1. Heatmap: per-author x per-head recovery scores (clustered by similarity)
2. Strip plot: per-head recovery distribution across authors

Usage:
    python scripts/fig_knockout_heatmap.py
    python scripts/fig_knockout_heatmap.py --input outputs/knockout_all_heads.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

FIGURES_DIR = Path("figures")
NUM_HEADS = 16

ACCENT_HIGH = "#C44E52"
ACCENT_MID = "#DD8452"
LIGHT_GRAY = "#cccccc"


def load_knockout(path: Path) -> tuple[list[str], np.ndarray]:
    """Load knockout JSON, return (authors, recovery_matrix[n_authors, 16])."""
    with open(path) as f:
        data = json.load(f)
    authors = list(data.keys())
    matrix = np.array([
        [data[a]["head_recovery"][f"H{h}"] for h in range(NUM_HEADS)]
        for a in authors
    ])
    return authors, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/knockout_all_heads.json")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    authors, matrix = load_knockout(Path(args.input))
    n_authors = len(authors)
    print(f"Loaded {n_authors} authors, {NUM_HEADS} heads")

    # --- Figure 1: Heatmap (authors clustered by recovery profile) ---
    # Cluster authors by similarity of their 16-dim recovery vectors
    Z = linkage(matrix, method="ward", metric="euclidean")
    row_order = leaves_list(Z)
    sorted_authors = [authors[i] for i in row_order]
    sorted_matrix = matrix[row_order]

    # Sort columns by mean recovery (most important heads left)
    col_means = matrix.mean(axis=0)
    col_order = np.argsort(-col_means)
    sorted_matrix = sorted_matrix[:, col_order]
    head_labels = [f"H{h}" for h in col_order]

    fig, ax = plt.subplots(figsize=(10, max(8, n_authors * 0.28)))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(sorted_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels(head_labels, fontsize=9)
    ax.set_yticks(range(n_authors))
    ax.set_yticklabels(sorted_authors, fontsize=7)
    ax.set_xlabel("Attention head (sorted by mean recovery)")
    ax.set_title("Head knockout recovery: which heads carry each author's style?\n"
                 "(1.0 = head alone recovers full adaptation, 0 = no effect, "
                 "negative = hurts)",
                 fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Recovery score", shrink=0.6, pad=0.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "knockout_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'knockout_heatmap.png'}")
    plt.close()

    # --- Figure 2: Strip plot (per-head distribution of recovery scores) ---
    means = matrix.mean(axis=0)
    order = list(np.argsort(-means))

    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(42)

    stds = matrix.std(axis=0)
    for xi, h in enumerate(order):
        vals = matrix[:, h]
        jitter = rng.uniform(-0.25, 0.25, len(vals))
        color = (ACCENT_HIGH if means[h] > 0.20
                 else ACCENT_MID if means[h] > 0.10
                 else LIGHT_GRAY)
        ax.scatter(xi + jitter, vals, s=12, alpha=0.45, color=color,
                   zorder=2, edgecolors="none")
        # Mean diamond with ±1 std error bar
        ax.errorbar(xi, means[h], yerr=stds[h], fmt="D", color="black",
                    markersize=6, capsize=3, capthick=1.2, elinewidth=1.2,
                    zorder=4)

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axhline(y=1/NUM_HEADS, color="gray", linestyle="--", alpha=0.5,
               label=f"uniform = {1/NUM_HEADS:.3f}")

    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels([f"H{h}" for h in order], fontsize=9)
    ax.set_ylabel("Recovery score")
    ax.set_xlabel("Attention head (sorted by mean recovery)")
    ax.set_title(f"Per-head knockout recovery across {n_authors} authors\n"
                 "(high spread = author-specific head)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Annotate most polarized head
    most_polar = int(np.argmax(stds))
    polar_xi = order.index(most_polar)
    ax.annotate(f"H{most_polar}: std={stds[most_polar]:.2f}\n(most variable head)",
                xy=(polar_xi, means[most_polar]),
                xytext=(polar_xi - 4, means[most_polar] + 0.45),
                fontsize=8, color=ACCENT_HIGH, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT_HIGH, lw=1.2))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "knockout_strip.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'knockout_strip.png'}")
    plt.close()

    # --- Figure 3: Best head per author (who depends on what?) ---
    best_heads = matrix.argmax(axis=1)
    head_counts = np.bincount(best_heads, minlength=NUM_HEADS)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [ACCENT_HIGH if c >= 8 else ACCENT_MID if c >= 4 else LIGHT_GRAY
              for c in head_counts]
    ax.bar(range(NUM_HEADS), head_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(NUM_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(NUM_HEADS)], fontsize=9)
    ax.set_ylabel("Number of authors")
    ax.set_xlabel("Attention head")
    ax.set_title(f"Which head is most important for each author? ({n_authors} authors)",
                 fontsize=11, fontweight="bold")

    for h in range(NUM_HEADS):
        if head_counts[h] > 0:
            ax.text(h, head_counts[h] + 0.3, str(head_counts[h]),
                    ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "knockout_best_head.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'knockout_best_head.png'}")
    plt.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("KNOCKOUT SUMMARY")
    print(f"{'=' * 60}")
    print(f"\nMean recovery per head (sorted):")
    for h in order:
        bar = "#" * int(max(0, means[h]) * 80)
        print(f"  H{h:<3d} mean={means[h]:+.3f}  std={stds[h]:.3f}  {bar}")

    print(f"\nBest head distribution:")
    for h in np.argsort(-head_counts):
        if head_counts[h] > 0:
            who = [authors[i] for i in range(n_authors) if best_heads[i] == h]
            print(f"  H{h}: {head_counts[h]} authors — {', '.join(who[:5])}"
                  + (f" (+{len(who)-5} more)" if len(who) > 5 else ""))


if __name__ == "__main__":
    main()