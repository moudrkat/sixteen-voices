#!/usr/bin/env python3
"""SAE feature analysis figures and feature interpretation.

Produces:
1. PCA scatter of authors colored by best-head (style_pca.png)
2. Feature-head correlation bar chart (feature_head_bars.png)
3. H14 polarization scatter (h14_features.png)
4. Feature interpretation: re-runs SAE on eval texts to label top features

Usage:
    uv run python scripts/fig_sae.py
    uv run python scripts/fig_sae.py --interpret  # also print feature interpretations
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

FIGURES_DIR = Path("figures")
DEFAULT_SAE_DIR = Path("outputs/sae_topk16_2048")
NUM_HEADS = 16

# Colors consistent with other figures
HEAD_COLORS = {
    3: "#55a868",   # green
    11: "#4c72b0",  # blue
    14: "#c44e52",  # red
}
DEFAULT_COLOR = "#cccccc"
ACCENT_HIGH = "#C44E52"
ACCENT_MID = "#DD8452"


def load_data(sae_dir):
    """Load SAE matrix, knockout scores, and analysis results."""
    d = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko_raw = json.load(f)
    knockout = np.array([
        [ko_raw[a]["head_recovery"][f"H{h}"] for h in range(NUM_HEADS)]
        for a in authors
    ])

    with open(sae_dir / "feature_head_analysis.json") as f:
        analysis = json.load(f)

    return authors, matrix, knockout, analysis


def fig_style_pca(authors, matrix, knockout):
    """PCA scatter of authors in feature space, colored by best head."""
    centered = matrix - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_explained = S ** 2 / (S ** 2).sum()

    best_head = knockout.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot non-key heads first (gray)
    for i, author in enumerate(authors):
        h = best_head[i]
        if h not in HEAD_COLORS:
            ax.scatter(coords[i, 0], coords[i, 1], c=DEFAULT_COLOR,
                       s=30, alpha=0.5, zorder=1)

    # Plot key heads on top
    for h, color in HEAD_COLORS.items():
        mask = best_head == h
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                   s=60, alpha=0.8, label=f"H{h} ({mask.sum()})",
                   zorder=2, edgecolors="white", linewidth=0.5)

    # Label some interesting authors
    label_authors = {"poe", "wilde", "carroll", "grimm", "minimalist",
                     "dark", "poet", "harris", "baker", "dialogue",
                     "reporter", "repeater", "homer", "cozy"}
    for i, author in enumerate(authors):
        if author in label_authors:
            ax.annotate(author, (coords[i, 0], coords[i, 1]),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%} variance)")
    ax.set_title("Authors in SAE feature space")
    ax.legend(title="Best head", loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_style_pca.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_style_pca.png'}")


def compute_bh_counts(authors, matrix, knockout, fdr=0.05):
    """Compute per-head significant feature counts using Benjamini-Hochberg."""
    n_features = matrix.shape[1]
    all_tests = []
    for h in range(NUM_HEADS):
        for f in range(n_features):
            r, p = stats.pearsonr(matrix[:, f], knockout[:, h])
            all_tests.append((h, f, r, p))

    all_p = np.array([t[3] for t in all_tests])
    n_tests = len(all_p)
    sorted_idx = np.argsort(all_p)
    sorted_p = all_p[sorted_idx]
    thresholds = fdr * (np.arange(1, n_tests + 1)) / n_tests

    max_k = 0
    for k in range(n_tests):
        if sorted_p[k] <= thresholds[k]:
            max_k = k
    sig_idx = set(sorted_idx[:max_k + 1].tolist())

    counts = {h: 0 for h in range(NUM_HEADS)}
    for i, (h, f, r, p) in enumerate(all_tests):
        if i in sig_idx:
            counts[h] += 1
    return counts


def fig_feature_head_bars(analysis, bh_counts=None):
    """Bar chart: significant features per head (BH-corrected)."""
    if bh_counts is not None:
        counts = [bh_counts[h] for h in range(NUM_HEADS)]
    else:
        sig = analysis["sig_features_per_head"]
        counts = [sig[f"H{h}"] for h in range(NUM_HEADS)]

    heads = [f"H{h}" for h in range(NUM_HEADS)]
    colors = [HEAD_COLORS.get(h, DEFAULT_COLOR) for h in range(NUM_HEADS)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(heads, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha="center", va="bottom", fontsize=13,
                fontweight="bold")

    ax.set_ylabel("Number of correlated SAE features", fontsize=14)
    ax.set_xlabel("Attention head", fontsize=14)
    ax.set_title("How many features does each head control?",
                 fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_feature_head_bars.png", dpi=300)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_feature_head_bars.png'}")


def fig_h14_polarization(authors, matrix, knockout, analysis):
    """Scatter: authors on top H14-discriminating features."""
    h14 = knockout[:, 14]
    pol = analysis.get("h14_polarization")
    if not pol or len(pol["top_features"]) < 2:
        print("Skipping H14 figure — not enough polarization data")
        return

    # Pick top 2 discriminating features for the scatter axes
    f1_idx = pol["top_features"][0][0]
    f2_idx = pol["top_features"][1][0]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Color by H14 recovery
    sc = ax.scatter(matrix[:, f1_idx], matrix[:, f2_idx],
                    c=h14, cmap="RdBu_r", vmin=-0.3, vmax=0.5,
                    s=50, edgecolors="white", linewidth=0.5, zorder=2)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("H14 recovery score")

    # Label extremes
    helped = h14 > 0.3
    hurt = h14 < -0.05
    for i, author in enumerate(authors):
        if helped[i] or hurt[i]:
            ax.annotate(author, (matrix[i, f1_idx], matrix[i, f2_idx]),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"Feature {f1_idx} activation")
    ax.set_ylabel(f"Feature {f2_idx} activation")
    ax.set_title("H14 polarization explained by SAE features")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_h14_polarization.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_h14_polarization.png'}")


def fig_effective_dim(matrix):
    """Cumulative variance explained plot."""
    centered = matrix - matrix.mean(axis=0)
    S = np.linalg.svd(centered, compute_uv=False)
    var = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="#4c72b0", linewidth=2)
    ax.axhline(0.5, color="#999", linestyle="--", linewidth=0.8)
    ax.axhline(0.8, color="#999", linestyle="--", linewidth=0.8)
    ax.axhline(0.9, color="#999", linestyle="--", linewidth=0.8)

    for thresh in [0.5, 0.8, 0.9]:
        n = int(np.searchsorted(cumvar, thresh)) + 1
        ax.plot(n, thresh, "o", color=ACCENT_HIGH, markersize=8, zorder=3)
        ax.annotate(f"{n}d = {thresh:.0%}", (n, thresh),
                    xytext=(10, -5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Effective dimensionality of style space")
    ax.set_xlim(0, 60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_effective_dim.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_effective_dim.png'}")


def interpret_features(authors, matrix, knockout):
    """Label key features by what they correlate with."""
    # Features to interpret: top correlated with H3, H11, H14, H15
    # + top H14-polarization features
    # + highest-variance features
    corr_matrix = np.zeros((matrix.shape[1], NUM_HEADS))
    for f in range(matrix.shape[1]):
        for h in range(NUM_HEADS):
            corr_matrix[f, h], _ = stats.pearsonr(matrix[:, f], knockout[:, h])

    # For each feature, describe its author profile
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0) + 1e-8

    # Collect interesting features
    interesting = set()
    for h in [3, 11, 14, 15]:
        top = np.argsort(np.abs(corr_matrix[:, h]))[-5:]
        interesting.update(top.tolist())

    # Highest-variance features
    var = matrix.var(axis=0)
    interesting.update(np.argsort(var)[-10:].tolist())

    # H14 polarization features
    h14 = knockout[:, 14]
    helped = h14 > 0.1
    hurt = h14 < -0.1
    if helped.sum() >= 3 and hurt.sum() >= 3:
        diff = matrix[helped].mean(axis=0) - matrix[hurt].mean(axis=0)
        interesting.update(np.argsort(np.abs(diff))[-5:].tolist())

    print(f"\n── Feature Interpretations ({len(interesting)} features) ──\n")

    for f in sorted(interesting):
        # Author profile
        z_scores = (matrix[:, f] - global_mean[f]) / global_std[f]
        top_authors = np.argsort(z_scores)[-3:][::-1]
        bot_authors = np.argsort(z_scores)[:3]

        # Head correlations
        head_corrs = [(h, corr_matrix[f, h]) for h in range(NUM_HEADS)]
        head_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        sig_heads = [(h, r) for h, r in head_corrs if abs(r) > 0.3]

        top_str = ", ".join(f"{authors[i]}({z_scores[i]:+.1f}σ)" for i in top_authors)
        bot_str = ", ".join(f"{authors[i]}({z_scores[i]:+.1f}σ)" for i in bot_authors)
        head_str = ", ".join(f"H{h}({r:+.2f})" for h, r in sig_heads) if sig_heads else "none"

        print(f"  f{f:3d} | var={var[f]:.3f} | heads: {head_str}")
        print(f"        ↑ {top_str}")
        print(f"        ↓ {bot_str}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpret", action="store_true",
                        help="Also print feature interpretations")
    parser.add_argument("--sae-dir", type=str, default=None,
                        help="SAE directory (default: outputs/sae_topk16_2048)")
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir) if args.sae_dir else DEFAULT_SAE_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    authors, matrix, knockout, analysis = load_data(sae_dir)

    fig_style_pca(authors, matrix, knockout)
    print("Computing Benjamini-Hochberg correction...")
    bh_counts = compute_bh_counts(authors, matrix, knockout)
    fig_feature_head_bars(analysis, bh_counts=bh_counts)
    fig_h14_polarization(authors, matrix, knockout, analysis)
    fig_effective_dim(matrix)

    if args.interpret:
        interpret_features(authors, matrix, knockout)

    print("\nDone.")


if __name__ == "__main__":
    main()