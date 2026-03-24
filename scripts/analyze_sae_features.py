#!/usr/bin/env python3
"""Analyze SAE features vs head knockout structure.

Reads the author-feature matrix from analyze_sae.py and correlates with
knockout_all_heads.json to find feature-head relationships.

Usage:
    uv run python scripts/analyze_sae_features.py
"""

import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats


def load_data():
    """Load SAE matrix and knockout scores."""
    d = torch.load("outputs/sae/author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()  # (77, 256)

    with open("outputs/knockout_all_heads.json") as f:
        ko_raw = json.load(f)

    # Align knockout to same author order — extract head_recovery scores
    knockout = np.array([
        [ko_raw[a]["head_recovery"][f"H{h}"] for h in range(16)]
        for a in authors
    ])  # (77, 16)
    return authors, matrix, knockout


def feature_head_correlation(matrix, knockout):
    """Correlate each SAE feature with each head's knockout recovery."""
    n_features = matrix.shape[1]
    n_heads = knockout.shape[1]
    corr = np.zeros((n_features, n_heads))
    pval = np.zeros((n_features, n_heads))

    for f in range(n_features):
        for h in range(n_heads):
            r, p = stats.pearsonr(matrix[:, f], knockout[:, h])
            corr[f, h] = r
            pval[f, h] = p

    return corr, pval


def author_clustering(authors, matrix, knockout):
    """PCA of feature space, compare with head-based grouping."""
    # Center
    centered = matrix - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :3] * S[:3]  # first 3 PCs

    # Variance explained
    var_explained = S ** 2 / (S ** 2).sum()

    # Which head is best for each author?
    best_head = knockout.argmax(axis=1)

    return coords, var_explained, best_head


def find_discriminating_features(authors, matrix):
    """Features that best separate individual authors from the rest."""
    n_authors, n_features = matrix.shape
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0) + 1e-8

    results = {}
    for i, author in enumerate(authors):
        # Z-score of this author's mean vs global
        z = (matrix[i] - global_mean) / global_std
        top_pos = np.argsort(z)[-3:][::-1]  # 3 most elevated
        top_neg = np.argsort(z)[:3]  # 3 most suppressed
        results[author] = {
            "elevated": [(int(f), float(z[f])) for f in top_pos],
            "suppressed": [(int(f), float(z[f])) for f in top_neg],
        }
    return results


def h14_polarization(authors, matrix, knockout):
    """Do SAE features explain H14's love/hate split?"""
    h14 = knockout[:, 14]
    helped = h14 > 0.1  # authors where H14 helps
    hurt = h14 < -0.1   # authors where H14 hurts

    if helped.sum() < 3 or hurt.sum() < 3:
        return None

    helped_mean = matrix[helped].mean(axis=0)
    hurt_mean = matrix[hurt].mean(axis=0)
    diff = helped_mean - hurt_mean
    std = np.sqrt(matrix[helped].var(axis=0) + matrix[hurt].var(axis=0) + 1e-8)
    z = diff / std

    top_features = np.argsort(np.abs(z))[-10:][::-1]
    return {
        "n_helped": int(helped.sum()),
        "n_hurt": int(hurt.sum()),
        "top_features": [(int(f), float(z[f]), float(diff[f])) for f in top_features],
    }


def main():
    authors, matrix, knockout = load_data()
    n_authors, n_features = matrix.shape
    n_heads = knockout.shape[1]

    print(f"=== SAE Feature ↔ Head Analysis ===")
    print(f"{n_authors} authors, {n_features} features, {n_heads} heads\n")

    # 1. Feature-head correlation
    print("── Feature-Head Correlations ──")
    corr, pval = feature_head_correlation(matrix, knockout)

    # For each head, which features correlate most?
    for h in range(n_heads):
        top = np.argsort(np.abs(corr[:, h]))[-5:][::-1]
        sig = [(int(f), corr[f, h], pval[f, h]) for f in top if pval[f, h] < 0.05]
        if sig:
            feats = ", ".join(f"f{f}({r:+.3f} p={p:.1e})" for f, r, p in sig)
            print(f"  H{h:2d}: {feats}")
        else:
            print(f"  H{h:2d}: no significant correlations")

    # Summary: which heads have the most correlated features?
    sig_counts = [(pval[:, h] < 0.05).sum() for h in range(n_heads)]
    print(f"\n  Features significantly correlated (p<0.05) per head:")
    for h in range(n_heads):
        bar = "█" * sig_counts[h]
        print(f"    H{h:2d}: {sig_counts[h]:3d} {bar}")

    # 2. PCA of feature space
    print("\n── Author Clustering (PCA of feature space) ──")
    coords, var_explained, best_head = author_clustering(authors, matrix, knockout)
    print(f"  Variance explained: PC1={var_explained[0]:.1%}, "
          f"PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%}")

    # Do authors with the same best-head cluster together in feature space?
    for h in [11, 14, 3]:
        mask = best_head == h
        if mask.sum() < 2:
            continue
        center = coords[mask].mean(axis=0)
        spread = np.linalg.norm(coords[mask] - center, axis=1).mean()
        overall_spread = np.linalg.norm(coords - coords.mean(axis=0), axis=1).mean()
        print(f"  H{h}-dominant ({mask.sum()} authors): "
              f"cluster spread={spread:.3f} vs overall={overall_spread:.3f} "
              f"(ratio={spread/overall_spread:.2f})")

    # 3. H14 polarization
    print("\n── H14 Polarization Analysis ──")
    h14_result = h14_polarization(authors, matrix, knockout)
    if h14_result:
        print(f"  H14 helps {h14_result['n_helped']} authors, hurts {h14_result['n_hurt']}")
        print(f"  Features that distinguish helped vs hurt:")
        for f, z, diff in h14_result["top_features"]:
            direction = "higher in helped" if diff > 0 else "higher in hurt"
            print(f"    f{f}: z={z:+.2f} ({direction}, Δ={diff:+.3f})")
    else:
        print("  Not enough polarized authors to analyze")

    # 4. Author-discriminating features
    print("\n── Most Distinctive Features Per Author ──")
    disc = find_discriminating_features(authors, matrix)
    for author in sorted(authors):
        elev = ", ".join(f"f{f}({z:+.1f}σ)" for f, z in disc[author]["elevated"])
        supp = ", ".join(f"f{f}({z:+.1f}σ)" for f, z in disc[author]["suppressed"])
        print(f"  {author:>20s}: ↑{elev}  ↓{supp}")

    # 5. Effective dimensionality
    print("\n── Effective Dimensionality ──")
    centered = matrix - matrix.mean(axis=0)
    S = np.linalg.svd(centered, compute_uv=False)
    var = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var)
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        n = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"  {threshold:.0%} variance in {n} dimensions")

    # Save structured results
    results = {
        "feature_head_corr": {
            f"H{h}": [
                {"feature": int(f), "r": float(corr[f, h]), "p": float(pval[f, h])}
                for f in np.argsort(np.abs(corr[:, h]))[-10:][::-1]
                if pval[f, h] < 0.05
            ]
            for h in range(n_heads)
        },
        "sig_features_per_head": {f"H{h}": int(c) for h, c in enumerate(sig_counts)},
        "pca_variance_explained": var_explained[:10].tolist(),
        "h14_polarization": h14_result,
        "effective_dim": {
            f"{t:.0%}": int(np.searchsorted(cumvar, t)) + 1
            for t in [0.5, 0.8, 0.9, 0.95]
        },
    }

    out_path = Path("outputs/sae/feature_head_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()