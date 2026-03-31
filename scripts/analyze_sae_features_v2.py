#!/usr/bin/env python3
"""Analyze SAE features vs head knockout structure.

Works with any SAE directory. Reads the author-feature matrix and correlates
with knockout scores to find feature-head relationships, PCA clustering,
H14 polarization, and effective dimensionality.

Usage:
    uv run python scripts/analyze_sae_features_v2.py
    uv run python scripts/analyze_sae_features_v2.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats


def load_data(sae_dir):
    """Load SAE matrix and knockout scores."""
    d = torch.load(Path(sae_dir) / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko_raw = json.load(f)

    knockout = np.array([
        [ko_raw[a]["head_recovery"][f"H{h}"] for h in range(16)]
        for a in authors
    ])
    return authors, matrix, knockout


def feature_head_correlation(matrix, knockout):
    """Correlate each SAE feature with each head's knockout recovery."""
    n_features = matrix.shape[1]
    n_heads = knockout.shape[1]
    corr = np.zeros((n_features, n_heads))
    pval = np.zeros((n_features, n_heads))

    alive = matrix.mean(axis=0) > 0.01

    for f in range(n_features):
        if not alive[f]:
            continue
        for h in range(n_heads):
            r, p = stats.pearsonr(matrix[:, f], knockout[:, h])
            corr[f, h] = r
            pval[f, h] = p

    return corr, pval, alive


def author_clustering(authors, matrix, knockout):
    """PCA of feature space, compare with head-based grouping."""
    # Only use alive features for PCA
    alive = matrix.mean(axis=0) > 0.01
    matrix_alive = matrix[:, alive]

    centered = matrix_alive - matrix_alive.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :3] * S[:3]

    var_explained = S ** 2 / (S ** 2).sum()
    best_head = knockout.argmax(axis=1)

    return coords, var_explained, best_head


def h14_polarization(authors, matrix, knockout):
    """Do SAE features explain H14's love/hate split?"""
    h14 = knockout[:, 14]
    helped = h14 > 0.1
    hurt = h14 < -0.1

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
        "helped_authors": [authors[i] for i in np.where(helped)[0]],
        "hurt_authors": [authors[i] for i in np.where(hurt)[0]],
        "top_features": [(int(f), float(z[f]), float(diff[f])) for f in top_features],
    }


def find_discriminating_features(authors, matrix):
    """Features that best separate individual authors from the rest."""
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0) + 1e-8

    results = {}
    for i, author in enumerate(authors):
        z = (matrix[i] - global_mean) / global_std
        top_pos = np.argsort(z)[-3:][::-1]
        top_neg = np.argsort(z)[:3]
        results[author] = {
            "elevated": [(int(f), float(z[f])) for f in top_pos],
            "suppressed": [(int(f), float(z[f])) for f in top_neg],
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE features vs head knockout structure")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048",
                        help="SAE directory with author_feature_matrix.pt")
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    authors, matrix, knockout = load_data(sae_dir)
    n_authors, n_features = matrix.shape
    n_heads = knockout.shape[1]
    n_alive = (matrix.mean(axis=0) > 0.01).sum()

    print(f"=== SAE Feature ↔ Head Analysis ===")
    print(f"SAE dir: {sae_dir}")
    print(f"{n_authors} authors, {n_features} features ({n_alive} alive), {n_heads} heads\n")

    # 1. Feature-head correlation
    print("── Feature-Head Correlations ──")
    corr, pval, alive = feature_head_correlation(matrix, knockout)

    for h in range(n_heads):
        top = np.argsort(np.abs(corr[:, h]))[-5:][::-1]
        sig = [(int(f), corr[f, h], pval[f, h]) for f in top
               if pval[f, h] < 0.05 and alive[f]]
        if sig:
            feats = ", ".join(f"f{f}({r:+.3f} p={p:.1e})" for f, r, p in sig)
            print(f"  H{h:2d}: {feats}")
        else:
            print(f"  H{h:2d}: no significant correlations")

    # Count significant features per head (p<0.01, |r|>0.3)
    sig_counts = {}
    for h in range(n_heads):
        count = sum(1 for f in range(n_features)
                    if alive[f] and pval[f, h] < 0.01 and abs(corr[f, h]) > 0.3)
        sig_counts[h] = count

    print(f"\n  Significant features per head (p<0.01, |r|>0.3):")
    for h in range(n_heads):
        bar = "█" * min(sig_counts[h], 120)
        print(f"    H{h:2d}: {sig_counts[h]:3d} {bar}")

    # 2. PCA
    print("\n── Author Clustering (PCA of feature space) ──")
    coords, var_explained, best_head = author_clustering(authors, matrix, knockout)
    print(f"  Variance explained: PC1={var_explained[0]:.1%}, "
          f"PC2={var_explained[1]:.1%}, PC3={var_explained[2]:.1%}")

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
        print(f"  H14 helps {h14_result['n_helped']} authors: "
              f"{', '.join(h14_result['helped_authors'][:8])}")
        print(f"  H14 hurts {h14_result['n_hurt']} authors: "
              f"{', '.join(h14_result['hurt_authors'][:8])}")
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
    alive_matrix = matrix[:, alive]
    centered = alive_matrix - alive_matrix.mean(axis=0)
    S = np.linalg.svd(centered, compute_uv=False)
    var = S ** 2 / (S ** 2).sum()
    cumvar = np.cumsum(var)
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        n = int(np.searchsorted(cumvar, threshold)) + 1
        print(f"  {threshold:.0%} variance in {n} dimensions")

    # 6. Head-independent features (MLP axis)
    print("\n── Head-Independent Features (MLP axis candidates) ──")
    uncorr = []
    for fi in range(n_features):
        if not alive[fi]:
            continue
        max_r = max(abs(corr[fi, h]) for h in range(n_heads))
        if max_r < 0.2:
            mean_act = matrix[:, fi].mean()
            uncorr.append((fi, max_r, mean_act))
    uncorr.sort(key=lambda x: -x[2])  # sort by mean activation
    print(f"  {len(uncorr)} features uncorrelated with any head (max|r| < 0.2)")
    for fi, mr, ma in uncorr[:15]:
        # Which authors use this feature most?
        top_a = matrix[:, fi].argsort()[-3:][::-1]
        astr = ", ".join(authors[a] for a in top_a)
        print(f"    f{fi:4d}: mean_act={ma:.3f}, max|r|={mr:.3f}  top: {astr}")

    # Save
    results = {
        "sae_dir": str(sae_dir),
        "n_features": n_features,
        "n_alive": int(alive.sum()),
        "sig_features_per_head": {f"H{h}": sig_counts[h] for h in range(n_heads)},
        "pca_variance_explained": var_explained[:10].tolist(),
        "h14_polarization": h14_result,
        "effective_dim": {
            f"{t:.0%}": int(np.searchsorted(cumvar, t)) + 1
            for t in [0.5, 0.8, 0.9, 0.95]
        },
        "head_independent_features": [
            {"feature": fi, "max_r": float(mr), "mean_act": float(ma)}
            for fi, mr, ma in uncorr
        ],
    }

    out_path = sae_dir / "feature_head_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
