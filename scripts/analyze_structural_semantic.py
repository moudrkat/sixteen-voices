#!/usr/bin/env python3
"""Analyze structural vs semantic features per author and per head.

Counts how many elevated features each author has in each category,
and how many features of each type flow through each head.

Usage:
    uv run python scripts/analyze_structural_semantic.py
    uv run python scripts/analyze_structural_semantic.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from scipy import stats


STRUCTURAL_FEATURES = {665, 883, 993, 60, 1777, 689, 329, 1385, 344,
                       1779, 627, 1604, 1524, 793}


def main():
    parser = argparse.ArgumentParser(
        description="Structural vs semantic feature analysis")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "structural_semantic.json")

    afm = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    authors = afm["authors"]
    matrix = afm["matrix"].numpy()
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0) + 1e-8

    with open("outputs/knockout_all_heads.json") as f:
        ko = json.load(f)
    knockout = np.array([
        [ko[a]["head_recovery"][f"H{h}"] for h in range(16)]
        for a in authors
    ])

    alive = matrix.mean(axis=0) > 0.01

    # Per-author analysis
    print("=== Features per author (z > 2) ===\n")
    author_results = []
    for author in authors:
        idx = authors.index(author)
        z = (matrix[idx] - global_mean) / global_std
        elevated = z > 2.0
        n_struct = sum(1 for fi in STRUCTURAL_FEATURES if elevated[fi])
        n_total = int(elevated.sum())
        n_sem = n_total - n_struct
        print(f"{author:>20s}: {n_struct:2d} structural, {n_sem:2d} semantic")
        author_results.append({
            "author": author,
            "n_structural": n_struct,
            "n_semantic": n_sem,
            "n_total": n_total,
        })

    # Per-head analysis
    print("\n=== Features per head (p<0.01, |r|>0.3) ===\n")
    head_results = []
    for h in range(16):
        n_struct = 0
        n_sem = 0
        for fi in range(matrix.shape[1]):
            if not alive[fi]:
                continue
            r, p = stats.pearsonr(matrix[:, fi], knockout[:, h])
            if p < 0.01 and abs(r) > 0.3:
                if fi in STRUCTURAL_FEATURES:
                    n_struct += 1
                else:
                    n_sem += 1
        total = n_struct + n_sem
        pct = f"{100 * n_struct / total:.0f}%" if total > 0 else "-"
        print(f"  H{h:2d}: {n_struct:3d} structural, {n_sem:3d} semantic "
              f"({pct} structural)")
        head_results.append({
            "head": h,
            "n_structural": n_struct,
            "n_semantic": n_sem,
        })

    # MLP features
    print("\n=== MLP (head-independent, max|r| < 0.2) ===\n")
    n_struct_mlp = 0
    n_sem_mlp = 0
    for fi in range(matrix.shape[1]):
        if not alive[fi]:
            continue
        max_r = max(
            abs(stats.pearsonr(matrix[:, fi], knockout[:, h])[0])
            for h in range(16))
        if max_r < 0.2:
            if fi in STRUCTURAL_FEATURES:
                n_struct_mlp += 1
            else:
                n_sem_mlp += 1
    print(f"  {n_struct_mlp} structural, {n_sem_mlp} semantic")

    results = {
        "structural_feature_indices": sorted(STRUCTURAL_FEATURES),
        "authors": author_results,
        "heads": head_results,
        "mlp": {"n_structural": n_struct_mlp, "n_semantic": n_sem_mlp},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()