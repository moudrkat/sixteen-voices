#!/usr/bin/env python3
"""PCA of LoRA adapter weight deltas across all 82 authors.

Flattens each adapter's ΔW (q_proj + v_proj, each [1024, 1024]) into
a single vector, stacks them into an [82, 2*1024*1024] matrix, and
runs PCA. Reports:
  - Variance explained by top-k components
  - What the top components look like (which authors load on which PCs)
  - Whether PCs correlate with H14 recovery / V-Q balance

Usage:
    uv run python scripts/adapter_pca.py

Outputs:
    outputs/adapter_pca.json
    figures/adapter_pca.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices.adapter import load_adapter_deltas

ADAPTERS_DIR = Path("outputs/authors")
VQ_PATH = Path("outputs/h14_vq_balance.json")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
OUTPUT_JSON = Path("outputs/adapter_pca.json")
OUTPUT_FIG = Path("figures/adapter_pca.png")

LABEL_AUTHORS = [
    "browne", "poe", "homer", "melville", "milton",
    "carroll", "grimm",
    "shelley", "twain", "burnett", "barrie",
    "unusual_vocab", "dialogue", "firstperson", "minimalist", "poet",
]

SYNTHETIC_AUTHORS = {
    "unusual_vocab": "vocabulary",
    "reporter": "vocabulary",
    "simple_vocab": "vocabulary",
    "poet": "mixed",
    "dark": "mixed",
    "cozy": "mixed",
    "dialogue": "structure",
    "firstperson": "structure",
    "questioner": "structure",
    "repeater": "structure",
    "minimalist": "structure",
    "fabulist": "structure",
    "rambler": "structure",
}


def main():
    with open(VQ_PATH) as f:
        vq_data = json.load(f)
    vq_by_author = {d["author"]: d for d in vq_data["per_author"]}

    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)

    # Load all adapter deltas
    authors = []
    vectors = []

    for author in sorted(ko.keys()):
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if not adapter_path.exists():
            continue

        deltas = load_adapter_deltas(str(adapter_path))
        # Flatten and concatenate q_proj + v_proj
        q_flat = deltas["q_proj"].flatten().numpy()
        v_flat = deltas["v_proj"].flatten().numpy()
        vec = np.concatenate([q_flat, v_flat])

        authors.append(author)
        vectors.append(vec)

    X = np.stack(vectors)  # [n_authors, 2*1024*1024]
    n, d = X.shape
    print(f"Matrix: {n} authors × {d:,} dimensions")

    # Center
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean

    # PCA via SVD (much faster than computing covariance for tall-skinny)
    # X_centered = U @ S @ Vt, where U is [n, n], S is [n], Vt is [n, d]
    print("Running SVD...")
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Variance explained
    var_explained = S ** 2 / np.sum(S ** 2)
    cumvar = np.cumsum(var_explained)

    print(f"\nVariance explained by top components:")
    for k in [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]:
        if k <= len(var_explained):
            print(f"  Top {k:3d}: {cumvar[k-1]:.3f}")

    # PC scores (projections)
    scores = U * S  # [n, n] — each row is an author's coordinates in PC space

    # Correlate top PCs with H14 recovery and V-Q balance
    h14 = np.array([ko[a]["head_recovery"]["H14"] for a in authors])
    vq = np.array([vq_by_author[a]["vq_balance"] if a in vq_by_author else 0 for a in authors])

    print(f"\n{'PC':>4s}  {'var%':>6s}  {'r(H14)':>8s}  {'r(V-Q)':>8s}")
    print("-" * 32)
    pc_correlations = []
    for pc in range(min(20, n)):
        rv_h14 = float(np.corrcoef(scores[:, pc], h14)[0, 1])
        rv_vq = float(np.corrcoef(scores[:, pc], vq)[0, 1])
        flag = " ***" if abs(rv_h14) > 0.3 or abs(rv_vq) > 0.3 else ""
        print(f"  PC{pc:02d}  {var_explained[pc]:.3f}   {rv_h14:+.3f}     {rv_vq:+.3f}{flag}")
        pc_correlations.append({
            "pc": pc,
            "var_explained": float(var_explained[pc]),
            "cumvar": float(cumvar[pc]),
            "r_h14": rv_h14,
            "r_vq": rv_vq,
        })

    # --- Figure: 4 panels ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel 1: Scree plot
    ax = axes[0, 0]
    ax.bar(range(min(30, n)), var_explained[:30], color="#555", alpha=0.7)
    ax.plot(range(min(30, n)), cumvar[:30], 'o-', color="#991b1b", markersize=3)
    ax.set_xlabel("Principal component", fontsize=9)
    ax.set_ylabel("Variance explained", fontsize=9)
    ax.set_title("Scree plot: adapter weight PCA", fontsize=11, fontweight="bold")
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.5)
    # Mark where cumvar hits 90%
    k90 = np.searchsorted(cumvar, 0.9) + 1
    ax.text(k90, 0.91, f"90% at PC{k90}", fontsize=8, color="gray")

    # Panel 2: PC1 vs PC2, colored by H14 recovery
    ax = axes[0, 1]
    colors = ["#991b1b" if h > 0.2 else "#1e40af" if h < -0.2 else "#888888"
              for h in h14]
    ax.scatter(scores[:, 0], scores[:, 1], c=colors, s=30, alpha=0.7,
               edgecolors="white", linewidth=0.3)

    author_to_idx = {a: i for i, a in enumerate(authors)}
    for author in LABEL_AUTHORS:
        if author not in author_to_idx:
            continue
        i = author_to_idx[author]
        color = "#991b1b" if h14[i] > 0.2 else \
                "#1e40af" if h14[i] < -0.2 else "#666666"
        ax.annotate(
            author.replace("_", " ").capitalize(),
            xy=(scores[i, 0], scores[i, 1]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    r0_h14 = float(np.corrcoef(scores[:, 0], h14)[0, 1])
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)", fontsize=9)
    ax.set_title(f"Adapter PCA: PC1 vs PC2\ncolored by H14 recovery",
                 fontsize=11, fontweight="bold")

    # Panel 3: Find the PC most correlated with H14, plot it
    best_pc_h14 = max(range(min(20, n)), key=lambda k: abs(pc_correlations[k]["r_h14"]))
    best_r = pc_correlations[best_pc_h14]["r_h14"]

    ax = axes[1, 0]
    ax.scatter(scores[:, best_pc_h14], h14, c=colors, s=30, alpha=0.7,
               edgecolors="white", linewidth=0.3)
    slope, intercept = np.polyfit(scores[:, best_pc_h14], h14, 1)
    x_line = np.linspace(scores[:, best_pc_h14].min(), scores[:, best_pc_h14].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#666", linestyle="--",
            linewidth=1, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)

    for author in LABEL_AUTHORS:
        if author not in author_to_idx:
            continue
        i = author_to_idx[author]
        color = "#991b1b" if h14[i] > 0.2 else \
                "#1e40af" if h14[i] < -0.2 else "#666666"
        ax.annotate(
            author.replace("_", " ").capitalize(),
            xy=(scores[i, best_pc_h14], h14[i]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax.set_xlabel(f"PC{best_pc_h14} score ({var_explained[best_pc_h14]:.1%} var)", fontsize=9)
    ax.set_ylabel("H14 knockout recovery", fontsize=9)
    ax.set_title(f"Best H14 predictor: PC{best_pc_h14}\n"
                 f"r = {best_r:+.2f}",
                 fontsize=11, fontweight="bold")

    # Panel 4: Find the PC most correlated with V-Q, plot it
    best_pc_vq = max(range(min(20, n)), key=lambda k: abs(pc_correlations[k]["r_vq"]))
    best_r_vq = pc_correlations[best_pc_vq]["r_vq"]

    ax = axes[1, 1]
    ax.scatter(scores[:, best_pc_vq], vq, c=colors, s=30, alpha=0.7,
               edgecolors="white", linewidth=0.3)
    slope3, intercept3 = np.polyfit(scores[:, best_pc_vq], vq, 1)
    x_line3 = np.linspace(scores[:, best_pc_vq].min(), scores[:, best_pc_vq].max(), 100)
    ax.plot(x_line3, slope3 * x_line3 + intercept3, color="#666", linestyle="--",
            linewidth=1, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)

    for author in LABEL_AUTHORS:
        if author not in author_to_idx:
            continue
        i = author_to_idx[author]
        color = "#991b1b" if h14[i] > 0.2 else \
                "#1e40af" if h14[i] < -0.2 else "#666666"
        ax.annotate(
            author.replace("_", " ").capitalize(),
            xy=(scores[i, best_pc_vq], vq[i]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax.set_xlabel(f"PC{best_pc_vq} score ({var_explained[best_pc_vq]:.1%} var)", fontsize=9)
    ax.set_ylabel("H14 V-Q balance", fontsize=9)
    ax.set_title(f"Best V-Q predictor: PC{best_pc_vq}\n"
                 f"r = {best_r_vq:+.2f}",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save PCA components for hypernetwork training
    pca_path = Path("outputs/adapter_pca_components.npz")
    np.savez(
        pca_path,
        mean=X_mean,
        components=Vt,  # [n, d] — top n principal directions
        singular_values=S,
        scores=scores,  # [n, n] — author projections
        authors=np.array(authors),
    )
    print(f"Saved PCA components → {pca_path}")

    # Top/bottom authors on key PCs
    for pc in [0, 1, best_pc_h14]:
        print(f"\n=== PC{pc} top/bottom authors ===")
        order = np.argsort(scores[:, pc])
        print("  Bottom:", ", ".join(f"{authors[i]}({scores[i,pc]:+.1f})" for i in order[:5]))
        print("  Top:   ", ", ".join(f"{authors[i]}({scores[i,pc]:+.1f})" for i in order[-5:][::-1]))

    # Save JSON
    output = {
        "n_authors": n,
        "n_dimensions": d,
        "variance_explained": [float(v) for v in var_explained[:min(50, n)]],
        "cumulative_variance": [float(v) for v in cumvar[:min(50, n)]],
        "pc_correlations": pc_correlations,
        "k_for_90pct": int(k90),
        "per_author": [
            {
                "author": authors[i],
                "h14_recovery": float(h14[i]),
                "vq_balance": float(vq[i]),
                **{f"PC{k}": float(scores[i, k]) for k in range(min(10, n))},
            }
            for i in range(n)
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()