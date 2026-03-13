#!/usr/bin/env python3
"""Head importance from LoRA weight norms — no inference needed.

For each trained adapter, computes the Frobenius norm of the ΔW slice
per head (64 rows each) for both Q and V projections. Produces:

1. Bar chart: mean relative head importance ± std across all authors
   (high std = author-specific head, low std = universal head)
2. Heatmap: per-author × per-head relative importance

Runs on whatever adapters are trained so far.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running from scripts/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sixteen_voices.adapter import load_adapter_deltas
from sixteen_voices.constants import NUM_HEADS, HEAD_DIM

AUTHORS_DIR = Path("outputs/authors")
FIGURES_DIR = Path("figures")


def per_head_norms(deltas: dict, proj: str) -> np.ndarray:
    """Frobenius norm of each head's 64-row slice of ΔW."""
    delta = deltas[proj]
    norms = []
    for h in range(NUM_HEADS):
        head_slice = delta[h * HEAD_DIM : (h + 1) * HEAD_DIM, :]
        norms.append(head_slice.norm().item())
    return np.array(norms)


def discover_adapters() -> dict[str, Path]:
    """Find all trained adapters in outputs/authors/."""
    adapters = {}
    for d in sorted(AUTHORS_DIR.iterdir()):
        adapter_dir = d / "adapter"
        if (adapter_dir / "adapter_model.safetensors").exists():
            adapters[d.name] = adapter_dir
    return adapters


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    adapters = discover_adapters()
    if not adapters:
        print("No trained adapters found in outputs/authors/")
        return

    authors = list(adapters.keys())
    print(f"Found {len(authors)} trained adapters")

    # Collect per-head norms
    head_data = {"q_proj": {}, "v_proj": {}}
    for author in authors:
        deltas = load_adapter_deltas(adapters[author])
        for proj in ["q_proj", "v_proj"]:
            head_data[proj][author] = per_head_norms(deltas, proj)

    # --- Figure 1: Strip plot (sorted by mean, jittered author dots) ---
    ACCENT_HIGH = "#C44E52"
    ACCENT_MID = "#DD8452"
    LIGHT_GRAY = "#cccccc"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    rng = np.random.default_rng(42)

    for ax, proj in zip(axes, ["q_proj", "v_proj"]):
        raw = np.array([head_data[proj][a] for a in authors])
        rel = raw / raw.sum(axis=1, keepdims=True)

        means = rel.mean(axis=0)
        order = list(np.argsort(means)[::-1])

        x = np.arange(NUM_HEADS)
        for xi, h in enumerate(order):
            vals = rel[:, h]
            jitter = rng.uniform(-0.2, 0.2, len(vals))
            color = (ACCENT_HIGH if means[h] > 0.07
                     else ACCENT_MID if means[h] > 0.06
                     else LIGHT_GRAY)
            ax.scatter(xi + jitter, vals, s=10, alpha=0.4, color=color,
                       zorder=2, edgecolors="none")
            ax.plot(xi, means[h], "D", color="black", markersize=5, zorder=4)

        ax.axhline(y=1.0 / NUM_HEADS, color="gray", linestyle="--",
                   alpha=0.5, label=f"uniform = {1/NUM_HEADS:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in order], fontsize=8)
        ax.set_ylabel("Relative importance (norm fraction)")
        ax.set_title(f"{proj} — head importance: means hide author-specific variance "
                     f"({len(authors)} authors)")
        ax.legend(fontsize=8)

        # Annotate the most variable head
        stds = rel.std(axis=0)
        most_var = order[0]  # already sorted by mean, find most-var instead
        most_var_h = int(np.argmax(stds))
        most_var_xi = order.index(most_var_h)
        ax.annotate(f"H{most_var_h}: highest variance\nacross authors",
                    xy=(most_var_xi, means[most_var_h]),
                    xytext=(most_var_xi + 3, means[most_var_h] + 0.015),
                    fontsize=8, color=ACCENT_HIGH, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT_HIGH, lw=1.2))

    fig.suptitle("Which heads carry the most LoRA weight change?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "head_importance.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'head_importance.png'}")
    plt.close()

    # --- Figure 2: Per-author heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(authors) * 0.28)))

    for ax, proj in zip(axes, ["q_proj", "v_proj"]):
        raw = np.array([head_data[proj][a] for a in authors])
        rel = raw / raw.sum(axis=1, keepdims=True)

        im = ax.imshow(rel, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(NUM_HEADS))
        ax.set_xticklabels([f"H{i}" for i in range(NUM_HEADS)], fontsize=8)
        ax.set_yticks(range(len(authors)))
        ax.set_yticklabels(authors, fontsize=8)
        ax.set_xlabel("Attention head")
        ax.set_title(f"{proj} — relative LoRA impact per head")
        plt.colorbar(im, ax=ax, label="Fraction of total norm", shrink=0.6)

    fig.suptitle("Per-author head importance (LoRA weight norms)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "head_importance_heatmap.png", dpi=150,
                bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'head_importance_heatmap.png'}")
    plt.close()

    # --- Figure 3: Author correlation (do authors use the same heads?) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 14))

    for ax, proj in zip(axes, ["q_proj", "v_proj"]):
        raw = np.array([head_data[proj][a] for a in authors])
        rel = raw / raw.sum(axis=1, keepdims=True)
        corr = np.corrcoef(rel)

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(authors)))
        ax.set_xticklabels(authors, rotation=90, fontsize=7)
        ax.set_yticks(range(len(authors)))
        ax.set_yticklabels(authors, fontsize=7)
        ax.set_title(f"{proj} — correlation of head profiles")
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Do different authors modify the same heads?\n"
                 "(1.0 = same head profile, −1.0 = opposite profile)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "head_correlation.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'head_correlation.png'}")
    plt.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("HEAD IMPORTANCE SUMMARY")
    print(f"{'=' * 60}")

    for proj in ["q_proj", "v_proj"]:
        raw = np.array([head_data[proj][a] for a in authors])
        rel = raw / raw.sum(axis=1, keepdims=True)
        mean = rel.mean(axis=0)
        std = rel.std(axis=0)

        print(f"\n{proj}:")
        print(f"  {'Head':<6s} {'Mean':>8s} {'Std':>8s}  bar")
        print(f"  {'-' * 40}")

        order = np.argsort(-mean)
        for h in order:
            bar = "#" * int(mean[h] * 200)
            print(f"  H{h:<4d} {mean[h]:>8.4f} {std[h]:>8.4f}  {bar}")

        var_order = np.argsort(-std)
        print(f"\n  Most variable (author-specific): "
              f"H{var_order[0]}, H{var_order[1]}, H{var_order[2]}")
        print(f"  Most stable (universal):         "
              f"H{var_order[-1]}, H{var_order[-2]}, H{var_order[-3]}")


if __name__ == "__main__":
    main()