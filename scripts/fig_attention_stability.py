#!/usr/bin/env python3
"""Do attention patterns change after LoRA? (Spoiler: no.)

Compares per-head attention entropy between the base model and all 82
adapted models. Shows that LoRA changes what heads output, not how
they attend.

Requires: outputs/head_attention_patterns.json
  (produced by: uv run python scripts/head_attention_patterns.py --with-adapters <all>)

Usage:
    uv run python scripts/fig_attention_stability.py

Outputs:
    outputs/attention_stability.json
    figures/attention_stability.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_JSON = Path("outputs/attention_stability.json")
OUTPUT_FIG = Path("figures/attention_stability.png")
PATTERNS_PATH = Path("outputs/head_attention_patterns.json")
NUM_HEADS = 16


def main():
    with open(PATTERNS_PATH) as f:
        data = json.load(f)

    base = data["base"]["heads"]
    authors = [k for k in data if k != "base"]
    n = len(authors)
    print(f"Base model + {n} adapted models")

    # Collect per-head entropy deltas across all authors
    head_names = [f"H{h}" for h in range(NUM_HEADS)]
    base_entropies = [base[h]["entropy"] for h in head_names]
    base_prev = [base[h]["prev_token_frac"] for h in head_names]

    entropy_deltas = np.zeros((n, NUM_HEADS))
    prev_deltas = np.zeros((n, NUM_HEADS))
    for i, author in enumerate(authors):
        for j, h in enumerate(head_names):
            entropy_deltas[i, j] = data[author]["heads"][h]["entropy"] - base[h]["entropy"]
            prev_deltas[i, j] = data[author]["heads"][h]["prev_token_frac"] - base[h]["prev_token_frac"]

    # Classify heads by base model pattern
    is_structural = []
    for h in head_names:
        pattern = base[h]["pattern"]
        is_structural.append("focused" in pattern or "previous-token" in pattern)

    # Summary stats
    results = {}
    print(f"\n{'Head':>4s}  {'base_ent':>8s}  {'mean_Δ':>7s}  {'max_|Δ|':>7s}  {'type':>8s}")
    print("-" * 42)
    for j, h in enumerate(head_names):
        mean_d = float(np.mean(entropy_deltas[:, j]))
        max_d = float(entropy_deltas[np.argmax(np.abs(entropy_deltas[:, j])), j])
        label = "struct" if is_structural[j] else "semantic"
        print(f"{h:>4s}  {base_entropies[j]:8.2f}  {mean_d:+7.3f}  {max_d:+7.3f}  {label:>8s}")
        results[h] = {
            "base_entropy": base_entropies[j],
            "mean_delta": mean_d,
            "max_abs_delta": float(np.max(np.abs(entropy_deltas[:, j]))),
            "std_delta": float(np.std(entropy_deltas[:, j])),
            "type": label,
        }

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Box plot of entropy deltas per head
    # Sort by base entropy (structural first)
    order = sorted(range(NUM_HEADS), key=lambda j: base_entropies[j])
    box_data = [entropy_deltas[:, j] for j in order]
    labels = [head_names[j] for j in order]

    bp = ax1.boxplot(box_data, labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5),
                     flierprops=dict(markersize=3, alpha=0.5))
    for i, patch in enumerate(bp["boxes"]):
        j = order[i]
        if is_structural[j]:
            patch.set_facecolor("#f59e0b")
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor("#8b5cf6")
            patch.set_alpha(0.6)

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Entropy change after LoRA", fontsize=9)
    ax1.set_title(f"Attention entropy: base vs adapted ({n} authors)\n"
                  "orange = structural, purple = semantic",
                  fontsize=11, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: Base entropy vs mean absolute delta
    mean_abs = [np.mean(np.abs(entropy_deltas[:, j])) for j in range(NUM_HEADS)]
    colors = ["#f59e0b" if is_structural[j] else "#8b5cf6" for j in range(NUM_HEADS)]

    ax2.scatter(base_entropies, mean_abs, c=colors, s=60, alpha=0.8,
                edgecolors="white", linewidth=0.5, zorder=3)

    for j, h in enumerate(head_names):
        ax2.annotate(h, xy=(base_entropies[j], mean_abs[j]),
                     xytext=(5, 3), textcoords="offset points",
                     fontsize=8, fontweight="bold",
                     color="#b45309" if is_structural[j] else "#6d28d9",
                     alpha=0.8)

    ax2.set_xlabel("Base model entropy (structural ← → semantic)", fontsize=9)
    ax2.set_ylabel("Mean |entropy change| after LoRA", fontsize=9)
    ax2.set_title("LoRA barely changes attention patterns\n"
                  "structural and semantic heads stay in place",
                  fontsize=11, fontweight="bold")

    # Add a reference line
    ax2.axhline(np.mean(mean_abs), color="gray", linewidth=0.5, linestyle=":",
                alpha=0.5)
    ax2.text(max(base_entropies) * 0.95, np.mean(mean_abs) + 0.005,
             f"mean = {np.mean(mean_abs):.3f}", fontsize=7, color="gray",
             ha="right")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "n_authors": n,
        "per_head": results,
        "overall_mean_abs_delta": float(np.mean(np.abs(entropy_deltas))),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()