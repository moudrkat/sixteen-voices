#!/usr/bin/env python3
"""Does V-sensitivity predict which heads polarize?

Plots base model V-sensitivity vs knockout recovery variance for all
16 heads. If H14 polarizes because it's the most V-sensitive, then
other sensitive heads should also show higher variance.

Requires:
    outputs/why_h14.json (from why_h14.py)
    outputs/knockout_all_heads.json (from knockout.py)

Usage:
    uv run python scripts/fig_sensitivity_vs_variance.py

Outputs:
    figures/sensitivity_vs_variance.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FIG = Path("figures/sensitivity_vs_variance.png")
NUM_HEADS = 16


def main():
    with open("outputs/why_h14.json") as f:
        sensitivity_data = json.load(f)

    with open("outputs/knockout_all_heads.json") as f:
        knockout_data = json.load(f)

    heads = [f"H{h}" for h in range(NUM_HEADS)]

    # V-sensitivity per head
    v_sens = [sensitivity_data["v_sensitivity"][h]["mean_logit_diff"] for h in heads]
    q_sens = [sensitivity_data["q_sensitivity"][h]["mean_kl"] for h in heads]

    # Knockout recovery stats per head
    recovery_std = []
    recovery_mean = []
    for h in heads:
        vals = [author_data["head_recovery"][h]
                for author_data in knockout_data.values()
                if isinstance(author_data, dict) and "head_recovery" in author_data
                and h in author_data["head_recovery"]]
        recovery_std.append(np.std(vals))
        recovery_mean.append(np.mean(vals))

    # Classify heads
    structural = {4, 6, 8, 12}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for h_idx in range(NUM_HEADS):
        if h_idx == 14:
            color, marker, s = "#991b1b", "s", 100
        elif h_idx == 11:
            color, marker, s = "#1e40af", "s", 100
        elif h_idx in structural:
            color, marker, s = "#f59e0b", "o", 60
        else:
            color, marker, s = "#8b5cf6", "o", 60

        ax1.scatter(v_sens[h_idx], recovery_std[h_idx],
                   c=color, marker=marker, s=s, alpha=0.8,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax1.annotate(heads[h_idx],
                    xy=(v_sens[h_idx], recovery_std[h_idx]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=8, fontweight="bold", alpha=0.7)

        ax2.scatter(q_sens[h_idx], recovery_std[h_idx],
                   c=color, marker=marker, s=s, alpha=0.8,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax2.annotate(heads[h_idx],
                    xy=(q_sens[h_idx], recovery_std[h_idx]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=8, fontweight="bold", alpha=0.7)

    # Correlation
    r_v = np.corrcoef(v_sens, recovery_std)[0, 1]
    r_q = np.corrcoef(q_sens, recovery_std)[0, 1]

    ax1.set_xlabel("V-perturbation sensitivity (base model)", fontsize=9)
    ax1.set_ylabel("Knockout recovery std (across 77 authors)", fontsize=9)
    ax1.set_title(f"V sensitivity vs polarization (r = {r_v:+.2f})",
                  fontsize=11, fontweight="bold")

    ax2.set_xlabel("Q-perturbation sensitivity (base model)", fontsize=9)
    ax2.set_ylabel("Knockout recovery std (across 77 authors)", fontsize=9)
    ax2.set_title(f"Q sensitivity vs polarization (r = {r_q:+.2f})",
                  fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#991b1b",
               markersize=10, label="H14"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#1e40af",
               markersize=10, label="H11"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#8b5cf6",
               markersize=8, label="Semantic"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#f59e0b",
               markersize=8, label="Structural"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
              ncol=4, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")
    print(f"r(V_sens, recovery_std) = {r_v:+.3f}")
    print(f"r(Q_sens, recovery_std) = {r_q:+.3f}")


if __name__ == "__main__":
    main()