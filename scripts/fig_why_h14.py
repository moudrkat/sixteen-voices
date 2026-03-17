#!/usr/bin/env python3
"""Focused figure: why H14 is the polarizing head.

Shows that H14 is the most sensitive head in the base model —
small weight changes produce the largest output changes.

Requires: outputs/why_h14.json (produced by why_h14.py)

Usage:
    uv run python scripts/fig_why_h14.py

Outputs:
    figures/why_h14_sensitivity.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FIG = Path("figures/why_h14_sensitivity.png")
DATA_PATH = Path("outputs/why_h14.json")
NUM_HEADS = 16


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    heads = [f"H{h}" for h in range(NUM_HEADS)]

    q_sens = [data["q_sensitivity"][h]["mean_kl"] for h in heads]
    v_sens = [data["v_sensitivity"][h]["mean_logit_diff"] for h in heads]

    # Classify heads (from attention patterns)
    structural = {4, 6, 8, 12}  # previous-token / local-window

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(NUM_HEADS)

    def make_colors(values):
        colors = []
        for h in range(NUM_HEADS):
            if h == 14:
                colors.append("#991b1b")
            elif h == 11:
                colors.append("#1e40af")
            elif h in structural:
                colors.append("#f59e0b")
            else:
                colors.append("#8b5cf6")
        return colors

    colors = make_colors(q_sens)

    # Panel 1: Q sensitivity
    bars1 = ax1.bar(x, q_sens, color=colors, alpha=0.8, edgecolor="white",
                    linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(heads, fontsize=8)
    ax1.set_ylabel("Attention KL divergence after Q perturbation", fontsize=9)
    ax1.set_title("Q sensitivity: how much does attention move?",
                  fontsize=11, fontweight="bold")

    # Annotate H14
    h14_q_rank = sorted(range(16), key=lambda h: -q_sens[h]).index(14) + 1
    ax1.annotate(f"rank #{h14_q_rank}",
                xy=(14, q_sens[14]), xytext=(14, q_sens[14] * 0.85),
                fontsize=9, fontweight="bold", color="#991b1b",
                ha="center")

    # Panel 2: V sensitivity
    bars2 = ax2.bar(x, v_sens, color=colors, alpha=0.8, edgecolor="white",
                    linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(heads, fontsize=8)
    ax2.set_ylabel("Mean |logit change| after V perturbation", fontsize=9)
    ax2.set_title("V sensitivity: how much does output change?",
                  fontsize=11, fontweight="bold")

    # Annotate H14
    ax2.annotate(f"rank #1",
                xy=(14, v_sens[14]), xytext=(14, v_sens[14] * 0.92),
                fontsize=9, fontweight="bold", color="#991b1b",
                ha="center")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#991b1b", alpha=0.8, label="H14"),
        Patch(facecolor="#1e40af", alpha=0.8, label="H11"),
        Patch(facecolor="#8b5cf6", alpha=0.8, label="Semantic heads"),
        Patch(facecolor="#f59e0b", alpha=0.8, label="Structural heads"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
              ncol=4, fontsize=9, frameon=False,
              bbox_to_anchor=(0.5, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")


if __name__ == "__main__":
    main()