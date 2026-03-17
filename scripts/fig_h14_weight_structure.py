#!/usr/bin/env python3
"""H14's weight structure: the V-sensitivity outlier.

Shows V-sensitivity per head alongside V-norm, making clear that
H14 is an outlier — most sensitive despite having the smallest V.

Requires: outputs/why_h14.json (from why_h14.py)
          outputs/knockout_all_heads.json (from knockout.py)

Usage:
    uv run python scripts/fig_h14_weight_structure.py

Outputs:
    figures/h14_weight_structure.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FIG = Path("figures/h14_weight_structure.png")
DATA_PATH = Path("outputs/why_h14.json")
NUM_HEADS = 16


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    with open("outputs/knockout_all_heads.json") as f:
        knockout_data = json.load(f)

    heads = [f"H{h}" for h in range(NUM_HEADS)]

    v_sens = [data["v_sensitivity"][h]["mean_logit_diff"] for h in heads]
    v_norms = [data["norms"][h]["v"] for h in heads]

    # Knockout recovery std per head
    recovery_std = []
    for h in heads:
        vals = [ad["head_recovery"][h] for ad in knockout_data.values()
                if isinstance(ad, dict) and "head_recovery" in ad]
        recovery_std.append(np.std(vals))

    structural = {4, 6, 8, 12}

    # Sort heads by V-sensitivity (descending)
    order = sorted(range(NUM_HEADS), key=lambda h: -v_sens[h])
    sorted_heads = [heads[h] for h in order]
    sorted_v_sens = [v_sens[h] for h in order]
    sorted_v_norms = [v_norms[h] for h in order]
    sorted_rec_std = [recovery_std[h] for h in order]

    def get_color(h_idx):
        if h_idx == 14:
            return "#991b1b"
        elif h_idx == 11:
            return "#1e40af"
        elif h_idx in structural:
            return "#f59e0b"
        return "#8b5cf6"

    colors = [get_color(order[i]) for i in range(NUM_HEADS)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    x = np.arange(NUM_HEADS)

    # Top: V-sensitivity (what we're explaining)
    ax1.bar(x, sorted_v_sens, color=colors, alpha=0.8, edgecolor="white",
            linewidth=0.5)
    ax1.set_ylabel("V-perturbation sensitivity\n(mean |logit change|)", fontsize=9)
    ax1.set_title("H14 is the most V-sensitive head in the base model",
                  fontsize=12, fontweight="bold")

    # Add V-norm as text on each bar
    for i in range(NUM_HEADS):
        ax1.text(i, sorted_v_sens[i] + 0.003,
                f"V={sorted_v_norms[i]:.1f}",
                ha="center", fontsize=6.5, color="#555", rotation=45)

    # Bottom: knockout recovery std (the consequence)
    ax2.bar(x, sorted_rec_std, color=colors, alpha=0.8, edgecolor="white",
            linewidth=0.5)
    ax2.set_ylabel("Knockout recovery std\n(across 77 authors)", fontsize=9)
    ax2.set_xlabel("Heads sorted by V-sensitivity (highest → lowest)", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_heads, fontsize=8)

    r = np.corrcoef(v_sens, recovery_std)[0, 1]
    ax2.set_title(f"More V-sensitive → more polarizing (r = {r:+.2f})",
                  fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#991b1b", alpha=0.8, label="H14"),
        Patch(facecolor="#1e40af", alpha=0.8, label="H11"),
        Patch(facecolor="#8b5cf6", alpha=0.8, label="Semantic"),
        Patch(facecolor="#f59e0b", alpha=0.8, label="Structural"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
              fontsize=8, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")
    print(f"r(V_sens, recovery_std) = {r:+.3f}")


if __name__ == "__main__":
    main()