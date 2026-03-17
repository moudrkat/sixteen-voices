#!/usr/bin/env python3
"""Mechanistic explanation of head roles from base model weights.

Shows that base model properties predict which heads matter for style:
- V-sensitivity (logit_impact / V_norm) predicts polarization (r=0.91)
- Q_norm predicts Q-sensitivity (r=0.76)
- Combined picture explains head roles

Requires:
    outputs/why_h14.json
    outputs/knockout_all_heads.json
    outputs/head_attention_patterns.json

Usage:
    uv run python scripts/fig_head_mechanics.py

Outputs:
    figures/head_mechanics.png
    outputs/head_mechanics.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FIG = Path("figures/head_mechanics.png")
OUTPUT_JSON = Path("outputs/head_mechanics.json")
NUM_HEADS = 16


def main():
    with open("outputs/why_h14.json") as f:
        sens_data = json.load(f)
    with open("outputs/knockout_all_heads.json") as f:
        knockout_data = json.load(f)
    with open("outputs/head_attention_patterns.json") as f:
        attn_data = json.load(f)

    heads = [f"H{h}" for h in range(16)]
    structural = {4, 6, 8, 12}

    # Extract data
    v_sens = [sens_data["v_sensitivity"][h]["mean_logit_diff"] for h in heads]
    q_sens = [sens_data["q_sensitivity"][h]["mean_kl"] for h in heads]
    v_norms = [sens_data["norms"][h]["v"] for h in heads]
    q_norms = [sens_data["norms"][h]["q"] for h in heads]
    wo_norms = [sens_data["output_proj"][h]["norm"] for h in heads]

    # Compute logit impacts (W_unembed @ W_O_h)
    # Pre-computed values from analysis
    logit_impacts = [56.564, 77.642, 78.091, 67.032, 56.652, 61.521, 56.526,
                     62.410, 61.943, 59.529, 72.581, 100.362, 54.199, 72.932,
                     68.555, 81.455]

    # Key ratios
    v_ratio = [logit_impacts[h] / v_norms[h] for h in range(NUM_HEADS)]

    # Knockout stats
    recovery_mean = []
    recovery_std = []
    for h in heads:
        vals = [ad["head_recovery"][h] for ad in knockout_data.values()
                if isinstance(ad, dict) and "head_recovery" in ad]
        recovery_mean.append(np.mean(vals))
        recovery_std.append(np.std(vals))

    # Base entropy
    base_entropy = [attn_data["base"]["heads"][h]["entropy"] for h in heads]

    def get_color(h):
        if h == 14: return "#991b1b"
        if h == 11: return "#1e40af"
        if h in structural: return "#f59e0b"
        return "#8b5cf6"

    def get_marker(h):
        if h in (11, 14): return "s"
        return "o"

    def get_size(h):
        if h in (11, 14): return 110
        return 60

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    def scatter_heads(ax, xs, ys, xlabel, ylabel, title):
        for h in range(NUM_HEADS):
            ax.scatter(xs[h], ys[h], c=get_color(h), marker=get_marker(h),
                      s=get_size(h), alpha=0.85, edgecolors="white",
                      linewidth=0.5, zorder=3 + (1 if h in (11,14) else 0))
            ax.annotate(heads[h], xy=(xs[h], ys[h]),
                       xytext=(5, 3), textcoords="offset points",
                       fontsize=7.5, fontweight="bold", alpha=0.7,
                       color=get_color(h))
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title}\nr = {r:+.2f}", fontsize=11, fontweight="bold")
        return r

    # Panel 1: logit_impact/V_norm → V-sensitivity
    r1 = scatter_heads(axes[0, 0], v_ratio, v_sens,
                       "Logit impact / V norm (amplification ratio)",
                       "V-perturbation sensitivity",
                       "Why H14 is V-sensitive")

    # Panel 2: V-sensitivity → knockout variance (polarization)
    r2 = scatter_heads(axes[0, 1], v_sens, recovery_std,
                       "V-perturbation sensitivity",
                       "Knockout recovery std (77 authors)",
                       "V-sensitive heads polarize")

    # Panel 3: Q_norm → Q-sensitivity
    r3 = scatter_heads(axes[1, 0], q_norms, q_sens,
                       "Q weight norm",
                       "Q-perturbation sensitivity (KL div)",
                       "Bigger Q weights → more Q-sensitive")

    # Panel 4: V-sensitivity → mean recovery (importance)
    r4 = scatter_heads(axes[1, 1], v_sens, recovery_mean,
                       "V-perturbation sensitivity",
                       "Mean knockout recovery (77 authors)",
                       "V-sensitive heads also matter more")

    # Legend
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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")

    # Also: the full chain
    # logit_impact/V_norm → V_sensitivity → recovery_std
    # Can we predict recovery_std directly from the ratio?
    r_direct = np.corrcoef(v_ratio, recovery_std)[0, 1]
    r_mean = np.corrcoef(v_ratio, recovery_mean)[0, 1]
    r_v_mean = np.corrcoef(v_sens, recovery_mean)[0, 1]
    print(f"\nFull chain:")
    print(f"  r(ratio, V_sens)       = {r1:+.3f}")
    print(f"  r(V_sens, rec_std)     = {r2:+.3f}")
    print(f"  r(ratio, rec_std)      = {r_direct:+.3f}  (direct)")
    print(f"  r(V_sens, rec_mean)    = {r_v_mean:+.3f}")
    print(f"  r(ratio, rec_mean)     = {r_mean:+.3f}")
    print(f"  r(Q_norm, Q_sens)      = {r3:+.3f}")

    # Save results
    results = {
        "per_head": {
            heads[h]: {
                "v_sensitivity": v_sens[h],
                "q_sensitivity": q_sens[h],
                "v_norm": v_norms[h],
                "q_norm": q_norms[h],
                "logit_impact": logit_impacts[h],
                "amplification_ratio": v_ratio[h],
                "recovery_mean": recovery_mean[h],
                "recovery_std": recovery_std[h],
                "base_entropy": base_entropy[h],
                "type": "structural" if h in structural else "semantic",
            }
            for h in range(NUM_HEADS)
        },
        "correlations": {
            "ratio_vs_v_sensitivity": r1,
            "v_sensitivity_vs_recovery_std": r2,
            "ratio_vs_recovery_std_direct": r_direct,
            "q_norm_vs_q_sensitivity": r3,
            "v_sensitivity_vs_recovery_mean": r_v_mean,
            "ratio_vs_recovery_mean": r_mean,
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()