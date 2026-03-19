#!/usr/bin/env python3
"""Clean knockout strip plot + H11-vs-H14 scatter.

Replaces the old cluttered strip plot with arrows.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
DATA_PATH = Path("outputs/knockout_all_heads.json")

C_H11 = "#2980B9"
C_H14 = "#C44E52"
C_H3 = "#6AB04C"
C_NEUTRAL = "#cccccc"
C_TEXT = "#333333"


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)

    heads = [f"H{i}" for i in range(16)]
    authors = sorted(raw.keys())
    recovery = {h: [] for h in heads}
    best_head = {}
    for a in authors:
        rec = raw[a]["head_recovery"]
        best = max(rec, key=rec.get)
        best_head[a] = best
        for h in heads:
            recovery[h].append(rec[h])
    return raw, heads, authors, recovery, best_head


def make_strip_plot(heads, authors, recovery):
    """Clean strip plot: dots + diamond means, no arrows, just labeled axes."""
    means = {h: np.mean(recovery[h]) for h in heads}
    order = sorted(heads, key=lambda h: means[h], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, h in enumerate(order):
        vals = recovery[h]

        # Color the head label by role
        if h == "H11":
            color = C_H11
        elif h == "H14":
            color = C_H14
        elif h == "H3":
            color = C_H3
        else:
            color = "#888888"

        # Jittered dots
        jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(vals))
        dot_color = "#dddddd" if h not in ("H11", "H14", "H3") else color
        dot_alpha = 0.25 if h not in ("H11", "H14", "H3") else 0.4
        ax.scatter([i] * len(vals) + jitter, vals,
                   s=15, color=dot_color, alpha=dot_alpha, zorder=2,
                   edgecolors="none")

        # Mean diamond + error bar
        m = means[h]
        s = np.std(vals)
        ax.errorbar(i, m, yerr=s, fmt="D", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5,
                    capsize=4, capthick=1.5, elinewidth=1.5, zorder=5)

    # Uniform baseline
    uniform = 1.0 / 16
    ax.axhline(uniform, color="#aaaaaa", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(order) - 0.5, uniform + 0.01, f"uniform = {uniform:.3f}",
            fontsize=8, color="#aaaaaa", ha="right")

    ax.axhline(0, color="#cccccc", linewidth=0.5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=11)
    # Color the tick labels
    for i, h in enumerate(order):
        if h == "H11":
            ax.get_xticklabels()[i].set_color(C_H11)
            ax.get_xticklabels()[i].set_fontweight("bold")
        elif h == "H14":
            ax.get_xticklabels()[i].set_color(C_H14)
            ax.get_xticklabels()[i].set_fontweight("bold")
        elif h == "H3":
            ax.get_xticklabels()[i].set_color(C_H3)
            ax.get_xticklabels()[i].set_fontweight("bold")

    ax.set_ylabel("Recovery score", fontsize=12)
    ax.set_xlabel("Attention head (sorted by mean recovery)", fontsize=12)
    ax.set_title("Per-head knockout recovery across 77 authors",
                 fontsize=14, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, len(order) - 0.5)

    plt.tight_layout()
    out = FIG_DIR / "knockout_strip_clean.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


def make_scatter(raw, authors):
    """H11 vs H14 recovery scatter — shows anticorrelation and two strategies."""
    h11 = [raw[a]["head_recovery"]["H11"] for a in authors]
    h14 = [raw[a]["head_recovery"]["H14"] for a in authors]
    h3 = [raw[a]["head_recovery"]["H3"] for a in authors]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: H11 vs H14 ---
    ax = axes[0]
    for i, a in enumerate(authors):
        best = max(raw[a]["head_recovery"], key=raw[a]["head_recovery"].get)
        if best == "H14":
            color = C_H14
            alpha = 0.8
        elif best == "H11":
            color = C_H11
            alpha = 0.5
        else:
            color = "#999999"
            alpha = 0.4
        ax.scatter(h11[i], h14[i], s=40, color=color, alpha=alpha,
                   edgecolors="white", linewidth=0.5, zorder=3)

    # Label a few interesting authors
    label_authors = {
        "homer": (-10, 10), "poe": (-10, 10), "melville": (8, 5),
        "carroll": (8, -10), "minimalist": (8, -10), "dialogue": (8, 5),
        "shelley": (8, -10), "browne": (-10, 10), "grimm": (8, 5),
    }
    for a, (dx, dy) in label_authors.items():
        if a in authors:
            idx = authors.index(a)
            ax.annotate(a, (h11[idx], h14[idx]), fontsize=7.5,
                        color="#666666", xytext=(dx, dy),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="#cccccc",
                                        lw=0.5))

    ax.axhline(0, color="#eeeeee", linewidth=0.5)
    ax.axvline(0, color="#eeeeee", linewidth=0.5)

    corr = np.corrcoef(h11, h14)[0, 1]
    ax.set_xlabel("H11 recovery", fontsize=12, color=C_H11)
    ax.set_ylabel("H14 recovery", fontsize=12, color=C_H14)
    ax.set_title(f"H11 vs H14  (r = {corr:.2f})", fontsize=13,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_H11,
               markersize=8, label="H11-dominant (51)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_H14,
               markersize=8, label="H14-dominant (18)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
               markersize=8, label="Other (8)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    # --- Right: H14 vs H3 ---
    ax = axes[1]
    for i, a in enumerate(authors):
        best = max(raw[a]["head_recovery"], key=raw[a]["head_recovery"].get)
        if best == "H14":
            color = C_H14
            alpha = 0.8
        elif best == "H11":
            color = C_H11
            alpha = 0.5
        else:
            color = "#999999"
            alpha = 0.4
        ax.scatter(h14[i], h3[i], s=40, color=color, alpha=alpha,
                   edgecolors="white", linewidth=0.5, zorder=3)

    corr_14_3 = np.corrcoef(h14, h3)[0, 1]
    ax.set_xlabel("H14 recovery", fontsize=12, color=C_H14)
    ax.set_ylabel("H3 recovery", fontsize=12, color=C_H3)
    ax.set_title(f"H14 vs H3  (r = {corr_14_3:.2f})", fontsize=13,
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(0, color="#eeeeee", linewidth=0.5)
    ax.axvline(0, color="#eeeeee", linewidth=0.5)

    fig.suptitle("Two adaptation strategies: H11-led vs H14-led",
                 fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = FIG_DIR / "knockout_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


def main():
    raw, heads, authors, recovery, best_head = load_data()
    make_strip_plot(heads, authors, recovery)
    make_scatter(raw, authors)


if __name__ == "__main__":
    main()