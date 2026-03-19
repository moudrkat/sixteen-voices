#!/usr/bin/env python3
"""Steering contrast figure: Carroll (H11-led) vs Poe (H14-led).

Two panels showing PPL curves for both heads in both authors.
The mirror-image pattern should be immediately visible.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures")
DATA_PATH = Path("outputs/steering_sweep.json")

C_H11 = "#2980B9"
C_H14 = "#C44E52"
C_H3 = "#6AB04C"
C_TEXT = "#333333"


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    authors = [
        ("carroll", "Carroll (H11-led)"),
        ("poe", "Poe (H14-led)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for col, (author, title) in enumerate(authors):
        ax = axes[col]
        curves = data[author]["curves"]
        full_ppl = float(list(curves.values())[0]["1.0"])
        rec = data[author]["head_recovery"]

        for head, color, ls in [
            ("H11", C_H11, "-"),
            ("H14", C_H14, "-"),
        ]:
            if head not in curves:
                continue
            curve = curves[head]
            scales = sorted(curve.keys(), key=float)
            xs = [float(s) for s in scales]
            ys = [float(curve[s]) for s in scales]

            label = f"{head} (recovery {rec[head]:.2f})"
            ax.plot(xs, ys, f"o{ls}", color=color, linewidth=2.5,
                    markersize=6, label=label, zorder=3)

        # Also plot H3 if available
        if "H3" in curves:
            curve = curves["H3"]
            scales = sorted(curve.keys(), key=float)
            xs = [float(s) for s in scales]
            ys = [float(curve[s]) for s in scales]
            label = f"H3 (recovery {rec['H3']:.2f})"
            ax.plot(xs, ys, "o-", color=C_H3, linewidth=2, markersize=5,
                    label=label, alpha=0.7, zorder=2)

        # Reference line at 1x
        ax.axvline(1.0, color="#dddddd", linewidth=1, zorder=1)

        ax.set_xlabel("Head scale", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.05, 2.05)

        # Annotate the kill effects
        for head, color in [("H11", C_H11), ("H14", C_H14)]:
            if head not in curves:
                continue
            kill_ppl = float(curves[head]["0.0"])
            delta = kill_ppl - full_ppl
            pct = delta / full_ppl * 100
            if pct > 15:  # only annotate if substantial
                ax.annotate(f"kill: +{pct:.0f}%",
                            xy=(0.0, kill_ppl),
                            xytext=(0.3, kill_ppl),
                            fontsize=8, color=color, fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.8),
                            va="center")

    fig.suptitle(
        "Steering the dominant head: same mechanism, different authors",
        fontsize=15, fontweight="bold", y=1.02)

    fig.text(0.5, -0.02,
             'Scaling one head\'s output at inference · TinyStories-1Layer-21M · '
             'LoRA rank 8 · seed=42',
             ha="center", fontsize=9, color="#aaaaaa")

    plt.tight_layout()
    out = FIG_DIR / "steering_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()