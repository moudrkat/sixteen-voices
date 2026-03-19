#!/usr/bin/env python3
"""Carroll steering figure: H11 (backbone) vs H14 (style) side by side.

Shows text at 0x, 0.5x, 1x, 1.5x, 2x for both heads.

Usage:
    python scripts/fig_carroll_steering.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path("figures")
DATA_PATH = Path("outputs/carroll_steering.json")

FONT_PROSE = "Noto Serif Display"
C_H11 = "#2980B9"    # blue for backbone
C_H14 = "#C44E52"    # red for style
C_TEXT = "#333333"

SCALE_LABELS = {
    "0.0": "0×  killed",
    "0.5": "0.5×",
    "1.0": "1×  normal",
    "1.5": "1.5×",
    "2.0": "2×  amplified",
}

SCALE_BG = {
    "0.0": 0.15,
    "0.5": 0.07,
    "1.0": 0.0,
    "1.5": 0.07,
    "2.0": 0.15,
}


def make_figure(data):
    heads = data["heads"]
    prompt = data["prompt"]
    seed = data["seed"]

    scales = ["0.0", "0.5", "1.0", "1.5", "2.0"]
    n = len(scales)

    fig, axes = plt.subplots(n, 2, figsize=(16, n * 2.2 + 1.5),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.08})

    fig.suptitle(
        "Steering Carroll:  H11 (backbone) vs H14 (style)",
        fontsize=18, fontweight="bold", y=0.98)

    for col, (head_key, color, subtitle) in enumerate([
        ("H11", C_H11, "H11 — backbone head (recovery 0.42)"),
        ("H14", C_H14, "H14 — style head (recovery 0.19)"),
    ]):
        axes[0, col].set_title(subtitle, fontsize=13, fontweight="bold",
                               color=color, pad=10)

        for row, scale in enumerate(scales):
            ax = axes[row, col]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            entry = heads[head_key][scale]
            text = entry["text"]
            ppl = entry["ppl"]

            bg_alpha = SCALE_BG[scale]
            bg_color = tuple(list(plt.matplotlib.colors.to_rgb(color)) + [bg_alpha])
            border_alpha = 0.3 if scale == "1.0" else 0.6
            lw = 1.0 if scale == "1.0" else 1.5

            ax.add_patch(FancyBboxPatch(
                (0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.02",
                facecolor=bg_color, edgecolor=color, linewidth=lw,
                alpha=border_alpha, transform=ax.transAxes))

            # Scale label
            label = SCALE_LABELS[scale]
            label_weight = "bold" if scale in ("0.0", "2.0") else "normal"
            ax.text(0.03, 0.88, label, fontsize=11, fontweight=label_weight,
                    color=color, transform=ax.transAxes, va="top")

            # PPL
            ax.text(0.97, 0.88, f"PPL: {ppl:.1f}", fontsize=9,
                    color="#999999", transform=ax.transAxes, va="top", ha="right")

            # Text
            wrapped = textwrap.fill(text[:180], width=55)
            ax.text(0.03, 0.62, wrapped, fontsize=9.5, color=C_TEXT,
                    style="italic", fontfamily=FONT_PROSE,
                    transform=ax.transAxes, va="top", linespacing=1.3)

    fig.text(0.5, 0.005,
             f'Prompt: "{prompt}" · seed={seed} · TinyStories-1Layer-21M · '
             f'LoRA rank 8 · scaling one head at inference',
             ha="center", fontsize=9, color="#aaaaaa")

    fig_path = FIG_DIR / "carroll_steering.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {fig_path}")


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    make_figure(data)


if __name__ == "__main__":
    main()