#!/usr/bin/env python3
"""Figure: Poe H14 steering — separate figure per prompt.

Shows how scaling H14 from 0× to 2× changes Poe's output.
Reads from outputs/poe_steering_prompts.json (seed=42).

Usage:
    uv run --extra viz python scripts/fig_poe_steering.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = Path("figures")
SAMPLES_JSON = Path("outputs/poe_steering_prompts.json")

PROMPTS = [
    "It was a dark and stormy",
    "In the dark of night",
]

SCALES = ["0.0", "0.5", "1.0", "1.5", "2.0"]

SCALE_COLORS = {
    "0.0": "#dbeafe",
    "0.5": "#e8f0fe",
    "1.0": "#f9fafb",
    "1.5": "#fef3c7",
    "2.0": "#fecaca",
}

SCALE_BORDERS = {
    "0.0": "#93c5fd",
    "0.5": "#bfdbfe",
    "1.0": "#d1d5db",
    "1.5": "#fcd34d",
    "2.0": "#f87171",
}

SCALE_LABELS = {
    "0.0": "0×    killed",
    "0.5": "0.5×",
    "1.0": "1×    normal",
    "1.5": "1.5×",
    "2.0": "2×    amplified",
}


def make_figure(prompt, samples, head, recovery, output_name):
    scales = SCALES

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.97,
            f'Steering Poe\'s H{head}  (recovery = {recovery:+.2f})',
            fontsize=16, fontweight="bold", ha="center", va="top",
            transform=ax.transAxes)
    ax.text(0.5, 0.93,
            f'Prompt: "{prompt}"  ·  seed = 42  ·  TinyStories-1Layer-21M + LoRA',
            fontsize=9, ha="center", va="top", color="#6b7280",
            transform=ax.transAxes)

    n = len(scales)
    box_height = 0.15
    gap = 0.02
    start_y = 0.88

    for i, scale in enumerate(scales):
        text = samples[scale]
        if len(text) > 220:
            cut = text[:220].rfind(" ")
            if cut < 100:
                cut = 220
            text = text[:cut] + " ..."

        wrapped = textwrap.fill(text, width=80)

        y_top = start_y - i * (box_height + gap)
        label = SCALE_LABELS[scale]
        bg = SCALE_COLORS[scale]
        border = SCALE_BORDERS[scale]

        rect = mpatches.FancyBboxPatch(
            (0.02, y_top - box_height), 0.96, box_height,
            boxstyle="round,pad=0.01",
            facecolor=bg, edgecolor=border, linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        label_color = "#1e40af" if float(scale) < 1 else "#991b1b" if float(scale) > 1 else "#374151"
        ax.text(0.04, y_top - 0.012, label,
                fontsize=10, fontweight="bold", va="top",
                color=label_color, transform=ax.transAxes)

        ax.text(0.04, y_top - 0.038, wrapped,
                fontsize=8.5, va="top", family="serif", style="italic",
                color="#1f2937", transform=ax.transAxes,
                linespacing=1.4)

    # Arrow
    arrow_x = 0.008
    bottom_y = start_y - (n - 1) * (box_height + gap) - box_height + 0.02
    ax.annotate("", xy=(arrow_x, bottom_y),
                xytext=(arrow_x, start_y),
                arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=2),
                transform=ax.transAxes)
    ax.text(arrow_x - 0.005, (start_y + bottom_y) / 2,
            "scale", fontsize=7, rotation=90, va="center", ha="center",
            color="#9ca3af", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved figures/{output_name}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(SAMPLES_JSON) as f:
        data = json.load(f)

    head = data["head"]

    # Load recovery from knockout data
    knockout_path = Path("outputs/knockout_all_heads.json")
    with open(knockout_path) as f:
        knockout = json.load(f)
    recovery = knockout["poe"]["head_recovery"][f"H{head}"]

    for prompt in PROMPTS:
        if prompt not in data["prompts"]:
            print(f"Skipping '{prompt}' — not in data")
            continue

        samples = data["prompts"][prompt]
        safe_name = prompt.lower().replace(" ", "_")[:30]
        output_name = f"poe_steering_{safe_name}.png"
        make_figure(prompt, samples, head, recovery, output_name)


if __name__ == "__main__":
    main()