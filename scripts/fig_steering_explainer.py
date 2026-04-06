#!/usr/bin/env python3
"""General steering explainer diagram.

Shows the core mechanic shared by Anthropic's emotion steering and
our SAE feature steering: find a direction in activation space,
add it during inference, behavior changes.

Usage:
    python scripts/fig_steering_explainer.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colors ──────────────────────────────────────────────────────────
C_TEXT = "#2C3E50"
C_MODEL = "#4A90D9"
C_STREAM = "#E8E8E8"
C_DIRECTION = "#E74C3C"
C_OUTPUT_NORMAL = "#27AE60"
C_OUTPUT_STEERED = "#E74C3C"
C_BG = "white"
C_ARROW = "#555555"
C_VECTOR = "#9B59B6"


def rounded_box(ax, x, y, w, h, label, color, fontsize=10,
                text_color="white", alpha=1.0, lw=1.5, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            zorder=zorder + 1)


def arrow_h(ax, x0, y, x1, **kwargs):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.5, **kwargs))


def draw(ax):
    # ── Layout ──
    # Two rows: top = normal, bottom = steered
    # Left to right: Input → [Model internals with activations] → Output

    row_gap = 4.5
    y_top = 5.5       # normal pass
    y_bot = 0.5       # steered pass

    # ── NORMAL PASS (top) ──
    ax.text(0.5, y_top + 3.2, "Normal inference", fontsize=14,
            fontweight="bold", color=C_TEXT, ha="left")

    # Input
    rounded_box(ax, 0.5, y_top + 0.6, 2.5, 1.5, "Input\ntext", "#F39C12",
                fontsize=11)

    # Arrow
    arrow_h(ax, 3.0, y_top + 1.35, 4.0)

    # Model block
    rounded_box(ax, 4.0, y_top + 0.0, 5.0, 2.7, "", C_MODEL, alpha=0.15,
                lw=2.0)
    ax.text(6.5, y_top + 2.35, "Model", fontsize=12, fontweight="bold",
            color=C_MODEL, ha="center")

    # Residual stream inside model
    stream_x = 4.5
    stream_w = 4.0
    stream_h = 0.7
    stream_y = y_top + 0.8
    box = FancyBboxPatch((stream_x, stream_y), stream_w, stream_h,
                         boxstyle="round,pad=0.08",
                         facecolor="#D5E8D4", edgecolor="#82B366",
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(stream_x + stream_w / 2, stream_y + stream_h / 2,
            "activations  (vector of numbers)",
            fontsize=9, ha="center", va="center", color=C_TEXT,
            style="italic", zorder=4)

    # Arrow out
    arrow_h(ax, 9.0, y_top + 1.35, 10.0)

    # Output
    rounded_box(ax, 10.0, y_top + 0.6, 3.5, 1.5, "", C_OUTPUT_NORMAL,
                fontsize=10)
    ax.text(11.75, y_top + 1.55, "Normal\noutput", fontsize=11,
            fontweight="bold", color="white", ha="center", va="center")

    # ── STEERED PASS (bottom) ──
    ax.text(0.5, y_bot + 3.6, "Steered inference", fontsize=14,
            fontweight="bold", color=C_TEXT, ha="left")
    ax.text(0.5, y_bot + 3.2, "same model, same weights — direction added during generation",
            fontsize=10, color="#666666", ha="left", style="italic")

    # Input
    rounded_box(ax, 0.5, y_bot + 0.6, 2.5, 1.5, "Same\ninput", "#F39C12",
                fontsize=11)

    # Arrow
    arrow_h(ax, 3.0, y_bot + 1.35, 4.0)

    # Model block
    rounded_box(ax, 4.0, y_bot + 0.0, 5.0, 2.7, "", C_MODEL, alpha=0.15,
                lw=2.0)
    ax.text(6.5, y_bot + 2.35, "Model", fontsize=12, fontweight="bold",
            color=C_MODEL, ha="center")

    # Residual stream inside model
    stream_y2 = y_bot + 0.8
    box2 = FancyBboxPatch((stream_x, stream_y2), stream_w, stream_h,
                          boxstyle="round,pad=0.08",
                          facecolor="#D5E8D4", edgecolor="#82B366",
                          linewidth=1.5, zorder=3)
    ax.add_patch(box2)
    ax.text(stream_x + stream_w / 2, stream_y2 + stream_h / 2,
            "activations",
            fontsize=9, ha="center", va="center", color=C_TEXT,
            style="italic", zorder=4)

    # ── THE DIRECTION BEING ADDED ──
    dir_x = stream_x + stream_w / 2
    dir_y_top = y_bot + 3.1
    dir_y_bot = stream_y2 + stream_h

    # Direction box (above, pointing down)
    dir_box_w = 3.6
    dir_box_h = 1.0
    dir_box_x = dir_x - dir_box_w / 2
    dir_box = FancyBboxPatch((dir_box_x, dir_y_top), dir_box_w, dir_box_h,
                             boxstyle="round,pad=0.1",
                             facecolor=C_DIRECTION, edgecolor="#C0392B",
                             linewidth=2.0, alpha=0.9, zorder=5)
    ax.add_patch(dir_box)
    ax.text(dir_x, dir_y_top + dir_box_h / 2,
            "direction vector\n(extracted beforehand by SAE / probing)",
            fontsize=10, ha="center", va="center", color="white",
            fontweight="bold", zorder=6)

    # "+" symbol at injection point
    plus_y = (dir_y_top + stream_y2 + stream_h) / 2
    ax.text(dir_x, plus_y, "+", fontsize=22, fontweight="bold",
            ha="center", va="center", color=C_DIRECTION, zorder=6)

    # Arrow from direction box down to stream
    ax.annotate("", xy=(dir_x, stream_y2 + stream_h + 0.05),
                xytext=(dir_x, dir_y_top - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_DIRECTION,
                                lw=2.5, connectionstyle="arc3,rad=0"),
                zorder=5)

    # Scale label
    ax.text(dir_x + 1.8, plus_y + 0.15, "× scale",
            fontsize=9, color=C_DIRECTION, style="italic",
            fontweight="bold", zorder=6)

    # Arrow out
    arrow_h(ax, 9.0, y_bot + 1.35, 10.0)

    # Output (steered)
    rounded_box(ax, 10.0, y_bot + 0.6, 3.5, 1.5, "", C_OUTPUT_STEERED,
                fontsize=10)
    ax.text(11.75, y_bot + 1.55, "Steered\noutput", fontsize=11,
            fontweight="bold", color="white", ha="center", va="center")

    # ── EXAMPLES on the right ──
    ex_x = 14.0
    ex_w = 7.0

    # Normal output example
    ex_y_top = y_top + 0.3
    box_n = FancyBboxPatch((ex_x, ex_y_top), ex_w, 2.2,
                           boxstyle="round,pad=0.1",
                           facecolor="#E8F5E9", edgecolor="#81C784",
                           linewidth=1.2, zorder=2)
    ax.add_patch(box_n)

    ax.text(ex_x + 0.3, ex_y_top + 1.85,
            "Anthropic: Claude responds helpfully, calmly",
            fontsize=8, color="#2E7D32", va="top", zorder=3)
    ax.text(ex_x + 0.3, ex_y_top + 1.35,
            'Ours: "the dark sky wept, the clouds\n'
            'seemed to go away and, in the night..."',
            fontsize=8, color="#2E7D32", va="top", style="italic", zorder=3)

    # Arrow from output to example
    ax.plot([13.5, ex_x], [y_top + 1.35, ex_y_top + 1.1],
            color="#81C784", lw=0.8, ls="--", zorder=1)

    # Steered output example
    ex_y_bot = y_bot + 0.0
    box_s = FancyBboxPatch((ex_x, ex_y_bot), ex_w, 2.7,
                           boxstyle="round,pad=0.1",
                           facecolor="#FFEBEE", edgecolor="#E57373",
                           linewidth=1.2, zorder=2)
    ax.add_patch(box_s)

    ax.text(ex_x + 0.3, ex_y_bot + 2.35,
            "Anthropic + desperate: Claude starts cheating,\n"
            "blackmailing — no visible emotional cues",
            fontsize=8, color="#C62828", va="top", zorder=3)
    ax.text(ex_x + 0.3, ex_y_bot + 1.45,
            'Ours + simplicity: "It was dark.\n'
            'I woke up. It was dark. It was night."',
            fontsize=8, color="#C62828", va="top", style="italic", zorder=3)

    # Direction label examples
    ax.text(ex_x + 0.3, ex_y_bot + 0.55,
            "Same mechanic: find a direction, add it, behavior changes",
            fontsize=8.5, color=C_TEXT, fontweight="bold", va="top", zorder=3)

    # Arrow from output to example
    ax.plot([13.5, ex_x], [y_bot + 1.35, ex_y_bot + 1.3],
            color="#E57373", lw=0.8, ls="--", zorder=1)

    # ── What is a "direction"? — small note at bottom ──
    note_y = -0.8
    ax.text(7.0, note_y,
            'A "direction" = a vector in the model\'s internal activation space.\n'
            'Anthropic: extracted from stories about emotions  →  "desperate," "calm," "curious"\n'
            'Ours: found by sparse autoencoder (SAE)  →  "simplicity," "dialogue," "questions"',
            fontsize=9, color="#666666", ha="center", va="top",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", lw=1.0))

    # ── Axis ──
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-1.8, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7.0, 9.0,
            "Activation Steering: The Core Idea",
            fontsize=18, fontweight="bold", color=C_TEXT, ha="center")
    ax.text(7.0, 8.5,
            "Find a direction in the model's internal space. Add it during generation. Behavior changes.",
            fontsize=10, color="#666666", ha="center", style="italic")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/steering_explainer.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(21, 11))
    fig.patch.set_facecolor(C_BG)
    draw(ax)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()