#!/usr/bin/env python3
"""Explainer figure: What is a 'direction in activation space'?

Uses a 2D analogy to show:
- Panel 1: Activation space — each point is the model's internal state for one text
- Panel 2: A direction is a line through that space. Texts further along it
            share a property (e.g. simplicity, desperation).
- Panel 3: Steering = push the point along that direction. Same model, shifted state.

Usage:
    python scripts/fig_what_is_direction.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Style ─────────────────────────────────────────────────────────
C_TEXT = "#2C3E50"
C_BG = "white"
C_GRID = "#EEEEEE"
C_POINT = "#4A90D9"
C_POINT_HIGHLIGHT = "#E74C3C"
C_DIRECTION = "#E74C3C"
C_DIRECTION2 = "#9B59B6"
C_STEER = "#27AE60"
C_FAINT = "#CCCCCC"

np.random.seed(42)


def make_cloud(n=60):
    """Generate a cloud of points representing model activations."""
    x = np.random.randn(n) * 1.5 + 0.5
    y = np.random.randn(n) * 1.5 + 0.5
    return x, y


def draw_panel1(ax):
    """Panel 1: What is activation space?"""
    x, y = make_cloud(80)

    ax.scatter(x, y, s=40, c=C_POINT, alpha=0.5, edgecolors="none", zorder=3)

    # Label a few points
    labels = [
        (2.5, 2.0, '"Once upon a time\nthere was a cat"'),
        (-2.0, -1.5, '"The dark sky\nwept with fury"'),
        (-1.5, 2.5, '"She said hello.\nHe said hi."'),
    ]
    for lx, ly, txt in labels:
        ax.scatter([lx], [ly], s=100, c=C_POINT, edgecolors="#333", linewidths=1.5,
                   zorder=5)
        ax.annotate(txt, (lx, ly), textcoords="offset points",
                    xytext=(15, 10), fontsize=13, color="#444444",
                    style="italic", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F9F9F9",
                              ec="#DDDDDD", lw=1.0),
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=1.0))

    # Axes labels
    ax.set_xlabel("dimension 1  (of 1024 in our model... thousands in Claude)", fontsize=14, color="#888888")
    ax.set_ylabel("dimension 2", fontsize=14, color="#888888")

    ax.set_title("Activation space", fontsize=22, fontweight="bold",
                 color=C_TEXT, pad=14)
    ax.text(0.5, -0.10,
            "Each dot = model's internal state\nfor one piece of text",
            transform=ax.transAxes, fontsize=14, color="#666666",
            ha="center", va="top", style="italic")

    _style_ax(ax)


def draw_panel2(ax):
    """Panel 2: A direction = a line. Points along it share a property."""
    x, y = make_cloud(80)

    # Define a direction (roughly 25 degrees)
    angle = np.radians(25)
    dx, dy = np.cos(angle), np.sin(angle)

    # Project points onto direction
    proj = x * dx + y * dy

    # Color by projection: more along direction = more red
    norm_proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
    colors = plt.cm.RdYlBu_r(norm_proj * 0.7 + 0.15)

    ax.scatter(x, y, s=40, c=colors, alpha=0.6, edgecolors="none", zorder=3)

    # Draw the direction arrow (long, through center)
    arr_len = 4.0
    cx, cy = 0.5, 0.5  # center of cloud
    ax.annotate("", xy=(cx + arr_len * dx, cy + arr_len * dy),
                xytext=(cx - arr_len * dx, cy - arr_len * dy),
                arrowprops=dict(arrowstyle="-|>", color=C_DIRECTION,
                                lw=3.5, mutation_scale=22),
                zorder=4)

    # Label ends
    ax.text(cx - arr_len * dx - 0.3, cy - arr_len * dy + 0.3,
            "less", fontsize=16, color=C_DIRECTION, fontweight="bold",
            ha="right", va="bottom")
    ax.text(cx + arr_len * dx + 0.3, cy + arr_len * dy - 0.3,
            "more", fontsize=16, color=C_DIRECTION, fontweight="bold",
            ha="left", va="top")

    # Direction label
    ax.text(cx + arr_len * dx * 0.3, cy + arr_len * dy * 0.3 + 1.2,
            'direction = "simplicity"',
            fontsize=16, color=C_DIRECTION, fontweight="bold",
            rotation=np.degrees(angle),
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_DIRECTION,
                      lw=2.0, alpha=0.95))

    # Add example texts at ends
    ax.text(cx - arr_len * dx + 0.2, cy - arr_len * dy - 0.5,
            '"The melancholic evening\nbrought whispers of ancient..."',
            fontsize=12, color="#666666", style="italic", ha="left")
    ax.text(cx + arr_len * dx - 1.0, cy + arr_len * dy + 0.6,
            '"It was dark.\nI went to sleep."',
            fontsize=12, color="#666666", style="italic", ha="left")

    ax.set_title("A direction = a line through that space",
                 fontsize=22, fontweight="bold", color=C_TEXT, pad=14)
    ax.text(0.5, -0.10,
            "Texts further along the line share\na property (shorter, simpler, more desperate...)",
            transform=ax.transAxes, fontsize=14, color="#666666",
            ha="center", va="top", style="italic")

    _style_ax(ax)


def draw_panel3(ax):
    """Panel 3: Steering = push the point along the direction."""
    x, y = make_cloud(40)
    ax.scatter(x, y, s=30, c=C_POINT, alpha=0.25, edgecolors="none", zorder=2)

    # Direction arrow (faint, background)
    angle = np.radians(25)
    dx, dy = np.cos(angle), np.sin(angle)
    arr_len = 4.0
    cx, cy = 0.5, 0.5
    ax.annotate("", xy=(cx + arr_len * dx, cy + arr_len * dy),
                xytext=(cx - arr_len * dx, cy - arr_len * dy),
                arrowprops=dict(arrowstyle="-|>", color=C_DIRECTION,
                                lw=2.0, alpha=0.3, mutation_scale=18),
                zorder=2)

    # Original point (model state before steering)
    orig_x, orig_y = -0.5, 0.0
    ax.scatter([orig_x], [orig_y], s=180, c=C_POINT, edgecolors="#333",
               linewidths=2.0, zorder=6)
    ax.annotate("model's state\n(normal)", (orig_x, orig_y),
                textcoords="offset points", xytext=(-70, -40),
                fontsize=14, color=C_POINT, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=C_POINT, lw=1.2),
                zorder=7)

    # Steered point (pushed along direction)
    scale = 3.0
    new_x = orig_x + scale * dx
    new_y = orig_y + scale * dy
    ax.scatter([new_x], [new_y], s=180, c=C_DIRECTION, edgecolors="#333",
               linewidths=2.0, zorder=6)
    ax.annotate("model's state\n(steered)", (new_x, new_y),
                textcoords="offset points", xytext=(65, 30),
                fontsize=14, color=C_DIRECTION, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=C_DIRECTION, lw=1.2),
                zorder=7)

    # Big steering arrow between points
    ax.annotate("",
                xy=(new_x - 0.15 * dx, new_y - 0.15 * dy),
                xytext=(orig_x + 0.15 * dx, orig_y + 0.15 * dy),
                arrowprops=dict(arrowstyle="-|>", color=C_STEER,
                                lw=4.5, mutation_scale=28),
                zorder=5)
    # Label the arrow
    mid_x = (orig_x + new_x) / 2
    mid_y = (orig_y + new_y) / 2
    ax.text(mid_x - 0.7, mid_y + 0.8,
            "+ scale x direction",
            fontsize=15, color=C_STEER, fontweight="bold",
            ha="center", rotation=np.degrees(angle),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_STEER,
                      lw=1.5, alpha=0.9))

    # Output examples
    ax.text(-3.2, -2.6,
            'output: "the dark sky wept,\nthe clouds seemed to go away"',
            fontsize=13, color=C_POINT, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec=C_POINT,
                      lw=1.0, alpha=0.8))

    ax.text(1.2, -2.6,
            'output: "It was dark.\nI went to sleep. It was dark."',
            fontsize=13, color=C_DIRECTION, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec=C_DIRECTION,
                      lw=1.0, alpha=0.8))

    ax.set_title("Steering = push along the direction",
                 fontsize=22, fontweight="bold", color=C_TEXT, pad=14)
    ax.text(0.5, -0.10,
            "Same model, same weights, same prompt.\n"
            "Just add direction x scale to activations at every token.",
            transform=ax.transAxes, fontsize=14, color="#666666",
            ha="center", va="top", style="italic")

    _style_ax(ax)


def _style_ax(ax):
    ax.set_xlim(-4.5, 5.0)
    ax.set_ylim(-3.5, 4.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="#CCCCCC")
    ax.tick_params(colors="#CCCCCC", labelsize=0)
    for spine in ax.spines.values():
        spine.set_color("#DDDDDD")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/what_is_direction.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor(C_BG)

    fig.suptitle("What is a \"direction in activation space\"?",
                 fontsize=28, fontweight="bold", color=C_TEXT, y=0.98)
    fig.text(0.5, 0.93,
             "In reality the space has 1024 dimensions (our model) or thousands (Claude). Here we squash it to 2D.",
             fontsize=16, color="#888888", ha="center", style="italic")

    draw_panel1(axes[0])
    draw_panel2(axes[1])
    draw_panel3(axes[2])

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()