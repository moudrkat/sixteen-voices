#!/usr/bin/env python3
"""Generate LinkedIn-style transplant figures for new experiments."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path("figures")
DATA_PATH = Path("outputs/transplant_new.json")

C_HEAD = "#6AB04C"
C_DONOR = "#C44E52"
C_TEXT = "#333333"
FONT_PROSE = "Noto Serif Display"

DONOR_COLORS = {
    "poe": "#C44E52",
    "carroll": "#2980B9",
}


def wrap_text(text, width=36):
    return textwrap.fill(text, width=width)


def draw_head_strip(ax, x, y, w, h, highlight_head=None, highlight_color=C_DONOR):
    hw = w / 16
    gap = hw * 0.08
    actual_hw = hw - gap
    for i in range(16):
        hx = x + i * hw
        color = highlight_color if i == highlight_head else C_HEAD
        alpha = 1.0 if i == highlight_head else 0.5
        lw = 2.0 if i == highlight_head else 0.5
        ec = highlight_color if i == highlight_head else "#888888"
        ax.add_patch(FancyBboxPatch(
            (hx, y), actual_hw, h, boxstyle="round,pad=0.01",
            facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha))
        if i == highlight_head:
            ax.text(hx + actual_hw / 2, y + h / 2, f"H{i}",
                    ha="center", va="center", fontsize=6, fontweight="bold",
                    color="white")


def make_figure(key, info):
    host = info["host"]
    donor = info["donor"]
    head = info["head"]
    pure = info["pure_text"]
    transplanted = info["transplant_text"]
    pure_ppl = info["pure_ppl"]
    t_ppl = info["transplant_ppl"]

    host_label = host.capitalize()
    donor_label = donor.capitalize()
    donor_color = DONOR_COLORS.get(donor, C_DONOR)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.3, 9)
    ax.axis("off")

    # Title
    ax.text(8.0, 8.7,
            f"What happens when you transplant one attention head?",
            fontsize=20, fontweight="bold", ha="center", va="top", color=C_TEXT)

    # Head strips
    strip_y = 7.5
    strip_w = 5.8

    ax.text(1.2, strip_y + 0.65, f"{host_label}'s adapter — 16 heads:",
            fontsize=12, fontweight="bold", color=C_TEXT)
    draw_head_strip(ax, 1.2, strip_y, strip_w, 0.42)

    h_x = 1.2 + head / 16 * strip_w + strip_w / 32
    ax.annotate("", xy=(h_x, strip_y),
                xytext=(h_x, strip_y - 0.55),
                arrowprops=dict(arrowstyle="-|>", color=donor_color, lw=2.5))
    ax.text(h_x + 0.35, strip_y - 0.4,
            f"← replace with\n    {donor_label}'s H{head}",
            fontsize=10, color=donor_color, fontweight="bold", va="center")

    ax.text(9.0, strip_y + 0.65,
            f"{host_label}'s adapter + {donor_label}'s H{head}:",
            fontsize=12, fontweight="bold", color=donor_color)
    draw_head_strip(ax, 9.0, strip_y, strip_w, 0.42,
                    highlight_head=head, highlight_color=donor_color)

    # Text boxes
    box_h = 5.2
    box_bot = 0.9

    # Pure
    ax.add_patch(FancyBboxPatch(
        (0.3, box_bot), 6.7, box_h, boxstyle="round,pad=0.18",
        facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=1.5))
    ax.text(0.7, box_bot + box_h - 0.25, f"Pure {host_label}:",
            fontsize=16, fontweight="bold", va="top", color="#555555")
    ax.text(0.7, box_bot + box_h - 0.8, wrap_text(pure[:250], 30),
            fontsize=15.5, va="top", fontfamily=FONT_PROSE, style="italic",
            color=C_TEXT, linespacing=1.5)
    ax.text(0.7, box_bot + 0.2, f"PPL: {pure_ppl:.0f}",
            fontsize=12, color="#999999")

    # Arrow
    mid_y = box_bot + box_h / 2
    ax.annotate("", xy=(8.5, mid_y), xytext=(7.5, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=donor_color, lw=3))

    # Transplanted
    bg = donor_color.lstrip("#")
    r, g, b = int(bg[:2], 16), int(bg[2:4], 16), int(bg[4:], 16)
    light_bg = f"#{min(255, r+200):02x}{min(255, g+200):02x}{min(255, b+200):02x}"

    ax.add_patch(FancyBboxPatch(
        (8.8, box_bot), 6.9, box_h, boxstyle="round,pad=0.18",
        facecolor=light_bg, edgecolor=donor_color, linewidth=2))
    ax.text(9.2, box_bot + box_h - 0.25,
            f"{host_label} + {donor_label}'s H{head}:",
            fontsize=16, fontweight="bold", va="top", color=donor_color)
    ax.text(9.2, box_bot + box_h - 0.8, wrap_text(transplanted[:250], 30),
            fontsize=15.5, va="top", fontfamily=FONT_PROSE, style="italic",
            color=C_TEXT, linespacing=1.5)
    ppl_delta = (t_ppl - pure_ppl) / pure_ppl * 100
    sign = "+" if ppl_delta >= 0 else ""
    ax.text(9.2, box_bot + 0.2,
            f"PPL: {t_ppl:.0f}  ({sign}{ppl_delta:.0f}%)",
            fontsize=12, color=donor_color)

    # Footer
    ax.text(8.0, 0.35,
            f'Prompt: "It was a dark and stormy" · seed=42 · '
            f'TinyStories-1Layer-21M · LoRA rank 8',
            fontsize=10, ha="center", va="bottom", color="#aaaaaa")
    ax.text(8.0, 0.0,
            f"Same model, same seed — only 64 of 1024 weight rows replaced in one head",
            fontsize=10, ha="center", va="bottom", color="#aaaaaa", style="italic")

    plt.tight_layout()
    out = FIG_DIR / f"transplant_linkedin_{key}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    plt.close()


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)
    for key, info in data.items():
        make_figure(key, info)


if __name__ == "__main__":
    main()