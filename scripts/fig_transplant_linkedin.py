#!/usr/bin/env python3
"""LinkedIn-optimized transplant figures: one per host, cursive text.

Landscape format for LinkedIn image preview. Shows the mechanism
(16 heads, H14 swapped) and the text result.

Usage:
    python scripts/fig_transplant_linkedin.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

FIGURES_DIR = Path("figures")
TRANSPLANT_JSON = Path("outputs/transplant_samples.json")

C_HEAD = "#6AB04C"
C_POE = "#C44E52"
C_TEXT = "#333333"

FONT_PROSE = "Noto Serif Display"


def wrap_text(text, width=36):
    import textwrap
    return textwrap.fill(text, width=width)


def draw_head_strip(ax, x, y, w, h, highlight_head=None, highlight_color=C_POE):
    """Draw a strip of 16 small head boxes."""
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


def make_figure(host_key, host_label, donor, prompt, info, seed=42):
    pure = info["pure_text"]
    transplanted = info["transplant_text"]
    pure_ppl = info["pure_ppl"]
    transplant_ppl = info["transplant_ppl"]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.3, 9)
    ax.axis("off")

    # ── Title ──
    ax.text(8.0, 8.7,
            "What happens when you transplant one attention head?",
            fontsize=20, fontweight="bold", ha="center", va="top",
            color=C_TEXT)

    # ── Head strip: host adapter ──
    strip_y = 7.5
    strip_w = 5.8
    ax.text(1.2, strip_y + 0.65, f"{host_label}'s adapter — 16 heads:",
            fontsize=12, fontweight="bold", color=C_TEXT)
    draw_head_strip(ax, 1.2, strip_y, strip_w, 0.42)

    # Arrow showing H14 replacement
    h14_x = 1.2 + 14/16 * strip_w + strip_w/32
    ax.annotate("", xy=(h14_x, strip_y),
                xytext=(h14_x, strip_y - 0.55),
                arrowprops=dict(arrowstyle="-|>", color=C_POE, lw=2.5))
    ax.text(h14_x + 0.35, strip_y - 0.4,
            f"← replace with\n    {donor}'s H14",
            fontsize=10, color=C_POE, fontweight="bold", va="center")

    # ── Head strip: after transplant ──
    ax.text(9.0, strip_y + 0.65, f"{host_label}'s adapter + {donor}'s H14:",
            fontsize=12, fontweight="bold", color=C_POE)
    draw_head_strip(ax, 9.0, strip_y, strip_w, 0.42, highlight_head=14,
                    highlight_color=C_POE)

    # ── Text boxes ──
    box_h = 5.2
    box_bot = 0.9

    # Pure host
    ax.add_patch(FancyBboxPatch(
        (0.3, box_bot), 6.7, box_h, boxstyle="round,pad=0.18",
        facecolor="#f5f5f5", edgecolor="#cccccc", linewidth=1.5))
    ax.text(0.7, box_bot + box_h - 0.25, f"Pure {host_label}:",
            fontsize=16, fontweight="bold", va="top", color="#555555")
    ax.text(0.7, box_bot + box_h - 0.8, wrap_text(pure, 30),
            fontsize=15.5, va="top", fontfamily=FONT_PROSE, style="italic",
            color=C_TEXT, linespacing=1.5)
    ax.text(0.7, box_bot + 0.2, f"PPL: {pure_ppl:.0f}",
            fontsize=12, color="#999999")

    # Arrow between boxes
    mid_y = box_bot + box_h / 2
    ax.annotate("", xy=(8.5, mid_y), xytext=(7.5, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=C_POE, lw=3))

    # Transplanted host
    ax.add_patch(FancyBboxPatch(
        (8.8, box_bot), 6.9, box_h, boxstyle="round,pad=0.18",
        facecolor="#fff5f5", edgecolor=C_POE, linewidth=2))
    ax.text(9.2, box_bot + box_h - 0.25, f"{host_label} + {donor}'s H14:",
            fontsize=16, fontweight="bold", va="top", color=C_POE)
    ax.text(9.2, box_bot + box_h - 0.8, wrap_text(transplanted, 30),
            fontsize=15.5, va="top", fontfamily=FONT_PROSE, style="italic",
            color=C_TEXT, linespacing=1.5)
    ppl_delta = (transplant_ppl - pure_ppl) / pure_ppl * 100
    ax.text(9.2, box_bot + 0.2,
            f"PPL: {transplant_ppl:.0f}  (+{ppl_delta:.0f}%)",
            fontsize=12, color=C_POE)

    # ── Footer ──
    ax.text(8.0, 0.35,
            f'Prompt: "{prompt}" · seed={seed} · TinyStories-1Layer-21M · LoRA rank 8',
            fontsize=10, ha="center", va="bottom", color="#aaaaaa")
    ax.text(8.0, 0.0,
            "Same model, same seed — only 64 of 1024 weight rows replaced",
            fontsize=10, ha="center", va="bottom", color="#aaaaaa", style="italic")

    plt.tight_layout()
    out = FIGURES_DIR / f"transplant_linkedin_{host_key}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRANSPLANT_JSON) as f:
        data = json.load(f)

    prompt = data["prompt"]
    donor = data["donor"].capitalize()

    labels = {
        "carroll": "Carroll",
        "grimm": "Grimm",
        "minimalist": "Minimalist*",
    }

    for host_key, info in data["pairs"].items():
        host_label = labels.get(host_key, host_key.capitalize())
        make_figure(host_key, host_label, donor, prompt, info,
                    seed=data.get("seed", 42))


if __name__ == "__main__":
    main()