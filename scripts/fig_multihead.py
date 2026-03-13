#!/usr/bin/env python3
"""Draw multi-head attention mechanism: how input splits into heads during inference.

Usage:
    python scripts/fig_multihead.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_ATTN = "#4A90D9"
C_TEXT = "#2C3E50"
C_ARROW = "#555555"
C_EMBED = "#F39C12"
C_HEAD_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabebe", "#469990", "#e6beff",
    "#9A6324", "#800000", "#aaffc3", "#808000",
]


def box(ax, x, y, w, h, label, color, fontsize=9, alpha=1.0,
        text_color="white", sublabel=None, sublabel_size=7):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                        facecolor=color, edgecolor="#333333",
                        linewidth=1.0, alpha=alpha, zorder=2)
    ax.add_patch(b)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.12, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.1, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=text_color, zorder=3, alpha=0.85)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="figures/multihead.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor("white")

    cx = 6.5  # center x

    # Flow top-to-bottom: high y = top of figure
    # ============ Title ============
    y_title = 10.0
    ax.text(cx, y_title, "How Multi-Head Attention Works",
            fontsize=15, ha="center", color=C_TEXT, fontweight="bold")

    # ============ STEP 1: Input ============
    y_input = 9.0
    input_w = 5.0
    box(ax, cx - input_w / 2, y_input, input_w, 0.6,
        "Input embedding:  x", C_EMBED, fontsize=10,
        sublabel="1024-dim vector")

    arrow(ax, cx, y_input, cx, y_input - 0.4)

    # ============ STEP 2: V projection (full) ============
    y_proj = 7.8
    proj_w = 5.0
    box(ax, cx - proj_w / 2, y_proj, proj_w, 0.7,
        "V projection:  V = W_v · x", C_ATTN, fontsize=10,
        sublabel="1024×1024 matrix → 1024-dim output")

    # "The key insight" annotation
    ax.text(cx, y_proj - 0.2, "this 1024-dim output is NOT used as one vector",
            fontsize=8, ha="center", color="#CC0000", fontweight="bold",
            style="italic")

    arrow(ax, cx, y_proj - 0.25, cx, y_proj - 0.5)

    # ============ STEP 3: Split into 16 heads (color bar) ============
    y_split = 6.8
    bar_w = 10.0
    bar_h = 0.5
    bar_x = cx - bar_w / 2

    # Background bar
    bg = FancyBboxPatch((bar_x, y_split), bar_w, bar_h,
                         boxstyle="round,pad=0.02",
                         facecolor="#eee", edgecolor="#999",
                         linewidth=0.8, zorder=1)
    ax.add_patch(bg)

    # Individual head slices in the bar
    head_w = bar_w / 16
    for i in range(16):
        hx = bar_x + i * head_w
        b = FancyBboxPatch((hx, y_split), head_w, bar_h,
                            boxstyle="square,pad=0.0",
                            facecolor=C_HEAD_COLORS[i],
                            edgecolor="#555", linewidth=0.4,
                            alpha=0.7, zorder=2)
        ax.add_patch(b)
        if i < 3 or i == 15:
            ax.text(hx + head_w / 2, y_split + bar_h / 2, f"H{i}",
                    ha="center", va="center", fontsize=5,
                    fontweight="bold", color="white", zorder=3)

    ax.text(bar_x + bar_w / 2, y_split + bar_h + 0.12,
            "V output split into 16 chunks of 64 dims each",
            fontsize=8, ha="center", color=C_TEXT, fontweight="bold")

    # Dimension labels below bar
    ax.text(bar_x + head_w / 2, y_split - 0.15, "dims\n0–63",
            fontsize=5, ha="center", color="#888")
    ax.text(bar_x + 1.5 * head_w, y_split - 0.15, "dims\n64–127",
            fontsize=5, ha="center", color="#888")
    ax.text(bar_x + 15.5 * head_w, y_split - 0.15, "dims\n960–1023",
            fontsize=5, ha="center", color="#888")

    # ============ STEP 4: Each head does attention independently ============
    ax.text(cx + 0.2, y_split - 0.45, "each head works independently",
            fontsize=7, ha="left", color="#888", style="italic")

    # Show 3 heads + dots
    head_box_w = 2.2
    head_box_h = 1.8
    head_spacing = 0.3
    total_w = 4 * head_box_w + 3 * head_spacing
    hx0 = cx - total_w / 2

    y_heads_top = 5.6
    heads_to_show = [0, 1, None, 15]  # None = dots

    for j, hi in enumerate(heads_to_show):
        hx = hx0 + j * (head_box_w + head_spacing)
        hy = y_heads_top - head_box_h

        if hi is None:
            ax.text(hx + head_box_w / 2, hy + head_box_h / 2, "· · ·",
                    fontsize=20, ha="center", va="center", color="#888")
            continue

        # Head background
        bg = FancyBboxPatch((hx, hy), head_box_w, head_box_h,
                             boxstyle="round,pad=0.06",
                             facecolor=C_HEAD_COLORS[hi],
                             edgecolor="#444", linewidth=1.2,
                             alpha=0.15, zorder=0)
        ax.add_patch(bg)

        # Head label
        ax.text(hx + head_box_w / 2, hy + head_box_h - 0.15,
                f"Head {hi}", fontsize=9, ha="center",
                color=C_HEAD_COLORS[hi], fontweight="bold")

        # Inside: Q_h, K_h → attention → V_h
        inner_y = hy + 0.15
        small_w = 0.8
        small_h = 0.35

        # Q_h and K_h
        box(ax, hx + 0.1, inner_y + 0.85, small_w, small_h,
            "Q_h", C_ATTN, fontsize=7, alpha=0.7)
        box(ax, hx + head_box_w - small_w - 0.1, inner_y + 0.85, small_w, small_h,
            "K_h", C_ATTN, fontsize=7, alpha=0.7)

        # Attention scores
        ax.text(hx + head_box_w / 2, inner_y + 0.65, "attention\nscores",
                fontsize=5.5, ha="center", color="#666", style="italic")

        # V_h
        box(ax, hx + (head_box_w - small_w) / 2, inner_y + 0.15, small_w, small_h,
            "V_h", C_ATTN, fontsize=7, alpha=0.7)

        # Output
        ax.text(hx + head_box_w / 2, inner_y - 0.1, "→ 64-dim out",
                fontsize=6, ha="center", color=C_TEXT, fontweight="bold")

        # Dashed line from bar down to this head
        bar_head_x = bar_x + hi * head_w + head_w / 2
        ax.plot([bar_head_x, hx + head_box_w / 2],
                [y_split, hy + head_box_h],
                color=C_HEAD_COLORS[hi], lw=0.8, ls="--", zorder=1)

    # ============ STEP 5: Concatenate ============
    hy_bottom = y_heads_top - head_box_h  # bottom of head boxes
    y_concat = hy_bottom - 0.8

    # Dashed lines from heads to concat
    for j, hi in enumerate(heads_to_show):
        if hi is None:
            continue
        hx = hx0 + j * (head_box_w + head_spacing)
        ax.plot([hx + head_box_w / 2, cx],
                [hy_bottom, y_concat + 0.5],
                color="#888", lw=0.7, ls="--", zorder=1)

    box(ax, cx - bar_w / 2, y_concat, bar_w, 0.5,
        "Concatenate all 16 head outputs → 1024-dim", C_ATTN,
        fontsize=9, alpha=0.8)

    arrow(ax, cx, y_concat, cx, y_concat - 0.4)

    # Out projection
    y_out = y_concat - 1.0
    box(ax, cx - 3, y_out, 6, 0.55,
        "Output projection:  O · concat(heads) → 1024-dim", C_ATTN,
        fontsize=9, alpha=0.9)

    # ============ Key insight box ============
    insight_y = y_out - 1.2
    ax.text(cx, insight_y,
            "Each head sees the full input but produces only 64 dims of output.\n"
            "LoRA changes what each head outputs → zeroing a head's ΔW rows\n"
            "removes that head's style contribution while keeping the others.",
            fontsize=9, ha="center", va="center", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1",
                      edgecolor="#F39C12", linewidth=1.2))

    # Axis
    ax.set_xlim(0, 13)
    ax.set_ylim(insight_y - 0.8, y_title + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()