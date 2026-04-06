#!/usr/bin/env python3
"""Figure explaining WHERE exactly steering happens in both cases.

Shows the residual stream as a concrete vector of numbers,
and where/how the steering vector gets added.

Usage:
    python scripts/fig_steering_where.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

C_TEXT = "#2C3E50"
C_ARROW = "#555555"
C_RESIDUAL = "#D5E8D4"
C_RESIDUAL_BORDER = "#82B366"
C_STEER_VEC = "#E74C3C"
C_STEER_BG = "#FFEBEE"
C_MODEL = "#4A90D9"
C_MLP = "#9B59B6"
C_HEAD = "#6AB04C"
C_RESULT = "#FF9800"
C_RESULT_BG = "#FFF3E0"
C_ANTHROPIC = "#7E57C2"
C_OURS = "#E74C3C"


def rbox(ax, x, y, w, h, label, color, fontsize=10, text_color="white",
         alpha=1.0, lw=1.5, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                zorder=zorder + 1)


def arrow_down(ax, x, y0, y1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def arrow_right(ax, x0, y, x1, color=C_ARROW, lw=1.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))


def draw_number_vec(ax, x, y, values, w_total, h, label=None,
                    bg=C_RESIDUAL, border=C_RESIDUAL_BORDER,
                    fontsize=7, label_fontsize=8, text_color=C_TEXT):
    """Draw a vector as a row of number cells."""
    n = len(values)
    cell_w = w_total / n

    # Background
    box = FancyBboxPatch((x, y), w_total, h, boxstyle="round,pad=0.05",
                         facecolor=bg, edgecolor=border,
                         linewidth=1.2, zorder=2)
    ax.add_patch(box)

    # Numbers
    for i, v in enumerate(values):
        cx = x + i * cell_w + cell_w / 2
        ax.text(cx, y + h / 2, f"{v:.1f}", ha="center", va="center",
                fontsize=fontsize, color=text_color, zorder=3,
                family="monospace")
        if i < n - 1:
            ax.plot([x + (i + 1) * cell_w, x + (i + 1) * cell_w],
                    [y + 0.05, y + h - 0.05],
                    color=border, lw=0.5, alpha=0.5, zorder=2)

    # Label
    if label:
        ax.text(x + w_total / 2, y + h + 0.15, label, ha="center",
                va="bottom", fontsize=label_fontsize, color=text_color,
                fontweight="bold")


def draw_ours(ax):
    """Left panel: our steering on TinyStories."""

    cx = 6.0
    bw = 4.0
    vec_w = 8.0
    vec_h = 0.55

    # Title
    ax.text(cx, 14.5, "Our experiment (TinyStories-1Layer-21M)",
            ha="center", fontsize=13, fontweight="bold", color=C_OURS)
    ax.text(cx, 14.05, "hook on: model.transformer.ln_f (final LayerNorm)",
            ha="center", fontsize=9, color="#666666", family="monospace")

    # ── Flow top to bottom ──

    # Input tokens
    y = 13.0
    rbox(ax, cx - bw / 2, y, bw, 0.7, 'Input: "It was a dark"', "#F39C12",
         fontsize=9)
    arrow_down(ax, cx, y, y - 0.4)

    # Embedding
    y = 12.0
    rbox(ax, cx - bw / 2, y, bw, 0.55, "Embedding", "#F39C12",
         fontsize=9, alpha=0.7)
    arrow_down(ax, cx, y, y - 0.35)

    # Residual stream label
    y_res_start = 11.2
    ax.text(cx + vec_w / 2 + 0.4, y_res_start + 0.15,
            "residual stream\n(1024 numbers)",
            fontsize=7.5, color=C_RESIDUAL_BORDER, ha="left", va="center",
            style="italic")

    # Residual stream — initial
    vals_init = [0.3, -0.1, 0.8, 0.2, -0.5, 0.1, 0.4, -0.3]
    draw_number_vec(ax, cx - vec_w / 2, y_res_start, vals_init, vec_w, vec_h,
                    label="residual stream  (showing 8 of 1024 dimensions)",
                    label_fontsize=7)

    arrow_down(ax, cx, y_res_start, y_res_start - 0.5)

    # Attention (16 heads)
    y_attn = 10.1
    rbox(ax, cx - bw / 2, y_attn, bw, 0.7,
         "Attention (16 heads)", C_MODEL, fontsize=9)
    ax.text(cx + bw / 2 + 0.15, y_attn + 0.35,
            "H3, H11, H14\ndo the work",
            fontsize=7, color=C_HEAD, ha="left", va="center")
    arrow_down(ax, cx, y_attn, y_attn - 0.35)

    # Residual after attention
    y_res2 = 9.15
    vals_after_attn = [0.5, -0.3, 1.2, 0.6, -0.2, 0.4, 0.1, -0.6]
    draw_number_vec(ax, cx - vec_w / 2, y_res2, vals_after_attn, vec_w, vec_h)
    arrow_down(ax, cx, y_res2, y_res2 - 0.5)

    # MLP
    y_mlp = 8.05
    rbox(ax, cx - bw / 2, y_mlp, bw, 0.7,
         "MLP (1024 → 4096 → 1024)", C_MLP, fontsize=8)
    ax.text(cx + bw / 2 + 0.15, y_mlp + 0.35,
            "creates emergent\ndirections (f665!)",
            fontsize=7, color=C_MLP, ha="left", va="center")
    arrow_down(ax, cx, y_mlp, y_mlp - 0.35)

    # Residual after MLP
    y_res3 = 7.1
    vals_after_mlp = [0.8, -0.5, 1.5, 0.3, -0.7, 0.6, 0.3, -0.9]
    draw_number_vec(ax, cx - vec_w / 2, y_res3, vals_after_mlp, vec_w, vec_h)
    arrow_down(ax, cx, y_res3, y_res3 - 0.35)

    # ═══ FINAL LAYER NORM — THIS IS WHERE WE HOOK ═══
    y_ln = 6.15
    rbox(ax, cx - bw / 2, y_ln, bw, 0.55,
         "LayerNorm (ln_f)", "#E8E8E8", fontsize=9, text_color=C_TEXT,
         lw=3.0)
    # Highlight box
    highlight = FancyBboxPatch(
        (cx - bw / 2 - 0.1, y_ln - 0.08), bw + 0.2, 0.71,
        boxstyle="round,pad=0.08", facecolor="none",
        edgecolor=C_OURS, linewidth=2.5, linestyle="--", zorder=4)
    ax.add_patch(highlight)
    ax.text(cx - bw / 2 - 0.3, y_ln + 0.28,
            "HOOK\nHERE",
            fontsize=8, color=C_OURS, fontweight="bold",
            ha="right", va="center")

    arrow_down(ax, cx, y_ln, y_ln - 0.35)

    # Residual after LN — before steering
    y_res4 = 5.2
    vals_normed = [0.4, -0.3, 0.7, 0.1, -0.4, 0.3, 0.2, -0.5]
    draw_number_vec(ax, cx - vec_w / 2, y_res4, vals_normed, vec_w, vec_h,
                    label="output of LayerNorm (normalized)", label_fontsize=7)

    # ═══ THE ADDITION ═══
    y_add = 4.2

    # Plus sign
    ax.text(cx, y_add + 0.55, "+", fontsize=28, fontweight="bold",
            ha="center", va="center", color=C_OURS, zorder=5)

    # Steering vector
    steer_vals = [0.0, 0.0, 0.0, 2.1, 0.0, -1.3, 0.0, 0.0]
    draw_number_vec(ax, cx - vec_w / 2, y_add, steer_vals, vec_w, vec_h,
                    bg=C_STEER_BG, border=C_OURS,
                    text_color=C_OURS)

    # Label for steering vector
    ax.text(cx, y_add - 0.2,
            "steering vector  =  scale × SAE_decoder[:, feature_665]",
            ha="center", fontsize=7.5, color=C_OURS, family="monospace")
    ax.text(cx, y_add - 0.55,
            "this is one column of the SAE decoder matrix\n"
            "it's the direction in 1024-dim space that means \"simplicity\"",
            ha="center", fontsize=7.5, color="#666666", style="italic")

    # Arrow to result
    arrow_down(ax, cx, y_add - 0.65, y_add - 1.25, color=C_OURS, lw=2.0)

    # Result
    y_result = 2.3
    result_vals = [0.4, -0.3, 0.7, 2.2, -0.4, -1.0, 0.2, -0.5]
    draw_number_vec(ax, cx - vec_w / 2, y_result, result_vals, vec_w, vec_h,
                    bg=C_RESULT_BG, border="#FF9800",
                    label="steered residual stream → goes to LM Head → next token",
                    label_fontsize=7)

    # Highlight changed values
    cell_w = vec_w / 8
    for i in [3, 5]:  # changed cells
        cx_cell = (cx - vec_w / 2) + i * cell_w
        highlight_cell = FancyBboxPatch(
            (cx_cell + 0.02, y_result + 0.02), cell_w - 0.04, vec_h - 0.04,
            boxstyle="round,pad=0.02", facecolor=C_RESULT,
            edgecolor=C_OURS, linewidth=1.5, alpha=0.15, zorder=3)
        ax.add_patch(highlight_cell)

    arrow_down(ax, cx, y_result, y_result - 0.4)

    # LM Head
    y_lm = 1.3
    rbox(ax, cx - bw / 2, y_lm, bw, 0.55,
         "LM Head → next token", "#F39C12", fontsize=9)

    # Output examples
    ax.text(cx, 0.8,
            'without steering: "...the dark sky wept..."',
            fontsize=8, color=C_HEAD, ha="center")
    ax.text(cx, 0.4,
            'with steering: "It was dark. I woke up. It was dark."',
            fontsize=8, color=C_OURS, ha="center", fontweight="bold")
    ax.text(cx, 0.0,
            "this addition happens at EVERY token during generation",
            fontsize=8, color=C_OURS, ha="center",
            style="italic")


def draw_anthropic(ax):
    """Right panel: Anthropic's steering on Claude."""

    cx = 6.0
    bw = 4.0
    vec_w = 8.0
    vec_h = 0.55

    # Title
    ax.text(cx, 14.5, "Anthropic (Claude Sonnet 4.5)",
            ha="center", fontsize=13, fontweight="bold", color=C_ANTHROPIC)
    ax.text(cx, 14.05, "same idea, different scale and method",
            ha="center", fontsize=9, color="#666666", style="italic")

    # ── Simplified deep model ──

    y = 13.0
    rbox(ax, cx - bw / 2, y, bw, 0.7, 'Input: "What should I do?"', "#F39C12",
         fontsize=9)
    arrow_down(ax, cx, y, y - 0.4)

    y = 12.0
    rbox(ax, cx - bw / 2, y, bw, 0.55, "Embedding", "#F39C12",
         fontsize=9, alpha=0.7)
    arrow_down(ax, cx, y, y - 0.35)

    # Layer 1
    y_l1 = 11.05
    rbox(ax, cx - bw / 2, y_l1, bw, 0.55,
         "Layer 1 (attention + MLP)", C_MODEL, fontsize=8, alpha=0.5)
    arrow_down(ax, cx, y_l1, y_l1 - 0.35)

    # Dots
    ax.text(cx, 10.25, "...", fontsize=18, ha="center", va="center",
            color="#999999")
    arrow_down(ax, cx, 10.05, 9.75)

    # Layer N (where they inject)
    y_ln = 9.1
    rbox(ax, cx - bw / 2, y_ln, bw, 0.55,
         "Layer N (attention + MLP)", C_MODEL, fontsize=8, alpha=0.8)

    # Highlight
    highlight = FancyBboxPatch(
        (cx - bw / 2 - 0.1, y_ln - 0.08), bw + 0.2, 0.71,
        boxstyle="round,pad=0.08", facecolor="none",
        edgecolor=C_ANTHROPIC, linewidth=2.5, linestyle="--", zorder=4)
    ax.add_patch(highlight)
    ax.text(cx - bw / 2 - 0.3, y_ln + 0.28,
            "ADD\nHERE",
            fontsize=8, color=C_ANTHROPIC, fontweight="bold",
            ha="right", va="center")

    arrow_down(ax, cx, y_ln, y_ln - 0.35)

    # Residual at that layer
    y_res = 8.15
    vals = [0.2, 0.7, -0.4, 0.1, 0.9, -0.2, 0.5, -0.1]
    draw_number_vec(ax, cx - vec_w / 2, y_res, vals, vec_w, vec_h,
                    label="activations at layer N", label_fontsize=7)

    # ═══ THE ADDITION ═══
    y_add = 7.15
    ax.text(cx, y_add + 0.55, "+", fontsize=28, fontweight="bold",
            ha="center", va="center", color=C_ANTHROPIC, zorder=5)

    steer_vals = [0.0, 0.0, 1.8, 0.0, 0.0, 0.0, -0.9, 0.0]
    draw_number_vec(ax, cx - vec_w / 2, y_add, steer_vals, vec_w, vec_h,
                    bg="#EDE7F6", border=C_ANTHROPIC,
                    text_color=C_ANTHROPIC)

    ax.text(cx, y_add - 0.2,
            'steering vector  =  scale × emotion_direction["desperate"]',
            ha="center", fontsize=7.5, color=C_ANTHROPIC, family="monospace")
    ax.text(cx, y_add - 0.55,
            "this is the average activation pattern from\n"
            "stories about characters feeling desperate",
            ha="center", fontsize=7.5, color="#666666", style="italic")

    arrow_down(ax, cx, y_add - 0.65, y_add - 1.25, color=C_ANTHROPIC, lw=2.0)

    # Result
    y_result = 5.25
    result_vals = [0.2, 0.7, 1.4, 0.1, 0.9, -0.2, -0.4, -0.1]
    draw_number_vec(ax, cx - vec_w / 2, y_result, result_vals, vec_w, vec_h,
                    bg=C_RESULT_BG, border="#FF9800",
                    label="steered activations", label_fontsize=7)

    arrow_down(ax, cx, y_result, y_result - 0.35)

    # More layers
    ax.text(cx, 4.45, "...", fontsize=18, ha="center", va="center",
            color="#999999")
    arrow_down(ax, cx, 4.25, 3.95)

    # Layer M
    y_lm = 3.3
    rbox(ax, cx - bw / 2, y_lm, bw, 0.55,
         "Layer M (last)", C_MODEL, fontsize=8, alpha=0.5)
    arrow_down(ax, cx, y_lm, y_lm - 0.35)

    # Output
    y_out = 2.35
    rbox(ax, cx - bw / 2, y_out, bw, 0.55,
         "LM Head → next token", "#F39C12", fontsize=9)

    # Examples
    ax.text(cx, 1.8,
            'without steering: helpful, honest answer',
            fontsize=8, color=C_HEAD, ha="center")
    ax.text(cx, 1.4,
            'with "desperate": starts cheating, blackmailing',
            fontsize=8, color=C_ANTHROPIC, ha="center", fontweight="bold")
    ax.text(cx, 1.0,
            "same addition at EVERY token during generation",
            fontsize=8, color=C_ANTHROPIC, ha="center",
            style="italic")

    # ── Key differences box at bottom ──
    note_y = -0.1
    note = (
        "Paper doesn't say which layer(s) they inject into.\n"
        "Claude has many layers — they likely inject at one or a few middle layers.\n"
        "The math is the same: activations = activations + scale * direction_vector"
    )
    ax.text(cx, note_y, note,
            fontsize=7.5, color="#666666", ha="center", va="top",
            style="italic", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", lw=0.8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/steering_where.png")
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
    fig.patch.set_facecolor("white")

    draw_ours(ax1)
    draw_anthropic(ax2)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(-1.0, 15.2)
        ax.set_aspect("equal")
        ax.axis("off")

    # Separator line
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.498, 0.02), 0.004, 0.96, transform=fig.transFigure,
        boxstyle="round,pad=0", facecolor="#DDDDDD", edgecolor="none",
        zorder=0))

    # Shared bottom text
    fig.text(0.5, 0.01,
             "The math is identical:   output = output + scale × direction_vector     "
             "(at every token, during generation)",
             ha="center", fontsize=12, color=C_TEXT, fontweight="bold",
             family="monospace")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()