#!/usr/bin/env python3
"""Explainer figure: What is the residual stream and where steering happens.

Shows the TinyStories-1Layer-21M architecture as a vertical flow:
- Token comes in → embedding (1024 numbers)
- Residual stream flows down
- Attention reads & writes back
- MLP reads & writes back
- ★ STEERING HAPPENS HERE: add direction vector (1024 numbers)
- Predict next token

Plus a zoomed-in view of what "add 1024 numbers" actually looks like.

Usage:
    python scripts/fig_residual_stream.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colors ──────────────────────────────────────────────────────────
C_TEXT = "#2C3E50"
C_BG = "white"
C_STREAM = "#3498DB"
C_STREAM_BG = "#EBF5FB"
C_ATTN = "#9B59B6"
C_MLP = "#F39C12"
C_STEER = "#E74C3C"
C_OUTPUT = "#27AE60"
C_EMBED = "#1ABC9C"
C_ARROW = "#555555"
C_FAINT = "#AAAAAA"


def rbox(ax, x, y, w, h, label, color, fontsize=18, text_color="white",
         alpha=1.0, lw=2.0, sublabel=None, sublabel_fs=14):
    """Rounded box with optional sublabel."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.3, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.35, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)


def arrow_v(ax, x, y0, y1, color=C_ARROW, lw=2.5):
    """Vertical arrow (top to bottom: y0 > y1)."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=20))


def draw_left(ax):
    """Left panel: the residual stream as a vertical flow."""
    cx = 6.0   # center x for the main column
    bw = 6.0   # box width
    bh = 1.4   # box height
    gap = 0.8  # gap between boxes

    # ── Positions (top to bottom) ──
    y_token = 20.0
    y_embed = y_token - bh - gap
    y_stream1 = y_embed - bh - gap - 0.3
    y_attn = y_stream1 - bh - gap
    y_stream2 = y_attn - bh - gap
    y_mlp = y_stream2 - bh - gap
    y_stream3 = y_mlp - bh - gap
    y_steer = y_stream3 - bh - gap - 0.3
    y_stream4 = y_steer - bh - 0.4 - gap
    y_predict = y_stream4 - bh - gap

    # ── Token input ──
    rbox(ax, cx - bw / 2, y_token, bw, bh, 'Input token: "dark"', "#7F8C8D",
         fontsize=17)

    # ── Embedding ──
    arrow_v(ax, cx, y_token, y_embed + bh)
    rbox(ax, cx - bw / 2, y_embed, bw, bh, "Embedding", C_EMBED,
         fontsize=18, sublabel="token → 1024 numbers")

    # ── Residual stream background stripe ──
    stream_top = y_embed
    stream_bot = y_stream4 + bh
    stripe = FancyBboxPatch((cx - 1.0, stream_bot - 0.3), 2.0,
                            stream_top - stream_bot + bh + 0.6,
                            boxstyle="round,pad=0.1",
                            facecolor=C_STREAM_BG, edgecolor=C_STREAM,
                            linewidth=2.0, alpha=0.3, zorder=1,
                            linestyle="--")
    ax.add_patch(stripe)

    # Residual stream label — far right, no overlap
    ax.text(cx + bw / 2 + 2.0, (stream_top + stream_bot + bh) / 2 + 1.5,
            "RESIDUAL STREAM",
            fontsize=18, color=C_STREAM, ha="left", va="center",
            fontweight="bold")
    ax.text(cx + bw / 2 + 2.0, (stream_top + stream_bot + bh) / 2,
            "1024 numbers\nflowing down\n\neveryone reads from it\nand writes back to it",
            fontsize=14, color=C_STREAM, ha="left", va="center",
            linespacing=1.5)

    # ── Stream → Attention ──
    arrow_v(ax, cx, y_embed, y_stream1 + bh)

    # Stream state 1
    ax.text(cx, y_stream1 + bh / 2,
            "[0.23, -1.07, 0.54, ... ]  1024 numbers",
            fontsize=13, color=C_STREAM, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_STREAM,
                      lw=1.5, alpha=0.9))

    # ── Attention ──
    arrow_v(ax, cx, y_stream1, y_attn + bh)
    attn_w = bw + 2.0
    rbox(ax, cx - attn_w / 2, y_attn, attn_w, bh,
         "16 Attention Heads", C_ATTN,
         fontsize=18, sublabel="read stream → compute → add result back")

    # Side annotation for attention
    ax.text(cx - attn_w / 2 - 0.5, y_attn + bh / 2,
            "each head:\n64 dims",
            fontsize=13, color=C_ATTN, ha="right", va="center",
            style="italic")

    # ── Stream after attention ──
    arrow_v(ax, cx, y_attn, y_stream2 + bh)
    ax.text(cx, y_stream2 + bh / 2,
            "[0.41, -0.89, 0.71, ... ]  1024 numbers",
            fontsize=13, color=C_STREAM, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_STREAM,
                      lw=1.5, alpha=0.9))

    # ── MLP ──
    arrow_v(ax, cx, y_stream2, y_mlp + bh)
    rbox(ax, cx - bw / 2, y_mlp, bw, bh, "MLP", C_MLP,
         fontsize=18, sublabel="1024 → 4096 → 1024")

    # ── Stream after MLP ──
    arrow_v(ax, cx, y_mlp, y_stream3 + bh)
    ax.text(cx, y_stream3 + bh / 2,
            "[0.58, -0.62, 0.93, ... ]  1024 numbers",
            fontsize=13, color=C_STREAM, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_STREAM,
                      lw=1.5, alpha=0.9))

    # ── ★ STEERING ──
    arrow_v(ax, cx, y_stream3, y_steer + bh + 0.4, color=C_STEER, lw=3.5)

    steer_w = bw + 2.5
    steer_h = bh + 0.5
    steer_box = FancyBboxPatch((cx - steer_w / 2, y_steer), steer_w, steer_h,
                               boxstyle="round,pad=0.15",
                               facecolor=C_STEER, edgecolor="#C0392B",
                               linewidth=3.0, alpha=0.95, zorder=3)
    ax.add_patch(steer_box)
    ax.text(cx, y_steer + steer_h / 2 + 0.2,
            "STEERING HAPPENS HERE",
            fontsize=19, color="white", ha="center", va="center",
            fontweight="bold", zorder=4)
    ax.text(cx, y_steer + steer_h / 2 - 0.3,
            "stream = stream + scale x direction",
            fontsize=14, color="white", ha="center", va="center",
            fontfamily="monospace", zorder=4, alpha=0.9)

    # Direction vector coming from left side
    dir_label_x = cx - steer_w / 2 - 4.5
    ax.annotate("",
                xy=(cx - steer_w / 2, y_steer + steer_h / 2),
                xytext=(dir_label_x + 3.5, y_steer + steer_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_STEER,
                                lw=3.0, mutation_scale=20))
    ax.text(dir_label_x + 1.5, y_steer + steer_h / 2 + 0.8,
            "direction vector",
            fontsize=16, color=C_STEER, ha="center", va="bottom",
            fontweight="bold")
    ax.text(dir_label_x + 1.5, y_steer + steer_h / 2 + 0.3,
            "(1024 numbers)",
            fontsize=14, color=C_STEER, ha="center", va="bottom")
    ax.text(dir_label_x + 1.5, y_steer + steer_h / 2 - 0.6,
            "from SAE decoder\nor linear probe",
            fontsize=12, color=C_STEER, ha="center", va="top",
            style="italic", alpha=0.8)

    # ── Stream after steering ──
    arrow_v(ax, cx, y_steer, y_stream4 + bh, color=C_STEER, lw=3.5)
    ax.text(cx, y_stream4 + bh / 2,
            "[0.82, -0.31, 1.47, ... ]  1024 numbers",
            fontsize=13, color=C_STEER, ha="center", va="center",
            fontfamily="monospace", fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFEBEE", ec=C_STEER,
                      lw=2.0, alpha=0.9))
    ax.text(cx + 4.8, y_stream4 + bh / 2,
            "shifted!",
            fontsize=15, color=C_STEER, ha="left", va="center",
            fontweight="bold")

    # ── Predict ──
    arrow_v(ax, cx, y_stream4, y_predict + bh)
    rbox(ax, cx - bw / 2, y_predict, bw, bh, "Predict next token", C_OUTPUT,
         fontsize=18)

    # ── Title ──
    ax.set_xlim(-4.0, 16.0)
    ax.set_ylim(y_predict - 1.2, y_token + bh + 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(cx, y_token + bh + 1.2,
            "TinyStories-1Layer-21M: one pass through the model",
            fontsize=24, fontweight="bold", color=C_TEXT, ha="center")


def draw_right(ax):
    """Right panel: zoomed-in view of the vector addition."""
    y_top = 16.0
    cx = 6.0

    ax.text(cx, y_top + 1.5,
            "What steering looks like — zoomed in",
            fontsize=24, fontweight="bold", color=C_TEXT, ha="center")
    ax.text(cx, y_top + 0.6,
            "It's literally adding numbers to numbers",
            fontsize=16, color="#888888", ha="center", style="italic")

    # ── The three vectors ──
    row_h = 2.2
    box_w = 10.5
    left = cx - box_w / 2

    # Residual stream
    y_r = y_top - 2.0
    rbox(ax, left, y_r, box_w, row_h, "", C_STREAM, alpha=0.15, lw=1.5)
    ax.text(left + 0.5, y_r + row_h / 2 + 0.45,
            "residual stream", fontsize=16, color=C_STREAM,
            fontweight="bold", va="center")
    ax.text(left + 0.5, y_r + row_h / 2 - 0.35,
            "[ 0.58,  -0.62,   0.93,   0.11,  -0.44,  ...  0.27 ]",
            fontsize=14, color=C_TEXT, fontfamily="monospace", va="center")
    ax.text(left + box_w - 0.4, y_r + row_h - 0.3,
            "1024 numbers", fontsize=13, color=C_FAINT,
            ha="right", va="top", style="italic")

    # Plus sign
    y_plus = y_r - 1.0
    ax.text(cx, y_plus, "+", fontsize=42, fontweight="bold",
            color=C_STEER, ha="center", va="center")

    # Direction vector
    y_d = y_plus - 2.0
    rbox(ax, left, y_d, box_w, row_h, "", C_STEER, alpha=0.15, lw=1.5)
    ax.text(left + 0.5, y_d + row_h / 2 + 0.45,
            "15 x direction vector  (scale=15, feature f665)",
            fontsize=16, color=C_STEER, fontweight="bold", va="center")
    ax.text(left + 0.5, y_d + row_h / 2 - 0.35,
            "[ 0.24,   0.31,   0.54,  -0.08,   0.19,  ...  0.05 ]",
            fontsize=14, color=C_TEXT, fontfamily="monospace", va="center")
    ax.text(left + box_w - 0.4, y_d + row_h - 0.3,
            "1024 numbers", fontsize=13, color=C_FAINT,
            ha="right", va="top", style="italic")

    # Equals sign
    y_eq = y_d - 1.0
    ax.text(cx, y_eq, "=", fontsize=42, fontweight="bold",
            color=C_TEXT, ha="center", va="center")

    # Result
    y_res = y_eq - 2.0
    rbox(ax, left, y_res, box_w, row_h, "", C_STEER, alpha=0.3, lw=2.5)
    ax.text(left + 0.5, y_res + row_h / 2 + 0.45,
            "steered residual stream", fontsize=16, color=C_STEER,
            fontweight="bold", va="center")
    ax.text(left + 0.5, y_res + row_h / 2 - 0.35,
            "[ 0.82,  -0.31,   1.47,   0.03,  -0.25,  ...  0.32 ]",
            fontsize=14, color=C_TEXT, fontfamily="monospace",
            fontweight="bold", va="center")
    ax.text(left + box_w - 0.4, y_res + row_h - 0.3,
            "1024 numbers", fontsize=13, color=C_FAINT,
            ha="right", va="top", style="italic")

    # ── Arrow to output ──
    y_arrow_end = y_res - 1.5
    ax.annotate("", xy=(cx, y_arrow_end),
                xytext=(cx, y_res),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=2.5, mutation_scale=20))

    # ── Output comparison ──
    y_out = y_arrow_end - 3.5
    out_w = box_w
    out_h = 3.2

    # Normal output
    out_box1 = FancyBboxPatch((left, y_out), out_w / 2 - 0.4, out_h,
                              boxstyle="round,pad=0.15",
                              facecolor="#E3F2FD", edgecolor=C_STREAM,
                              linewidth=1.5, zorder=2)
    ax.add_patch(out_box1)
    ax.text(left + out_w / 4 - 0.2, y_out + out_h - 0.5,
            "Without steering:", fontsize=15, color=C_STREAM,
            fontweight="bold", ha="center", va="top")
    ax.text(left + out_w / 4 - 0.2, y_out + out_h / 2 - 0.3,
            '"the dark sky wept.\nThe clouds seemed\nto go away and,\nin the night..."',
            fontsize=13, color="#444444", ha="center", va="center",
            style="italic")

    # Steered output
    out_box2 = FancyBboxPatch((left + out_w / 2 + 0.4, y_out),
                              out_w / 2 - 0.4, out_h,
                              boxstyle="round,pad=0.15",
                              facecolor="#FFEBEE", edgecolor=C_STEER,
                              linewidth=1.5, zorder=2)
    ax.add_patch(out_box2)
    ax.text(left + 3 * out_w / 4 + 0.2, y_out + out_h - 0.5,
            "With steering:", fontsize=15, color=C_STEER,
            fontweight="bold", ha="center", va="top")
    ax.text(left + 3 * out_w / 4 + 0.2, y_out + out_h / 2 - 0.3,
            '"It was dark.\nI went to sleep.\nIt was dark.\nI woke up."',
            fontsize=13, color="#444444", ha="center", va="center",
            style="italic")

    # ── Bottom note ──
    ax.text(cx, y_out - 1.0,
            "That's the whole trick. Add 1024 numbers to 1024 numbers.\n"
            "Do it at every token during generation. Behavior changes.",
            fontsize=15, color="#666666", ha="center", va="top",
            style="italic", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5",
                      ec="#DDDDDD", lw=1.0))

    ax.set_xlim(-1.5, 13.5)
    ax.set_ylim(y_out - 3.0, y_top + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/residual_stream.png")
    args = parser.parse_args()

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full two-panel figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(28, 24),
                                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(C_BG)
    draw_left(ax_left)
    draw_right(ax_right)
    plt.tight_layout(pad=2.0)
    out = Path(args.output)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved -> {out}")
    plt.close(fig)

    # Left panel only (for embedding in the app)
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 24))
    fig2.patch.set_facecolor(C_BG)
    draw_left(ax2)
    plt.tight_layout(pad=1.0)
    out2 = out_dir / "residual_stream_model.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved -> {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()