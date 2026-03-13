#!/usr/bin/env python3
"""Draw model architecture diagram: 1-layer GPT-Neo + LoRA on q_proj/v_proj.

Usage:
    python scripts/fig_architecture.py
    python scripts/fig_architecture.py --output figures/architecture.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colors
C_BASE = "#E8E8E8"       # base model blocks
C_ATTN = "#4A90D9"       # attention
C_LORA = "#E8573A"       # LoRA adapters
C_LORA_LIGHT = "#F4A793" # LoRA inner
C_HEAD = "#6AB04C"       # individual heads
C_MLP = "#9B59B6"        # MLP / FFN
C_EMBED = "#F39C12"      # embeddings
C_TEXT = "#2C3E50"        # text color
C_ARROW = "#555555"


def rounded_box(ax, x, y, w, h, label, color, fontsize=9, text_color="white",
                alpha=1.0, style="round,pad=0.1", lw=1.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
    return box


def arrow(ax, x0, y0, x1, y1, **kwargs):
    style = "Simple,tail_width=2,head_width=8,head_length=5"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, color=C_ARROW, lw=1.2, zorder=1,
        connectionstyle="arc3,rad=0", **kwargs,
    ))


def draw_architecture(ax):
    # --- Layout: everything flows bottom-to-top, centered on cx ---
    cx = 5.0
    bw = 6.0           # main block width
    gap = 0.4          # vertical gap
    bh = 0.55          # standard block height

    # Head strip dimensions
    n_heads = 16
    hw = 0.34          # head box width
    hgap = 0.03        # gap between heads
    head_h = 0.4
    head_total_w = n_heads * hw + (n_heads - 1) * hgap  # ~5.89

    # ===================== BOTTOM TO TOP =====================

    # --- Token + Position Embedding ---
    y = 0.5
    rounded_box(ax, cx - bw / 2, y, bw, bh,
                "Token + Position Embedding  (50,257 × 1024)", C_EMBED, fontsize=8)
    arrow(ax, cx, y + bh, cx, y + bh + gap)

    # --- LayerNorm 1 ---
    y_ln1 = y + bh + gap
    rounded_box(ax, cx - bw / 2, y_ln1, bw, 0.4,
                "LayerNorm", C_BASE, fontsize=9, text_color=C_TEXT)
    arrow(ax, cx, y_ln1 + 0.4, cx, y_ln1 + 0.4 + gap * 0.5)

    # --- Attention outer box ---
    y_attn = y_ln1 + 0.4 + gap * 0.5
    attn_h = 3.6
    attn_pad = 0.25
    attn_box = FancyBboxPatch(
        (cx - bw / 2 - attn_pad, y_attn), bw + 2 * attn_pad, attn_h,
        boxstyle="round,pad=0.12", facecolor=C_ATTN, edgecolor="#333333",
        linewidth=1.8, alpha=0.10, zorder=0)
    ax.add_patch(attn_box)
    ax.text(cx, y_attn + attn_h - 0.22, "Multi-Head Attention",
            ha="center", fontsize=10, color=C_ATTN, fontweight="bold", zorder=3)

    # --- Q, K, V projections (inside attention box) ---
    proj_w = 1.6
    proj_h = 0.55
    proj_spacing = 0.2
    total_proj_w = 3 * proj_w + 2 * proj_spacing
    y_proj = y_attn + 0.55

    qx = cx - total_proj_w / 2
    kx = qx + proj_w + proj_spacing
    vx = kx + proj_w + proj_spacing

    rounded_box(ax, qx, y_proj, proj_w, proj_h,
                "Q proj", C_ATTN, fontsize=8, alpha=0.9)
    rounded_box(ax, kx, y_proj, proj_w, proj_h,
                "K proj", C_ATTN, fontsize=8, alpha=0.50)
    rounded_box(ax, vx, y_proj, proj_w, proj_h,
                "V proj", C_ATTN, fontsize=8, alpha=0.9)

    # LoRA badges — below Q and V
    lora_w, lora_h = 0.9, 0.25
    rounded_box(ax, qx + (proj_w - lora_w) / 2, y_proj - lora_h - 0.08,
                lora_w, lora_h, "LoRA r=8", C_LORA, fontsize=5.5,
                style="round,pad=0.04", lw=1.0)
    rounded_box(ax, vx + (proj_w - lora_w) / 2, y_proj - lora_h - 0.08,
                lora_w, lora_h, "LoRA r=8", C_LORA, fontsize=5.5,
                style="round,pad=0.04", lw=1.0)

    # --- Arrow from Q/K/V up to heads ---
    y_mid = y_proj + proj_h + 0.15
    ax.annotate("", xy=(cx, y_mid + 0.15),
                xytext=(cx, y_proj + proj_h),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- 16 Heads ---
    y_heads = y_mid + 0.2
    head_x0 = cx - head_total_w / 2
    for i in range(n_heads):
        hx = head_x0 + i * (hw + hgap)
        box = FancyBboxPatch(
            (hx, y_heads), hw, head_h, boxstyle="round,pad=0.03",
            facecolor=C_HEAD, edgecolor="#444444", linewidth=0.7,
            alpha=0.8, zorder=2)
        ax.add_patch(box)
        ax.text(hx + hw / 2, y_heads + head_h / 2, f"H{i}",
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="white", zorder=3)

    # Head strip label
    ax.text(cx, y_heads + head_h + 0.15,
            "16 heads × 64d  —  specialization learned per author",
            ha="center", fontsize=7, color=C_TEXT, style="italic")

    # --- Arrow from heads to concat ---
    y_pre_out = y_heads + head_h + 0.35
    ax.annotate("", xy=(cx, y_pre_out + 0.1),
                xytext=(cx, y_heads + head_h),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- Concat + Out proj ---
    y_out = y_pre_out + 0.15
    out_w = 4.5
    rounded_box(ax, cx - out_w / 2, y_out, out_w, 0.45,
                "Concat → Out proj  (1024→1024)", C_ATTN, fontsize=7.5, alpha=0.8)

    # --- Residual + LayerNorm 2 ---
    y_res = y_attn + attn_h + gap * 0.6
    ax.text(cx + bw / 2 + attn_pad + 0.1, (y_attn + attn_h + y_res) / 2,
            "+ residual", fontsize=7, color=C_TEXT, style="italic")
    arrow(ax, cx, y_attn + attn_h, cx, y_res)

    rounded_box(ax, cx - bw / 2, y_res, bw, 0.4,
                "LayerNorm", C_BASE, fontsize=9, text_color=C_TEXT)
    arrow(ax, cx, y_res + 0.4, cx, y_res + 0.4 + gap)

    # --- FFN ---
    y_mlp = y_res + 0.4 + gap
    rounded_box(ax, cx - bw / 2, y_mlp, bw, bh,
                "FFN  (1024 → 4096 → 1024)", C_MLP, fontsize=8.5)
    ax.text(cx + bw / 2 + attn_pad + 0.1, y_mlp + bh / 2,
            "+ residual", fontsize=7, color=C_TEXT, style="italic", va="center")
    arrow(ax, cx, y_mlp + bh, cx, y_mlp + bh + gap)

    # --- LM Head ---
    y_lm = y_mlp + bh + gap
    rounded_box(ax, cx - bw / 2, y_lm, bw, bh,
                "LM Head  (1024 → 50,257)", C_EMBED, fontsize=8.5)

    # --- Title ---
    ax.text(cx, y_lm + bh + 0.55, "TinyStories-1Layer-21M + LoRA",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)
    ax.text(cx, y_lm + bh + 0.25,
            "1 layer · 16 heads · 21M params · LoRA rank 8 on Q, V",
            ha="center", fontsize=8.5, color="#666666")

    # --- Legend ---
    legend_items = [
        (C_EMBED, "Embeddings"),
        (C_ATTN, "Attention"),
        (C_HEAD, "Heads"),
        (C_LORA, "LoRA"),
        (C_MLP, "FFN"),
    ]
    legend_y = 0.05
    lx_start = cx - 3.5
    for i, (color, label) in enumerate(legend_items):
        lx = lx_start + i * 1.5
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y), 0.22, 0.2, boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="#333333", lw=0.7, zorder=2))
        ax.text(lx + 0.32, legend_y + 0.1, label,
                fontsize=7, va="center", color=C_TEXT)

    # --- Axis ---
    ax.set_xlim(cx - bw / 2 - 1, cx + bw / 2 + 1.5)
    ax.set_ylim(-0.2, y_lm + bh + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Draw model architecture diagram")
    parser.add_argument("--output", type=str, default="figures/architecture.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(10, 11))
    fig.patch.set_facecolor("white")
    draw_architecture(ax)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()