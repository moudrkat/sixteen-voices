#!/usr/bin/env python3
"""Architecture diagram annotated with all findings from both articles.

Usage:
    python scripts/fig_architecture_annotated.py
    python scripts/fig_architecture_annotated.py --output figures/architecture_annotated.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colors ──────────────────────────────────────────────────────────
C_BASE = "#E8E8E8"
C_ATTN = "#4A90D9"
C_LORA = "#E8573A"
C_HEAD = "#6AB04C"
C_MLP = "#9B59B6"
C_EMBED = "#F39C12"
C_TEXT = "#2C3E50"
C_ARROW = "#555555"

# Finding-specific head colors
C_H11 = "#2E86C1"   # blue — dominant workhorse
C_H14 = "#E74C3C"   # red — polarizing wildcard
C_H3 = "#27AE60"    # green — quiet workhorse

# Annotation colors
C_NOTE_BG = "#FFFDE7"      # warm cream for callout boxes
C_NOTE_BORDER = "#BDBDBD"
C_FINDING = "#1A237E"       # dark blue for finding text
C_SAE = "#FF6F00"           # orange for SAE annotations


def rounded_box(ax, x, y, w, h, label, color, fontsize=9, text_color="white",
                alpha=1.0, style="round,pad=0.1", lw=1.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)
    return box


def callout(ax, x, y, w, text, *, anchor_x, anchor_y,
            fontsize=8, bg=C_NOTE_BG, border=C_NOTE_BORDER,
            text_color=C_FINDING, line_color="#999999",
            align="left", valign="top"):
    """Draw a callout box with a line connecting to an anchor point."""
    n_lines = text.count("\n") + 1
    line_h = fontsize * 0.038
    pad = 0.15
    h = n_lines * line_h + 2 * pad

    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         facecolor=bg, edgecolor=border,
                         linewidth=1.0, alpha=0.92, zorder=5)
    ax.add_patch(box)

    text_x = x + 0.1 if align == "left" else x + w - 0.1
    ha = "left" if align == "left" else "right"
    ax.text(text_x, y + h - pad, text, ha=ha, va=valign,
            fontsize=fontsize, color=text_color, zorder=6,
            linespacing=1.35)

    # Connector line
    ax.plot([anchor_x, x if anchor_x < x + w / 2 else x + w],
            [anchor_y, y + h / 2],
            color=line_color, lw=0.8, ls="--", zorder=4)
    return h


def draw(ax):
    # ── Layout constants (same as original) ──────────────────────
    cx = 5.0
    bw = 6.0
    gap = 0.4
    bh = 0.55

    n_heads = 16
    hw = 0.34
    hgap = 0.03
    head_h = 0.4
    head_total_w = n_heads * hw + (n_heads - 1) * hgap

    # Wider figure — annotations go on both sides
    left_margin = cx - bw / 2 - 6.5
    right_margin = cx + bw / 2 + 6.5

    # ══════════ BOTTOM TO TOP ══════════

    # --- Token + Position Embedding ---
    y = 0.5
    rounded_box(ax, cx - bw / 2, y, bw, bh,
                "Token + Position Embedding  (50,257 × 1024)", C_EMBED, fontsize=8)
    ax.annotate("", xy=(cx, y + bh + gap * 0.5),
                xytext=(cx, y + bh),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- LayerNorm 1 ---
    y_ln1 = y + bh + gap
    rounded_box(ax, cx - bw / 2, y_ln1, bw, 0.4,
                "LayerNorm", C_BASE, fontsize=9, text_color=C_TEXT)
    ax.annotate("", xy=(cx, y_ln1 + 0.4 + gap * 0.25),
                xytext=(cx, y_ln1 + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

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

    # --- Q, K, V projections ---
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

    # LoRA badges
    lora_w, lora_h = 0.9, 0.25
    rounded_box(ax, qx + (proj_w - lora_w) / 2, y_proj - lora_h - 0.08,
                lora_w, lora_h, "LoRA r=8", C_LORA, fontsize=5.5,
                style="round,pad=0.04", lw=1.0)
    rounded_box(ax, vx + (proj_w - lora_w) / 2, y_proj - lora_h - 0.08,
                lora_w, lora_h, "LoRA r=8", C_LORA, fontsize=5.5,
                style="round,pad=0.04", lw=1.0)

    # --- Arrow Q/K/V → heads ---
    y_mid = y_proj + proj_h + 0.15
    ax.annotate("", xy=(cx, y_mid + 0.15),
                xytext=(cx, y_proj + proj_h),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- 16 Heads (color-coded) ---
    y_heads = y_mid + 0.2
    head_x0 = cx - head_total_w / 2
    head_colors = {}
    for i in range(n_heads):
        if i == 11:
            c = C_H11
        elif i == 14:
            c = C_H14
        elif i == 3:
            c = C_H3
        else:
            c = C_HEAD
        hx = head_x0 + i * (hw + hgap)
        head_colors[i] = (hx, c)
        box = FancyBboxPatch(
            (hx, y_heads), hw, head_h, boxstyle="round,pad=0.03",
            facecolor=c, edgecolor="#444444", linewidth=0.7,
            alpha=0.85 if i in (3, 11, 14) else 0.45, zorder=2)
        ax.add_patch(box)
        ax.text(hx + hw / 2, y_heads + head_h / 2, f"H{i}",
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="white", zorder=3)

    ax.text(cx, y_heads + head_h + 0.15,
            "16 heads × 64d  —  specialization learned per author",
            ha="center", fontsize=7, color=C_TEXT, style="italic")

    # --- Arrow heads → concat ---
    y_pre_out = y_heads + head_h + 0.35
    ax.annotate("", xy=(cx, y_pre_out + 0.1),
                xytext=(cx, y_heads + head_h),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- Concat + Out proj ---
    y_out = y_pre_out + 0.15
    out_w = 4.5
    rounded_box(ax, cx - out_w / 2, y_out, out_w, 0.45,
                "Concat → Out proj  (1024→1024)", C_ATTN, fontsize=7.5, alpha=0.8)

    # --- Residual arrow + label ---
    y_res = y_attn + attn_h + gap * 0.6

    # Residual bypass line (right side)
    res_x = cx + bw / 2 + attn_pad + 0.15
    ax.plot([res_x, res_x], [y_attn, y_res + 0.2],
            color=C_ARROW, lw=1.0, ls="-", alpha=0.4, zorder=0)
    ax.annotate("", xy=(cx + bw / 2, y_res + 0.2),
                xytext=(res_x, y_res + 0.2),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=0.8, alpha=0.4))
    ax.text(res_x + 0.08, (y_attn + y_res + 0.2) / 2,
            "+ residual", fontsize=6.5, color=C_TEXT, style="italic",
            rotation=90, ha="left", va="center")

    ax.annotate("", xy=(cx, y_res),
                xytext=(cx, y_attn + attn_h),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- LayerNorm 2 ---
    rounded_box(ax, cx - bw / 2, y_res, bw, 0.4,
                "LayerNorm", C_BASE, fontsize=9, text_color=C_TEXT)
    ax.annotate("", xy=(cx, y_res + 0.4 + gap * 0.5),
                xytext=(cx, y_res + 0.4),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # ── SAE reads here ──
    sae_y = y_res + 0.15
    ax.annotate("SAE reads\nhere",
                xy=(cx - bw / 2, sae_y),
                xytext=(cx - bw / 2 - 1.2, sae_y),
                fontsize=7, color=C_SAE, fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_SAE, lw=1.5),
                zorder=6)

    # --- FFN / MLP ---
    y_mlp = y_res + 0.4 + gap
    rounded_box(ax, cx - bw / 2, y_mlp, bw, bh,
                "MLP  (1024 → 4096 → 1024)", C_MLP, fontsize=8.5)

    # Residual bypass line (right side)
    res2_x = res_x
    ax.plot([res2_x, res2_x], [y_res + 0.4, y_mlp + bh + 0.15],
            color=C_ARROW, lw=1.0, ls="-", alpha=0.4, zorder=0)
    ax.annotate("", xy=(cx + bw / 2, y_mlp + bh + 0.15),
                xytext=(res2_x, y_mlp + bh + 0.15),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=0.8, alpha=0.4))
    ax.text(res2_x + 0.08, (y_res + 0.4 + y_mlp + bh) / 2,
            "+ residual", fontsize=6.5, color=C_TEXT, style="italic",
            rotation=90, ha="left", va="center")

    ax.annotate("", xy=(cx, y_mlp + bh + gap * 0.5),
                xytext=(cx, y_mlp + bh),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))

    # --- LM Head ---
    y_lm = y_mlp + bh + gap
    rounded_box(ax, cx - bw / 2, y_lm, bw, bh,
                "LM Head  (1024 → 50,257)", C_EMBED, fontsize=8.5)

    # --- Title ---
    ax.text(cx, y_lm + bh + 0.65, "TinyStories-1Layer-21M + LoRA",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)
    ax.text(cx, y_lm + bh + 0.35,
            "1 layer · 16 heads · 21M params · 77 LoRA adapters · 1,232 knockout experiments",
            ha="center", fontsize=7.5, color="#666666")

    # ══════════ ANNOTATIONS — LEFT SIDE ══════════

    note_w = 4.0
    left_x = cx - bw / 2 - attn_pad - note_w - 1.2

    # ── V > Q callout (points to V proj) ──
    callout(ax, left_x, y_proj - 0.3, note_w,
            "V changes carry style,\n"
            "Q changes don't\n"
            "V-only > Q-only for 68/77 authors (88%)\n"
            "LoRA changes what heads output,\n"
            "not where they look",
            anchor_x=vx, anchor_y=y_proj + proj_h / 2,
            fontsize=7.5, text_color=C_FINDING)

    # ── LoRA callout (points to LoRA badge) ──
    callout(ax, left_x, y_proj - 1.8, note_w,
            "LoRAs amplify, they don't create\n"
            "98.8% of features already exist\n"
            "in the base model\n"
            "77 adapters × 33k params each (0.15%)",
            anchor_x=vx + (proj_w - lora_w) / 2,
            anchor_y=y_proj - lora_h / 2,
            fontsize=7.5, text_color=C_FINDING)

    # ── MLP callout (points to MLP box) ──
    callout(ax, left_x, y_mlp - 0.6, note_w,
            "Emergent directions\n"
            "27 head-independent features\n"
            "Strongest: simplicity (f665)\n"
            "No head controls it · invisible to knockouts\n"
            "Activation steering: 100% win rate",
            anchor_x=cx - bw / 2, anchor_y=y_mlp + bh / 2,
            fontsize=7.5, text_color=C_MLP)

    # ══════════ ANNOTATIONS — RIGHT SIDE ══════════

    right_x = cx + bw / 2 + attn_pad + 1.3

    # ── H3 callout (top right) ──
    h3_hx = head_colors[3][0]
    callout(ax, right_x, y_heads + 3.2, note_w,
            "H3 — quiet workhorse\n"
            "Consistent 2nd for almost all authors\n"
            "107 interpretable SAE features\n"
            "Never the main lever (0/77 dominant)\n"
            "Broadest feature landscape",
            anchor_x=h3_hx + hw / 2, anchor_y=y_heads + head_h,
            fontsize=7.5, text_color=C_H3)

    # ── H11 callout (middle right) ──
    h11_hx = head_colors[11][0]
    callout(ax, right_x, y_heads + 1.5, note_w,
            "H11 — dominant workhorse\n"
            "Best head for 51/77 authors (66%)\n"
            "Mean recovery: 0.38\n"
            "Zero feature overlap with other heads\n"
            "What it computes: unknown",
            anchor_x=h11_hx + hw, anchor_y=y_heads + head_h,
            fontsize=7.5, text_color=C_H11)

    # ── H14 callout (lower right) ──
    h14_hx = head_colors[14][0]
    callout(ax, right_x, y_heads - 0.7, note_w,
            "H14 — polarizing wildcard\n"
            "Best for 18 authors (Homer, Poe, Milton)\n"
            "Hurts 9 authors (Shelley, Wilde, Stoker)\n"
            "Suppresses \"I\" + conversational verbs\n"
            "Amplifies rare vocabulary\n"
            "End-to-end explainable",
            anchor_x=h14_hx + hw, anchor_y=y_heads + head_h / 2,
            fontsize=7.5, text_color=C_H14)

    # ── SAE feature types (bottom right) ──
    callout(ax, right_x, y_proj - 1.8, note_w,
            "SAE: 2048 features, 314 alive, ~25 interpretable\n"
            "Structural features → steer universally\n"
            "  (sentence length, punctuation, dialogue)\n"
            "Semantic features → detect but don't steer\n"
            "  (atmosphere, vocabulary, character voice)\n"
            "Detection ≠ steering",
            anchor_x=cx - bw / 2 - 1.2, anchor_y=sae_y,
            fontsize=7.5, text_color=C_SAE,
            bg="#FFF3E0", border="#FFB74D")

    # ── Anticorrelation note between H11 and H14 ──
    h11_mid_x = h11_hx + hw / 2
    h14_mid_x = h14_hx + hw / 2
    mid_x = (h11_mid_x + h14_mid_x) / 2
    ax.annotate("anticorrelated  r = −0.39",
                xy=(mid_x, y_heads - 0.18),
                fontsize=7, ha="center", color="#C62828",
                fontweight="bold", zorder=6)
    ax.annotate("", xy=(h11_mid_x, y_heads - 0.05),
                xytext=(h14_mid_x, y_heads - 0.05),
                arrowprops=dict(arrowstyle="<->", color="#C62828",
                                lw=1.5, connectionstyle="arc3,rad=-0.35"),
                zorder=5)

    # ── Legend ──
    legend_items = [
        (C_EMBED, "Embeddings"),
        (C_ATTN, "Attention"),
        (C_H11, "H11 (dominant)"),
        (C_H3, "H3 (workhorse)"),
        (C_H14, "H14 (polarizing)"),
        (C_HEAD, "Other heads"),
        (C_LORA, "LoRA"),
        (C_MLP, "MLP"),
    ]
    legend_y = 0.05
    lx_start = cx - 4.5
    for i, (color, label) in enumerate(legend_items):
        lx = lx_start + i * 1.25
        ax.add_patch(FancyBboxPatch(
            (lx, legend_y), 0.18, 0.18, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#333333", lw=0.6, zorder=2))
        ax.text(lx + 0.26, legend_y + 0.09, label,
                fontsize=5.5, va="center", color=C_TEXT)

    # --- Axis ---
    ax.set_xlim(left_margin, right_margin)
    ax.set_ylim(-0.3, y_lm + bh + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/architecture_annotated.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(20, 13))
    fig.patch.set_facecolor("white")
    draw(ax)
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()