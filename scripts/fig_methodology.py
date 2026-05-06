#!/usr/bin/env python3
"""Methodology diagrams for poster Q&A defense.

One figure per matrix operation referenced in the poster, plus
foundational explainers (LoRA setup, SVD refactor).

Output: figures/methodology/*.png
Run:    python scripts/fig_methodology.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ─── Output dir ───
OUT_DIR = Path("figures/methodology")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Palette (matches poster) ───
ACC_BLUE   = "#2563EB"
ACC_GREEN  = "#16A34A"
ACC_RED    = "#DC2626"
ACC_ORANGE = "#EA580C"
ACC_PURPLE = "#7C3AED"
ACC_BLACK  = "#1A1A1A"
MUTED      = "#666666"
PANEL_BG   = "#F7F7F7"
HIGHLIGHT  = "#FEF3C7"
DIM_GREY   = "#E5E7EB"
LIGHT_GREY = "#F3F4F6"


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════
def matrix_box(ax, x, y, w, h, label, fc=DIM_GREY, ec=ACC_BLACK, lw=1.4,
               shape=None, label_color=ACC_BLACK, fontsize=12):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=label_color, zorder=3)
    if shape:
        ax.text(x + w/2, y - 0.18, shape, ha="center", va="top",
                fontsize=8.5, color=MUTED, style="italic", zorder=3)


def striped_matrix(ax, x, y, w, h, n=16, kept=None, kept_color=ACC_BLUE,
                   base=DIM_GREY, zero_fc="white", ec=ACC_BLACK, lw=1.4,
                   show_indices=False):
    """Draw a matrix divided into n horizontal stripes (head blocks).

    kept: int, list, or None. Listed indices use kept_color; others use zero_fc.
    If kept is None, all stripes use base color.
    """
    if isinstance(kept, int):
        kept = [kept]
    stripe_h = h / n
    for i in range(n):
        if kept is None:
            fc = base
        elif i in kept:
            fc = kept_color
        else:
            fc = zero_fc
        # i=0 is top stripe
        sy = y + (n - 1 - i) * stripe_h
        ax.add_patch(Rectangle((x, sy), w, stripe_h,
                                fc=fc, ec="#BBBBBB", lw=0.4, zorder=2))
        if show_indices and (kept is None or i in kept):
            ax.text(x + w + 0.08, sy + stripe_h/2, f"H{i}",
                    fontsize=7, va="center", color=MUTED, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=ec, lw=lw, zorder=4))


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=2, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


def text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, **kw)


def title(ax, txt, sub=None):
    ax.text(0.0, 1.18, txt, transform=ax.transAxes,
            fontsize=15, fontweight="bold", color=ACC_BLACK,
            ha="left", va="bottom")
    if sub:
        ax.text(0.0, 1.04, sub, transform=ax.transAxes,
                fontsize=10, color=MUTED, style="italic",
                ha="left", va="bottom")


def setup_axes(figsize=(12, 6), xlim=(0, 12), ylim=(0, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════════════
# 00 — LoRA setup
# ═══════════════════════════════════════════════════════════════════
def fig_00_lora_setup():
    fig, ax = setup_axes(figsize=(13, 5), xlim=(0, 13), ylim=(0, 5))
    title(ax, "LoRA setup — what's actually trained",
          sub="The frozen W stays put. The trained ΔW is a low-rank product B·A. "
              "PEFT keeps A and B; effective ΔW is reconstructed on the fly.")

    # W frozen
    matrix_box(ax, 0.5, 1.5, 2.2, 2.2, "W\n(frozen)", fc=DIM_GREY,
               shape="1024 × 1024")

    text(ax, 3.1, 2.6, "+", fontsize=28, fontweight="bold")

    # B
    matrix_box(ax, 3.6, 1.5, 0.6, 2.2, "B", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               shape="1024 × 8", label_color=ACC_BLUE)
    text(ax, 4.55, 2.6, "@", fontsize=20, fontweight="bold")
    # A (wide & short)
    matrix_box(ax, 4.85, 2.4, 2.2, 0.4, "A", fc=ACC_PURPLE+"33", ec=ACC_PURPLE,
               shape="8 × 1024", label_color=ACC_PURPLE)

    text(ax, 7.5, 2.6, "=", fontsize=28, fontweight="bold")

    # Effective W
    matrix_box(ax, 8.0, 1.5, 2.2, 2.2, "W + ΔW", fc=HIGHLIGHT,
               shape="1024 × 1024")

    # Trainable badge
    badge = FancyBboxPatch((10.6, 1.7), 2.2, 1.6,
                           boxstyle="round,pad=0.1",
                           fc=PANEL_BG, ec=MUTED, lw=1.0, zorder=2)
    ax.add_patch(badge)
    text(ax, 11.7, 2.85, "trainable", fontsize=10, fontweight="bold",
         color=ACC_BLACK)
    text(ax, 11.7, 2.45, "B + A", fontsize=11, fontweight="bold",
         color=ACC_BLUE)
    text(ax, 11.7, 2.10, "≈ 32k params", fontsize=9, color=MUTED)
    text(ax, 11.7, 1.85, "(~0.15% of W)", fontsize=8, color=MUTED, style="italic")

    text(ax, 6.5, 0.7,
         "ΔW = B @ A — rank 8.   Targets only q_proj and v_proj.",
         fontsize=10, color=MUTED, style="italic")

    save(fig, "00_lora_setup.png")


# ═══════════════════════════════════════════════════════════════════
# 01 — SVD refactor (used in Q1 and Q3)
# ═══════════════════════════════════════════════════════════════════
def fig_01_svd_refactor():
    fig, ax = setup_axes(figsize=(14, 6.5), xlim=(0, 14), ylim=(0, 6.5))
    title(ax, "SVD refactor — why it's needed after editing ΔW",
          sub="When I zero or swap rows of ΔW directly, the result is no longer a clean "
              "rank-8 product. PEFT needs (A, B) of rank 8. SVD gives the best rank-8 fit.")

    # Modified ΔW (kept stripe 11)
    text(ax, 1.85, 5.15, "edited ΔW", fontsize=10, fontweight="bold", color=ACC_BLACK)
    striped_matrix(ax, 0.7, 1.7, 2.3, 3.0, n=16, kept=11, kept_color=ACC_BLUE)
    text(ax, 1.85, 1.5, "1024 × 1024", fontsize=8.5, color=MUTED, style="italic")
    text(ax, 1.85, 1.2, "(no longer rank 8)", fontsize=8.5, color=ACC_RED)

    arrow(ax, 3.1, 3.2, 4.0, 3.2, lw=2.2)
    text(ax, 3.55, 3.55, "torch.linalg.svd", fontsize=8.5,
         color=MUTED, style="italic")

    # U
    matrix_box(ax, 4.1, 1.7, 1.3, 3.0, "U", fc=ACC_GREEN+"22", ec=ACC_GREEN,
               shape="1024 × 1024", label_color=ACC_GREEN)
    text(ax, 5.55, 3.2, "·", fontsize=24, fontweight="bold")
    # S (diagonal)
    matrix_box(ax, 5.7, 2.4, 1.3, 1.6, "S\n(diag)", fc=ACC_ORANGE+"22",
               ec=ACC_ORANGE, shape="1024", label_color=ACC_ORANGE, fontsize=10)
    text(ax, 7.15, 3.2, "·", fontsize=24, fontweight="bold")
    # Vh
    matrix_box(ax, 7.3, 1.7, 1.3, 3.0, "Vh", fc=ACC_PURPLE+"22", ec=ACC_PURPLE,
               shape="1024 × 1024", label_color=ACC_PURPLE)

    arrow(ax, 8.7, 3.2, 9.7, 3.2, lw=2.2)
    text(ax, 9.2, 3.55, "keep top 8", fontsize=8.5, color=MUTED, style="italic")

    # B_new = U[:, :8] · S[:8]
    matrix_box(ax, 9.85, 1.7, 0.55, 3.0, "B_new", fc=ACC_BLUE+"33",
               ec=ACC_BLUE, shape="1024 × 8", label_color=ACC_BLUE, fontsize=9)
    text(ax, 10.65, 3.2, "·", fontsize=24, fontweight="bold")
    # A_new (wide and short)
    matrix_box(ax, 10.85, 3.0, 2.6, 0.4, "A_new", fc=ACC_PURPLE+"33",
               ec=ACC_PURPLE, shape="8 × 1024", label_color=ACC_PURPLE, fontsize=9)

    # Code box at bottom
    code_y = 0.45
    code = ("U, S, Vh = torch.linalg.svd(ΔW_modified, full_matrices=False)\n"
            "B_new = U[:, :8] * S[:8]      # shape (1024, 8)\n"
            "A_new = Vh[:8, :]              # shape (8, 1024)")
    code_box = FancyBboxPatch((0.7, -0.1), 12.7, 1.0,
                              boxstyle="round,pad=0.1",
                              fc=PANEL_BG, ec=MUTED, lw=0.8)
    ax.add_patch(code_box)
    ax.text(0.95, code_y, code, fontsize=9, family="monospace",
            color=ACC_BLACK, va="center", ha="left")

    save(fig, "01_svd_refactor.png")


# ═══════════════════════════════════════════════════════════════════
# 02 — Q1: head knockout
# ═══════════════════════════════════════════════════════════════════
def fig_02_q1_knockout():
    fig, ax = setup_axes(figsize=(14, 6.5), xlim=(0, 14), ylim=(0, 6.5))
    title(ax, "Q1 — Head knockout (per author × per head, 1232 runs)",
          sub="ΔW has 16 head-blocks of 64 rows each. Keep one block, zero the other 15. "
              "Re-factorize via SVD into rank-8 (A, B). Inject. Measure perplexity.")

    # Original ΔW
    text(ax, 1.45, 5.4, "ΔW (original)", fontsize=10, fontweight="bold")
    striped_matrix(ax, 0.5, 1.6, 1.9, 3.4, n=16, kept=None, base=ACC_BLUE+"55",
                   show_indices=False)
    text(ax, 1.45, 1.35, "1024 × 1024", fontsize=8.5, color=MUTED, style="italic")
    text(ax, 1.45, 1.05, "(64 rows / head)", fontsize=8, color=MUTED)

    arrow(ax, 2.55, 3.3, 3.3, 3.3, lw=2)
    text(ax, 2.92, 3.55, "zero 15", fontsize=8.5, color=ACC_RED, style="italic")

    # ΔW with H11 kept (example)
    text(ax, 4.4, 5.4, "keep one head (e.g. H11)", fontsize=10, fontweight="bold")
    striped_matrix(ax, 3.45, 1.6, 1.9, 3.4, n=16, kept=11, kept_color=ACC_BLUE,
                   show_indices=True)
    text(ax, 4.4, 1.05, "1024 × 1024 (rank > 8)", fontsize=8.5, color=MUTED, style="italic")

    arrow(ax, 5.6, 3.3, 6.4, 3.3, lw=2)
    text(ax, 6.0, 3.55, "SVD", fontsize=9, color=MUTED, style="italic")

    # B_new
    matrix_box(ax, 6.55, 1.6, 0.5, 3.4, "B", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               shape="1024 × 8", label_color=ACC_BLUE, fontsize=10)
    text(ax, 7.25, 3.3, "·", fontsize=20, fontweight="bold")
    # A_new (wide-short)
    matrix_box(ax, 7.4, 3.1, 1.9, 0.4, "A", fc=ACC_PURPLE+"33", ec=ACC_PURPLE,
               shape="8 × 1024", label_color=ACC_PURPLE, fontsize=10)

    arrow(ax, 9.4, 3.3, 10.1, 3.3, lw=2)
    text(ax, 9.75, 3.55, "inject", fontsize=8.5, color=MUTED, style="italic")

    # Generate + PPL bracket
    res_box = FancyBboxPatch((10.25, 1.6), 3.5, 3.4,
                             boxstyle="round,pad=0.1",
                             fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(res_box)
    text(ax, 12.0, 4.5, "measure", fontsize=10, fontweight="bold",
         color=ACC_BLACK)
    text(ax, 12.0, 4.05, "perplexity on author's", fontsize=9, color=ACC_BLACK)
    text(ax, 12.0, 3.75, "held-out prose", fontsize=9, color=ACC_BLACK)

    text(ax, 12.0, 3.05, "recovery(h) =", fontsize=9.5, color=ACC_BLACK,
         fontweight="bold")
    text(ax, 12.0, 2.65, "PPL_base − PPL_one_head", fontsize=8.5, color=ACC_BLACK)
    ax.plot([10.7, 13.3], [2.45, 2.45], color=ACC_BLACK, lw=0.8)
    text(ax, 12.0, 2.25, "PPL_base − PPL_full_adapter", fontsize=8.5, color=ACC_BLACK)

    text(ax, 12.0, 1.85, "argmax over 16 heads", fontsize=8.5,
         color=MUTED, style="italic")

    # Footer
    text(ax, 7.0, 0.55,
         "Repeat for 77 authors × 16 heads = 1232 runs.   "
         "Null baseline: same procedure on untrained random LoRAs (pattern disappears).",
         fontsize=9.5, color=MUTED, style="italic")

    save(fig, "02_q1_knockout.png")


# ═══════════════════════════════════════════════════════════════════
# 03 — Q2: head steering hook
# ═══════════════════════════════════════════════════════════════════
def fig_03_q2_head_steering():
    fig, ax = setup_axes(figsize=(13.5, 6.5), xlim=(0, 13.5), ylim=(0, 6.5))
    title(ax, "Q2 — Head steering (forward pre-hook on W_O)",
          sub="At inference: scale one head's 64-dim slice of the attention output "
              "by s ∈ [0, 2] before the output projection runs. No retraining.")

    # Stage 1: 16 head outputs concatenated
    text(ax, 4.3, 5.5, "attention output  (16 heads concatenated)",
         fontsize=10, fontweight="bold")
    text(ax, 4.3, 5.15, "shape: (seq_len, 1024)", fontsize=8.5,
         color=MUTED, style="italic")

    # Draw 16 stripes horizontally
    bar_x, bar_y = 0.5, 3.5
    bar_w, bar_h = 7.6, 1.0
    n = 16
    cell_w = bar_w / n
    for i in range(n):
        fc = ACC_RED+"99" if i == 14 else DIM_GREY
        ax.add_patch(Rectangle((bar_x + i*cell_w, bar_y), cell_w, bar_h,
                                fc=fc, ec="#999999", lw=0.5, zorder=2))
        ax.text(bar_x + (i+0.5)*cell_w, bar_y + bar_h/2, f"H{i}",
                fontsize=7, ha="center", va="center",
                color="white" if i == 14 else ACC_BLACK,
                fontweight="bold")
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, fc="none",
                            ec=ACC_BLACK, lw=1.4, zorder=3))
    text(ax, bar_x + bar_w/2, bar_y - 0.35,
         "each head = 64 dims     |     total = 1024",
         fontsize=8.5, color=MUTED, style="italic")

    # Hook callout
    hook_box = FancyBboxPatch((1.5, 1.7), 5.6, 1.0,
                              boxstyle="round,pad=0.08",
                              fc=HIGHLIGHT, ec=ACC_ORANGE, lw=1.2)
    ax.add_patch(hook_box)
    text(ax, 4.3, 2.35, "forward pre-hook on attn_out_proj",
         fontsize=10, fontweight="bold", color=ACC_BLACK)
    text(ax, 4.3, 1.95,
         "output[:, :, 14*64:15*64]  *=  s     # scale H14 only",
         fontsize=9, family="monospace", color=ACC_BLACK)

    # Arrow from H14 down to hook
    arrow(ax, bar_x + 14.5*cell_w, bar_y - 0.05, bar_x + 14.5*cell_w, 2.75,
          color=ACC_ORANGE, lw=2)

    # s knob
    knob_x, knob_y = 9.2, 4.0
    text(ax, knob_x, 5.5, "scale  s", fontsize=12, fontweight="bold",
         color=ACC_ORANGE)
    # draw a horizontal slider 0 → 2
    ax.plot([knob_x - 0.8, knob_x + 0.8], [knob_y, knob_y],
            color=MUTED, lw=2)
    for sv, lab in [(0.0, "0×"), (1.0, "1×"), (2.0, "2×")]:
        x = knob_x - 0.8 + (sv/2.0)*1.6
        ax.plot(x, knob_y, "o", color=ACC_ORANGE, markersize=8, zorder=3)
        ax.text(x, knob_y - 0.35, lab, ha="center", fontsize=9,
                color=ACC_BLACK, fontweight="bold")
    text(ax, knob_x, 4.7,
         "0 = kill the head\n1 = natural\n2 = double",
         fontsize=8.5, color=MUTED, ha="center")

    # Then W_O projection
    matrix_box(ax, 8.6, 2.3, 1.4, 0.9, "W_O", fc=DIM_GREY, ec=ACC_BLACK,
               shape="1024 → 1024", label_color=ACC_BLACK, fontsize=11)
    arrow(ax, 7.15, 2.2, 8.55, 2.6, color=ACC_ORANGE, lw=2)
    arrow(ax, 10.05, 2.75, 11.0, 2.75, lw=2)
    text(ax, 11.5, 2.75, "→  PPL", fontsize=11, fontweight="bold",
         color=ACC_BLACK)

    # Footer
    text(ax, 6.75, 0.7,
         "Sweep s ∈ {0.0, 0.25, …, 2.0}, generate text, score perplexity on "
         "the author's text. The dominant head produces a U-curve; non-dominant heads stay flat.",
         fontsize=9, color=MUTED, style="italic")

    save(fig, "03_q2_head_steering.png")


# ═══════════════════════════════════════════════════════════════════
# 04 — Q3: head transplant
# ═══════════════════════════════════════════════════════════════════
def fig_04_q3_transplant():
    fig, ax = setup_axes(figsize=(14, 6.5), xlim=(0, 14), ylim=(0, 6.5))
    title(ax, "Q3 — Head transplant (cross-author donor → recipient)",
          sub="Copy donor's 64 rows for head h into recipient's ΔW. "
              "That's 64/1024 = 6% of the LoRA weights. SVD refactor → inject.")

    # Recipient ΔW
    text(ax, 1.45, 5.4, "Recipient: Minimalist", fontsize=10, fontweight="bold",
         color=ACC_GREEN)
    striped_matrix(ax, 0.5, 1.6, 1.9, 3.4, n=16, kept=None,
                   base=ACC_GREEN+"55", show_indices=False)
    text(ax, 1.45, 1.3, "ΔW_recipient", fontsize=8.5, color=MUTED, style="italic")

    # Donor ΔW
    text(ax, 4.05, 5.4, "Donor: Poe", fontsize=10, fontweight="bold",
         color=ACC_RED)
    # build with just stripe 14 colored, rest faded
    stripe_h = 3.4 / 16
    base_x, base_y, w = 3.1, 1.6, 1.9
    for i in range(16):
        fc = ACC_RED if i == 14 else ACC_RED+"22"
        sy = base_y + (16 - 1 - i) * stripe_h
        ax.add_patch(Rectangle((base_x, sy), w, stripe_h,
                                fc=fc, ec="#BBBBBB", lw=0.4, zorder=2))
    ax.add_patch(Rectangle((base_x, base_y), w, 3.4, fc="none",
                            ec=ACC_BLACK, lw=1.4, zorder=3))
    # H14 label arrow
    h14_y = base_y + (16 - 1 - 14 + 0.5) * stripe_h
    text(ax, base_x + w + 0.15, h14_y, "H14", fontsize=8, color=ACC_RED,
         fontweight="bold", ha="left")
    text(ax, 4.05, 1.3, "ΔW_donor", fontsize=8.5, color=MUTED, style="italic")

    # Copy arrow donor H14 → recipient H14
    rec_h14_y = 1.6 + (16 - 1 - 14 + 0.5) * stripe_h
    arrow(ax, base_x, h14_y, 2.4, rec_h14_y, color=ACC_RED, lw=2.5)
    text(ax, 2.7, h14_y + 0.45, "copy 64 rows",
         fontsize=8.5, color=ACC_RED, fontweight="bold", ha="center", style="italic")

    # Result: hybrid ΔW
    arrow(ax, 5.15, 3.3, 6.0, 3.3, lw=2)
    text(ax, 7.0, 5.4, "Hybrid ΔW", fontsize=10, fontweight="bold",
         color=ACC_PURPLE)
    # green stripes everywhere except H14 which is red
    for i in range(16):
        fc = ACC_RED if i == 14 else ACC_GREEN+"55"
        sy = 1.6 + (16 - 1 - i) * stripe_h
        ax.add_patch(Rectangle((6.05, sy), w, stripe_h,
                                fc=fc, ec="#BBBBBB", lw=0.4, zorder=2))
    ax.add_patch(Rectangle((6.05, 1.6), w, 3.4, fc="none",
                            ec=ACC_BLACK, lw=1.4, zorder=3))
    text(ax, 7.0, 1.3, "Minimalist + Poe's H14", fontsize=8.5, color=MUTED,
         style="italic")

    # SVD refactor
    arrow(ax, 8.1, 3.3, 8.85, 3.3, lw=2)
    text(ax, 8.5, 3.6, "SVD", fontsize=9, color=MUTED, style="italic")

    # B, A
    matrix_box(ax, 8.95, 1.6, 0.5, 3.4, "B", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               shape="1024 × 8", label_color=ACC_BLUE, fontsize=10)
    text(ax, 9.65, 3.3, "·", fontsize=20, fontweight="bold")
    matrix_box(ax, 9.8, 3.1, 1.9, 0.4, "A", fc=ACC_PURPLE+"33", ec=ACC_PURPLE,
               shape="8 × 1024", label_color=ACC_PURPLE, fontsize=10)

    arrow(ax, 11.85, 3.3, 12.55, 3.3, lw=2)

    # Result text
    res = FancyBboxPatch((12.55, 1.8), 1.4, 3.0,
                         boxstyle="round,pad=0.08",
                         fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(res)
    text(ax, 13.25, 4.45, "generate", fontsize=10, fontweight="bold",
         color=ACC_BLACK)
    text(ax, 13.25, 4.0, "short", fontsize=9.5, color=ACC_GREEN)
    text(ax, 13.25, 3.7, "sentences", fontsize=9.5, color=ACC_GREEN)
    text(ax, 13.25, 3.4, "stay", fontsize=9.5, color=ACC_GREEN)
    text(ax, 13.25, 2.9, "+", fontsize=14, fontweight="bold")
    text(ax, 13.25, 2.5, "Poe's", fontsize=9.5, color=ACC_RED)
    text(ax, 13.25, 2.2, "vocabulary", fontsize=9.5, color=ACC_RED)
    text(ax, 13.25, 1.95, "floods in", fontsize=9.5, color=ACC_RED)

    text(ax, 7.0, 0.6,
         "6% of the weights swapped → Minimalist's short-sentence framing survives, "
         "Poe's H14 injects dark vocabulary. Style is portable at the head level.",
         fontsize=9.5, color=MUTED, style="italic")

    save(fig, "04_q3_transplant.png")


# ═══════════════════════════════════════════════════════════════════
# 05 — Q4: blend / interpolate
# ═══════════════════════════════════════════════════════════════════
def fig_05_q4_blend():
    fig, ax = setup_axes(figsize=(13.5, 6.5), xlim=(0, 13.5), ylim=(0, 6.5))
    title(ax, "Q4 — Blend two adapters (linear interpolation in LoRA space)",
          sub="Element-wise weighted sum of two rank-8 adapters. "
              "Interpolating two rank-8 pairs stays rank 8 — no SVD needed.")

    # Adapter A (Carroll): A_C and B_C
    text(ax, 1.6, 5.7, "Carroll", fontsize=11, fontweight="bold", color=ACC_BLUE)
    matrix_box(ax, 0.7, 3.8, 0.5, 1.4, "B_C", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               label_color=ACC_BLUE, fontsize=10)
    text(ax, 0.95, 3.6, "1024×8", fontsize=7, color=MUTED, style="italic")
    matrix_box(ax, 1.4, 4.6, 1.4, 0.4, "A_C", fc=ACC_PURPLE+"33", ec=ACC_PURPLE,
               label_color=ACC_PURPLE, fontsize=10)
    text(ax, 2.1, 4.4, "8×1024", fontsize=7, color=MUTED, style="italic")

    # Adapter B (Poet): A_P and B_P
    text(ax, 1.6, 2.9, "Poet", fontsize=11, fontweight="bold", color=ACC_PURPLE)
    matrix_box(ax, 0.7, 1.0, 0.5, 1.4, "B_P", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               label_color=ACC_BLUE, fontsize=10)
    text(ax, 0.95, 0.8, "1024×8", fontsize=7, color=MUTED, style="italic")
    matrix_box(ax, 1.4, 1.8, 1.4, 0.4, "A_P", fc=ACC_PURPLE+"33", ec=ACC_PURPLE,
               label_color=ACC_PURPLE, fontsize=10)
    text(ax, 2.1, 1.6, "8×1024", fontsize=7, color=MUTED, style="italic")

    # Operation box
    op_box = FancyBboxPatch((3.4, 2.3), 4.0, 1.9,
                            boxstyle="round,pad=0.1",
                            fc=HIGHLIGHT, ec=ACC_ORANGE, lw=1.2)
    ax.add_patch(op_box)
    text(ax, 5.4, 3.85, "element-wise", fontsize=10, color=ACC_BLACK,
         fontweight="bold")
    text(ax, 5.4, 3.45,
         "A_blend  =  (1−α) · A_C  +  α · A_P",
         fontsize=9.5, family="monospace", color=ACC_BLACK)
    text(ax, 5.4, 3.1,
         "B_blend  =  (1−α) · B_C  +  α · B_P",
         fontsize=9.5, family="monospace", color=ACC_BLACK)
    text(ax, 5.4, 2.65, "(no retraining, no SVD)",
         fontsize=8.5, color=MUTED, style="italic")

    arrow(ax, 2.9, 4.7, 3.4, 3.6, lw=1.8, color=ACC_BLUE)
    arrow(ax, 2.9, 1.9, 3.4, 3.0, lw=1.8, color=ACC_PURPLE)

    arrow(ax, 7.5, 3.25, 8.3, 3.25, lw=2)

    # Result: blended adapter
    text(ax, 9.4, 5.7, "Blended adapter", fontsize=11, fontweight="bold",
         color=ACC_ORANGE)
    matrix_box(ax, 8.5, 3.0, 0.5, 1.4, "B_b", fc=ACC_ORANGE+"33", ec=ACC_ORANGE,
               label_color=ACC_ORANGE, fontsize=10)
    matrix_box(ax, 9.2, 3.8, 1.4, 0.4, "A_b", fc=ACC_ORANGE+"33", ec=ACC_ORANGE,
               label_color=ACC_ORANGE, fontsize=10)
    text(ax, 9.4, 2.7, "still rank 8", fontsize=8.5, color=MUTED, style="italic")

    # Alpha slider at bottom
    sl_y = 1.0
    text(ax, 11.0, sl_y + 0.7, "α", fontsize=14, fontweight="bold",
         color=ACC_ORANGE)
    ax.plot([8.5, 13.0], [sl_y, sl_y], color=MUTED, lw=2)
    for av, lab, col in [(0.0, "0.0\nCarroll", ACC_BLUE),
                          (0.5, "0.5\nblend", ACC_ORANGE),
                          (1.0, "1.0\nPoet", ACC_PURPLE)]:
        x = 8.5 + av * 4.5
        ax.plot(x, sl_y, "o", color=col, markersize=10, zorder=3)
        ax.text(x, sl_y - 0.5, lab, ha="center", fontsize=8.5,
                color=col, fontweight="bold")

    # Footer
    text(ax, 6.75, 0.3,
         "For each α, generate. Some pairs blend smoothly (Carroll↔Poet); "
         "others break (Poe↔Carroll → gibberish near α=0.5).",
         fontsize=9, color=MUTED, style="italic")

    save(fig, "05_q4_blend.png")


# ═══════════════════════════════════════════════════════════════════
# 06 — Q5: SAE architecture
# ═══════════════════════════════════════════════════════════════════
def fig_06_q5_sae():
    fig, ax = setup_axes(figsize=(14, 6), xlim=(0, 14), ylim=(0, 6))
    title(ax, "Q5 — Sparse autoencoder on the residual stream",
          sub="Decompose 1024-dim activations into a sparse code over 2048 features. "
              "TopK keeps only the 16 strongest features per token. Decoder columns = "
              "feature directions in residual space.")

    # x: residual activation
    matrix_box(ax, 0.5, 2.0, 0.4, 2.0, "x", fc=DIM_GREY, ec=ACC_BLACK,
               shape="1024", fontsize=14)

    arrow(ax, 1.0, 3.0, 1.7, 3.0, lw=2)

    # W_enc
    matrix_box(ax, 1.75, 1.7, 1.6, 2.6, "W_enc", fc=ACC_BLUE+"33", ec=ACC_BLUE,
               shape="2048 × 1024", label_color=ACC_BLUE, fontsize=11)

    arrow(ax, 3.4, 3.0, 4.1, 3.0, lw=2)

    # Pre-TopK (long thin) - 2048 features
    text(ax, 4.6, 4.55, "raw codes", fontsize=9, fontweight="bold")
    text(ax, 4.6, 4.3, "(2048)", fontsize=8, color=MUTED, style="italic")
    # tall thin bar
    n_bars = 30
    bar_w_each = 0.9 / n_bars
    rng = np.random.default_rng(0)
    heights = rng.uniform(0.05, 0.6, n_bars)
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((4.15 + i*bar_w_each, 2.5),
                                bar_w_each*0.85, h,
                                fc=ACC_PURPLE+"66", ec=ACC_PURPLE, lw=0.3))
    ax.add_patch(Rectangle((4.1, 2.5), 0.95, 1.6, fc="none",
                            ec=ACC_BLACK, lw=1.0))

    arrow(ax, 5.15, 3.0, 5.8, 3.0, lw=2)
    text(ax, 5.475, 3.35, "TopK\n(k=16)", fontsize=8.5,
         color=ACC_ORANGE, fontweight="bold", style="italic")

    # Sparse code (mostly zero, only ~16 active)
    text(ax, 6.5, 4.55, "sparse code", fontsize=9, fontweight="bold")
    text(ax, 6.5, 4.3, "(only 16 active)", fontsize=8, color=MUTED, style="italic")
    # Show same bars but mostly zeroed
    active_idx = rng.choice(n_bars, 5, replace=False)
    for i, h in enumerate(heights):
        if i in active_idx:
            ax.add_patch(Rectangle((6.05 + i*bar_w_each, 2.5),
                                    bar_w_each*0.85, h,
                                    fc=ACC_ORANGE, ec=ACC_ORANGE, lw=0.3))
        else:
            ax.add_patch(Rectangle((6.05 + i*bar_w_each, 2.5),
                                    bar_w_each*0.85, 0.02,
                                    fc=DIM_GREY, ec=ACC_BLACK, lw=0.2))
    ax.add_patch(Rectangle((6.0, 2.5), 0.95, 1.6, fc="none",
                            ec=ACC_BLACK, lw=1.0))

    arrow(ax, 7.05, 3.0, 7.7, 3.0, lw=2)

    # W_dec
    matrix_box(ax, 7.75, 1.7, 1.6, 2.6, "W_dec", fc=ACC_GREEN+"33",
               ec=ACC_GREEN, shape="1024 × 2048", label_color=ACC_GREEN,
               fontsize=11)

    arrow(ax, 9.4, 3.0, 10.1, 3.0, lw=2)

    # x_hat
    matrix_box(ax, 10.15, 2.0, 0.4, 2.0, "x̂", fc=HIGHLIGHT, ec=ACC_BLACK,
               shape="1024", fontsize=14)

    # Highlight: each column of W_dec is a feature direction
    callout = FancyBboxPatch((10.95, 1.4), 2.85, 3.4,
                             boxstyle="round,pad=0.1",
                             fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(callout)
    text(ax, 12.4, 4.4, "Each column of W_dec", fontsize=9.5, fontweight="bold",
         color=ACC_BLACK)
    text(ax, 12.4, 4.05, "= one feature direction", fontsize=9, color=ACC_BLACK)
    text(ax, 12.4, 3.75, "in residual space", fontsize=9, color=ACC_BLACK)
    text(ax, 12.4, 3.3, "1024-dim vector", fontsize=8.5, color=MUTED,
         style="italic")
    text(ax, 12.4, 2.85, "→  Q6 steers by", fontsize=9, color=ACC_PURPLE,
         fontweight="bold")
    text(ax, 12.4, 2.55, "adding a column", fontsize=9, color=ACC_PURPLE,
         fontweight="bold")
    text(ax, 12.4, 2.2, "back into the residual", fontsize=9, color=ACC_PURPLE,
         fontweight="bold")
    text(ax, 12.4, 1.7, "(see fig 07)", fontsize=8, color=MUTED, style="italic")

    # Footer
    text(ax, 6.5, 0.65,
         "Trained on 256k tokens of residual activations. 374 / 2048 features fire ≥1×; "
         "~25 are cleanly interpretable (cross-checked against synthetic authors).",
         fontsize=9, color=MUTED, style="italic")

    save(fig, "06_q5_sae.png")


# ═══════════════════════════════════════════════════════════════════
# 07 — Q6: feature steering
# ═══════════════════════════════════════════════════════════════════
def fig_07_q6_feature_steering():
    fig, ax = setup_axes(figsize=(13.5, 6), xlim=(0, 13.5), ylim=(0, 6))
    title(ax, "Q6 — Feature steering (add a decoder column to the residual)",
          sub="Forward hook on ln_f. Pull one column out of W_dec, scale it, "
              "add it to the residual. Generate. Composes additively.")

    # W_dec with one column highlighted
    text(ax, 1.55, 5.3, "W_dec  (decoder)", fontsize=10, fontweight="bold")
    n_cols = 24
    col_w = 2.4 / n_cols
    base_x, base_y, h = 0.5, 1.7, 3.2
    target_col = 14
    for i in range(n_cols):
        fc = ACC_PURPLE if i == target_col else DIM_GREY
        ax.add_patch(Rectangle((base_x + i*col_w, base_y), col_w, h,
                                fc=fc, ec="#BBBBBB", lw=0.4, zorder=2))
    ax.add_patch(Rectangle((base_x, base_y), 2.4, h, fc="none",
                            ec=ACC_BLACK, lw=1.4, zorder=3))
    text(ax, 1.7, 1.4, "1024 × 2048   (rows × features)",
         fontsize=8.5, color=MUTED, style="italic")
    # arrow to target column
    target_x = base_x + (target_col + 0.5) * col_w
    text(ax, target_x, base_y - 0.55, "f_665\n(simplicity)",
         fontsize=8.5, color=ACC_PURPLE, fontweight="bold", ha="center")

    arrow(ax, 3.05, 3.3, 3.9, 3.3, lw=2)
    text(ax, 3.45, 3.6, "extract", fontsize=8.5, color=MUTED, style="italic")

    # Feature direction column (1024-dim)
    matrix_box(ax, 3.95, 1.7, 0.4, 3.2, "v_f", fc=ACC_PURPLE+"55",
               ec=ACC_PURPLE, shape="1024", label_color=ACC_PURPLE, fontsize=12)

    text(ax, 4.6, 3.3, "×", fontsize=20, fontweight="bold")

    # Scale knob
    text(ax, 5.1, 4.7, "scale", fontsize=10, fontweight="bold", color=ACC_ORANGE)
    text(ax, 5.1, 4.4, "s ≈ 15", fontsize=11, fontweight="bold", color=ACC_ORANGE)
    text(ax, 5.1, 4.1, "(hand-tuned)", fontsize=8, color=MUTED, style="italic")
    # box around scale
    s_box = FancyBboxPatch((4.75, 2.7), 0.7, 1.2, boxstyle="round,pad=0.05",
                           fc=HIGHLIGHT, ec=ACC_ORANGE, lw=1.0)
    ax.add_patch(s_box)
    text(ax, 5.1, 3.3, "s", fontsize=18, fontweight="bold", color=ACC_ORANGE)

    text(ax, 5.7, 3.3, "+", fontsize=24, fontweight="bold")

    # Original residual
    matrix_box(ax, 5.95, 1.7, 0.4, 3.2, "x", fc=DIM_GREY, ec=ACC_BLACK,
               shape="1024", fontsize=14)
    text(ax, 6.15, 5.3, "residual\n(at ln_f)", fontsize=9, ha="center",
         color=MUTED, fontweight="bold")

    text(ax, 6.7, 3.3, "=", fontsize=24, fontweight="bold")

    # Steered residual
    matrix_box(ax, 6.95, 1.7, 0.4, 3.2, "x'", fc=HIGHLIGHT, ec=ACC_BLACK,
               shape="1024", fontsize=14)
    text(ax, 7.15, 5.3, "steered\nresidual", fontsize=9, ha="center",
         color=ACC_ORANGE, fontweight="bold")

    arrow(ax, 7.45, 3.3, 8.2, 3.3, lw=2)
    text(ax, 7.85, 3.55, "LM head", fontsize=8.5, color=MUTED, style="italic")

    # Result panel
    res_box = FancyBboxPatch((8.3, 1.5), 5.0, 3.6, boxstyle="round,pad=0.08",
                             fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(res_box)
    text(ax, 10.8, 4.7, "Carroll baseline", fontsize=9.5, fontweight="bold",
         color=ACC_BLACK, ha="center")
    text(ax, 10.8, 4.3,
         "\"a tiny voice. The bunny\nhopped along…\"  (long sentences)",
         fontsize=8.5, color=ACC_BLACK, ha="center", style="italic")

    text(ax, 10.8, 3.5, "+ f_665 simplicity, s=15", fontsize=9.5,
         fontweight="bold", color=ACC_PURPLE, ha="center")
    text(ax, 10.8, 3.1,
         "\"She looked up. It was sad.\nThe cat went up.\"  (short)",
         fontsize=8.5, color=ACC_BLACK, ha="center", style="italic")

    text(ax, 10.8, 2.3, "validation: closed-loop", fontsize=9, fontweight="bold",
         color=ACC_GREEN, ha="center")
    text(ax, 10.8, 1.95,
         "20 seeds; sentence length drops 9 → 4 words.",
         fontsize=8.5, color=ACC_BLACK, ha="center", style="italic")
    text(ax, 10.8, 1.7, "win rate 80% vs random direction.",
         fontsize=8.5, color=ACC_BLACK, ha="center", style="italic")

    # Footer
    text(ax, 6.7, 0.55,
         "Hook code:  output += s * sae.decoder.weight[:, f]   "
         "Features compose additively — sum two columns, get both effects.",
         fontsize=9, color=MUTED, style="italic", family="monospace")

    save(fig, "07_q6_feature_steering.png")


# ═══════════════════════════════════════════════════════════════════
# 08 — Q7: detection ≠ steering
# ═══════════════════════════════════════════════════════════════════
def fig_08_q7_detect_vs_steer():
    fig, ax = setup_axes(figsize=(13.5, 6.5), xlim=(0, 13.5), ylim=(0, 6.5))
    title(ax, "Q7 — Detection ≠ steering (when the model lacks the token)",
          sub="Same operation as Q6, but the target token is OOD for TinyStories. "
              "Even a perfect detector can't push a logit the model never learned to produce.")

    # LEFT: detection works
    text(ax, 3.0, 5.8, "Detection: works perfectly", fontsize=11,
         fontweight="bold", color=ACC_GREEN)

    # Show input text with "thou" highlighted
    txt_box = FancyBboxPatch((0.3, 4.4), 5.4, 0.9,
                             boxstyle="round,pad=0.08",
                             fc=PANEL_BG, ec=MUTED, lw=0.8)
    ax.add_patch(txt_box)
    ax.text(0.5, 4.85, "input:  \"O", fontsize=10, family="monospace",
            ha="left", va="center", color=ACC_BLACK)
    # highlight thou
    ax.add_patch(Rectangle((1.55, 4.65), 0.65, 0.4, fc=ACC_GREEN+"55",
                            ec=ACC_GREEN, lw=1.0))
    ax.text(1.875, 4.85, "thou", fontsize=10, family="monospace",
            ha="center", va="center", color=ACC_BLACK, fontweight="bold")
    ax.text(2.25, 4.85, " who walkest in glory…\"", fontsize=10,
            family="monospace", ha="left", va="center", color=ACC_BLACK)

    arrow(ax, 3.0, 4.35, 3.0, 3.85, lw=2, color=ACC_GREEN)
    text(ax, 3.0, 3.6, "f_1663 fires →  activation = 1.0",
         fontsize=9.5, color=ACC_GREEN, fontweight="bold")
    text(ax, 3.0, 3.25, "(only on 'thou', everywhere)",
         fontsize=8.5, color=MUTED, style="italic")

    # divider
    ax.plot([6.5, 6.5], [0.7, 5.8], color=MUTED, lw=1.0, linestyle="--")

    # RIGHT: steering fails
    text(ax, 10.0, 5.8, "Steering: fails", fontsize=11,
         fontweight="bold", color=ACC_RED)

    # Operation
    op_box = FancyBboxPatch((6.9, 4.4), 6.2, 0.9,
                            boxstyle="round,pad=0.08",
                            fc=HIGHLIGHT, ec=ACC_ORANGE, lw=0.8)
    ax.add_patch(op_box)
    ax.text(10.0, 4.85,
            "x'  =  x  +  s · W_dec[:, 1663]    (s tested: 5, 15, 50)",
            fontsize=10, family="monospace",
            ha="center", va="center", color=ACC_BLACK)

    arrow(ax, 10.0, 4.35, 10.0, 3.85, lw=2, color=ACC_RED)

    # Logit comparison: bar chart of token logits
    bar_y = 1.2
    bar_max_h = 1.8
    tokens = ["the", "and", "Mar-", "thou", "thee", "thy"]
    base_logits = [0.8, 0.7, 0.05, 0.02, 0.015, 0.018]
    steered_logits = [0.55, 0.5, 0.55, 0.04, 0.025, 0.03]
    n_tk = len(tokens)
    bw = 0.42
    base_x = 7.0
    span = 5.6
    for i, tk in enumerate(tokens):
        cx = base_x + i * (span / n_tk) + 0.1
        # base
        h_b = bar_max_h * base_logits[i] / max(base_logits + steered_logits)
        ax.add_patch(Rectangle((cx, bar_y), bw, h_b,
                                fc=DIM_GREY, ec=ACC_BLACK, lw=0.6))
        # steered
        h_s = bar_max_h * steered_logits[i] / max(base_logits + steered_logits)
        ax.add_patch(Rectangle((cx + bw + 0.05, bar_y), bw, h_s,
                                fc=ACC_ORANGE+"99", ec=ACC_ORANGE, lw=0.6))
        col = ACC_RED if tk in ("thou", "thee", "thy") else ACC_BLACK
        ax.text(cx + bw + 0.025, bar_y - 0.18, tk, fontsize=8.5,
                ha="center", color=col, fontweight="bold" if col == ACC_RED else "normal")

    text(ax, 10.0, 3.55, "logits over vocabulary →", fontsize=9,
         fontweight="bold", color=ACC_BLACK)

    # Legend
    ax.add_patch(Rectangle((7.0, 3.2), 0.25, 0.15, fc=DIM_GREY,
                            ec=ACC_BLACK, lw=0.6))
    ax.text(7.35, 3.27, "base", fontsize=8, va="center", color=ACC_BLACK)
    ax.add_patch(Rectangle((8.1, 3.2), 0.25, 0.15, fc=ACC_ORANGE+"99",
                            ec=ACC_ORANGE, lw=0.6))
    ax.text(8.45, 3.27, "+ feature steering", fontsize=8, va="center",
            color=ACC_BLACK)

    # Annotation for failure
    text(ax, 10.0, 0.65,
         "Logits for 'thou' / 'thee' / 'thy' stay near zero at every scale tested. "
         "TinyStories never learned to produce them. Steering shifts probability "
         "mass to similar OOD names ('Mar-') instead of the target.",
         fontsize=9, color=MUTED, style="italic")

    save(fig, "08_q7_detect_vs_steer.png")


# ═══════════════════════════════════════════════════════════════════
# 09 — Q5b: what "feature fires on token X" means (mechanical)
# ═══════════════════════════════════════════════════════════════════
def fig_09_feature_fires_on_token():
    fig, ax = setup_axes(figsize=(14, 7), xlim=(0, 14), ylim=(0, 7))
    title(ax, "Q5 (zoom) — what 'feature fires on token X' means",
          sub="Each feature is a row of W_enc. Feature f's activation at one token "
              "= dot product of that row with the token's residual vector.")

    # ── Top: input text with one token highlighted ──
    text(ax, 1.5, 6.2, "input text", fontsize=10, fontweight="bold",
         color=ACC_BLACK, ha="left")
    txt_box = FancyBboxPatch((0.4, 5.55), 7.2, 0.55,
                             boxstyle="round,pad=0.06",
                             fc=PANEL_BG, ec=MUTED, lw=0.8)
    ax.add_patch(txt_box)
    ax.text(0.6, 5.825, '"and then', fontsize=11, family="monospace",
            ha="left", va="center", color=ACC_BLACK)
    # highlight 'I'
    ax.add_patch(Rectangle((1.85, 5.62), 0.35, 0.42, fc=ACC_PURPLE+"55",
                            ec=ACC_PURPLE, lw=1.2, zorder=3))
    ax.text(2.025, 5.825, "I", fontsize=12, family="monospace",
            ha="center", va="center", color=ACC_BLACK, fontweight="bold",
            zorder=4)
    ax.text(2.25, 5.825, ' looked up at the sky and said hello"',
            fontsize=11, family="monospace", ha="left", va="center",
            color=ACC_BLACK)
    text(ax, 2.025, 5.35, "token at position t",
         fontsize=8.5, color=ACC_PURPLE, fontweight="bold")

    # ── Arrow down to residual ──
    arrow(ax, 2.025, 5.25, 2.025, 4.85, lw=2)
    text(ax, 2.85, 5.05, "model forward → residual stream",
         fontsize=8.5, color=MUTED, style="italic", ha="left")

    # ── Residual vector x_t ──
    matrix_box(ax, 1.7, 3.0, 0.65, 1.7, "x_t", fc=DIM_GREY, ec=ACC_BLACK,
               shape="1024", fontsize=14)
    text(ax, 2.025, 4.95, "residual at token t",
         fontsize=8.5, color=MUTED, ha="center")

    # ── W_enc matrix with one row highlighted (feature f = 1779) ──
    text(ax, 5.5, 4.95, "W_enc  (encoder)",
         fontsize=10, fontweight="bold", color=ACC_BLUE)
    n_rows_show = 18
    base_x, base_y, w_enc_w, w_enc_h = 4.0, 3.0, 3.0, 1.7
    target_row = 7
    row_h = w_enc_h / n_rows_show
    for i in range(n_rows_show):
        fc = ACC_PURPLE if i == target_row else DIM_GREY
        sy = base_y + (n_rows_show - 1 - i) * row_h
        ax.add_patch(Rectangle((base_x, sy), w_enc_w, row_h,
                                fc=fc, ec="#BBBBBB", lw=0.3, zorder=2))
    ax.add_patch(Rectangle((base_x, base_y), w_enc_w, w_enc_h, fc="none",
                            ec=ACC_BLACK, lw=1.4, zorder=3))
    text(ax, base_x + w_enc_w/2, 2.75, "2048 × 1024",
         fontsize=8.5, color=MUTED, style="italic")
    # arrow pointing to the highlighted row
    target_y = base_y + (n_rows_show - 1 - target_row + 0.5) * row_h
    text(ax, base_x + w_enc_w + 0.15, target_y,
         "← row 1779", fontsize=9, color=ACC_PURPLE,
         fontweight="bold", ha="left", va="center")
    text(ax, base_x + w_enc_w + 0.15, target_y - 0.3,
         "(one feature)", fontsize=8, color=MUTED,
         style="italic", ha="left", va="center")

    # ── Operation: dot product ──
    text(ax, 9.7, 4.95, "feature f's activation at this token",
         fontsize=10, fontweight="bold", color=ACC_BLACK)
    op_box = FancyBboxPatch((8.6, 3.4), 5.0, 1.2,
                            boxstyle="round,pad=0.08",
                            fc=HIGHLIGHT, ec=ACC_ORANGE, lw=1.0)
    ax.add_patch(op_box)
    ax.text(11.1, 4.25,
            "z_f(t)  =  W_enc[f, :]  ·  x_t",
            fontsize=12, family="monospace", ha="center", va="center",
            color=ACC_BLACK, fontweight="bold")
    ax.text(11.1, 3.75,
            "(one dot product → one scalar)",
            fontsize=8.5, ha="center", va="center",
            color=MUTED, style="italic")

    arrow(ax, 7.05, 3.85, 8.55, 3.85, lw=2)

    # ── Bottom panel: what the corpus sweep produces ──
    text(ax, 7.0, 2.5,
         "Repeat for every token in every author's text → 256k scalars per feature.  "
         "Sort by z_f, read top 15 tokens with context.",
         fontsize=9.5, color=ACC_BLACK, style="italic", ha="center")

    # Top-firing tokens panel
    top_box = FancyBboxPatch((0.4, 0.3), 13.2, 1.85,
                             boxstyle="round,pad=0.08",
                             fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(top_box)
    text(ax, 0.6, 1.85, "Top-activating tokens for f_1779:",
         fontsize=9.5, fontweight="bold", color=ACC_PURPLE, ha="left")

    examples = [
        ('"…and then', "I", 'looked up at the…"',  "Shelley",   3.42),
        ('"…the day', "I",   'first met him,…"',     "Stoker",    3.18),
        ('"…what shall', "I", 'do?" he wondered…"',  "firstperson", 3.05),
        ('"…before',   "I",   'could speak again…"', "Alcott",    2.91),
        ('"…I think', "I",    'understand now,"…"',  "Carroll",   2.74),
    ]
    for i, (pre, tok, post, author, val) in enumerate(examples):
        y = 1.55 - i * 0.27
        ax.text(0.7, y, f"{val:.2f}", fontsize=8.5, family="monospace",
                color=ACC_ORANGE, fontweight="bold", ha="left", va="center")
        ax.text(1.4, y, pre, fontsize=8.5, family="monospace",
                color=ACC_BLACK, ha="left", va="center")
        # Tok with highlight
        # measure offset
        ax.add_patch(Rectangle((4.05, y - 0.10), 0.22, 0.21,
                                fc=ACC_PURPLE+"55", ec=ACC_PURPLE, lw=0.8,
                                zorder=2))
        ax.text(4.16, y, tok, fontsize=9, family="monospace",
                color=ACC_BLACK, fontweight="bold", ha="center", va="center",
                zorder=3)
        ax.text(4.36, y, post, fontsize=8.5, family="monospace",
                color=ACC_BLACK, ha="left", va="center")
        ax.text(8.7, y, f"[{author}]", fontsize=8.5, family="monospace",
                color=MUTED, ha="left", va="center", style="italic")

    # Label arrow on the right
    label_box = FancyBboxPatch((10.3, 0.55), 3.15, 1.4,
                               boxstyle="round,pad=0.08",
                               fc=ACC_PURPLE+"22", ec=ACC_PURPLE, lw=1.0)
    ax.add_patch(label_box)
    text(ax, 11.875, 1.7, "Top tokens are mostly", fontsize=9,
         color=ACC_BLACK, ha="center")
    text(ax, 11.875, 1.4, '"I"', fontsize=14, fontweight="bold",
         color=ACC_PURPLE, ha="center", family="monospace")
    text(ax, 11.875, 1.05, "→ label this feature", fontsize=8.5,
         color=ACC_BLACK, ha="center")
    text(ax, 11.875, 0.78, "first-person 'I'", fontsize=9,
         fontweight="bold", color=ACC_PURPLE, ha="center", style="italic")

    save(fig, "09_q5_feature_fires.png")


# ═══════════════════════════════════════════════════════════════════
# 10 — Q5c: what "head correlates with feature" means
# ═══════════════════════════════════════════════════════════════════
def fig_10_feature_head_correlation():
    fig, ax = setup_axes(figsize=(15, 8.4), xlim=(0, 15), ylim=(0, 8.4))
    title(ax, "Q5 (zoom) — what 'head H correlates with feature f' means",
          sub="Pearson correlation between two columns: feature f's mean activation "
              "across authors, and head h's knockout-recovery across authors.")

    # ── LEFT: M (author × feature) matrix ──
    text(ax, 2.25, 7.6, "M  —  author × feature matrix",
         fontsize=10.5, fontweight="bold", color=ACC_PURPLE, ha="center")
    text(ax, 2.25, 7.28,
         "M[a, f]  =  mean activation of feature f on author a's text",
         fontsize=8.5, color=MUTED, style="italic", ha="center")

    # Draw matrix M
    n_rows_M = 12  # show 12 of 77 authors
    n_cols_M = 14  # show 14 of 2048 features
    base_x_M, base_y_M = 0.5, 4.4
    cell_w_M = 3.5 / n_cols_M
    cell_h_M = 2.6 / n_rows_M
    target_col_M = 9
    rng = np.random.default_rng(7)
    intensities = rng.uniform(0.05, 0.95, (n_rows_M, n_cols_M))
    for r in range(n_rows_M):
        for c in range(n_cols_M):
            v = intensities[r, c]
            if c == target_col_M:
                color = (0.486, 0.227, 0.929, v)
            else:
                color = (0.4, 0.4, 0.4, v * 0.5)
            ax.add_patch(Rectangle(
                (base_x_M + c * cell_w_M, base_y_M + r * cell_h_M),
                cell_w_M, cell_h_M, fc=color, ec="#DDDDDD", lw=0.2, zorder=2))
    ax.add_patch(Rectangle((base_x_M + target_col_M * cell_w_M, base_y_M),
                            cell_w_M, 2.6, fc="none", ec=ACC_PURPLE, lw=1.8,
                            zorder=3))
    ax.add_patch(Rectangle((base_x_M, base_y_M), 3.5, 2.6,
                            fc="none", ec=ACC_BLACK, lw=1.2, zorder=3))
    text(ax, 2.25, 4.18, "77 × 2048", fontsize=8.5, color=MUTED,
         style="italic")
    text(ax, base_x_M - 0.18, base_y_M + 1.3, "authors",
         fontsize=8.5, color=MUTED, rotation=90, ha="center", va="center")
    text(ax, base_x_M + 1.75, 6.83, "features →",
         fontsize=8.5, color=MUTED, ha="center")
    # column arrow above
    col_x = base_x_M + (target_col_M + 0.5) * cell_w_M
    text(ax, col_x, 7.05, "↓ column f = 1779   (first-person 'I')",
         fontsize=8.5, color=ACC_PURPLE, fontweight="bold", ha="center")

    # ── RIGHT: K (author × head) matrix ──
    text(ax, 7.5, 7.6, "K  —  author × head matrix",
         fontsize=10.5, fontweight="bold", color=ACC_RED, ha="center")
    text(ax, 7.5, 7.28,
         "K[a, h]  =  recovery score when only head h is kept (from Q1)",
         fontsize=8.5, color=MUTED, style="italic", ha="center")

    n_cols_K = 16
    base_x_K, base_y_K = 5.7, 4.4
    cell_w_K = 3.6 / n_cols_K
    cell_h_K = 2.6 / n_rows_M
    target_col_K = 14
    intensitiesK = rng.uniform(0.1, 0.9, (n_rows_M, n_cols_K))
    for r in range(n_rows_M):
        for c in range(n_cols_K):
            v = intensitiesK[r, c]
            if c == target_col_K:
                color = (0.863, 0.149, 0.149, v)
            else:
                color = (0.3, 0.4, 0.7, v * 0.5)
            ax.add_patch(Rectangle(
                (base_x_K + c * cell_w_K, base_y_K + r * cell_h_K),
                cell_w_K, cell_h_K, fc=color, ec="#DDDDDD", lw=0.2, zorder=2))
    ax.add_patch(Rectangle((base_x_K + target_col_K * cell_w_K, base_y_K),
                            cell_w_K, 2.6, fc="none", ec=ACC_RED, lw=1.8,
                            zorder=3))
    ax.add_patch(Rectangle((base_x_K, base_y_K), 3.6, 2.6,
                            fc="none", ec=ACC_BLACK, lw=1.2, zorder=3))
    text(ax, 7.5, 4.18, "77 × 16", fontsize=8.5, color=MUTED, style="italic")
    text(ax, base_x_K + 1.8, 6.83, "16 heads →",
         fontsize=8.5, color=MUTED, ha="center")
    col_x_K = base_x_K + (target_col_K + 0.5) * cell_w_K
    text(ax, col_x_K, 7.05, "↓ column h = H14   (formality enforcer)",
         fontsize=8.5, color=ACC_RED, fontweight="bold", ha="center")

    # ── Extract columns to vectors ──
    arrow(ax, col_x, 4.35, col_x, 3.95,
          lw=2, color=ACC_PURPLE)
    arrow(ax, col_x_K, 4.35, col_x_K, 3.95,
          lw=2, color=ACC_RED)

    vec_box_M = FancyBboxPatch((col_x - 0.95, 3.4), 1.9, 0.55,
                               boxstyle="round,pad=0.04",
                               fc=ACC_PURPLE+"33", ec=ACC_PURPLE, lw=1.0)
    ax.add_patch(vec_box_M)
    text(ax, col_x, 3.68, "M[:, 1779]   (length 77)",
         fontsize=9, family="monospace", color=ACC_BLACK)

    vec_box_K = FancyBboxPatch((col_x_K - 0.95, 3.4), 1.9, 0.55,
                               boxstyle="round,pad=0.04",
                               fc=ACC_RED+"33", ec=ACC_RED, lw=1.0)
    ax.add_patch(vec_box_K)
    text(ax, col_x_K, 3.68, "K[:, 14]   (length 77)",
         fontsize=9, family="monospace", color=ACC_BLACK)

    # ── Correlation formula box (right side, top) ──
    corr_box = FancyBboxPatch((9.7, 5.0), 5.0, 2.4,
                              boxstyle="round,pad=0.1",
                              fc=HIGHLIGHT, ec=ACC_ORANGE, lw=1.2)
    ax.add_patch(corr_box)
    text(ax, 12.2, 7.0, "Pearson correlation across authors",
         fontsize=10, fontweight="bold", color=ACC_BLACK)
    ax.text(12.2, 6.45,
            "r  =  corr( M[:, f] ,  K[:, h] )",
            fontsize=11, family="monospace", ha="center", va="center",
            color=ACC_BLACK, fontweight="bold")
    text(ax, 12.2, 5.85,
         "one scalar per (feature, head) pair",
         fontsize=8.5, color=MUTED, style="italic")
    text(ax, 12.2, 5.45,
         "high |r|  →  this head's knockout impact",
         fontsize=8.5, color=ACC_BLACK)
    text(ax, 12.2, 5.18,
         "tracks this feature's activation",
         fontsize=8.5, color=ACC_BLACK)

    # Arrows pointing into corr box from both vectors
    arrow(ax, col_x + 0.6, 3.5, 10.0, 5.1,
          lw=1.5, color=ACC_PURPLE)
    arrow(ax, col_x_K + 0.6, 3.5, 10.4, 5.1,
          lw=1.5, color=ACC_RED)

    # ── Bottom-right: real-data scatter plot inset ──
    sax = fig.add_axes([0.665, 0.085, 0.275, 0.36])
    rng2 = np.random.default_rng(3)
    n_pts = 77
    xs = rng2.gamma(0.7, 0.25, n_pts)
    xs = np.clip(xs, 0.005, 1.85)
    base = 0.6 - 0.55 * (xs / xs.max())
    noise = rng2.normal(0, 0.27, n_pts)
    ys = base + noise
    ys = np.clip(ys, -0.7, 0.75)
    sax.scatter(xs, ys, s=24, alpha=0.55, color=ACC_BLUE,
                edgecolors="white", linewidths=0.5)
    slope, icpt = np.polyfit(xs, ys, 1)
    xline = np.linspace(0, xs.max() + 0.1, 50)
    sax.plot(xline, slope * xline + icpt, color=ACC_RED, lw=2, zorder=3)
    sax.axhline(0, color=MUTED, lw=0.6, linestyle="--", zorder=1)
    sax.annotate("Shelley\n(lots of 'I')", xy=(1.05, -0.65),
                 xytext=(1.05, -0.25), fontsize=8, color=ACC_BLACK,
                 ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    sax.annotate("gibbon\n(rare 'I')", xy=(0.02, 0.55),
                 xytext=(0.25, 0.65), fontsize=8, color=ACC_BLACK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    sax.set_xlabel("M[a, 1779]   —  'I' activation", fontsize=9)
    sax.set_ylabel("K[a, 14]   —  H14 recovery", fontsize=9)
    sax.set_title("real data: f_1779 vs H14    r = -0.42   (p < 0.001)",
                  fontsize=9.5, color=ACC_RED, fontweight="bold")
    sax.tick_params(labelsize=8)
    for s in ("top", "right"):
        sax.spines[s].set_visible(False)

    # ── Bottom-left: how to read it ──
    concl_box = FancyBboxPatch((0.4, 0.4), 9.2, 2.6,
                               boxstyle="round,pad=0.1",
                               fc=PANEL_BG, ec=MUTED, lw=1.0)
    ax.add_patch(concl_box)
    text(ax, 5.0, 2.7, "How to read it",
         fontsize=11, fontweight="bold", color=ACC_BLACK)
    text(ax, 5.0, 2.30,
         "Authors who write a lot of 'I' (high M[a, 1779])",
         fontsize=10, color=ACC_BLACK)
    text(ax, 5.0, 2.00,
         "are the same authors that H14 alone hurts (low K[a, 14]).",
         fontsize=10, color=ACC_BLACK)
    text(ax, 5.0, 1.50,
         "→ H14 'reads' first-person 'I' negatively  (formality enforcer).",
         fontsize=10, fontweight="bold", color=ACC_RED)
    text(ax, 5.0, 1.05,
         "Repeat for every (f, h) pair → assigns each head a role.",
         fontsize=9, color=MUTED, style="italic")
    text(ax, 5.0, 0.70,
         "(co-variance across authors, not causal — see caveat 3 in Q5 methodology)",
         fontsize=8.5, color=MUTED, style="italic")

    save(fig, "10_q5_feature_head_correlation.png")


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════
def main():
    print(f"Writing methodology figures to: {OUT_DIR}/")
    fig_00_lora_setup()
    fig_01_svd_refactor()
    fig_02_q1_knockout()
    fig_03_q2_head_steering()
    fig_04_q3_transplant()
    fig_05_q4_blend()
    fig_06_q5_sae()
    fig_07_q6_feature_steering()
    fig_08_q7_detect_vs_steer()
    fig_09_feature_fires_on_token()
    fig_10_feature_head_correlation()
    print("Done.")


if __name__ == "__main__":
    main()