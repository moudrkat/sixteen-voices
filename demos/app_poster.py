#!/usr/bin/env python3
"""Streamlit app: Poster companion — SAE Feature Steering.

One slider. Pick an author, drag simplicity, see the text change.

Usage:
    streamlit run demos/app_poster.py
"""

import json
import sys
from pathlib import Path

# Ensure the package is importable when running from Streamlit Cloud
# (where `pip install -e .` may not have run)
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import streamlit as st
import torch

from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_base_model as _load_base,
    load_tokenizer as _load_tokenizer,
)
from sixteen_voices.sae import SparseAutoencoder

ADAPTERS_DIR = Path("outputs/authors")
SAE_DIR = Path("outputs/sae_topk16_2048")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")

PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "The door opened slowly",
]

# Simplicity feature — SAE decoder column 665
SIMPLICITY_ID = 665

AUTHORS = {
    "(base model)": "Raw TinyStories — no author style.",
    "poe": "Edgar Allan Poe — trained on The Raven and Other Poems + The Works of E.A. Poe. Dark, atmospheric, ornate.",
    "carroll": "Lewis Carroll — trained on Alice's Adventures in Wonderland, Through the Looking-Glass, Sylvie and Bruno. Playful, absurdist, Victorian.",
    "grimm": "Brothers Grimm — trained on Grimm's Fairy Tales + Household Stories. Folk structure, fairy tale voice.",
}

# ── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_model(author):
    if author == "(base model)":
        return _load_base()
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_sae():
    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    return SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)


@st.cache_resource
def load_knockout_data():
    with open(KNOCKOUT_PATH) as f:
        return json.load(f)


# ── Core functions ──────────────────────────────────────────────────

def build_steering_vector(sae, scale):
    """Build simplicity steering vector from SAE decoder column."""
    w = sae.decoder.weight.detach()
    return scale * w[:, SIMPLICITY_ID]


def generate(model, tokenizer, prompt, max_new=70, seed=42,
             feature_vec=None):
    """Generate text, optionally with feature steering."""
    torch.manual_seed(seed)

    hooks = []
    if feature_vec is not None and feature_vec.abs().max() > 0.01:
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (output[0] + feature_vec,) + output[1:]
            return output + feature_vec
        h = model.transformer.ln_f.register_forward_hook(hook_fn)
        hooks.append(h)

    try:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        plen = ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new, temperature=0.7,
                do_sample=True, top_k=50, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return text


# ── Capture all intermediate activations ───────────────────────────

def _clear_all_hooks(model):
    """Remove any lingering hooks from the model (safety net for crashes)."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def capture_all_activations(model, tokenizer, prompt):
    """Hook every stage and return real activations for the last token.

    Returns dict with keys:
        post_embed, post_attn, post_mlp, post_ln_f
    Each is a 1-D tensor of 1024 numbers (last token position).
    """
    _clear_all_hooks(model)

    acts = {}
    hooks = []

    # 1. After embedding = input to the first block
    def _hook_embed(module, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states", args[0])
        acts["post_embed"] = t[0, -1, :].detach().clone()
    hooks.append(
        model.transformer.h[0].register_forward_pre_hook(_hook_embed,
                                                          with_kwargs=True))

    # 2. After attention = input to ln_2 (residual after attn add)
    def _hook_attn(module, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states", args[0])
        acts["post_attn"] = t[0, -1, :].detach().clone()
    hooks.append(
        model.transformer.h[0].ln_2.register_forward_pre_hook(_hook_attn,
                                                               with_kwargs=True))

    # 3. After MLP = output of full block (residual after mlp add)
    def _hook_block(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        acts["post_mlp"] = t[0, -1, :].detach().clone()
    hooks.append(model.transformer.h[0].register_forward_hook(_hook_block))

    # 4. After LayerNorm = ln_f output (where steering injects)
    def _hook_lnf(module, input, output):
        t = output if not isinstance(output, tuple) else output[0]
        acts["post_ln_f"] = t[0, -1, :].detach().clone()
    hooks.append(model.transformer.ln_f.register_forward_hook(_hook_lnf))

    try:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in hooks:
            h.remove()

    return acts


# ── Interactive model diagram ──────────────────────────────────────

_C_TEXT = "#2C3E50"
_C_STREAM = "#3498DB"
_C_STREAM_BG = "#EBF5FB"
_C_ATTN = "#9B59B6"
_C_MLP = "#F39C12"
_C_STEER = "#E74C3C"
_C_OUTPUT = "#27AE60"
_C_EMBED = "#1ABC9C"
_C_ARROW = "#555555"


def _rbox(ax, x, y, w, h, label, color, fontsize=15, text_color="white",
          alpha=1.0, lw=2.0, sublabel=None, sublabel_fs=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor="#333333",
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.22, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.22, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold", zorder=4)


def _arrow_v(ax, x, y0, y1, color=_C_ARROW, lw=2.0):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=16))


def _fmt(vec, idxs):
    vals = [f"{vec[i].item():+.2f}" for i in idxs]
    return "[" + ", ".join(vals) + ", ...]"


def _stream_label(ax, cx, y, vec, idxs, color=_C_STREAM, bold=False,
                  bg="white", tag=None):
    """Draw a real-values label at position y. Returns new y below it."""
    txt = _fmt(vec, idxs)
    ax.text(cx, y, txt,
            fontsize=8.5, color=color, ha="center", fontfamily="monospace",
            fontweight="bold" if bold else "normal", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec=color,
                      lw=1.5 if bold else 1.0, alpha=0.9))
    if tag:
        ax.text(cx + 4.2, y, tag,
                fontsize=9, color=color, ha="left", va="center",
                fontweight="bold", style="italic")
    return y - 0.55


def draw_model_with_steering(all_acts, direction, scaled_dir, steered_vec,
                              scale, prompt, text_baseline, text_steered,
                              head_recovery=None, author=None):
    """Draw the full model architecture with ALL real numbers."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 26))
    fig.patch.set_facecolor("white")

    cx = 5.5
    bw = 5.5
    bh = 1.1
    gap = 0.7

    # 6 representative dimensions
    idxs = torch.linspace(0, 1023, 6).long()

    y = 32.0

    # ── Title ──
    ax.text(cx, y + 1.5,
            "TinyStories-1Layer-21M",
            fontsize=20, fontweight="bold", color=_C_TEXT, ha="center")
    ax.text(cx, y + 0.9,
            "all numbers are real — captured from your prompt right now",
            fontsize=11, color="#888888", ha="center", style="italic",
            zorder=10)

    # ── Input tokens (full prompt!) ──
    y -= 0.8
    _rbox(ax, cx - bw / 2, y - bh, bw, bh, f'Input: "{prompt}"',
          "#7F8C8D", fontsize=13)
    y -= bh + gap + 0.5

    # ── Embedding ──
    _arrow_v(ax, cx, y + bh + 0.5, y + bh)
    _rbox(ax, cx - bw / 2, y, bw, bh, "Embedding", _C_EMBED,
          sublabel="each token → 1024 numbers")
    y -= bh + gap + 0.2

    # ── POST-EMBEDDING — real ──
    _arrow_v(ax, cx, y + bh + 0.2, y + 0.4)
    y = _stream_label(ax, cx, y + 0.05, all_acts["post_embed"], idxs,
                      tag="after embedding")

    # ── Residual stream label — positioned between embed and attention ──
    ax.text(cx + bw / 2 + 1.5, y + 1.0,
            "RESIDUAL STREAM",
            fontsize=13, color=_C_STREAM, ha="left", va="top",
            fontweight="bold")
    ax.text(cx + bw / 2 + 1.5, y + 0.4,
            "1024 numbers flowing down\nvalues shown for last\ntoken position",
            fontsize=9, color=_C_STREAM, ha="left", va="top",
            linespacing=1.4)

    # ── Residual stream background stripe ──
    stripe_top = y + 1.5
    stripe_bot = y - 13.5
    stripe = FancyBboxPatch((cx - 0.6, stripe_bot), 1.2,
                            stripe_top - stripe_bot,
                            boxstyle="round,pad=0.08",
                            facecolor=_C_STREAM_BG, edgecolor=_C_STREAM,
                            linewidth=1.5, alpha=0.25, zorder=1,
                            linestyle="--")
    ax.add_patch(stripe)

    # ── Attention ──
    y -= 0.4
    _arrow_v(ax, cx, y + 0.3, y - 0.15)
    y -= 0.25

    if head_recovery is not None:
        # Draw 16 individual head boxes, colored by recovery score
        attn_w = bw + 1.2
        n_cols = 8
        n_rows = 2
        pad = 0.12
        cell_w = (attn_w - pad * (n_cols + 1)) / n_cols
        cell_h = 0.42
        block_h = n_rows * cell_h + (n_rows + 1) * pad + 0.5
        block_top = y
        block_bot = y - block_h

        # Outer box
        outer = FancyBboxPatch(
            (cx - attn_w / 2, block_bot), attn_w, block_h,
            boxstyle="round,pad=0.12", facecolor="#F3E5F5",
            edgecolor="#333333", linewidth=2.0, alpha=0.6, zorder=2)
        ax.add_patch(outer)

        ax.text(cx, block_top - 0.15,
                "16 Attention Heads", fontsize=13, color=_C_ATTN,
                ha="center", va="top", fontweight="bold", zorder=4)
        ax.text(cx, block_top - 0.48,
                "read stream → compute → add result back",
                fontsize=10, color=_C_ATTN, ha="center", va="top",
                style="italic", zorder=4, alpha=0.85)

        recoveries = [head_recovery.get(f"H{i}", 0.0) for i in range(16)]
        max_rec = max(max(recoveries), 0.01)

        for i in range(16):
            row = i // n_cols
            col = i % n_cols
            hx = (cx - attn_w / 2) + pad + col * (cell_w + pad)
            hy = block_top - 0.7 - row * (cell_h + pad)

            rec = recoveries[i]
            # Color: bright purple for high recovery, light grey for low/negative
            if rec > 0:
                intensity = min(rec / max_rec, 1.0)
                r = int(155 * (1 - intensity) + 155 * intensity)
                g = int(155 * (1 - intensity) + 89 * intensity)
                b = int(155 * (1 - intensity) + 182 * intensity)
                fc = f"#{r:02x}{g:02x}{b:02x}"
                tc = "white" if intensity > 0.5 else _C_TEXT
            else:
                fc = "#E0E0E0"
                tc = "#999999"

            box = FancyBboxPatch(
                (hx, hy - cell_h), cell_w, cell_h,
                boxstyle="round,pad=0.04", facecolor=fc,
                edgecolor="#666666", linewidth=1.0, zorder=3)
            ax.add_patch(box)

            ax.text(hx + cell_w / 2, hy - cell_h / 2 + 0.06,
                    f"H{i}", fontsize=7.5, color=tc, ha="center",
                    va="center", fontweight="bold", zorder=4)
            ax.text(hx + cell_w / 2, hy - cell_h / 2 - 0.1,
                    f"{rec:.0%}", fontsize=6, color=tc, ha="center",
                    va="center", zorder=4)

        # Legend
        author_label = author if author != "(base model)" else "base"
        ax.text(cx - attn_w / 2 - 0.3, block_bot + block_h / 2,
                f"recovery\nfor {author_label}",
                fontsize=8, color=_C_ATTN, ha="right", va="center",
                style="italic", linespacing=1.4)

        y = block_bot - gap
    else:
        attn_w = bw + 1.2
        _rbox(ax, cx - attn_w / 2, y - bh, attn_w, bh,
              "16 Attention Heads", _C_ATTN, fontsize=14,
              sublabel="read stream → compute → add result back")
        ax.text(cx - attn_w / 2 - 0.3, y - bh / 2,
                "each: 64d", fontsize=9, color=_C_ATTN, ha="right",
                style="italic")
        y -= bh + gap

    # ── POST-ATTENTION — real ──
    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_attn"], idxs,
                      tag="after attention")

    # ── MLP ──
    y -= 0.3
    _arrow_v(ax, cx, y + 0.3, y - 0.15)
    y -= 0.25
    _rbox(ax, cx - bw / 2, y - bh, bw, bh, "MLP", _C_MLP,
          sublabel="1024 → 4096 → 1024")
    ax.text(cx + bw / 2 + 0.3, y - bh / 2,
            "simplicity direction\nemerges here —\nno single head\ncontrols it",
            fontsize=8, color=_C_MLP, ha="left", va="center",
            style="italic", linespacing=1.3)
    y -= bh + gap

    # ── POST-MLP — real ──
    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_mlp"], idxs,
                      tag="after MLP")

    # ── LayerNorm ──
    y -= 0.3
    _arrow_v(ax, cx, y + 0.3, y - 0.15)
    y -= 0.25
    ln_w = bw - 0.5
    _rbox(ax, cx - ln_w / 2, y - bh * 0.7, ln_w, bh * 0.7, "LayerNorm",
          "#95A5A6", fontsize=12)
    y -= bh * 0.7 + gap

    # ── POST-LN_F (before steering) — real ──
    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_ln_f"], idxs,
                      bold=True, tag="before steering")

    # ═══════════════════════════════════════════════════════════════
    # ★ STEERING
    # ═══════════════════════════════════════════════════════════════
    y -= 0.4
    _arrow_v(ax, cx, y + 0.35, y - 0.05, color=_C_STEER, lw=2.5)
    y -= 0.15
    steer_h = bh + 0.3
    steer_w = bw + 1.5
    steer_box = FancyBboxPatch((cx - steer_w / 2, y - steer_h),
                               steer_w, steer_h,
                               boxstyle="round,pad=0.12",
                               facecolor=_C_STEER, edgecolor="#C0392B",
                               linewidth=2.5, alpha=0.95, zorder=3)
    ax.add_patch(steer_box)
    ax.text(cx, y - steer_h / 2 + 0.15,
            "STEERING HAPPENS HERE",
            fontsize=14, color="white", ha="center", va="center",
            fontweight="bold", zorder=4)
    ax.text(cx, y - steer_h / 2 - 0.2,
            "stream = stream + scale x direction",
            fontsize=10, color="white", ha="center", va="center",
            fontfamily="monospace", zorder=4, alpha=0.9)

    # Direction vector coming from left — with real values
    dir_x = cx - steer_w / 2 - 3.5
    dir_y = y - steer_h / 2
    ax.annotate("",
                xy=(cx - steer_w / 2, dir_y),
                xytext=(dir_x + 2.5, dir_y),
                arrowprops=dict(arrowstyle="-|>", color=_C_STEER,
                                lw=2.5, mutation_scale=16))
    ax.text(dir_x + 1.0, dir_y + 0.55,
            "direction vector", fontsize=12, color=_C_STEER,
            ha="center", fontweight="bold")
    ax.text(dir_x + 1.0, dir_y + 0.15,
            f"f665 x {scale:.0f}", fontsize=10, color=_C_STEER,
            ha="center")
    ax.text(dir_x + 1.0, dir_y - 0.3,
            _fmt(scaled_dir, idxs),
            fontsize=7, color=_C_STEER, ha="center",
            fontfamily="monospace")
    y -= steer_h + gap + 0.2

    # ── POST-STEERING — real steered values ──
    _arrow_v(ax, cx, y + 0.3, y + 0.05, color=_C_STEER, lw=2.5)
    y = _stream_label(ax, cx, y - 0.2, steered_vec, idxs,
                      color=_C_STEER, bold=True, bg="#FFEBEE",
                      tag="after steering!")

    # ── Predict next token ──
    y -= 0.4
    _arrow_v(ax, cx, y + 0.35, y - 0.05)
    _rbox(ax, cx - bw / 2, y - bh - 0.1, bw, bh,
          "Predict next token", _C_OUTPUT, fontsize=14)
    y -= bh + gap + 0.8

    # ── Loop arrow: repeat for each generated token ──
    loop_x = cx + bw / 2 + 0.8
    loop_top = y - 0.1   # top of "predict" box
    loop_bot = y - bh - 0.1  # bottom of "predict" box
    # Arrow curving from bottom-right of predict back up to right side
    ax.annotate("",
                xy=(loop_x + 1.5, loop_top + 0.5),
                xytext=(loop_x + 1.5, loop_bot - 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=1.5, mutation_scale=14,
                                connectionstyle="arc3,rad=-0.5"))
    ax.text(loop_x + 2.5, (loop_top + loop_bot) / 2,
            "repeat for\neach token",
            fontsize=10, color="#888888", ha="left", va="center",
            style="italic")

    # ── Output comparison — stacked vertically to avoid overlap ──
    out_w = bw + 1.0
    out_h = 2.0
    out_left = cx - out_w / 2

    def _wrap_text(text, max_chars=45):
        """Truncate and wrap text to fit in output boxes."""
        if len(text) > max_chars * 2:
            text = text[:max_chars * 2] + "..."
        words = text.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) + 1 > max_chars:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            lines.append(line)
        return "\n".join(lines[:3])  # max 3 lines

    # Baseline
    box_base = FancyBboxPatch((out_left, y - out_h), out_w, out_h,
                              boxstyle="round,pad=0.12",
                              facecolor="#E3F2FD", edgecolor=_C_STREAM,
                              linewidth=1.2, zorder=2)
    ax.add_patch(box_base)
    ax.text(out_left + 0.3, y - 0.25,
            "Without steering:", fontsize=11, color=_C_STREAM,
            fontweight="bold", ha="left", va="top")
    ax.text(out_left + 0.3, y - 0.7,
            f'"{_wrap_text(text_baseline)}"',
            fontsize=8.5, color="#444444", ha="left", va="top",
            style="italic", linespacing=1.3)
    y -= out_h + 0.4

    # Steered
    box_steer = FancyBboxPatch((out_left, y - out_h), out_w, out_h,
                               boxstyle="round,pad=0.12",
                               facecolor="#FFEBEE", edgecolor=_C_STEER,
                               linewidth=1.2, zorder=2)
    ax.add_patch(box_steer)
    ax.text(out_left + 0.3, y - 0.25,
            "With steering:", fontsize=11, color=_C_STEER,
            fontweight="bold", ha="left", va="top")
    ax.text(out_left + 0.3, y - 0.7,
            f'"{_wrap_text(text_steered)}"',
            fontsize=8.5, color="#444444", ha="left", va="top",
            style="italic", linespacing=1.3)

    # ── Axes ──
    ax.set_xlim(-3.5, 14.0)
    ax.set_ylim(y - out_h - 0.8, 34.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig


# ── Main app ────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sixteen Voices — Steering Lab",
        page_icon="~",
        layout="centered",
    )

    st.title("Steering a Tiny Language Model")
    st.caption(
        "Pick an author voice. Drag the simplicity knob. "
        "Watch gothic prose turn into kindergarten sentences."
    )

    st.markdown(
        "This is a tiny AI that writes children's stories. "
        "I taught it to imitate 77 different authors — Poe, Carroll, "
        "the Brothers Grimm, and many more — and then opened it up "
        "to understand *how* it captures each voice. "
        "Along the way I found internal \"knobs\" that control "
        "things like simplicity, dialogue, and formality. "
        "This demo lets you turn one of those knobs yourself "
        "and watch the writing change in real time.\n\n"
        "What you see here is a small slice of a larger experiment "
        "- a lot of poking around inside "
        "a one-layer transformer. The full project is "
        "[on GitHub](https://github.com/moudrkat/sixteen-voices)."
    )

    tokenizer = load_tokenizer()
    sae = load_sae()
    knockout_data = load_knockout_data()

    # ── Author ──────────────────────────────────────────────────────
    author = st.selectbox("Author", list(AUTHORS.keys()), index=2)
    st.caption(AUTHORS[author])

    # ── Simplicity slider ───────────────────────────────────────────
    scale = st.slider(
        "Simplicity",
        min_value=0.0, max_value=15.0,
        value=8.0, step=0.5,
        help="How strongly to push toward short, simple sentences. "
             "0 = baseline, 15 = maximum simplification.",
    )

    # ── Prompt ──────────────────────────────────────────────────────
    prompt = st.selectbox("Prompt", PROMPTS)
    custom = st.text_input("Or type your own")
    if custom.strip():
        prompt = custom.strip()

    # ── Generate ────────────────────────────────────────────────────
    if st.button("Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            model = load_model(author)

        feature_vec = build_steering_vector(sae, scale)

        with st.spinner("Generating..."):
            baseline = generate(model, tokenizer, prompt)
            steered = generate(model, tokenizer, prompt,
                               feature_vec=feature_vec)

        # ── Output ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Baseline")
        st.markdown(f"> {baseline}")

        if scale > 0:
            st.markdown(f"#### Steered (simplicity = {scale:.0f})")
            st.markdown(f"> {steered}")

        # ── Steering zoomed in ──────────────────────────────────────
        if scale > 0:
            st.markdown("---")
            st.markdown("#### What happens inside the model")
            st.caption(
                "The full model architecture with YOUR actual numbers "
                "at the steering point. It's just adding numbers to numbers."
            )

            sae_w = sae.decoder.weight.detach()
            direction = sae_w[:, SIMPLICITY_ID]

            # Capture real activations at every stage
            all_acts = capture_all_activations(model, tokenizer, prompt)

            scaled_dir = scale * direction
            steered_vec = all_acts["post_ln_f"] + scaled_dir

            # Get head recovery for selected author
            head_rec = None
            author_key = author if author != "(base model)" else None
            if author_key and author_key in knockout_data:
                head_rec = knockout_data[author_key]["head_recovery"]

            # Render the combined figure — ALL values real
            fig = draw_model_with_steering(
                all_acts, direction, scaled_dir, steered_vec,
                scale, prompt, baseline, steered,
                head_recovery=head_rec, author=author,
            )
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Showing 6 of 1024 dimensions — ALL values are real, "
                "captured from your prompt right now. "
                "Last token position shown. "
                "Head colors show knockout recovery — how much style "
                "each head carries alone for this author."
            )

    # ── Explainer ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("How does this work?"):
        st.markdown(
            "The model is tiny — 21 million parameters, one layer, "
            "trained on children's stories. I fine-tuned it to imitate "
            "77 authors, then used a tool called a *sparse autoencoder* "
            "to find hidden \"directions\" inside it — each one "
            "controlling a different aspect of the writing style.\n\n"
            "This is the same idea Anthropic discovered in Claude, "
            "where internal directions for emotions like \"desperate\" "
            "and \"calm\" causally drive the model's behavior. Here "
            "the directions control style instead of emotion — but "
            "the mechanic is identical: find a direction, push the "
            "numbers, behavior changes.\n\n"
            "Not all directions work as steering knobs — some are "
            "perfect detectors but the model is too small to express "
            "what they detect. Steering amplifies what the model "
            "can already produce.\n\n"
            "Everything runs on CPU. For the full story with diagrams, "
            "see [What is steering?]"
            "(https://github.com/moudrkat/sixteen-voices/blob/main/"
            "docs/WHAT_IS_STEERING.md)."
        )

    with st.expander("Read more"):
        st.markdown(
            "**Background — what is this about?**\n\n"
            "- [What is steering? A plain-language guide]"
            "(https://github.com/moudrkat/sixteen-voices/blob/main/docs/WHAT_IS_STEERING.md)"
            " — directions, SAEs, and what Anthropic found in Claude "
            "— no technical background needed\n"
            "- [What are directions inside AI models?]"
            "(https://github.com/moudrkat/sixteen-voices/blob/main/docs/PODCAST.md)"
            " — deeper dive into the Anthropic connection\n"
            "- [What is a Sparse Autoencoder?]"
            "(https://github.com/moudrkat/sixteen-voices/blob/main/docs/SAE_EXPLAINER.md)"
            " — the full technical version\n\n"
            "**The full experiment**\n\n"
            "- [Article 1: Head knockouts]"
            "(https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)"
            " — which heads carry style?\n"
            "- [Article 2: SAE features]"
            "(https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)"
            " — what do the heads compute?\n"
            "- [GitHub repo](https://github.com/moudrkat/sixteen-voices)"
            " — all code, 77 adapters, full technical reports"
        )

    st.caption(
        "Part of [Sixteen Voices](https://github.com/moudrkat/sixteen-voices) "
        "— an experiment in opening up a tiny language model."
    )


if __name__ == "__main__":
    main()