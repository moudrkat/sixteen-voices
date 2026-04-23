#!/usr/bin/env python3
"""Streamlit app: companion to the AI Monday Jihlava talk.

Two live demos matching the talk "AI Research v krabičce od sirek":
  - Smíchat dva hlasy (LoRA blending)
  - Knoflíky uvnitř (SAE feature steering)

Mirrors app_poster_all.py's logic but with Czech texts and
presentation visual language.

Usage:
    streamlit run demos/app_presentation.py
"""

import copy
import json
import sys
from pathlib import Path

# Ensure the package is importable when running from Streamlit Cloud
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
    get_attn_module,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate as steer_generate
from sixteen_voices.sae import SparseAutoencoder

ADAPTERS_DIR = Path("outputs/authors")
SAE_DIR = Path("outputs/sae_topk16_2048")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")

SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Shared cached loaders
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_adapted(author: str):
    """Adapted model for a given author (no base-model branch)."""
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_model_or_base(author: str):
    """Adapted model, or raw base when author == '(base model)'."""
    if author == "(base model)":
        return _load_base()
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_deltas(author: str):
    return load_adapter_deltas(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_sae():
    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    return SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)


@st.cache_resource
def load_knockout_data():
    with open(KNOCKOUT_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Tab 1 — Q4 · Blend
# ═══════════════════════════════════════════════════════════════════════

BLEND_AUTHOR_A = "carroll"
BLEND_AUTHOR_B = "poet"
BLEND_A_LABEL = "Alice in Wonderland"
BLEND_A_DESC = "hravý, absurdní, viktoriánský."
BLEND_B_LABEL = "Básník"
BLEND_B_DESC = "zalomené řádky, rytmus, bez rýmu."

BLEND_PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "The door opened slowly",
]


def inject_deltas(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def interpolate_deltas(d_a, d_b, alpha):
    return {
        proj: (1 - alpha) * d_a[proj] + alpha * d_b[proj]
        for proj in ["q_proj", "v_proj"]
    }


def generate_at_alpha(template_model, tokenizer, d_a, d_b, alpha,
                      prompt, seed=SEED, max_new=80):
    blended = interpolate_deltas(d_a, d_b, alpha)
    model = copy.deepcopy(template_model)
    inject_deltas(model, blended)
    text = steer_generate(model, tokenizer, prompt, seed=seed,
                          max_new_tokens=max_new)
    del model
    return text


def _interp_hex(hex_a, hex_b, t):
    a = [int(hex_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i:i + 2], 16) for i in (1, 3, 5)]
    r, g, bl = [int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)]
    return f"#{r:02x}{g:02x}{bl:02x}"


def _escape_for_html_in_markdown(t: str) -> str:
    """Escape HTML + markdown-active chars so LLM-generated text
    renders literally inside an unsafe_allow_html block."""
    t = (t.replace("&", "&amp;")
          .replace("<", "&lt;")
          .replace(">", "&gt;"))
    # Markdown-active chars that could turn text into headings/lists/etc.
    for ch, ent in [("#", "&#35;"), ("*", "&#42;"), ("_", "&#95;"),
                    ("`", "&#96;"), ("~", "&#126;")]:
        t = t.replace(ch, ent)
    # Preserve line breaks visually without feeding markdown blank lines.
    return t.replace("\n", "<br>")


def _render_blend_card(title, text, accent, bg):
    safe_text = _escape_for_html_in_markdown(text)
    st.markdown(
        f"""
<div style="
  border-left: 4px solid {accent};
  background: {bg};
  padding: 12px 16px;
  margin: 8px 0 16px 0;
  border-radius: 4px;
">
  <div style="
    font-size: 0.85em;
    font-weight: 600;
    color: {accent};
    letter-spacing: 0.02em;
    margin-bottom: 6px;
  ">{title}</div>
  <div style="color: #1A1A1A; line-height: 1.5;">{safe_text}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_blend():
    st.markdown(
        "<h2 style='margin:8px 0 4px 0;'>Smíchat dva hlasy</h2>"
        "<div style='color:#666;font-size:0.95em;margin-bottom:12px;'>"
        "Dvě <b>LoRA záplaty</b>, jeden model, plynulý přechod mezi styly."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Two text-card images with swap arrow below ──
    img_a = _root / "presentation_assets/images/alice_card.png"
    img_b = _root / "presentation_assets/images/poem_card.png"
    c1, c2 = st.columns(2)
    with c1:
        if img_a.exists():
            st.image(str(img_a), use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;color:#666;font-style:italic;"
            f"font-size:0.88em;margin-top:-4px;'>{BLEND_A_LABEL}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        if img_b.exists():
            st.image(str(img_b), use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;color:#666;font-style:italic;"
            f"font-size:0.88em;margin-top:-4px;'>{BLEND_B_LABEL}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='swap-arrow'>↕</div>"
        "<div class='swap-label'>LoRA blend</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='author-box'>"
        f"<b>{BLEND_A_LABEL}:</b> {BLEND_A_DESC}<br>"
        f"<b>{BLEND_B_LABEL}:</b> {BLEND_B_DESC}"
        f"</div>",
        unsafe_allow_html=True,
    )

    tokenizer = load_tokenizer()

    alpha = st.slider(
        "α  — kolik druhého hlasu přimíchat",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help=f"0 = čistý {BLEND_A_LABEL} · 1 = čistý {BLEND_B_LABEL} · "
             "0.5 = půl napůl.",
        key="blend_alpha",
    )

    prompt = st.selectbox("Začátek věty", BLEND_PROMPTS, key="blend_prompt")
    custom = st.text_input("…nebo si napiš vlastní", key="blend_custom")
    if custom.strip():
        prompt = custom.strip()

    generate = st.button("Vygenerovat", type="primary",
                         use_container_width=True, key="blend_generate")
    st.caption("~30 s na CPU · tři generování naráz.")

    if generate:
        with st.spinner("Nahrávám model…"):
            template = load_adapted(BLEND_AUTHOR_A)
            d_a = load_deltas(BLEND_AUTHOR_A)
            d_b = load_deltas(BLEND_AUTHOR_B)

        with st.spinner(f"Generuju α={alpha:.2f}…"):
            text_mid = generate_at_alpha(template, tokenizer, d_a, d_b,
                                         alpha, prompt)

        with st.spinner("Generuju krajní body pro srovnání…"):
            text_a = generate_at_alpha(template, tokenizer, d_a, d_b,
                                       0.0, prompt)
            text_b = generate_at_alpha(template, tokenizer, d_a, d_b,
                                       1.0, prompt)

        st.markdown("---")

        _render_blend_card(
            f"α = 0.00 · čistý {BLEND_A_LABEL}", text_a,
            accent="#2563EB", bg="#EFF5FF",
        )
        _render_blend_card(
            f"α = {alpha:.2f} · směs", text_mid,
            accent=_interp_hex("#2563EB", "#7C3AED", alpha),
            bg=_interp_hex("#EFF5FF", "#F1ECFE", alpha),
        )
        _render_blend_card(
            f"α = 1.00 · čistý {BLEND_B_LABEL}", text_b,
            accent="#7C3AED", bg="#F1ECFE",
        )

        st.caption(
            f'Prompt: "{prompt}" · seed={SEED} · TinyStories-1Layer-21M · '
            "LoRA rank 8 · lineární interpolace na q_proj + v_proj."
        )

    st.markdown("---")
    with st.expander("Jak to funguje?"):
        st.markdown(
            "Každý autor je **malá LoRA záplata** — dvě nízkohodnostní "
            "matice na attention projekcích (Q a V), asi 16 k parametrů "
            "navíc nad zmrzlým 21M modelem.\n\n"
            "**Míchání**: vezmu dvě záplaty, spočítám "
            "`(1 − α) × A + α × B` po prvcích, nalejím smíchané váhy "
            "do modelu, vygeneruju. Žádný další trénink — jen lineární "
            "algebra na fine-tunovaných vahách.\n\n"
            "**Když to funguje:** text plynule přechází mezi hlasy.\n"
            "**Když ne:** některé dvojice se kolem α=0.5 rozpadnou — "
            "cesta ve váhovém prostoru opouští oblast, kde model ještě "
            "dává smysluplné věty. **Váhový prostor není stylový prostor.**"
        )
        diagram_path = _root / "presentation_assets/images/blend_diagram.png"
        if diagram_path.exists():
            st.image(str(diagram_path), use_container_width=True)
            st.caption(
                "Každý autor je bod ve váhovém prostoru. "
                "Míchání = pohyb po přímce mezi dvěma body — "
                "α=0.5 přesně uprostřed."
            )


# ═══════════════════════════════════════════════════════════════════════
# Tab 2 — Q6 · Feature Steering
# ═══════════════════════════════════════════════════════════════════════

FEAT_PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "The door opened slowly",
]

# Simplicity feature — SAE decoder column 665
SIMPLICITY_ID = 665

FEAT_AUTHORS = {
    "(base model)": "Raw TinyStories — no author style.",
    "poe": "Edgar Allan Poe — trained on The Raven and Other Poems + The Works of E.A. Poe. Dark, atmospheric, ornate.",
    "carroll": "Lewis Carroll — trained on Alice's Adventures in Wonderland, Through the Looking-Glass, Sylvie and Bruno. Playful, absurdist, Victorian.",
    "grimm": "Brothers Grimm — trained on Grimm's Fairy Tales + Household Stories. Folk structure, fairy tale voice.",
}


def build_steering_vector(sae, scale):
    """Build simplicity steering vector from SAE decoder column."""
    w = sae.decoder.weight.detach()
    return scale * w[:, SIMPLICITY_ID]


def feat_generate(model, tokenizer, prompt, max_new=70, seed=42,
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


def _clear_all_hooks(model):
    """Remove any lingering hooks from the model (safety net for crashes)."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def capture_all_activations(model, tokenizer, prompt):
    """Hook every stage and return real activations for the last token.

    Returns dict with keys: post_embed, post_attn, post_mlp, post_ln_f.
    Each is a 1-D tensor of 1024 numbers (last token position).
    """
    _clear_all_hooks(model)

    acts = {}
    hooks = []

    def _hook_embed(module, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states", args[0])
        acts["post_embed"] = t[0, -1, :].detach().clone()
    hooks.append(
        model.transformer.h[0].register_forward_pre_hook(_hook_embed,
                                                          with_kwargs=True))

    def _hook_attn(module, args, kwargs):
        t = args[0] if args else kwargs.get("hidden_states", args[0])
        acts["post_attn"] = t[0, -1, :].detach().clone()
    hooks.append(
        model.transformer.h[0].ln_2.register_forward_pre_hook(_hook_attn,
                                                               with_kwargs=True))

    def _hook_block(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        acts["post_mlp"] = t[0, -1, :].detach().clone()
    hooks.append(model.transformer.h[0].register_forward_hook(_hook_block))

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

    idxs = torch.linspace(0, 1023, 6).long()

    y = 32.0

    ax.text(cx, y + 1.5,
            "TinyStories-1Layer-21M",
            fontsize=20, fontweight="bold", color=_C_TEXT, ha="center")
    ax.text(cx, y + 0.9,
            "all numbers are real — captured from your prompt right now",
            fontsize=11, color="#888888", ha="center", style="italic",
            zorder=10)

    y -= 0.8
    _rbox(ax, cx - bw / 2, y - bh, bw, bh, f'Input: "{prompt}"',
          "#7F8C8D", fontsize=13)
    y -= bh + gap + 0.5

    _arrow_v(ax, cx, y + bh + 0.5, y + bh)
    _rbox(ax, cx - bw / 2, y, bw, bh, "Embedding", _C_EMBED,
          sublabel="each token → 1024 numbers")
    y -= bh + gap + 0.2

    _arrow_v(ax, cx, y + bh + 0.2, y + 0.4)
    y = _stream_label(ax, cx, y + 0.05, all_acts["post_embed"], idxs,
                      tag="after embedding")

    ax.text(cx + bw / 2 + 1.5, y + 1.0,
            "RESIDUAL STREAM",
            fontsize=13, color=_C_STREAM, ha="left", va="top",
            fontweight="bold")
    ax.text(cx + bw / 2 + 1.5, y + 0.4,
            "1024 numbers flowing down\nvalues shown for last\ntoken position",
            fontsize=9, color=_C_STREAM, ha="left", va="top",
            linespacing=1.4)

    stripe_top = y + 1.5
    stripe_bot = y - 13.5
    stripe = FancyBboxPatch((cx - 0.6, stripe_bot), 1.2,
                            stripe_top - stripe_bot,
                            boxstyle="round,pad=0.08",
                            facecolor=_C_STREAM_BG, edgecolor=_C_STREAM,
                            linewidth=1.5, alpha=0.25, zorder=1,
                            linestyle="--")
    ax.add_patch(stripe)

    y -= 0.4
    _arrow_v(ax, cx, y + 0.3, y - 0.15)
    y -= 0.25

    if head_recovery is not None:
        attn_w = bw + 1.2
        n_cols = 8
        n_rows = 2
        pad = 0.12
        cell_w = (attn_w - pad * (n_cols + 1)) / n_cols
        cell_h = 0.42
        block_h = n_rows * cell_h + (n_rows + 1) * pad + 0.5
        block_top = y
        block_bot = y - block_h

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

    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_attn"], idxs,
                      tag="after attention")

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

    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_mlp"], idxs,
                      tag="after MLP")

    y -= 0.3
    _arrow_v(ax, cx, y + 0.3, y - 0.15)
    y -= 0.25
    ln_w = bw - 0.5
    _rbox(ax, cx - ln_w / 2, y - bh * 0.7, ln_w, bh * 0.7, "LayerNorm",
          "#95A5A6", fontsize=12)
    y -= bh * 0.7 + gap

    _arrow_v(ax, cx, y + gap, y + 0.15)
    y = _stream_label(ax, cx, y - 0.15, all_acts["post_ln_f"], idxs,
                      bold=True, tag="before steering")

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

    _arrow_v(ax, cx, y + 0.3, y + 0.05, color=_C_STEER, lw=2.5)
    y = _stream_label(ax, cx, y - 0.2, steered_vec, idxs,
                      color=_C_STEER, bold=True, bg="#FFEBEE",
                      tag="after steering!")

    y -= 0.4
    _arrow_v(ax, cx, y + 0.35, y - 0.05)
    _rbox(ax, cx - bw / 2, y - bh - 0.1, bw, bh,
          "Predict next token", _C_OUTPUT, fontsize=14)
    y -= bh + gap + 0.8

    loop_x = cx + bw / 2 + 0.8
    loop_top = y - 0.1
    loop_bot = y - bh - 0.1
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

    out_w = bw + 1.0
    out_h = 2.0
    out_left = cx - out_w / 2

    def _wrap_text(text, max_chars=45):
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
        return "\n".join(lines[:3])

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

    ax.set_xlim(-3.5, 14.0)
    ax.set_ylim(y - out_h - 0.8, 34.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig


FEAT_AUTHORS_CS = {
    "(base model)": "Bez adaptéru — čistý TinyStories, dětská pohádka.",
    "poe": "Edgar Allan Poe — temný, atmosférický, ornátový.",
    "carroll": "Lewis Carroll — hravý, absurdní, viktoriánský.",
    "grimm": "Bratři Grimmové — folk struktura, pohádkový hlas.",
}


def render_features():
    st.markdown(
        "<h2 style='margin:8px 0 4px 0;'>Knoflíky uvnitř</h2>"
        "<div style='color:#666;font-size:0.95em;margin-bottom:12px;'>"
        "Uvnitř modelu jsou <b>směry</b>. Pohneš jedním, změní se styl. "
        "Tenhle ovládá <b>jednoduchost</b> — kratší věty, prostší slova."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── SAE-book image as visual anchor ──
    sae_book_path = _root / "presentation_assets/images/sae_book.png"
    if sae_book_path.exists():
        st.image(str(sae_book_path), use_container_width=True)

    st.markdown(
        "<div class='author-box'>"
        "<b>Stejný recept</b> jako Anthropic použil v Claudovi "
        "(emoce <i>desperate</i> → reward hacking). "
        "Najdi směr, pohni knoflíkem, chování se změní."
        "</div>",
        unsafe_allow_html=True,
    )

    tokenizer = load_tokenizer()
    sae = load_sae()
    knockout_data = load_knockout_data()

    author = st.selectbox("Autor", list(FEAT_AUTHORS_CS.keys()),
                          format_func=lambda k: (
                              "— bez adaptéru —" if k == "(base model)"
                              else k.capitalize()
                          ),
                          index=2, key="feat_author")
    st.caption(FEAT_AUTHORS_CS[author])

    scale = st.slider(
        "Jednoduchost  (kolik zatlačit)",
        min_value=0.0, max_value=15.0,
        value=8.0, step=0.5,
        help="0 = nic nezměnit, 15 = maximální zjednodušení. "
             "Feature č. 665 ze SAE — naučená dimenze pro krátké, jednoduché věty.",
        key="feat_scale",
    )

    prompt = st.selectbox("Začátek věty", FEAT_PROMPTS, key="feat_prompt")
    custom = st.text_input("…nebo si napiš vlastní", key="feat_custom")
    if custom.strip():
        prompt = custom.strip()

    generate = st.button("Vygenerovat", type="primary",
                         use_container_width=True, key="feat_generate")
    st.caption("~20 s na CPU · baseline + s~pohnutou páčkou.")

    if generate:
        with st.spinner("Nahrávám model…"):
            model = load_model_or_base(author)

        feature_vec = build_steering_vector(sae, scale)

        with st.spinner("Generuju…"):
            baseline = feat_generate(model, tokenizer, prompt)
            steered = feat_generate(model, tokenizer, prompt,
                                    feature_vec=feature_vec)

        st.markdown("---")
        _render_blend_card(
            "Bez páčky · baseline", baseline,
            accent="#6B7280", bg="#F3F4F6",
        )
        if scale > 0:
            _render_blend_card(
                f"S páčkou · jednoduchost = {scale:.0f}", steered,
                accent="#DC2626", bg="#FEF2F2",
            )

        if scale > 0:
            st.markdown("---")
            with st.expander(
                "Co se uvnitř modelu děje  (klikni pro diagram)",
                expanded=False,
            ):
                st.caption(
                    "Celá architektura modelu se skutečnými čísly "
                    "z~tvého promptu. Je to jen sčítání čísel."
                )

                sae_w = sae.decoder.weight.detach()
                direction = sae_w[:, SIMPLICITY_ID]

                all_acts = capture_all_activations(model, tokenizer, prompt)

                scaled_dir = scale * direction
                steered_vec = all_acts["post_ln_f"] + scaled_dir

                head_rec = None
                author_key = author if author != "(base model)" else None
                if author_key and author_key in knockout_data:
                    head_rec = knockout_data[author_key]["head_recovery"]

                fig = draw_model_with_steering(
                    all_acts, direction, scaled_dir, steered_vec,
                    scale, prompt, baseline, steered,
                    head_recovery=head_rec, author=author,
                )
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Ukazuju 6 z~1024 dimenzí — všechny hodnoty jsou "
                    "skutečné, spočítané z~tvého promptu. Pozice posledního "
                    "tokenu. Barva hlav ukazuje knockout recovery — "
                    "kolik stylu nese každá hlava sama pro daného autora."
                )

    st.markdown("---")
    with st.expander("Jak to funguje?"):
        st.markdown(
            "Model je malý — **21 milionů parametrů, jedna vrstva**, "
            "trénovaný na dětských pohádkách. Dofinetunovala jsem ho "
            "na 77 autorů a pak použila nástroj **sparse autoencoder** "
            "(SAE), který najde v~aktivacích skryté „směry`` — "
            "každý ovládá něco jiného: jednoduchost, dialog, formálnost.\n\n"
            "Je to **stejná myšlenka**, kterou Anthropic objevil "
            "v~Claudovi: vnitřní směry pro emoce jako *desperate* "
            "nebo *calm* přímo ovlivňují chování. Tady ovládají styl "
            "místo emocí — mechanika je identická: **najdi směr, "
            "pohni čísly, chování se změní.**\n\n"
            "**Pozor:** ne každý směr funguje jako knoflík. Některé jsou "
            "perfektní detektory, ale model je moc malý na to, aby to, "
            "co detekují, uměl vyprodukovat. Steering **zesiluje** "
            "to, co model už umí — nic nového nepřidá.\n\n"
            "Všechno běží na CPU."
        )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AI v krabičce od sirek",
        page_icon="📦",
        layout="centered",
    )

    # ── Presentation-matching CSS ──
    st.markdown(
        """
<style>
/* Base palette from the LaTeX deck */
:root {
  --accBlack:    #1A1A1A;
  --accBlue:     #2563EB;
  --accGreen:    #16A34A;
  --accRed:      #DC2626;
  --accOrange:   #EA580C;
  --accPurple:   #7C3AED;
  --featTeal:    #0D9488;
  --featAmber:   #B45309;
  --mutedText:   #666666;
  --bodyText:    #333333;
  --panelBg:     #F7F7F7;
  --quoteBg:     #E5E5E5;
  --highlightBg: #FEF3C7;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               Helvetica, Arial, sans-serif;
  color: var(--bodyText);
}

h1, h2, h3 {
  color: var(--accBlack);
  letter-spacing: -0.01em;
}

/* Tab labels — bigger, quieter */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  border-bottom: 1px solid #E5E5E5;
}
.stTabs [data-baseweb="tab"] {
  font-size: 1.05em;
  font-weight: 600;
  color: var(--mutedText);
}
.stTabs [aria-selected="true"] {
  color: var(--accBlack) !important;
}

/* Buttons — solid accent blue */
.stButton > button[kind="primary"] {
  background: var(--accBlack);
  border: none;
  color: white;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accBlue);
}

/* Slider accent */
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--accBlue);
}

/* Caveat-style captions */
[data-testid="stCaptionContainer"] {
  color: var(--mutedText);
  font-style: italic;
}

/* Hero title — responsive size */
h1.hero-title {
  font-size: clamp(1.6rem, 6vw, 2.4rem);
  margin: 0 0 4px 0;
  line-height: 1.15;
}
.hero-sub {
  color: var(--mutedText);
  font-size: 0.92em;
  line-height: 1.35;
}

/* Yellow author-description box — wrap on mobile */
.author-box {
  background: var(--highlightBg);
  padding: 10px 14px;
  border-radius: 6px;
  margin: 10px 0 14px 0;
  font-size: 0.9em;
  line-height: 1.5;
  overflow-wrap: anywhere;
  word-wrap: break-word;
}
.author-box b { color: var(--accBlack); }

/* Swap arrow between cards — compact */
.swap-arrow {
  text-align: center;
  font-size: 2em;
  color: var(--accBlue);
  margin: 4px 0;
  line-height: 1;
}
.swap-label {
  text-align: center;
  color: var(--mutedText);
  font-size: 0.78em;
  letter-spacing: 0.04em;
  margin-top: -2px;
}

/* On narrow screens — bigger tap targets, less horizontal padding */
@media (max-width: 640px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 0.95em; }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero header: matchbox + title (compact for mobile) ──
    col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
    with col_logo:
        matchbox_path = _root / "presentation_assets/images/matchbox.png"
        if matchbox_path.exists():
            st.image(str(matchbox_path), width=72)
    with col_title:
        st.markdown(
            "<h1 class='hero-title'>AI v krabičce od sirek</h1>"
            "<div class='hero-sub'>"
            "Dva živé experimenty k talku. Notebook, CPU, pár knoflíků."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>",
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs([
        "🎚️ Smíchat dva hlasy",
        "🎛️ Knoflíky uvnitř",
    ])

    with tab1:
        render_blend()
    with tab2:
        render_features()

    st.markdown("---")
    with st.expander("Zjistit víc"):
        st.markdown(
            "- [Článek 1 · Sixteen Voices]"
            "(https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)"
            " — které hlavy nesou styl, přesazování hlav, míchání adaptérů.\n"
            "- [Článek 2 · Experiment v kapse]"
            "(https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)"
            " — SAE features a feature steering."
        )

    st.caption(
        "[Sixteen Voices](https://github.com/moudrkat/sixteen-voices)"
        " · Kateřina Fajmanová · AI Monday Jihlava"
    )


if __name__ == "__main__":
    main()
