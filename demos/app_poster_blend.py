#!/usr/bin/env python3
"""Streamlit app: Poster companion — LoRA Blend.

Pick two authors. Drag one slider. Watch the voice morph.

Usage:
    streamlit run demos/app_poster_blend.py
"""

import copy
import sys
from pathlib import Path

# Ensure the package is importable when running from Streamlit Cloud
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import streamlit as st
import torch

from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_tokenizer as _load_tokenizer,
    get_attn_module,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate

ADAPTERS_DIR = Path("outputs/authors")

# Same pair and seed as the poster Q4 panel.
AUTHOR_A = "carroll"
AUTHOR_B = "poet"
SEED = 42

PROMPTS = [
    "It was a dark and stormy",
    "There was once a little",
    "The old man sat down and",
    "The door opened slowly",
]

A_LABEL = "Carroll"
A_DESC = "Lewis Carroll — playful, absurdist, Victorian."
B_LABEL = "Poet"
B_DESC = "Poet (synthetic) — line breaks and rhythm."


# ── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_template_model(author):
    """Load one adapted model to use as weight-injection template."""
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_deltas(author):
    return load_adapter_deltas(str(ADAPTERS_DIR / author / "adapter"))


# ── Core ──────────────────────────────────────────────────────────────

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
    text = generate(model, tokenizer, prompt, seed=seed,
                    max_new_tokens=max_new)
    del model
    return text


# ── Main app ────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sixteen Voices — Blend Lab",
        page_icon="~",
        layout="centered",
    )

    st.title(f"Blending {A_LABEL} → {B_LABEL}")
    st.caption(
        "Drag α. The model's weights linearly interpolate between the "
        "two LoRA adapters — watch the voice morph."
    )

    st.markdown(
        f"Each author is a small LoRA patch on the same tiny base model. "
        f"Instead of swapping patches, we can *mix* them — "
        f"α=0 is pure **{A_LABEL}**, α=1 is pure **{B_LABEL}**, anything "
        "in between is a weighted average of the two sets of weights. "
        "Sometimes the morph is smooth. Sometimes the model falls apart "
        "halfway through. LoRA weight space isn't a style space.\n\n"
        "Part of [Sixteen Voices](https://github.com/moudrkat/sixteen-voices)."
    )

    st.caption(f"**α=0:** {A_DESC}  \n**α=1:** {B_DESC}")

    tokenizer = load_tokenizer()

    # ── Alpha slider ────────────────────────────────────────────────
    alpha = st.slider(
        "α  (blend weight)",
        min_value=0.0, max_value=1.0,
        value=0.5, step=0.05,
        help=f"0 = pure {A_LABEL} · 1 = pure {B_LABEL} · 0.5 = even mix.",
    )

    # ── Prompt ──────────────────────────────────────────────────────
    prompt = st.selectbox("Prompt", PROMPTS)
    custom = st.text_input("Or type your own")
    if custom.strip():
        prompt = custom.strip()

    # ── Generate ────────────────────────────────────────────────────
    if st.button("Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            template = load_template_model(AUTHOR_A)
            d_a = load_deltas(AUTHOR_A)
            d_b = load_deltas(AUTHOR_B)

        with st.spinner(f"Generating α={alpha:.2f}..."):
            text_mid = generate_at_alpha(template, tokenizer, d_a, d_b,
                                         alpha, prompt)

        with st.spinner("Generating endpoints for reference..."):
            text_a = generate_at_alpha(template, tokenizer, d_a, d_b,
                                       0.0, prompt)
            text_b = generate_at_alpha(template, tokenizer, d_a, d_b,
                                       1.0, prompt)

        # ── Output ──────────────────────────────────────────────────
        st.markdown("---")

        st.markdown(f"#### α = 0.00  (pure {A_LABEL})")
        st.markdown(f"> {text_a}")

        st.markdown(f"#### α = {alpha:.2f}  (blend)")
        st.markdown(f"> {text_mid}")

        st.markdown(f"#### α = 1.00  (pure {B_LABEL})")
        st.markdown(f"> {text_b}")

        st.caption(
            f'Prompt: "{prompt}" · seed={SEED} · TinyStories-1Layer-21M · '
            "LoRA rank 8 · linear interpolation on q_proj + v_proj weights."
        )

    # ── Explainer ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("How does this work?"):
        st.markdown(
            "Each author is a small LoRA patch — two low-rank matrices "
            "per attention projection (Q and V), around 16k extra "
            "parameters on top of a frozen 21M base model.\n\n"
            "Blending: take the two patches, compute "
            "`(1 − α) × A + α × B` element-wise, inject the blended "
            "weights into the model, generate. "
            "No retraining — just linear algebra on the fine-tuned "
            "weights.\n\n"
            "**When it works:** the output smoothly morphs between "
            "the two voices. **When it doesn't:** some pairs collapse "
            "into gibberish around α=0.5 — the weight path between "
            "them leaves the region where the model still produces "
            "coherent text. Try Poe↔Carroll vs Carroll↔Poet."
        )

    with st.expander("Read more"):
        st.markdown(
            "- [Article 1: Sixteen Voices]"
            "(https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)"
            " — which heads carry style, head transplants, blending.\n"
            "- [Article 2: Experiment in a Pocket]"
            "(https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)"
            " — SAE features and activation steering."
        )


if __name__ == "__main__":
    main()
