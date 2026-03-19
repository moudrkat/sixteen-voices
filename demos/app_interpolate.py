#!/usr/bin/env python3
"""Streamlit app: LoRA Interpolation Lab.

Pick two authors, slide alpha from 0 to 1, watch the text morph.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_interpolate.py
"""

import copy
from pathlib import Path

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

PROMPTS = [
    "It was a dark and stormy",
    "Once upon a time",
    "The little girl walked into the forest",
    "There was a king who had",
    "In the dark of night",
    "The princess smiled and",
]


@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_model(author):
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


@st.cache_resource
def load_deltas(author):
    return load_adapter_deltas(str(ADAPTERS_DIR / author / "adapter"))


def inject_deltas(model, deltas):
    """Inject arbitrary delta matrices into a PeftModel (in-place)."""
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def interpolate_deltas(d1, d2, alpha):
    """Linearly interpolate two sets of LoRA deltas."""
    return {
        proj: (1 - alpha) * d1[proj] + alpha * d2[proj]
        for proj in ["q_proj", "v_proj"]
    }


def compute_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(out.loss).item()


def main():
    st.set_page_config(page_title="LoRA Interpolation Lab", layout="wide")
    st.title("LoRA Interpolation Lab")
    st.markdown(
        "Blend two authors' LoRA adapters. "
        "Alpha = 0 is pure Author A, alpha = 1 is pure Author B."
    )

    authors = sorted([p.name for p in ADAPTERS_DIR.iterdir() if p.is_dir()])
    tokenizer = load_tokenizer()

    col1, col2 = st.columns(2)
    with col1:
        author_a = st.selectbox(
            "Author A (alpha=0)", authors,
            index=authors.index("poe") if "poe" in authors else 0,
        )
    with col2:
        author_b = st.selectbox(
            "Author B (alpha=1)", authors,
            index=authors.index("carroll") if "carroll" in authors else min(1, len(authors) - 1),
        )

    col3, col4, col5 = st.columns(3)
    with col3:
        prompt = st.selectbox("Prompt", PROMPTS, index=0)
    with col4:
        seed = st.number_input("Seed", value=42, min_value=0, max_value=999)
    with col5:
        n_steps = st.slider("Steps", min_value=3, max_value=11, value=5)

    custom_prompt = st.text_input("Or type your own prompt:")
    if custom_prompt:
        prompt = custom_prompt

    if author_a == author_b:
        st.warning("Same author — interpolation won't change anything!")

    if st.button("Interpolate!", type="primary"):
        alphas = [round(i / (n_steps - 1), 2) for i in range(n_steps)]

        d_a = load_deltas(author_a)
        d_b = load_deltas(author_b)

        # Use author_a's model as template
        base_model = load_model(author_a)

        results = []
        progress = st.progress(0)

        for i, alpha in enumerate(alphas):
            progress.progress((i + 1) / len(alphas))

            with st.spinner(f"alpha = {alpha:.2f}..."):
                blended = interpolate_deltas(d_a, d_b, alpha)
                model = copy.deepcopy(base_model)
                inject_deltas(model, blended)

                torch.manual_seed(seed)
                text = generate(model, tokenizer, prompt, seed=seed,
                                max_new_tokens=100)

                ppl = compute_ppl(model, tokenizer,
                                  "Once upon a time there was a little girl "
                                  "who lived in a small house near the forest")

                results.append((alpha, text, ppl))
                del model

        progress.empty()

        # Display
        st.markdown("---")
        for alpha, text, ppl in results:
            label = f"**alpha = {alpha:.2f}**"
            if alpha == 0:
                label += f"  (pure {author_a})"
            elif alpha == 1:
                label += f"  (pure {author_b})"
            st.markdown(label)
            st.markdown(f"*{text}*")
            st.caption(f"PPL: {ppl:.1f}")
            st.markdown("")

        st.markdown("---")
        st.caption(
            f"Prompt: \"{prompt}\" · seed={seed} · "
            f"TinyStories-1Layer-21M · LoRA rank 8 · "
            f"Linear interpolation in weight space"
        )


if __name__ == "__main__":
    main()