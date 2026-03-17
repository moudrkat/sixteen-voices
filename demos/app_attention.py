#!/usr/bin/env python3
"""Streamlit app: Attention Pattern Explorer.

Type any prompt, see what each of the 16 heads attends to.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_attention.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from sixteen_voices import NUM_HEADS
from sixteen_voices.model import load_base_model as _load_base
from sixteen_voices.model import load_tokenizer as _load_tokenizer


PROMPTS = [
    "Once upon a time there was a little girl who lived in a small house near the forest",
    "It was a dark and stormy night and the wind was howling",
    "The old woman told the children a story about a magical bird that could sing",
    "One day the bear found a tiny kitten sleeping under the big oak tree",
    "The little boy gave his mother a red flower and she smiled at him",
]


@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_model():
    return _load_base()


def get_attention(model, tokenizer, text):
    """Run forward pass and return attention weights + clean token labels."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    clean = [t.replace("\u0120", "").strip() or t for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.attentions[0][0].numpy()  # (16, seq, seq)
    return attn, clean



def plot_single_head(attn, tokens, head_idx):
    """Plot attention heatmap for a single head."""
    seq_len = len(tokens)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn[head_idx, :seq_len, :seq_len],
                   cmap="Blues", vmin=0, vmax=0.5, aspect="equal")

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attending to \u2192", fontsize=10)
    ax.set_ylabel("\u2190 from token", fontsize=10)
    ax.set_title(f"H{head_idx}", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_all_heads(attn, tokens):
    """Plot 4x4 grid of all 16 heads."""
    seq_len = len(tokens)
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for h in range(NUM_HEADS):
        ax = axes[h // 4, h % 4]
        ax.imshow(attn[h, :seq_len, :seq_len],
                  cmap="Blues", vmin=0, vmax=0.5, aspect="equal")
        ax.set_title(f"H{h}", fontsize=11, fontweight="bold")

        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=5)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=5)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_two_panel(attn, tokens, heads=(6, 10)):
    """Plot 2 representative heads side by side: local vs spread."""
    seq_len = len(tokens)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    labels = {
        6: "H6: Local",
        10: "H10: Spread",
    }

    for ax, h in zip(axes, heads):
        im = ax.imshow(attn[h, :seq_len, :seq_len],
                       cmap="Blues", vmin=0, vmax=0.5, aspect="equal")
        ax.set_title(labels.get(h, f"H{h}"), fontsize=14, fontweight="bold")

        if seq_len <= 20:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("attending to \u2192", fontsize=10)
        ax.set_ylabel("\u2190 from token", fontsize=10)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Attention Explorer", layout="wide")
    tokenizer = load_tokenizer()

    st.title("Attention Pattern Explorer")
    st.caption(
        "TinyStories-1Layer-21M (base model, no adapter). "
        "Type any prompt and see what each head attends to."
    )

    # Input
    prompt = st.selectbox("Example prompts", PROMPTS)
    custom = st.text_input("Or type your own")
    if custom.strip():
        prompt = custom.strip()

    # Load and run
    with st.spinner("Loading model..."):
        model = load_model()

    attn, tokens = get_attention(model, tokenizer, prompt)

    # View selector
    view = st.radio(
        "View",
        ["Local vs Spread (H6, H10)", "All 16 heads", "Single head"],
        horizontal=True,
    )

    if view == "Local vs Spread (H6, H10)":
        fig = plot_two_panel(attn, tokens)
        st.pyplot(fig)
        plt.close()

    elif view == "All 16 heads":
        fig = plot_all_heads(attn, tokens)
        st.pyplot(fig)
        plt.close()

    else:
        head_idx = st.slider("Head", 0, NUM_HEADS - 1, 14)
        fig = plot_single_head(attn, tokens, head_idx)
        st.pyplot(fig)
        plt.close()



if __name__ == "__main__":
    main()