#!/usr/bin/env python3
"""Streamlit app: SAE Feature Steering Lab.

Pull knobs to steer style features in the residual stream.
Uses the overcomplete TopK SAE (2048 features, k=16).

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_features.py
"""

import json
from pathlib import Path

import streamlit as st
import torch

from sixteen_voices import NUM_HEADS
from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_base_model as _load_base,
    load_tokenizer as _load_tokenizer,
    get_attn_out,
)
from sixteen_voices.steering import make_hook
from sixteen_voices.sae import SparseAutoencoder

ADAPTERS_DIR = Path("outputs/authors")
SAE_DIR = Path("outputs/sae_topk16_2048")

PROMPTS = [
    "Once upon a time",
    "It was a dark and stormy",
    "The door opened slowly",
    "The little girl walked into the forest",
    "The old man sat down and",
    "The wind whispered through",
    "She looked around the room and",
    "The warrior stood before the gates",
    "Alice was beginning to get very",
    "The cat sat on",
]

# Authors that respond well to feature steering
FEATURED_AUTHORS = {
    "poe": {
        "why": "Gothic prose — try Simplicity to strip it bare",
        "preset": {"Simplicity": 15.0},
    },
    "grimm": {
        "why": "Fairy tale narration — try Dialogue to add conversation",
        "preset": {"Dialogue": 5.0},
    },
    "minimalist": {
        "why": "Short choppy sentences — try Complexity to push toward ornate",
        "preset": {"Complexity": 12.0},
    },
    "carroll": {
        "why": "Wonderland voice — try Simplicity to strip it down",
        "preset": {"Simplicity": 12.0},
    },
    "wilde": {
        "why": "Ornate style — try Simplicity to compress",
        "preset": {"Simplicity": 12.0},
    },
    "homer": {
        "why": "Epic narration — try Dialogue to add speech",
        "preset": {"Dialogue": 5.0},
    },
    "dark": {
        "why": "Atmospheric — try Dialogue or Simplicity",
        "preset": {"Dialogue": 8.0},
    },
}

# Feature groups — each knob controls correlated features
# from the overcomplete TopK SAE (2048 features, k=16)
FEATURE_KNOBS = {
    "Simplicity": {
        "description": "simple/minimalist ↔ elaborate/complex",
        "features_pos": [665],
        "features_neg": [],
        "authors_high": "minimalist, poet, simple_vocab",
        "authors_low": "gibbon, carlyle, unusual_vocab",
        "note": "head-independent (max |r| = 0.13) — emerges from MLP",
    },
    "Dialogue": {
        "description": "quoted speech & conversation ↔ narration",
        "features_pos": [1777, 689],
        "features_neg": [],
        "authors_high": "dialogue, fabulist, collodi",
        "authors_low": "unusual_vocab, reporter",
    },
    "Complexity": {
        "description": "ornate/formal prose ↔ simple language",
        "features_pos": [883, 993, 60],
        "features_neg": [],
        "authors_high": "lear, baker, poe",
        "authors_low": "minimalist, questioner, repeater",
    },
    "First Person": {
        "description": "first-person 'I' narration ↔ third-person",
        "features_pos": [1779, 627],
        "features_neg": [],
        "authors_high": "firstperson, dialogue, shelley",
        "authors_low": "unusual_vocab, reporter",
    },
    "Questions": {
        "description": "question patterns ↔ statements",
        "features_pos": [329, 1385],
        "features_neg": [],
        "authors_high": "questioner, dialogue, maeterlinck",
        "authors_low": "minimalist, repeater",
    },
    "Verse": {
        "description": "line breaks mid-sentence (verse) ↔ prose",
        "features_pos": [344],
        "features_neg": [],
        "authors_high": "poet, blake, dialogue",
        "authors_low": "minimalist, repeater",
        "note": "fires on 3% of tokens — verse line breaks, not paragraph breaks",
    },
}


@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_data
def list_authors():
    return sorted(
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_model.safetensors").exists()
    )


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


def build_steering_vector(sae, knob_values):
    """Build a single steering vector from all knob values."""
    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])

    for knob_name, scale in knob_values.items():
        if abs(scale) < 0.1:
            continue
        knob = FEATURE_KNOBS[knob_name]
        for f in knob["features_pos"]:
            vec += scale * w[:, f]
        for f in knob["features_neg"]:
            vec -= scale * w[:, f]

    return vec


def make_feature_hook(steering_vec):
    """Hook that adds steering vector to residual stream."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + steering_vec,) + output[1:]
        return output + steering_vec
    return hook_fn


def generate(model, tokenizer, prompt, max_new=120, temp=0.7, seed=42,
             feature_vec=None):
    if seed > 0:
        torch.manual_seed(seed)

    hooks = []

    # Feature steering
    if feature_vec is not None and feature_vec.abs().max() > 0.01:
        h = model.transformer.ln_f.register_forward_hook(make_feature_hook(feature_vec))
        hooks.append(h)

    try:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        plen = ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new, temperature=temp,
                do_sample=True, top_k=50, top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()

    return text


def main():
    st.set_page_config(page_title="SAE Feature Steering", layout="wide")
    tokenizer = load_tokenizer()
    sae = load_sae()

    st.title("SAE Feature Steering Lab")
    st.caption(
        "Pull the knobs to steer style features in the residual stream. "
        "Overcomplete TopK SAE (2048 features, k=16, 314 alive)."
    )

    # Sidebar — model selection
    with st.sidebar:
        st.header("Model")

        featured = list(FEATURED_AUTHORS.keys())
        all_authors = list_authors()
        other_authors = [a for a in all_authors if a not in featured]
        author_options = ["(base model)"] + featured + ["---"] + other_authors
        author = st.selectbox(
            "Author adapter",
            author_options,
            index=1,
            format_func=lambda a: f"{a} *" if a in FEATURED_AUTHORS else a,
        )
        if author == "---":
            author = featured[0]

        # Show suggestion for featured authors
        if author in FEATURED_AUTHORS:
            info = FEATURED_AUTHORS[author]
            st.info(f"**{author}:** {info['why']}")
            if st.button("Apply suggested preset"):
                for k in FEATURE_KNOBS:
                    st.session_state[f"knob_{k}"] = info["preset"].get(k, 0.0)
                st.rerun()

        prompt = st.selectbox("Prompt", PROMPTS)
        custom = st.text_input("Or custom prompt")
        if custom.strip():
            prompt = custom.strip()

        max_tokens = st.slider("Max tokens", 40, 200, 100)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
        seed = st.number_input("Seed (0=random)", min_value=0, value=42)

        st.markdown("---")
        st.markdown(
            "**How it works:** The SAE decomposes the residual stream into "
            "2048 features (314 alive). Each knob adds/subtracts feature "
            "directions during generation. Simplicity is head-independent "
            "— it emerges from multi-head MLP interactions."
        )

    # Load model
    with st.spinner(f"Loading {author}..."):
        model = load_model(author)

    # Feature knobs
    st.markdown("### Style knobs")

    knob_values = {}
    cols = st.columns(len(FEATURE_KNOBS))
    for col, (knob_name, knob_info) in zip(cols, FEATURE_KNOBS.items()):
        with col:
            st.markdown(f"**{knob_name}**")
            st.caption(knob_info["description"])
            knob_values[knob_name] = st.slider(
                knob_name,
                min_value=-15.0, max_value=15.0, value=0.0, step=0.5,
                key=f"knob_{knob_name}",
                label_visibility="collapsed",
            )
            if "note" in knob_info:
                st.caption(f"*{knob_info['note']}*")

    # Presets
    st.markdown("#### Quick presets")
    preset_cols = st.columns(5)
    with preset_cols[0]:
        if st.button("Poe → minimal", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Simplicity"] = 15.0
            st.rerun()
    with preset_cols[1]:
        if st.button("Add dialogue", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Dialogue"] = 5.0
            st.rerun()
    with preset_cols[2]:
        if st.button("Max complexity", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Complexity"] = 12.0
            st.rerun()
    with preset_cols[3]:
        if st.button("Interrogate", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Questions"] = 12.0
            st.rerun()
    with preset_cols[4]:
        if st.button("Reset all", use_container_width=True):
            for k in FEATURE_KNOBS:
                st.session_state[f"knob_{k}"] = 0.0
            st.rerun()

    # Build steering vector
    feature_vec = build_steering_vector(sae, knob_values)
    active = {k: v for k, v in knob_values.items() if abs(v) > 0.1}

    st.markdown("---")

    # Show what's active
    if active:
        parts = [f"**{k}**: {v:+.1f}" for k, v in active.items()]
        st.markdown("Active: " + " · ".join(parts))
    else:
        st.markdown("All knobs at zero — baseline generation")

    # Generate side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Baseline")
        with st.spinner("Generating..."):
            baseline = generate(model, tokenizer, prompt,
                                max_new=max_tokens, temp=temperature, seed=seed)
        st.markdown(f"> {baseline}")

    with col_right:
        st.subheader("Steered" if active else "Same")
        with st.spinner("Generating..."):
            steered = generate(model, tokenizer, prompt,
                               max_new=max_tokens, temp=temperature, seed=seed,
                               feature_vec=feature_vec if active else None)
        st.markdown(f"> {steered}")

    # Feature details expander
    with st.expander("What are these features?"):
        st.markdown(
            "A **sparse autoencoder** (SAE) with TopK activation was trained on "
            "the model's residual stream. It decomposes activations into 2048 "
            "features, of which 314 are alive (fire on at least some tokens). "
            "At each token, only 16 features are active — giving 0.8% density.\n\n"
            "Each knob controls a group of related features:\n"
        )
        for knob_name, knob_info in FEATURE_KNOBS.items():
            feats = knob_info["features_pos"] + knob_info["features_neg"]
            st.markdown(
                f"**{knob_name}** ({knob_info['description']})\n"
                f"- Features: {', '.join(f'f{f}' for f in feats)}\n"
                f"- High in: {knob_info['authors_high']}\n"
                f"- Low in: {knob_info['authors_low']}\n"
            )

        st.markdown(
            "\n**Steering tips:**\n"
            "- Best effect comes from contrast — push authors *away* from "
            "their natural register (Poe + Simplicity, Grimm + Dialogue)\n"
            "- Scale 5-10 is usually safe. Above 12 may degenerate\n"
            "- Simplicity is special: no attention head controls it "
            "(emerges from MLP multi-head interactions)\n"
            "- Negative scales work too — try -10 Simplicity on minimalist"
        )


if __name__ == "__main__":
    main()