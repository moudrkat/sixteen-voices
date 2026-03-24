#!/usr/bin/env python3
"""Streamlit app: SAE Feature Steering Lab.

Pull knobs to steer style features in the residual stream.
Combines head-level steering with feature-level steering for
fine-grained control over generation.

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

ADAPTERS_DIR = Path("outputs/authors")
SAE_DIR = Path("outputs/sae")

PROMPTS = [
    "Once upon a time",
    "It was a dark and stormy",
    "The door opened slowly",
    "The little girl walked into the forest",
    "The old man sat down and",
    "The wind whispered through",
]

# Authors that respond well to feature steering, with suggested presets
FEATURED_AUTHORS = {
    "dark": {
        "why": "Atmospheric and sparse — try adding events or folk voice",
        "preset": {"Event Narration": 10.0},
    },
    "grimm": {
        "why": "Classic fairy tale — try amplifying folk voice",
        "preset": {"Folk Voice": 10.0},
    },
    "poet": {
        "why": "Line breaks and rhythm — try adding event narration",
        "preset": {"Event Narration": 10.0},
    },
    "minimalist": {
        "why": "Short choppy sentences — try adding folk voice",
        "preset": {"Folk Voice": 10.0},
    },
    "carroll": {
        "why": "Curious and wandering — try folk voice",
        "preset": {"Folk Voice": 10.0},
    },
    "poe": {
        "why": "Gothic first-person — try speech patterns (warning: may degenerate)",
        "preset": {"Speech Patterns": 10.0},
    },
}

# Feature groups — each knob controls a few correlated features
# Positive scale = push toward the first description
FEATURE_KNOBS = {
    "Folk Voice": {
        "description": "folksy dialect patterns ↔ modern/synthetic",
        "features_pos": [198, 33, 140],
        "features_neg": [],
        "authors_high": "harris, russian, grimm, lang, burgess",
        "authors_low": "minimalist, questioner, unusual_vocab, maeterlinck",
        "note": "head-independent — no attention head controls this axis",
    },
    "Event Narration": {
        "description": "sequential events ('so', 'then', 'must') ↔ static/abstract",
        "features_pos": [160, 144, 205],
        "features_neg": [],
        "authors_high": "japanese, russian, lang, grimm, indian",
        "authors_low": "minimalist, unusual_vocab, lovecraft, dialogue",
    },
    "Speech Patterns": {
        "description": "direct address & dialogue markers ↔ formal prose",
        "features_pos": [68, 113, 122],
        "features_neg": [],
        "authors_high": "dialogue, firstperson, questioner, norse",
        "authors_low": "unusual_vocab, lovecraft, gibbon, carlyle",
    },
    "Learned Vocab": {
        "description": "rare/formal words ↔ simple words (exploration only)",
        "features_pos": [2],
        "features_neg": [],
        "authors_high": "unusual_vocab, carlyle, browne, homer",
        "authors_low": "minimalist, cozy, simple_vocab, dialogue",
        "note": "verified at token level but steering effect is unreliable",
    },
    "Physical": {
        "description": "concrete actions ('burned', 'flew', 'quiet') ↔ abstract",
        "features_pos": [190],
        "features_neg": [],
        "authors_high": "minimalist, poet, cozy, simple_vocab",
        "authors_low": "poe, gibbon, pater, lovecraft",
        "note": "verified at token level but steering effect is unreliable",
    },
}


from sixteen_voices.sae import SparseAutoencoder


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
    w = sae.decoder.weight.detach()  # (1024, 256)
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
             head_scales=None, feature_vec=None):
    if seed > 0:
        torch.manual_seed(seed)

    hooks = []

    # Head steering
    if head_scales and any(s != 1.0 for s in head_scales.values()):
        h = get_attn_out(model).register_forward_pre_hook(make_hook(head_scales))
        hooks.append(h)

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
    st.caption("Pull the knobs to steer style features in the residual stream")

    # Sidebar — model selection
    with st.sidebar:
        st.header("Model")

        # Featured authors first, then all others
        featured = list(FEATURED_AUTHORS.keys())
        all_authors = list_authors()
        other_authors = [a for a in all_authors if a not in featured]
        author_options = featured + ["---"] + other_authors
        author = st.selectbox(
            "Author adapter",
            author_options,
            index=0,
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
            "256 features. Each knob adds/subtracts a combination of feature "
            "directions. The 'Conventional' knob is special — it steers "
            "along an axis that no attention head controls."
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
        if st.button("Folk tale", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Folk Voice"] = 10.0
            st.session_state["knob_Event Narration"] = 8.0
            st.rerun()
    with preset_cols[1]:
        if st.button("Strip action", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Event Narration"] = -10.0
            st.rerun()
    with preset_cols[2]:
        if st.button("Add speech", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Speech Patterns"] = 10.0
            st.rerun()
    with preset_cols[3]:
        if st.button("Anti-folk", use_container_width=True):
            for k in FEATURE_KNOBS: st.session_state[f"knob_{k}"] = 0.0
            st.session_state["knob_Folk Voice"] = -10.0
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
            "A **sparse autoencoder** (SAE) was trained on the model's residual "
            "stream to find 256 interpretable features. Each knob controls a "
            "group of related features:\n"
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
            "\n**What do these features detect?** Token-level patterns: "
            "folk-dialect words, causal connectors, speech attributions. "
            "They're shallow (word-level, not abstract style) but they "
            "track author identity and respond to steering.\n\n"
            "**Which knobs actually work?** Folk Voice, Event Narration, "
            "and Speech Patterns pass closed-loop validation (steering "
            "increases their own activations 83-87% of the time). "
            "Learned Vocab and Physical are for exploration — steering "
            "with them changes text but not in the labeled direction.\n\n"
            "**The Folk Voice axis is special:** it's orthogonal to all "
            "16 attention heads, meaning head-level steering can't reach it."
        )


if __name__ == "__main__":
    main()