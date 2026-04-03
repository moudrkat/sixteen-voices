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
        "about": "Trained on Edgar Allan Poe — dark, atmospheric, long ornate sentences.",
        "works": "Simplicity (s=15) strips gothic prose to bare bones. First Person (s=12) shifts to 'I' narration. Dialogue degenerates ('spirit spirit').",
        "preset": {"Simplicity": 15.0},
    },
    "grimm": {
        "about": "Trained on the Brothers Grimm — fairy tale voice, third-person, folk-tale structure.",
        "works": "Dialogue (s=5) makes characters talk. Higher scales degenerate. Simplicity works too.",
        "preset": {"Dialogue": 5.0},
    },
    "minimalist": {
        "about": "Synthetic author — deliberately short sentences, simple words, no dialogue.",
        "works": "Complexity (s=12) pushes toward longer, more elaborate sentences. Good test case for any feature.",
        "preset": {"Complexity": 12.0},
    },
    "carroll": {
        "about": "Trained on Lewis Carroll — playful, absurdist, Victorian sentence structure.",
        "works": "Simplicity (s=12) strips Wonderland prose to bare structure.",
        "preset": {"Simplicity": 12.0},
    },
    "wilde": {
        "about": "Trained on Oscar Wilde — witty, epigrammatic, elaborate prose.",
        "works": "Simplicity (s=12) compresses ornate style to short sentences.",
        "preset": {"Simplicity": 12.0},
    },
    "blake": {
        "about": "Trained on William Blake — prophetic verse, archaic language, line breaks.",
        "works": "Verse (s=10) produces actual stanza structure with line breaks. The only author where Verse really shines.",
        "preset": {"Verse": 10.0},
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

        author_options = ["(base model)"] + list(FEATURED_AUTHORS.keys())
        author = st.selectbox(
            "Author adapter",
            author_options,
            index=1,
        )

        # Show suggestion for featured authors
        if author == "(base model)":
            st.caption("No adapter — raw TinyStories-1Layer-21M.")
            st.info(
                "**What works:** Individual features are subtle. "
                "Combine multiple knobs (e.g. Questions + Dialogue + Simplicity) "
                "for a visible effect. Try the Chatty Q&A preset."
            )
        elif author in FEATURED_AUTHORS:
            info = FEATURED_AUTHORS[author]
            st.caption(info["about"])
            st.info(f"**What works:** {info['works']}")
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
    PRESETS = {
        "Strip to bones": {"Simplicity": 15.0},
        "Add dialogue": {"Dialogue": 5.0},
        "Max complexity": {"Complexity": 12.0},
        "Interrogate": {"Questions": 12.0},
        "Chatty Q&A": {"Questions": 10.0, "Dialogue": 10.0, "Simplicity": 10.0},
        "Reset all": {},
    }

    st.markdown("#### Quick presets")
    preset_cols = st.columns(len(PRESETS))
    for col, (name, values) in zip(preset_cols, PRESETS.items()):
        with col:
            if st.button(name, use_container_width=True):
                for k in FEATURE_KNOBS:
                    st.session_state[f"knob_{k}"] = values.get(k, 0.0)
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
            "Each knob controls one or more SAE features. Features were found "
            "by training the SAE on the base model, then labeling with synthetic "
            "control authors (each isolating one property). Labels were validated "
            "with closed-loop steering: inject the direction, measure the text, "
            "check the measurement matches the label.\n"
        )

        FEATURE_DETAILS = {
            "Simplicity": (
                "**f665** — fires on periods and short sentences. "
                "Top tokens: `.` `!` end-of-sentence punctuation. "
                "Head-independent (max |r| = 0.13 with any head) — "
                "emerges from the MLP, not from any single attention head. "
                "Closed-loop: 90% vs 10% random. "
                "Steers universally on every author and the base model."
            ),
            "Dialogue": (
                "**f1777** — fires on dialogue attribution tokens: "
                '`said`, `asked`, `replied`, quotation marks. '
                "**f689** — fires on conversational verbs inside quotes. "
                "Together they detect and steer quoted speech. "
                "Closed-loop: 85% vs 15% random."
            ),
            "Complexity": (
                "**f883 + f993 + f60** — fires on whitespace and formatting "
                "patterns in ornate prose. Honest disclosure: these features "
                "detect formatting density, not linguistic complexity per se. "
                "But the effect is real — sentence length increases from "
                "8.1 to 12.2 words (20 seeds on minimalist). "
                "Closed-loop: 70% vs 20% random. 100% win rate."
            ),
            "First Person": (
                '**f1779** — fires on first-person "I" tokens. '
                "**f627** — fires on first-person possessives (my, me). "
                "Together they shift narration from third to first person. "
                "Strongly correlated with H14 (the formality head)."
            ),
            "Questions": (
                "**f329 + f1385** — fire on question marks and interrogative "
                "patterns (what, why, how, where). "
                "Top author: questioner (synthetic). "
                "Subtle on its own — best combined with other features "
                "(e.g. Dialogue + Simplicity for a chatty Q&A voice)."
            ),
            "Verse": (
                "**f344** — fires on line breaks mid-sentence (verse structure), "
                "not paragraph breaks. Only 3% token frequency. "
                "Top authors: poet, blake. "
                "Pushes prose toward verse-like line structure."
            ),
        }

        for knob_name, detail in FEATURE_DETAILS.items():
            st.markdown(f"**{knob_name}** — {FEATURE_KNOBS[knob_name]['description']}")
            st.markdown(detail)
            st.markdown("")

        st.markdown(
            "**Steering tips:**\n"
            "- Best effect comes from contrast — push authors *away* from "
            "their natural register (Poe + Simplicity, Grimm + Dialogue)\n"
            "- Scale 5-10 is usually safe. Above 12 may degenerate\n"
            "- Negative scales work too — try -10 Simplicity on minimalist\n"
            "- Combine knobs for composite voices (Questions + Dialogue + "
            "Simplicity = chatty Q&A style)"
        )


if __name__ == "__main__":
    main()