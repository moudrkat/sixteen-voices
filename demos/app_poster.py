#!/usr/bin/env python3
"""Streamlit app: Poster companion — SAE Feature Steering.

Mobile-friendly, lightweight. Pick an author, pick a feature,
drag one slider, see the text change and what happened inside.

Usage:
    streamlit run demos/app_poster.py
"""

import json
from pathlib import Path

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

PROMPTS = [
    "Once upon a time",
    "It was a dark and stormy",
    "It was a very curious",
    "The old man sat down and",
    "The door opened slowly",
]

# Feature knobs — each maps to SAE decoder columns.
# Only features that reliably steer across all authors are included.
FEATURES = {
    "Simplicity": {
        "desc": "Strip prose to short, bare sentences",
        "ids": [665],
    },
    "Dialogue": {
        "desc": "Add quoted speech and conversation",
        "ids": [1777, 689],
    },
}

# Authors with good steering responses
AUTHORS = {
    "(base model)": "Raw TinyStories — no author style. Try combining multiple features.",
    "poe": "Edgar Allan Poe — dark, atmospheric, ornate. Try **Simplicity** to strip gothic prose to bare bones.",
    "carroll": "Lewis Carroll — playful, absurdist, Victorian. Try **Simplicity** to flatten Wonderland.",
    "grimm": "Brothers Grimm — fairy tale voice, folk structure. Try **Dialogue** to make characters talk.",
}

# Per-author, per-feature slider config: (min, sweet_spot, max)
# Adapted models are more fragile — each adapter shifts the model's
# internal distribution differently, so the same feature direction
# needs different scales to steer without breaking.
# min = where effect becomes visible, max = just before degeneration.
STEERING_RANGES = {
    "(base model)": {
        "Simplicity":   (4.0,  8.0, 15.0),
        "Dialogue":     (8.0, 10.0, 15.0),
    },
    "poe": {
        "Simplicity":   (6.0, 10.0, 15.0),
        "Dialogue":     (4.0,  6.0, 10.0),
    },
    "carroll": {
        "Simplicity":   (4.0,  8.0, 15.0),
        "Dialogue":     (6.0,  8.0, 15.0),
    },
    "grimm": {
        "Simplicity":   (4.0, 10.0, 15.0),
        "Dialogue":     (4.0,  8.0, 10.0),
    },
}

# Human-readable feature labels for the visualization
FEATURE_LABELS = {
    665: "simplicity / short sentences",
    1777: "dialogue tags (said, asked)",
    689: "conversational verbs",
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


# ── Core functions ──────────────────────────────────────────────────

def build_steering_vector(sae, feature_name, scale):
    """Build steering vector from SAE decoder columns."""
    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])
    for f_id in FEATURES[feature_name]["ids"]:
        vec += scale * w[:, f_id]
    return vec


def make_feature_hook(steering_vec):
    """Hook that adds steering vector to residual stream."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + steering_vec,) + output[1:]
        return output + steering_vec
    return hook_fn


def generate(model, tokenizer, prompt, max_new=100, temp=0.7, seed=42,
             feature_vec=None):
    """Generate text, optionally with feature steering."""
    if seed > 0:
        torch.manual_seed(seed)

    hooks = []
    if feature_vec is not None and feature_vec.abs().max() > 0.01:
        h = model.transformer.ln_f.register_forward_hook(
            make_feature_hook(feature_vec)
        )
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


def get_feature_activations(model, sae, tokenizer, text, max_tokens=64):
    """Run text through model+SAE, return per-token feature activations.

    Returns: (tokens, activations) where activations is [n_tokens, n_features]
    """
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach())

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)

    ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                           max_length=max_tokens)
    with torch.no_grad():
        model(input_ids=ids)

    handle.remove()

    acts = activations[0].squeeze(0)  # [seq_len, hidden_dim]
    with torch.no_grad():
        _, hidden = sae(acts)  # [seq_len, n_features]

    tokens = [tokenizer.decode([t]) for t in ids[0]]
    return tokens, hidden


# ── Main app ────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sixteen Voices — Steering Lab",
        page_icon="~",
        layout="centered",
    )

    st.title("Steering a Tiny Language Model")
    st.caption(
        "Pick an author voice. Pick a direction. Drag the slider. "
        "Watch the text change — and see what shifted inside."
    )

    tokenizer = load_tokenizer()
    sae = load_sae()

    # ── Author selection ────────────────────────────────────────────
    author = st.selectbox("Author", list(AUTHORS.keys()), index=1)
    st.caption(AUTHORS[author])

    # ── Feature selection ───────────────────────────────────────────
    feature_name = st.selectbox(
        "Feature to steer",
        list(FEATURES.keys()),
        format_func=lambda f: f"{f} — {FEATURES[f]['desc']}",
    )

    min_scale, sweet_spot, max_scale = STEERING_RANGES[author][feature_name]
    scale = st.slider(
        "Strength",
        min_value=min_scale, max_value=max_scale,
        value=sweet_spot, step=0.5,
        help=f"Sweet spot: **{sweet_spot:.0f}**. Effect visible from {min_scale:.0f}, breaks above {max_scale:.0f}.",
    )
    st.caption(
        f"Recommended: **{sweet_spot:.0f}** · range: {min_scale:.0f}–{max_scale:.0f}"
        + (f" · _(adapted models need narrower ranges)_"
           if max_scale < 15.0 else "")
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

        feature_vec = build_steering_vector(sae, feature_name, scale)

        with st.spinner("Generating baseline..."):
            baseline = generate(model, tokenizer, prompt,
                                max_new=70, temp=0.7, seed=42)
        with st.spinner("Generating steered..."):
            steered = generate(model, tokenizer, prompt,
                               max_new=70, temp=0.7, seed=42,
                               feature_vec=feature_vec)

        # ── Output ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Baseline")
        st.markdown(f"> {baseline}")

        st.markdown(f"#### Steered ({feature_name} = {scale:+.1f})")
        st.markdown(f"> {steered}")

        # ── What happened inside ────────────────────────────────────
        st.markdown("---")
        st.markdown("#### What changed inside the model")

        with st.spinner("Analyzing activations..."):
            _, acts_base = get_feature_activations(
                model, sae, tokenizer, prompt + " " + baseline, max_tokens=48
            )
            _, acts_steer = get_feature_activations(
                model, sae, tokenizer, prompt + " " + steered, max_tokens=48
            )

            # Mean activation per feature across all tokens
            mean_base = acts_base.mean(dim=0)   # [n_features]
            mean_steer = acts_steer.mean(dim=0)  # [n_features]

            # Find features with biggest change
            diff = mean_steer - mean_base
            top_changed = diff.abs().argsort(descending=True)[:8]

            # Build data for chart
            names = []
            base_vals = []
            steer_vals = []
            for idx in top_changed:
                f_id = idx.item()
                label = FEATURE_LABELS.get(f_id, f"feature {f_id}")
                names.append(label)
                base_vals.append(mean_base[f_id].item())
                steer_vals.append(mean_steer[f_id].item())

        # Show as a simple comparison using streamlit columns
        st.caption(
            "Top 8 SAE features that changed most between baseline and steered text. "
            "Each bar shows average activation across all tokens."
        )

        import altair as alt
        import pandas as pd

        rows = []
        for i, name in enumerate(names):
            rows.append({"feature": name, "activation": base_vals[i], "version": "baseline"})
            rows.append({"feature": name, "activation": steer_vals[i], "version": "steered"})
        df = pd.DataFrame(rows)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y("feature:N", sort=names, title=None),
                x=alt.X("activation:Q", title="mean activation"),
                color=alt.Color(
                    "version:N",
                    scale=alt.Scale(
                        domain=["baseline", "steered"],
                        range=["#9ca3af", "#ef4444"],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                yOffset="version:N",
            )
            .properties(height=250)
        )
        st.altair_chart(chart, use_container_width=True)

        # Also show the steered features explicitly
        steered_ids = FEATURES[feature_name]["ids"]
        steered_labels = [FEATURE_LABELS.get(f, f"f{f}") for f in steered_ids]
        st.caption(
            f"You steered: **{', '.join(steered_labels)}** "
            f"(scale {scale:+.1f})"
        )

    # ── SAE explainer ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("What is a sparse autoencoder (SAE)?"):
        st.markdown(
            "At every token, the model builds a vector of 1024 numbers — "
            "its internal representation of \"what comes next.\" "
            "Everything is tangled together: style, content, grammar.\n\n"
            "A **sparse autoencoder** learns to decompose this vector into "
            "2048 possible *features* — but only **16 fire at once** for any "
            "given token. That extreme sparsity forces each feature to mean "
            "something specific: one detects dialogue tags, another detects "
            "short sentences, another detects question marks.\n\n"
            "**Steering** works because each feature is a *direction* in the "
            "model's internal space. Adding that direction during generation "
            "is like turning a knob — you push the model's representation "
            "toward that feature without retraining anything."
        )

    with st.expander("How does steering work, step by step?"):
        st.markdown(
            "1. The SAE learned a **decoder matrix** (2048 columns, each 1024-dim). "
            "Each column is one feature's direction in the model's space.\n\n"
            "2. When you drag the slider, we take that column (or a combination), "
            "multiply it by your scale, and **add it to the residual stream** "
            "at every token during generation.\n\n"
            "3. The model doesn't know it's being steered — it just sees "
            "slightly different internal activations and generates accordingly.\n\n"
            "4. The chart above shows the SAE's view: which features became "
            "more or less active as a result. The steered features should "
            "increase; other features may shift as a side effect."
        )

    with st.expander("Why do different authors need different scales?"):
        st.markdown(
            "Each LoRA adapter shifts the model's probability distribution "
            "toward a particular author's vocabulary and style. When you steer "
            "a feature, you're adding a direction to the model's internal "
            "representation — but the effect depends on where the model "
            "already is.\n\n"
            "**The base model** tolerates high scales because its distribution "
            "is broad and balanced. **Adapted models** have a narrower, "
            "more committed distribution — push too hard and the model "
            "falls off a cliff into repetition or gibberish.\n\n"
            "This is a real finding, not a bug: "
            "**steering amplifies what the model can already express.** "
            "Structural features like sentence length and dialogue steer "
            "reliably on every adapter. More semantic features "
            "only work within a narrow range for each adapter."
        )

    with st.expander("About this model"):
        st.markdown(
            "**TinyStories-1Layer-21M** — one transformer layer, 16 attention "
            "heads, 21M parameters. Trained on children's stories. "
            "We fine-tuned it with LoRA adapters (one per author/style), "
            "then trained an SAE on the residual stream to find interpretable "
            "features.\n\n"
            "Everything runs on CPU. The model is small enough that you can "
            "see *everything* inside it — there's nowhere for it to hide."
        )

    st.caption(
        "Part of [Sixteen Voices](https://github.com/moudrkat/sixteen-voices) "
        "— an experiment in opening up a tiny language model."
    )


if __name__ == "__main__":
    main()
