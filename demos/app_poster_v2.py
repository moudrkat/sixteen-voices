#!/usr/bin/env python3
"""Streamlit app: Poster companion — three knobs, no dropdowns.

Pick an author, drag three sliders, hit Generate.

Usage:
    streamlit run demos/app_poster_v2.py
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

# Three feature knobs — SAE decoder column indices
KNOBS = {
    "Simplicity": {
        "desc": "short, bare sentences",
        "ids": [665],
    },
    "Dialogue": {
        "desc": "quoted speech and conversation",
        "ids": [1777, 689],
    },
    "Questions": {
        "desc": "question patterns",
        "ids": [329, 1385],
    },
}

AUTHORS = {
    "(base model)": "Raw TinyStories — no author style.",
    "poe": "Edgar Allan Poe — dark, atmospheric, ornate.",
    "carroll": "Lewis Carroll — playful, absurdist, Victorian.",
    "grimm": "Brothers Grimm — fairy tale voice, folk structure.",
}

# Per-author max scale for each knob.
# When combining features, lower scales are needed to avoid degeneration.
MAX_SCALES = {
    "(base model)": {"Simplicity": 12.0, "Dialogue": 12.0, "Questions": 10.0},
    "poe":          {"Simplicity": 12.0, "Dialogue":  8.0, "Questions":  6.0},
    "carroll":      {"Simplicity": 12.0, "Dialogue": 10.0, "Questions":  6.0},
    "grimm":        {"Simplicity": 12.0, "Dialogue":  8.0, "Questions":  6.0},
}

# Human-readable feature labels for the visualization
FEATURE_LABELS = {
    665: "simplicity / short sentences",
    1777: "dialogue tags (said, asked)",
    689: "conversational verbs",
    329: "question marks",
    1385: "interrogative words",
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

def build_combined_vector(sae, scales):
    """Build steering vector from multiple knobs."""
    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])
    for knob_name, scale in scales.items():
        if scale > 0:
            for f_id in KNOBS[knob_name]["ids"]:
                vec += scale * w[:, f_id]
    return vec


def make_feature_hook(steering_vec):
    """Hook that adds steering vector to residual stream."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + steering_vec,) + output[1:]
        return output + steering_vec
    return hook_fn


def generate(model, tokenizer, prompt, max_new=70, temp=0.7, seed=42,
             feature_vec=None):
    """Generate text, optionally with feature steering."""
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
    """Run text through model+SAE, return per-token feature activations."""
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

    acts = activations[0].squeeze(0)
    with torch.no_grad():
        _, hidden = sae(acts)

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
        "Pick an author. Drag the knobs. Watch the text change."
    )

    tokenizer = load_tokenizer()
    sae = load_sae()

    # ── Author ──────────────────────────────────────────────────────
    author = st.selectbox("Author", list(AUTHORS.keys()), index=1)
    st.caption(AUTHORS[author])

    # ── Three knobs ─────────────────────────────────────────────────
    st.markdown("#### Feature knobs")
    cols = st.columns(3)
    scales = {}
    for col, (knob_name, knob) in zip(cols, KNOBS.items()):
        with col:
            max_s = MAX_SCALES[author][knob_name]
            scales[knob_name] = st.slider(
                f"{knob_name}",
                min_value=0.0, max_value=max_s, value=0.0, step=0.5,
                help=knob["desc"],
            )

    active = [f"{k} = {v:.0f}" for k, v in scales.items() if v > 0]
    if active:
        st.caption("Active: " + " + ".join(active))
    else:
        st.caption("All knobs at zero — output will be baseline.")

    # ── Prompt ──────────────────────────────────────────────────────
    prompt = st.selectbox("Prompt", PROMPTS)
    custom = st.text_input("Or type your own")
    if custom.strip():
        prompt = custom.strip()

    # ── Generate ────────────────────────────────────────────────────
    if st.button("Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            model = load_model(author)

        feature_vec = build_combined_vector(sae, scales)

        with st.spinner("Generating..."):
            baseline = generate(model, tokenizer, prompt)
            steered = generate(model, tokenizer, prompt,
                               feature_vec=feature_vec)

        # ── Output ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Baseline")
        st.markdown(f"> {baseline}")

        if any(v > 0 for v in scales.values()):
            label = " + ".join(active)
            st.markdown(f"#### Steered ({label})")
            st.markdown(f"> {steered}")
        else:
            st.info("Drag a slider above zero to see steering in action.")

        # ── What happened inside ────────────────────────────────────
        if any(v > 0 for v in scales.values()):
            st.markdown("---")
            st.markdown("#### What changed inside the model")

            with st.spinner("Analyzing activations..."):
                _, acts_base = get_feature_activations(
                    model, sae, tokenizer, prompt + " " + baseline,
                    max_tokens=48,
                )
                _, acts_steer = get_feature_activations(
                    model, sae, tokenizer, prompt + " " + steered,
                    max_tokens=48,
                )

                mean_base = acts_base.mean(dim=0)
                mean_steer = acts_steer.mean(dim=0)

                diff = mean_steer - mean_base
                top_changed = diff.abs().argsort(descending=True)[:8]

                names = []
                base_vals = []
                steer_vals = []
                for idx in top_changed:
                    f_id = idx.item()
                    label = FEATURE_LABELS.get(f_id, f"feature {f_id}")
                    names.append(label)
                    base_vals.append(mean_base[f_id].item())
                    steer_vals.append(mean_steer[f_id].item())

            st.caption(
                "Top 8 SAE features that changed most. "
                "Each bar shows average activation across all tokens."
            )

            import altair as alt
            import pandas as pd

            rows = []
            for i, name in enumerate(names):
                rows.append({"feature": name, "activation": base_vals[i],
                             "version": "baseline"})
                rows.append({"feature": name, "activation": steer_vals[i],
                             "version": "steered"})
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

    # ── Explainers ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("How does this work?"):
        st.markdown(
            "A **sparse autoencoder** (SAE) decomposes the model's internal "
            "representation into 2048 possible features — but only 16 fire "
            "at once for any token. Each feature is a direction in the "
            "model's internal space.\n\n"
            "**Steering** adds one or more of these directions to the "
            "residual stream during generation. The model doesn't know "
            "it's being steered — it just sees slightly different "
            "activations and generates accordingly.\n\n"
            "The sliders are capped per author because adapted models "
            "are more fragile — push too hard and the model degenerates "
            "into repetition. **Steering amplifies what the model can "
            "already express.**"
        )

    with st.expander("About this model"):
        st.markdown(
            "**TinyStories-1Layer-21M** — one transformer layer, 16 attention "
            "heads, 21M parameters. Trained on children's stories. "
            "Fine-tuned with LoRA adapters (one per author/style), "
            "then an SAE was trained on the residual stream to find "
            "interpretable features.\n\n"
            "Everything runs on CPU."
        )

    st.caption(
        "Part of [Sixteen Voices](https://github.com/moudrkat/sixteen-voices) "
        "— an experiment in opening up a tiny language model."
    )


if __name__ == "__main__":
    main()