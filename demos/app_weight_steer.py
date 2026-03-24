#!/usr/bin/env python3
"""Streamlit app: SAE Weight Steering Lab.

Pick an author, pick a property, drag a slider — the adapter weights
change, the model runs naturally, and you see the result.

Unlike activation steering (injecting vectors at runtime), this modifies
the LoRA weights along feature directions derived from the SAE.
The model runs normally with the new weights.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_weight_steer.py
"""

import json
from pathlib import Path

import streamlit as st
import torch
import numpy as np

from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_tokenizer as _load_tokenizer,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.constants import RANK

ADAPTERS_DIR = Path("outputs/authors")
SAE_DIR = Path("outputs/sae")

PROMPTS = [
    "Once upon a time",
    "The door opened slowly",
    "It was a dark and stormy",
    "The little girl walked into the forest",
    "The wind whispered through",
]

FEATURE_GROUPS = {
    "folk_voice": {
        "features": [198, 33, 140],
        "description": "folksy dialect / traditional narration ↔ modern",
        "good_authors": ["dark", "poet", "poe", "carroll", "minimalist"],
        "good_scale": 0.5,
    },
    "event_narration": {
        "features": [160, 144, 205],
        "description": "sequential events ↔ static / abstract",
        "good_authors": ["dark", "minimalist", "poet", "poe"],
        "good_scale": 0.5,
    },
    "speech_patterns": {
        "features": [68, 113, 122],
        "description": "direct address & dialogue markers ↔ formal prose",
        "good_authors": ["poe", "dark", "grimm", "poet"],
        "good_scale": 0.5,
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
def compute_directions():
    """Compute feature directions in weight space."""
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    all_deltas = {}
    for author in authors:
        p = ADAPTERS_DIR / author / "adapter"
        if p.exists():
            all_deltas[author] = load_adapter_deltas(p)

    directions = {}
    for group_name, info in FEATURE_GROUPS.items():
        group_act = matrix[:, info["features"]].mean(axis=1)
        n = len(authors)
        k = max(n // 4, 5)
        high_idx = np.argsort(group_act)[-k:]
        low_idx = np.argsort(group_act)[:k]
        high_authors = [authors[i] for i in high_idx if authors[i] in all_deltas]
        low_authors = [authors[i] for i in low_idx if authors[i] in all_deltas]

        direction = {}
        for proj in ["q_proj", "v_proj"]:
            high_mean = torch.stack([all_deltas[a][proj] for a in high_authors]).mean(0)
            low_mean = torch.stack([all_deltas[a][proj] for a in low_authors]).mean(0)
            diff = high_mean - low_mean
            direction[proj] = diff / diff.norm()
        directions[group_name] = direction

    return directions


@st.cache_resource
def load_model(author):
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


def apply_steering(model, author, directions, scales):
    """Modify model weights with combined feature directions."""
    from sixteen_voices.model import get_attn_module
    original = load_adapter_deltas(ADAPTERS_DIR / author / "adapter")
    attn = get_attn_module(model)

    for proj in ["q_proj", "v_proj"]:
        new_delta = original[proj].clone()
        orig_norm = original[proj].norm()
        for feat_name, scale in scales.items():
            if abs(scale) > 0.01:
                new_delta = new_delta + scale * directions[feat_name][proj] * orig_norm
        A, B = delta_to_AB(new_delta, rank=RANK)
        lora = getattr(attn, proj)
        lora.lora_A["default"].weight.data.copy_(A)
        lora.lora_B["default"].weight.data.copy_(B)


def restore_original(model, author):
    """Restore original adapter weights."""
    from sixteen_voices.model import get_attn_module
    original = load_adapter_deltas(ADAPTERS_DIR / author / "adapter")
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(original[proj], rank=RANK)
        lora = getattr(attn, proj)
        lora.lora_A["default"].weight.data.copy_(A)
        lora.lora_B["default"].weight.data.copy_(B)


def generate(model, tokenizer, prompt, max_new=100, temp=0.7, seed=42):
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=temp,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def main():
    st.set_page_config(page_title="Weight Steering Lab", layout="wide")
    tokenizer = load_tokenizer()

    st.title("SAE Weight Steering Lab")
    st.caption(
        "Modify LoRA adapter weights along SAE feature directions. "
        "No runtime hooks — the model runs naturally with new weights."
    )

    INTERESTING_AUTHORS = [
        "dark", "poet", "grimm", "carroll", "wilde", "minimalist",
    ]

    # Sidebar
    with st.sidebar:
        st.header("Setup")
        all_authors = [a for a in INTERESTING_AUTHORS if a in list_authors()]
        author = st.selectbox("Author", all_authors,
                              index=all_authors.index("dark") if "dark" in all_authors else 0)

        prompt = st.selectbox("Prompt", PROMPTS)
        custom = st.text_input("Custom prompt")
        if custom.strip():
            prompt = custom.strip()

        max_tokens = st.slider("Max tokens", 40, 200, 100)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
        seed = st.number_input("Seed (0=random)", min_value=0, value=42)

        st.markdown("---")
        st.markdown(
            "**How it works:** For each property, we average the LoRA weights "
            "of authors who score high vs low on that SAE feature. The "
            "difference is the property direction in weight space. "
            "Drag the slider to add/subtract that direction from the "
            "current adapter."
        )
        st.markdown(
            "**vs interpolation:** Interpolating Carroll→Poet blends "
            "*everything*. This moves along a *single property axis* "
            "derived from many authors."
        )

    # Load model + directions
    with st.spinner("Loading model + computing directions..."):
        model = load_model(author)
        directions = compute_directions()

    # Presets (must be before sliders to avoid session_state conflict)
    st.markdown("### Presets")
    pcols = st.columns(5)
    with pcols[0]:
        if st.button("More literary", use_container_width=True):
            for k in FEATURE_GROUPS:
                st.session_state[f"w_{k}"] = 0.0
            st.session_state["w_complexity"] = 0.5
            st.rerun()
    with pcols[1]:
        if st.button("More simple", use_container_width=True):
            for k in FEATURE_GROUPS:
                st.session_state[f"w_{k}"] = 0.0
            st.session_state["w_complexity"] = -0.5
            st.rerun()
    with pcols[2]:
        if st.button("More dialogue", use_container_width=True):
            for k in FEATURE_GROUPS:
                st.session_state[f"w_{k}"] = 0.0
            st.session_state["w_dialogue"] = 0.3
            st.rerun()
    with pcols[3]:
        if st.button("More folksy", use_container_width=True):
            for k in FEATURE_GROUPS:
                st.session_state[f"w_{k}"] = 0.0
            st.session_state["w_conventional"] = 0.5
            st.rerun()
    with pcols[4]:
        if st.button("Reset", use_container_width=True):
            for k in FEATURE_GROUPS:
                st.session_state[f"w_{k}"] = 0.0
            st.rerun()

    # Property sliders
    st.markdown("### Property sliders")
    st.caption("0 = original adapter · positive = more of this property · negative = less")

    scales = {}
    cols = st.columns(len(FEATURE_GROUPS))
    for col, (feat_name, info) in zip(cols, FEATURE_GROUPS.items()):
        with col:
            st.markdown(f"**{feat_name.title()}**")
            st.caption(info["description"])
            scales[feat_name] = st.slider(
                feat_name,
                min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
                key=f"w_{feat_name}",
                label_visibility="collapsed",
            )
            if author in info["good_authors"]:
                st.caption(f"*works well for {author}*")

    active = {k: v for k, v in scales.items() if abs(v) > 0.01}

    st.markdown("---")

    # Generate side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Original {author}")
        restore_original(model, author)
        with st.spinner("Generating..."):
            baseline = generate(model, tokenizer, prompt,
                                max_new=max_tokens, temp=temperature, seed=seed)
        st.markdown(f"> {baseline}")

    with col_right:
        if active:
            parts = [f"{k}: {v:+.2f}" for k, v in active.items()]
            st.subheader(f"Steered ({', '.join(parts)})")
        else:
            st.subheader("Same (no steering)")

        apply_steering(model, author, directions, scales)
        with st.spinner("Generating..."):
            steered = generate(model, tokenizer, prompt,
                               max_new=max_tokens, temp=temperature, seed=seed)
        st.markdown(f"> {steered}")

        # Restore for next run
        restore_original(model, author)

    # Explanation
    with st.expander("How is this different from activation steering?"):
        st.markdown(
            "**Activation steering** injects a vector into the residual stream "
            "every token. It fights the model's computation — a constant push.\n\n"
            "**Weight steering** modifies the LoRA weights once. The model then "
            "runs naturally. It's like the adapter was trained on a slightly "
            "different author mixture.\n\n"
            "**vs interpolation:** `(1-α)×Carroll + α×Poet` blends all "
            "properties at once. Weight steering moves along one property "
            "axis, derived from averaging many authors who share that property. "
            "'Make it more complex' vs 'make it more like Poet.'"
        )


if __name__ == "__main__":
    main()