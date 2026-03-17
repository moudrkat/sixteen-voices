#!/usr/bin/env python3
"""Streamlit app: Head Transplant Lab.

Pick a host author, a donor author, and a head — see what happens
when you graft one head's LoRA weights into another adapter.

Supports both weight-space transplant (SVD) and activation transplant.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_transplant.py
"""

import copy
from pathlib import Path

import streamlit as st
import torch

from sixteen_voices import HEAD_DIM, NUM_HEADS
from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_tokenizer as _load_tokenizer,
    get_attn_module,
    get_attn_out,
)
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate

ADAPTERS_DIR = Path("outputs/authors")

PROMPTS = [
    "Once upon a time",
    "The little girl walked into the forest",
    "It was a dark and stormy",
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


def transplant_head_weights(recipient_delta, donor_delta, head):
    result = recipient_delta.clone()
    s, e = head * HEAD_DIM, (head + 1) * HEAD_DIM
    result[s:e, :] = donor_delta[s:e, :]
    return result


def generate_activation_transplant(host_model, donor_model, tokenizer,
                                   prompt, head, seed, max_new_tokens=100):
    """Token-by-token generation with activation transplant."""
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    plen = input_ids.shape[1]

    host_attn = get_attn_module(host_model)
    donor_attn = get_attn_module(donor_model)

    for _ in range(max_new_tokens):
        donor_head_out = {}

        def capture_hook(module, args):
            h = args[0]
            s = head * HEAD_DIM
            donor_head_out["val"] = h[:, -1:, s:s + HEAD_DIM].clone()

        hook_d = donor_attn.out_proj.register_forward_pre_hook(capture_hook)
        with torch.no_grad():
            donor_model(input_ids)
        hook_d.remove()

        def replace_hook(module, args):
            h = args[0].clone()
            s = head * HEAD_DIM
            h[:, -1:, s:s + HEAD_DIM] = donor_head_out["val"]
            return (h,) + args[1:]

        hook_h = host_attn.out_proj.register_forward_pre_hook(replace_hook)
        with torch.no_grad():
            logits = host_model(input_ids).logits[:, -1, :]
        hook_h.remove()

        logits = logits / 0.8
        top_k_logits, top_k_indices = torch.topk(logits, 50)
        probs = torch.softmax(top_k_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        next_token = top_k_indices.gather(1, idx)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0][plen:], skip_special_tokens=True).strip()


def main():
    st.set_page_config(page_title="Head Transplant Lab", layout="wide")
    st.title("Head Transplant Lab")
    st.markdown(
        "Graft one author's attention head into another's adapter. "
        "See how the output changes."
    )

    authors = sorted([p.name for p in ADAPTERS_DIR.iterdir() if p.is_dir()])
    tokenizer = load_tokenizer()

    # Sidebar controls
    col1, col2, col3 = st.columns(3)
    with col1:
        host = st.selectbox("Host (keeps structure)", authors,
                            index=authors.index("carroll") if "carroll" in authors else 0)
    with col2:
        donor = st.selectbox("Donor (provides head)", authors,
                             index=authors.index("poe") if "poe" in authors else min(1, len(authors) - 1))
    with col3:
        head = st.selectbox("Head to transplant",
                            [f"H{i}" for i in range(NUM_HEADS)],
                            index=14)
        head_idx = int(head[1:])

    col4, col5, col6 = st.columns(3)
    with col4:
        prompt = st.selectbox("Prompt", PROMPTS, index=2)
    with col5:
        seed = st.number_input("Seed", value=42, min_value=0, max_value=999)
    with col6:
        method = st.radio("Method", ["Both", "Weight (SVD)", "Activation"],
                          horizontal=True)

    custom_prompt = st.text_input("Or type your own prompt:")
    if custom_prompt:
        prompt = custom_prompt

    if host == donor:
        st.warning("Host and donor are the same — transplant won't change anything!")

    if st.button("Transplant!", type="primary"):
        # Pure host
        with st.spinner(f"Generating pure {host}..."):
            host_model = load_model(host)
            pure_text = generate(host_model, tokenizer, prompt, seed=seed,
                                 max_new_tokens=100)

        results = {}

        # Weight transplant
        if method in ["Both", "Weight (SVD)"]:
            with st.spinner("Weight transplant (SVD)..."):
                h_deltas = load_deltas(host)
                d_deltas = load_deltas(donor)
                t_deltas = {}
                for proj in ["q_proj", "v_proj"]:
                    t_deltas[proj] = transplant_head_weights(
                        h_deltas[proj], d_deltas[proj], head_idx)

                # Deep copy cached model to inject into
                w_model = copy.deepcopy(host_model)
                attn = get_attn_module(w_model)
                for proj in ["q_proj", "v_proj"]:
                    A, B = delta_to_AB(t_deltas[proj])
                    getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
                    getattr(attn, proj).lora_B["default"].weight.data.copy_(B)

                weight_text = generate(w_model, tokenizer, prompt, seed=seed,
                                       max_new_tokens=100)
                results["Weight (SVD)"] = weight_text
                del w_model

        # Activation transplant
        if method in ["Both", "Activation"]:
            with st.spinner("Activation transplant..."):
                donor_model = load_model(donor)
                act_text = generate_activation_transplant(
                    host_model, donor_model, tokenizer, prompt,
                    head_idx, seed, max_new_tokens=100)
                results["Activation"] = act_text

        # Display
        st.markdown("---")

        if method == "Both":
            cols = st.columns(3)
            with cols[0]:
                st.subheader(f"Pure {host.capitalize()}")
                st.markdown(f"*{pure_text}*")
            with cols[1]:
                st.subheader(f"Weight transplant")
                st.caption(f"{host} + {donor}'s {head} (SVD, lossy)")
                st.markdown(f"*{results['Weight (SVD)']}*")
            with cols[2]:
                st.subheader(f"Activation transplant")
                st.caption(f"{host} + {donor}'s {head} (exact)")
                st.markdown(f"*{results['Activation']}*")
        else:
            cols = st.columns(2)
            with cols[0]:
                st.subheader(f"Pure {host.capitalize()}")
                st.markdown(f"*{pure_text}*")
            with cols[1]:
                key = list(results.keys())[0]
                st.subheader(f"{key} transplant")
                st.caption(f"{host} + {donor}'s {head}")
                st.markdown(f"*{results[key]}*")

        st.markdown("---")
        st.caption(
            f"Prompt: \"{prompt}\" · seed={seed} · "
            f"TinyStories-1Layer-21M · LoRA rank 8"
        )


if __name__ == "__main__":
    main()