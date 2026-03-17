#!/usr/bin/env python3
"""Streamlit app: Head Knockout Lab.

Pick an author, see which heads matter most, toggle them on/off,
watch the text change.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_steer.py
"""

import json
from pathlib import Path

import streamlit as st
import torch

from sixteen_voices import HEAD_DIM, NUM_HEADS
from sixteen_voices.model import load_adapted_model as _load_adapted, load_base_model as _load_base
from sixteen_voices.model import load_tokenizer as _load_tokenizer, get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
KNOCKOUT_JSON = Path("outputs/knockout_all_heads.json")

PROMPTS = [
    "It was a dark and stormy",
    "Once upon a time",
    "The little girl walked into the forest",
    "There was a king who had",
    "In the dark of night",
    "The princess smiled and",
]

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)


@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_data
def list_authors():
    return sorted(
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_model.safetensors").exists()
    )


@st.cache_data
def load_knockout_data():
    if KNOCKOUT_JSON.exists():
        with open(KNOCKOUT_JSON) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_model(adapter_path):
    return _load_adapted(adapter_path)


@st.cache_resource
def load_base():
    return _load_base()


def compute_ppl(model, tokenizer, text, head_scales=None):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    hook = None
    if head_scales:
        hook = get_attn_out(model).register_forward_pre_hook(make_hook(head_scales))
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    if hook:
        hook.remove()
    return torch.exp(out.loss).item()


def generate(model, tokenizer, prompt, max_new=120, temp=0.7, seed=42, head_scales=None):
    if seed > 0:
        torch.manual_seed(seed)
    hook = None
    if head_scales and any(s != 1.0 for s in head_scales.values()):
        hook = get_attn_out(model).register_forward_pre_hook(make_hook(head_scales))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=temp,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    if hook:
        hook.remove()
    return text


def recovery_color(score):
    """Return emoji + color hint based on recovery score."""
    if score > 0.4:
        return "🔴", "red"    # very important
    elif score > 0.2:
        return "🟠", "orange"  # moderately important
    elif score > 0.05:
        return "🟡", "yellow"  # slightly important
    elif score > -0.05:
        return "⚪", "gray"    # negligible
    else:
        return "🔵", "blue"    # hurts (negative recovery)


def main():
    st.set_page_config(page_title="Head Knockout Lab", layout="wide")
    tokenizer = load_tokenizer()
    knockout_data = load_knockout_data()

    st.title("Head Knockout Lab")

    # Sidebar
    with st.sidebar:
        st.header("Setup")
        authors = list_authors()
        author = st.selectbox("Author", authors,
                              index=authors.index("poe") if "poe" in authors else 0)
        prompt = st.selectbox("Prompt", PROMPTS)
        custom = st.text_input("Or custom prompt")
        if custom.strip():
            prompt = custom.strip()
        max_tokens = st.slider("Max tokens", 40, 200, 120)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
        seed = st.number_input("Seed (0=random)", min_value=0, value=42)

        pass  # explanation moved to main area expander

    # Load model
    with st.spinner("Loading model..."):
        model = load_model(str(ADAPTERS_DIR / author / "adapter"))

    # Show head importance from knockout data
    if knockout_data and author in knockout_data:
        recovery = knockout_data[author]["head_recovery"]

        # Architecture explainer
        with st.expander("Model architecture"):
            arch_path = Path("figures/architecture.png")
            if arch_path.exists():
                st.image(str(arch_path), use_container_width=True)
            st.markdown(
                "**TinyStories-1Layer-21M.** One attention layer with 16 heads "
                "(64 dimensions each). Each adapter adds a LoRA bypass "
                "(rank 8) to the Q and V projections — ~33k trainable "
                "parameters (0.15% of the model). We trained 77 adapters, "
                "one per author."
            )

        st.markdown("### Head importance for " + author.capitalize())

        # Bar chart of this author's head recovery
        import matplotlib.pyplot as plt
        import numpy as np

        scores = [recovery[f"H{h}"] for h in range(NUM_HEADS)]
        fig, ax = plt.subplots(figsize=(10, 2.2))
        colors = []
        for s in scores:
            if s > 0.4:
                colors.append("#ef4444")
            elif s > 0.2:
                colors.append("#f97316")
            elif s > 0.05:
                colors.append("#eab308")
            elif s > -0.05:
                colors.append("#9ca3af")
            else:
                colors.append("#3b82f6")
        ax.bar(range(NUM_HEADS), scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(NUM_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(NUM_HEADS)], fontsize=9)
        ax.set_ylabel("Recovery", fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.set_title(f"How much does each head contribute to {author.capitalize()}'s style?",
                     fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🔴 critical  🟠 important  🟡 mild  ⚪ negligible  🔵 hurts")

        with st.expander("How is this measured?"):
            st.markdown(
                "Each LoRA adapter adds a weight change **ΔW = B·A** to the "
                "frozen model. This matrix has 1024 rows — 64 per head. "
                "We can zero out all rows except one head's block to isolate "
                "its contribution.\n\n"
                "Then we measure **perplexity** (how surprised the model is "
                "by the author's text) and compute a recovery score:"
            )
            st.latex(
                r"\text{recovery}_h = "
                r"\frac{\text{PPL}_{\text{base}} - \text{PPL}_h}"
                r"{\text{PPL}_{\text{base}} - \text{PPL}_{\text{full}}}"
            )
            st.markdown(
                "Where:\n"
                "- **PPL_base** = perplexity with no adapter (the raw model)\n"
                "- **PPL_full** = perplexity with all 16 heads active\n"
                "- **PPL_h** = perplexity with only head *h* active\n\n"
                "| Score | Meaning |\n"
                "|-------|---------|\n"
                "| **1.0** | This head alone recovers the full adaptation |\n"
                "| **0.0** | This head contributes nothing |\n"
                "| **< 0** | This head alone makes things *worse* than no adapter |\n\n"
                "The scores come from 1,232 experiments (77 authors × 16 "
                "heads). Try toggling heads below — kill an important one "
                "(🔴) and watch the style collapse!"
            )
            strip_path = Path("figures/knockout_strip.png")
            if strip_path.exists():
                st.markdown("**Recovery across all 77 authors:**")
                st.image(str(strip_path), use_container_width=True)

        # Sort by importance for display
        sorted_heads = sorted(range(NUM_HEADS),
                              key=lambda h: recovery[f"H{h}"], reverse=True)
        best_head = sorted_heads[0]
        worst_head = sorted_heads[-1]

        # Initialize head scales
        if "head_scales" not in st.session_state or st.session_state.get("last_author") != author:
            st.session_state.head_scales = {h: 1.0 for h in range(NUM_HEADS)}
            st.session_state.last_author = author

        # Steering sliders — two rows of 8
        st.markdown("### Steering sliders")
        st.caption("0× = killed · 1× = normal · 2× = amplified")
        for row in range(2):
            cols = st.columns(8)
            for col_idx in range(8):
                h = row * 8 + col_idx
                score = recovery[f"H{h}"]
                emoji, _ = recovery_color(score)
                with cols[col_idx]:
                    st.session_state.head_scales[h] = st.slider(
                        f"{emoji} H{h} ({score:+.2f})",
                        min_value=0.0, max_value=2.0,
                        value=st.session_state.head_scales[h],
                        step=0.1, key=f"slider_{h}",
                    )

        # Quick presets
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            if st.button("Kill all except best", use_container_width=True):
                for h in range(NUM_HEADS):
                    st.session_state.head_scales[h] = 2.0 if h == best_head else 0.0
                st.rerun()
        with col_b:
            if st.button("Kill worst", use_container_width=True):
                for h in range(NUM_HEADS):
                    st.session_state.head_scales[h] = 0.0 if h == worst_head else 1.0
                st.rerun()
        with col_c:
            if st.button("Top 4 at 2×", use_container_width=True):
                top4 = set(sorted_heads[:4])
                for h in range(NUM_HEADS):
                    st.session_state.head_scales[h] = 2.0 if h in top4 else 0.0
                st.rerun()
        with col_d:
            if st.button("All to zero", use_container_width=True):
                for h in range(NUM_HEADS):
                    st.session_state.head_scales[h] = 0.0
                st.rerun()
        with col_e:
            if st.button("Reset all (1×)", use_container_width=True):
                for h in range(NUM_HEADS):
                    st.session_state.head_scales[h] = 1.0
                st.rerun()

        # Status
        modified = {h: s for h, s in st.session_state.head_scales.items() if s != 1.0}
        if modified:
            parts = [f"H{h}={s:.1f}×" for h, s in sorted(modified.items())]
            st.markdown(f"**Modified:** {', '.join(parts)}")
        else:
            st.markdown("**All heads at 1× (normal)**")
    else:
        st.session_state.head_scales = {h: 1.0 for h in range(NUM_HEADS)}

    st.markdown("---")

    # Generate
    scales = dict(st.session_state.head_scales)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Full adapter (all heads at 1×)")
        with st.spinner("Generating..."):
            full_text = generate(model, tokenizer, prompt,
                                max_new=max_tokens, temp=temperature, seed=seed)
            full_ppl = compute_ppl(model, tokenizer, EVAL_TEXT)
        st.markdown(f"*{full_text}*")
        st.caption(f"PPL: {full_ppl:.0f}")

    with col_right:
        any_changed = any(s != 1.0 for s in scales.values())
        st.subheader("Steered" if any_changed else "Same (no changes)")
        with st.spinner("Generating..."):
            ko_text = generate(model, tokenizer, prompt,
                               max_new=max_tokens, temp=temperature,
                               seed=seed, head_scales=scales)
            ko_ppl = compute_ppl(model, tokenizer, EVAL_TEXT, scales)
        st.markdown(f"*{ko_text}*")
        ppl_change = (ko_ppl - full_ppl) / full_ppl * 100
        color = "red" if ppl_change > 50 else "orange" if ppl_change > 20 else "green"
        st.caption(f"PPL: {ko_ppl:.0f} (:{color}[{ppl_change:+.0f}%])")


if __name__ == "__main__":
    main()