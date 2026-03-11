#!/usr/bin/env python3
"""Streamlit app: Head Knockout Challenge.

How many of the 16 attention heads can you remove before the story collapses?
Pick an author adapter, then knock out heads one by one.

Usage:
    pip install -e ".[demo]"
    streamlit run demos/app_steer.py
"""

from pathlib import Path

import streamlit as st
import torch

from sixteen_voices import HEAD_DIM, NUM_HEADS
from sixteen_voices.model import load_adapted_model as _load_adapted, load_base_model as _load_base
from sixteen_voices.model import load_tokenizer as _load_tokenizer, get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")

PROMPTS = [
    "Once upon a time",
    "The little girl walked into the forest",
    "There was a king who had",
    "In the dark of night",
    "It was a dark and stormy",
    "The princess smiled and",
]

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)

HEAD_NAMES = {
    3: "H3 (style)", 11: "H11 (structural)",
    13: "H13 (clustering)", 14: "H14 (familiarity)",
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
def load_model(adapter_path):
    return _load_adapted(adapter_path)


@st.cache_resource
def load_base():
    return _load_base()


def compute_ppl(model, tokenizer, text, head_scales=None):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    hook = None
    if head_scales:
        hook = get_attn_out(model).register_forward_hook(make_hook(head_scales))
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
        hook = get_attn_out(model).register_forward_hook(make_hook(head_scales))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=temp,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if hook:
        hook.remove()
    return text


def init_state():
    for key, default in [
        ("alive_heads", set(range(NUM_HEADS))),
        ("kill_order", []),
        ("ppl_history", []),
        ("game_started", False),
        ("last_killed", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def reset():
    st.session_state.alive_heads = set(range(NUM_HEADS))
    st.session_state.kill_order = []
    st.session_state.ppl_history = []
    st.session_state.game_started = False
    st.session_state.last_killed = None


def head_name(h):
    return HEAD_NAMES.get(h, f"H{h}")


def main():
    st.set_page_config(page_title="Head Knockout Challenge", layout="wide")
    init_state()
    tokenizer = load_tokenizer()

    st.title("Head Knockout Challenge")
    st.markdown("**16 attention heads. Kill them one by one. "
                "How many can you remove before the story falls apart?**")

    with st.sidebar:
        st.header("Setup")
        mode = st.radio("Model", ["Author LoRA", "Base model"])
        authors = list_authors()
        author = (st.selectbox("Author", authors,
                               index=authors.index("grimm") if "grimm" in authors else 0)
                  if mode == "Author LoRA" else None)
        prompt = st.selectbox("Prompt", PROMPTS)
        custom = st.text_input("Or custom prompt")
        if custom.strip():
            prompt = custom.strip()
        max_tokens = st.slider("Max tokens", 40, 200, 120)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
        seed = st.number_input("Seed (0=random)", min_value=0, value=42)
        if st.button("Reset game", use_container_width=True):
            reset()
            st.rerun()

    with st.spinner("Loading model..."):
        model = load_model(str(ADAPTERS_DIR / author / "adapter")) if author else load_base()

    n_alive = len(st.session_state.alive_heads)
    st.markdown(f"### Heads alive: {n_alive}/16 | Killed: {NUM_HEADS - n_alive}")

    cols = st.columns(8)
    kill = None
    for h in range(NUM_HEADS):
        with cols[h % 8]:
            if h in st.session_state.alive_heads:
                if st.button(f"🟢 {head_name(h)}", key=f"k_{h}", use_container_width=True):
                    kill = h
            else:
                st.button(f"💀 {head_name(h)}", key=f"d_{h}", use_container_width=True, disabled=True)

    if kill is not None:
        st.session_state.alive_heads.discard(kill)
        st.session_state.kill_order.append(kill)
        st.session_state.last_killed = kill
        st.session_state.game_started = True
        st.rerun()

    if st.button("Generate story", type="primary", use_container_width=True) or st.session_state.game_started:
        scales = {h: (1.0 if h in st.session_state.alive_heads else 0.0)
                  for h in range(NUM_HEADS)}

        with st.spinner("Computing..."):
            ppl = compute_ppl(model, tokenizer, EVAL_TEXT, scales)
            if not st.session_state.ppl_history:
                baseline = compute_ppl(model, tokenizer, EVAL_TEXT)
                st.session_state.ppl_history.append(("baseline", baseline))
            else:
                baseline = st.session_state.ppl_history[0][1]
            if st.session_state.kill_order:
                st.session_state.ppl_history.append(
                    (f"-{head_name(st.session_state.kill_order[-1])}", ppl))

        ratio = ppl / baseline
        if ratio < 1.5:
            label, color = "HEALTHY", "green"
        elif ratio < 2.0:
            label, color = "DEGRADED", "orange"
        elif ratio < 4.0:
            label, color = "CRITICAL", "red"
        else:
            label, color = "COLLAPSED", "red"

        health = max(0, min(100, int(100 / ratio)))
        st.markdown(f"### Model health: :{color}[{label}] ({health}%)")
        st.progress(health / 100)
        st.metric("Perplexity", f"{ppl:.0f}", f"{ratio:.1f}x baseline")

        with st.spinner("Generating..."):
            text = generate(model, tokenizer, prompt,
                            max_new=max_tokens, temp=temperature,
                            seed=seed, head_scales=scales)
        st.markdown(f"**{prompt}**{text[len(prompt):]}")


if __name__ == "__main__":
    main()
