#!/usr/bin/env python3
"""Streamlit app: Interactive LoRA & Multi-Head Attention Explainer.

A visual, educational walkthrough of how LoRA adapts a transformer and
how attention heads specialize per author.

Usage:
    streamlit run demos/app_explainer.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from sixteen_voices import (
    HEAD_DIM, HIDDEN_DIM, NUM_HEADS, RANK,
    get_attn_out, load_adapted_model, load_base_model, load_tokenizer,
    make_hook,
)

ADAPTERS_DIR = Path("outputs/authors")


# ── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def get_tokenizer():
    return load_tokenizer()


@st.cache_data
def list_authors():
    return sorted(
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_model.safetensors").exists()
    )


@st.cache_resource
def get_base_model():
    return load_base_model()


@st.cache_resource
def get_adapted_model(author):
    return load_adapted_model(str(ADAPTERS_DIR / author / "adapter"))


def get_delta_W(model, proj="v_proj"):
    """Extract merged ΔW = B @ A for a given projection."""
    from sixteen_voices.model import get_attn_module
    attn = get_attn_module(model)
    lora = getattr(attn, proj)
    A = lora.lora_A["default"].weight.data  # (rank, in)
    B = lora.lora_B["default"].weight.data  # (out, rank)
    scaling = lora.scaling["default"]
    return (B @ A * scaling).cpu().numpy()


def generate_text(model, tokenizer, prompt, head_scales=None,
                  max_new=80, temp=0.7, seed=42):
    if seed > 0:
        torch.manual_seed(seed)
    hook = None
    if head_scales and any(s != 1.0 for s in head_scales.values()):
        hook = get_attn_out(model).register_forward_pre_hook(make_hook(head_scales))
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


# ── Page config ─────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LoRA Explainer — Sixteen Voices",
        layout="wide",
    )

    pages = [
        "1. The Model",
        "2. What is LoRA?",
        "3. Multi-Head Attention",
        "4. ΔW: What LoRA Actually Changes",
        "5. Head Playground",
    ]

    with st.sidebar:
        st.title("LoRA Explainer")
        st.caption("An interactive guide to LoRA and attention heads")
        page = st.radio("Section", pages)

    if page == pages[0]:
        page_model()
    elif page == pages[1]:
        page_lora()
    elif page == pages[2]:
        page_multihead()
    elif page == pages[3]:
        page_delta()
    elif page == pages[4]:
        page_playground()


# ═════════════════════════════════════════════════════════════════════
# PAGE 1: The Model
# ═════════════════════════════════════════════════════════════════════

def page_model():
    st.header("The Model: TinyStories-1Layer-21M")

    st.markdown("""
    We use the smallest interesting transformer we could find:

    | | |
    |---|---|
    | **Architecture** | GPT-Neo (autoregressive, decoder-only) |
    | **Layers** | 1 |
    | **Attention heads** | 16 |
    | **Hidden dimension** | 1024 (= 16 heads × 64 dims each) |
    | **Vocabulary** | 50,257 tokens |
    | **Total parameters** | ~21 million |
    | **Trained on** | TinyStories — simple children's stories |

    """)

    st.markdown("### Why so small?")
    st.markdown("""
    Because we want to **see everything**. With one layer and 16 heads,
    there's nowhere for the model to hide. Every computation is one step:

    $$\\text{input} \\xrightarrow{\\text{16 heads in parallel}} \\text{output}$$

    No deep circuits, no residual stream accumulation across layers. Just
    16 heads, each getting one shot at the input.
    """)

    st.markdown("### What can it do?")
    st.info("""
    This model generates simple children's stories. Its entire world is
    "Once upon a time, Lily went to the park." Don't expect Lovecraft.
    """)

    tokenizer = get_tokenizer()
    prompt = st.text_input("Try it yourself:", "Once upon a time")
    if st.button("Generate (base model)"):
        model = get_base_model()
        with st.spinner("Generating..."):
            text = generate_text(model, tokenizer, prompt)
        st.markdown(f"**{prompt}**{text[len(prompt):]}")


# ═════════════════════════════════════════════════════════════════════
# PAGE 2: What is LoRA?
# ═════════════════════════════════════════════════════════════════════

def page_lora():
    st.header("What is LoRA?")

    st.markdown("""
    ### The problem

    Fine-tuning means updating the model's weight matrices. Our attention
    projections are **1024 × 1024** matrices — that's **1,048,576
    parameters** each. For a small model that's fine, but the same
    technique on a large model means billions of parameters to update.

    ### The trick

    Instead of updating the full matrix, LoRA adds a **low-rank bypass**:
    """)

    st.latex(r"h = W \cdot x + \underbrace{B \cdot A}_{=\,\Delta W} \cdot x")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Matrix | Shape | Parameters | Status |
        |--------|-------|-----------|--------|
        | **W** | 1024 × 1024 | 1,048,576 | frozen |
        | **A** | 8 × 1024 | 8,192 | trained |
        | **B** | 1024 × 8 | 8,192 | trained |
        """)
    with col2:
        st.markdown(f"""
        **Bottleneck rank:** {RANK}

        **Trainable params per projection:**
        {RANK * HIDDEN_DIM * 2:,} (A + B)

        **Two projections (Q + V):**
        {RANK * HIDDEN_DIM * 2 * 2:,} total

        **That's {RANK * HIDDEN_DIM * 2 * 2 / (HIDDEN_DIM * HIDDEN_DIM * 2) * 100:.2f}%**
        of the full weight matrices.
        """)

    st.markdown("### The intuition")
    st.markdown("""
    Think of it as a detour. The main highway (W) stays as it is. LoRA
    builds a **narrow side road** (A compresses 1024 dims → 8, then B
    expands 8 → 1024) that nudges the output in a new direction.

    The side road is very narrow (rank 8), so it can only make
    low-dimensional adjustments — but that turns out to be enough to shift
    the model's vocabulary and patterns toward a specific author.
    """)

    st.markdown("### The effective weight change")
    st.latex(r"\Delta W = B \cdot A \quad \text{(1024 × 1024, but rank 8)}")
    st.markdown("""
    After training, you can multiply B × A to get ΔW — a full-sized
    matrix that shows *exactly* what LoRA changed. This is what we
    dissect in the next sections.
    """)

    # Interactive rank explorer
    st.markdown("---")
    st.markdown("### Play with the numbers")
    st.caption("This is just a calculator — our adapters are all rank 8. "
               "Change the rank to see how it affects parameter count.")
    rank = st.slider("LoRA rank (hypothetical)", 1, 64, RANK)
    hidden = HIDDEN_DIM
    n_proj = 2  # Q and V
    trainable = rank * hidden * 2 * n_proj
    full = hidden * hidden * n_proj
    st.metric("Trainable parameters", f"{trainable:,}")
    st.metric("Full fine-tuning would be", f"{full:,}")
    st.metric("Compression ratio", f"{trainable / full * 100:.2f}%")

    if rank == 1:
        st.caption("Rank 1 = each projection change is a single vector outer product. Very constrained.")
    elif rank >= 32:
        st.caption("High rank = more expressive, but also more parameters. Diminishing returns.")


# ═════════════════════════════════════════════════════════════════════
# PAGE 3: Multi-Head Attention
# ═════════════════════════════════════════════════════════════════════

def page_multihead():
    st.header("Multi-Head Attention")

    st.markdown(f"""
    ### One input, {NUM_HEADS} parallel views

    The input (a {HIDDEN_DIM}-dimensional vector) is projected into
    queries (Q), keys (K), and values (V), each {HIDDEN_DIM} dimensions.
    Then something crucial happens: **each projection is split into
    {NUM_HEADS} chunks of {HEAD_DIM} dimensions**.
    """)

    st.latex(
        r"Q = W_Q \cdot x \in \mathbb{R}^{1024}"
        r"\quad\longrightarrow\quad"
        r"Q_0 \in \mathbb{R}^{64},\;"
        r"Q_1 \in \mathbb{R}^{64},\;"
        r"\ldots,\;"
        r"Q_{15} \in \mathbb{R}^{64}"
    )

    st.markdown("""
    Each head independently computes attention over its own 64 dimensions:
    """)

    st.latex(
        r"\text{head}_h = \text{softmax}\!\left(\frac{Q_h \cdot K_h^\top}{\sqrt{64}}\right) \cdot V_h"
    )

    st.markdown(f"""
    Then all {NUM_HEADS} outputs are concatenated and projected back:
    """)

    st.latex(
        r"\text{output} = W_O \cdot \text{concat}(\text{head}_0, \text{head}_1, \ldots, \text{head}_{15})"
    )

    st.markdown("""
    ### Why this matters for LoRA

    Each head "owns" 64 rows of the weight matrix. LoRA modifies the full
    1024×1024 matrix, but the change can be **sliced by head**:

    - Rows 0–63 → Head 0's change
    - Rows 64–127 → Head 1's change
    - ...
    - Rows 960–1023 → Head 15's change

    This means we can ask: **how much did LoRA change each head?** And we
    can surgically remove one head's change to see what happens.
    """)

    st.markdown("### The key dimensions")
    st.markdown(f"""
    | | |
    |---|---|
    | Input dimension | {HIDDEN_DIM} |
    | Number of heads | {NUM_HEADS} |
    | Dimensions per head | {HEAD_DIM} |
    | {NUM_HEADS} × {HEAD_DIM} | = {NUM_HEADS * HEAD_DIM} ✓ |
    """)


# ═════════════════════════════════════════════════════════════════════
# PAGE 4: ΔW — What LoRA Actually Changes
# ═════════════════════════════════════════════════════════════════════

def page_delta():
    st.header("ΔW: What LoRA Actually Changes")

    authors = list_authors()
    if not authors:
        st.warning("No trained adapters found. Run `make train` first.")
        return

    st.markdown("""
    ΔW = B·A is the effective weight change that LoRA makes. It's a
    1024×1024 matrix, but with only rank 8 — meaning the change lives in
    an 8-dimensional subspace. Let's look at it.
    """)

    col1, col2 = st.columns(2)
    with col1:
        author1 = st.selectbox("Author A", authors, index=0)
    with col2:
        idx2 = min(1, len(authors) - 1)
        author2 = st.selectbox("Author B", authors, index=idx2)

    proj = st.radio("Projection", ["v_proj", "q_proj"], horizontal=True)

    model1 = get_adapted_model(author1)
    model2 = get_adapted_model(author2)

    dw1 = get_delta_W(model1, proj)
    dw2 = get_delta_W(model2, proj)

    # Global color scale
    vmax = max(np.abs(dw1).max(), np.abs(dw2).max())

    import matplotlib.pyplot as plt

    st.markdown("### Full ΔW matrices")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(dw1, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_title(f"{author1} — {proj}")
    ax1.set_xlabel("input dim")
    ax1.set_ylabel("output dim (rows = heads)")
    for i in range(1, NUM_HEADS):
        ax1.axhline(i * HEAD_DIM - 0.5, color="black", lw=0.3, alpha=0.5)

    im = ax2.imshow(dw2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_title(f"{author2} — {proj}")
    ax2.set_xlabel("input dim")
    for i in range(1, NUM_HEADS):
        ax2.axhline(i * HEAD_DIM - 0.5, color="black", lw=0.3, alpha=0.5)
    fig.colorbar(im, ax=[ax1, ax2], shrink=0.8, label="weight change")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    The thin horizontal lines show head boundaries (every 64 rows).
    Notice how different authors have different patterns — some heads
    are heavily modified, others barely touched.
    """)

    # Per-head norm comparison
    st.markdown("### Per-head change magnitude")
    st.markdown("""
    How much did each head's weights change? We measure the Frobenius
    norm (total magnitude) of each head's 64×1024 block in ΔW:
    """)
    st.latex(r"\|\Delta W_h\|_F = \sqrt{\sum_{i,j} (\Delta W_h)_{ij}^2}")

    norms1 = [np.linalg.norm(dw1[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(NUM_HEADS)]
    norms2 = [np.linalg.norm(dw2[h * HEAD_DIM:(h + 1) * HEAD_DIM]) for h in range(NUM_HEADS)]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(NUM_HEADS)
    w = 0.35
    ax.bar(x - w / 2, norms1, w, label=author1, color="#e6194b", alpha=0.8)
    ax.bar(x + w / 2, norms2, w, label=author2, color="#4363d8", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{i}" for i in range(NUM_HEADS)])
    ax.set_ylabel("‖ΔW_h‖_F")
    ax.set_title(f"Per-head weight change magnitude ({proj})")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    If two authors have very different bar charts, their adapters are
    "wiring" the heads differently. If a head has a tall bar for one
    author and a short bar for another, that head is more important for
    the first author.

    (Norm ≠ importance — a large change could be in an unimportant
    direction. But it's a first approximation.)
    """)


# ═════════════════════════════════════════════════════════════════════
# PAGE 5: Head Playground
# ═════════════════════════════════════════════════════════════════════

def page_playground():
    st.header("Head Playground")

    st.markdown("""
    Sixteen sliders, one per head. Scale each head's output from 0
    (silenced) to 2 (amplified). Watch how the generated text changes.

    The model computes attention as usual, but before concatenation, each
    head's 64-dim output is multiplied by its scale factor:
    """)
    st.latex(
        r"\text{head}_h^{\text{steered}} = s_h \cdot \text{head}_h"
        r"\qquad s_h \in [0, 2]"
    )

    authors = list_authors()
    if not authors:
        st.warning("No trained adapters found. Run `make train` first.")
        return

    with st.sidebar:
        author = st.selectbox("Author", authors,
                              index=authors.index("barrie") if "barrie" in authors else 0)
        prompt = st.text_input("Prompt", "Once upon a time")
        max_tokens = st.slider("Max tokens", 40, 200, 80)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.05)
        seed = st.number_input("Seed (0=random)", min_value=0, value=42)

        st.markdown("---")
        preset = st.selectbox("Presets", [
            "All on (default)",
            "Only head 0",
            "Only head 15",
            "Kill even heads",
            "Kill odd heads",
            "All off (silence)",
        ])

    # Presets
    defaults = {
        "All on (default)": {h: 1.0 for h in range(NUM_HEADS)},
        "Only head 0": {h: (1.0 if h == 0 else 0.0) for h in range(NUM_HEADS)},
        "Only head 15": {h: (1.0 if h == 15 else 0.0) for h in range(NUM_HEADS)},
        "Kill even heads": {h: (0.0 if h % 2 == 0 else 1.0) for h in range(NUM_HEADS)},
        "Kill odd heads": {h: (1.0 if h % 2 == 0 else 0.0) for h in range(NUM_HEADS)},
        "All off (silence)": {h: 0.0 for h in range(NUM_HEADS)},
    }

    preset_scales = defaults[preset]

    st.markdown("### Head scales")
    cols = st.columns(4)
    scales = {}
    for h in range(NUM_HEADS):
        with cols[h % 4]:
            scales[h] = st.slider(
                f"H{h}", 0.0, 2.0,
                value=preset_scales.get(h, 1.0),
                step=0.1, key=f"head_{h}",
            )

    active = sum(1 for s in scales.values() if s > 0.01)
    silenced = NUM_HEADS - active
    boosted = sum(1 for s in scales.values() if s > 1.01)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Active heads", f"{active}/{NUM_HEADS}")
    mcol2.metric("Silenced", silenced)
    mcol3.metric("Boosted (>1.0)", boosted)

    if st.button("Generate", type="primary", use_container_width=True):
        tokenizer = get_tokenizer()
        model = get_adapted_model(author)

        with st.spinner("Generating with full model..."):
            full_text = generate_text(model, tokenizer, prompt,
                                      max_new=max_tokens, temp=temperature,
                                      seed=seed)

        with st.spinner("Generating with steered heads..."):
            steered_text = generate_text(model, tokenizer, prompt,
                                         head_scales=scales,
                                         max_new=max_tokens, temp=temperature,
                                         seed=seed)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### All heads = 1.0")
            st.markdown(f"**{prompt}**{full_text[len(prompt):]}")
        with col2:
            st.markdown("#### Your configuration")
            st.markdown(f"**{prompt}**{steered_text[len(prompt):]}")

        # Show which heads are changed
        changed = [f"H{h}={scales[h]:.1f}" for h in range(NUM_HEADS) if abs(scales[h] - 1.0) > 0.01]
        if changed:
            st.caption(f"Modified heads: {', '.join(changed)}")
        else:
            st.caption("No heads modified — both outputs use the same configuration.")


if __name__ == "__main__":
    main()