#!/usr/bin/env python3
"""Streamlit app: Poster companion, Kill a Head.

One author. One tap. See what happens when the dominant head is killed.

Mobile-first: no sliders, no head picker. Just the demo.

Usage:
    streamlit run demos/app_poster_steer.py
"""

import sys
from pathlib import Path

# Ensure the package is importable when running from Streamlit Cloud
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import streamlit as st

from sixteen_voices.model import (
    load_adapted_model as _load_adapted,
    load_tokenizer as _load_tokenizer,
    get_attn_out,
)
from sixteen_voices.steering import make_hook, generate as steer_generate

ADAPTERS_DIR = Path("outputs/authors")

SEED = 42
PROMPT = "It was a dark and stormy"

# Two authors, one H11-led, one H14-led. Matches poster Q2.
AUTHORS = {
    "carroll": ("Carroll", 11, "H11-led. Playful, Victorian Wonderland."),
    "poe":     ("Poe",     14, "H14-led. Dark, elevated register."),
}


# ── Cached loaders ──────────────────────────────────────────────────

@st.cache_resource
def load_tokenizer():
    return _load_tokenizer()


@st.cache_resource
def load_model(author):
    return _load_adapted(str(ADAPTERS_DIR / author / "adapter"))


# ── Core ──────────────────────────────────────────────────────────────

def generate_with_kill(model, tokenizer, prompt, head_to_kill,
                       seed=SEED, max_new=80):
    """Generate text with one head's output zeroed. head_to_kill=None → baseline."""
    head_scales = {head_to_kill: 0.0} if head_to_kill is not None else None
    attn_out = get_attn_out(model) if head_scales else None
    return steer_generate(
        model, tokenizer, prompt,
        head_scales=head_scales,
        attn_out=attn_out,
        seed=seed,
        max_new_tokens=max_new,
    )


# ── Main app ────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Sixteen Voices — Kill a Head",
        page_icon="~",
        layout="centered",
    )

    st.title("Kill a Head")
    st.caption(
        "Pick an author. Tap Generate. Compare what the model writes "
        "when the dominant head is on, off, or when a different head "
        "is off instead."
    )

    st.markdown(
        "**What's happening.** Each of these fine-tuned authors relies "
        "on one attention head (out of 16) to carry most of its style. "
        "For **Carroll** it's H11. For **Poe** it's H14. The button "
        "runs the same prompt three times: once with all heads on, once "
        "with the dominant head zeroed out at inference time, and once "
        "with a different strong head zeroed out for contrast. "
        "Killing the right head should wreck the voice. Killing the wrong "
        "one should barely matter.\n\n"
        f"Output is deterministic (seed = {SEED}), so the same prompt "
        "gives the same text each time. If you want variation, change "
        "the prompt.\n\n"
        "Part of [Sixteen Voices](https://github.com/moudrkat/sixteen-voices)."
    )

    tokenizer = load_tokenizer()

    # ── Author ──────────────────────────────────────────────────────
    author = st.selectbox(
        "Author",
        list(AUTHORS.keys()),
        index=0,
        format_func=lambda k: AUTHORS[k][0],
    )
    label, dominant, blurb = AUTHORS[author]
    st.caption(blurb)

    # ── Prompt (fixed, matches the poster) ──────────────────────────
    st.markdown(f"**Prompt:** *\u201c{PROMPT}\u201d*")

    # ── Generate ────────────────────────────────────────────────────
    if st.button("Generate", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            model = load_model(author)

        with st.spinner("Baseline..."):
            text_base = generate_with_kill(model, tokenizer, PROMPT, None)

        with st.spinner(f"Killing H{dominant} (the dominant head)..."):
            text_kill_dom = generate_with_kill(model, tokenizer, PROMPT, dominant)

        other = 14 if dominant == 11 else 11
        with st.spinner(f"Killing H{other} (the other strong head, for contrast)..."):
            text_kill_other = generate_with_kill(model, tokenizer, PROMPT, other)

        # ── Output ──────────────────────────────────────────────────
        st.markdown("---")

        st.markdown(f"#### {label}, natural (all 16 heads on)")
        st.markdown(f"> {text_base}")

        st.markdown(f"#### 🔴 Kill H{dominant} (the dominant head for {label})")
        st.markdown(f"> {text_kill_dom}")
        st.caption(
            f"Removing H{dominant} should break {label}'s voice. "
            "It's the head carrying most of the style."
        )

        st.markdown(f"#### Kill H{other} (control, not {label}'s dominant head)")
        st.markdown(f"> {text_kill_other}")
        st.caption(
            f"Removing H{other} should barely matter for {label}. "
            "That's the point: the dominant head is author-specific."
        )

        st.caption(
            f'Prompt: "{PROMPT}" · seed={SEED} · TinyStories-1Layer-21M · '
            "one head's output set to 0 via forward pre-hook."
        )

    # ── Explainer ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("How does this work?"):
        st.markdown(
            "The attention layer has 16 parallel heads, each producing "
            "a 64-dim slice of the output. A forward hook intercepts the "
            "pre-projection tensor and **multiplies one head's slice by 0** "
            "before the forward pass continues. That head still attends, "
            "but its contribution is thrown away.\n\n"
            "- For **Carroll**, H11 carries the style. Kill it → generic "
            "rabbit story, Alice and her voice gone. Kill H14 → barely "
            "noticeable.\n"
            "- For **Poe**, it's flipped. Kill H14 → the dark register "
            "collapses into nonsense. Kill H11 → stays broadly Poe.\n\n"
            "**Two authors, two different dominant heads, same tiny model.**"
        )

    with st.expander("Read more"):
        st.markdown(
            "- [Article 1: Sixteen Voices]"
            "(https://www.linkedin.com/pulse/sixteen-voices-interpretability-experiment-tiny-kate%C5%99ina-fajmanov%C3%A1-jmfnf)"
            " — the full head-knockout story.\n"
            "- [Article 2: Experiment in a Pocket]"
            "(https://www.linkedin.com/pulse/experiment-pocket-opening-tiny-model-finding-knobs-kate%C5%99ina-fajmanov%C3%A1-crodf)"
            " — what those heads actually compute (SAE features)."
        )


if __name__ == "__main__":
    main()
