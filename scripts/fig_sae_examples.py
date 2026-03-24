#!/usr/bin/env python3
"""Generate a clean before/after text figure for SAE steering.

Shows 3 best examples side-by-side: baseline vs steered.
Big readable text, minimal chrome.

Usage:
    uv run python scripts/fig_sae_examples.py
"""

from pathlib import Path
from textwrap import fill

import json
import torch
import matplotlib.pyplot as plt

from sixteen_voices.model import load_adapted_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
SAE_DIR = Path("outputs/sae")


def gen(model, tokenizer, prompt, vec=None, seed=42, max_new=70):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + vec,) + output[1:]
        return output + vec
    hook = model.transformer.ln_f.register_forward_hook(hook_fn) if vec is not None else None
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=0.7,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
    if hook:
        hook.remove()
    return text


def truncate(text, n=150):
    if len(text) > n:
        text = text[:n].rsplit(" ", 1)[0] + " ..."
    return text


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)
    w = sae.decoder.weight.detach()

    tokenizer = load_tokenizer()

    folk_voice_vec = 8 * (w[:, 198] + w[:, 33] + w[:, 140])
    event_narration_vec = 8 * (w[:, 160] + w[:, 144] + w[:, 205])

    # Generate the 3 best examples
    examples = []

    # 1. Poet + complexity: line breaks vanish
    model = load_adapted_model("outputs/authors/poet/adapter")
    b = gen(model, tokenizer, "The wind whispered through")
    s = gen(model, tokenizer, "The wind whispered through", complexity_vec)
    examples.append({
        "author": "POET",
        "knob": "Complexity +",
        "color": "#4c72b0",
        "baseline": truncate(b),
        "steered": truncate(s),
        "note": "Line breaks vanish. Verse becomes prose.",
    })

    # 2. Dark + complexity: creepy → adventurous
    model = load_adapted_model("outputs/authors/dark/adapter")
    b = gen(model, tokenizer, "Once upon a time")
    s = gen(model, tokenizer, "Once upon a time", complexity_vec)
    examples.append({
        "author": "DARK",
        "knob": "Complexity +",
        "color": "#c44e52",
        "baseline": truncate(b),
        "steered": truncate(s),
        "note": "Creepy atmosphere replaced by curious exploration.",
    })

    # 3. Grimm + dialogue: third person → first person
    model = load_adapted_model("outputs/authors/grimm/adapter")
    b = gen(model, tokenizer, "Once upon a time")
    s = gen(model, tokenizer, "Once upon a time", dialogue_vec)
    examples.append({
        "author": "GRIMM",
        "knob": "Dialogue +",
        "color": "#8172b2",
        "baseline": truncate(b),
        "steered": truncate(s),
        "note": "Third-person fairy tale shifts to first person.",
    })

    # Build figure
    fig, axes = plt.subplots(len(examples), 2, figsize=(16, len(examples) * 3.2),
                              gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})

    fig.suptitle("SAE Feature Steering", fontsize=18, fontweight="bold", y=0.97)

    for i, ex in enumerate(examples):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        for ax in [ax_left, ax_right]:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        # Left: baseline
        if i == 0:
            ax_left.set_title("Baseline", fontsize=13, fontweight="bold",
                              color="#666", pad=10)
            ax_right.set_title("Steered", fontsize=13, fontweight="bold",
                               color="#666", pad=10)

        # Author label on far left
        ax_left.text(-0.02, 0.95, ex["author"],
                     fontsize=14, fontweight="bold", color=ex["color"],
                     transform=ax_left.transAxes, va="top", ha="right")

        # Knob badge
        ax_right.text(1.02, 0.95, ex["knob"],
                      fontsize=10, color=ex["color"],
                      transform=ax_right.transAxes, va="top", ha="left",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor=ex["color"], alpha=0.12,
                                edgecolor=ex["color"], linewidth=0.8))

        # Baseline text
        ax_left.text(0.05, 0.85, fill(ex["baseline"], width=55),
                     fontsize=11, color="#444", family="serif", style="italic",
                     transform=ax_left.transAxes, va="top",
                     linespacing=1.4)

        # Steered text
        ax_right.text(0.05, 0.85, fill(ex["steered"], width=55),
                      fontsize=11, color=ex["color"], family="serif",
                      style="italic", fontweight="bold",
                      transform=ax_right.transAxes, va="top",
                      linespacing=1.4)

        # Note
        ax_right.text(0.05, 0.08, ex["note"],
                      fontsize=9.5, color="#888",
                      transform=ax_right.transAxes, va="bottom")

        # Separator
        if i < len(examples) - 1:
            line_y = axes[i, 0].get_position().y0 - 0.01
            fig.add_artist(plt.Line2D([0.08, 0.92], [line_y, line_y],
                                       transform=fig.transFigure,
                                       color="#ddd", linewidth=1))

    fig.tight_layout(rect=[0.06, 0.02, 0.98, 0.94])
    fig.savefig(FIGURES_DIR / "sae_steering_examples.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_steering_examples.png'}")


if __name__ == "__main__":
    main()