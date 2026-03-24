#!/usr/bin/env python3
"""Generate the "killer" steering figure for LinkedIn.

Shows generated text from multiple authors, baseline vs steered,
arranged to show what the feature knobs actually do.

Usage:
    uv run python scripts/fig_sae_steering.py
"""

import json
from pathlib import Path
from textwrap import fill

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

from sixteen_voices.model import load_adapted_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
SAE_DIR = Path("outputs/sae")


def gen(model, tokenizer, prompt, vec=None, seed=42, max_new=60):
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


def fig_steering_showcase(sae, tokenizer):
    """The main figure: baseline vs steered for several authors."""
    w = sae.decoder.weight.detach()

    folk_voice_vec = 8 * (w[:, 198] + w[:, 33] + w[:, 140])
    event_narration_vec = 8 * (w[:, 160] + w[:, 144] + w[:, 205])
    speech_patterns_vec = 8 * (w[:, 68] + w[:, 113] + w[:, 122])

    cases = [
        {
            "author": "poet",
            "knob": "Complexity +",
            "vec": complexity_vec,
            "prompt": "The wind whispered through",
            "color": "#4c72b0",
        },
        {
            "author": "minimalist",
            "knob": "Complexity +",
            "vec": complexity_vec,
            "prompt": "Once upon a time",
            "color": "#55a868",
        },
        {
            "author": "dark",
            "knob": "Complexity +",
            "vec": complexity_vec,
            "prompt": "Once upon a time",
            "color": "#c44e52",
        },
        {
            "author": "grimm",
            "knob": "Dialogue +",
            "vec": dialogue_vec,
            "prompt": "Once upon a time",
            "color": "#8172b2",
        },
        {
            "author": "carroll",
            "knob": "Conventional +",
            "vec": conventional_vec,
            "prompt": "Once upon a time",
            "color": "#ccb974",
        },
    ]

    fig, axes = plt.subplots(len(cases), 1, figsize=(14, len(cases) * 2.4))
    fig.suptitle("SAE Feature Steering: same direction, different authors",
                 fontsize=16, fontweight="bold", y=0.98)

    for i, case in enumerate(cases):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        model = load_adapted_model(f"outputs/authors/{case['author']}/adapter")
        baseline = gen(model, tokenizer, case["prompt"])
        steered = gen(model, tokenizer, case["prompt"], case["vec"])

        # Truncate to ~100 chars
        baseline_short = baseline[:120].rsplit(" ", 1)[0] + "..."
        steered_short = steered[:120].rsplit(" ", 1)[0] + "..."

        # Author label + knob
        ax.text(0.0, 0.85, f"{case['author'].upper()}",
                fontsize=13, fontweight="bold", color=case["color"],
                transform=ax.transAxes, va="top")
        ax.text(0.0, 0.58, f"  {case['knob']}",
                fontsize=10, color="#666",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=case["color"],
                          alpha=0.15, edgecolor="none"))

        # Arrow
        ax.annotate("", xy=(0.49, 0.45), xytext=(0.49, 0.75),
                     arrowprops=dict(arrowstyle="->", color=case["color"],
                                     lw=2.5),
                     transform=ax.transAxes)

        # Baseline text (top right)
        ax.text(0.15, 0.85,
                fill(baseline_short, width=70),
                fontsize=9.5, color="#333", family="serif",
                transform=ax.transAxes, va="top",
                style="italic")

        # Steered text (bottom right)
        ax.text(0.15, 0.35,
                fill(steered_short, width=70),
                fontsize=9.5, color=case["color"], family="serif",
                transform=ax.transAxes, va="top",
                fontweight="bold", style="italic")

        # Separator
        if i < len(cases) - 1:
            ax.plot([0, 1], [-0.05, -0.05], color="#ddd", linewidth=0.8,
                    transform=ax.transAxes, clip_on=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "sae_steering_showcase.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_steering_showcase.png'}")


def fig_style_space_with_arrows(sae, tokenizer):
    """PCA scatter with steering arrows overlaid."""
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko_raw = json.load(f)
    knockout = np.array([
        [ko_raw[a]["head_recovery"][f"H{h}"] for h in range(16)]
        for a in authors
    ])
    best_head = knockout.argmax(axis=1)

    # PCA
    centered = matrix - matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_explained = S ** 2 / (S ** 2).sum()

    # Project feature directions into PCA space
    w = sae.decoder.weight.detach().numpy()  # (1024, 256)

    # We need to project the SAE directions into the author-feature PCA space
    # The PCA was on the 77x256 author-feature matrix
    # Feature direction = change in feature activations
    # Folk voice features: 198, 33, 140 → direction in 256-dim feature space
    folk_voice_dir_256 = np.zeros(256)
    folk_voice_dir_256[[198, 33, 140]] = 1.0
    folk_voice_dir_256 -= centered.mean(axis=0)  # not needed, just direction
    # Project into PCA
    folk_voice_pca = Vt[:2] @ folk_voice_dir_256

    event_narration_dir_256 = np.zeros(256)
    event_narration_dir_256[[160, 144, 205]] = 1.0
    event_narration_pca = Vt[:2] @ event_narration_dir_256

    HEAD_COLORS = {3: "#55a868", 11: "#4c72b0", 14: "#c44e52"}
    DEFAULT_COLOR = "#cccccc"

    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot authors
    for i, author in enumerate(authors):
        h = best_head[i]
        c = HEAD_COLORS.get(h, DEFAULT_COLOR)
        s = 60 if h in HEAD_COLORS else 30
        alpha = 0.8 if h in HEAD_COLORS else 0.4
        ax.scatter(coords[i, 0], coords[i, 1], c=c, s=s, alpha=alpha,
                   edgecolors="white", linewidth=0.5, zorder=2)

    # Label interesting authors
    label_set = {"poe", "carroll", "grimm", "minimalist", "dark", "poet",
                 "wilde", "homer", "dialogue", "harris", "cozy", "lear"}
    for i, author in enumerate(authors):
        if author in label_set:
            ax.annotate(author, (coords[i, 0], coords[i, 1]),
                        fontsize=7.5, alpha=0.75,
                        xytext=(5, 5), textcoords="offset points")

    # Draw feature direction arrows from center
    cx, cy = coords.mean(axis=0)
    arrow_scale = 15

    ax.annotate("", xy=(cx + complexity_pca[0] * arrow_scale,
                         cy + complexity_pca[1] * arrow_scale),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#c44e52",
                                lw=3, mutation_scale=20))
    ax.text(cx + complexity_pca[0] * arrow_scale * 1.1,
            cy + complexity_pca[1] * arrow_scale * 1.1,
            "COMPLEXITY\n(H3-controlled)",
            fontsize=10, fontweight="bold", color="#c44e52",
            ha="center", va="center")

    ax.annotate("", xy=(cx + conventional_pca[0] * arrow_scale,
                         cy + conventional_pca[1] * arrow_scale),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#dd8452",
                                lw=3, mutation_scale=20))
    ax.text(cx + conventional_pca[0] * arrow_scale * 1.15,
            cy + conventional_pca[1] * arrow_scale * 1.15,
            "CONVENTIONAL\n(head-independent)",
            fontsize=10, fontweight="bold", color="#dd8452",
            ha="center", va="center")

    # Legend
    for h, c in HEAD_COLORS.items():
        mask = best_head == h
        ax.scatter([], [], c=c, s=60, label=f"H{h} dominant ({mask.sum()})")
    ax.scatter([], [], c=DEFAULT_COLOR, s=30, alpha=0.5, label="Other heads")
    ax.legend(loc="lower left", fontsize=9)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.0%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.0%} variance)", fontsize=11)
    ax.set_title("77 authors in SAE feature space — with steering directions",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sae_style_space_arrows.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'sae_style_space_arrows.png'}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SAE...")
    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)

    tokenizer = load_tokenizer()

    print("Generating steering showcase...")
    fig_steering_showcase(sae, tokenizer)

    print("Generating style space with arrows...")
    fig_style_space_with_arrows(sae, tokenizer)

    print("Done.")


if __name__ == "__main__":
    main()