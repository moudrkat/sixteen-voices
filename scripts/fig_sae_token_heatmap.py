#!/usr/bin/env python3
"""Generate Anthropic-style token activation heatmaps for SAE features.

For each featured feature, shows a text passage with each token colored
by activation intensity — like the Golden Gate Bridge visualization.

Usage:
    uv run python scripts/fig_sae_token_heatmap.py
    uv run python scripts/fig_sae_token_heatmap.py --features 665 1779 1777 1663
    uv run python scripts/fig_sae_token_heatmap.py --authors minimalist dialogue
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

from sixteen_voices import load_base_model, load_tokenizer, TextChunkDataset
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
SAE_DIR = Path("outputs/sae_topk16_2048")

# Project palette
BLUE = "#4c72b0"
GREEN = "#55a868"
RED = "#c44e52"
ORANGE = "#e8a735"
DARK = "#333333"
GRAY = "#999999"

# Featured features: (feature_id, label, best_author, color_base)
FEATURED = [
    (665,  "simplicity (periods)",    "minimalist",  RED),
    (1779, 'first-person "I"',        "firstperson", BLUE),
    (1777, "dialogue attribution",    "dialogue",    GREEN),
    (1663, 'archaic "thou"',          "milton",      ORANGE),
    (776,  '"said" detector',         "dialogue",    BLUE),
    (9,    "question marks",          "questioner",  RED),
    (1518, '"Marilla" detector',      "montgomery_marilla",  GREEN),
    (746,  "semicolons (formality)",  "byron",       ORANGE),
    (1621, 'archaic "thy"',           "blake",       ORANGE),
]


def get_token_activations(model, tokenizer, sae, text, hook_point="residual",
                          seq_len=128):
    """Run text through model+SAE, return per-token activations for all features."""
    if hook_point == "residual":
        target = model.transformer.ln_f
    else:
        raise ValueError(f"Unsupported hook_point: {hook_point}")

    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach())

    handle = target.register_forward_hook(hook_fn)

    dataset = TextChunkDataset(text, tokenizer, max_length=seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    all_ids = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model(input_ids=batch["input_ids"])
            all_ids.append(batch["input_ids"])
            break  # just first chunk

    handle.remove()

    acts = activations[0].reshape(-1, activations[0].shape[-1])
    ids = all_ids[0].reshape(-1)[:acts.shape[0]]

    with torch.no_grad():
        _, hidden = sae(acts)

    tokens = [tokenizer.decode([tid]) for tid in ids.tolist()]
    return tokens, hidden  # hidden: (n_tokens, n_features)


def render_token_heatmap(ax, tokens, activations, feature_idx, label,
                         color_base, max_tokens=80):
    """Render a single feature's token heatmap on an axes."""
    acts = activations[:max_tokens, feature_idx].numpy()
    tokens = tokens[:max_tokens]

    # Normalize activations for color intensity
    act_max = acts.max()
    if act_max > 0:
        normed = acts / act_max
    else:
        normed = acts

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Convert color_base to RGB
    base_rgb = mcolors.to_rgb(color_base)

    # Layout tokens in wrapped lines
    x, y = 0.02, 0.85
    line_height = 0.18
    font_size = 9

    # Header
    firing = (acts > 0.01).sum()
    ax.text(0.0, 0.97, f"f{feature_idx} — {label}",
            fontsize=11, fontweight="bold", color=color_base,
            transform=ax.transAxes, va="top")
    ax.text(1.0, 0.97,
            f"fires on {firing}/{len(acts)} tokens ({100*firing/len(acts):.0f}%)",
            fontsize=8, color=GRAY, transform=ax.transAxes, va="top", ha="right")

    for i, (tok, norm_val, raw_val) in enumerate(zip(tokens, normed, acts)):
        # Measure token width (approximate)
        tok_display = tok.replace("\n", "↵").replace("\r", "")
        char_width = len(tok_display) * 0.0095 + 0.005

        # Wrap to next line if needed
        if x + char_width > 0.98:
            x = 0.02
            y -= line_height
            if y < 0.0:
                break

        # Background color: white to color_base based on activation
        alpha = min(norm_val ** 0.7, 1.0)  # slightly compress dynamic range
        if alpha > 0.02:
            bg_color = (*base_rgb, alpha * 0.7)
            bbox = dict(
                boxstyle="round,pad=0.05",
                facecolor=bg_color,
                edgecolor="none",
            )
        else:
            bbox = None

        # Text color: dark for low activation, white for high
        text_color = "white" if alpha > 0.6 else DARK

        ax.text(x, y, tok_display, fontsize=font_size, fontfamily="monospace",
                color=text_color, va="top", transform=ax.transAxes,
                bbox=bbox)

        x += char_width

    return ax


def render_token_heatmap_large(ax, tokens, activations, feature_idx, label,
                               color_base, max_tokens=80):
    """Render a single feature's token heatmap — large, poster-quality."""
    acts = activations[:max_tokens, feature_idx].numpy()
    tokens = tokens[:max_tokens]

    act_max = acts.max()
    normed = acts / act_max if act_max > 0 else acts

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    base_rgb = mcolors.to_rgb(color_base)

    x, y = 0.02, 0.82
    line_height = 0.16
    font_size = 13

    # Header — no feature ID, just the label
    firing = (acts > 0.01).sum()
    ax.text(0.0, 0.97, label,
            fontsize=17, fontweight="bold", color=color_base,
            transform=ax.transAxes, va="top")
    ax.text(1.0, 0.97,
            f"fires on {firing}/{len(acts)} tokens ({100*firing/len(acts):.0f}%)",
            fontsize=12, color=GRAY, transform=ax.transAxes, va="top", ha="right")

    for i, (tok, norm_val, raw_val) in enumerate(zip(tokens, normed, acts)):
        tok_display = tok.replace("\n", "↵").replace("\r", "")
        char_width = len(tok_display) * 0.012 + 0.006

        if x + char_width > 0.98:
            x = 0.02
            y -= line_height
            if y < 0.0:
                break

        alpha = min(norm_val ** 0.7, 1.0)
        if alpha > 0.02:
            bg_color = (*base_rgb, alpha * 0.7)
            bbox = dict(boxstyle="round,pad=0.06", facecolor=bg_color,
                        edgecolor="none")
        else:
            bbox = None

        text_color = "white" if alpha > 0.6 else DARK
        ax.text(x, y, tok_display, fontsize=font_size, fontfamily="monospace",
                color=text_color, va="top", transform=ax.transAxes, bbox=bbox)
        x += char_width


def render_article_heatmap(tokens_by_author, activations_by_author, output_path):
    """Render the article figure: 2 panels — first-person and dialogue features."""
    panels = [
        (1779, 'First-person "I"', "firstperson", BLUE),
        (1777, "Dialogue attribution", "dialogue", GREEN),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(18, 7))

    for i, (feat_idx, label, author, color) in enumerate(panels):
        tokens = tokens_by_author[author]
        acts = activations_by_author[author]
        render_token_heatmap_large(axes[i], tokens, acts, feat_idx, label, color)

    fig.suptitle("What SAE Features Detect",
                 fontsize=22, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def render_multi_feature_panel(tokens_by_author, activations_by_author,
                               featured_features, output_path):
    """Render a multi-panel figure with one row per feature."""
    n_features = len(featured_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, 2.8 * n_features))

    if n_features == 1:
        axes = [axes]

    for i, (feat_idx, label, best_author, color) in enumerate(featured_features):
        if best_author in tokens_by_author:
            tokens = tokens_by_author[best_author]
            acts = activations_by_author[best_author]
        else:
            # Fallback to first available
            first_author = list(tokens_by_author.keys())[0]
            tokens = tokens_by_author[first_author]
            acts = activations_by_author[first_author]

        render_token_heatmap(axes[i], tokens, acts, feat_idx, label, color)

    fig.suptitle("Feature Activation Heatmaps — What Each Feature Detects",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def render_single_feature_detail(tokens_by_author, activations_by_author,
                                 feat_idx, label, color, output_path):
    """Render a detailed single-feature figure across multiple authors."""
    authors_to_show = list(tokens_by_author.keys())[:6]
    n = len(authors_to_show)

    fig, axes = plt.subplots(n, 1, figsize=(16, 2.2 * n))
    if n == 1:
        axes = [axes]

    for i, author in enumerate(authors_to_show):
        tokens = tokens_by_author[author]
        acts = activations_by_author[author]
        render_token_heatmap(
            axes[i], tokens, acts, feat_idx,
            f"{label} — {author}", color,
        )

    fig.suptitle(f"f{feat_idx}: {label} — across authors",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Anthropic-style token activation heatmaps")
    parser.add_argument("--sae-dir", default=str(SAE_DIR))
    parser.add_argument("--features", type=int, nargs="*", default=None,
                        help="Specific feature indices to plot")
    parser.add_argument("--authors", nargs="*", default=None,
                        help="Authors to use for text samples")
    parser.add_argument("--detail-features", type=int, nargs="*", default=None,
                        help="Features to show in multi-author detail view")
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)

    print("Loading SAE, model, tokenizer...")
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)
    tokenizer = load_tokenizer()
    model = load_base_model()

    # Determine which features and authors to use
    if args.features:
        featured = [(f, f"f{f}", "minimalist", BLUE) for f in args.features]
    else:
        featured = FEATURED

    # Collect needed authors
    needed_authors = set()
    for _, _, author, _ in featured:
        needed_authors.add(author)
    if args.authors:
        needed_authors = set(args.authors)

    # Add a few contrasting authors for detail views
    extra_authors = {"minimalist", "dialogue", "poe", "homer", "grimm", "montgomery"}
    needed_authors |= extra_authors

    # Load texts and get activations
    text_dir = Path("data/eval")
    tokens_by_author = {}
    activations_by_author = {}

    for author in sorted(needed_authors):
        text_path = text_dir / f"{author}.txt"
        if not text_path.exists():
            print(f"  Skipping {author} (no eval text)")
            continue

        print(f"  Processing {author}...")
        text = text_path.read_text()
        tokens, hidden = get_token_activations(
            model, tokenizer, sae, text, config["hook_point"]
        )
        tokens_by_author[author] = tokens
        activations_by_author[author] = hidden

    # 0. Article figure: just first-person + dialogue
    if "firstperson" in tokens_by_author and "dialogue" in tokens_by_author:
        print("\nGenerating article heatmap (2-panel)...")
        render_article_heatmap(
            tokens_by_author, activations_by_author,
            FIGURES_DIR / "sae_token_heatmap_article.png",
        )

    # 1. Multi-feature overview panel
    print("\nGenerating multi-feature overview...")
    render_multi_feature_panel(
        tokens_by_author, activations_by_author, featured,
        FIGURES_DIR / "sae_token_heatmap.png",
    )

    # 2. Detail views for key features
    detail_features = args.detail_features or [665, 1779, 1663]
    for feat_idx in detail_features:
        # Find matching featured entry
        match = next((f for f in FEATURED if f[0] == feat_idx), None)
        if match:
            _, label, _, color = match
        else:
            label, color = f"f{feat_idx}", BLUE

        print(f"Generating detail view for f{feat_idx}...")
        render_single_feature_detail(
            tokens_by_author, activations_by_author,
            feat_idx, label, color,
            FIGURES_DIR / f"sae_token_heatmap_f{feat_idx}.png",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()