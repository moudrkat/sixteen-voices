#!/usr/bin/env python3
"""Generate a grid figure: one author × multiple features, showing how
each feature sculpts the same voice differently.

Each row = one author. Each column = baseline + one feature applied.
Same prompt and seed throughout, so differences are purely from steering.

Usage:
    uv run python scripts/fig_sae_feature_grid.py
    uv run python scripts/fig_sae_feature_grid.py --seed 123
"""

import argparse
import json
from pathlib import Path
from textwrap import fill

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.model import load_adapted_model
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
SAE_DIR = Path("outputs/sae_topk16_2048")

# Colors
DARK = "#333333"
GRAY = "#999999"
LIGHT_GRAY = "#f5f5f5"
BLUE = "#4c72b0"
GREEN = "#55a868"
RED = "#c44e52"
ORANGE = "#e8a735"
PURPLE = "#8172b3"

FEATURES = {
    "simplicity": {"indices": {665: 1.0}, "scale": 15.0, "color": RED,
                   "label": "+ simplicity", "short": "Short sentences"},
    "first_person": {"indices": {1779: 1.0}, "scale": 12.0, "color": BLUE,
                     "label": "+ first-person", "short": '"I" narration'},
    "dialogue": {"indices": {1777: 1.0, 689: 1.0}, "scale": 8.0, "color": GREEN,
                 "label": "+ dialogue", "short": "Conversation"},
    "verse": {"indices": {344: 1.0}, "scale": 12.0, "color": ORANGE,
              "label": "+ verse", "short": "Line breaks"},
    "complexity": {"indices": {883: 1.0, 993: 1.0, 60: 1.0}, "scale": 12.0,
                   "color": PURPLE, "label": "+ complexity", "short": "Ornate prose"},
}

AUTHORS = [
    {
        "name": "poe",
        "prompt": "It was a dark and stormy",
        "features": ["simplicity", "first_person", "dialogue", "verse"],
        "display": "Poe (gothic)",
    },
    {
        "name": "grimm",
        "prompt": "Once upon a time there lived",
        "features": ["simplicity", "dialogue", "verse", "complexity"],
        "display": "Grimm (fairy tale)",
    },
    {
        "name": "carroll",
        "prompt": "Alice was beginning to get very",
        "features": ["simplicity", "first_person", "verse", "dialogue"],
        "display": "Carroll (whimsical)",
    },
    {
        "name": "homer",
        "prompt": "The warrior stood before the gates",
        "features": ["simplicity", "first_person", "dialogue", "verse"],
        "display": "Homer (epic)",
    },
]


def make_steering_hook(sae, feature_indices, scale):
    w = sae.decoder.weight.detach()
    steering_vec = torch.zeros(w.shape[0])
    for feat_idx, weight in feature_indices.items():
        steering_vec += weight * scale * w[:, feat_idx]

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + steering_vec.to(output[0].device),) + output[1:]
        return output + steering_vec.to(output.device)
    return hook_fn


def generate(model, tokenizer, prompt, max_new=80, seed=42, hook_fn=None):
    if hook_fn:
        handle = model.transformer.ln_f.register_forward_hook(hook_fn)
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
    if hook_fn:
        handle.remove()
    return text


def render_grid(results, output_path):
    """Render the author × feature grid.

    results: list of dicts with keys: author_display, columns (list of
             {label, color, text})
    """
    n_rows = len(results)
    n_cols = max(len(r["columns"]) for r in results)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row_idx, row_data in enumerate(results):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            if col_idx >= len(row_data["columns"]):
                continue

            col = row_data["columns"][col_idx]
            text = col["text"]
            color = col["color"]
            label = col["label"]

            # Header
            if row_idx == 0:
                ax.text(0.5, 1.12, label, ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color=color,
                        transform=ax.transAxes)

            # Author label on left
            if col_idx == 0:
                ax.text(-0.08, 0.5, row_data["author_display"],
                        ha="right", va="center", fontsize=11,
                        fontweight="bold", color=DARK, rotation=90,
                        transform=ax.transAxes)

            # Text box
            wrapped = fill(text, width=38)
            # Truncate to ~4 lines
            lines = wrapped.split("\n")[:5]
            display_text = "\n".join(lines)
            if len(lines) >= 5 and len(wrapped.split("\n")) > 5:
                display_text += "..."

            # Background
            bg = FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.03",
                facecolor=color if col_idx > 0 else LIGHT_GRAY,
                alpha=0.08 if col_idx > 0 else 0.5,
                edgecolor=color if col_idx > 0 else GRAY,
                linewidth=1.5,
                transform=ax.transAxes,
            )
            ax.add_patch(bg)

            ax.text(0.5, 0.52, display_text, ha="center", va="center",
                    fontsize=8.5, color=DARK, fontfamily="serif",
                    fontstyle="italic", transform=ax.transAxes,
                    linespacing=1.4)

    fig.suptitle(
        "One Voice, Many Directions — Steering with SAE Features",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--output", default=str(FIGURES_DIR / "sae_feature_grid.png"))
    args = parser.parse_args()

    # Load SAE
    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    results = []
    model_cache = {}

    for author_cfg in AUTHORS:
        author = author_cfg["name"]
        prompt = author_cfg["prompt"]
        feat_names = author_cfg["features"]

        print(f"\n{author_cfg['display']}:")

        # Load model
        if author not in model_cache:
            model_cache[author] = load_adapted_model(
                f"outputs/authors/{author}/adapter")
        model = model_cache[author]

        # Baseline
        baseline = generate(model, tokenizer, prompt, args.max_tokens, args.seed)
        print(f"  baseline: {baseline[:80]}...")

        columns = [{"label": "Baseline", "color": GRAY, "text": baseline}]

        # Each feature
        for feat_name in feat_names:
            feat = FEATURES[feat_name]
            hook = make_steering_hook(sae, feat["indices"], feat["scale"])
            steered = generate(model, tokenizer, prompt, args.max_tokens,
                               args.seed, hook_fn=hook)
            print(f"  {feat['label']}: {steered[:80]}...")
            columns.append({
                "label": feat["label"],
                "color": feat["color"],
                "text": steered,
            })

        results.append({
            "author_display": author_cfg["display"],
            "columns": columns,
        })

    # Save raw results as JSON too
    json_path = SAE_DIR / "feature_grid_results.json"
    json_data = []
    for r in results:
        json_data.append({
            "author": r["author_display"],
            "columns": [{"label": c["label"], "text": c["text"]} for c in r["columns"]],
        })
    with open(json_path, "w") as f:
        json.dump({"seed": args.seed, "results": json_data}, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Render
    render_grid(results, args.output)


if __name__ == "__main__":
    main()