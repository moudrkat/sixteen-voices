#!/usr/bin/env python3
"""Generate the showcase figure: each author steered by multiple features.

Shows baseline + steered text side by side for each author × feature combo.
Designed for article and LinkedIn.

Usage:
    uv run python scripts/fig_sae_showcase.py
    uv run python scripts/fig_sae_showcase.py --seed 123
"""

import argparse
import json
from pathlib import Path
from textwrap import fill

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sixteen_voices import load_tokenizer
from sixteen_voices.model import load_adapted_model, load_base_model
from sixteen_voices.sae import SparseAutoencoder

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
SAE_DIR = Path("outputs/sae_topk16_2048")

DARK = "#333333"
GRAY = "#888888"
LIGHT = "#f7f7f7"
RED = "#c44e52"
BLUE = "#4c72b0"
GREEN = "#55a868"
ORANGE = "#e8a735"

FEATURES = {
    "simplicity": {"indices": {665: 1.0}, "scale": 15.0, "color": RED,
                   "label": "simplicity"},
    "first_person": {"indices": {1779: 1.0}, "scale": 12.0, "color": BLUE,
                     "label": 'first-person "I"'},
    "dialogue": {"indices": {1777: 1.0, 689: 1.0}, "scale": 8.0, "color": GREEN,
                 "label": "dialogue"},
}

ROWS = [
    {"author": "poe", "prompt": "It was a dark and stormy",
     "display": "Poe", "features": ["simplicity", "first_person", "dialogue"]},
    {"author": "carroll", "prompt": "Alice was beginning to get very",
     "display": "Carroll", "features": ["simplicity", "first_person", "dialogue"]},
    {"author": "grimm", "prompt": "Once upon a time there lived",
     "display": "Grimm", "features": ["simplicity", "first_person", "dialogue"]},
]


def steer_generate(model, tokenizer, sae, prompt, feature_indices, scale,
                   seed=42, max_new=100, method="addition"):
    if method == "clamp":
        return clamp_generate(model, tokenizer, sae, prompt, feature_indices,
                              scale, seed, max_new)

    w = sae.decoder.weight.detach()
    vec = torch.zeros(w.shape[0])
    for fidx, weight in feature_indices.items():
        vec += weight * scale * w[:, fidx]

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].device),) + out[1:]
        return out + vec.to(out.device)

    handle = model.transformer.ln_f.register_forward_hook(hook)
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
    handle.remove()
    return text


def clamp_generate(model, tokenizer, sae, prompt, feature_indices, scale,
                   seed=42, max_new=100):
    """Golden Gate Bridge approach: encode → clamp feature → decode → replace."""
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out

        # Encode through SAE
        pre_act = sae.encoder(x)
        if sae.activation == "topk":
            topk = torch.topk(pre_act, k=sae.k, dim=-1)
            hidden = torch.zeros_like(pre_act)
            hidden.scatter_(-1, topk.indices, torch.relu(topk.values))
        else:
            hidden = torch.relu(pre_act)

        # Clamp target features to scale value
        for fidx, weight in feature_indices.items():
            hidden[..., fidx] = weight * scale

        # Decode back to residual stream
        x_hat = sae.decoder(hidden)

        if isinstance(out, tuple):
            return (x_hat,) + out[1:]
        return x_hat

    handle = model.transformer.ln_f.register_forward_hook(hook)
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
    handle.remove()
    return text


def baseline_generate(model, tokenizer, prompt, seed=42, max_new=100):
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, temperature=0.7,
            do_sample=True, top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def truncate_text(text, max_chars=180):
    """Clean truncation at word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "\u2026"


def render(results, output_path, seed):
    """Render the showcase figure."""
    n_rows = len(results)
    n_cols = 4  # baseline + 3 features

    col_labels = ["baseline"] + [FEATURES[f]["label"] for f in ROWS[0]["features"]]
    col_colors = [GRAY] + [FEATURES[f]["color"] for f in ROWS[0]["features"]]

    fig_w = 24
    fig_h = 5.5 * n_rows + 3.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # Layout first so we can place headers in figure coords
    fig.tight_layout(rect=[0.06, 0.01, 1.0, 0.86], h_pad=4.0, w_pad=2.5)

    # Column headers — placed in figure coords above the top row
    for col_idx in range(n_cols):
        ax = axes[0][col_idx]
        bbox = ax.get_position()
        col_x = (bbox.x0 + bbox.x1) / 2
        col_y = bbox.y1 + 0.025
        prefix = "" if col_idx == 0 else "+ "
        fig.text(col_x, col_y, f"{prefix}{col_labels[col_idx]}",
                 ha="center", va="bottom", fontsize=22,
                 fontweight="bold", color=col_colors[col_idx])

    for row_idx, row in enumerate(results):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            color = col_colors[col_idx]
            text = row["texts"][col_idx]
            text = truncate_text(text, 150)

            # Row label
            if col_idx == 0:
                ax.text(-0.08, 0.5, row["display"],
                        ha="right", va="center", fontsize=24,
                        fontweight="bold", color=DARK, rotation=90,
                        transform=ax.transAxes)

            # Background box
            bg_alpha = 0.06 if col_idx > 0 else 0.0
            bg_color = color if col_idx > 0 else LIGHT
            box = FancyBboxPatch(
                (0.01, 0.01), 0.98, 0.98,
                boxstyle="round,pad=0.04",
                facecolor=bg_color, alpha=bg_alpha if col_idx > 0 else 0.4,
                edgecolor=color, linewidth=2.0 if col_idx > 0 else 0.8,
                transform=ax.transAxes,
            )
            ax.add_patch(box)

            # Wrap and display text
            wrapped = fill(text, width=26)
            ax.text(0.5, 0.5, wrapped, ha="center", va="center",
                    fontsize=15, color=DARK, fontfamily="serif",
                    fontstyle="italic", transform=ax.transAxes,
                    linespacing=1.35)

    fig.suptitle(
        "One Voice, Many Directions",
        fontsize=34, fontweight="bold", y=0.97,
    )
    fig.text(0.5, 0.925,
             "Same author, same prompt, same seed \u2014 only the SAE feature direction changes",
             ha="center", fontsize=19, color=GRAY)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)
    tokenizer = load_tokenizer()

    model_cache = {}
    results = []

    for row_cfg in ROWS:
        author = row_cfg["author"]
        prompt = row_cfg["prompt"]

        print(f"\n{row_cfg['display']}:")

        if author not in model_cache:
            model_cache[author] = load_adapted_model(
                f"outputs/authors/{author}/adapter")
        model = model_cache[author]

        # Baseline
        bl = baseline_generate(model, tokenizer, prompt, args.seed, args.max_tokens)
        print(f"  baseline: {bl[:100]}")
        texts = [bl]

        # Features
        for feat_name in row_cfg["features"]:
            feat = FEATURES[feat_name]
            st = steer_generate(model, tokenizer, sae, prompt,
                                feat["indices"], feat["scale"],
                                args.seed, args.max_tokens)
            print(f"  + {feat['label']}: {st[:100]}")
            texts.append(st)

        results.append({"display": row_cfg["display"], "texts": texts})

    # Save JSON
    json_out = {"seed": args.seed, "results": [
        {"author": r["display"], "texts": r["texts"]} for r in results
    ]}
    json_path = SAE_DIR / "showcase_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved results to {json_path}")

    render(results, FIGURES_DIR / "sae_showcase.png", args.seed)


if __name__ == "__main__":
    main()