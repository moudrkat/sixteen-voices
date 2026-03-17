#!/usr/bin/env python3
"""Per-token head attribution across authors.

For each prompt × author, generates text token-by-token and decomposes
each token's logit into contributions from each of the 16 heads. Shows
which heads "push" for each generated word.

In this 1-layer model, the decomposition is exact:
    logit(token) = embedding_bias + Σ_h (head_h_output @ W_O_h @ W_unembed[token])

Usage:
    uv run python scripts/fig_token_attribution.py [--recompute]

Outputs:
    figures/token_attribution.png
    outputs/token_attribution.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from sixteen_voices import (
    load_base_model, load_tokenizer, load_adapted_model,
    NUM_HEADS, HEAD_DIM,
)

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_FIG = Path("figures/token_attribution.png")
OUTPUT_JSON = Path("outputs/token_attribution.json")

PROMPTS = [
    "Once upon a time, in a dark",
    "The old man looked at her and said",
    "She was not like the other",
    "The water was",
    "It was a cold night and the",
    "The king sat on his throne and",
    "Deep in the forest there lived",
    "The ship sailed across the",
]

# Authors that show interesting H14 contrast
AUTHORS = ["poe", "twain", "carroll", "melville", "burnett", "homer", "browne", "yeats"]

# Which prompts to show in the figure (best contrast)
FIG_PROMPTS = [
    "Once upon a time, in a dark",
    "The water was",
    "It was a cold night and the",
    "The ship sailed across the",
]

MAX_TOKENS = 40
SEED = 42


def get_head_contributions(model, tokenizer, input_ids):
    """Decompose next-token logits into per-head contributions.

    Returns:
        head_contribs: [NUM_HEADS, vocab_size] — each head's contribution to logits
        residual_contribs: [vocab_size] — contribution from everything except attention heads
        full_logits: [vocab_size] — the actual model output
    """
    captured = {}

    def capture_hook(module, args):
        captured["head_out"] = args[0].detach()

    if hasattr(model, "peft_config"):
        attn = model.base_model.model.transformer.h[0].attn.attention
        lm_head = model.base_model.model.lm_head
    else:
        attn = model.transformer.h[0].attn.attention
        lm_head = model.lm_head

    hook = attn.out_proj.register_forward_pre_hook(capture_hook)

    with torch.no_grad():
        outputs = model(input_ids)
        full_logits = outputs.logits[0, -1]

    hook.remove()

    head_out = captured["head_out"][0, -1]  # [1024]
    w_o = attn.out_proj.weight.data
    w_unembed = lm_head.weight.data

    head_contribs = torch.zeros(NUM_HEADS, w_unembed.shape[0])
    for h in range(NUM_HEADS):
        h_out = head_out[h * HEAD_DIM : (h + 1) * HEAD_DIM]
        h_residual = w_o[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] @ h_out
        head_contribs[h] = w_unembed @ h_residual

    residual = full_logits - head_contribs.sum(dim=0)

    return head_contribs, residual, full_logits


def generate_with_attribution(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
    """Generate tokens one at a time, recording per-head attribution."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    tokens = []
    attributions = []

    torch.manual_seed(SEED)

    for _ in range(max_tokens):
        head_contribs, residual, full_logits = get_head_contributions(
            model, tokenizer, input_ids
        )

        # Sample with temperature for readable text (greedy causes repetition)
        probs = torch.softmax(full_logits / 0.8, dim=0)
        next_token = torch.multinomial(probs, 1).unsqueeze(0)
        token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)

        token_id = next_token.item()
        per_head = head_contribs[:, token_id].numpy()

        tokens.append(token_str)
        attributions.append(per_head)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokens, np.array(attributions)


def compute_all(tokenizer, base_model, authors):
    """Run attribution for all prompts × authors."""
    all_results = {}

    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        prompt_results = {}

        print(f"  base...")
        tokens, attribs = generate_with_attribution(base_model, tokenizer, prompt)
        prompt_results["base"] = {
            "tokens": tokens,
            "attributions": attribs.tolist(),
        }
        print(f"    → {''.join(tokens)}")

        for author in authors:
            print(f"  {author}...")
            model = load_adapted_model(ADAPTERS_DIR / author / "adapter")
            tokens, attribs = generate_with_attribution(model, tokenizer, prompt)
            prompt_results[author] = {
                "tokens": tokens,
                "attributions": attribs.tolist(),
            }
            print(f"    → {''.join(tokens)}")
            del model

        all_results[prompt] = prompt_results

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")

    return all_results


def make_figure(all_results, authors):
    """Create attribution heatmap showing top-3 head contributions per token."""
    prompts = [p for p in FIG_PROMPTS if p in all_results]
    all_sources = ["base"] + [a for a in authors if a in list(all_results.values())[0]]
    n_prompts = len(prompts)
    n_sources = len(all_sources)

    fig, axes = plt.subplots(n_prompts, 1, figsize=(20, 3.2 * n_prompts + 0.5))
    if n_prompts == 1:
        axes = [axes]

    # Color map for key heads
    head_cmap = {
        14: "#991b1b",  # red
        11: "#1e40af",  # blue
        3:  "#7c3aed",  # purple
    }
    default_color = "#6b7280"

    for pi, prompt in enumerate(prompts):
        ax = axes[pi]
        data = all_results[prompt]

        max_tokens = max(len(data[s]["tokens"]) for s in all_sources if s in data)

        for si, source in enumerate(all_sources):
            if source not in data:
                continue

            tokens = data[source]["tokens"]
            attribs = np.array(data[source]["attributions"])
            y = n_sources - 1 - si

            for ti, token in enumerate(tokens):
                contribs = attribs[ti]

                # Normalize: fraction of positive contribution per head
                pos = np.maximum(contribs, 0)
                total_pos = pos.sum()
                if total_pos > 0:
                    fracs = pos / total_pos
                else:
                    fracs = np.zeros(NUM_HEADS)

                # Draw stacked mini-bar inside the cell showing top heads
                top3 = np.argsort(fracs)[::-1][:3]
                dom_h = top3[0]
                dom_frac = fracs[dom_h]

                # Background color from dominant head
                color = head_cmap.get(dom_h, default_color)
                r, g, b = mcolors.to_rgb(color)
                alpha = 0.15 + 0.75 * dom_frac

                ax.add_patch(plt.Rectangle(
                    (ti, y), 1, 1,
                    facecolor=(r, g, b, alpha),
                    edgecolor="white", linewidth=0.8
                ))

                # Token text
                display = token.replace("\n", "↵").strip()
                if len(display) > 9:
                    display = display[:8] + "…"
                ax.text(ti + 0.5, y + 0.68, display,
                       ha="center", va="center", fontsize=6.5,
                       fontfamily="monospace")

                # Head label — show dominant + second if close
                label = f"H{dom_h}"
                second_h = top3[1]
                if fracs[second_h] > 0.2:
                    label += f"+{second_h}"
                ax.text(ti + 0.5, y + 0.32, label,
                       ha="center", va="center", fontsize=5.5,
                       fontweight="bold", alpha=0.8,
                       color="white" if alpha > 0.5 else "#374151")

        ax.set_xlim(0, max_tokens)
        ax.set_ylim(0, n_sources)
        ax.set_yticks([i + 0.5 for i in range(n_sources)])
        ax.set_yticklabels(list(reversed(all_sources)), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_title(f'"{prompt}"', fontsize=12, fontweight="bold", pad=10)
        ax.set_aspect("equal")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#991b1b", label="H14 dominant"),
        Patch(facecolor="#1e40af", label="H11 dominant"),
        Patch(facecolor="#7c3aed", label="H3 dominant"),
        Patch(facecolor="#6b7280", label="Other head"),
    ]
    fig.legend(handles=legend_elements, loc="upper right",
              ncol=4, fontsize=9, frameon=True,
              bbox_to_anchor=(0.98, 1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")


def main():
    recompute = "--recompute" in sys.argv

    authors = [a for a in AUTHORS
               if (ADAPTERS_DIR / a / "adapter" / "adapter_config.json").exists()]
    print(f"Authors: {authors}")

    if not recompute and OUTPUT_JSON.exists():
        print("Loading cached results (use --recompute to regenerate)")
        with open(OUTPUT_JSON) as f:
            all_results = json.load(f)
    else:
        tokenizer = load_tokenizer()
        base_model = load_base_model()
        all_results = compute_all(tokenizer, base_model, authors)

    make_figure(all_results, authors)
    make_h14_heatmap(all_results, authors)


def make_h14_heatmap(all_results, authors):
    """H14-specific contribution heatmap: red = H14 pushing, blue = H14 opposing."""
    prompts = [p for p in FIG_PROMPTS if p in all_results]
    all_sources = ["base"] + [a for a in authors if a in list(all_results.values())[0]]
    n_prompts = len(prompts)
    n_sources = len(all_sources)

    fig, axes = plt.subplots(n_prompts, 1, figsize=(22, 3.0 * n_prompts + 0.5))
    if n_prompts == 1:
        axes = [axes]

    # Collect all H14 values to set consistent color scale
    all_h14 = []
    for prompt in prompts:
        data = all_results[prompt]
        for source in all_sources:
            if source not in data:
                continue
            attribs = np.array(data[source]["attributions"])
            all_h14.extend(attribs[:, 14].tolist())

    vmax = np.percentile(np.abs(all_h14), 95)

    for pi, prompt in enumerate(prompts):
        ax = axes[pi]
        data = all_results[prompt]

        max_tokens = max(len(data[s]["tokens"]) for s in all_sources if s in data)

        for si, source in enumerate(all_sources):
            if source not in data:
                continue

            tokens = data[source]["tokens"]
            attribs = np.array(data[source]["attributions"])
            y = n_sources - 1 - si

            for ti, token in enumerate(tokens):
                h14_val = attribs[ti, 14]

                # Red = positive H14, blue = negative, white = neutral
                intensity = np.clip(h14_val / vmax, -1, 1)
                if intensity > 0:
                    r, g, b = 0.6 + 0.4 * intensity, 0.1 + 0.2 * (1 - intensity), 0.1 + 0.2 * (1 - intensity)
                elif intensity < 0:
                    r, g, b = 0.1 + 0.2 * (1 + intensity), 0.1 + 0.2 * (1 + intensity), 0.6 + 0.4 * (-intensity)
                else:
                    r, g, b = 0.95, 0.95, 0.95

                ax.add_patch(plt.Rectangle(
                    (ti, y), 1, 1,
                    facecolor=(r, g, b),
                    edgecolor="white", linewidth=0.5
                ))

                # Token text
                display = token.replace("\n", "↵").strip()
                if len(display) > 9:
                    display = display[:8] + "…"
                text_color = "white" if abs(intensity) > 0.6 else "#374151"
                ax.text(ti + 0.5, y + 0.5, display,
                       ha="center", va="center", fontsize=5.5,
                       fontfamily="monospace", color=text_color)

        ax.set_xlim(0, max_tokens)
        ax.set_ylim(0, n_sources)
        ax.set_yticks([i + 0.5 for i in range(n_sources)])
        ax.set_yticklabels(list(reversed(all_sources)), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_title(f'"{prompt}"', fontsize=12, fontweight="bold", pad=8)
        ax.set_aspect("equal")

    # Colorbar
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.colors as mc

    cmap = mc.LinearSegmentedColormap.from_list("h14",
        [(0.1, 0.1, 1.0), (0.95, 0.95, 0.95), (1.0, 0.1, 0.1)])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(-vmax, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("H14 logit contribution to chosen token", fontsize=10)

    fig.suptitle("H14 contribution per token across authors",
                fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout(rect=[0, 0, 0.92, 1.0])
    h14_path = Path("figures/token_attribution_h14.png")
    fig.savefig(h14_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {h14_path}")


if __name__ == "__main__":
    main()