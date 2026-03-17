#!/usr/bin/env python3
"""Figure: Poe style dial — scale all semantic heads together.

The key finding: we identified which heads are "style" heads (semantic,
diffuse attention) vs "structural" heads (previous-token, local). Now
we dial only the style heads from 0.5× to 1.25×, keeping structural
heads untouched. Smooth transition from less-Poe to more-Poe.

Usage:
    uv run --extra viz python scripts/fig_poe_style_dial.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from sixteen_voices import load_adapted_model, load_tokenizer
from sixteen_voices.model import get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_DIR = Path("figures")

# Poe's style heads: recovery > 0.2 from knockout experiments
# (only heads that actually contribute to Poe's style, not all semantic heads)
STYLE_HEADS = [2, 3, 5, 13, 14, 15]
SCALES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SEED = 42
MAX_NEW = 100
TEMP = 0.7

PROMPTS = [
    "It was a dark and stormy",
    "In the dark of night",
]

SCALE_COLORS = {
    0.25: "#bfdbfe",
    0.5:  "#dbeafe",
    0.75: "#eff6ff",
    1.0:  "#f9fafb",
    1.25: "#fef3c7",
    1.5:  "#fecaca",
}

SCALE_BORDERS = {
    0.25: "#60a5fa",
    0.5:  "#93c5fd",
    0.75: "#bfdbfe",
    1.0:  "#d1d5db",
    1.25: "#fcd34d",
    1.5:  "#f87171",
}

SCALE_LABELS = {
    0.25: "0.25× least Poe",
    0.5:  "0.5×",
    0.75: "0.75×",
    1.0:  "1×    normal",
    1.25: "1.25×",
    1.5:  "1.5×  most Poe",
}


def generate(model, tokenizer, prompt, scale):
    torch.manual_seed(SEED)
    scales = {h: scale for h in STYLE_HEADS}
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_hook(scales))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, temperature=TEMP,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    hook.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def make_figure(prompt, samples, output_name):
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    heads_str = ", ".join(f"H{h}" for h in STYLE_HEADS)
    ax.text(0.5, 0.97,
            "Dialing Poe's style heads",
            fontsize=16, fontweight="bold", ha="center", va="top",
            transform=ax.transAxes)
    ax.text(0.5, 0.93,
            f'6 Poe-specific heads ({heads_str}) · other 10 heads untouched',
            fontsize=9, ha="center", va="top", color="#6b7280",
            transform=ax.transAxes)
    ax.text(0.5, 0.90,
            f'Prompt: "{prompt}"  ·  seed = 42',
            fontsize=9, ha="center", va="top", color="#6b7280",
            transform=ax.transAxes)

    n = len(SCALES)
    box_height = 0.12
    gap = 0.015
    start_y = 0.84

    for i, scale in enumerate(SCALES):
        text = samples[scale]
        if len(text) > 260:
            cut = text[:260].rfind(" ")
            if cut < 100:
                cut = 260
            text = text[:cut] + " ..."

        wrapped = textwrap.fill(text, width=80)

        y_top = start_y - i * (box_height + gap)
        label = SCALE_LABELS[scale]
        bg = SCALE_COLORS[scale]
        border = SCALE_BORDERS[scale]

        rect = mpatches.FancyBboxPatch(
            (0.02, y_top - box_height), 0.96, box_height,
            boxstyle="round,pad=0.01",
            facecolor=bg, edgecolor=border, linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        label_color = "#1e40af" if scale < 1 else "#991b1b" if scale > 1 else "#374151"
        ax.text(0.04, y_top - 0.015, label,
                fontsize=10, fontweight="bold", va="top",
                color=label_color, transform=ax.transAxes)

        ax.text(0.04, y_top - 0.045, wrapped,
                fontsize=8.5, va="top", family="serif", style="italic",
                color="#1f2937", transform=ax.transAxes,
                linespacing=1.4)

    # Arrow
    arrow_x = 0.008
    bottom_y = start_y - (n - 1) * (box_height + gap) - box_height + 0.02
    ax.annotate("", xy=(arrow_x, bottom_y),
                xytext=(arrow_x, start_y),
                arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=2),
                transform=ax.transAxes)
    ax.text(arrow_x - 0.005, (start_y + bottom_y) / 2,
            "style", fontsize=7, rotation=90, va="center", ha="center",
            color="#9ca3af", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / output_name, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved figures/{output_name}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tokenizer = load_tokenizer()
    model = load_adapted_model(str(ADAPTERS_DIR / "poe" / "adapter"))

    all_samples = {}

    for prompt in PROMPTS:
        samples = {}
        for scale in SCALES:
            samples[scale] = generate(model, tokenizer, prompt, scale)
            print(f'  {scale}×: {samples[scale][:100]}...')
        all_samples[prompt] = samples

        safe = prompt.lower().replace(" ", "_")[:30]
        make_figure(prompt, samples, f"poe_style_dial_{safe}.png")

    # Save
    out = {
        "author": "poe",
        "style_heads": STYLE_HEADS,
        "seed": SEED,
        "prompts": {
            p: {str(s): t for s, t in samps.items()}
            for p, samps in all_samples.items()
        },
    }
    with open("outputs/poe_style_dial_samples.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved outputs/poe_style_dial_samples.json")


if __name__ == "__main__":
    main()