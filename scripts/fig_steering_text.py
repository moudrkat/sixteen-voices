#!/usr/bin/env python3
"""Figure: steered text samples — show how scaling heads changes the prose.

Shows text generated at 0×, 0.5×, 1× (normal), 1.5×, 2× for an author's
most important head. Visual proof that you can dial style up and down.

Usage:
    uv run --extra viz python scripts/fig_steering_text.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from sixteen_voices import NUM_HEADS, load_adapted_model, load_tokenizer
from sixteen_voices.model import get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
KNOCKOUT_JSON = Path("outputs/knockout_all_heads.json")
OUTPUT_DIR = Path("figures")

PROMPT = "It was a dark and stormy"
MAX_NEW = 80
TEMP = 0.7
SEED = 42
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


def generate_steered(model, tokenizer, prompt, head, scale):
    torch.manual_seed(SEED)
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_hook({head: scale}))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, temperature=TEMP,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    hook.remove()
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Remove prompt prefix
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def wrap_text(text, width=55):
    """Wrap text to given width."""
    words = text.split()
    lines = []
    current = []
    length = 0
    for w in words:
        if length + len(w) + 1 > width and current:
            lines.append(" ".join(current))
            current = [w]
            length = len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def make_author_panel(ax, author, head, samples, recovery):
    """Draw a single author's steering panel."""
    scale_labels = {0.0: "0×  (killed)", 0.5: "0.5×", 1.0: "1×  (normal)",
                    1.5: "1.5×", 2.0: "2×  (amplified)"}

    # Background gradient from light to dark
    n = len(SCALES)
    text_parts = []
    y = 0.95

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = f"{author.capitalize()} — H{head} (recovery = {recovery:+.2f})"
    ax.text(0.5, 0.99, title, fontsize=12, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)

    colors = ["#dbeafe", "#bfdbfe", "#f9fafb", "#fef3c7", "#fecaca"]
    spacing = 0.18

    for i, scale in enumerate(SCALES):
        text = samples[scale]
        wrapped = wrap_text(text, width=65)
        label = scale_labels[scale]

        box_y = 0.92 - i * spacing

        # Background box
        bbox = dict(boxstyle="round,pad=0.4", facecolor=colors[i],
                    edgecolor="#d1d5db", alpha=0.8)

        ax.text(0.02, box_y, f"{label}:", fontsize=8, fontweight="bold",
                va="top", transform=ax.transAxes, color="#374151")
        ax.text(0.02, box_y - 0.025, wrapped, fontsize=7.5,
                va="top", transform=ax.transAxes, family="serif",
                color="#1f2937", bbox=bbox, style="italic")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tokenizer = load_tokenizer()

    with open(KNOCKOUT_JSON) as f:
        knockout = json.load(f)

    # Authors and their most interesting heads
    authors = [
        ("poe", None),     # will pick best head
        ("carroll", None),
        ("grimm", None),
    ]

    # Find best head per author
    for i, (author, head) in enumerate(authors):
        if head is None:
            recovery = knockout[author]["head_recovery"]
            ranked = sorted(range(NUM_HEADS),
                            key=lambda h: abs(recovery[f"H{h}"]), reverse=True)
            authors[i] = (author, ranked[0])

    fig, axes = plt.subplots(len(authors), 1, figsize=(10, 5.5 * len(authors)))
    if len(authors) == 1:
        axes = [axes]

    all_samples = {}

    for ax, (author, head) in zip(axes, authors):
        print(f"Generating for {author}, H{head}...")
        adapter_path = ADAPTERS_DIR / author / "adapter"
        model = load_adapted_model(adapter_path)
        recovery = knockout[author]["head_recovery"][f"H{head}"]

        samples = {}
        for scale in SCALES:
            text = generate_steered(model, tokenizer, PROMPT, head, scale)
            samples[scale] = text
            print(f"  {scale}×: {text[:80]}...")

        make_author_panel(ax, author, head, samples, recovery)
        all_samples[author] = {
            "head": head,
            "recovery": recovery,
            "prompt": PROMPT,
            "seed": SEED,
            "samples": {str(s): t for s, t in samples.items()},
        }
        del model

    fig.suptitle(f'Steering: "{PROMPT}" — scaling one head from 0× to 2×',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "steering_text.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figures/steering_text.png")

    # Save samples JSON
    with open("outputs/steering_text_samples.json", "w") as f:
        json.dump(all_samples, f, indent=2)
    print("Saved outputs/steering_text_samples.json")


if __name__ == "__main__":
    main()