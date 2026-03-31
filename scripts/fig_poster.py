#!/usr/bin/env python3
"""Generate poster figures for ML Prague.

Produces:
1. Steering before/after text comparison panels
2. Two-layer style diagram (structural vs semantic)
3. Pipeline flow diagram
4. Semantic feature token highlights

Usage:
    uv run python scripts/fig_poster.py
"""

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def fig_steering_examples():
    """Before/after steering text panels — the 'wow' figure."""
    examples = [
        {
            "title": "Poe + Simplicity",
            "subtitle": "sentence length: 23.9 → 4.9 words (20/20 seeds)",
            "baseline": (
                "and the trees began to have to stop him from his bed. "
                "The dark and sky wept. The dark sky above the clouds "
                "seemed to go away and, in the night above the clouds."
            ),
            "steered": (
                "It was dark. I went to sleep. It was dark. I woke up. "
                "It was dark. We could find a car. It was dark and it "
                "was night."
            ),
            "color": "#e8a735",
            "feature": "f665 (simplicity, head-independent)",
        },
        {
            "title": "Grimm + Dialogue",
            "subtitle": "narration → characters talk (scale 5)",
            "baseline": (
                "a little frog who was very sad. He sailed on, he had "
                "had a dream of sailing on it. The queen was very wise."
            ),
            "steered": (
                "a little frog. The frog loved to bounce. The girl said, "
                "'I have to go to the pond!' So the girl asked her father."
            ),
            "color": "#55a868",
            "feature": "f1777 + f689 (dialogue, H3-controlled)",
        },
        {
            "title": "Grimm + Questions",
            "subtitle": "declarative → interrogative",
            "baseline": (
                "a very clever fox. He loved to go to the forest; "
                "so he went so far in and he saw a rabbit."
            ),
            "steered": (
                "'It is like a little frog?' 'I want to be that?' "
                "said the frog, 'I go to the mill. I will give a try?'"
            ),
            "color": "#c44e52",
            "feature": "f329 (questions)",
        },
    ]

    fig, axes = plt.subplots(len(examples), 1, figsize=(14, len(examples) * 3.2))
    fig.suptitle("SAE Feature Steering: inject a direction, change the style",
                 fontsize=18, fontweight="bold", y=0.98)

    for i, ex in enumerate(examples):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Title + feature
        ax.text(0.0, 0.95, ex["title"], fontsize=15, fontweight="bold",
                color=ex["color"], transform=ax.transAxes, va="top")
        ax.text(0.0, 0.78, ex["subtitle"], fontsize=10, color="#666",
                transform=ax.transAxes, va="top")
        ax.text(0.0, 0.65, ex["feature"], fontsize=9, color="#999",
                transform=ax.transAxes, va="top", style="italic")

        # Baseline
        ax.text(0.02, 0.50, "Baseline:", fontsize=9, fontweight="bold",
                color="#999", transform=ax.transAxes, va="top")
        ax.text(0.02, 0.38, fill(ex["baseline"], width=85),
                fontsize=10, color="#666", family="serif",
                transform=ax.transAxes, va="top", style="italic")

        # Arrow
        ax.annotate("", xy=(0.5, 0.18), xytext=(0.5, 0.30),
                     arrowprops=dict(arrowstyle="->", color=ex["color"],
                                     lw=2.5), transform=ax.transAxes)

        # Steered
        ax.text(0.02, 0.17, "Steered:", fontsize=9, fontweight="bold",
                color=ex["color"], transform=ax.transAxes, va="top")
        ax.text(0.02, 0.05, fill(ex["steered"], width=85),
                fontsize=10, color=ex["color"], family="serif",
                transform=ax.transAxes, va="top", fontweight="bold",
                style="italic")

        # Separator
        if i < len(examples) - 1:
            ax.plot([0, 1], [-0.02, -0.02], color="#ddd", linewidth=1,
                    transform=ax.transAxes, clip_on=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES_DIR / "poster_steering_examples.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_two_layers():
    """The main finding: structural vs semantic style layers."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(8, 8.7, "Two Layers of Style",
            ha="center", fontsize=22, fontweight="bold")
    ax.text(8, 8.2, "The SAE decomposes author style into shared structure and unique identity",
            ha="center", fontsize=12, color="#666")

    # ── Left panel: Structural ──
    struct_box = FancyBboxPatch((0.5, 1.0), 6.5, 6.5,
                                 boxstyle="round,pad=0.3",
                                 facecolor="#55a868", alpha=0.08,
                                 edgecolor="#55a868", linewidth=2)
    ax.add_patch(struct_box)

    ax.text(3.75, 7.2, "STRUCTURAL", ha="center", fontsize=16,
            fontweight="bold", color="#55a868")
    ax.text(3.75, 6.7, "Shared axes · Universal knobs · 75-100% win rate",
            ha="center", fontsize=9, color="#55a868")

    struct_features = [
        ("Simplicity (f665)", "sentence length ↓", "9.1 → 6.0 words", "HEAD-INDEPENDENT"),
        ("Complexity (f883+)", "sentence length ↑", "8.1 → 12.2 words", "H3-controlled"),
        ("Dialogue (f1777+)", "quotes ↑", "1.3 → 3.4 marks", "H3-controlled"),
        ("Questions (f329)", "? marks ↑", "declarative → interrogative", ""),
        ("Verse (f344)", "line breaks", "prose → verse structure", ""),
    ]

    for j, (name, effect, metric, note) in enumerate(struct_features):
        y = 6.0 - j * 1.0
        ax.text(1.0, y, name, fontsize=11, fontweight="bold", color="#333")
        ax.text(3.5, y, effect, fontsize=10, color="#55a868")
        ax.text(5.2, y, metric, fontsize=9, color="#666", style="italic")
        if note:
            ax.text(1.0, y - 0.3, note, fontsize=7, color="#999")

    ax.text(3.75, 1.3, "Works on ANY model — base or adapted",
            ha="center", fontsize=10, fontweight="bold", color="#55a868",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#55a868",
                      alpha=0.1, edgecolor="#55a868"))

    # ── Right panel: Semantic ──
    sem_box = FancyBboxPatch((8.5, 1.0), 7.0, 6.5,
                              boxstyle="round,pad=0.3",
                              facecolor="#c44e52", alpha=0.08,
                              edgecolor="#c44e52", linewidth=2)
    ax.add_patch(sem_box)

    ax.text(12.0, 7.2, "SEMANTIC", ha="center", fontsize=16,
            fontweight="bold", color="#c44e52")
    ax.text(12.0, 6.7, "Unique identity · Detectable everywhere · Adapter-specific steering",
            ha="center", fontsize=9, color="#c44e52")

    sem_features = [
        ("Dark f1224", '"not quite a smile, not quite a frown"', "uncanny negation"),
        ("Dark f562", '"looking in, looking in, looking in"', "obsessive observation"),
        ("Cozy f1988", '"steam rose from the meat"', "food descriptions"),
        ("Cozy f930", '"wool soft against her fingers"', "tactile comfort"),
        ("Carroll f815", '"purring, not growling," said Alice', "Wonderland dialogue"),
    ]

    for j, (name, quote, label) in enumerate(sem_features):
        y = 6.0 - j * 1.0
        ax.text(9.0, y, name, fontsize=11, fontweight="bold", color="#333")
        ax.text(11.0, y, quote, fontsize=9, color="#c44e52", style="italic")
        ax.text(9.0, y - 0.3, label, fontsize=8, color="#999")

    ax.text(12.0, 1.3, "Steers ONLY with matching adapter",
            ha="center", fontsize=10, fontweight="bold", color="#c44e52",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#c44e52",
                      alpha=0.1, edgecolor="#c44e52"))

    # Connecting text at bottom
    ax.text(8, 0.4,
            "Every author is primarily semantic: Harris has 0 structural, 40 semantic features. "
            "What makes an author unique isn't sentence length — it's content.",
            ha="center", fontsize=10, color="#666", style="italic")

    out = FIGURES_DIR / "poster_two_layers.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_pipeline():
    """Pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(8, 3.7, "The Pipeline", ha="center", fontsize=16,
            fontweight="bold")

    steps = [
        {"x": 1.3, "label": "Design\nsynth controls", "detail": "minimalist,\ndialogue, cozy...",
         "color": "#4c72b0"},
        {"x": 4.0, "label": "Train\nSAE", "detail": "TopK, 2048\nfeatures, k=16",
         "color": "#55a868"},
        {"x": 6.7, "label": "Label with\ncontrols", "detail": "3-way check:\ntokens + synthetics",
         "color": "#e8a735"},
        {"x": 9.4, "label": "Connect\nto heads", "detail": "feature-head\ncorrelation (BH)",
         "color": "#c44e52"},
        {"x": 12.1, "label": "Steer &\nmeasure", "detail": "win rates\nacross 20 seeds",
         "color": "#8172b2"},
        {"x": 14.8, "label": "Two layers\nof style", "detail": "structural\n+ semantic",
         "color": "#333333"},
    ]

    for i, step in enumerate(steps):
        x = step["x"]
        box = FancyBboxPatch((x - 1.0, 1.0), 2.0, 2.0,
                              boxstyle="round,pad=0.2",
                              facecolor=step["color"], alpha=0.12,
                              edgecolor=step["color"], linewidth=2)
        ax.add_patch(box)

        ax.text(x, 2.5, step["label"], ha="center", va="center",
                fontsize=10, fontweight="bold", color=step["color"])
        ax.text(x, 1.5, step["detail"], ha="center", va="center",
                fontsize=8, color="#666", linespacing=1.3)

        # Arrow to next
        if i < len(steps) - 1:
            next_x = steps[i + 1]["x"]
            ax.annotate("", xy=(next_x - 1.15, 2.0),
                        xytext=(x + 1.15, 2.0),
                        arrowprops=dict(arrowstyle="-|>", color="#999",
                                        lw=2, mutation_scale=15))

    ax.text(8, 0.3,
            "Synthetics are ground truth that existed before the SAE. "
            "Labels are grounded, not post-hoc guesses.",
            ha="center", fontsize=10, color="#666", style="italic")

    out = FIGURES_DIR / "poster_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def fig_head_roles_compact():
    """Compact head roles for poster — just the key finding."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(6, 4.7, "What Each Head Computes", ha="center",
            fontsize=16, fontweight="bold")

    heads = [
        {"x": 1.5, "name": "H11", "color": "#4c72b0",
         "line1": "Workhorse", "line2": "66% of authors",
         "line3": "17 SAE features\nMOSTLY OPAQUE"},
        {"x": 4.5, "name": "H3", "color": "#55a868",
         "line1": "Style Reader", "line2": "107 SAE features",
         "line3": "reads ALL\ninterpretable axes"},
        {"x": 7.5, "name": "H14", "color": "#c44e52",
         "line1": "Formality Enforcer", "line2": "46 features",
         "line3": "helps Homer/Milton\nhurts Shelley/Wilde"},
        {"x": 10.5, "name": "MLP", "color": "#e8a735",
         "line1": "Multi-Head Mix", "line2": "27 features",
         "line3": "head-independent\nincl. STRONGEST\nstyle direction"},
    ]

    for h in heads:
        box = FancyBboxPatch((h["x"] - 1.3, 0.5), 2.6, 3.5,
                              boxstyle="round,pad=0.2",
                              facecolor=h["color"], alpha=0.1,
                              edgecolor=h["color"], linewidth=2)
        ax.add_patch(box)

        ax.text(h["x"], 3.7, h["name"], ha="center", fontsize=16,
                fontweight="bold", color=h["color"])
        ax.text(h["x"], 3.1, h["line1"], ha="center", fontsize=11,
                fontweight="bold", color="#333")
        ax.text(h["x"], 2.6, h["line2"], ha="center", fontsize=10,
                color="#666")
        ax.text(h["x"], 1.6, h["line3"], ha="center", fontsize=9,
                color="#555", linespacing=1.4)

    out = FIGURES_DIR / "poster_head_roles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print("Generating poster figures...")
    fig_steering_examples()
    fig_two_layers()
    fig_pipeline()
    fig_head_roles_compact()
    print("Done.")


if __name__ == "__main__":
    main()
