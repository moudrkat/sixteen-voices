#!/usr/bin/env python3
"""Shuffle test: is each author's style about word choice or structure?

For each author:
  1. Take eval text (cleaned)
  2. Shuffle words within each sentence (destroys structure, keeps vocabulary)
  3. Score original and shuffled with both base and adapted model
  4. Compute: vocab_fraction = how much of the adaptation benefit survives shuffling

If adapted model improves PPL on both original AND shuffled text equally,
the style is vocabulary-driven (word choice matters, not order).
If the improvement vanishes on shuffled text, the style is structure-driven.

    vocab_fraction = (base_ppl_shuf - adapted_ppl_shuf) / (base_ppl_orig - adapted_ppl_orig)

Close to 1.0 = vocabulary-driven. Close to 0 or negative = structure-driven.

Then correlates with V-Q balance and H14 recovery.

Usage:
    uv run python scripts/h14_shuffle_test.py

Outputs:
    outputs/h14_shuffle_test.json
    figures/h14_shuffle_test.png
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices import load_base_model, load_adapted_model, load_tokenizer, load_eval_text
from sixteen_voices.text import compute_perplexity
ADAPTERS_DIR = Path("outputs/authors")
VQ_PATH = Path("outputs/h14_vq_balance.json")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
OUTPUT_JSON = Path("outputs/h14_shuffle_test.json")
OUTPUT_FIG = Path("figures/h14_shuffle_test.png")

LABEL_AUTHORS = [
    "browne", "poe", "homer", "melville", "milton",
    "carroll", "grimm",
    "shelley", "twain", "burnett", "barrie",
    "unusual_vocab", "dialogue", "firstperson", "minimalist",
]

SYNTHETIC_AUTHORS = {
    "unusual_vocab": "vocabulary",
    "reporter": "vocabulary",
    "simple_vocab": "vocabulary",
    "poet": "mixed",
    "dark": "mixed",
    "cozy": "mixed",
    "dialogue": "structure",
    "firstperson": "structure",
    "questioner": "structure",
    "repeater": "structure",
    "minimalist": "structure",
    "fabulist": "structure",
    "rambler": "structure",
}


def shuffle_within_sentences(text: str, rng: random.Random) -> str:
    """Shuffle words within each sentence, preserving sentence boundaries."""
    # Split into sentences (approximate)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    shuffled = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 2:
            # Keep first word capitalization, shuffle the rest
            first = words[0]
            rest = words[1:]
            rng.shuffle(rest)
            shuffled.append(first + " " + " ".join(rest))
        else:
            shuffled.append(sent)
    return " ".join(shuffled)


def main():
    # Load V-Q balance and knockout data
    with open(VQ_PATH) as f:
        vq_data = json.load(f)
    vq_by_author = {d["author"]: d for d in vq_data["per_author"]}

    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)

    # Load model and tokenizer once
    print("Loading base model...")
    tokenizer = load_tokenizer()
    base_model = load_base_model()

    rng = random.Random(42)
    results = []

    authors = sorted(ko.keys())
    for i, author in enumerate(authors):
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if not adapter_path.exists():
            continue

        try:
            text = load_eval_text(author, length=5000)
        except FileNotFoundError:
            continue

        words = text.split()
        if len(words) < 200:
            continue
        mid = len(words) // 3
        eval_text = " ".join(words[mid:mid + 500])

        # Create shuffled version
        shuffled_text = shuffle_within_sentences(eval_text, rng)

        # Score with base model
        with torch.no_grad():
            base_ppl_orig = compute_perplexity(base_model, tokenizer, eval_text)
            base_ppl_shuf = compute_perplexity(base_model, tokenizer, shuffled_text)

        # Score with adapted model
        adapted_model = load_adapted_model(adapter_path)
        with torch.no_grad():
            adapt_ppl_orig = compute_perplexity(adapted_model, tokenizer, eval_text)
            adapt_ppl_shuf = compute_perplexity(adapted_model, tokenizer, shuffled_text)

        # How much does the adapter help on original vs shuffled?
        benefit_orig = base_ppl_orig - adapt_ppl_orig
        benefit_shuf = base_ppl_shuf - adapt_ppl_shuf

        # Vocab fraction: how much benefit survives shuffling
        if abs(benefit_orig) > 0.1:
            vocab_frac = benefit_shuf / benefit_orig
        else:
            vocab_frac = 0.0

        h14_recovery = ko[author]["head_recovery"]["H14"]
        vq_balance = vq_by_author[author]["vq_balance"] if author in vq_by_author else None

        results.append({
            "author": author,
            "base_ppl_orig": float(base_ppl_orig),
            "base_ppl_shuf": float(base_ppl_shuf),
            "adapt_ppl_orig": float(adapt_ppl_orig),
            "adapt_ppl_shuf": float(adapt_ppl_shuf),
            "benefit_orig": float(benefit_orig),
            "benefit_shuf": float(benefit_shuf),
            "vocab_frac": float(vocab_frac),
            "h14_recovery": h14_recovery,
            "vq_balance": vq_balance,
        })

        print(f"  [{i+1}/{len(authors)}] {author:20s}  "
              f"benefit_orig={benefit_orig:+.1f}  benefit_shuf={benefit_shuf:+.1f}  "
              f"vocab_frac={vocab_frac:.2f}  H14={h14_recovery:+.2f}")

        # Free adapter memory
        del adapted_model

    # Filter to authors with V-Q data
    matched = [r for r in results if r["vq_balance"] is not None]

    vf = np.array([r["vocab_frac"] for r in matched])
    vq = np.array([r["vq_balance"] for r in matched])
    h14 = np.array([r["h14_recovery"] for r in matched])

    # Clip extreme vocab_frac values for robust correlation
    vf_clipped = np.clip(vf, -2, 3)

    r_vf_vq = float(np.corrcoef(vf_clipped, vq)[0, 1])
    r_vf_h14 = float(np.corrcoef(vf_clipped, h14)[0, 1])

    print(f"\nN = {len(matched)}")
    print(f"Correlation(vocab_frac, V-Q balance) = {r_vf_vq:+.3f}")
    print(f"Correlation(vocab_frac, H14 recovery) = {r_vf_h14:+.3f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: vocab_frac vs V-Q balance
    colors = ["#555555"] * len(matched)
    for i, r in enumerate(matched):
        if r["author"] in SYNTHETIC_AUTHORS:
            st = SYNTHETIC_AUTHORS[r["author"]]
            colors[i] = {"vocabulary": "#991b1b", "mixed": "#888", "structure": "#1e40af"}[st]

    ax1.scatter(vf_clipped, vq, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    slope, intercept = np.polyfit(vf_clipped, vq, 1)
    x_line = np.linspace(vf_clipped.min(), vf_clipped.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax1.axvline(1, color="gray", linewidth=0.5, alpha=0.3, linestyle=":")

    author_to_r = {r["author"]: r for r in matched}
    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r["h14_recovery"] > 0.2 else \
                "#1e40af" if r["h14_recovery"] < -0.2 else "#666666"
        ax1.annotate(
            author.replace("_", " ").capitalize(),
            xy=(np.clip(r["vocab_frac"], -2, 3), r["vq_balance"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax1.set_xlabel("Vocab fraction\n(share of adaptation benefit surviving word shuffle)", fontsize=9)
    ax1.set_ylabel("H14 V-Q balance (from LoRA weights)", fontsize=9)
    ax1.set_title(f"Shuffle test vs V-Q balance\nr = {r_vf_vq:+.2f}", fontsize=11, fontweight="bold")

    # Panel 2: vocab_frac vs H14 recovery
    colors2 = ["#991b1b" if r["h14_recovery"] > 0.2 else
               "#1e40af" if r["h14_recovery"] < -0.2 else
               "#888888" for r in matched]

    ax2.scatter(vf_clipped, h14, c=colors2, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    slope2, intercept2 = np.polyfit(vf_clipped, h14, 1)
    ax2.plot(x_line, slope2 * x_line + intercept2, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax2.axvline(1, color="gray", linewidth=0.5, alpha=0.3, linestyle=":")

    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r["h14_recovery"] > 0.2 else \
                "#1e40af" if r["h14_recovery"] < -0.2 else "#666666"
        ax2.annotate(
            author.replace("_", " ").capitalize(),
            xy=(np.clip(r["vocab_frac"], -2, 3), r["h14_recovery"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax2.set_xlabel("Vocab fraction\n(share of adaptation benefit surviving word shuffle)", fontsize=9)
    ax2.set_ylabel("H14 knockout recovery", fontsize=9)
    ax2.set_title(f"Shuffle test vs H14 recovery\nr = {r_vf_h14:+.2f}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "n_authors": len(matched),
        "correlations": {
            "vocab_frac_vs_vq": r_vf_vq,
            "vocab_frac_vs_h14": r_vf_h14,
        },
        "per_author": sorted(results, key=lambda r: r["vocab_frac"], reverse=True),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()