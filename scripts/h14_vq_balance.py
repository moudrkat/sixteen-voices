#!/usr/bin/env python3
"""Why is H14 polarizing? The V-Q balance explanation.

For each author, computes what fraction of H14's LoRA weight change
goes into the value projection vs the query projection. Authors where
LoRA concentrated H14's change in V (what the head outputs) have
positive H14 recovery. Authors where LoRA concentrated in Q (what
the head attends to) have negative recovery.

Correlation: r = 0.71, explaining ~50% of H14 variance.

Usage:
    uv run python scripts/h14_vq_balance.py

Outputs:
    outputs/h14_vq_balance.json
    figures/h14_vq_balance.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices.adapter import load_adapter_deltas

KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_JSON = Path("outputs/h14_vq_balance.json")
OUTPUT_FIG = Path("figures/h14_vq_balance.png")
HEAD_DIM = 64
NUM_HEADS = 16

# Authors to label in the scatter plots
LABEL_AUTHORS = [
    "browne", "poe", "homer", "melville", "milton",
    "carroll", "grimm",
    "shelley", "twain", "burnett", "barrie",
]

# Synthetic controls with known properties
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

STYLE_TYPE_COLORS = {
    "vocabulary": "#991b1b",
    "mixed": "#888888",
    "structure": "#1e40af",
}


def main():
    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)

    data = []

    for author in ko:
        adapter_path = ADAPTERS_DIR / author / "adapter"
        if not adapter_path.exists():
            continue

        deltas = load_adapter_deltas(str(adapter_path))

        # Per-head Frobenius norms for Q and V projections
        q_norms = [
            float(torch.norm(deltas["q_proj"][h * HEAD_DIM:(h + 1) * HEAD_DIM]).item())
            for h in range(NUM_HEADS)
        ]
        v_norms = [
            float(torch.norm(deltas["v_proj"][h * HEAD_DIM:(h + 1) * HEAD_DIM]).item())
            for h in range(NUM_HEADS)
        ]

        q_total = sum(q_norms)
        v_total = sum(v_norms)

        h14_q_frac = q_norms[14] / q_total if q_total > 0 else 0
        h14_v_frac = v_norms[14] / v_total if v_total > 0 else 0
        vq_balance = h14_v_frac - h14_q_frac

        data.append({
            "author": author,
            "h14_recovery": ko[author]["head_recovery"]["H14"],
            "h14_v_frac": h14_v_frac,
            "h14_q_frac": h14_q_frac,
            "vq_balance": vq_balance,
            "h14_v_norm": v_norms[14],
            "h14_q_norm": q_norms[14],
        })

    # Correlation
    h14 = np.array([d["h14_recovery"] for d in data])
    vq = np.array([d["vq_balance"] for d in data])
    v_frac = np.array([d["h14_v_frac"] for d in data])
    q_frac = np.array([d["h14_q_frac"] for d in data])

    r_vq = float(np.corrcoef(vq, h14)[0, 1])
    r_v = float(np.corrcoef(v_frac, h14)[0, 1])
    r_q = float(np.corrcoef(q_frac, h14)[0, 1])

    print(f"N = {len(data)}")
    print(f"Correlation(V-Q balance, H14 recovery) = {r_vq:+.3f}  (R² = {r_vq**2:.3f})")
    print(f"Correlation(V fraction,  H14 recovery) = {r_v:+.3f}")
    print(f"Correlation(Q fraction,  H14 recovery) = {r_q:+.3f}")

    # --- Figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: V-Q balance vs H14 recovery (the main result)
    colors = ["#991b1b" if d["h14_recovery"] > 0.2 else
              "#1e40af" if d["h14_recovery"] < -0.2 else
              "#888888" for d in data]

    ax1.scatter(vq, h14, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)

    # Regression line
    slope, intercept = np.polyfit(vq, h14, 1)
    x_line = np.linspace(vq.min(), vq.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)

    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Labels
    author_to_d = {d["author"]: d for d in data}
    for author in LABEL_AUTHORS:
        if author not in author_to_d:
            continue
        d = author_to_d[author]
        color = "#991b1b" if d["h14_recovery"] > 0.2 else \
                "#1e40af" if d["h14_recovery"] < -0.2 else "#666666"
        ax1.annotate(
            author.capitalize(),
            xy=(d["vq_balance"], d["h14_recovery"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax1.set_xlabel("H14 V-Q balance\n(V fraction − Q fraction of LoRA weight)", fontsize=9)
    ax1.set_ylabel("H14 knockout recovery", fontsize=9)
    ax1.set_title(f"V-Q balance predicts H14 recovery\nr = {r_vq:+.2f}, R² = {r_vq**2:.2f}",
                  fontsize=11, fontweight="bold")

    # Panel 2: V fraction vs Q fraction, colored by H14 sign
    ax2.scatter(q_frac, v_frac, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)

    # Diagonal (equal V and Q)
    lim = max(q_frac.max(), v_frac.max()) * 1.05
    ax2.plot([0, lim], [0, lim], color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.text(lim * 0.65, lim * 0.72, "V = Q", fontsize=7, color="gray", rotation=38)
    ax2.text(lim * 0.3, lim * 0.8, "more V\n(output)", fontsize=8, color="#991b1b", alpha=0.4,
             ha="center")
    ax2.text(lim * 0.8, lim * 0.3, "more Q\n(routing)", fontsize=8, color="#1e40af", alpha=0.4,
             ha="center")

    for author in LABEL_AUTHORS:
        if author not in author_to_d:
            continue
        d = author_to_d[author]
        color = "#991b1b" if d["h14_recovery"] > 0.2 else \
                "#1e40af" if d["h14_recovery"] < -0.2 else "#666666"
        ax2.annotate(
            author.capitalize(),
            xy=(d["h14_q_frac"], d["h14_v_frac"]),
            xytext=(6, 3), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax2.set_xlabel("H14 Q fraction (of total Q weight)", fontsize=9)
    ax2.set_ylabel("H14 V fraction (of total V weight)", fontsize=9)
    ax2.set_title("H14: value vs query weight allocation\n"
                  "red = H14 helps, blue = H14 hurts",
                  fontsize=11, fontweight="bold")

    # Panel 3: Synthetic controls — V-Q balance by style type
    synth_data = [d for d in data if d["author"] in SYNTHETIC_AUTHORS]
    if synth_data:
        # Sort by V-Q balance
        synth_data.sort(key=lambda d: d["vq_balance"], reverse=True)
        synth_names = [d["author"] for d in synth_data]
        synth_vq = [d["vq_balance"] for d in synth_data]
        synth_colors = [STYLE_TYPE_COLORS[SYNTHETIC_AUTHORS[d["author"]]] for d in synth_data]

        bars = ax3.barh(range(len(synth_data)), synth_vq, color=synth_colors,
                        edgecolor="white", linewidth=0.5, alpha=0.85)
        ax3.set_yticks(range(len(synth_data)))
        ax3.set_yticklabels([n.replace("_", " ") for n in synth_names], fontsize=8)
        ax3.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax3.set_xlabel("H14 V-Q balance", fontsize=9)
        ax3.set_title("Synthetic controls\nby style type", fontsize=11, fontweight="bold")
        ax3.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor="#991b1b", label="vocabulary-defined"),
            Patch(facecolor="#888888", label="mixed"),
            Patch(facecolor="#1e40af", label="structure-defined"),
        ]
        ax3.legend(handles=legend_items, fontsize=7, loc="lower right")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "n_authors": len(data),
        "correlations": {
            "vq_balance_vs_h14": r_vq,
            "v_frac_vs_h14": r_v,
            "q_frac_vs_h14": r_q,
        },
        "per_author": sorted(data, key=lambda d: d["h14_recovery"]),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()