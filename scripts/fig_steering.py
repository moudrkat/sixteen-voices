#!/usr/bin/env python3
"""Figure: steering curves — how PPL changes as you scale each head from 0× to 2×.

Shows:
1. Steering curves for selected interesting authors (Poe, Grimm, Twain, Browne)
2. Aggregate: for all 82 authors, how does scaling H11 vs H14 differ?

Usage:
    uv run --extra viz python scripts/fig_steering.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SWEEP_JSON = Path("outputs/steering_sweep.json")
OUTPUT_DIR = Path("figures")


def main():
    with open(SWEEP_JSON) as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Figure 1: Individual author steering curves ---
    showcase = ["poe", "grimm", "twain", "browne"]
    showcase = [a for a in showcase if a in data]

    fig, axes = plt.subplots(1, len(showcase), figsize=(4 * len(showcase), 3.5),
                             sharey=False)
    if len(showcase) == 1:
        axes = [axes]

    for ax, author in zip(axes, showcase):
        d = data[author]
        full_ppl = d["full_ppl"]
        base_ppl = d["base_ppl"]

        for head_name, curve in sorted(d["curves"].items()):
            scales = sorted(curve.keys(), key=float)
            ppls = [curve[s] for s in scales]
            x = [float(s) for s in scales]

            # Normalize: show as % change from full adapter
            y = [(p - full_ppl) / full_ppl * 100 for p in ppls]

            rec = d["head_recovery"].get(head_name, 0)
            lw = 2.5 if abs(rec) > 0.3 else 1.2
            alpha = 1.0 if abs(rec) > 0.3 else 0.5
            ax.plot(x, y, "o-", label=f"{head_name} (rec={rec:+.2f})",
                    linewidth=lw, alpha=alpha, markersize=4)

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=1.0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Scale factor")
        ax.set_title(author.capitalize(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.1, 2.1)

    axes[0].set_ylabel("PPL change from full adapter (%)")
    fig.suptitle("Steering curves: PPL vs head scale factor", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "steering_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/steering_curves.png")

    # --- Figure 2: Aggregate H11 vs H14 across all authors ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, target_head in zip(axes, ["H11", "H14"]):
        for author, d in sorted(data.items()):
            if target_head not in d["curves"]:
                continue
            curve = d["curves"][target_head]
            full_ppl = d["full_ppl"]
            rec = d["head_recovery"].get(target_head, 0)

            scales = sorted(curve.keys(), key=float)
            x = [float(s) for s in scales]
            y = [(curve[s] - full_ppl) / full_ppl * 100 for s in scales]

            # Color by recovery
            if rec > 0.3:
                color, alpha = "#ef4444", 0.4
            elif rec < -0.2:
                color, alpha = "#3b82f6", 0.4
            else:
                color, alpha = "#9ca3af", 0.2

            ax.plot(x, y, "-", color=color, alpha=alpha, linewidth=0.8)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axvline(x=1.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel("PPL change (%)")
        ax.set_title(f"{target_head} steering across all authors", fontsize=12)
        ax.set_xlim(-0.1, 2.1)

        # Legend
        from matplotlib.lines import Line2D
        ax.legend(
            [Line2D([0], [0], color="#ef4444", lw=2),
             Line2D([0], [0], color="#3b82f6", lw=2),
             Line2D([0], [0], color="#9ca3af", lw=2)],
            ["rec > 0.3 (important)", "rec < -0.2 (hurts)", "near zero"],
            fontsize=8, loc="upper left",
        )

    fig.suptitle(f"How all {len(data)} authors respond to H11 vs H14 scaling", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "steering_aggregate.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/steering_aggregate.png")

    # --- Figure 3: Head roles summary ---
    # Combine attention patterns + knockout importance
    attn_path = Path("outputs/head_attention_patterns.json")
    knockout_path = Path("outputs/knockout_all_heads.json")

    if attn_path.exists() and knockout_path.exists():
        with open(attn_path) as f:
            attn_data = json.load(f)
        with open(knockout_path) as f:
            knockout_data = json.load(f)

        base_attn = attn_data["base"]["heads"]

        # Compute mean recovery per head across all authors
        mean_recovery = {}
        std_recovery = {}
        for h in range(16):
            recs = [knockout_data[a]["head_recovery"][f"H{h}"]
                    for a in knockout_data]
            mean_recovery[h] = np.mean(recs)
            std_recovery[h] = np.std(recs)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))

        # Panel 1: Mean recovery vs entropy
        ax = axes[0, 0]
        entropies = [base_attn[f"H{h}"]["entropy"] for h in range(16)]
        means = [mean_recovery[h] for h in range(16)]
        stds = [std_recovery[h] for h in range(16)]
        ax.errorbar(entropies, means, yerr=stds, fmt="o", capsize=3, color="#6366f1")
        for h in range(16):
            ax.annotate(f"H{h}", (entropies[h], means[h]),
                        fontsize=7, ha="left", va="bottom")
        ax.set_xlabel("Attention entropy (base model)")
        ax.set_ylabel("Mean knockout recovery")
        ax.set_title("Recovery vs attention focus")
        ax.axhline(y=0, color="gray", linewidth=0.5)

        # Panel 2: Mean recovery vs prev-token fraction
        ax = axes[0, 1]
        prev_fracs = [base_attn[f"H{h}"]["prev_token_frac"] for h in range(16)]
        ax.errorbar(prev_fracs, means, yerr=stds, fmt="o", capsize=3, color="#6366f1")
        for h in range(16):
            ax.annotate(f"H{h}", (prev_fracs[h], means[h]),
                        fontsize=7, ha="left", va="bottom")
        ax.set_xlabel("Previous-token attention (base model)")
        ax.set_ylabel("Mean knockout recovery")
        ax.set_title("Recovery vs previous-token bias")
        ax.axhline(y=0, color="gray", linewidth=0.5)

        # Panel 3: Recovery std (variability) per head
        ax = axes[1, 0]
        colors = ["#ef4444" if s > 0.2 else "#9ca3af" for s in stds]
        ax.bar(range(16), stds, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(16))
        ax.set_xticklabels([f"H{h}" for h in range(16)], fontsize=8)
        ax.set_ylabel("Std of recovery across authors")
        ax.set_title("Which heads are most author-specific?")

        # Panel 4: Head type classification
        ax = axes[1, 1]
        patterns = [base_attn[f"H{h}"]["pattern"] for h in range(16)]
        # Categorize
        cats = []
        for p in patterns:
            if "previous-token" in p:
                cats.append("prev-token")
            elif "focused" in p:
                cats.append("focused")
            elif "local-window" in p:
                cats.append("local")
            else:
                cats.append("semantic")
        cat_colors = {
            "prev-token": "#f97316",
            "focused": "#eab308",
            "local": "#22c55e",
            "semantic": "#6366f1",
        }
        bar_colors = [cat_colors[c] for c in cats]
        ax.bar(range(16), means, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(16))
        ax.set_xticklabels([f"H{h}" for h in range(16)], fontsize=8)
        ax.set_ylabel("Mean recovery")
        ax.set_title("Head roles (color = attention type)")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        from matplotlib.patches import Patch
        ax.legend(
            [Patch(color=c) for c in cat_colors.values()],
            cat_colors.keys(), fontsize=8, loc="upper right",
        )

        fig.suptitle("Head roles in TinyStories-1Layer-21M", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "head_roles.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved figures/head_roles.png")


if __name__ == "__main__":
    main()