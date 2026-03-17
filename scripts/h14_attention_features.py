#!/usr/bin/env python3
"""Do base model attention statistics predict V-Q balance / H14 recovery?

For each author, runs the base model (no adapter) on their text and
extracts per-head attention statistics:
  - Entropy (diffuse vs focused attention)
  - Mean attention to previous token (positional bias)
  - Max attention weight (how peaked the distribution is)

Then correlates these with V-Q balance and H14 recovery.

If the base model's attention patterns on an author's text already differ
in ways that predict V-Q balance, a hypernetwork could use these as
input features.

Usage:
    uv run python scripts/h14_attention_features.py

Outputs:
    outputs/h14_attention_features.json
    figures/h14_attention_features.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices import load_base_model, load_tokenizer, load_eval_text
from sixteen_voices.model import get_attn_module
VQ_PATH = Path("outputs/h14_vq_balance.json")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
OUTPUT_JSON = Path("outputs/h14_attention_features.json")
OUTPUT_FIG = Path("figures/h14_attention_features.png")

NUM_HEADS = 16
MAX_TOKENS = 512

LABEL_AUTHORS = [
    "browne", "poe", "homer", "melville", "milton",
    "carroll", "grimm",
    "shelley", "twain", "burnett", "barrie",
    "unusual_vocab", "dialogue", "firstperson",
]


def extract_attention_stats(model, tokenizer, text: str) -> dict:
    """Run base model on text and extract per-head attention statistics."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    seq_len = inputs["input_ids"].shape[1]

    if seq_len < 10:
        return None

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, num_heads, seq_len, seq_len)
    attn = outputs.attentions[0][0]  # (num_heads, seq_len, seq_len)

    stats = {}
    for h in range(NUM_HEADS):
        head_attn = attn[h]  # (seq_len, seq_len)

        # Skip first token (has nowhere to attend but itself)
        head_attn = head_attn[1:]  # (seq_len-1, seq_len)

        # Entropy per position, then average
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1)
        mean_entropy = float(entropy.mean().item())

        # Max attention weight per position (how peaked)
        max_attn = float(head_attn.max(dim=-1).values.mean().item())

        # Attention to previous token (diagonal -1)
        prev_attn_vals = []
        for t in range(head_attn.shape[0]):
            actual_t = t + 1  # because we skipped first token
            if actual_t > 0:
                prev_attn_vals.append(float(head_attn[t, actual_t - 1].item()))
        mean_prev_attn = np.mean(prev_attn_vals) if prev_attn_vals else 0.0

        # Attention to first token (BOS)
        mean_bos_attn = float(head_attn[:, 0].mean().item())

        # Attention spread: std of attention weights per position
        attn_std = float(head_attn.std(dim=-1).mean().item())

        stats[f"H{h}"] = {
            "entropy": mean_entropy,
            "max_attn": max_attn,
            "prev_token_attn": mean_prev_attn,
            "bos_attn": mean_bos_attn,
            "attn_std": attn_std,
        }

    return stats


def main():
    with open(VQ_PATH) as f:
        vq_data = json.load(f)
    vq_by_author = {d["author"]: d for d in vq_data["per_author"]}

    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)

    print("Loading base model...")
    tokenizer = load_tokenizer()
    model = load_base_model()

    results = []
    authors = sorted(ko.keys())

    for i, author in enumerate(authors):
        if author not in vq_by_author:
            continue

        try:
            text = load_eval_text(author, length=5000)
        except FileNotFoundError:
            continue
        words = text.split()
        if len(words) < 200:
            continue

        # Take a chunk from the middle
        mid = len(words) // 3
        eval_text = " ".join(words[mid:mid + 400])

        stats = extract_attention_stats(model, tokenizer, eval_text)
        if stats is None:
            continue

        h14_stats = stats["H14"]
        entry = {
            "author": author,
            "h14_recovery": ko[author]["head_recovery"]["H14"],
            "vq_balance": vq_by_author[author]["vq_balance"],
            # H14 attention features
            "h14_entropy": h14_stats["entropy"],
            "h14_max_attn": h14_stats["max_attn"],
            "h14_prev_token": h14_stats["prev_token_attn"],
            "h14_bos_attn": h14_stats["bos_attn"],
            "h14_attn_std": h14_stats["attn_std"],
        }

        # Also store all heads' entropy for comparison
        for h in range(NUM_HEADS):
            entry[f"h{h}_entropy"] = stats[f"H{h}"]["entropy"]
            entry[f"h{h}_prev_token"] = stats[f"H{h}"]["prev_token_attn"]

        results.append(entry)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(authors)}] processed {author}")

    print(f"\nN = {len(results)}")

    # Correlations: H14 features vs V-Q balance and H14 recovery
    vq = np.array([r["vq_balance"] for r in results])
    h14 = np.array([r["h14_recovery"] for r in results])

    h14_metrics = ["h14_entropy", "h14_max_attn", "h14_prev_token", "h14_bos_attn", "h14_attn_std"]

    print(f"\n{'H14 attention feature':25s}  {'r(V-Q)':>8s}  {'r(H14)':>8s}")
    print("-" * 45)
    best_metric = None
    best_r = 0
    for m in h14_metrics:
        vals = np.array([r[m] for r in results])
        rv = float(np.corrcoef(vals, vq)[0, 1])
        rh = float(np.corrcoef(vals, h14)[0, 1])
        print(f"{m:25s}  {rv:+.3f}     {rh:+.3f}")
        if abs(rh) > abs(best_r):
            best_r = rh
            best_metric = m

    # Also check ALL heads' entropy vs H14 recovery
    print(f"\n{'Per-head entropy':25s}  {'r(V-Q)':>8s}  {'r(H14)':>8s}")
    print("-" * 45)
    for h in range(NUM_HEADS):
        vals = np.array([r[f"h{h}_entropy"] for r in results])
        rv = float(np.corrcoef(vals, vq)[0, 1])
        rh = float(np.corrcoef(vals, h14)[0, 1])
        flag = " ***" if abs(rh) > 0.3 else " *" if abs(rh) > 0.2 else ""
        print(f"h{h:02d}_entropy              {rv:+.3f}     {rh:+.3f}{flag}")

    # Check per-head prev_token attention
    print(f"\n{'Per-head prev_token':25s}  {'r(V-Q)':>8s}  {'r(H14)':>8s}")
    print("-" * 45)
    for h in range(NUM_HEADS):
        vals = np.array([r[f"h{h}_prev_token"] for r in results])
        rv = float(np.corrcoef(vals, vq)[0, 1])
        rh = float(np.corrcoef(vals, h14)[0, 1])
        flag = " ***" if abs(rh) > 0.3 else " *" if abs(rh) > 0.2 else ""
        print(f"h{h:02d}_prev_token           {rv:+.3f}     {rh:+.3f}{flag}")

    # Multi-feature: combine all heads' entropy into a feature vector
    # Use all 16 heads' entropy + prev_token as features
    from numpy.linalg import lstsq
    X = np.column_stack([
        np.array([r[f"h{h}_entropy"] for r in results]) for h in range(NUM_HEADS)
    ] + [
        np.array([r[f"h{h}_prev_token"] for r in results]) for h in range(NUM_HEADS)
    ])
    # Add intercept
    X = np.column_stack([X, np.ones(len(results))])

    # Predict V-Q balance
    coef_vq, res_vq, _, _ = lstsq(X, vq, rcond=None)
    pred_vq = X @ coef_vq
    ss_res = np.sum((vq - pred_vq) ** 2)
    ss_tot = np.sum((vq - vq.mean()) ** 2)
    r2_vq = 1 - ss_res / ss_tot

    # Predict H14 recovery
    coef_h14, res_h14, _, _ = lstsq(X, h14, rcond=None)
    pred_h14 = X @ coef_h14
    ss_res_h14 = np.sum((h14 - pred_h14) ** 2)
    ss_tot_h14 = np.sum((h14 - h14.mean()) ** 2)
    r2_h14 = 1 - ss_res_h14 / ss_tot_h14

    print(f"\nMulti-feature (32 features: 16 entropies + 16 prev_token):")
    print(f"  R²(V-Q balance) = {r2_vq:.3f}")
    print(f"  R²(H14 recovery) = {r2_h14:.3f}")
    print(f"  (Warning: {X.shape[1]} features / {len(results)} samples — likely overfit)")

    # Reduced feature set: just H14 entropy + H14 prev_token
    X_small = np.column_stack([
        np.array([r["h14_entropy"] for r in results]),
        np.array([r["h14_prev_token"] for r in results]),
        np.ones(len(results)),
    ])
    coef_s, _, _, _ = lstsq(X_small, h14, rcond=None)
    pred_s = X_small @ coef_s
    r2_small = 1 - np.sum((h14 - pred_s) ** 2) / ss_tot_h14

    coef_sv, _, _, _ = lstsq(X_small, vq, rcond=None)
    pred_sv = X_small @ coef_sv
    r2_small_vq = 1 - np.sum((vq - pred_sv) ** 2) / ss_tot

    print(f"\n2-feature (H14 entropy + H14 prev_token):")
    print(f"  R²(V-Q balance) = {r2_small_vq:.3f}")
    print(f"  R²(H14 recovery) = {r2_small:.3f}")

    # --- Figure ---
    bm = best_metric
    bm_vals = np.array([r[bm] for r in results])
    r_bm_h14 = float(np.corrcoef(bm_vals, h14)[0, 1])
    r_bm_vq = float(np.corrcoef(bm_vals, vq)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = ["#991b1b" if r["h14_recovery"] > 0.2 else
              "#1e40af" if r["h14_recovery"] < -0.2 else
              "#888888" for r in results]

    # Panel 1: best metric vs H14 recovery
    ax1.scatter(bm_vals, h14, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    slope, intercept = np.polyfit(bm_vals, h14, 1)
    x_line = np.linspace(bm_vals.min(), bm_vals.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.3)

    author_to_r = {r["author"]: r for r in results}
    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r["h14_recovery"] > 0.2 else \
                "#1e40af" if r["h14_recovery"] < -0.2 else "#666666"
        ax1.annotate(
            author.replace("_", " ").capitalize(),
            xy=(r[bm], r["h14_recovery"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax1.set_xlabel(f"Base model {bm.replace('h14_', 'H14 ')}\n(no adapter, just base model on author text)",
                   fontsize=9)
    ax1.set_ylabel("H14 knockout recovery", fontsize=9)
    ax1.set_title(f"Base model attention predicts H14 recovery?\n"
                  f"r = {r_bm_h14:+.2f}",
                  fontsize=11, fontweight="bold")

    # Panel 2: best metric vs V-Q balance
    ax2.scatter(bm_vals, vq, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)
    slope2, intercept2 = np.polyfit(bm_vals, vq, 1)
    ax2.plot(x_line, slope2 * x_line + intercept2, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)

    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r["h14_recovery"] > 0.2 else \
                "#1e40af" if r["h14_recovery"] < -0.2 else "#666666"
        ax2.annotate(
            author.replace("_", " ").capitalize(),
            xy=(r[bm], r["vq_balance"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax2.set_xlabel(f"Base model {bm.replace('h14_', 'H14 ')}\n(no adapter, just base model on author text)",
                   fontsize=9)
    ax2.set_ylabel("H14 V-Q balance (from LoRA weights)", fontsize=9)
    ax2.set_title(f"Base model attention predicts V-Q balance?\n"
                  f"r = {r_bm_vq:+.2f}",
                  fontsize=11, fontweight="bold")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "n_authors": len(results),
        "best_single_feature": best_metric,
        "correlations": {
            f"{best_metric}_vs_h14": r_bm_h14,
            f"{best_metric}_vs_vq": r_bm_vq,
        },
        "multifeature_r2": {
            "full_32_features_vq": r2_vq,
            "full_32_features_h14": r2_h14,
            "h14_2_features_vq": r2_small_vq,
            "h14_2_features_h14": r2_small,
        },
        "per_author": results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()