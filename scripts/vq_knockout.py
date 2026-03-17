#!/usr/bin/env python3
"""V-only vs Q-only knockout: does V really work better in isolation?

For H14 across all 82 authors, tests three conditions:
  - V-only: keep only H14's V LoRA, zero all Q LoRA
  - Q-only: keep only H14's Q LoRA, zero all V LoRA
  - Both:   keep H14's Q+V LoRA (existing knockout result)

If the V-Q mechanism is real, V-only should consistently recover more
than Q-only for V-heavy authors, and the difference should correlate
with the V-Q balance.

Usage:
    uv run python scripts/vq_knockout.py

Outputs:
    outputs/vq_knockout.json
    figures/vq_knockout.png
"""

import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sixteen_voices import load_base_model, load_tokenizer, load_eval_text
from sixteen_voices.adapter import load_adapter_deltas, knockout_all_except
from sixteen_voices.text import compute_perplexity
ADAPTERS_DIR = Path("outputs/authors")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
VQ_BALANCE_PATH = Path("outputs/h14_vq_balance.json")
OUTPUT_PATH = Path("outputs/vq_knockout.json")
FIG_PATH = Path("figures/vq_knockout.png")

TARGET_HEAD = 14


def main():
    with open(KNOCKOUT_PATH) as f:
        existing = json.load(f)

    # Load V-Q balance data if available
    vq_balance = {}
    if VQ_BALANCE_PATH.exists():
        with open(VQ_BALANCE_PATH) as f:
            vq_data = json.load(f)
        for a in vq_data.get("per_author", []):
            vq_balance[a["author"]] = a.get("vq_balance", 0)

    author_dirs = sorted(
        d.name for d in ADAPTERS_DIR.iterdir()
        if (d / "adapter" / "adapter_config.json").exists()
    )
    print(f"Found {len(author_dirs)} authors")

    tokenizer = load_tokenizer()
    base_model = load_base_model()
    h = TARGET_HEAD

    results = {}
    t0 = time.time()

    for i, name in enumerate(author_dirs):
        if name not in existing:
            continue

        try:
            eval_text = load_eval_text(name, length=5000)
        except FileNotFoundError:
            continue

        base_ppl = existing[name]["base_ppl"]
        full_ppl = existing[name]["full_ppl"]
        both_recovery = existing[name]["head_recovery"][f"H{h}"]
        ppl_range = base_ppl - full_ppl

        if ppl_range <= 0:
            continue

        deltas = load_adapter_deltas(ADAPTERS_DIR / name / "adapter")

        # V-only: keep H14's V rows, zero all Q
        v_delta = knockout_all_except(deltas["v_proj"], h)
        ko = copy.deepcopy(base_model)
        attn = ko.transformer.h[0].attn.attention
        with torch.no_grad():
            attn.v_proj.weight.add_(v_delta)
        v_ppl = compute_perplexity(ko, tokenizer, eval_text)
        del ko

        # Q-only: keep H14's Q rows, zero all V
        q_delta = knockout_all_except(deltas["q_proj"], h)
        ko = copy.deepcopy(base_model)
        attn = ko.transformer.h[0].attn.attention
        with torch.no_grad():
            attn.q_proj.weight.add_(q_delta)
        q_ppl = compute_perplexity(ko, tokenizer, eval_text)
        del ko

        v_rec = (base_ppl - v_ppl) / ppl_range
        q_rec = (base_ppl - q_ppl) / ppl_range

        results[name] = {
            "v_only_recovery": round(float(v_rec), 4),
            "q_only_recovery": round(float(q_rec), 4),
            "both_recovery": both_recovery,
            "vq_balance": vq_balance.get(name, None),
        }

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(author_dirs) - i - 1)
        print(f"  [{i+1}/{len(author_dirs)}] {name:20s}  "
              f"V={v_rec:+.3f}  Q={q_rec:+.3f}  both={both_recovery:+.3f}  "
              f"({elapsed:.0f}s, ~{eta:.0f}s left)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"head": TARGET_HEAD, "authors": results}, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")

    # Stats
    v_recs = [r["v_only_recovery"] for r in results.values()]
    q_recs = [r["q_only_recovery"] for r in results.values()]
    n_v_wins = sum(1 for v, q in zip(v_recs, q_recs) if v > q)
    print(f"\nV-only > Q-only: {n_v_wins}/{len(results)} authors")
    print(f"Mean V-only: {np.mean(v_recs):+.3f}")
    print(f"Mean Q-only: {np.mean(q_recs):+.3f}")

    # Correlation with V-Q balance
    authors_with_vq = [a for a in results if results[a]["vq_balance"] is not None]
    if authors_with_vq:
        vq_vals = [results[a]["vq_balance"] for a in authors_with_vq]
        v_minus_q = [results[a]["v_only_recovery"] - results[a]["q_only_recovery"]
                     for a in authors_with_vq]
        r = np.corrcoef(vq_vals, v_minus_q)[0, 1]
        print(f"r(V-Q balance, V-only minus Q-only recovery) = {r:+.3f}")

    # Figure
    make_figure(results)
    print(f"\nTotal: {time.time() - t0:.0f}s")


def make_figure(results):
    authors = sorted(results.keys())
    v_rec = np.array([results[a]["v_only_recovery"] for a in authors])
    q_rec = np.array([results[a]["q_only_recovery"] for a in authors])
    both_rec = np.array([results[a]["both_recovery"] for a in authors])
    vq_diff = v_rec - q_rec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: V-only vs Q-only scatter
    # Color by which wins
    for i, a in enumerate(authors):
        color = "#991b1b" if vq_diff[i] > 0 else "#1e40af"
        ax1.scatter(q_rec[i], v_rec[i], c=color, s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.3)

    lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
            max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Q-only recovery", fontsize=10)
    ax1.set_ylabel("V-only recovery", fontsize=10)
    n_v = np.sum(vq_diff > 0)
    ax1.set_title(f"H14: V-only vs Q-only knockout\n"
                  f"V wins for {n_v}/{len(authors)} authors",
                  fontsize=11, fontweight="bold")

    # Label some extreme points
    extremes = np.argsort(np.abs(vq_diff))[-6:]
    for idx in extremes:
        ax1.annotate(authors[idx], xy=(q_rec[idx], v_rec[idx]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, alpha=0.8)

    # Panel 2: V-Q balance vs (V-only minus Q-only)
    authors_with_vq = [a for a in authors if results[a]["vq_balance"] is not None]
    if authors_with_vq:
        vq_vals = [results[a]["vq_balance"] for a in authors_with_vq]
        v_minus_q_rec = [results[a]["v_only_recovery"] - results[a]["q_only_recovery"]
                         for a in authors_with_vq]
        colors = ["#991b1b" if d > 0 else "#1e40af" for d in v_minus_q_rec]
        ax2.scatter(vq_vals, v_minus_q_rec, c=colors, s=30, alpha=0.7,
                   edgecolors="white", linewidth=0.3)
        ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        r = np.corrcoef(vq_vals, v_minus_q_rec)[0, 1]
        ax2.set_xlabel("V-Q balance (weight distribution)", fontsize=10)
        ax2.set_ylabel("V-only minus Q-only recovery", fontsize=10)
        ax2.set_title(f"V-Q balance predicts which projection\n"
                      f"works better in isolation (r = {r:+.2f})",
                      fontsize=11, fontweight="bold")

        extremes2 = np.argsort([abs(v) for v in v_minus_q_rec])[-6:]
        for idx in extremes2:
            ax2.annotate(authors_with_vq[idx],
                        xy=(vq_vals[idx], v_minus_q_rec[idx]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=6.5, alpha=0.8)

    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()