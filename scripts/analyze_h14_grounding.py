#!/usr/bin/env python3
"""Ground H14's SAE feature correlations in training text properties.

Measures text-level properties (I-usage, conversational verbs, sentence length,
etc.) for all authors and correlates them with H14 knockout recovery scores.

Usage:
    uv run python scripts/analyze_h14_grounding.py
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

SAE_DIR = Path("outputs/sae_topk16_2048")
AUTHORS_DIR = Path("data/authors")

CONV_VERBS = {
    "am", "was", "'m", "'ve", "'ll", "'d",
    "said", "asked", "replied", "think", "know",
    "want", "feel", "like", "need",
}


def measure_text(path):
    """Measure text-level properties of an author's training data."""
    with open(path) as f:
        text = f.read()

    words = text.split()
    total = len(words)
    if total < 100:
        return None

    # First-person "I" percentage
    i_count = words.count("I")
    i_pct = i_count / total * 100

    # Conversational verb percentage
    conv_count = sum(1 for w in words if w.lower() in CONV_VERBS)
    conv_pct = conv_count / total * 100

    # Average sentence length
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) > 2]
    avg_sent = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    # Question marks per 1000 words
    q_per_k = text.count("?") / total * 1000

    # Quotes per 1000 words
    quotes_per_k = text.count('"') / total * 1000

    return {
        "i_pct": i_pct,
        "conv_pct": conv_pct,
        "avg_sent": avg_sent,
        "q_per_k": q_per_k,
        "quotes_per_k": quotes_per_k,
    }


def main():
    # Load knockout scores
    with open("outputs/knockout_all_heads.json") as f:
        ko = json.load(f)

    # Load SAE feature-author matrix
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    sae_authors = d["authors"]
    matrix = d["matrix"].numpy()

    # Measure all authors
    results = []
    for author in sorted(ko.keys()):
        path = AUTHORS_DIR / f"{author}.txt"
        if not path.exists():
            continue
        metrics = measure_text(path)
        if metrics is None:
            continue
        metrics["author"] = author
        metrics["h14"] = ko[author]["head_recovery"]["H14"]
        results.append(metrics)

    # Print sorted by H14
    results.sort(key=lambda r: r["h14"], reverse=True)

    print(f"{'Author':<18} {'H14':>6} {'I%':>6} {'Conv%':>6} {'AvgSnt':>7} {'Q/1k':>6} {'Quot/1k':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['author']:<18} {r['h14']:>6.3f} {r['i_pct']:>6.1f} {r['conv_pct']:>6.1f} "
              f"{r['avg_sent']:>7.1f} {r['q_per_k']:>6.1f} {r['quotes_per_k']:>7.1f}")

    # Correlations with H14
    h14s = np.array([r["h14"] for r in results])
    print(f"\n{'Metric':<18} {'r':>8} {'p':>10} {'sig':>5}")
    print("-" * 45)
    for metric in ["i_pct", "conv_pct", "avg_sent", "q_per_k", "quotes_per_k"]:
        vals = np.array([r[metric] for r in results])
        r, p = stats.pearsonr(h14s, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{metric:<18} {r:>+8.3f} {p:>10.4f} {sig:>5}")

    # Top positive SAE feature correlations with H14
    print("\n\n=== SAE features most correlated with H14 ===")
    n_features = matrix.shape[1]
    corrs = []
    for f in range(n_features):
        col = matrix[:, f]
        if col.std() < 1e-8:
            continue
        h14_for_sae = np.array([ko[a]["head_recovery"]["H14"] for a in sae_authors])
        r, p = stats.pearsonr(col, h14_for_sae)
        corrs.append((f, r, p))
    corrs.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 5 POSITIVE:")
    for f, r, p in corrs[:5]:
        vals = matrix[:, f]
        top_idx = np.argsort(vals)[-3:][::-1]
        top_str = ", ".join(f"{sae_authors[i]}" for i in top_idx)
        print(f"  f{f:4d}  r={r:+.3f}  p={p:.4f}  top: {top_str}")

    print("\nTop 5 NEGATIVE:")
    for f, r, p in corrs[-5:]:
        vals = matrix[:, f]
        top_idx = np.argsort(vals)[-3:][::-1]
        top_str = ", ".join(f"{sae_authors[i]}" for i in top_idx)
        print(f"  f{f:4d}  r={r:+.3f}  p={p:.4f}  top: {top_str}")

    # Save results
    out = {
        "description": "H14 grounding: text-level correlations with H14 knockout recovery",
        "n_authors": len(results),
        "correlations": {},
        "authors": [],
    }
    for metric in ["i_pct", "conv_pct", "avg_sent", "q_per_k", "quotes_per_k"]:
        vals = np.array([r[metric] for r in results])
        r, p = stats.pearsonr(h14s, vals)
        out["correlations"][metric] = {"r": round(r, 4), "p": round(p, 6)}
    for r in results:
        out["authors"].append(r)

    out_path = SAE_DIR / "h14_grounding.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()