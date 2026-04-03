#!/usr/bin/env python3
"""Profile all 16 attention heads: SAE features, text-level grounding, author patterns.

For each head, reports:
- Knockout stats (mean recovery, std, how many authors it's dominant for)
- SAE features correlated with it (BH-corrected and at |r|>0.3)
- Which authors score highest/lowest
- Text-level property correlations (I-usage, conversational verbs, sentence length, word length)
- Attempted one-line interpretation

Usage:
    uv run python scripts/analyze_head_profiles.py
    uv run python scripts/analyze_head_profiles.py --head 14   # single head deep dive
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy import stats

SAE_DIR = Path("outputs/sae_topk16_2048")
AUTHORS_DIR = Path("data/authors")
NUM_HEADS = 16

CONV_VERBS = {
    "am", "was", "'m", "'ve", "'ll", "'d",
    "said", "asked", "replied", "think", "know",
    "want", "feel", "like", "need",
}


def load_data():
    """Load SAE matrix and knockout scores."""
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko = json.load(f)

    # Knockout matrix: authors × heads
    knockout = np.array([
        [ko[a]["head_recovery"][f"H{h}"] for h in range(NUM_HEADS)]
        for a in authors
    ])

    return authors, matrix, knockout, ko


def measure_text_properties(authors):
    """Measure text-level properties for all authors."""
    props = {}
    for author in authors:
        path = AUTHORS_DIR / f"{author}.txt"
        if not path.exists():
            continue
        with open(path) as f:
            text = f.read()
        words = text.split()
        total = len(words)
        if total < 100:
            continue

        i_pct = words.count("I") / total * 100
        conv_pct = sum(1 for w in words if w.lower() in CONV_VERBS) / total * 100
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip().split()) > 2]
        avg_sent = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        avg_word_len = np.mean([len(w) for w in words])
        q_per_k = text.count("?") / total * 1000
        quote_per_k = text.count('"') / total * 1000

        props[author] = {
            "i_pct": i_pct,
            "conv_pct": conv_pct,
            "avg_sent": avg_sent,
            "avg_word_len": avg_word_len,
            "q_per_k": q_per_k,
            "quote_per_k": quote_per_k,
        }
    return props


def bh_correction(p_values, fdr=0.05):
    """Benjamini-Hochberg correction. Returns set of indices that are significant."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = fdr * (np.arange(1, n + 1)) / n

    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k
    if max_k < 0:
        return set()
    return set(sorted_idx[:max_k + 1].tolist())


def compute_global_bh(matrix, knockout, fdr=0.05):
    """Compute global BH correction across all head×feature pairs."""
    n_features = matrix.shape[1]
    all_tests = []
    for h in range(NUM_HEADS):
        for f in range(n_features):
            if matrix[:, f].std() < 1e-8:
                all_tests.append((h, f, 0.0, 1.0))
            else:
                r, p = stats.pearsonr(matrix[:, f], knockout[:, h])
                all_tests.append((h, f, r, p))
    p_array = np.array([t[3] for t in all_tests])
    sig_idx = bh_correction(p_array, fdr)
    sig_pairs = set()
    for i in sig_idx:
        h, f, r, p = all_tests[i]
        sig_pairs.add((h, f))
    return sig_pairs


def profile_head(h, authors, matrix, knockout, text_props, global_sig=None):
    """Generate a full profile for one head.

    global_sig: set of (head, feature) pairs that pass global BH correction.
    If None, falls back to per-head BH (less strict).
    """
    h_scores = knockout[:, h]
    n_features = matrix.shape[1]

    # Basic stats
    mean_rec = h_scores.mean()
    std_rec = h_scores.std()

    # Dominant for which authors
    best_head = knockout.argmax(axis=1)
    dominant_mask = best_head == h
    dominant_authors = [authors[i] for i in range(len(authors)) if dominant_mask[i]]

    # Feature correlations
    all_corrs = []
    all_p = []
    for f in range(n_features):
        if matrix[:, f].std() < 1e-8:
            all_corrs.append(0)
            all_p.append(1.0)
            continue
        r, p = stats.pearsonr(matrix[:, f], h_scores)
        all_corrs.append(r)
        all_p.append(p)
    all_corrs = np.array(all_corrs)
    all_p = np.array(all_p)

    # BH correction — use global if provided, else per-head fallback
    if global_sig is not None:
        bh_features = [(f, all_corrs[f], all_p[f]) for f in range(n_features)
                       if (h, f) in global_sig]
    else:
        bh_sig = bh_correction(all_p)
        bh_features = [(f, all_corrs[f], all_p[f]) for f in bh_sig]
    bh_features.sort(key=lambda x: abs(x[1]), reverse=True)

    # Looser threshold
    loose_features = [(f, all_corrs[f], all_p[f]) for f in range(n_features)
                      if abs(all_corrs[f]) > 0.3 and all_p[f] < 0.01]
    loose_features.sort(key=lambda x: abs(x[1]), reverse=True)

    # Top/bottom authors
    sorted_auth = sorted(zip(authors, h_scores), key=lambda x: x[1], reverse=True)
    top5 = sorted_auth[:5]
    bot5 = sorted_auth[-5:]

    # Text property correlations
    text_corrs = {}
    shared_authors = [a for a in authors if a in text_props]
    if len(shared_authors) >= 10:
        h_vals = np.array([knockout[list(authors).index(a), h] for a in shared_authors])
        for metric in ["i_pct", "conv_pct", "avg_sent", "avg_word_len", "q_per_k", "quote_per_k"]:
            vals = np.array([text_props[a][metric] for a in shared_authors])
            r, p = stats.pearsonr(h_vals, vals)
            text_corrs[metric] = {"r": r, "p": p}

    # For top features, show which authors fire highest
    feature_profiles = []
    for f, r, p in (bh_features or loose_features)[:10]:
        vals = matrix[:, f]
        top_idx = np.argsort(vals)[-5:][::-1]
        feature_profiles.append({
            "feature": f,
            "r": r,
            "p": p,
            "top_authors": [(authors[i], float(vals[i])) for i in top_idx],
        })

    return {
        "head": h,
        "mean_recovery": float(mean_rec),
        "std_recovery": float(std_rec),
        "dominant_for": dominant_authors,
        "n_dominant": len(dominant_authors),
        "n_bh_features": len(bh_features),
        "n_loose_features": len(loose_features),
        "top_authors": [(a, float(s)) for a, s in top5],
        "bot_authors": [(a, float(s)) for a, s in bot5],
        "text_correlations": {k: {"r": round(v["r"], 4), "p": round(v["p"], 6)}
                              for k, v in text_corrs.items()},
        "feature_profiles": feature_profiles,
    }


def print_profile(profile, verbose=False):
    """Pretty-print a head profile."""
    h = profile["head"]
    print(f"\n{'='*70}")
    print(f"H{h}  |  mean={profile['mean_recovery']:+.3f}  std={profile['std_recovery']:.3f}  "
          f"|  dominant for {profile['n_dominant']} authors  "
          f"|  {profile['n_bh_features']} BH features, {profile['n_loose_features']} loose")

    if profile["dominant_for"]:
        dom = profile["dominant_for"]
        if len(dom) <= 8:
            print(f"  Dominant for: {', '.join(dom)}")
        else:
            print(f"  Dominant for: {', '.join(dom[:5])}... (+{len(dom)-5} more)")

    print(f"  Top:    {', '.join(f'{a}({s:+.2f})' for a, s in profile['top_authors'])}")
    print(f"  Bottom: {', '.join(f'{a}({s:+.2f})' for a, s in profile['bot_authors'])}")

    # Text correlations
    sig_text = []
    for metric, vals in profile["text_correlations"].items():
        sig = ""
        if vals["p"] < 0.001: sig = "***"
        elif vals["p"] < 0.01: sig = "**"
        elif vals["p"] < 0.05: sig = "*"
        if sig:
            sig_text.append(f"{metric}(r={vals['r']:+.3f}{sig})")
    if sig_text:
        print(f"  Text grounding: {', '.join(sig_text)}")
    else:
        print(f"  Text grounding: none significant")

    # Top features
    if profile["feature_profiles"]:
        print(f"  Top features:")
        for fp in profile["feature_profiles"][:5]:
            top_str = ", ".join(f"{a}" for a, v in fp["top_authors"][:4])
            print(f"    f{fp['feature']:4d} (r={fp['r']:+.3f}): {top_str}")

    if verbose and profile["feature_profiles"]:
        print(f"\n  All {len(profile['feature_profiles'])} features:")
        for fp in profile["feature_profiles"]:
            top_str = ", ".join(f"{a}({v:.1f})" for a, v in fp["top_authors"])
            print(f"    f{fp['feature']:4d} r={fp['r']:+.3f} p={fp['p']:.6f} | {top_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=None, help="Profile a single head in detail")
    parser.add_argument("--sae-dir", type=str, default=None)
    args = parser.parse_args()

    global SAE_DIR
    if args.sae_dir:
        SAE_DIR = Path(args.sae_dir)

    authors, matrix, knockout, ko = load_data()
    text_props = measure_text_properties(authors)

    print("Computing global Benjamini-Hochberg correction...")
    global_sig = compute_global_bh(matrix, knockout)

    if args.head is not None:
        profile = profile_head(args.head, authors, matrix, knockout, text_props,
                               global_sig=global_sig)
        print_profile(profile, verbose=True)
    else:
        all_profiles = []
        for h in range(NUM_HEADS):
            profile = profile_head(h, authors, matrix, knockout, text_props,
                                   global_sig=global_sig)
            print_profile(profile)
            all_profiles.append(profile)

        # Save JSON
        out_path = SAE_DIR / "head_profiles.json"
        with open(out_path, "w") as f:
            json.dump(all_profiles, f, indent=2)
        print(f"\n\nSaved {out_path}")

        # Summary table
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"{'Head':<5} {'Mean':>6} {'Std':>5} {'Dom':>4} {'BH':>4} {'Loose':>6}  Text grounding")
        print("-" * 70)
        for p in all_profiles:
            sig = []
            for m, v in p["text_correlations"].items():
                if v["p"] < 0.05:
                    sig.append(f"{m}({v['r']:+.2f})")
            sig_str = ", ".join(sig) if sig else "—"
            print(f"H{p['head']:<4} {p['mean_recovery']:>+6.3f} {p['std_recovery']:>5.3f} "
                  f"{p['n_dominant']:>4} {p['n_bh_features']:>4} {p['n_loose_features']:>6}  {sig_str}")


if __name__ == "__main__":
    main()