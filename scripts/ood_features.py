#!/usr/bin/env python3
"""What predicts H14 importance? Text feature correlation analysis.

Tests vocabulary complexity, word length, sentence length, etc. as predictors
of H14 recovery score. Finding: word complexity (pct_long_words, pct_rare_words)
predicts H14 importance (r~0.47), not base PPL or sentence length.

Usage:
    python scripts/ood_features.py
    python scripts/ood_features.py --plot
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from sixteen_voices import extract_prose

KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
DATA_DIR = Path("data/authors")

# TinyStories-like common vocabulary
TINYSTORIES_COMMON = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "so", "very", "too", "also", "just", "then", "now",
    "he", "she", "it", "they", "we", "i", "you", "his", "her", "its",
    "their", "our", "my", "your", "him", "them", "us", "me",
    "said", "went", "got", "came", "saw", "looked", "wanted", "liked",
    "played", "ran", "jumped", "walked", "cried", "laughed", "smiled",
    "happy", "sad", "big", "little", "small", "new", "old", "good",
    "bad", "pretty", "nice", "beautiful", "best", "favorite",
    "mom", "mommy", "dad", "daddy", "boy", "girl", "baby", "friend",
    "dog", "cat", "bird", "fish", "bunny", "bear", "lion", "monkey",
    "house", "home", "room", "door", "window", "bed", "table", "chair",
    "tree", "flower", "garden", "park", "forest", "river", "lake", "sea",
    "day", "night", "morning", "time", "today", "tomorrow",
    "red", "blue", "green", "yellow", "pink", "purple", "orange", "white",
    "one", "two", "three", "four", "five", "all", "some", "many",
    "thing", "way", "place", "name", "story", "game", "toy", "ball",
    "eat", "drink", "sleep", "help", "give", "take", "make", "put",
    "find", "know", "think", "feel", "love", "like", "want", "need",
}


def compute_text_features(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return {}

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    word_counts = Counter(words)
    n_words = len(words)
    n_unique = len(word_counts)

    ttr = n_unique / n_words if n_words > 0 else 0
    avg_word_len = np.mean([len(w) for w in words])
    pct_long = sum(1 for w in words if len(w) > 8) / n_words * 100
    pct_rare = sum(1 for w in words if w not in TINYSTORIES_COMMON) / n_words * 100
    sent_lengths = [len(re.findall(r"[a-zA-Z]+", s)) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    hapax = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax / n_unique if n_unique > 0 else 0

    return {
        "n_words": n_words,
        "n_unique": n_unique,
        "ttr": round(ttr, 4),
        "avg_word_len": round(avg_word_len, 2),
        "pct_long_words": round(pct_long, 1),
        "pct_rare_words": round(pct_rare, 1),
        "avg_sent_len": round(avg_sent_len, 1),
        "hapax_ratio": round(hapax_ratio, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Text features vs H14 importance")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/ood_features.json")
    args = parser.parse_args()

    if not KNOCKOUT_PATH.exists():
        print(f"Run scripts/knockout.py first ({KNOCKOUT_PATH} not found)")
        return
    knockout = json.load(open(KNOCKOUT_PATH))

    results = {}
    for author in sorted(knockout.keys()):
        txt_path = DATA_DIR / f"{author}.txt"
        if not txt_path.exists():
            continue
        text = extract_prose(txt_path.read_text(), length=10000)
        features = compute_text_features(text)
        features["base_ppl"] = knockout[author]["base_ppl"]
        features["full_ppl"] = knockout[author]["full_ppl"]
        features["h14_recovery"] = knockout[author]["head_recovery"]["H14"]
        results[author] = features

    authors = sorted(results.keys())
    h14_vals = np.array([results[a]["h14_recovery"] for a in authors])

    feature_names = ["base_ppl", "ttr", "avg_word_len", "pct_long_words",
                     "pct_rare_words", "avg_sent_len", "hapax_ratio"]

    print(f"\nCorrelation with H14 recovery ({len(authors)} authors)")
    print(f"{'='*70}")
    print(f"  {'Feature':20s}  {'Pearson r':>10s}  {'p-value':>10s}  {'Spearman':>10s}  {'p-value':>10s}")

    best_feature, best_r = None, 0
    for feat in feature_names:
        vals = np.array([results[a][feat] for a in authors])
        r_p, p_p = stats.pearsonr(vals, h14_vals)
        r_s, p_s = stats.spearmanr(vals, h14_vals)
        sig = "*" if p_p < 0.05 else " "
        print(f"  {feat:20s}  {r_p:+10.3f}  {p_p:10.4f}{sig}  {r_s:+10.3f}  {p_s:10.4f}")
        if abs(r_p) > abs(best_r):
            best_r, best_feature = r_p, feat

    print(f"\nBest predictor: {best_feature} (r={best_r:+.3f})")

    # Top/bottom authors
    sorted_h14 = sorted(authors, key=lambda a: results[a]["h14_recovery"], reverse=True)
    print(f"\nTop 10 H14-positive:")
    print(f"  {'Author':15s}  {'H14':>6s}  {'%Long':>6s}  {'%Rare':>6s}  {'WdLen':>6s}")
    for a in sorted_h14[:10]:
        r = results[a]
        print(f"  {a:15s}  {r['h14_recovery']:+6.2f}  {r['pct_long_words']:6.1f}  "
              f"{r['pct_rare_words']:6.1f}  {r['avg_word_len']:6.2f}")

    print(f"\nTop 10 H14-negative:")
    for a in sorted_h14[-10:]:
        r = results[a]
        print(f"  {a:15s}  {r['h14_recovery']:+6.2f}  {r['pct_long_words']:6.1f}  "
              f"{r['pct_rare_words']:6.1f}  {r['avg_word_len']:6.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plot_feats = ["ttr", "avg_word_len", "pct_long_words",
                      "pct_rare_words", "avg_sent_len", "base_ppl"]
        for ax, feat in zip(axes.flat, plot_feats):
            vals = [results[a][feat] for a in authors]
            ax.scatter(vals, h14_vals, alpha=0.6, s=30)
            r, p = stats.pearsonr(vals, h14_vals)
            ax.set_xlabel(feat)
            ax.set_ylabel("H14 recovery")
            ax.set_title(f"{feat} (r={r:+.2f}, p={p:.3f})")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = Path("outputs/plots/ood_features.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
