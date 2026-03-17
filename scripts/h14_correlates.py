#!/usr/bin/env python3
"""What predicts H14 recovery? Correlate text-level metrics with H14 scores.

Uses cleaned text (same preprocessing as training) for all metrics.
Tests whether the H14 polarization can be explained by measurable
properties of each author's prose.

Usage:
    uv run python scripts/h14_correlates.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from sixteen_voices import load_eval_text

KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
EVAL_PATH = Path("outputs/eval_adapters.json")
OUTPUT_PATH = Path("outputs/h14_correlates.json")

# Common/simple words — roughly the TinyStories vocabulary core
SIMPLE_WORDS = set(
    "the a an and but or in on at to for of is was were are be been "
    "he she it they we i you my his her its our their this that these those "
    "said went got had has have do did does can could will would shall should "
    "not no yes up down out into with from by as if so then than very much "
    "little big small one two three all some any many more most other new old "
    "good bad happy sad nice day time girl boy man mom dad come go see look "
    "make take give tell think know want like just now here there when what "
    "how why who where which way thing because before after about over through "
    "again back still even also too".split()
)


def compute_text_metrics(text: str) -> dict:
    """Compute prose-level metrics from cleaned text."""
    words_raw = text.split()
    # Strip punctuation for word-level metrics
    words = [w.strip('.,;:!?\'"()[]{}—–-…""''') .lower() for w in words_raw]
    words = [w for w in words if w and len(w) > 0]

    if len(words) < 50:
        return None

    counts = Counter(words)
    unique = set(words)

    # Sentence splitting (approximate)
    sentences = []
    for s in text.replace('!', '.').replace('?', '.').split('.'):
        s = s.strip()
        if len(s.split()) >= 3:
            sentences.append(s)

    return {
        "n_words": len(words),
        "avg_word_len": float(np.mean([len(w) for w in words])),
        "pct_long_words": sum(1 for w in words if len(w) > 7) / len(words),
        "pct_short_words": sum(1 for w in words if len(w) <= 3) / len(words),
        "simple_word_frac": sum(1 for w in words if w in SIMPLE_WORDS) / len(words),
        "type_token_ratio": len(unique) / len(words),
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / len(unique),
        "avg_sent_len": float(np.mean([len(s.split()) for s in sentences])) if sentences else 0,
        "comma_density": text.count(',') / len(words),
        "semicolon_density": text.count(';') / len(words),
        "dialogue_frac": text.count('"') / len(text) * 100,
    }


def main():
    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)
    with open(EVAL_PATH) as f:
        ev = {item['author']: item for item in json.load(f)}

    results = []

    for author in ko:
        if author not in ev:
            continue

        try:
            text = load_eval_text(author, length=0)
        except FileNotFoundError:
            continue

        metrics = compute_text_metrics(text)
        if metrics is None:
            print(f"  Skipping {author} — too short after cleaning")
            continue

        metrics["author"] = author
        metrics["h14_recovery"] = ko[author]["head_recovery"]["H14"]
        metrics["base_ppl"] = ev[author]["base_ppl"]
        metrics["adapted_ppl"] = ev[author]["adapted_ppl"]
        metrics["ppl_ratio"] = ev[author]["ratio"]
        results.append(metrics)

    print(f"\nAnalyzed {len(results)} authors")

    # Compute correlations
    h14 = np.array([r["h14_recovery"] for r in results])
    metric_names = [k for k in results[0] if k not in ("author", "h14_recovery")]

    print(f"\n{'Metric':25s}  {'r':>7s}  {'p<0.05':>7s}")
    print("-" * 42)

    correlations = {}
    for m in metric_names:
        vals = np.array([r[m] for r in results])
        r = float(np.corrcoef(vals, h14)[0, 1])
        # Approximate significance: t = r * sqrt(n-2) / sqrt(1-r^2)
        n = len(results)
        if abs(r) < 1.0:
            t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
            # Two-tailed p-value approximation (rough)
            significant = abs(t) > 2.0  # ~p<0.05 for n>30
        else:
            significant = True
        correlations[m] = {"r": r, "significant": bool(significant)}
        print(f"{m:25s}  {r:+.3f}  {'*' if significant else '':>7s}")

    # Sort by absolute correlation
    print(f"\n=== Sorted by |r| ===")
    for m, v in sorted(correlations.items(), key=lambda x: -abs(x[1]["r"])):
        print(f"  {m:25s}  r={v['r']:+.3f}  {'*' if v['significant'] else ''}")

    # Show top and bottom authors with the best metric
    best_metric = max(correlations, key=lambda k: abs(correlations[k]["r"]))
    print(f"\n=== H14 vs {best_metric} (top correlator) ===")
    sorted_results = sorted(results, key=lambda x: x["h14_recovery"])
    print(f"\n{'Author':20s}  {best_metric:>15s}  {'H14':>6s}")
    print("-" * 45)
    for r in sorted_results[:8]:
        print(f"{r['author']:20s}  {r[best_metric]:15.3f}  {r['h14_recovery']:+.3f}")
    print("...")
    for r in sorted_results[-8:]:
        print(f"{r['author']:20s}  {r[best_metric]:15.3f}  {r['h14_recovery']:+.3f}")

    # Save
    output = {
        "n_authors": len(results),
        "correlations": correlations,
        "per_author": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()