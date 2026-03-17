#!/usr/bin/env python3
"""Vocab-driven vs structure-driven style: an independent text metric.

For each author, measures:
  - Vocabulary surprise: KL divergence of author's unigram distribution
    from the corpus-average unigram distribution. High = unusual words.
  - Structure surprise: excess bigram surprise beyond what unigrams predict.
    Computed as: H(bigram) - H(unigram). High = unusual word ordering.
  - Vocab-structure ratio: vocab_surprise / (vocab_surprise + structure_surprise).
    Close to 1 = style is about word choice. Close to 0 = style is about structure.

Then correlates this ratio with the V-Q balance from LoRA weights.

Usage:
    uv run python scripts/h14_vocab_vs_structure.py

Outputs:
    outputs/h14_vocab_vs_structure.json
    figures/h14_vocab_vs_structure.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sixteen_voices import get_eval_authors, load_eval_text
VQ_PATH = Path("outputs/h14_vq_balance.json")
OUTPUT_JSON = Path("outputs/h14_vocab_vs_structure.json")
OUTPUT_FIG = Path("figures/h14_vocab_vs_structure.png")

LABEL_AUTHORS = [
    "browne", "poe", "homer", "melville", "milton",
    "carroll", "grimm",
    "shelley", "twain", "burnett", "barrie",
    # synthetics
    "unusual_vocab", "dialogue", "firstperson", "reporter",
    "minimalist", "poet", "dark",
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


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation stripping tokenizer."""
    words = text.lower().split()
    words = [w.strip('.,;:!?\'"()[]{}—–-…""''') for w in words]
    return [w for w in words if w]


def unigram_kl(author_counts: Counter, corpus_counts: Counter, vocab: set) -> float:
    """KL divergence D(author || corpus) over shared vocabulary.

    Uses add-1 smoothing to handle zero counts.
    """
    author_total = sum(author_counts[w] for w in vocab) + len(vocab)
    corpus_total = sum(corpus_counts[w] for w in vocab) + len(vocab)

    kl = 0.0
    for w in vocab:
        p = (author_counts[w] + 1) / author_total  # author
        q = (corpus_counts[w] + 1) / corpus_total   # corpus average
        if p > 0:
            kl += p * np.log(p / q)
    return float(kl)


def bigram_excess_entropy(words: list[str]) -> float:
    """Conditional entropy of bigrams: H(w2|w1) = H(bigram) - H(unigram).

    Measures how much knowing the previous word reduces uncertainty —
    higher means LESS predictable (more random) word ordering patterns.
    """
    if len(words) < 100:
        return 0.0

    unigram_counts = Counter(words)
    bigram_counts = Counter(zip(words[:-1], words[1:]))
    n_bigrams = len(words) - 1

    # H(bigram) — entropy of bigram distribution
    h_bigram = 0.0
    for count in bigram_counts.values():
        p = count / n_bigrams
        if p > 0:
            h_bigram -= p * np.log2(p)

    # H(unigram) — entropy of unigram distribution
    n_unigrams = sum(unigram_counts.values())
    h_unigram = 0.0
    for count in unigram_counts.values():
        p = count / n_unigrams
        if p > 0:
            h_unigram -= p * np.log2(p)

    # Conditional entropy H(w2|w1) = H(w1,w2) - H(w1)
    # Higher = word order is LESS predictable (more random)
    # Lower = word order is MORE predictable (more structured patterns)
    return float(h_bigram - h_unigram)


def structure_score(words: list[str], text: str) -> tuple[float, float, float]:
    """Composite structure distinctiveness score.

    Combines:
    - Bigram repetition rate: fraction of bigrams that appear 3+ times
      (high = repetitive structural patterns like "said the", "and the")
    - Sentence length variance (normalized)
    - Dialogue density
    """
    if len(words) < 100:
        return 0.0

    bigram_counts = Counter(zip(words[:-1], words[1:]))
    n_bigrams = len(words) - 1
    repeated_bigrams = sum(c for c in bigram_counts.values() if c >= 3) / n_bigrams

    # Sentence length variance
    sentences = []
    for s in text.replace('!', '.').replace('?', '.').split('.'):
        s = s.strip()
        ws = s.split()
        if len(ws) >= 3:
            sentences.append(len(ws))
    sent_var = float(np.std(sentences)) if len(sentences) > 5 else 0.0

    # Dialogue density
    dialogue = text.count('"') / max(len(text), 1) * 100

    return repeated_bigrams, sent_var, dialogue


def main():
    # Load V-Q balance data
    with open(VQ_PATH) as f:
        vq_data = json.load(f)
    vq_by_author = {d["author"]: d for d in vq_data["per_author"]}

    # Build corpus-wide unigram distribution from all authors
    author_texts = {}
    author_words = {}
    corpus_counts = Counter()

    for author in get_eval_authors():
        try:
            text = load_eval_text(author, length=0)
        except FileNotFoundError:
            continue
        words = tokenize(text)
        if len(words) < 200:
            continue
        author_texts[author] = text
        author_words[author] = words
        corpus_counts.update(words)

    # Shared vocabulary (words appearing in at least 3 authors)
    word_author_count = Counter()
    for author, words in author_words.items():
        for w in set(words):
            word_author_count[w] += 1
    vocab = {w for w, c in word_author_count.items() if c >= 3}
    print(f"Corpus: {len(author_words)} authors, {len(vocab)} shared vocab words")

    # Compute metrics per author
    results = []
    for author in sorted(author_words):
        words = author_words[author]
        text = author_texts[author]
        a_counts = Counter(words)

        # Vocabulary surprise: KL(author || corpus)
        vocab_surprise = unigram_kl(a_counts, corpus_counts, vocab)

        # Structure: bigram conditional entropy and other features
        cond_entropy = bigram_excess_entropy(words)
        rep_bigrams, sent_var, dialogue = structure_score(words, text)

        results.append({
            "author": author,
            "vocab_surprise": vocab_surprise,
            "cond_entropy": cond_entropy,
            "repeated_bigram_rate": rep_bigrams,
            "sent_len_std": sent_var,
            "dialogue_density": dialogue,
            "n_words": len(words),
        })

    # Compute vocab-structure ratio
    # Normalize vocab_surprise and a structure composite to [0,1] range
    vs = np.array([r["vocab_surprise"] for r in results])
    # Structure composite: use repeated bigram rate (structural repetition)
    # + dialogue density (structural pattern) — both z-scored
    rep = np.array([r["repeated_bigram_rate"] for r in results])
    dial = np.array([r["dialogue_density"] for r in results])
    sent = np.array([r["sent_len_std"] for r in results])

    # Z-score each
    def zscore(x):
        return (x - x.mean()) / x.std() if x.std() > 0 else x * 0

    vs_z = zscore(vs)
    rep_z = zscore(rep)
    dial_z = zscore(dial)
    sent_z = zscore(sent)

    # Vocab score: just vocab surprise (z-scored)
    vocab_scores = vs_z

    # Structure score: average of structure indicators
    struct_scores = (rep_z + dial_z + sent_z) / 3

    # Ratio: positive = more vocab-driven, negative = more structure-driven
    vocab_minus_struct = vocab_scores - struct_scores

    for i, r in enumerate(results):
        r["vocab_z"] = float(vocab_scores[i])
        r["struct_z"] = float(struct_scores[i])
        r["vocab_minus_struct"] = float(vocab_minus_struct[i])

    # Correlate with V-Q balance
    matched = []
    for r in results:
        if r["author"] in vq_by_author:
            r["vq_balance"] = vq_by_author[r["author"]]["vq_balance"]
            r["h14_recovery"] = vq_by_author[r["author"]]["h14_recovery"]
            matched.append(r)

    vms = np.array([r["vocab_minus_struct"] for r in matched])
    vq = np.array([r["vq_balance"] for r in matched])
    h14 = np.array([r["h14_recovery"] for r in matched])

    r_vms_vq = float(np.corrcoef(vms, vq)[0, 1])
    r_vms_h14 = float(np.corrcoef(vms, h14)[0, 1])
    r_vs_vq = float(np.corrcoef([r["vocab_surprise"] for r in matched], vq)[0, 1])

    print(f"\nN = {len(matched)} matched authors")
    print(f"Correlation(vocab−struct, V-Q balance) = {r_vms_vq:+.3f}")
    print(f"Correlation(vocab−struct, H14 recovery) = {r_vms_h14:+.3f}")
    print(f"Correlation(vocab_surprise, V-Q balance) = {r_vs_vq:+.3f}")

    # Individual metric correlations with V-Q balance
    print(f"\n{'Metric':25s}  {'r(V-Q)':>8s}  {'r(H14)':>8s}")
    print("-" * 45)
    for metric in ["vocab_surprise", "repeated_bigram_rate", "dialogue_density",
                    "sent_len_std", "cond_entropy"]:
        vals = np.array([r[metric] for r in matched])
        rv = float(np.corrcoef(vals, vq)[0, 1])
        rh = float(np.corrcoef(vals, h14)[0, 1])
        print(f"{metric:25s}  {rv:+.3f}     {rh:+.3f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: vocab-struct score vs V-Q balance
    colors = []
    for r in matched:
        if r["author"] in SYNTHETIC_AUTHORS:
            style_colors = {"vocabulary": "#991b1b", "mixed": "#888888", "structure": "#1e40af"}
            colors.append(style_colors.get(SYNTHETIC_AUTHORS[r["author"]], "#888"))
        else:
            colors.append("#555555")

    ax1.scatter(vms, vq, c=colors, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)

    # Regression line
    slope, intercept = np.polyfit(vms, vq, 1)
    x_line = np.linspace(vms.min(), vms.max(), 100)
    ax1.plot(x_line, slope * x_line + intercept, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)

    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Labels
    author_to_r = {r["author"]: r for r in matched}
    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r.get("h14_recovery", 0) > 0.2 else \
                "#1e40af" if r.get("h14_recovery", 0) < -0.2 else "#666666"
        ax1.annotate(
            author.replace("_", " ").capitalize(),
            xy=(r["vocab_minus_struct"], r["vq_balance"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax1.set_xlabel("Text metric: vocab − structure score (z-scored)", fontsize=9)
    ax1.set_ylabel("H14 V-Q balance (from LoRA weights)", fontsize=9)
    ax1.set_title(f"Text-level style type vs weight-level V-Q balance\n"
                  f"r = {r_vms_vq:+.2f}",
                  fontsize=11, fontweight="bold")

    # Panel 2: vocab-struct score vs H14 recovery directly
    colors2 = ["#991b1b" if r["h14_recovery"] > 0.2 else
               "#1e40af" if r["h14_recovery"] < -0.2 else
               "#888888" for r in matched]

    ax2.scatter(vms, h14, c=colors2, s=30, alpha=0.7, edgecolors="white", linewidth=0.3)

    slope2, intercept2 = np.polyfit(vms, h14, 1)
    x_line2 = np.linspace(vms.min(), vms.max(), 100)
    ax2.plot(x_line2, slope2 * x_line2 + intercept2, color="#666", linestyle="--",
             linewidth=1, alpha=0.7)

    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax2.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    for author in LABEL_AUTHORS:
        if author not in author_to_r:
            continue
        r = author_to_r[author]
        color = "#991b1b" if r["h14_recovery"] > 0.2 else \
                "#1e40af" if r["h14_recovery"] < -0.2 else "#666666"
        ax2.annotate(
            author.replace("_", " ").capitalize(),
            xy=(r["vocab_minus_struct"], r["h14_recovery"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7, color=color, fontweight="bold", alpha=0.8,
        )

    ax2.set_xlabel("Text metric: vocab − structure score (z-scored)", fontsize=9)
    ax2.set_ylabel("H14 knockout recovery", fontsize=9)
    ax2.set_title(f"Text-level style type vs H14 recovery\n"
                  f"r = {r_vms_h14:+.2f}",
                  fontsize=11, fontweight="bold")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "n_authors": len(matched),
        "correlations": {
            "vocab_minus_struct_vs_vq": r_vms_vq,
            "vocab_minus_struct_vs_h14": r_vms_h14,
            "vocab_surprise_vs_vq": r_vs_vq,
        },
        "per_author": sorted(matched, key=lambda r: r["vocab_minus_struct"], reverse=True),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()