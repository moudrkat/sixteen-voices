#!/usr/bin/env python3
"""Classify all alive SAE features by monosemanticity.

Reads the feature report (from analyze_sae.py) and categorizes each feature:
  - monosemantic_token: fires on one specific token/subword
  - monosemantic_structural: fires on one structural property (punctuation, syntax)
  - function_word: fires on common function words (the, a, was, etc.)
  - formatting_artifact: fires on whitespace/indentation/encoding noise
  - near_dead: <=0.1% firing rate or max activation < 3.0
  - polysemantic: mixed token patterns, no dominant single concept

Usage:
    uv run python scripts/classify_sae_features.py
    uv run python scripts/classify_sae_features.py --report outputs/sae_topk16_2048/feature_report_all.txt
    uv run python scripts/classify_sae_features.py --output outputs/sae_topk16_2048/feature_classification.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


FUNCTION_WORDS = {
    w.lower()
    for w in [
        "the", "a", "an", "and", "of", "to", "in", "is", "it", "for", "that",
        "was", "on", "are", "with", "as", "at", "be", "or", "from", "but",
        "not", "by", "this", "had", "his", "her", "he", "she", "they", "we",
        "you", "my", "all", "been", "have", "has", "which", "were", "their",
        "would", "there", "so", "him", "them", "its", "who", "no", "if",
        "do", "did", "can", "could", "will", "shall", "may", "might",
        "one", "up", "out", "then", "than", "now", "very", "just", "about",
        "into", "over", "after", "more", "also", "some", "what", "when",
        "how", "where", "your", "our",
    ]
}


def parse_feature_report(path: str) -> list[dict]:
    """Parse feature_report_all.txt into structured feature records."""
    with open(path) as f:
        text = f.read()

    blocks = re.split(r"(?=── Feature)", text)
    blocks = [b for b in blocks if b.strip().startswith("── Feature")]

    features = []
    for b in blocks:
        header = re.search(
            r"Feature\s+(\d+) │ mean=([\d.]+)\s+max=([\d.]+)\s+fires on ([\d.]+)%", b
        )
        if not header:
            continue

        fid = int(header.group(1))
        mean_act = float(header.group(2))
        max_act = float(header.group(3))
        fire_rate = float(header.group(4))

        # Extract token hits (the ←«token» markers)
        tokens = re.findall(r"←«(.+?)»", b)

        # Extract top/bottom authors line
        top_match = re.search(r"Top authors: (.+)", b)
        bot_match = re.search(r"Bottom authors: (.+)", b)
        top_authors = top_match.group(1).strip() if top_match else ""
        bot_authors = bot_match.group(1).strip() if bot_match else ""

        features.append(
            {
                "fid": fid,
                "mean": mean_act,
                "max": max_act,
                "fire_rate": fire_rate,
                "tokens": tokens,
                "top_authors": top_authors,
                "bot_authors": bot_authors,
            }
        )

    return features


def classify_feature(feat: dict) -> tuple[str, str]:
    """Return (category, reason) for a single feature."""
    tokens = feat["tokens"]
    tokens_clean = [t.strip() for t in tokens if t.strip()]

    # Near-dead
    if feat["fire_rate"] <= 0.1 or feat["max"] < 3.0:
        return "near_dead", f"fire_rate={feat['fire_rate']}%, max={feat['max']:.1f}"

    # Formatting artifacts: >50% of top tokens are empty/whitespace/digits
    empty_or_ws = sum(1 for t in tokens if not t.strip() or t.strip().isdigit())
    if len(tokens) > 0 and empty_or_ws / len(tokens) > 0.5:
        return "formatting_artifact", f"{empty_or_ws}/{len(tokens)} empty/whitespace tokens"

    if not tokens_clean:
        return "near_dead", "no non-empty tokens"

    # Token frequency analysis
    token_counts = Counter(t.lower() for t in tokens_clean)
    most_common_token, most_common_count = token_counts.most_common(1)[0]
    dominance = most_common_count / len(tokens_clean)

    # Single-token dominated (>50% same token)
    if dominance > 0.5:
        if most_common_token in FUNCTION_WORDS:
            return "function_word", f"'{most_common_token}' ({dominance:.0%} of hits)"
        else:
            return "monosemantic_token", f"'{most_common_token}' ({dominance:.0%} of hits)"

    # Punctuation-dominated (>40% punctuation tokens)
    punct_count = sum(1 for t in tokens_clean if t in ".!?,;:\"'")
    if punct_count / len(tokens_clean) > 0.4:
        return "monosemantic_structural", f"punctuation ({punct_count}/{len(tokens_clean)})"

    # Function-word mix (>60% function words, but no single one dominates)
    fw_count = sum(1 for t in tokens_clean if t.lower() in FUNCTION_WORDS)
    if fw_count / len(tokens_clean) > 0.6:
        return "function_word", f"mixed function words ({fw_count}/{len(tokens_clean)})"

    # Everything else
    sample = ", ".join(tokens_clean[:6])
    return "polysemantic", f"mixed: [{sample}]"


def classify_all(features: list[dict]) -> list[dict]:
    """Classify all features, return annotated list."""
    for feat in features:
        category, reason = classify_feature(feat)
        feat["category"] = category
        feat["reason"] = reason
    return features


def print_report(features: list[dict]):
    """Print human-readable classification report."""
    by_cat = {}
    for f in features:
        by_cat.setdefault(f["category"], []).append(f)

    total = len(features)
    print(f"SAE Feature Monosemanticity Report")
    print(f"{'=' * 60}")
    print(f"Total alive features analyzed: {total}")
    print()

    # Summary table
    print(f"{'Category':<30s} {'Count':>5s} {'%':>6s}")
    print(f"{'-' * 43}")
    for cat in [
        "monosemantic_token",
        "monosemantic_structural",
        "function_word",
        "formatting_artifact",
        "polysemantic",
        "near_dead",
    ]:
        feats = by_cat.get(cat, [])
        pct = 100 * len(feats) / total if total > 0 else 0
        print(f"  {cat:<28s} {len(feats):5d} {pct:5.1f}%")

    genuinely_mono = len(by_cat.get("monosemantic_token", [])) + len(
        by_cat.get("monosemantic_structural", [])
    )
    print(f"\n  Genuinely monosemantic:       {genuinely_mono:5d} {100*genuinely_mono/total:5.1f}%")

    # Detail per category
    for cat, label in [
        ("monosemantic_token", "MONOSEMANTIC: TOKEN DETECTORS"),
        ("monosemantic_structural", "MONOSEMANTIC: STRUCTURAL DETECTORS"),
        ("function_word", "FUNCTION WORD DETECTORS"),
        ("formatting_artifact", "FORMATTING ARTIFACTS"),
        ("polysemantic", "POLYSEMANTIC / UNINTERPRETABLE"),
        ("near_dead", "NEAR-DEAD"),
    ]:
        feats = by_cat.get(cat, [])
        if not feats:
            continue
        print(f"\n{'=' * 60}")
        print(f"{label} ({len(feats)} features)")
        print(f"{'=' * 60}")
        for f in sorted(feats, key=lambda x: -x["fire_rate"]):
            print(
                f"  f{f['fid']:4d} ({f['fire_rate']:5.1f}%): {f['reason']}"
            )
            if f["top_authors"]:
                print(f"         top: {f['top_authors'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Classify SAE features by monosemanticity")
    parser.add_argument(
        "--report",
        default="outputs/sae_topk16_2048/feature_report_all.txt",
        help="Path to feature_report_all.txt from analyze_sae.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save JSON classification (default: <sae-dir>/feature_classification.json)",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Save text report to file",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: {report_path} not found. Run analyze_sae.py first:")
        print(f"  uv run python scripts/analyze_sae.py --sae-dir outputs/sae_topk16_2048 "
              f"--top-features 400 --output {report_path}")
        return

    print(f"Parsing {report_path} ...")
    features = parse_feature_report(str(report_path))
    print(f"  {len(features)} features parsed")

    print("Classifying ...")
    features = classify_all(features)

    # Print report
    print()
    print_report(features)

    # Save JSON
    output_path = args.output
    if output_path is None:
        output_path = report_path.parent / "feature_classification.json"
    output_path = Path(output_path)

    json_out = {}
    for f in features:
        json_out[f["fid"]] = {
            "category": f["category"],
            "reason": f["reason"],
            "fire_rate": f["fire_rate"],
            "mean": f["mean"],
            "max": f["max"],
            "tokens_sample": f["tokens"][:10],
            "top_authors": f["top_authors"],
            "bot_authors": f["bot_authors"],
        }
    with open(output_path, "w") as fp:
        json.dump(json_out, fp, indent=2)
    print(f"\nSaved classification to {output_path}")

    # Save text report
    if args.report_output:
        import io
        import sys

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_report(features)
        sys.stdout = old_stdout
        Path(args.report_output).write_text(buf.getvalue())
        print(f"Saved text report to {args.report_output}")


if __name__ == "__main__":
    main()