#!/usr/bin/env python3
"""Deep analysis of all 16 attention heads via SAE features.

For each head:
1. Which SAE features correlate with it (BH-corrected)
2. What tokens those features fire on (token-level evidence)
3. Which text properties predict the head's effect
4. Clustering: which heads read similar features
5. One-line summary

Usage:
    uv run python scripts/analyze_heads_deep.py
    uv run python scripts/analyze_heads_deep.py --head 3    # deep dive on one head
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy import stats

from sixteen_voices import load_tokenizer
from sixteen_voices.model import load_base_model
from sixteen_voices.sae import SparseAutoencoder

SAE_DIR = Path("outputs/sae_topk16_2048")
AUTHORS_DIR = Path("data/authors")
NUM_HEADS = 16

CONV_VERBS = {
    "am", "was", "'m", "'ve", "'ll", "'d",
    "said", "asked", "replied", "think", "know",
    "want", "feel", "like", "need",
}


def load_all_data():
    """Load SAE, knockout, author-feature matrix, tokenizer, model."""
    d = torch.load(SAE_DIR / "author_feature_matrix.pt", weights_only=False)
    authors = d["authors"]
    matrix = d["matrix"].numpy()

    with open("outputs/knockout_all_heads.json") as f:
        ko = json.load(f)

    knockout = np.array([
        [ko[a]["head_recovery"][f"H{h}"] for h in range(NUM_HEADS)]
        for a in authors
    ])

    with open(SAE_DIR / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(SAE_DIR / "sae_weights.pt", config)

    tokenizer = load_tokenizer()
    model = load_base_model()

    return authors, matrix, knockout, ko, sae, tokenizer, model


def bh_correction(p_values, fdr=0.05):
    """Benjamini-Hochberg correction. Returns set of significant indices."""
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


def get_head_features(matrix, knockout):
    """For each head, find correlated features at multiple thresholds."""
    n_features = matrix.shape[1]
    head_features = {}

    for h in range(NUM_HEADS):
        h_scores = knockout[:, h]
        corrs = []
        p_vals = []
        for f in range(n_features):
            if matrix[:, f].std() < 1e-8:
                corrs.append(0.0)
                p_vals.append(1.0)
                continue
            r, p = stats.pearsonr(matrix[:, f], h_scores)
            corrs.append(r)
            p_vals.append(p)
        corrs = np.array(corrs)
        p_vals = np.array(p_vals)

        bh_sig = bh_correction(p_vals)
        bh_feats = [(f, corrs[f], p_vals[f]) for f in bh_sig]
        bh_feats.sort(key=lambda x: abs(x[1]), reverse=True)

        loose = [(f, corrs[f], p_vals[f]) for f in range(n_features)
                 if abs(corrs[f]) > 0.3 and p_vals[f] < 0.01]
        loose.sort(key=lambda x: abs(x[1]), reverse=True)

        head_features[h] = {
            "bh": bh_feats,
            "loose": loose,
            "corrs": corrs,
            "p_vals": p_vals,
        }
    return head_features


def get_token_activations(sae, model, tokenizer, texts, target_features):
    """Run SAE on texts and find which tokens activate each target feature."""
    feature_tokens = defaultdict(Counter)

    for text in texts:
        ids = tokenizer.encode(text, return_tensors="pt", truncation=True,
                               max_length=512)
        tokens = [tokenizer.decode([t]) for t in ids[0]]

        with torch.no_grad():
            out = model(ids, output_hidden_states=False)
            # Get residual stream at ln_f input
            hidden = model.transformer.ln_f(
                model.transformer.h[0](
                    model.transformer.wte(ids) + model.transformer.wpe(
                        torch.arange(ids.shape[1]).unsqueeze(0)
                    )
                )[0]
            )

        # Run SAE encoder
        with torch.no_grad():
            pre_act = sae.encoder(hidden[0])
            if sae.activation == "topk":
                topk = torch.topk(pre_act, k=sae.k, dim=-1)
                acts = torch.zeros_like(pre_act)
                acts.scatter_(-1, topk.indices, torch.relu(topk.values))
            else:
                acts = torch.relu(pre_act)

        # For each target feature, record which tokens activate it
        for f in target_features:
            for pos in range(len(tokens)):
                val = acts[pos, f].item()
                if val > 0:
                    tok = tokens[pos].strip()
                    if tok:
                        feature_tokens[f][tok] += 1

    return feature_tokens


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
        excl_per_k = text.count("!") / total * 1000

        props[author] = {
            "i_pct": i_pct, "conv_pct": conv_pct, "avg_sent": avg_sent,
            "avg_word_len": avg_word_len, "q_per_k": q_per_k,
            "quote_per_k": quote_per_k, "excl_per_k": excl_per_k,
        }
    return props


def head_similarity(head_features):
    """Compute pairwise Jaccard similarity between heads based on feature sets."""
    sets = {}
    for h in range(NUM_HEADS):
        sets[h] = {f for f, r, p in head_features[h]["loose"]}

    print("\n\n=== HEAD SIMILARITY (Jaccard on loose features) ===")
    print(f"{'':>4}", end="")
    for h in range(NUM_HEADS):
        print(f"  H{h:>2}", end="")
    print()

    for h1 in range(NUM_HEADS):
        print(f"H{h1:<3}", end="")
        for h2 in range(NUM_HEADS):
            if not sets[h1] or not sets[h2]:
                print(f"    -", end="")
            else:
                jaccard = len(sets[h1] & sets[h2]) / len(sets[h1] | sets[h2])
                if h1 == h2:
                    print(f"    .", end="")
                elif jaccard > 0.3:
                    print(f"  {jaccard:.2f}"[0:5], end="")
                elif jaccard > 0.1:
                    print(f"  {jaccard:.2f}"[0:5], end="")
                else:
                    print(f"    -", end="")

        print()


def analyze_head(h, authors, matrix, knockout, head_features, text_props,
                 feature_tokens=None):
    """Deep analysis of a single head."""
    h_scores = knockout[:, h]
    hf = head_features[h]

    print(f"\n{'='*70}")
    print(f"H{h} DEEP DIVE")
    print(f"{'='*70}")

    # Basic stats
    mean_rec = h_scores.mean()
    std_rec = h_scores.std()
    best_head = knockout.argmax(axis=1)
    dominant = [authors[i] for i in range(len(authors)) if best_head[i] == h]

    print(f"\nKnockout: mean={mean_rec:+.3f}, std={std_rec:.3f}")
    print(f"Dominant for {len(dominant)} authors: {', '.join(dominant[:10])}"
          + (f"... (+{len(dominant)-10})" if len(dominant) > 10 else ""))
    print(f"BH features: {len(hf['bh'])}, loose features: {len(hf['loose'])}")

    # Top/bottom authors
    sorted_auth = sorted(zip(authors, h_scores), key=lambda x: x[1], reverse=True)
    print(f"\nTop 8:    {', '.join(f'{a}({s:+.3f})' for a, s in sorted_auth[:8])}")
    print(f"Bottom 5: {', '.join(f'{a}({s:+.3f})' for a, s in sorted_auth[-5:])}")

    # Text property correlations
    shared = [a for a in authors if a in text_props]
    h_vals = np.array([knockout[list(authors).index(a), h] for a in shared])
    print(f"\nText-level correlations (n={len(shared)}):")
    for metric in ["i_pct", "conv_pct", "avg_sent", "avg_word_len", "q_per_k",
                    "quote_per_k", "excl_per_k"]:
        vals = np.array([text_props[a][metric] for a in shared])
        r, p = stats.pearsonr(h_vals, vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        marker = " <--" if sig else ""
        print(f"  {metric:<16} r={r:+.3f}  p={p:.4f} {sig}{marker}")

    # Feature analysis
    feats_to_show = hf["bh"][:15] if hf["bh"] else hf["loose"][:15]
    if feats_to_show:
        print(f"\nTop correlated features:")
        for f, r, p in feats_to_show:
            vals = matrix[:, f]
            top_idx = np.argsort(vals)[-4:][::-1]
            bot_idx = np.argsort(vals)[:2]
            top_str = ", ".join(f"{authors[i]}" for i in top_idx)
            bot_str = ", ".join(f"{authors[i]}" for i in bot_idx)

            bh_tag = "BH" if any(bf[0] == f for bf in hf["bh"]) else "  "
            print(f"  [{bh_tag}] f{f:4d} r={r:+.3f}  HIGH: {top_str}  LOW: {bot_str}")

            # Token-level if available
            if feature_tokens and f in feature_tokens:
                top_tokens = feature_tokens[f].most_common(8)
                tok_str = ", ".join(f'"{t}"({c})' for t, c in top_tokens)
                print(f"         tokens: {tok_str}")

    # What's unique to this head?
    this_feats = {f for f, r, p in hf["loose"]}
    other_feats = set()
    for oh in range(NUM_HEADS):
        if oh == h:
            continue
        other_feats |= {f for f, r, p in head_features[oh]["loose"]}
    exclusive = this_feats - other_feats

    if exclusive:
        print(f"\nFeatures UNIQUE to H{h} ({len(exclusive)}):")
        for f in sorted(exclusive)[:5]:
            vals = matrix[:, f]
            top_idx = np.argsort(vals)[-3:][::-1]
            top_str = ", ".join(f"{authors[i]}" for i in top_idx)
            r = hf["corrs"][f]
            print(f"  f{f:4d} r={r:+.3f}: {top_str}")

    # Overlap with other heads
    print(f"\nOverlap with other heads:")
    for oh in range(NUM_HEADS):
        if oh == h:
            continue
        oh_feats = {f for f, r, p in head_features[oh]["loose"]}
        overlap = this_feats & oh_feats
        if overlap:
            print(f"  H{oh:2d}: {len(overlap):3d} shared features")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--skip-tokens", action="store_true",
                        help="Skip token-level analysis (faster)")
    args = parser.parse_args()

    print("Loading data...")
    authors, matrix, knockout, ko, sae, tokenizer, model = load_all_data()
    text_props = measure_text_properties(authors)
    print("Computing feature-head correlations...")
    head_features = get_head_features(matrix, knockout)

    # Token-level analysis: run SAE on sample texts
    feature_tokens = None
    if not args.skip_tokens:
        # Collect all features we care about (top features from all heads)
        all_target_features = set()
        for h in range(NUM_HEADS):
            feats = head_features[h]["bh"][:10] or head_features[h]["loose"][:10]
            for f, r, p in feats:
                all_target_features.add(f)

        # Load sample texts
        print(f"Running SAE on sample texts for {len(all_target_features)} features...")
        sample_texts = []
        sample_authors = ["poe", "homer", "carroll", "grimm", "shelley",
                          "wilde", "minimalist", "dialogue", "firstperson",
                          "unusual_vocab", "blake", "tennyson", "stoker",
                          "cozy", "questioner", "reporter"]
        for a in sample_authors:
            path = AUTHORS_DIR / f"{a}.txt"
            if path.exists():
                with open(path) as f:
                    text = f.read()
                # Take a middle chunk (skip headers)
                words = text.split()
                start = min(500, len(words) // 4)
                chunk = " ".join(words[start:start + 300])
                sample_texts.append(chunk)

        feature_tokens = get_token_activations(sae, model, tokenizer,
                                               sample_texts, all_target_features)
        print(f"Got token activations for {len(feature_tokens)} features")

    if args.head is not None:
        analyze_head(args.head, authors, matrix, knockout, head_features,
                     text_props, feature_tokens)
    else:
        for h in range(NUM_HEADS):
            analyze_head(h, authors, matrix, knockout, head_features,
                         text_props, feature_tokens)

        head_similarity(head_features)

        # Save results
        results = {
            "description": "Deep head analysis: features, text grounding, token evidence",
            "heads": [],
        }
        for h in range(NUM_HEADS):
            hf = head_features[h]
            h_scores = knockout[:, h]
            best_head = knockout.argmax(axis=1)

            shared = [a for a in authors if a in text_props]
            h_vals = np.array([knockout[list(authors).index(a), h] for a in shared])
            text_corrs = {}
            for metric in ["i_pct", "conv_pct", "avg_sent", "avg_word_len",
                           "q_per_k", "quote_per_k", "excl_per_k"]:
                vals = np.array([text_props[a][metric] for a in shared])
                r, p = stats.pearsonr(h_vals, vals)
                text_corrs[metric] = {"r": round(float(r), 4), "p": round(float(p), 6)}

            sorted_auth = sorted(zip(authors, h_scores.tolist()),
                                 key=lambda x: x[1], reverse=True)

            top_feats = []
            for f, r, p in (hf["bh"][:10] or hf["loose"][:10]):
                entry = {"feature": int(f), "r": round(float(r), 4),
                         "p": round(float(p), 6)}
                if feature_tokens and f in feature_tokens:
                    entry["top_tokens"] = feature_tokens[f].most_common(10)
                vals = matrix[:, f]
                top_idx = np.argsort(vals)[-4:][::-1]
                entry["top_authors"] = [authors[i] for i in top_idx]
                top_feats.append(entry)

            results["heads"].append({
                "head": h,
                "mean_recovery": round(float(h_scores.mean()), 4),
                "std_recovery": round(float(h_scores.std()), 4),
                "n_dominant": int((best_head == h).sum()),
                "dominant_for": [authors[i] for i in range(len(authors))
                                 if best_head[i] == h],
                "n_bh_features": len(hf["bh"]),
                "n_loose_features": len(hf["loose"]),
                "top_authors": sorted_auth[:8],
                "bot_authors": sorted_auth[-5:],
                "text_correlations": text_corrs,
                "top_features": top_feats,
            })

        out_path = SAE_DIR / "heads_deep_analysis.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nSaved {out_path}")


if __name__ == "__main__":
    main()