#!/usr/bin/env python3
"""Interpret SAE features: show top-activating tokens and author attribution.

For each feature, shows:
  - Top tokens/contexts that maximally activate it
  - Which authors trigger it most
  - A quick "what does this feature mean?" summary

Usage:
    uv run python scripts/analyze_sae.py
    uv run python scripts/analyze_sae.py --top-features 50
    uv run python scripts/analyze_sae.py --output outputs/sae/feature_report.txt
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch

from sixteen_voices import load_base_model, load_tokenizer, TextChunkDataset
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.constants import HIDDEN_DIM


def collect_per_author(model, tokenizer, sae, author_texts, hook_point,
                       seq_len=128, batch_size=16):
    """Run each author's text through model+SAE, return per-author activations."""

    block = model.transformer.h[0]
    if hook_point == "residual":
        target = model.transformer.ln_f
    elif hook_point == "mlp":
        target = block.mlp
    elif hook_point == "attn":
        target = block.attn
    else:
        raise ValueError(f"Unknown hook_point: {hook_point}")

    results = {}  # author -> (mean_feature_activation, n_tokens)

    for author, text in author_texts.items():
        activations = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output.detach())

        handle = target.register_forward_hook(hook_fn)

        dataset = TextChunkDataset(text, tokenizer, max_length=seq_len)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        model.eval()
        all_ids = []
        with torch.no_grad():
            for batch in loader:
                model(input_ids=batch["input_ids"])
                all_ids.append(batch["input_ids"])

        handle.remove()

        if not activations:
            continue

        acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
        ids = torch.cat([i.reshape(-1) for i in all_ids], dim=0)[:acts.shape[0]]

        with torch.no_grad():
            _, hidden = sae(acts)

        results[author] = {
            "hidden": hidden,       # (n_tokens, n_features)
            "ids": ids,             # (n_tokens,)
            "n_tokens": hidden.shape[0],
        }

    return results


def analyze(author_data, tokenizer, n_features, top_features=30,
            top_tokens_per_feature=15, context_window=3):
    """Analyze features across all authors."""

    # Stack all activations
    all_hidden = []
    all_ids = []
    all_authors = []  # which author each token belongs to
    for author, data in author_data.items():
        all_hidden.append(data["hidden"])
        all_ids.append(data["ids"])
        all_authors.extend([author] * data["n_tokens"])

    all_hidden = torch.cat(all_hidden, dim=0)  # (total_tokens, n_features)
    all_ids = torch.cat(all_ids, dim=0)

    # Feature stats
    mean_act = all_hidden.mean(dim=0)
    max_act = all_hidden.max(dim=0).values
    sparsity = (all_hidden > 0.01).float().mean(dim=0)
    dead_mask = max_act < 0.01

    n_dead = dead_mask.sum().item()
    n_alive = n_features - n_dead

    print(f"=== SAE Feature Analysis ===")
    print(f"Features: {n_features} ({n_alive} alive, {n_dead} dead)")
    print(f"Total tokens: {all_hidden.shape[0]:,}")
    print(f"Authors: {len(author_data)}")
    print()

    # Per-author mean activation per feature
    author_list = list(author_data.keys())
    author_means = {}
    for author, data in author_data.items():
        author_means[author] = data["hidden"].mean(dim=0)  # (n_features,)

    # Rank features by mean activation
    ranked = mean_act.argsort(descending=True)

    lines = []
    shown = 0
    for feat_idx in ranked:
        f = feat_idx.item()
        if dead_mask[f]:
            continue

        header = (f"── Feature {f:4d} │ mean={mean_act[f]:.4f}  "
                  f"max={max_act[f]:.4f}  "
                  f"fires on {100*sparsity[f].item():.1f}% of tokens ──")
        print(header)
        lines.append(header)

        # Top activating tokens with context
        top_pos = all_hidden[:, f].argsort(descending=True)[:top_tokens_per_feature]
        print("  Top tokens:")
        for pos in top_pos:
            p = pos.item()
            act_val = all_hidden[p, f].item()
            author = all_authors[p]

            # Get context window
            start = max(0, p - context_window)
            end = min(len(all_ids), p + context_window + 1)
            context_ids = all_ids[start:end].tolist()
            context_tokens = tokenizer.decode(context_ids)
            # Highlight the target token
            target_token = tokenizer.decode([all_ids[p].item()])

            line = f"    [{author:>15s}] ({act_val:.2f}) ...{context_tokens}...  ←«{target_token.strip()}»"
            print(line)
            lines.append(line)

        # Top authors for this feature
        author_scores = [(a, author_means[a][f].item()) for a in author_list]
        author_scores.sort(key=lambda x: x[1], reverse=True)
        top_authors = author_scores[:5]
        bot_authors = author_scores[-3:]

        author_line = "  Top authors: " + ", ".join(
            f"{a}({v:.3f})" for a, v in top_authors
        )
        print(author_line)
        lines.append(author_line)

        bot_line = "  Bottom authors: " + ", ".join(
            f"{a}({v:.3f})" for a, v in bot_authors
        )
        print(bot_line)
        lines.append(bot_line)

        print()
        lines.append("")

        shown += 1
        if shown >= top_features:
            break

    # Summary: feature-author correlation matrix overview
    print(f"\n=== Author specialization ===")
    print("Features most specific to each author (highest ratio vs global mean):\n")
    lines.append("\n=== Author specialization ===")

    global_mean = mean_act  # (n_features,)
    for author in sorted(author_list):
        a_mean = author_means[author]
        # Ratio of author mean to global mean (avoid div by zero)
        ratio = a_mean / (global_mean + 1e-8)
        top3 = ratio.argsort(descending=True)[:3]
        feats = ", ".join(
            f"f{i.item()}({ratio[i].item():.1f}x)" for i in top3
        )
        line = f"  {author:>20s}: {feats}"
        print(line)
        lines.append(line)

    return lines


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument("--sae-dir", default="outputs/sae",
                        help="Directory with sae_weights.pt and sae_config.json")
    parser.add_argument("--top-features", type=int, default=30,
                        help="How many features to show in detail")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file")
    parser.add_argument("--authors", type=str, nargs="+", default=None,
                        help="Only analyze these authors (e.g. --authors carroll wilde poe)")
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Use clean eval texts from data/eval/ (default)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Use training texts from data/authors/ instead of eval")
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)

    print(f"Loading SAE: {config['n_features']} features, "
          f"hook={config['hook_point']}, sparsity={config['sparsity_coeff']}")

    # Load SAE
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    # Load model
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_base_model()

    # Load author texts
    text_dir = Path("data/eval") if not args.no_eval else Path("data/authors")
    print(f"Loading texts from {text_dir}/ ...")
    author_texts = {}
    for f in sorted(text_dir.glob("*.txt")):
        author = f.stem
        if args.authors and author not in args.authors:
            continue
        author_texts[author] = f.read_text()
    if not author_texts:
        print(f"ERROR: no authors found. Available: "
              f"{[f.stem for f in sorted(text_dir.glob('*.txt'))]}")
        return
    print(f"  {len(author_texts)} authors: {', '.join(sorted(author_texts))}")

    # Collect per-author activations
    print("Running model + SAE on each author...")
    author_data = collect_per_author(
        model, tokenizer, sae, author_texts, config["hook_point"]
    )

    # Save per-author mean activation matrix (authors x features)
    # This is the main structured output for downstream analysis
    author_list = sorted(author_data.keys())
    matrix = torch.stack([author_data[a]["hidden"].mean(dim=0) for a in author_list])
    matrix_path = sae_dir / "author_feature_matrix.pt"
    torch.save({"authors": author_list, "matrix": matrix}, matrix_path)
    print(f"Saved author x feature matrix ({len(author_list)} x {config['n_features']}) "
          f"to {matrix_path}")

    # Analyze
    lines = analyze(
        author_data, tokenizer, config["n_features"],
        top_features=args.top_features,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()