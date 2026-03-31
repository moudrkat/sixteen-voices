#!/usr/bin/env python3
"""Show what semantic features detect — top-activating tokens with context.

Usage:
    uv run python scripts/analyze_semantic_tokens.py
    uv run python scripts/analyze_semantic_tokens.py --sae-dir outputs/sae_topk16_2048
    uv run python scripts/analyze_semantic_tokens.py --authors dark cozy carroll harris
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sixteen_voices import load_base_model, load_tokenizer, TextChunkDataset
from sixteen_voices.sae import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(
        description="Analyze semantic features — what tokens they fire on")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--authors", nargs="*",
                        default=["dark", "cozy", "carroll", "harris", "poe"])
    parser.add_argument("--top-k", type=int, default=6,
                        help="Top tokens to show per feature")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "semantic_tokens.json")

    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)

    afm = torch.load(sae_dir / "author_feature_matrix.pt", weights_only=False)
    all_authors = afm["authors"]
    matrix = afm["matrix"]
    global_mean = matrix.mean(dim=0)
    global_std = matrix.std(dim=0) + 1e-8

    tokenizer = load_tokenizer()
    model = load_base_model()
    text_dir = Path("data/authors")

    STRUCTURAL = {665, 883, 993, 60, 1777, 689, 329, 1385, 344,
                  1779, 627, 1604, 1524, 793}

    results = []

    for author in args.authors:
        text_path = text_dir / f"{author}.txt"
        if not text_path.exists():
            print(f"  {author}: no text file, skipping")
            continue

        # Find top semantic features for this author
        idx = all_authors.index(author)
        z = (matrix[idx] - global_mean) / global_std
        top_features = []
        for fi in z.argsort(descending=True):
            fi = fi.item()
            if fi in STRUCTURAL:
                continue
            if matrix[idx, fi].item() < 0.1:
                continue
            top_features.append(fi)
            if len(top_features) >= 3:
                break

        if not top_features:
            print(f"  {author}: no semantic features found")
            continue

        # Run author text through model + SAE
        text = text_path.read_text()
        dataset = TextChunkDataset(text, tokenizer, max_length=128)
        torch.manual_seed(42)
        if len(dataset) > 200:
            indices = torch.randperm(len(dataset))[:200].tolist()
            dataset = Subset(dataset, indices)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        activations, all_ids = [], []
        hook = model.transformer.ln_f.register_forward_hook(
            lambda mod, inp, out: activations.append(
                inp[0].detach() if isinstance(inp, tuple)
                else out.detach()))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                model(input_ids=batch["input_ids"])
                all_ids.append(batch["input_ids"])
        hook.remove()

        acts = torch.cat(
            [a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
        ids = torch.cat([i.reshape(-1) for i in all_ids], dim=0)
        with torch.no_grad():
            _, hidden = sae(acts)

        print(f"\n=== {author.upper()} ===")
        author_result = {"author": author, "features": []}

        for fi in top_features:
            top_pos = hidden[:, fi].argsort(descending=True)[:args.top_k]
            tokens_ctx = []
            for pos in top_pos:
                p = pos.item()
                act = hidden[p, fi].item()
                start = max(0, p - 5)
                end = min(len(ids), p + 6)
                context = tokenizer.decode(
                    ids[start:end].tolist()).replace("\n", " ")
                tokens_ctx.append({"activation": act, "context": context})

            z_val = z[fi].item()
            print(f"  f{fi} (z={z_val:.1f}):")
            for tc in tokens_ctx:
                print(f"    {tc['activation']:5.1f}  ...{tc['context']}...")

            author_result["features"].append({
                "feature": fi,
                "z_score": z_val,
                "top_tokens": tokens_ctx,
            })

        results.append(author_result)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()