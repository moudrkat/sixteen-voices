#!/usr/bin/env python3
"""Train a sparse autoencoder on the residual stream of TinyStories-1Layer-21M.

Collects activations from the model, trains a simple SAE, and saves
the encoder/decoder weights + feature activation stats.

The SAE decomposes the residual stream (after attention+MLP, before
final layernorm) into sparse, hopefully-interpretable features.

Usage:
    uv run python scripts/train_sae.py
    uv run python scripts/train_sae.py --n-features 2048 --sparsity 5e-3
    uv run python scripts/train_sae.py --hook-point mlp  # train on MLP output instead

What it does:
    1. Runs the base model on text chunks, collects activations
    2. Trains: hidden = ReLU(W_enc @ x + b_enc)
              x_hat  = W_dec @ hidden + b_dec
              loss   = MSE(x, x_hat) + λ * L1(hidden)
    3. Saves weights, dead feature stats, top-activating tokens
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sixteen_voices import load_base_model, load_tokenizer, TextChunkDataset
from sixteen_voices.constants import HIDDEN_DIM


class SparseAutoencoder(nn.Module):
    """Vanilla SAE: linear encoder with ReLU, linear decoder, L1 penalty."""

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize decoder columns to unit norm (Bricken et al. 2023)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        x_hat = self.decoder(hidden)
        return x_hat, hidden

    @property
    def n_features(self):
        return self.encoder.out_features


def collect_activations(
    model, tokenizer, texts: list[str], hook_point: str,
    max_chunks: int = 2000, seq_len: int = 128, batch_size: int = 16,
):
    """Run model on text, collect activations from a hook point.

    hook_point:
        "residual" — after attention+MLP block, before final LN
        "mlp"      — MLP output only (before residual add)
        "attn"     — attention output only (before residual add)
    """
    # TextChunkDataset takes a single string — concatenate all texts
    combined = "\n\n".join(texts)
    dataset = TextChunkDataset(combined, tokenizer, max_length=seq_len)
    if len(dataset) > max_chunks:
        indices = torch.randperm(len(dataset))[:max_chunks].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations = []
    hook_handle = None

    def hook_fn(module, input, output):
        # output shape varies by module; we want the hidden states
        if isinstance(output, tuple):
            output = output[0]
        activations.append(output.detach())

    # Register hook at the right place
    block = model.transformer.h[0]
    if hook_point == "residual":
        # After the full block — hook on final layernorm input
        target = model.transformer.ln_f
    elif hook_point == "mlp":
        target = block.mlp
    elif hook_point == "attn":
        target = block.attn
    else:
        raise ValueError(f"Unknown hook_point: {hook_point}")

    hook_handle = target.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            model(input_ids=batch["input_ids"])
            if (i + 1) % 20 == 0:
                n_tokens = sum(a.shape[0] * a.shape[1] for a in activations)
                print(f"  collected {n_tokens:,} token activations...")

    hook_handle.remove()

    # Flatten: (batch, seq, hidden) -> (n_tokens, hidden)
    all_acts = torch.cat([a.reshape(-1, a.shape[-1]) for a in activations], dim=0)
    print(f"  total: {all_acts.shape[0]:,} tokens × {all_acts.shape[1]} dims")
    return all_acts


def train_sae(
    activations: torch.Tensor,
    n_features: int = 2048,
    sparsity_coeff: float = 1e-3,
    lr: float = 3e-4,
    batch_size: int = 256,
    n_epochs: int = 5,
):
    """Train SAE on collected activations."""
    input_dim = activations.shape[1]
    sae = SparseAutoencoder(input_dim, n_features)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\nTraining SAE: {input_dim} → {n_features} features, "
          f"λ={sparsity_coeff}, {n_epochs} epochs")
    print(f"  {activations.shape[0]:,} training tokens\n")

    for epoch in range(n_epochs):
        total_recon = 0.0
        total_l1 = 0.0
        n_batches = 0

        for (batch,) in loader:
            x_hat, hidden = sae(batch)

            recon_loss = nn.functional.mse_loss(x_hat, batch)
            l1_loss = hidden.abs().mean()
            loss = recon_loss + sparsity_coeff * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder columns to unit norm (keeps features comparable)
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )

            total_recon += recon_loss.item()
            total_l1 += l1_loss.item()
            n_batches += 1

        avg_recon = total_recon / n_batches
        avg_l1 = total_l1 / n_batches
        print(f"  epoch {epoch+1}/{n_epochs}  "
              f"recon={avg_recon:.6f}  L1={avg_l1:.4f}  "
              f"loss={avg_recon + sparsity_coeff * avg_l1:.6f}")

    return sae


def analyze_features(sae, activations, tokenizer, input_ids_flat, top_k=10):
    """Basic feature analysis: dead features, top-activating tokens."""
    sae.eval()
    with torch.no_grad():
        _, hidden = sae(activations)  # (n_tokens, n_features)

    # Dead features: never activate above threshold
    max_activation = hidden.max(dim=0).values
    dead_mask = max_activation < 0.01
    n_dead = dead_mask.sum().item()
    n_alive = sae.n_features - n_dead

    print(f"\n--- Feature analysis ---")
    print(f"  alive: {n_alive}/{sae.n_features}  "
          f"dead: {n_dead}/{sae.n_features} "
          f"({100*n_dead/sae.n_features:.0f}%)")

    # Mean activations per feature
    mean_act = hidden.mean(dim=0)
    # Sparsity: fraction of tokens each feature fires on
    sparsity = (hidden > 0.01).float().mean(dim=0)

    print(f"  mean sparsity: {sparsity[~dead_mask].mean():.4f} "
          f"(fraction of tokens each alive feature fires on)")

    # Top features by mean activation
    top_features = mean_act.argsort(descending=True)[:20]
    print(f"\n  top 20 features by mean activation:")
    for feat_idx in top_features:
        f = feat_idx.item()
        print(f"    feature {f:4d}: mean={mean_act[f]:.4f}  "
              f"sparsity={sparsity[f]:.4f}  max={max_activation[f]:.4f}")

    # For each of top 10 alive features, show top-activating tokens
    if input_ids_flat is not None:
        print(f"\n  top-activating tokens for 10 strongest features:")
        alive_by_mean = mean_act.argsort(descending=True)
        shown = 0
        for feat_idx in alive_by_mean:
            f = feat_idx.item()
            if dead_mask[f]:
                continue
            top_positions = hidden[:, f].argsort(descending=True)[:top_k]
            token_ids = input_ids_flat[top_positions].tolist()
            tokens = [tokenizer.decode([t]).strip() for t in token_ids]
            acts = hidden[top_positions, f].tolist()
            tok_str = ", ".join(f"'{t}'({a:.2f})" for t, a in zip(tokens, acts))
            print(f"    feature {f:4d}: {tok_str}")
            shown += 1
            if shown >= 10:
                break

    stats = {
        "n_features": sae.n_features,
        "n_alive": n_alive,
        "n_dead": n_dead,
        "mean_sparsity": sparsity[~dead_mask].mean().item(),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train SAE on TinyStories residual stream")
    parser.add_argument("--hook-point", default="residual",
                        choices=["residual", "mlp", "attn"],
                        help="Where to collect activations")
    parser.add_argument("--n-features", type=int, default=2048,
                        help="Number of SAE features (default: 2048)")
    parser.add_argument("--sparsity", type=float, default=1e-3,
                        help="L1 sparsity coefficient (default: 1e-3)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=2000,
                        help="Max text chunks to collect activations from")
    parser.add_argument("--output", type=str, default="outputs/sae",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer()
    model = load_base_model()

    print("Loading text data...")
    text_dir = Path("data/authors")
    texts = []
    for f in sorted(text_dir.glob("*.txt")):
        texts.append(f.read_text())
    print(f"  {len(texts)} text files")

    # Collect activations
    print(f"\nCollecting activations from '{args.hook_point}'...")
    activations = collect_activations(
        model, tokenizer, texts, args.hook_point,
        max_chunks=args.max_chunks,
    )

    # Also collect input_ids for token analysis
    print("Collecting input_ids for token analysis...")
    combined = "\n\n".join(texts)
    dataset = TextChunkDataset(combined, tokenizer, max_length=128)
    if len(dataset) > args.max_chunks:
        indices = torch.randperm(len(dataset))[:args.max_chunks].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    all_ids = torch.cat([dataset[i]["input_ids"].unsqueeze(0) for i in range(len(dataset))], dim=0)
    input_ids_flat = all_ids.reshape(-1)
    # Truncate to match activations (in case of slight mismatch)
    input_ids_flat = input_ids_flat[:activations.shape[0]]

    # Train
    sae = train_sae(
        activations,
        n_features=args.n_features,
        sparsity_coeff=args.sparsity,
        lr=args.lr,
        n_epochs=args.epochs,
    )

    # Analyze
    stats = analyze_features(sae, activations, tokenizer, input_ids_flat)

    # Save
    torch.save(sae.state_dict(), output_dir / "sae_weights.pt")
    config = {
        "hook_point": args.hook_point,
        "input_dim": HIDDEN_DIM,
        "n_features": args.n_features,
        "sparsity_coeff": args.sparsity,
        "lr": args.lr,
        "epochs": args.epochs,
        "n_tokens": activations.shape[0],
        **stats,
    }
    with open(output_dir / "sae_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  sae_weights.pt  — model weights")
    print(f"  sae_config.json — config + stats")


if __name__ == "__main__":
    main()