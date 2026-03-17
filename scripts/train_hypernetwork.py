#!/usr/bin/env python3
"""Train a hypernetwork to predict LoRA adapter weights from text.

Architecture:
  1. Run base model on author text → extract hidden states (mean-pooled)
  2. Small MLP maps hidden states → top-k PCA coefficients
  3. Reconstruct adapter: coefficients @ PCA_components + mean

The PCA decomposition from adapter_pca.py compresses each adapter from
2M parameters to ~30 coefficients. The hypernetwork learns to predict
those 30 numbers from the base model's representation of the text.

Training uses leave-one-out cross-validation (82 authors, predict each
from the other 81).

Usage:
    uv run python scripts/train_hypernetwork.py [--n-components 30] [--epochs 500]

Outputs:
    outputs/hypernetwork_results.json
    outputs/hypernetwork_model.pt
    figures/hypernetwork_eval.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.text import clean_text, compute_perplexity

AUTHORS_DIR = Path("data/authors")
PCA_PATH = Path("outputs/adapter_pca_components.npz")
KNOCKOUT_PATH = Path("outputs/knockout_all_heads.json")
OUTPUT_JSON = Path("outputs/hypernetwork_results.json")
OUTPUT_MODEL = Path("outputs/hypernetwork_model.pt")
OUTPUT_FIG = Path("figures/hypernetwork_eval.png")

HIDDEN_DIM = 1024  # base model hidden size
MAX_TOKENS = 512


class HyperNetwork(nn.Module):
    """Predicts PCA coefficients from base model hidden states."""

    def __init__(self, input_dim: int, n_components: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_components),
        )

    def forward(self, x):
        return self.net(x)


def extract_hidden_states(model, tokenizer, text: str) -> torch.Tensor:
    """Run base model on text, return mean-pooled hidden states [1024]."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # Last hidden state: [1, seq_len, hidden_dim]
    hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
    return hidden.mean(dim=0)  # [hidden_dim]


def extract_multi_features(model, tokenizer, text: str) -> torch.Tensor:
    """Extract richer features: mean + std of hidden states + attention entropy."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
    mean_h = hidden.mean(dim=0)  # [1024]
    std_h = hidden.std(dim=0)    # [1024]

    # Per-head attention entropy (16 values)
    attn = outputs.attentions[0][0]  # [num_heads, seq_len, seq_len]
    eps = 1e-10
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean(dim=-1)  # [num_heads]

    return torch.cat([mean_h, std_h, entropy])  # [1024 + 1024 + 16 = 2064]


def reconstruct_adapter(coefficients: np.ndarray, pca_mean: np.ndarray,
                        pca_components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct adapter weight deltas from PCA coefficients.

    Returns (q_proj_delta, v_proj_delta) each [1024, 1024].
    """
    flat = coefficients @ pca_components[:len(coefficients)] + pca_mean
    d = 1024
    q_delta = flat[:d * d].reshape(d, d)
    v_delta = flat[d * d:].reshape(d, d)
    return q_delta, v_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=30,
                        help="Number of PCA components to predict")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--features", choices=["mean", "multi"], default="multi",
                        help="Feature extraction mode")
    args = parser.parse_args()

    # Load PCA
    print("Loading PCA components...")
    pca = np.load(PCA_PATH, allow_pickle=True)
    pca_mean = pca["mean"]
    pca_components = pca["components"]  # [n_authors, 2*1024*1024]
    pca_scores = pca["scores"]          # [n_authors, n_authors]
    pca_authors = list(pca["authors"])

    k = min(args.n_components, len(pca_authors))
    target_scores = pca_scores[:, :k]   # [n_authors, k]

    print(f"Predicting top {k} PCA components")
    cumvar = np.cumsum(pca["singular_values"] ** 2)
    cumvar /= cumvar[-1]
    print(f"These capture {cumvar[k-1]:.1%} of adapter weight variance")

    # Load base model and extract features
    print("Loading base model...")
    tokenizer = load_tokenizer()
    model = load_base_model()

    print("Extracting features from author texts...")
    features = []
    valid_indices = []

    for i, author in enumerate(pca_authors):
        txt_path = AUTHORS_DIR / f"{author}.txt"
        if not txt_path.exists():
            print(f"  Skipping {author} — no text file")
            continue

        raw = txt_path.read_text(errors='ignore')
        text = clean_text(raw)
        words = text.split()
        if len(words) < 200:
            print(f"  Skipping {author} — too short")
            continue

        # Take from middle to avoid frontmatter
        mid = len(words) // 3
        eval_text = " ".join(words[mid:mid + 400])

        if args.features == "multi":
            feat = extract_multi_features(model, tokenizer, eval_text)
        else:
            feat = extract_hidden_states(model, tokenizer, eval_text)

        features.append(feat)
        valid_indices.append(i)

    X = torch.stack(features)  # [n, feat_dim]
    Y = torch.tensor(target_scores[valid_indices], dtype=torch.float32)  # [n, k]
    valid_authors = [pca_authors[i] for i in valid_indices]
    n = len(valid_authors)
    feat_dim = X.shape[1]

    print(f"Features: {n} authors × {feat_dim} dimensions")
    print(f"Targets: {n} authors × {k} PCA coefficients")

    # Normalize features
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-6)
    X_norm = (X - X_mean) / X_std

    # Normalize targets
    Y_mean = Y.mean(dim=0)
    Y_std = Y.std(dim=0).clamp(min=1e-6)
    Y_norm = (Y - Y_mean) / Y_std

    # --- Train on all data first (for the final model) ---
    print(f"\nTraining on all {n} authors...")
    hnet = HyperNetwork(feat_dim, k)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    losses = []
    for epoch in range(args.epochs):
        hnet.train()
        pred = hnet(X_norm)
        loss = nn.functional.mse_loss(pred, Y_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}")

    # Training set accuracy
    hnet.eval()
    with torch.no_grad():
        pred_norm = hnet(X_norm)
        pred_scores = pred_norm * Y_std + Y_mean
        train_r2 = 1 - torch.sum((pred_scores - Y) ** 2) / torch.sum((Y - Y.mean(dim=0)) ** 2)
        print(f"\nTrain R² (all components): {train_r2.item():.3f}")

        # Per-component R²
        for c in range(min(5, k)):
            ss_res = torch.sum((pred_scores[:, c] - Y[:, c]) ** 2)
            ss_tot = torch.sum((Y[:, c] - Y[:, c].mean()) ** 2)
            r2c = 1 - ss_res / ss_tot
            print(f"  PC{c} R² = {r2c.item():.3f}")

    # --- Leave-one-out cross-validation ---
    print(f"\nLeave-one-out cross-validation...")
    loo_predictions = torch.zeros_like(Y)

    for i in range(n):
        # Train on all except i
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False

        X_train = X_norm[mask]
        Y_train = Y_norm[mask]
        X_test = X_norm[i:i+1]

        loo_net = HyperNetwork(feat_dim, k)
        opt = torch.optim.Adam(loo_net.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

        for epoch in range(args.epochs):
            loo_net.train()
            pred = loo_net(X_train)
            loss = nn.functional.mse_loss(pred, Y_train)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        loo_net.eval()
        with torch.no_grad():
            pred_norm = loo_net(X_test)
            loo_predictions[i] = pred_norm[0] * Y_std + Y_mean

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n}]")

    # LOO R²
    loo_r2 = 1 - torch.sum((loo_predictions - Y) ** 2) / torch.sum((Y - Y.mean(dim=0)) ** 2)
    print(f"\nLOO R² (all components): {loo_r2.item():.3f}")

    # Per-component LOO R²
    loo_r2_per_pc = []
    for c in range(min(10, k)):
        ss_res = torch.sum((loo_predictions[:, c] - Y[:, c]) ** 2)
        ss_tot = torch.sum((Y[:, c] - Y[:, c].mean()) ** 2)
        r2c = float((1 - ss_res / ss_tot).item())
        loo_r2_per_pc.append(r2c)
        print(f"  PC{c} LOO R² = {r2c:.3f}")

    # Reconstruct adapters from LOO predictions and measure weight MSE
    loo_pred_np = loo_predictions.numpy()
    actual_np = Y.numpy()

    # Also compute: does the predicted adapter actually produce good perplexity?
    # (This is expensive — skip for now, just measure weight-space error)
    weight_cosine_sims = []
    for i in range(n):
        pred_flat = loo_pred_np[i] @ pca_components[:k] + pca_mean
        actual_flat = actual_np[i] @ pca_components[:k] + pca_mean
        cos = np.dot(pred_flat, actual_flat) / (np.linalg.norm(pred_flat) * np.linalg.norm(actual_flat) + 1e-10)
        weight_cosine_sims.append(float(cos))
    mean_cosine = np.mean(weight_cosine_sims)
    print(f"\nMean weight cosine similarity (LOO): {mean_cosine:.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Panel 1: Training loss curve
    ax = axes[0, 0]
    ax.plot(losses, color="#555", linewidth=0.5)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("MSE loss (normalized)", fontsize=9)
    ax.set_title("Training loss", fontsize=11, fontweight="bold")
    ax.set_yscale("log")

    # Panel 2: LOO R² per component
    ax = axes[0, 1]
    colors_bar = ["#991b1b" if r > 0 else "#1e40af" for r in loo_r2_per_pc]
    ax.bar(range(len(loo_r2_per_pc)), loo_r2_per_pc, color=colors_bar, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Principal component", fontsize=9)
    ax.set_ylabel("LOO R²", fontsize=9)
    ax.set_title(f"Per-component LOO R²\noverall LOO R² = {loo_r2.item():.3f}",
                 fontsize=11, fontweight="bold")

    # Panel 3: Predicted vs actual PC1 (LOO)
    ax = axes[1, 0]
    with open(KNOCKOUT_PATH) as f:
        ko = json.load(f)
    h14 = np.array([ko[a]["head_recovery"]["H14"] for a in valid_authors])
    colors = ["#991b1b" if h > 0.2 else "#1e40af" if h < -0.2 else "#888888"
              for h in h14]

    ax.scatter(actual_np[:, 0], loo_pred_np[:, 0], c=colors, s=30, alpha=0.7,
               edgecolors="white", linewidth=0.3)
    lim = max(abs(actual_np[:, 0]).max(), abs(loo_pred_np[:, 0]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Actual PC0 score", fontsize=9)
    ax.set_ylabel("Predicted PC0 score (LOO)", fontsize=9)
    ax.set_title(f"PC0: predicted vs actual\nLOO R² = {loo_r2_per_pc[0]:.3f}",
                 fontsize=11, fontweight="bold")

    for j, author in enumerate(valid_authors):
        if author in ["browne", "poe", "homer", "milton", "burnett", "twain",
                       "unusual_vocab", "dialogue", "dunsany", "yeats"]:
            color = "#991b1b" if h14[j] > 0.2 else \
                    "#1e40af" if h14[j] < -0.2 else "#666666"
            ax.annotate(author.replace("_", " ").capitalize(),
                        xy=(actual_np[j, 0], loo_pred_np[j, 0]),
                        xytext=(6, 3), textcoords="offset points",
                        fontsize=7, color=color, fontweight="bold", alpha=0.8)

    # Panel 4: Weight cosine similarity distribution
    ax = axes[1, 1]
    ax.hist(weight_cosine_sims, bins=20, color="#555", alpha=0.7, edgecolor="white")
    ax.axvline(mean_cosine, color="#991b1b", linewidth=1.5, linestyle="--",
               label=f"mean = {mean_cosine:.3f}")
    ax.set_xlabel("Weight cosine similarity (predicted vs actual)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("LOO adapter reconstruction quality",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save model
    torch.save({
        "model_state_dict": hnet.state_dict(),
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "n_components": k,
        "feat_dim": feat_dim,
        "features_mode": args.features,
    }, OUTPUT_MODEL)
    print(f"Saved model → {OUTPUT_MODEL}")

    # Save results
    output = {
        "n_authors": n,
        "n_components": k,
        "variance_captured": float(cumvar[k-1]),
        "feat_dim": feat_dim,
        "train_r2": float(train_r2.item()),
        "loo_r2": float(loo_r2.item()),
        "loo_r2_per_pc": loo_r2_per_pc,
        "mean_weight_cosine": mean_cosine,
        "per_author": [
            {
                "author": valid_authors[i],
                "weight_cosine": weight_cosine_sims[i],
                "h14_recovery": float(h14[i]),
            }
            for i in range(n)
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()