#!/usr/bin/env python3
"""Retraining stability test: does H14 stay on top across random seeds?

Trains an author's LoRA adapter multiple times with different random
seeds, runs knockout on each, and checks whether the head ranking
is stable.

Usage:
    uv run python scripts/retrain_stability.py                    # default: poe, 5 seeds
    uv run python scripts/retrain_stability.py --author melville  # different author
    uv run python scripts/retrain_stability.py --seeds 10         # more seeds

Outputs:
    outputs/retrain_stability.json
    figures/retrain_stability.png
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sixteen_voices import TextChunkDataset, create_lora_model, load_base_model, load_eval_text, load_tokenizer
from sixteen_voices.adapter import load_adapter_deltas, knockout_all_except
from sixteen_voices.text import clean_text, compute_perplexity

DATA_DIR = Path("data/authors")  # training data stays raw
OUTPUT_JSON = Path("outputs/retrain_stability.json")
OUTPUT_FIG = Path("figures/retrain_stability.png")

EPOCHS = 8
BATCH_SIZE = 4
LR = 5e-4
MAX_LENGTH = 512
STRIDE = 256
VAL_SPLIT = 0.1
MAX_WORDS = 50_000
NUM_HEADS = 16
HEAD_DIM = 64


def train_with_seed(text, tokenizer, base_model, seed, tag="lora"):
    """Train a LoRA adapter with a specific random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TextChunkDataset(text, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))

    model = create_lora_model(base_model=base_model)

    # Training loop (inlined to control seed)
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    for epoch in range(EPOCHS):
        for batch in dataloader:
            loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Quick val check
        model.eval()
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loss = sum(
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.item()
            for b in val_dl
        ) / len(val_dl)
        print(f"  [{tag}] epoch {epoch+1}/{EPOCHS} — val_loss: {val_loss:.4f}")
        model.train()

    model.eval()
    return model, val_loss


def run_knockout(model, base_model, tokenizer, eval_text):
    """Run per-head knockout and return recovery dict."""
    base_ppl = compute_perplexity(base_model, tokenizer, eval_text)
    adapted_ppl = compute_perplexity(model, tokenizer, eval_text)
    full_delta = base_ppl - adapted_ppl

    if abs(full_delta) < 0.1:
        return None

    # Get adapter deltas
    # Save to temp dir and reload
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        deltas = load_adapter_deltas(tmp)

    import copy

    recovery = {}
    for h in range(NUM_HEADS):
        q_ko = knockout_all_except(deltas["q_proj"], h)
        v_ko = knockout_all_except(deltas["v_proj"], h)

        ko_base = copy.deepcopy(base_model)
        attn = ko_base.transformer.h[0].attn.attention
        with torch.no_grad():
            attn.q_proj.weight.add_(q_ko)
            attn.v_proj.weight.add_(v_ko)
        ko_base.eval()
        ko_ppl = compute_perplexity(ko_base, tokenizer, eval_text)
        recovery[f"H{h}"] = float((base_ppl - ko_ppl) / full_delta)
        del ko_base

    return recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default="poe")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    author = args.author
    n_seeds = args.seeds
    seeds = list(range(42, 42 + n_seeds))

    print(f"Retraining stability test: {author}, {n_seeds} seeds")

    # Load text
    txt_path = DATA_DIR / f"{author}.txt"
    raw = txt_path.read_text(errors='ignore')
    text = clean_text(raw)
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    print(f"Text: {len(text.split()):,} words")

    # Eval text from pre-extracted clean prose
    eval_text = load_eval_text(author, length=3500)
    eval_words = eval_text.split()
    mid = len(eval_words) // 3
    eval_text = " ".join(eval_words[mid:mid + 500])

    tokenizer = load_tokenizer()
    base_model = load_base_model()

    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        t0 = time.time()

        model, val_loss = train_with_seed(text, tokenizer, base_model, seed,
                                           tag=f"{author}_s{seed}")
        elapsed = time.time() - t0
        print(f"  Trained in {elapsed:.0f}s, val_loss={val_loss:.4f}")

        # Save adapter for reproducibility
        adapter_dir = Path(f"outputs/retrain_stability/{author}/seed_{seed}/adapter")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        print(f"  Saved adapter → {adapter_dir}")

        print(f"  Running knockout...")
        recovery = run_knockout(model, base_model, tokenizer, eval_text)

        if recovery is None:
            print(f"  Skipping — no adaptation detected")
            continue

        # Rank heads by recovery
        ranked = sorted(recovery.items(), key=lambda x: -x[1])
        top3 = [(h, f"{v:+.3f}") for h, v in ranked[:3]]
        h14_rank = [i for i, (h, _) in enumerate(ranked) if h == "H14"][0] + 1
        h14_val = recovery["H14"]

        print(f"  Top 3: {top3}")
        print(f"  H14: rank={h14_rank}, recovery={h14_val:+.3f}")

        all_results.append({
            "seed": seed,
            "val_loss": val_loss,
            "train_time": elapsed,
            "recovery": recovery,
            "h14_rank": h14_rank,
            "h14_recovery": h14_val,
            "top_head": ranked[0][0],
        })

        del model

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY: {author}, {len(all_results)} seeds")
    print(f"{'='*50}")

    h14_ranks = [r["h14_rank"] for r in all_results]
    h14_recoveries = [r["h14_recovery"] for r in all_results]
    top_heads = [r["top_head"] for r in all_results]

    print(f"H14 rank across seeds: {h14_ranks}")
    print(f"H14 recovery across seeds: [{', '.join(f'{v:+.3f}' for v in h14_recoveries)}]")
    print(f"Top head across seeds: {top_heads}")
    print(f"H14 rank: mean={np.mean(h14_ranks):.1f}, std={np.std(h14_ranks):.1f}")
    print(f"H14 recovery: mean={np.mean(h14_recoveries):+.3f}, std={np.std(h14_recoveries):.3f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Recovery per head across seeds (box plot style)
    head_data = {f"H{h}": [] for h in range(NUM_HEADS)}
    for r in all_results:
        for h in range(NUM_HEADS):
            head_data[f"H{h}"].append(r["recovery"][f"H{h}"])

    # Sort by mean recovery
    head_order = sorted(head_data.keys(),
                        key=lambda h: np.mean(head_data[h]), reverse=True)
    box_data = [head_data[h] for h in head_order]

    bp = ax1.boxplot(box_data, labels=head_order, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))
    for i, patch in enumerate(bp['boxes']):
        if head_order[i] == "H14":
            patch.set_facecolor("#991b1b")
            patch.set_alpha(0.7)
        elif head_order[i] in ("H11", "H3"):
            patch.set_facecolor("#666")
            patch.set_alpha(0.5)
        else:
            patch.set_facecolor("#cccccc")
            patch.set_alpha(0.5)

    ax1.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Knockout recovery", fontsize=9)
    ax1.set_title(f"{author.capitalize()}: head recovery across {n_seeds} seeds",
                  fontsize=11, fontweight="bold")
    ax1.tick_params(axis='x', rotation=45)

    # Panel 2: H14 recovery per seed
    ax2.bar(range(len(all_results)), h14_recoveries,
            color=["#991b1b" if v > 0.2 else "#1e40af" if v < -0.2 else "#888"
                   for v in h14_recoveries],
            alpha=0.7, edgecolor="white")
    ax2.set_xticks(range(len(all_results)))
    ax2.set_xticklabels([f"seed {r['seed']}" for r in all_results], fontsize=8)
    ax2.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("H14 knockout recovery", fontsize=9)
    ax2.set_title(f"H14 stability: mean={np.mean(h14_recoveries):+.2f}, "
                  f"std={np.std(h14_recoveries):.2f}",
                  fontsize=11, fontweight="bold")

    plt.tight_layout()
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved {OUTPUT_FIG}")

    # Save JSON
    output = {
        "author": author,
        "n_seeds": len(all_results),
        "seeds": seeds,
        "h14_rank_mean": float(np.mean(h14_ranks)),
        "h14_rank_std": float(np.std(h14_ranks)),
        "h14_recovery_mean": float(np.mean(h14_recoveries)),
        "h14_recovery_std": float(np.std(h14_recoveries)),
        "top_heads": top_heads,
        "per_seed": all_results,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()