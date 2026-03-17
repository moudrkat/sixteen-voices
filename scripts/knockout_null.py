#!/usr/bin/env python3
"""Null baseline: knockout on random (untrained) LoRA adapters.

Creates fresh LoRA adapters (random init) and runs the same knockout analysis.
Establishes that head specialization is learned, not an artifact of random weights.

Usage:
    python scripts/knockout_null.py
    python scripts/knockout_null.py --seeds 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sixteen_voices import (
    NUM_HEADS,
    HEAD_DIM,
    RANK,
    compute_perplexity,
    load_base_model,
    load_eval_text,
    load_tokenizer,
)
from sixteen_voices.adapter import knockout_all_except, delta_to_AB
from sixteen_voices.model import create_lora_model, get_attn_module
TEST_AUTHORS = ["shelley", "grimm", "homer", "poe", "carroll", "alcott"]


def run_null_knockout(seed, tokenizer, eval_texts):
    """Create a random LoRA and run knockout on each eval text."""
    torch.manual_seed(seed)

    model = create_lora_model()
    model.eval()

    # PEFT initializes B=0 (no-op). Fill with random weights matching trained scale.
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A = getattr(attn, proj).lora_A["default"].weight.data
        B = getattr(attn, proj).lora_B["default"].weight.data
        torch.nn.init.normal_(A, std=0.02)
        torch.nn.init.normal_(B, std=0.02)

    # Extract deltas
    deltas = {}
    for proj in ["q_proj", "v_proj"]:
        A = getattr(attn, proj).lora_A["default"].weight.data.clone()
        B = getattr(attn, proj).lora_B["default"].weight.data.clone()
        deltas[proj] = B @ A

    base_model = load_base_model()

    results_per_text = {}
    for author, text in eval_texts.items():
        # Restore original random LoRA weights
        for proj in ["q_proj", "v_proj"]:
            A_new, B_new = delta_to_AB(deltas[proj])
            getattr(attn, proj).lora_A["default"].weight.data.copy_(A_new)
            getattr(attn, proj).lora_B["default"].weight.data.copy_(B_new)

        full_ppl = compute_perplexity(model, tokenizer, text)
        base_ppl = compute_perplexity(base_model, tokenizer, text)
        improvement = base_ppl - full_ppl

        head_recovery = {}
        for h in range(NUM_HEADS):
            for proj in ["q_proj", "v_proj"]:
                new_delta = knockout_all_except(deltas[proj], h)
                A_new, B_new = delta_to_AB(new_delta)
                getattr(attn, proj).lora_A["default"].weight.data.copy_(A_new)
                getattr(attn, proj).lora_B["default"].weight.data.copy_(B_new)

            h_ppl = compute_perplexity(model, tokenizer, text)
            if abs(improvement) < 0.01:
                rec = 0.0
            else:
                rec = (base_ppl - h_ppl) / improvement
            head_recovery[f"H{h}"] = round(rec, 3)

        results_per_text[author] = {
            "base_ppl": round(base_ppl, 2),
            "random_lora_ppl": round(full_ppl, 2),
            "improvement": round(improvement, 2),
            "head_recovery": head_recovery,
        }

    del model, base_model
    return results_per_text


def main():
    parser = argparse.ArgumentParser(description="Null baseline knockout")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--output", type=str, default="outputs/knockout_null_baseline.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    eval_texts = {}
    for author in TEST_AUTHORS:
        try:
            eval_texts[author] = load_eval_text(author)
        except FileNotFoundError:
            pass
    print(f"Eval texts: {list(eval_texts.keys())}")

    all_results = {}
    for seed in range(args.seeds):
        print(f"\n{'='*60}\nRandom seed {seed}\n{'='*60}")
        results = run_null_knockout(seed, tokenizer, eval_texts)
        all_results[f"seed_{seed}"] = results

        for author, r in results.items():
            rec = r["head_recovery"]
            vals = list(rec.values())
            std = np.std(vals)
            best_h = max(rec, key=rec.get)
            rec_str = " ".join(f"{rec[f'H{h}']:+.2f}" for h in range(NUM_HEADS))
            print(f"  {author:12s}  std={std:.3f}  best={best_h}  {rec_str}")

    # Aggregate
    print(f"\n{'='*60}\nAGGREGATE (mean +/- std across seeds)\n{'='*60}")
    for author in eval_texts:
        all_stds = []
        for seed_key in all_results:
            vals = list(all_results[seed_key][author]["head_recovery"].values())
            all_stds.append(np.std(vals))
        print(f"  {author:12s}  head_std={np.mean(all_stds):.3f}+/-{np.std(all_stds):.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")

    # Compare with trained if available
    trained_path = Path("outputs/knockout_all_heads.json")
    if trained_path.exists():
        trained = json.load(open(trained_path))
        print(f"\n{'='*60}\nCOMPARISON: random vs trained\n{'='*60}")
        for author in eval_texts:
            if author not in trained:
                continue
            t_vals = list(trained[author]["head_recovery"].values())
            t_std = np.std(t_vals)
            r_stds = [np.std(list(all_results[sk][author]["head_recovery"].values()))
                       for sk in all_results]
            print(f"  {author:12s}  trained_std={t_std:.3f}  "
                  f"random_std={np.mean(r_stds):.3f}  "
                  f"ratio={t_std / max(np.mean(r_stds), 0.001):.1f}x")


if __name__ == "__main__":
    main()
