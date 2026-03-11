#!/usr/bin/env python3
"""Cross-author head transplant: graft one author's head into another's adapter.

If style lives in specific heads, transplanting Shelley's H14 into Carroll's adapter
should shift Carroll toward Shelley's vocabulary — while keeping Carroll's structure.

Usage:
    python scripts/transplant.py carroll shelley
    python scripts/transplant.py carroll shelley --heads 14 11
    python scripts/transplant.py carroll shelley --all-heads
"""

import argparse
from pathlib import Path

import torch

from sixteen_voices import (
    HEAD_DIM,
    NUM_HEADS,
    compute_perplexity,
    extract_prose,
    generate,
    load_adapted_model,
    load_adapter_deltas,
    load_base_model,
    load_tokenizer,
)
from sixteen_voices.adapter import delta_to_AB
from sixteen_voices.model import get_attn_module

ADAPTERS_DIR = Path("outputs/authors")
DATA_DIR = Path("data/authors")

PROMPTS = [
    "Once upon a time",
    "The little girl walked into the forest",
    "There was an old woman who",
    "It was a dark and stormy",
]


def transplant_heads(recipient_delta, donor_delta, heads):
    """Replace specific head rows in recipient with donor's."""
    result = recipient_delta.clone()
    for h in heads:
        start = h * HEAD_DIM
        end = start + HEAD_DIM
        result[start:end, :] = donor_delta[start:end, :]
    return result


def inject_deltas(model, deltas_dict):
    """Write new A, B into the PEFT model for all projections."""
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas_dict[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def run_condition(label, adapter_path, tokenizer, deltas_override=None,
                  seed=42, ppl_texts=None):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    model = load_adapted_model(adapter_path)
    if deltas_override:
        inject_deltas(model, deltas_override)

    if ppl_texts:
        for name, text in ppl_texts.items():
            if text:
                ppl = compute_perplexity(model, tokenizer, text)
                print(f"  Perplexity on {name}: {ppl:.2f}")

    for prompt in PROMPTS:
        torch.manual_seed(seed)
        text = generate(model, tokenizer, prompt, seed=seed)
        print(f"\n  \"{prompt}\"")
        print(f"  -> {text}")

    del model


def main():
    parser = argparse.ArgumentParser(description="Head transplant experiment")
    parser.add_argument("recipient", help="Recipient author (keeps structure)")
    parser.add_argument("donor", help="Donor author (provides style head)")
    parser.add_argument("--heads", nargs="*", type=int, default=[14])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--also-reverse", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    r_path = str(ADAPTERS_DIR / args.recipient / "adapter")
    d_path = str(ADAPTERS_DIR / args.donor / "adapter")

    r_deltas = load_adapter_deltas(r_path)
    d_deltas = load_adapter_deltas(d_path)

    ppl_texts = {}
    for author in [args.recipient, args.donor]:
        path = DATA_DIR / f"{author}.txt"
        ppl_texts[author] = extract_prose(path.read_text()) if path.exists() else None

    # Baselines
    run_condition(f"PURE {args.recipient.upper()}", r_path, tokenizer,
                  seed=args.seed, ppl_texts=ppl_texts)
    run_condition(f"PURE {args.donor.upper()}", d_path, tokenizer,
                  seed=args.seed, ppl_texts=ppl_texts)

    if args.all_heads:
        for h in range(NUM_HEADS):
            t = {p: transplant_heads(r_deltas[p], d_deltas[p], [h]) for p in ["q_proj", "v_proj"]}
            run_condition(
                f"TRANSPLANT H{h}: {args.donor}->H{h}->{args.recipient}",
                r_path, tokenizer, deltas_override=t, seed=args.seed, ppl_texts=ppl_texts,
            )
    else:
        t = {p: transplant_heads(r_deltas[p], d_deltas[p], args.heads)
             for p in ["q_proj", "v_proj"]}
        heads_str = ",".join(f"H{h}" for h in args.heads)
        run_condition(
            f"TRANSPLANT: {args.recipient} + {args.donor}'s {heads_str}",
            r_path, tokenizer, deltas_override=t, seed=args.seed, ppl_texts=ppl_texts,
        )

        if args.also_reverse:
            rev = {p: transplant_heads(d_deltas[p], r_deltas[p], args.heads)
                   for p in ["q_proj", "v_proj"]}
            run_condition(
                f"REVERSE: {args.donor} + {args.recipient}'s {heads_str}",
                d_path, tokenizer, deltas_override=rev, seed=args.seed, ppl_texts=ppl_texts,
            )

    # Base model reference
    print(f"\n{'='*70}\n  BASE MODEL (no LoRA)\n{'='*70}")
    base = load_base_model()
    for name, text in ppl_texts.items():
        if text:
            ppl = compute_perplexity(base, tokenizer, text)
            print(f"  Perplexity on {name}: {ppl:.2f}")
    for prompt in PROMPTS:
        text = generate(base, tokenizer, prompt, seed=args.seed)
        print(f"\n  \"{prompt}\"\n  -> {text}")


if __name__ == "__main__":
    main()
