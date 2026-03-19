#!/usr/bin/env python3
"""Run specific transplant experiments and generate LinkedIn-style figures."""

import copy
import json
from pathlib import Path

import torch

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_module
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB
from sixteen_voices.steering import generate
from sixteen_voices import HEAD_DIM

ADAPTERS_DIR = Path("outputs/authors")
PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 160

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)

# (host, donor, head)
EXPERIMENTS = [
    ("minimalist", "poe", 11),
    ("minimalist", "carroll", 14),
    ("minimalist", "carroll", 11),
]


def transplant_head(host_delta, donor_delta, head):
    result = host_delta.clone()
    s, e = head * HEAD_DIM, (head + 1) * HEAD_DIM
    result[s:e, :] = donor_delta[s:e, :]
    return result


def inject_deltas(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def compute_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(out.loss).item()


def main():
    tokenizer = load_tokenizer()

    all_authors = set()
    for h, d, _ in EXPERIMENTS:
        all_authors.update([h, d])

    deltas = {a: load_adapter_deltas(str(ADAPTERS_DIR / a / "adapter")) for a in all_authors}
    models = {a: load_adapted_model(str(ADAPTERS_DIR / a / "adapter")) for a in all_authors}

    results = {}
    for host, donor, head in EXPERIMENTS:
        key = f"{host}_{donor}_H{head}"
        print(f"\n{'='*70}")
        print(f"  {host} + {donor}'s H{head}")
        print(f"{'='*70}")

        # Pure host
        torch.manual_seed(SEED)
        pure = generate(models[host], tokenizer, PROMPT, seed=SEED, max_new_tokens=MAX_NEW)
        pure_ppl = compute_ppl(models[host], tokenizer, EVAL_TEXT)
        print(f"  Pure {host} (PPL={pure_ppl:.1f}):")
        print(f"    {pure[:200]}")

        # Transplant
        t_deltas = {}
        for proj in ["q_proj", "v_proj"]:
            t_deltas[proj] = transplant_head(deltas[host][proj], deltas[donor][proj], head)

        model = copy.deepcopy(models[host])
        inject_deltas(model, t_deltas)
        torch.manual_seed(SEED)
        transplanted = generate(model, tokenizer, PROMPT, seed=SEED, max_new_tokens=MAX_NEW)
        t_ppl = compute_ppl(model, tokenizer, EVAL_TEXT)
        print(f"  + {donor}'s H{head} (PPL={t_ppl:.1f}):")
        print(f"    {transplanted[:200]}")

        results[key] = {
            "host": host,
            "donor": donor,
            "head": head,
            "pure_text": pure,
            "pure_ppl": pure_ppl,
            "transplant_text": transplanted,
            "transplant_ppl": t_ppl,
        }
        del model

    with open("outputs/transplant_new.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved outputs/transplant_new.json")


if __name__ == "__main__":
    main()