#!/usr/bin/env python3
"""Try several transplant pairs: donor H14 into various hosts."""

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
MAX_NEW = 100

PAIRS = [
    # (host, donor, head)
    ("dialogue", "poe", 14),
    ("poet", "poe", 14),
    ("minimalist", "poe", 14),
    ("dialogue", "carroll", 14),
    ("poet", "carroll", 14),
    ("minimalist", "carroll", 14),
    ("poet", "carroll", 11),
    ("dialogue", "carroll", 11),
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
    for h, d, _ in PAIRS:
        all_authors.update([h, d])

    deltas = {a: load_adapter_deltas(str(ADAPTERS_DIR / a / "adapter")) for a in all_authors}
    models = {a: load_adapted_model(str(ADAPTERS_DIR / a / "adapter")) for a in all_authors}

    eval_text = "Once upon a time there was a little girl who lived in a small house near the forest"

    for host, donor, head in PAIRS:
        print(f"\n{'='*70}")
        print(f"  {host} + {donor}'s H{head}")
        print(f"{'='*70}")

        # Pure host
        pure = generate(models[host], tokenizer, PROMPT, seed=SEED, max_new_tokens=MAX_NEW)
        pure_ppl = compute_ppl(models[host], tokenizer, eval_text)
        print(f"  Pure {host} (PPL={pure_ppl:.1f}):")
        print(f"    {pure[:150]}")

        # Transplant
        t_deltas = {}
        for proj in ["q_proj", "v_proj"]:
            t_deltas[proj] = transplant_head(deltas[host][proj], deltas[donor][proj], head)

        model = copy.deepcopy(models[host])
        inject_deltas(model, t_deltas)
        transplanted = generate(model, tokenizer, PROMPT, seed=SEED, max_new_tokens=MAX_NEW)
        t_ppl = compute_ppl(model, tokenizer, eval_text)
        print(f"  + {donor}'s H{head} (PPL={t_ppl:.1f}):")
        print(f"    {transplanted[:150]}")
        del model


if __name__ == "__main__":
    main()