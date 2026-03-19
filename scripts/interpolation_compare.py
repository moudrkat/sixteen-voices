#!/usr/bin/env python3
"""Run full 11-step interpolation for two pairs side by side."""

import copy
import json
from pathlib import Path

import torch

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_module
from sixteen_voices.adapter import load_adapter_deltas, delta_to_AB

ADAPTERS_DIR = Path("outputs/authors")
PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 80
EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers"
)

PAIRS = [
    ("grimm", "minimalist"),
    ("carroll", "poet"),
]

ALPHAS = [round(i / 10, 1) for i in range(11)]


def inject_deltas(model, deltas):
    attn = get_attn_module(model)
    for proj in ["q_proj", "v_proj"]:
        A, B = delta_to_AB(deltas[proj])
        getattr(attn, proj).lora_A["default"].weight.data.copy_(A)
        getattr(attn, proj).lora_B["default"].weight.data.copy_(B)


def interpolate_deltas(d1, d2, alpha):
    return {proj: (1 - alpha) * d1[proj] + alpha * d2[proj] for proj in ["q_proj", "v_proj"]}


def compute_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(out.loss).item()


def generate(model, tokenizer, prompt, seed=42, max_new=80):
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    plen = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                             temperature=0.8, top_k=50, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()


def main():
    tokenizer = load_tokenizer()
    all_authors = set()
    for a, b in PAIRS:
        all_authors.update([a, b])

    deltas = {a: load_adapter_deltas(str(ADAPTERS_DIR / a / "adapter")) for a in all_authors}
    template = load_adapted_model(str(ADAPTERS_DIR / "carroll" / "adapter"))

    for author_a, author_b in PAIRS:
        print(f"\n{'='*70}")
        print(f"  {author_a} → {author_b}")
        print(f"{'='*70}")
        for alpha in ALPHAS:
            blended = interpolate_deltas(deltas[author_a], deltas[author_b], alpha)
            model = copy.deepcopy(template)
            inject_deltas(model, blended)
            text = generate(model, tokenizer, PROMPT, seed=SEED, max_new=MAX_NEW)
            ppl = compute_ppl(model, tokenizer, EVAL_TEXT)
            print(f"  α={alpha:.1f}  PPL={ppl:.1f}  {text[:130]}")
            del model


if __name__ == "__main__":
    main()