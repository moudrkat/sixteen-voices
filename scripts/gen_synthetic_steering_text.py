#!/usr/bin/env python3
"""Generate steering text samples for synthetic adapters' dominant heads.

PPL measured on each adapter's OWN eval text (data/eval/<adapter>.txt),
not on a generic TinyStories sentence — so PPL spike when killing
H_dom is meaningful (in-distribution test).

Saves to outputs/synthetic_steering_text.json.
"""

import json
from pathlib import Path

from sixteen_voices.model import load_adapted_model, load_tokenizer, get_attn_out
from sixteen_voices.steering import generate, steered_perplexity
from sixteen_voices.text import load_eval_text

ADAPTERS_DIR = Path("outputs/authors")
OUT_PATH = Path("outputs/synthetic_steering_text.json")
EVAL_LENGTH = 2000  # words, same as steering_sweep.py

# (adapter_name, dominant_head) — dominant head from outputs/knockout_all_heads.json
ADAPTERS = [
    ("minimalist",     11),
    ("questioner",     15),
    ("firstperson",    11),
    ("dialogue",       10),
    ("poet",           11),  # rec 0.517 (high)
    ("unusual_vocab",  14),  # rec 0.598 (highest)
    ("cozy",           11),  # rec 0.253 (low)
    ("dark",           11),  # rec 0.243 (low)
]

PROMPT = "It was a dark and stormy"
SEED = 42
MAX_NEW = 80
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


def main():
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Always overwrite (eval text changed — old PPLs are stale)
    results = {}

    for adapter_name, head_idx in ADAPTERS:
        head_key = f"H{head_idx}"
        print(f"\n=== {adapter_name} × {head_key} ===")

        # Load adapter's own eval text (in-distribution PPL test)
        eval_text = load_eval_text(adapter_name, length=EVAL_LENGTH * 7)
        words = eval_text.split()
        if len(words) > EVAL_LENGTH:
            eval_text = " ".join(words[:EVAL_LENGTH])
        print(f"  eval text: {len(eval_text.split())} words from data/eval/{adapter_name}.txt")

        print(f"  Loading adapter {adapter_name}...")
        model = load_adapted_model(str(ADAPTERS_DIR / adapter_name / "adapter"))
        attn_out = get_attn_out(model)

        scale_results = {}
        for scale in SCALES:
            print(f"  scale={scale}...")
            head_scales = {head_idx: scale}
            text = generate(model, tokenizer, PROMPT,
                            head_scales=head_scales,
                            attn_out=attn_out,
                            seed=SEED, max_new_tokens=MAX_NEW)
            ppl = steered_perplexity(model, tokenizer, eval_text,
                                     head_scales=head_scales,
                                     attn_out=attn_out)
            scale_results[str(scale)] = {
                "text": text,
                "ppl": round(float(ppl), 1),
            }
            print(f"    PPL={ppl:.1f}: {text[:90]}...")

        results[adapter_name] = {
            "head": head_key,
            "scales": scale_results,
        }

        # Free memory before next adapter
        del model, attn_out

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")

    # Summary print: 0×, 1×, 2× for each adapter
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, info in results.items():
        print(f"\n{name} × {info['head']}")
        for s in ("0.0", "1.0", "2.0"):
            r = info["scales"][s]
            print(f"  {s}× PPL={r['ppl']}: {r['text'][:120]}")


if __name__ == "__main__":
    main()
