#!/usr/bin/env python3
"""Multi-head steering: scale attention heads at inference and measure effect.

Tests head steering configurations per author, comparing PPL and generated text.

Usage:
    python scripts/steer.py
    python scripts/steer.py --authors shelley poe
    python scripts/steer.py --prompt "The dark castle stood" --seed 42
"""

import argparse
import json
from pathlib import Path

from sixteen_voices import (
    extract_prose,
    generate,
    get_attn_out,
    load_adapted_model,
    load_tokenizer,
    steered_perplexity,
)

ADAPTERS_DIR = Path("outputs/authors")
DATA_DIR = Path("data/authors")

DEFAULT_PROMPT = "The raven sat on the old tree and"
DEFAULT_SEED = 123

# Per-author configs from knockout results:
# sign = H14 polarity, best_head = strongest non-H14/non-H11 head
AUTHOR_CONFIGS = {
    "carroll":    {"sign": "+", "best_head": 1},
    "poe":        {"sign": "+", "best_head": 3},
    "grimm":      {"sign": "+", "best_head": 1},
    "minimalist": {"sign": "+", "best_head": 15},
    "shelley":    {"sign": "-", "best_head": 10},
    "wilde":      {"sign": "-", "best_head": 2},
    "lovecraft":  {"sign": "+", "best_head": 15},
    "tennyson":   {"sign": "+", "best_head": 3},
}


def main():
    parser = argparse.ArgumentParser(description="Multi-head steering experiment")
    parser.add_argument("--authors", nargs="+", default=list(AUTHOR_CONFIGS.keys()))
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default="outputs/multihead_text.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    results = {}

    for author in args.authors:
        if author not in AUTHOR_CONFIGS:
            print(f"  Skip {author} (no config)")
            continue
        cfg = AUTHOR_CONFIGS[author]

        adapter_path = ADAPTERS_DIR / author / "adapter"
        txt_path = DATA_DIR / f"{author}.txt"
        if not adapter_path.exists() or not txt_path.exists():
            print(f"  Skip {author}")
            continue

        eval_text = extract_prose(txt_path.read_text())
        model = load_adapted_model(str(adapter_path))
        attn_out = get_attn_out(model)
        bh = cfg["best_head"]

        # Build configs to test
        configs = {
            "baseline": None,
            "H14_x2":   {14: 2.0},
            "H14_x0":   {14: 0.0},
        }
        if bh is not None:
            configs[f"H{bh}_x2"] = {bh: 2.0}
            configs[f"H14x2+H{bh}x2"] = {14: 2.0, bh: 2.0}
            configs[f"H14x2+H{bh}x1.5"] = {14: 2.0, bh: 1.5}
            configs[f"H14x1.5+H{bh}x1.5"] = {14: 1.5, bh: 1.5}
            configs[f"H14x2+H{bh}x0.5"] = {14: 2.0, bh: 0.5}
        configs["top3_x1.5"] = {3: 1.5, 11: 1.5, 14: 1.5}
        configs["H14x2+H3x0.5+H11x0.5"] = {14: 2.0, 3: 0.5, 11: 0.5}

        print(f"\n{'='*70}")
        print(f"  {author.upper()} [H14{cfg['sign']}]  best_other=H{bh}")
        print(f"  Prompt: \"{args.prompt}\"  seed={args.seed}")
        print(f"{'='*70}")

        author_results = {}
        baseline_ppl = None

        for name, hs in configs.items():
            p = steered_perplexity(model, tokenizer, eval_text, hs, attn_out)
            t = generate(model, tokenizer, args.prompt, head_scales=hs,
                         attn_out=attn_out, seed=args.seed)
            author_results[name] = {"ppl": round(p, 1), "text": t}

            if name == "baseline":
                baseline_ppl = p

            chg = (p - baseline_ppl) / baseline_ppl * 100 if baseline_ppl else 0
            print(f"\n  {name:30s}  PPL={p:8.1f} ({chg:+6.1f}%)")
            print(f"    {t[:160]}")

        results[author] = author_results
        del model

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
