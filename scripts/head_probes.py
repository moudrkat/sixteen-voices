#!/usr/bin/env python3
"""Probe what each attention head does — with and without adapters.

For each head, generates text with only that head active (all others at 0×)
and with that head killed (only that head at 0×). Also runs on the base model
(no adapter) to see what heads do by default in TinyStories.

Measures:
- PPL on a standard eval text with each head solo / killed
- Generated text samples with each head solo / killed
- Vocabulary analysis: what words appear more/less when each head is active

Usage:
    uv run python scripts/head_probes.py
    uv run python scripts/head_probes.py --base-only
    uv run python scripts/head_probes.py --authors poe grimm
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from sixteen_voices import (
    NUM_HEADS,
    load_adapted_model,
    load_base_model,
    load_tokenizer,
)
from sixteen_voices.model import get_attn_out
from sixteen_voices.steering import make_hook

ADAPTERS_DIR = Path("outputs/authors")
OUTPUT_PATH = Path("outputs/head_probes.json")

PROMPTS = [
    "Once upon a time",
    "It was a dark and stormy",
    "The little girl walked into",
    "There was a king who had",
    "In the morning the sun",
]

MAX_NEW = 60
TEMPERATURE = 0.7
SEED = 42

EVAL_TEXT = (
    "Once upon a time there was a little girl who lived in a small house "
    "near the forest with her mother and father and a cat named Whiskers. "
    "Every day she would go to the garden and play with the flowers and "
    "the butterflies and the birds that sang in the trees."
)


def generate_steered(model, tokenizer, prompt, head_scales, seed=SEED):
    """Generate text with head steering."""
    torch.manual_seed(seed)
    attn_out = get_attn_out(model)
    hook = attn_out.register_forward_pre_hook(make_hook(head_scales))
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, temperature=TEMPERATURE,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    hook.remove()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_normal(model, tokenizer, prompt, seed=SEED):
    """Generate without steering."""
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=MAX_NEW, temperature=TEMPERATURE,
            do_sample=True, top_k=50, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def compute_ppl(model, tokenizer, text, head_scales=None):
    """Compute PPL optionally with steering."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    hook = None
    if head_scales:
        attn_out = get_attn_out(model)
        hook = attn_out.register_forward_pre_hook(make_hook(head_scales))
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    if hook:
        hook.remove()
    return torch.exp(out.loss).item()


def word_freq(texts):
    """Count word frequencies across texts."""
    c = Counter()
    for t in texts:
        words = t.lower().split()
        c.update(words)
    return c


def probe_model(model, tokenizer, label):
    """Run full head probe on a model."""
    print(f"\n{'='*60}")
    print(f"Probing: {label}")
    print(f"{'='*60}")

    result = {"label": label}

    # Normal generation and PPL
    normal_ppl = compute_ppl(model, tokenizer, EVAL_TEXT)
    normal_texts = [generate_normal(model, tokenizer, p) for p in PROMPTS]
    result["normal_ppl"] = round(normal_ppl, 2)
    result["normal_samples"] = normal_texts[:2]  # just keep 2 for readability

    # Per-head: solo (only this head) and kill (everything except this head)
    head_data = {}
    for h in range(NUM_HEADS):
        # Solo: only this head at 1×, all others at 0×
        solo_scales = {i: (1.0 if i == h else 0.0) for i in range(NUM_HEADS)}
        solo_ppl = compute_ppl(model, tokenizer, EVAL_TEXT, solo_scales)
        solo_texts = [generate_steered(model, tokenizer, p, solo_scales) for p in PROMPTS[:2]]

        # Kill: this head at 0×, all others at 1×
        kill_scales = {h: 0.0}
        kill_ppl = compute_ppl(model, tokenizer, EVAL_TEXT, kill_scales)

        # Amplify: this head at 2×
        amp_scales = {h: 2.0}
        amp_ppl = compute_ppl(model, tokenizer, EVAL_TEXT, amp_scales)
        amp_texts = [generate_steered(model, tokenizer, p, amp_scales) for p in PROMPTS[:2]]

        # Word frequency diff: amplified vs normal
        normal_words = word_freq(normal_texts)
        amp_words = word_freq(amp_texts)
        # Words that appear much more when amplified
        boosted = []
        for word, count in amp_words.most_common(50):
            normal_count = normal_words.get(word, 0)
            if count > normal_count + 1:
                boosted.append(word)

        head_data[f"H{h}"] = {
            "solo_ppl": round(solo_ppl, 2),
            "kill_ppl": round(kill_ppl, 2),
            "amp_ppl": round(amp_ppl, 2),
            "ppl_impact": round(kill_ppl - normal_ppl, 2),  # +ve = this head helps
            "solo_samples": solo_texts,
            "amp_samples": amp_texts,
            "boosted_words": boosted[:10],
        }

        impact = kill_ppl - normal_ppl
        icon = "!!" if abs(impact) > 2 else "." if abs(impact) < 0.5 else " "
        print(f"  H{h:2d}: solo={solo_ppl:8.1f}  kill={kill_ppl:8.1f}  "
              f"amp={amp_ppl:8.1f}  impact={impact:+.1f} {icon}")

    result["heads"] = head_data
    return result


def main():
    parser = argparse.ArgumentParser(description="Head probe analysis")
    parser.add_argument("--authors", nargs="+", help="Probe these adapted models")
    parser.add_argument("--base-only", action="store_true", help="Only probe base model")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    results = {}

    # Always probe base model
    print("Loading base model...")
    base_model = load_base_model()
    results["base"] = probe_model(base_model, tokenizer, "TinyStories-1Layer-21M (no adapter)")

    if not args.base_only:
        authors = args.authors or ["poe", "grimm", "twain", "browne", "minimalist"]
        for author in authors:
            adapter_path = ADAPTERS_DIR / author / "adapter"
            if not adapter_path.exists():
                print(f"Skipping {author} — no adapter found")
                continue
            model = load_adapted_model(adapter_path)
            results[author] = probe_model(model, tokenizer, f"{author} adapter")
            del model

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Head importance (PPL impact when killed)")
    print(f"{'='*60}")
    header = f"{'':>12s}"
    for h in range(NUM_HEADS):
        header += f" H{h:2d}"
    print(header)
    for name, data in results.items():
        row = f"{name:>12s}"
        for h in range(NUM_HEADS):
            impact = data["heads"][f"H{h}"]["ppl_impact"]
            row += f" {impact:+4.0f}" if abs(impact) >= 1 else f" {impact:+4.1f}"
        print(row)


if __name__ == "__main__":
    main()