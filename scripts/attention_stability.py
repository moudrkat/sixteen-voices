#!/usr/bin/env python3
"""Test attention entropy stability across many prompts.

Generates prompts from 6 templates and measures per-head entropy
to determine which heads are consistently focused vs. spread.

Usage:
    uv run python scripts/attention_stability.py
    uv run python scripts/attention_stability.py --n-prompts 1000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sixteen_voices import NUM_HEADS, load_base_model, load_tokenizer

OUTPUT_PATH = Path("outputs/attention_stability.json")

# Word pools
SUBJECTS = [
    "the girl", "the boy", "the king", "the queen", "the cat", "the dog",
    "the bird", "the fish", "the old woman", "the knight", "the princess",
    "the farmer", "the witch", "the dragon", "the bear", "the mouse",
    "the owl", "the rabbit", "the fox", "the wolf", "a child", "the giant",
    "the fairy", "the pirate", "the baker", "the teacher", "my friend",
    "the baby", "the spider", "the frog", "the lion", "the turtle",
    "the monkey", "the goat", "the hen", "the crow", "the deer", "the lamb",
    "the snake", "the eagle",
]
VERBS = [
    "walked into", "ran towards", "looked at", "found", "saw", "heard",
    "touched", "carried", "opened", "climbed", "flew over", "swam across",
    "hid behind", "jumped over", "sat near", "stood beside", "fell into",
    "sang about", "dreamed of", "whispered to", "painted", "built",
    "chased", "followed", "called", "loved", "watched", "held", "broke",
    "fixed",
]
PLACES = [
    "the forest", "the castle", "the river", "the mountain", "the garden",
    "the house", "the bridge", "the cave", "the tower", "the tree",
    "the door", "the window", "the lake", "the moon", "the sun",
    "the star", "the flower", "the stone", "the road", "the sea",
    "the village", "the hill", "the pond", "the field", "the wall",
    "the well", "the church", "the market", "the barn", "the shore",
]
ADJECTIVES = [
    "big", "small", "dark", "bright", "old", "young", "beautiful", "scary",
    "quiet", "loud", "cold", "warm", "deep", "tall", "tiny", "golden",
    "silver", "magical", "broken", "hidden",
]
ADVERBS = [
    "quickly", "slowly", "carefully", "happily", "sadly", "silently",
    "bravely", "gently", "suddenly", "finally",
]
TIMES = [
    "one morning", "at night", "at dawn", "in winter", "in summer",
    "long ago", "one evening", "at sunset", "before sunrise", "after the rain",
]

TEMPLATE_NAMES = [
    "S-V-P", "T-S-V-AP", "S-adv-V-P-V-P",
    "there-was-A-S-P", "dialogue", "long",
]


def make_prompt(template_id):
    """Generate a prompt from the given template."""
    s = random.choice
    if template_id == 0:  # Subject verb place
        return f"{s(SUBJECTS)} {s(VERBS)} {s(PLACES)}"
    elif template_id == 1:  # Time, subject verb adj place
        return (f"{s(TIMES)} {s(SUBJECTS)} {s(VERBS)} "
                f"the {s(ADJECTIVES)} {s(PLACES).replace('the ', '')}")
    elif template_id == 2:  # Subject adverb verb place and verb place
        return (f"{s(SUBJECTS)} {s(ADVERBS)} {s(VERBS)} "
                f"{s(PLACES)} and {s(VERBS)} {s(PLACES)}")
    elif template_id == 3:  # There was adj subject in place
        return (f"there was a {s(ADJECTIVES)} "
                f"{s(SUBJECTS).replace('the ', '')} in {s(PLACES)}")
    elif template_id == 4:  # Dialogue
        return (f"{s(SUBJECTS)} said I will "
                f"{s(VERBS).split()[0]} the {s(ADJECTIVES)} "
                f"{s(PLACES).replace('the ', '')}")
    elif template_id == 5:  # Long
        return (f"{s(TIMES)} {s(SUBJECTS)} {s(ADVERBS)} {s(VERBS)} "
                f"{s(PLACES)} and {s(SUBJECTS)} {s(VERBS)} "
                f"{s(PLACES)} {s(ADVERBS)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = load_tokenizer()
    model = load_base_model()

    # Generate prompts
    prompts = []
    template_ids = []
    for i in range(args.n_prompts):
        tid = i % 6
        prompts.append(make_prompt(tid))
        template_ids.append(tid)

    print(f"Running {len(prompts)} prompts across 6 templates (seed={args.seed})...")

    all_entropies = {h: [] for h in range(NUM_HEADS)}
    template_entropies = {t: {h: [] for h in range(NUM_HEADS)} for t in range(6)}

    for idx, prompt in enumerate(prompts):
        if idx % 2000 == 0:
            print(f"  {idx}/{len(prompts)}...")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attn = outputs.attentions[0][0].numpy()
        seq_len = attn.shape[-1]

        for h in range(NUM_HEADS):
            a = attn[h, :seq_len, :seq_len]
            ents = []
            for i in range(1, seq_len):
                row = a[i, :i + 1]
                row_nz = row[row > 1e-8]
                if len(row_nz) > 0:
                    ents.append(-float((row_nz * np.log2(row_nz)).sum()))
            me = float(np.mean(ents)) if ents else 0.0
            all_entropies[h].append(me)
            template_entropies[template_ids[idx]][h].append(me)

    # Compute "most diffuse" counts
    most_diffuse = {h: 0 for h in range(NUM_HEADS)}
    for i in range(len(prompts)):
        ents = [all_entropies[h][i] for h in range(NUM_HEADS)]
        most_diffuse[np.argmax(ents)] += 1

    # Build results
    results = {
        "n_prompts": len(prompts),
        "seed": args.seed,
        "n_templates": 6,
        "overall": {},
        "most_diffuse_counts": {f"H{h}": most_diffuse[h] for h in range(NUM_HEADS)},
        "per_template": {},
    }

    print(f"\n{'Head':>6s} {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}")
    print("-" * 42)
    for h in range(NUM_HEADS):
        e = np.array(all_entropies[h])
        results["overall"][f"H{h}"] = {
            "mean": round(float(e.mean()), 4),
            "std": round(float(e.std()), 4),
            "min": round(float(e.min()), 4),
            "max": round(float(e.max()), 4),
        }
        print(f"  H{h:2d}   {e.mean():6.3f}   {e.std():6.3f}   {e.min():6.3f}   {e.max():6.3f}")

    print(f"\nMost diffuse head across {len(prompts)} prompts:")
    for h in sorted(most_diffuse, key=most_diffuse.get, reverse=True):
        if most_diffuse[h] > 0:
            pct = most_diffuse[h] / len(prompts) * 100
            print(f"  H{h}: {most_diffuse[h]}/{len(prompts)} ({pct:.1f}%)")

    # Per-template stats
    print("\nPer template:")
    for t in range(6):
        n = len(template_entropies[t][0])
        md = {h: 0 for h in range(NUM_HEADS)}
        for i in range(n):
            ents = [template_entropies[t][h][i] for h in range(NUM_HEADS)]
            md[np.argmax(ents)] += 1
        means = {h: float(np.mean(template_entropies[t][h])) for h in range(NUM_HEADS)}
        top3 = sorted(means, key=means.get, reverse=True)[:3]
        winner = max(md, key=md.get)

        results["per_template"][TEMPLATE_NAMES[t]] = {
            "n": n,
            "most_diffuse_counts": {f"H{h}": md[h] for h in range(NUM_HEADS) if md[h] > 0},
            "head_means": {f"H{h}": round(means[h], 4) for h in range(NUM_HEADS)},
        }
        print(f"  {TEMPLATE_NAMES[t]:>20s} (n={n}): "
              f"winner H{winner} ({md[winner]/n*100:.0f}%), "
              f"top3: H{top3[0]}({means[top3[0]]:.2f}) "
              f"H{top3[1]}({means[top3[1]]:.2f}) "
              f"H{top3[2]}({means[top3[2]]:.2f})")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()