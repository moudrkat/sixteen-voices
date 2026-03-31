#!/usr/bin/env python3
"""Closed-loop SAE steering validation.

Steer the model, generate text, run it through the unsteered SAE,
check if the targeted feature's activation increased. Compare against
random directions.

Usage:
    uv run python scripts/validate_closedloop.py
    uv run python scripts/validate_closedloop.py --sae-dir outputs/sae_topk16_2048
"""

import argparse
import json
from pathlib import Path

import torch

from sixteen_voices import load_base_model, load_tokenizer
from sixteen_voices.sae import SparseAutoencoder
from sixteen_voices.steering import generate


FEATURES = {
    "simplicity": ([665], 8.0),
    "complexity": ([883, 993, 60], 8.0),
    "dialogue": ([1777, 689], 8.0),
    "questions": ([329], 8.0),
    "verse": ([344], 5.0),
    "first_person": ([1779, 627], 8.0),
    "dark_negation": ([1224], 10.0),
    "dark_looking": ([562], 10.0),
    "cozy_food": ([1988], 10.0),
    "cozy_tactile": ([930], 10.0),
    "carroll": ([815], 10.0),
    "dialect": ([61, 111], 10.0),
    "rhetorical_q": ([1385], 8.0),
}


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop SAE steering validation")
    parser.add_argument("--sae-dir", default="outputs/sae_topk16_2048")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae_dir = Path(args.sae_dir)
    output_path = (Path(args.output) if args.output
                   else sae_dir / "closedloop_validation.json")

    with open(sae_dir / "sae_config.json") as f:
        config = json.load(f)
    sae = SparseAutoencoder.load(sae_dir / "sae_weights.pt", config)
    w = sae.decoder.weight.detach()

    tokenizer = load_tokenizer()
    model = load_base_model()

    results = []

    for group_name, (feat_indices, scale) in FEATURES.items():
        vec = sum(scale * w[:, fi] for fi in feat_indices)

        targeted_up = 0
        random_up = 0

        for seed in range(args.n_seeds):
            baseline_text = generate(model, tokenizer, args.prompt,
                                     max_new_tokens=60, seed=seed)

            hook = model.transformer.ln_f.register_forward_hook(
                lambda mod, inp, out, v=vec: (
                    (out[0] + v,) + out[1:] if isinstance(out, tuple)
                    else out + v))
            steered_text = generate(model, tokenizer, args.prompt,
                                    max_new_tokens=60, seed=seed)
            hook.remove()

            rand_vec = torch.randn_like(vec)
            rand_vec = rand_vec / rand_vec.norm() * vec.norm()
            hook = model.transformer.ln_f.register_forward_hook(
                lambda mod, inp, out, v=rand_vec: (
                    (out[0] + v,) + out[1:] if isinstance(out, tuple)
                    else out + v))
            random_text = generate(model, tokenizer, args.prompt,
                                   max_new_tokens=60, seed=seed)
            hook.remove()

            acts = {}
            for text_type, text in [("baseline", baseline_text),
                                     ("steered", steered_text),
                                     ("random", random_text)]:
                ids = tokenizer.encode(text, return_tensors="pt")
                activations = []
                h = model.transformer.ln_f.register_forward_hook(
                    lambda mod, inp, out: activations.append(
                        inp[0].detach() if isinstance(inp, tuple)
                        else out.detach()))
                with torch.no_grad():
                    model(input_ids=ids)
                h.remove()
                a = activations[0].reshape(-1, activations[0].shape[-1])
                with torch.no_grad():
                    _, hidden = sae(a)
                acts[text_type] = sum(
                    hidden[:, fi].mean().item() for fi in feat_indices
                ) / len(feat_indices)

            if acts["steered"] > acts["baseline"]:
                targeted_up += 1
            if acts["random"] > acts["baseline"]:
                random_up += 1

        signal = ("YES" if targeted_up >= 7 and targeted_up > random_up * 1.5
                  else ("weak" if targeted_up > random_up else "NO"))
        print(f"{group_name:>15s}: targeted {targeted_up:2d}/{args.n_seeds}  "
              f"random {random_up:2d}/{args.n_seeds}  {signal}")

        results.append({
            "feature_group": group_name,
            "feature_indices": feat_indices,
            "scale": scale,
            "targeted_up": targeted_up,
            "random_up": random_up,
            "n_seeds": args.n_seeds,
            "signal": signal,
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()