#!/usr/bin/env python3
"""Train LoRA and measure vocabulary redirection at every step.

Tracks how base-to-adapted vocabulary overlap evolves during training.
Does redirection happen gradually or as a phase transition?

Usage:
    python scripts/redirection_timeline.py --author shelley
    python scripts/redirection_timeline.py --author shelley grimm homer
    python scripts/redirection_timeline.py --author shelley --measure-every 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sixteen_voices import (
    HEAD_DIM,
    NUM_HEADS,
    TextChunkDataset,
    get_attn_out,
    load_base_model,
    load_tokenizer,
)
from sixteen_voices.model import create_lora_model

DATA_DIR = Path("data/authors")
TOP_K = 50
FOCUS_HEADS = [3, 7, 12, 14]

EPOCHS = 8
BATCH_SIZE = 4
LR = 5e-4
MAX_LENGTH = 512
STRIDE = 256

TEXT = "Once upon a time there was a little"


def make_knockout_hook(head_idx):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        s = head_idx * HEAD_DIM
        h[:, :, s : s + HEAD_DIM] = 0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def get_promoted_words(model, tokenizer, text, head_idx):
    attn_out = get_attn_out(model)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        full = {
            tokenizer.decode(i.item()).strip()
            for i in torch.topk(
                torch.softmax(model(**inputs).logits[0, -1], -1), TOP_K
            ).indices
        }

    hook = attn_out.register_forward_hook(make_knockout_hook(head_idx))
    with torch.no_grad():
        ko = {
            tokenizer.decode(i.item()).strip()
            for i in torch.topk(
                torch.softmax(model(**inputs).logits[0, -1], -1), TOP_K
            ).indices
        }
    hook.remove()
    return full - ko


def measure_overlap(model, tokenizer, base_promoted):
    overlaps = {}
    for h in FOCUS_HEADS:
        adapted = get_promoted_words(model, tokenizer, TEXT, h)
        base_top10 = set(list(base_promoted[h])[:10])
        adapted_top10 = set(list(adapted)[:10])
        overlaps[h] = len(base_top10 & adapted_top10) / max(len(base_top10), 1)
    return overlaps


def train_with_measurements(author, tokenizer, base_promoted, measure_every=10):
    txt_path = DATA_DIR / f"{author}.txt"
    if not txt_path.exists():
        print(f"  ERROR: no text for '{author}'")
        return None

    text = txt_path.read_text(encoding="utf-8")
    dataset = TextChunkDataset(text, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    print(f"  {len(dataset)} chunks -> {n_train} train / {n_val} val")

    model = create_lora_model()
    model.train()
    dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Step 0 measurement
    model.eval()
    overlap0 = measure_overlap(model, tokenizer, base_promoted)
    model.train()
    measurements = [{
        "step": 0, "epoch": 0, "loss": None,
        "overlap": overlap0,
        "mean_overlap": float(np.mean(list(overlap0.values()))),
    }]
    print(f"    step 0: mean_overlap={measurements[0]['mean_overlap']:.1%}")

    step = 0
    for epoch in range(EPOCHS):
        for batch in dataloader:
            loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % measure_every == 0:
                model.eval()
                overlap = measure_overlap(model, tokenizer, base_promoted)
                model.train()
                mean_ov = float(np.mean(list(overlap.values())))
                measurements.append({
                    "step": step, "epoch": epoch + 1,
                    "loss": loss.item(),
                    "overlap": overlap,
                    "mean_overlap": mean_ov,
                })
                print(f"    step {step}: loss={loss.item():.4f}, mean_overlap={mean_ov:.1%}")

    del model
    return measurements


def main():
    parser = argparse.ArgumentParser(description="Vocabulary redirection timeline")
    parser.add_argument("--author", nargs="+", default=["shelley", "grimm", "homer"])
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--output", type=str, default="outputs/redirection_timeline.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # Base model promoted words
    print("Base model vocabulary...")
    base_model = load_base_model()
    base_promoted = {}
    for h in FOCUS_HEADS:
        base_promoted[h] = get_promoted_words(base_model, tokenizer, TEXT, h)
        print(f"  H{h}: {len(base_promoted[h])} promoted words")
    del base_model

    all_data = {}
    for author in args.author:
        print(f"\n{'='*60}\n  Training {author} with redirection tracking\n{'='*60}")
        measurements = train_with_measurements(
            author, tokenizer, base_promoted, measure_every=args.measure_every
        )
        if measurements:
            all_data[author] = measurements

    # Save
    serializable = {}
    for author, ms in all_data.items():
        serializable[author] = [
            {**m, "overlap": {str(k): v for k, v in m["overlap"].items()}}
            for m in ms
        ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for author, ms in all_data.items():
        initial = ms[0]["mean_overlap"]
        final = ms[-1]["mean_overlap"]
        half_target = initial - (initial - final) * 0.5
        half_step = next((m["step"] for m in ms if m["mean_overlap"] <= half_target and m["step"] > 0), None)
        total = ms[-1]["step"]
        pct = f" ({half_step / total:.0%} through)" if half_step else ""
        print(f"  {author:>10s}: {initial:.0%} -> {final:.0%}, "
              f"50% redirection at step {half_step}/{total}{pct}")


if __name__ == "__main__":
    main()
