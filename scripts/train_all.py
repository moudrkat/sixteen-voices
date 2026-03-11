#!/usr/bin/env python3
"""Batch-train LoRA adapters for all author texts.

Usage:
    python scripts/train_all.py                        # train all authors
    python scripts/train_all.py --only shelley poe     # specific authors
    python scripts/train_all.py --retrain              # re-train existing
"""

import argparse
from pathlib import Path

from sixteen_voices import TextChunkDataset, create_lora_model, load_tokenizer

from train_lora import train

DATA_DIR = Path("data/authors")
ADAPTERS_DIR = Path("outputs/authors")

EPOCHS = 8
BATCH_SIZE = 4
LR = 5e-4
MAX_LENGTH = 512
STRIDE = 256
VAL_SPLIT = 0.1


def train_author(name, text, tokenizer, skip_existing=True, epochs=EPOCHS):
    from torch.utils.data import Subset

    adapter_dir = ADAPTERS_DIR / name / "adapter"
    if skip_existing and (adapter_dir / "adapter_config.json").exists():
        print(f"  [{name}] already done, skipping")
        return

    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    dataset = TextChunkDataset(text, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    print(f"  {len(dataset)} chunks -> {n_train} train / {n_val} val")

    model = create_lora_model()
    train_losses, val_losses = train(
        model, train_ds, val_ds,
        num_epochs=epochs, lr=LR, batch_size=BATCH_SIZE, tag=name,
    )

    import json
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    with open(adapter_dir.parent / "loss.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)
    print(f"  Saved -> {adapter_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Batch-train all author LoRAs")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"No data directory at {DATA_DIR}/")
        print("Run: python data/get_books.py && python data/get_author_datasets.py")
        return

    tokenizer = load_tokenizer()

    authors = {p.stem: p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.txt"))}
    if args.only:
        authors = {k: v for k, v in authors.items() if k in args.only}

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training {len(authors)} author LoRAs ({args.epochs} epochs, batch={BATCH_SIZE})")

    for name, text in authors.items():
        train_author(name, text, tokenizer, skip_existing=not args.retrain, epochs=args.epochs)

    done = sum(
        1 for name in authors
        if (ADAPTERS_DIR / name / "adapter" / "adapter_config.json").exists()
    )
    print(f"\nDone! {done}/{len(authors)} adapters in {ADAPTERS_DIR}/")


if __name__ == "__main__":
    main()
