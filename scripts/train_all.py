#!/usr/bin/env python3
"""Batch-train LoRA adapters for all author texts.

Usage:
    python scripts/train_all.py                        # train all authors
    python scripts/train_all.py --only shelley poe     # specific authors
    python scripts/train_all.py --retrain              # re-train existing
    python scripts/train_all.py --max-words 50000      # cap words per author
"""

import argparse
import json
import time
from pathlib import Path

from sixteen_voices import TextChunkDataset, create_lora_model, load_base_model, load_tokenizer
from sixteen_voices.text import clean_text

from train_lora import train

DATA_DIR = Path("data/authors")
ADAPTERS_DIR = Path("outputs/authors")

EPOCHS = 8
BATCH_SIZE = 4
LR = 5e-4
MAX_LENGTH = 512
STRIDE = 256
VAL_SPLIT = 0.1
MAX_WORDS = 50_000


def truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Truncate text to max_words. Returns (text, was_truncated)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    # Find the byte position after max_words words, then cut at last newline
    truncated = " ".join(words[:max_words])
    # Try to end at a paragraph boundary
    last_para = truncated.rfind("\n\n")
    if last_para > len(truncated) // 2:
        truncated = truncated[:last_para]
    return truncated, True


def train_author(name, raw_text, tokenizer, base_model, skip_existing=True,
                 epochs=EPOCHS, max_words=MAX_WORDS):
    from torch.utils.data import Subset

    adapter_dir = ADAPTERS_DIR / name / "adapter"
    meta_path = ADAPTERS_DIR / name / "meta.json"

    if skip_existing and (adapter_dir / "adapter_config.json").exists():
        print(f"  [{name}] already done, skipping")
        return

    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    # Clean and truncate
    raw_words = len(raw_text.split())
    cleaned = clean_text(raw_text)
    clean_words = len(cleaned.split())
    text, was_truncated = truncate_words(cleaned, max_words)
    train_words = len(text.split())

    print(f"  {raw_words:,}w raw -> {clean_words:,}w clean -> {train_words:,}w train"
          + (" (truncated)" if was_truncated else ""))

    dataset = TextChunkDataset(text, tokenizer, max_length=MAX_LENGTH, stride=STRIDE)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    print(f"  {len(dataset)} chunks -> {n_train} train / {n_val} val")

    model = create_lora_model(base_model=base_model)
    t0 = time.time()
    train_losses, val_losses = train(
        model, train_ds, val_ds,
        num_epochs=epochs, lr=LR, batch_size=BATCH_SIZE, tag=name,
    )
    elapsed = time.time() - t0

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)

    # Save metadata
    meta = {
        "author": name,
        "raw_words": raw_words,
        "clean_words": clean_words,
        "train_words": train_words,
        "truncated": was_truncated,
        "max_words": max_words,
        "n_chunks": len(dataset),
        "n_train": n_train,
        "n_val": n_val,
        "epochs": epochs,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "train_seconds": round(elapsed, 1),
        "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
        "final_val_loss": val_losses[-1]["val_loss"] if val_losses else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved -> {adapter_dir}/  ({elapsed:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Batch-train all author LoRAs")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"No data directory at {DATA_DIR}/")
        print("Run: python data/get_books.py && python data/get_author_datasets.py")
        return

    tokenizer = load_tokenizer()
    base_model = load_base_model()

    authors = {p.stem: p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.txt"))}
    if args.only:
        authors = {k: v for k, v in authors.items() if k in args.only}

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training {len(authors)} author LoRAs "
          f"({args.epochs} epochs, batch={BATCH_SIZE}, max_words={args.max_words:,})")

    for name, text in authors.items():
        train_author(name, text, tokenizer, base_model,
                     skip_existing=not args.retrain,
                     epochs=args.epochs, max_words=args.max_words)

    done = sum(
        1 for name in authors
        if (ADAPTERS_DIR / name / "adapter" / "adapter_config.json").exists()
    )
    print(f"\nDone! {done}/{len(authors)} adapters in {ADAPTERS_DIR}/")


if __name__ == "__main__":
    main()