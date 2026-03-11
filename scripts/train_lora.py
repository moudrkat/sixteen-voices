#!/usr/bin/env python3
"""Train a LoRA adapter on a single text file.

Usage:
    python scripts/train_lora.py data/authors/grimm.txt --output outputs/authors/grimm/adapter
    python scripts/train_lora.py data/authors/shelley.txt --epochs 10 --lr 3e-4
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sixteen_voices import TextChunkDataset, create_lora_model, load_tokenizer
from sixteen_voices.model import load_base_model


def train(model, train_dataset, val_dataset, num_epochs, lr, batch_size, tag="lora"):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    train_losses, val_losses = [], []
    step = 0

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            train_losses.append({"step": step, "loss": loss.item()})
            if step % 50 == 0:
                print(f"  [{tag}] step {step}, loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total = sum(
            model(input_ids=b["input_ids"], labels=b["labels"]).loss.item()
            for b in val_dl
        )
        val_loss = total / len(val_dl)
        val_losses.append({"epoch": epoch + 1, "step": step, "val_loss": val_loss})
        print(f"  [{tag}] epoch {epoch+1}/{num_epochs} — val_loss: {val_loss:.4f}")
        model.train()

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on text")
    parser.add_argument("text_file", type=str, help="Path to text file")
    parser.add_argument("--output", type=str, default=None, help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    text_path = Path(args.text_file)
    name = text_path.stem
    out_dir = Path(args.output) if args.output else Path(f"outputs/{name}/adapter")

    tokenizer = load_tokenizer()
    text = text_path.read_text(encoding="utf-8")

    dataset = TextChunkDataset(text, tokenizer, max_length=args.max_length, stride=args.stride)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds = Subset(dataset, range(n_train))
    val_ds = Subset(dataset, range(n_train, len(dataset)))
    print(f"  {len(dataset)} chunks -> {n_train} train / {n_val} val")

    model = create_lora_model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable} trainable / {total} total params")

    train_losses, val_losses = train(
        model, train_ds, val_ds, num_epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, tag=name,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    loss_file = out_dir.parent / "loss.json"
    with open(loss_file, "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)
    print(f"  Saved -> {out_dir}/")


if __name__ == "__main__":
    main()
