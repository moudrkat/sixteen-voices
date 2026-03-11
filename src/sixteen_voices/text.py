"""Text utilities — prose extraction and perplexity."""

import torch


def extract_prose(text: str, length: int = 5000) -> str:
    """Skip TOC/headers/frontmatter, return actual prose.

    Finds the first line with 60+ characters that is mostly lowercase
    (real prose, not chapter headings or metadata).
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) > 60:
            lower_ratio = sum(1 for c in stripped if c.islower()) / len(stripped)
            if lower_ratio > 0.5:
                start = sum(len(l) + 1 for l in lines[:i])
                return text[start : start + length]
    # fallback: skip first 2000 chars
    return text[2000 : 2000 + length]


def compute_perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    """Compute perplexity of text under model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()
