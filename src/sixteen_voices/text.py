"""Text utilities — prose extraction, cleaning, and perplexity."""

import re

import torch


def clean_text(text: str) -> str:
    """Remove non-prose noise from Gutenberg texts for training.

    Strips: illustration tags, book separators, TOC blocks, frontmatter
    (title pages, publisher info), footnote markers, and excess whitespace.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove === Book Title === separators from get_author_datasets.py
    text = re.sub(r"^=== .+ ===$", "", text, flags=re.MULTILINE)

    # Remove [Illustration] and [Illustration: ...]
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)

    # Remove footnote references like [1], [23]
    text = re.sub(r"\[(\d{1,3})\]", "", text)

    # Remove TOC blocks: "CONTENTS" heading followed by short lines
    text = _strip_toc_blocks(text)

    # Remove frontmatter blocks (title pages, publisher info)
    text = _strip_frontmatter(text)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _strip_toc_blocks(text: str) -> str:
    """Remove table-of-contents sections."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*CONTENTS\s*$", lines[i], re.IGNORECASE):
            # Skip CONTENTS heading and following short lines (TOC entries)
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                # TOC entries: short lines, often indented or title-case
                if len(stripped) == 0 or (len(stripped) < 80 and not _is_prose(stripped)):
                    i += 1
                else:
                    break
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _strip_frontmatter(text: str) -> str:
    """Remove title pages and publisher metadata blocks.

    Detects blocks of consecutive short non-prose lines (5+ in a row)
    and removes them. This catches title pages, dedications, publisher
    info, and other metadata that appears throughout multi-book files.

    Safety: if stripping would remove >50% of lines, skip it entirely
    (the text is likely poetry or short-line style, not frontmatter).
    """
    lines = text.split("\n")
    out = []
    buf = []  # buffer of consecutive non-prose lines

    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0 or not _is_prose(stripped):
            buf.append(line)
        else:
            # Flush buffer: keep if short (normal paragraph breaks),
            # discard if long (frontmatter block)
            if len(buf) < 5:
                out.extend(buf)
            buf = []
            out.append(line)

    # Flush remaining buffer
    if len(buf) < 5:
        out.extend(buf)

    # Safety: if we'd remove more than half, the text is probably
    # poetry/short-line style — return original
    if len(out) < len(lines) * 0.5:
        return text

    return "\n".join(out)


def _is_prose(line: str) -> bool:
    """Heuristic: is this line actual prose (not a heading/metadata)?"""
    if len(line) < 40:
        return False
    lower_ratio = sum(1 for c in line if c.islower()) / len(line)
    return lower_ratio > 0.4


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
