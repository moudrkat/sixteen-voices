"""Shared utilities for downloading from Project Gutenberg."""

import re
import urllib.request

_URL_TEMPLATES = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]


def fetch_text(book_id: int) -> str | None:
    """Download a book from Project Gutenberg by ID. Returns text or None."""
    for tmpl in _URL_TEMPLATES:
        url = tmpl.format(id=book_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
        except Exception:
            continue
    return None


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    start = re.search(
        r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*{3}",
        text, re.IGNORECASE,
    )
    if start:
        text = text[start.end():]

    end = re.search(
        r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        text, re.IGNORECASE,
    )
    if end:
        text = text[:end.start()]

    return text.strip()
