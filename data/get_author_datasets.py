#!/usr/bin/env python3
"""Combine per-book texts into per-author corpora.

Downloads from Gutenberg if needed, then concatenates all books per author
into data/authors/{author}.txt.

Usage:
    python data/get_author_datasets.py
    python data/get_author_datasets.py --authors carroll grimm
    python data/get_author_datasets.py --list
"""

import argparse
import time
from pathlib import Path

from gutenberg import fetch_text, strip_gutenberg
from get_books import BOOKS

AUTHORS_DIR = Path(__file__).parent / "authors"


def _build_author_books():
    """Derive author -> [(id, title)] from BOOKS."""
    groups = {}
    for _key, (book_id, title, author) in BOOKS.items():
        groups.setdefault(author, []).append((book_id, title))
    return groups


AUTHOR_BOOKS = _build_author_books()


def download_author(author: str, books: list, force: bool = False) -> bool:
    out_path = AUTHORS_DIR / f"{author}.txt"
    if out_path.exists() and not force:
        print(f"  [{author}] already exists -- skip")
        return True

    parts = []
    for book_id, title in books:
        print(f"  [{author}] fetching: {title} (id={book_id})...", end=" ", flush=True)
        raw = fetch_text(book_id)
        if raw is None:
            print("FAILED (skipped)")
            continue
        clean = strip_gutenberg(raw)
        words = len(clean.split())
        print(f"{words:,} words")
        parts.append(f"=== {title} ===\n\n{clean}")
        time.sleep(1)

    if not parts:
        print(f"  [{author}] nothing downloaded")
        return False

    combined = "\n\n\n".join(parts)
    out_path.write_text(combined, encoding="utf-8")
    total_words = len(combined.split())
    print(f"  [{author}] saved: {total_words:,} words\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build per-author corpora")
    parser.add_argument("--authors", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for author, books in sorted(AUTHOR_BOOKS.items()):
            print(f"\n{author}:")
            for book_id, title in books:
                print(f"  [{book_id}] {title}")
        return

    AUTHORS_DIR.mkdir(exist_ok=True)
    targets = {a: AUTHOR_BOOKS[a] for a in (args.authors or AUTHOR_BOOKS)}

    print(f"Building {len(targets)} author corpora...\n")
    ok, fail = 0, 0
    for author, books in targets.items():
        if download_author(author, books, force=args.force):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
