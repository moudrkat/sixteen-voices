#!/usr/bin/env python3
"""Pre-extract eval prose for all authors and save to data/eval/.

This makes eval text reproducible and auditable — every script reads
from data/eval/{author}.txt instead of running heuristics at eval time.

Re-run this script whenever source data or extract_prose logic changes.

Usage:
    python scripts/extract_prose_texts.py
    python scripts/extract_prose_texts.py --only poe browne
    python scripts/extract_prose_texts.py --length 10000
    python scripts/extract_prose_texts.py --preview poe   # print without saving
"""

import argparse
from pathlib import Path

from sixteen_voices.text import extract_prose

DATA_DIR = Path("data/authors")
EVAL_DIR = Path("data/eval")

DEFAULT_LENGTH = 14000  # ~2000 words * 7 chars/word

# --------------------------------------------------------------------------
# Per-author skip offsets (line numbers).
# Where the generic heuristic lands on editor prefaces / translator intros
# instead of actual author content, we skip to a known good line.
# These were determined by manual inspection of data/authors/*.txt.
# --------------------------------------------------------------------------
SKIP_LINES = {
    # Gutenberg prefaces / editor intros that fool the heuristic.
    # Each value is the line number in data/authors/{name}.txt where
    # actual author content begins.
    "aesop": 848,        # "A LION was awakened" — first actual fable
    "african": 1271,     # actual folk tales start
    "blake": 91,         # "Piping down the valleys wild"
    "browne": 2371,      # Chapter I of Religio Medici (actual Browne)
    "chinese": 333,      # "Once upon a time there were two brothers"
    "egyptian": 1244,    # actual mythology content
    "filipino": 327,     # "One day Aponibolinayen..."
    "gibbon": 2064,      # "In the second century of the Christian Æra"
    "greek_myth": 97,    # actual myths start after preface
    "harris": 373,       # actual Uncle Remus stories
    "homer": 25738,      # actual Homer prose translation
    "indian": 868,       # actual fairy tales
    "italian": 1428,     # "Once upon a time there was a man"
    "jacobs": 190,       # "Once upon a time there was a woman"
    "japanese": 102,     # "Long, long ago there lived, in Japan"
    "korean": 184,       # "In the days of King Sung-jong"
    "lear": 308,         # "There was an Old Man with a beard"
    "lofting": 258,      # "He lived in a little town called Puddleby"
    "maeterlinck": 501,  # actual dialogue starts
    "norse": 1555,       # actual tales (after scholarly preface)
    "poe": 1933,         # actual stories start at "Hans Pfaall"
    "russian": 150,      # "Ivan returned home..."
    "spyri": 79,         # Chapter I actual text
}

# Corrupted source files — wrong book / wrong language entirely.
# These should be excluded from evals.
EXCLUDED = {
    "dunsany",          # Dutch text, not Lord Dunsany
    "yeats",            # Finnish text, not W.B. Yeats
    "macdonald",        # CIA World Factbook, not George MacDonald
    "native_american",  # "The New Pun Book", not Native American folklore
    "potter",           # File contains Elizabeth Bibesco after Potter tales;
                        # Potter's short-lined prose with [Illustration] breaks
                        # defeats the heuristic → extracts Bibesco instead
}


def extract_one(author: str, length: int) -> str | None:
    src = DATA_DIR / f"{author}.txt"
    if not src.exists():
        return None
    raw = src.read_text(encoding="utf-8")

    skip = SKIP_LINES.get(author, 0)
    if skip > 0:
        lines = raw.split("\n")
        raw = "\n".join(lines[skip:])

    return extract_prose(raw, length=length)


def main():
    parser = argparse.ArgumentParser(description="Pre-extract eval prose")
    parser.add_argument("--only", nargs="+", help="Only these authors")
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH,
                        help="Characters to extract (default: %(default)s)")
    parser.add_argument("--preview", metavar="AUTHOR",
                        help="Print extracted text for one author, don't save")
    args = parser.parse_args()

    if args.preview:
        if args.preview in EXCLUDED:
            print(f"{args.preview} is EXCLUDED (corrupted source)")
            return
        text = extract_one(args.preview, args.length)
        if text is None:
            print(f"No source file for {args.preview}")
            return
        skip = SKIP_LINES.get(args.preview, 0)
        label = f" (skip={skip})" if skip else ""
        print(f"--- {args.preview}{label} ({len(text)} chars, {len(text.split())} words) ---")
        print(text[:2000])
        print(f"\n... ({len(text)} chars total)")
        return

    authors = sorted(p.stem for p in DATA_DIR.glob("*.txt"))
    if args.only:
        authors = [a for a in authors if a in args.only]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting eval text for {len(authors)} authors -> {EVAL_DIR}/\n")

    excluded_count = 0
    for author in authors:
        if author in EXCLUDED:
            print(f"  {author:20s}  EXCLUDED (corrupted source)")
            # Remove stale eval file if it exists
            stale = EVAL_DIR / f"{author}.txt"
            if stale.exists():
                stale.unlink()
            excluded_count += 1
            continue

        text = extract_one(author, args.length)
        if text is None:
            print(f"  {author:20s}  SKIP (no source)")
            continue

        skip = SKIP_LINES.get(author, 0)
        tag = f"  skip={skip}" if skip else ""

        out = EVAL_DIR / f"{author}.txt"
        out.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"  {author:20s}  {words:5d} words  {len(text):6d} chars{tag}")

    print(f"\nDone. {excluded_count} excluded, rest saved to {EVAL_DIR}/")
    print("Commit these files so eval results are reproducible.")


if __name__ == "__main__":
    main()