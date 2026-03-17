"""Text utilities — prose extraction, cleaning, and perplexity."""

import re
from pathlib import Path

import torch

# Paths relative to project root (scripts are run from there)
AUTHORS_DIR = Path("data/authors")
EVAL_DIR = Path("data/eval")


def get_eval_authors() -> list[str]:
    """Return sorted list of author names that have clean eval texts."""
    return sorted(p.stem for p in EVAL_DIR.glob("*.txt"))


def load_eval_text(author: str, length: int = 5000) -> str:
    """Load clean eval text for an author.

    Reads from data/eval/ (pre-extracted prose).  Falls back to
    data/authors/ + extract_prose() if no eval file exists.
    """
    eval_path = EVAL_DIR / f"{author}.txt"
    if eval_path.exists():
        text = eval_path.read_text(encoding="utf-8")
        return text[:length] if length else text

    # Fallback: extract on the fly from raw author file
    raw_path = AUTHORS_DIR / f"{author}.txt"
    if raw_path.exists():
        return extract_prose(raw_path.read_text(encoding="utf-8"), length=length)

    raise FileNotFoundError(f"No text found for author '{author}' in {EVAL_DIR} or {AUTHORS_DIR}")


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
    """Skip TOC/headers/frontmatter/prefaces, return actual prose.

    Strategy: find prose blocks that appear after structural breaks
    (chapter headings, === separators, centered titles) to skip past
    editor prefaces which are valid prose but not the author's voice.
    Falls back to the first prose block if no post-break block exists.
    """
    lines = text.split("\n")
    MIN_CONSECUTIVE = 5

    def _find_prose_block(start_line: int) -> int | None:
        """Return line index of first prose block at or after start_line.

        Tolerates blank lines between paragraphs — counts only non-blank
        lines toward MIN_CONSECUTIVE so paragraph-per-line files work.
        """
        for i in range(start_line, len(lines)):
            if not _is_prose_line(lines[i].strip()):
                continue
            # Found a candidate start; count prose lines in a window
            prose_count = 0
            for j in range(i, min(i + MIN_CONSECUTIVE * 3, len(lines))):
                s = lines[j].strip()
                if not s:
                    continue  # skip blank lines
                if _is_prose_line(s):
                    prose_count += 1
                    if prose_count >= MIN_CONSECUTIVE:
                        return i
                else:
                    break  # non-blank, non-prose → not a prose block
        return None

    # Words that signal prefatory sections (not actual story content)
    PREFACE_WORDS = {
        "preface", "introduction", "foreword", "contents",
        "acknowledgment", "dedication", "editor", "translator",
        "appreciation", "memoir", "life of", "death of",
        "bibliography", "appendix",
    }

    def _is_story_break(i: int) -> bool:
        """Is line i a chapter/story heading (not preface/TOC)?"""
        s = lines[i].strip()
        if not s:
            return False
        low = s.lower()
        # Skip prefatory headings
        if any(w in low for w in PREFACE_WORDS):
            return False
        # === Book Title === separators — skip the very first one (line 0-2)
        if s.startswith("===") and s.endswith("===") and i > 5:
            return True
        # CHAPTER / Chapter / Story / Tale headings
        if re.match(r"^(CHAPTER|Chapter|STORY|Story|TALE|Tale)\s", s):
            return True
        # Roman numeral headings for story sections (I., II., etc.)
        if re.match(r"^[IVXLC]+\.\s", s) and len(s) < 80:
            return True
        # ALL CAPS title — require blank lines around it and skip
        # title-page metadata (publisher, author credits, etc.)
        TITLE_PAGE_WORDS = {
            "by", "with", "from", "published", "translated",
            "edited", "london", "new york", "harper", "press",
            "illustrated", "copyright", "edition",
        }
        if (s.isupper() and 20 < len(s) < 80
                and i > 0 and not lines[i - 1].strip()
                and i < len(lines) - 1 and not lines[i + 1].strip()
                and not any(w in low for w in TITLE_PAGE_WORDS)):
            return True
        return False

    # Pass 1: find prose after a story/chapter break (skips prefaces)
    for i in range(len(lines)):
        if _is_story_break(i):
            block = _find_prose_block(i + 1)
            if block is not None:
                start = sum(len(l) + 1 for l in lines[:block])
                return text[start : start + length]

    # Pass 2: fall back to first prose block anywhere
    block = _find_prose_block(0)
    if block is not None:
        start = sum(len(l) + 1 for l in lines[:block])
        return text[start : start + length]

    # Final fallback: start from beginning (no prose blocks found at all)
    return text[:length]


def _is_prose_line(line: str) -> bool:
    """Is this line part of a narrative prose paragraph?"""
    if len(line) < 40:
        return False

    lower_ratio = sum(1 for c in line if c.islower()) / len(line)
    if lower_ratio < 0.5:
        return False

    low = line.lower()

    # Reject known metadata patterns
    if any(w in low for w in [
        "project gutenberg", "e-text prepared", "transcribed from",
        "distributed proofreading", "etext",
        "produced by", "produced from", "proofreading team",
        "transcriber's note", "transcriber note",
        "list of illustrations", "illustration",
        "copyright", "all rights reserved",
        "millennium fulcrum", "edition produced",
        "this book has two types of notes",
        "this collection of", "this volume",
        "thanks are due", "acknowledgment",
        "footnotes are", "editor's note",
        "preface", "foreword",
    ]):
        return False

    # Reject TOC entries (numbered chapters, roman numerals with titles)
    if re.match(r"^\s*CHAPTER\s+[IVXLC\d]", line):
        return False
    if re.match(r"^\s*[IVXLC]+\s{2,}", line):
        return False
    if re.match(r"^\s*Chapter\s+[IVXLC\d]", line):
        return False

    # Reject PG catalog / edition metadata
    if any(w in low for w in [
        "pg #", "edition (pg", "foonote", "history of the decline",
    ]):
        return False

    # Reject bibliography (many quoted titles)
    if line.count('"') >= 4 or line.count('\u201c') >= 3:
        return False

    # Reject ASCII boxes
    if line.startswith("|") or line.startswith("+"):
        return False

    return True


def compute_perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    """Compute perplexity of text under model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()
