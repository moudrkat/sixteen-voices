"""Tests for text utilities."""

from sixteen_voices.text import extract_prose


def test_extract_prose_skips_toc(sample_text_with_toc):
    """extract_prose should skip TOC and return actual prose."""
    result = extract_prose(sample_text_with_toc, length=500)
    # Should NOT start with "TABLE OF CONTENTS"
    assert not result.startswith("TABLE")
    # Should contain actual prose
    assert "Alice" in result


def test_extract_prose_respects_length(sample_text_with_toc):
    """Output should not exceed requested length."""
    result = extract_prose(sample_text_with_toc, length=200)
    assert len(result) <= 200


def test_extract_prose_pure_prose(sample_prose_only):
    """On pure prose, should return from the beginning."""
    result = extract_prose(sample_prose_only, length=500)
    assert "Alice" in result


def test_extract_prose_fallback():
    """On text with no qualifying line, falls back to offset 2000."""
    # 2000 chars of filler, then distinctive content
    filler = "A" * 2000
    target = "BCDEFGHIJK" * 50  # 500 chars of recognizable content
    text = filler + target
    result = extract_prose(text, length=200)
    # Should start at offset 2000, i.e. the B's
    assert result.startswith("B")


def test_extract_prose_finds_lowercase_line():
    """Should find the first line >60 chars that is mostly lowercase."""
    text = (
        "CHAPTER I — THE BEGINNING OF ALL THINGS AND MORE STUFF TO MAKE IT LONG\n"
        "THIS IS STILL ALL CAPS AND SHOULD BE SKIPPED BECAUSE ITS UPPERCASE TEXT\n"
        "The little girl walked slowly through the dark forest, her eyes wide with wonder and fear.\n"
        "More prose continues here after the first qualifying line was found.\n"
    )
    result = extract_prose(text, length=500)
    assert result.startswith("The little girl")
