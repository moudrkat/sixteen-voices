"""Tests for text utilities."""

from sixteen_voices.text import extract_prose, _is_prose_line


def test_is_prose_line_accepts_prose():
    """Long lowercase line should be detected as prose."""
    assert _is_prose_line(
        "Alice was beginning to get very tired of sitting by her sister on the bank."
    )


def test_is_prose_line_rejects_short():
    """Short lines are not prose."""
    assert not _is_prose_line("CHAPTER I")


def test_is_prose_line_rejects_uppercase():
    """ALL CAPS lines are not prose."""
    assert not _is_prose_line(
        "THE BEGINNING OF ALL THINGS AND MORE STUFF TO MAKE IT VERY LONG INDEED"
    )


def test_is_prose_line_rejects_gutenberg():
    """Gutenberg boilerplate should not count as prose."""
    assert not _is_prose_line(
        "Produced by David Widger from a text prepared by Project Gutenberg volunteers"
    )


def test_extract_prose_finds_prose_block(sample_text_with_toc):
    """extract_prose should find the prose block and include Alice."""
    result = extract_prose(sample_text_with_toc, length=500)
    assert "Alice" in result


def test_extract_prose_respects_length(sample_text_with_toc):
    """Output should not exceed requested length."""
    result = extract_prose(sample_text_with_toc, length=200)
    assert len(result) <= 200


def test_extract_prose_pure_prose(sample_prose_only):
    """On pure prose, should return from the beginning."""
    result = extract_prose(sample_prose_only, length=500)
    assert "Alice" in result


def test_extract_prose_fallback_returns_something():
    """On text with no qualifying prose, should still return content."""
    filler = "A" * 2000
    text = filler
    result = extract_prose(text, length=200)
    # Should return something (falls back to beginning)
    assert len(result) > 0


def test_extract_prose_skips_chapter_heading():
    """Should prefer prose after a chapter heading over raw start."""
    lines = [
        "PREFACE",
        "",
        "This is an editor's note about the text that is long enough to be detected as prose by our heuristic function.",
        "The editor continues rambling about the provenance of this text and its historical significance in detail.",
        "More editorial content follows here with additional unnecessary details about the manuscript and its history.",
        "The editor discusses at length the various editions and translations that have been produced over the years.",
        "Further editorial material about the author's life and times and the cultural context of the work itself.",
        "Even more editorial prose that we would like to skip past to get to the actual story content below.",
        "",
        "CHAPTER I",
        "",
        "It was a bright cold day in April, and the clocks were striking thirteen, and the world was changing fast.",
        "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly.",
        "The hallway smelt of boiled cabbage and old rag mats, and at one end there was a coloured poster too large.",
        "It depicted simply an enormous face, more than a metre wide, and the face of a man about forty-five years old.",
        "It was one of those pictures which are so contrived that the eyes follow you about when you move around.",
        "BIG BROTHER IS WATCHING YOU, the caption beneath it ran, and Winston felt the familiar sense of dread.",
    ]
    text = "\n".join(lines)
    result = extract_prose(text, length=500)
    # Should find prose after "CHAPTER I", not the editor's preface
    assert "bright cold day" in result
