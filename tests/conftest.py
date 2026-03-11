"""Test fixtures — mock tensors, no model download needed."""

import torch
import pytest


@pytest.fixture
def mock_delta():
    """A 1024x1024 delta matrix with known structure."""
    torch.manual_seed(0)
    return torch.randn(1024, 1024) * 0.01


@pytest.fixture
def mock_deltas(mock_delta):
    """Dict with q_proj and v_proj deltas."""
    torch.manual_seed(1)
    return {
        "q_proj": mock_delta,
        "v_proj": torch.randn(1024, 1024) * 0.01,
    }


@pytest.fixture
def sample_text_with_toc():
    """Text with a table of contents followed by prose."""
    return """TABLE OF CONTENTS

CHAPTER I — THE RABBIT HOLE
CHAPTER II — THE POOL OF TEARS
CHAPTER III — A CAUCUS-RACE

CHAPTER I

THE RABBIT HOLE

Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-POCKET, and looked at it, and then hurried on, Alice started to her feet."""


@pytest.fixture
def sample_prose_only():
    """Pure prose, no TOC."""
    return """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?"

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."""
