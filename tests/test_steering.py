"""Tests for steering hooks."""

import torch
from sixteen_voices.steering import make_hook
from sixteen_voices.constants import HEAD_DIM


def test_make_hook_scales_correct_dims():
    """Hook should scale only the specified head's dimensions."""
    head_scales = {3: 2.0}
    hook_fn = make_hook(head_scales)

    # Simulate attention output: (batch=1, seq=5, hidden=1024)
    original = torch.ones(1, 5, 1024)

    class FakeModule:
        pass

    result = hook_fn(FakeModule(), (original,))

    # Pre-hook returns a tuple
    assert isinstance(result, tuple)
    modified = result[0]

    # Head 3 dims should be scaled by 2.0
    start = 3 * HEAD_DIM
    end = 4 * HEAD_DIM
    assert (modified[0, 0, start:end] == 2.0).all()

    # Other heads should be unchanged (1.0)
    assert (modified[0, 0, :start] == 1.0).all()
    assert (modified[0, 0, end:] == 1.0).all()


def test_make_hook_multiple_heads():
    """Hook should handle multiple heads simultaneously."""
    head_scales = {0: 0.0, 15: 3.0}
    hook_fn = make_hook(head_scales)

    original = torch.ones(1, 1, 1024)
    result = hook_fn(None, (original,))
    modified = result[0]

    # Head 0 should be zeroed
    assert (modified[0, 0, :HEAD_DIM] == 0.0).all()
    # Head 15 should be 3x
    assert (modified[0, 0, 15 * HEAD_DIM:] == 3.0).all()
    # Middle heads unchanged
    assert (modified[0, 0, HEAD_DIM:15 * HEAD_DIM] == 1.0).all()


def test_make_hook_does_not_mutate_input():
    """Hook should clone the input, not modify it in-place."""
    head_scales = {7: 0.5}
    hook_fn = make_hook(head_scales)

    original = torch.ones(1, 1, 1024)
    original_copy = original.clone()
    hook_fn(None, (original,))

    # Original tensor should be unchanged
    assert torch.equal(original, original_copy)
