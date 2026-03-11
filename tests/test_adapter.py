"""Tests for adapter weight manipulation."""

import torch
from sixteen_voices.adapter import knockout_all_except, delta_to_AB
from sixteen_voices.constants import HEAD_DIM, RANK


def test_knockout_zeros_other_heads(mock_delta):
    """knockout_all_except should zero all rows except the kept head."""
    result = knockout_all_except(mock_delta, keep_head=3)

    # Head 3's rows should be preserved
    start, end = 3 * HEAD_DIM, 4 * HEAD_DIM
    assert torch.allclose(result[start:end], mock_delta[start:end])

    # All other rows should be zero
    assert result[:start].abs().sum() == 0
    assert result[end:].abs().sum() == 0


def test_knockout_preserves_shape(mock_delta):
    """Output should have same shape as input."""
    result = knockout_all_except(mock_delta, keep_head=0)
    assert result.shape == mock_delta.shape


def test_delta_to_AB_roundtrip(mock_delta):
    """SVD reconstruction should approximate the original delta."""
    A, B = delta_to_AB(mock_delta, rank=RANK)

    # Check shapes
    assert A.shape == (RANK, 1024)
    assert B.shape == (1024, RANK)

    # Reconstruction should be close (rank-8 approx of rank-1024 matrix)
    reconstructed = B @ A
    assert reconstructed.shape == mock_delta.shape


def test_delta_to_AB_higher_rank_is_closer(mock_delta):
    """Higher rank should give better approximation."""
    A4, B4 = delta_to_AB(mock_delta, rank=4)
    A8, B8 = delta_to_AB(mock_delta, rank=8)

    err4 = (B4 @ A4 - mock_delta).norm()
    err8 = (B8 @ A8 - mock_delta).norm()

    assert err8 <= err4


def test_knockout_each_head_independent(mock_delta):
    """Keeping different heads should give different results."""
    r0 = knockout_all_except(mock_delta, keep_head=0)
    r1 = knockout_all_except(mock_delta, keep_head=1)
    assert not torch.allclose(r0, r1)
