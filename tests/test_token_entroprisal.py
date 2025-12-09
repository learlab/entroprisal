"""Tests for TokenEntropisalCalculator."""

import polars as pl
import pytest

from entroprisal import TokenEntropisalCalculator


@pytest.fixture
def sample_ngrams():
    """Create sample n-gram data for testing."""
    return pl.LazyFrame(
        {
            "token_0": ["the", "the", "the", "cat", "cat", "sat"],
            "token_1": ["cat", "cat", "dog", "sat", "sat", "on"],
            "token_2": ["sat", "ran", "ran", "on", "down", "the"],
            "token_3": ["down", "away", "fast", "the", "slowly", "mat"],
            "count": [100, 50, 30, 80, 20, 60],
        }
    )


def test_init(sample_ngrams):
    """Test calculator initialization."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    assert calc is not None
    assert calc.min_frequency == 10


def test_calculate_metrics(sample_ngrams):
    """Test basic metric calculation."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    metrics = calc.calculate_metrics(["the", "cat", "sat"])

    assert "n_tokens" in metrics
    assert metrics["n_tokens"] == 3
    assert "mean_token_length" in metrics


def test_calculate_batch(sample_ngrams):
    """Test batch processing."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    token_lists = [
        ["the", "cat"],
        ["the", "dog"],
    ]

    results = calc.calculate_batch(token_lists)
    assert len(results) == 2
    assert "n_tokens" in results.columns
