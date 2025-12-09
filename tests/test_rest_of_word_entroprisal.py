"""Tests for RestOfWordEntropisalCalculator."""

import pandas as pd
import pytest

from entroprisal import RestOfWordEntropisalCalculator


@pytest.fixture
def sample_words():
    """Create sample word frequency data for testing."""
    return pd.DataFrame(
        {
            "WORD": ["cat", "dog", "the", "and", "can", "car", "cats", "dogs"],
            "COUNT": [1000, 800, 5000, 4000, 600, 700, 400, 350],
        }
    )


def test_init(sample_words):
    """Test calculator initialization."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    assert calc is not None
    assert len(calc.df) == 8


def test_calculate_metrics(sample_words):
    """Test basic metric calculation."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    metrics = calc.calculate_metrics("the cat and dog")

    assert "mean_word_length" in metrics
    # Left-to-right metrics (entropy and surprisal)
    assert "lr_c1_entropy" in metrics
    assert "lr_c1_surprisal" in metrics
    # Right-to-left metrics (entropy and surprisal)
    assert "rl_c1_entropy" in metrics
    assert "rl_c1_surprisal" in metrics


def test_calculate_batch(sample_words):
    """Test batch processing."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    texts = ["the cat", "the dog"]

    results = calc.calculate_batch(texts)
    assert len(results) == 2


def test_get_word_frequency(sample_words):
    """Test word frequency lookup."""
    calc = RestOfWordEntropisalCalculator(sample_words)

    freq = calc.get_word_frequency("cat")
    assert freq == 1000

    # Non-existent word
    freq = calc.get_word_frequency("xyz")
    assert freq == 0


# Add more tests as needed
